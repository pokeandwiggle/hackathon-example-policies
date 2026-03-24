"""Temporal ensemble blending for action chunks.

Eliminates position/orientation discontinuities ("steps") at action chunk
boundaries by blending overlapping predictions from consecutive chunks.

Two blending strategies are used depending on the configuration:

1. **Temporal ensemble** (``chunk_size > n_action_steps``):
   The old chunk's tail (predictions beyond ``n_action_steps``) and the new
   chunk's head (first ``overlap`` steps) refer to the *same* future
   timesteps.  We blend them with a linear ramp (LERP for position,
   SLERP for quaternions) so the transition is smooth.

2. **Offset-decay** (``chunk_size == n_action_steps``):
   No overlapping predictions exist, but there may still be a position/
   orientation jump between the last sent action and the first action of
   the new chunk.  We smooth this by blending the first few actions of
   the new chunk toward the last-sent action with a decaying weight.

All blending is performed in *translated* (absolute TCP) space,
which makes it action-mode agnostic.
"""

from typing import Optional

import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from ...utils.action_order import (
    DUAL_ABS_LEFT_POS_IDXS,
    DUAL_ABS_LEFT_QUAT_IDXS,
    DUAL_ABS_RIGHT_POS_IDXS,
    DUAL_ABS_RIGHT_QUAT_IDXS,
)


# ---------------------------------------------------------------------------
# Quaternion SLERP
# ---------------------------------------------------------------------------


def slerp_quat(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between unit quaternions (xyzw).

    Handles hemisphere consistency (shortest path) and falls back to
    linear interpolation for nearly identical quaternions.
    """
    q0 = q0.float()
    q1 = q1.float()

    slerp = Slerp([0, 1], R.from_quat([q0.cpu().numpy(), q1.cpu().numpy()]))
    R_interp = slerp(t)
    q_interp = torch.from_numpy(R_interp.as_quat()).to(q0.device)
    return q_interp


def linear_blend(v0: torch.Tensor, v1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Linear blend between two tensors."""
    return (1.0 - alpha) * v0 + alpha * v1


# ---------------------------------------------------------------------------
# Per-action blending helper
# ---------------------------------------------------------------------------


def blend_abs_tcp_actions(
    old: torch.Tensor, new: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Blend two 16-dim absolute TCP actions.

    Layout: ``[L_pos(3), L_quat(4), R_pos(3), R_quat(4), grip_L, grip_R]``

    * Position dims: LERP
    * Quaternion dims: SLERP (shortest-path)
    * Grippers: taken from *new* (no interpolation — discrete)

    Args:
        old: (16,) previous chunk's prediction.
        new: (16,) current chunk's prediction.
        alpha: Blend weight. 0 → all *old*; 1 → all *new*.
    """
    result = new.clone()

    old_left_pos = old[DUAL_ABS_LEFT_POS_IDXS]
    old_left_quat = old[DUAL_ABS_LEFT_QUAT_IDXS]
    old_right_pos = old[DUAL_ABS_RIGHT_POS_IDXS]
    old_right_quat = old[DUAL_ABS_RIGHT_QUAT_IDXS]
    new_left_pos = new[DUAL_ABS_LEFT_POS_IDXS]
    new_left_quat = new[DUAL_ABS_LEFT_QUAT_IDXS]
    new_right_pos = new[DUAL_ABS_RIGHT_POS_IDXS]
    new_right_quat = new[DUAL_ABS_RIGHT_QUAT_IDXS]

    # Interpolate positions and quaternions separately, then combine back into
    # a single action vector.
    blend_left_pos = linear_blend(old_left_pos, new_left_pos, alpha)
    blend_right_pos = linear_blend(old_right_pos, new_right_pos, alpha)
    blend_left_quat = slerp_quat(old_left_quat, new_left_quat, alpha)
    blend_right_quat = slerp_quat(old_right_quat, new_right_quat, alpha)

    result[DUAL_ABS_LEFT_POS_IDXS] = blend_left_pos
    result[DUAL_ABS_RIGHT_POS_IDXS] = blend_right_pos
    result[DUAL_ABS_LEFT_QUAT_IDXS] = blend_left_quat
    result[DUAL_ABS_RIGHT_QUAT_IDXS] = blend_right_quat

    return result


# ---------------------------------------------------------------------------
# ActionChunkBlender
# ---------------------------------------------------------------------------


class ActionChunkBlender:
    """Smooth action chunk boundaries via temporal ensemble / offset-decay.

    Operates entirely in **translated (absolute TCP 16-dim)** space.

    Usage::

        blender = ActionChunkBlender(chunk_size=16, n_action_steps=8)

        # On every new chunk:
        translated = [translator.translate(a, obs)[0] for a in raw_chunk]
        blender.on_new_chunk(translated)

        # On every step (including the first step of new chunk):
        action_to_send = blender.pop_action()  # (1, 16)
    """

    def __init__(
        self,
        chunk_size: int,
        n_action_steps: int,
        decay_steps: int = 8,
    ):
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self.overlap = max(0, chunk_size - n_action_steps)
        self.decay_steps = decay_steps

        # Translated actions for the current chunk (set by on_new_chunk)
        self._current_chunk: Optional[list[torch.Tensor]] = None

        # Tail of previous translated chunk (overlap portion)
        self._prev_chunk_tail: Optional[list[torch.Tensor]] = None

        # Last action actually sent to the robot
        self._last_sent_action: Optional[torch.Tensor] = None

        # Step counter within the current chunk
        self._step_in_chunk: int = 0

    def reset(self):
        """Clear all state for a new rollout."""
        self._current_chunk = None
        self._prev_chunk_tail = None
        self._last_sent_action = None
        self._step_in_chunk = 0

    @property
    def has_chunk(self) -> bool:
        """Whether the blender has actions ready to pop."""
        return self._current_chunk is not None and self._step_in_chunk < len(
            self._current_chunk
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_new_chunk(self, translated_actions: list[torch.Tensor]) -> None:
        """Register a new fully-translated action chunk and apply blending.

        Args:
            translated_actions: List of ``chunk_size`` tensors, each ``(16,)``
                in absolute TCP space.  Modified **in-place** during blending.
        """
        # --- Temporal ensemble: blend new head with old tail ---
        if self.overlap > 0 and self._prev_chunk_tail is not None:
            n_blend = min(
                self.overlap, len(self._prev_chunk_tail), len(translated_actions)
            )
            print("=== ActionChunkBlender: Temporal Ensemble Blending ===")
            print(f"Blending {n_blend} overlapping steps with temporal ensemble")
            print(f"Previous chunk tail length: {len(self._prev_chunk_tail)}")
            print(f"Current chunk length: {len(translated_actions)}")
            print(f"Overlap: {self.overlap}, Blend steps: {n_blend}")

            for k in range(n_blend):
                alpha = (k + 1) / n_blend
                old = self._prev_chunk_tail[k]
                new = translated_actions[k]
                translated_actions[k] = blend_abs_tcp_actions(old, new, alpha)

        # --- Offset-decay (fallback when there is no overlap) ---
        elif self.overlap <= 0 and self._last_sent_action is not None:
            raise NotImplementedError(
                "Offset-decay blending is not implemented yet."
                "Please set chunk_size > n_action_steps for temporal ensemble blending."
            )
            # K = min(self.decay_steps, self.n_action_steps)
            # for k in range(K):
            #     alpha = (k + 1) / (K + 1)
            #     translated_actions[k] = blend_abs_tcp_actions(
            #         self._last_sent_action, translated_actions[k], alpha
            #     )

        # Store THIS chunk's tail for blending with the NEXT chunk
        if self.overlap > 0:
            self._prev_chunk_tail = [
                a.clone() for a in translated_actions[self.n_action_steps :]
            ]

        # Only keep the executable portion (n_action_steps) for pop_action.
        # The tail beyond n_action_steps is only used as blending reference
        # for the next chunk — it must NOT be sent to the robot.
        self._current_chunk = translated_actions[: self.n_action_steps]
        self._step_in_chunk = 0

    def pop_action(self) -> torch.Tensor:
        """Pop the next blended action.

        Returns:
            ``(1, 16)`` absolute TCP action tensor (batch dim included for
            compatibility with ``send_action``).
        """
        assert self._current_chunk is not None, "pop_action called before on_new_chunk"
        assert self._step_in_chunk < len(self._current_chunk), (
            f"Chunk exhausted: popped {self._step_in_chunk}/{len(self._current_chunk)}"
        )

        action = self._current_chunk[self._step_in_chunk]
        self._last_sent_action = action.clone()
        self._step_in_chunk += 1
        return action.unsqueeze(0)  # (1, 16)
