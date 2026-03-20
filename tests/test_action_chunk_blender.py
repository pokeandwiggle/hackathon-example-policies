"""Tests for ActionChunkBlender — temporal ensemble and offset-decay blending.

Covers:
    1. SLERP quaternion interpolation
    2. blend_abs_tcp_actions (per-action LERP/SLERP)
    3. ActionChunkBlender with temporal ensemble (overlap > 0)
    4. ActionChunkBlender with offset-decay (overlap == 0)
    5. Reset and edge cases
"""

import pytest
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np

from example_policies.robot_deploy.deploy_core.action_chunk_blender import (
    ActionChunkBlender,
    blend_abs_tcp_actions,
    slerp_quat,
)
from example_policies.utils.action_order import (
    DUAL_ABS_LEFT_POS_IDXS,
    DUAL_ABS_LEFT_QUAT_IDXS,
    DUAL_ABS_RIGHT_POS_IDXS,
    DUAL_ABS_RIGHT_QUAT_IDXS,
)

ATOL = 1e-5


# =============================================================================
# Helpers
# =============================================================================


def _random_quat(rng: np.random.Generator) -> torch.Tensor:
    """Random unit quaternion (xyzw) as a torch tensor."""
    return torch.from_numpy(R.random(random_state=rng).as_quat()).float()


def _make_abs_tcp_action(
    left_pos: list[float],
    left_quat: torch.Tensor,
    right_pos: list[float],
    right_quat: torch.Tensor,
    left_gripper: float = 0.04,
    right_gripper: float = 0.04,
) -> torch.Tensor:
    """Build a 16-dim absolute TCP action tensor."""
    action = torch.zeros(16)
    action[DUAL_ABS_LEFT_POS_IDXS] = torch.tensor(left_pos)
    action[DUAL_ABS_LEFT_QUAT_IDXS] = left_quat
    action[DUAL_ABS_RIGHT_POS_IDXS] = torch.tensor(right_pos)
    action[DUAL_ABS_RIGHT_QUAT_IDXS] = right_quat
    action[14] = left_gripper
    action[15] = right_gripper
    return action


# =============================================================================
# 1. SLERP Tests
# =============================================================================


class TestSlerpQuat:
    """Test quaternion spherical linear interpolation."""

    def test_identity_endpoints(self):
        """slerp(q, q, t) should return q for any t."""
        rng = np.random.default_rng(42)
        q = _random_quat(rng)
        for t in [0.0, 0.5, 1.0]:
            result = slerp_quat(q, q, t)
            dot = torch.abs(torch.dot(result, q))
            torch.testing.assert_close(dot, torch.tensor(1.0), atol=ATOL, rtol=1e-5)

    def test_t0_returns_q0(self):
        """slerp(q0, q1, 0) should return q0."""
        rng = np.random.default_rng(42)
        q0 = _random_quat(rng)
        q1 = _random_quat(rng)
        result = slerp_quat(q0, q1, 0.0)
        dot = torch.abs(torch.dot(result, q0))
        torch.testing.assert_close(dot, torch.tensor(1.0), atol=ATOL, rtol=1e-5)

    def test_t1_returns_q1(self):
        """slerp(q0, q1, 1) should return q1."""
        rng = np.random.default_rng(42)
        q0 = _random_quat(rng)
        q1 = _random_quat(rng)
        result = slerp_quat(q0, q1, 1.0)
        dot = torch.abs(torch.dot(result, q1))
        torch.testing.assert_close(dot, torch.tensor(1.0), atol=ATOL, rtol=1e-5)

    def test_midpoint_is_interpolated(self):
        """slerp at t=0.5 should be equidistant from both endpoints."""
        rng = np.random.default_rng(42)
        q0 = _random_quat(rng)
        q1 = _random_quat(rng)
        mid = slerp_quat(q0, q1, 0.5)

        # Geodesic distance from mid to q0 and q1 should be equal
        d0 = torch.acos((torch.abs(torch.dot(mid, q0))).clamp(max=1.0))
        d1 = torch.acos((torch.abs(torch.dot(mid, q1))).clamp(max=1.0))
        torch.testing.assert_close(d0, d1, atol=1e-4, rtol=1e-4)

    def test_shortest_path(self):
        """slerp should always take the shortest path (hemisphere flip)."""
        rng = np.random.default_rng(42)
        q0 = _random_quat(rng)
        q1 = -_random_quat(rng)  # negate → opposite hemisphere
        result = slerp_quat(q0, q1, 0.5)
        assert result.shape == (4,)
        # Result should be a unit quaternion
        torch.testing.assert_close(result.norm(), torch.tensor(1.0), atol=ATOL, rtol=1e-5)

    def test_unit_quaternion_output(self):
        """Output should always be a unit quaternion."""
        rng = np.random.default_rng(123)
        for _ in range(20):
            q0 = _random_quat(rng)
            q1 = _random_quat(rng)
            t = rng.random()
            result = slerp_quat(q0, q1, t)
            torch.testing.assert_close(result.norm(), torch.tensor(1.0), atol=ATOL, rtol=1e-5)


# =============================================================================
# 2. blend_abs_tcp_actions Tests
# =============================================================================


class TestBlendAbsTcpActions:
    """Test per-action blending."""

    def _make_pair(self, rng):
        q0_l = _random_quat(rng)
        q0_r = _random_quat(rng)
        q1_l = _random_quat(rng)
        q1_r = _random_quat(rng)
        a0 = _make_abs_tcp_action([0.0, 0.0, 0.0], q0_l, [1.0, 1.0, 1.0], q0_r, 0.0, 0.0)
        a1 = _make_abs_tcp_action([1.0, 1.0, 1.0], q1_l, [2.0, 2.0, 2.0], q1_r, 1.0, 1.0)
        return a0, a1

    def test_alpha_0_returns_old(self):
        """alpha=0 should return old (except grippers = new)."""
        rng = np.random.default_rng(42)
        old, new = self._make_pair(rng)
        result = blend_abs_tcp_actions(old, new, 0.0)
        # Position should be old
        torch.testing.assert_close(result[DUAL_ABS_LEFT_POS_IDXS], old[DUAL_ABS_LEFT_POS_IDXS], atol=ATOL, rtol=1e-5)
        torch.testing.assert_close(result[DUAL_ABS_RIGHT_POS_IDXS], old[DUAL_ABS_RIGHT_POS_IDXS], atol=ATOL, rtol=1e-5)
        # Quaternion should be old
        dot_l = torch.abs(torch.dot(result[DUAL_ABS_LEFT_QUAT_IDXS], old[DUAL_ABS_LEFT_QUAT_IDXS]))
        torch.testing.assert_close(dot_l, torch.tensor(1.0), atol=ATOL, rtol=1e-5)
        # Grippers should be new
        assert result[14] == new[14]
        assert result[15] == new[15]

    def test_alpha_1_returns_new(self):
        """alpha=1 should return new entirely."""
        rng = np.random.default_rng(42)
        old, new = self._make_pair(rng)
        result = blend_abs_tcp_actions(old, new, 1.0)
        # Position should match exactly
        torch.testing.assert_close(result[DUAL_ABS_LEFT_POS_IDXS], new[DUAL_ABS_LEFT_POS_IDXS], atol=ATOL, rtol=1e-5)
        torch.testing.assert_close(result[DUAL_ABS_RIGHT_POS_IDXS], new[DUAL_ABS_RIGHT_POS_IDXS], atol=ATOL, rtol=1e-5)
        # Quaternions should represent the same rotation (SLERP may flip sign)
        dot_l = torch.abs(torch.dot(result[DUAL_ABS_LEFT_QUAT_IDXS], new[DUAL_ABS_LEFT_QUAT_IDXS]))
        torch.testing.assert_close(dot_l, torch.tensor(1.0), atol=1e-3, rtol=1e-3)
        dot_r = torch.abs(torch.dot(result[DUAL_ABS_RIGHT_QUAT_IDXS], new[DUAL_ABS_RIGHT_QUAT_IDXS]))
        torch.testing.assert_close(dot_r, torch.tensor(1.0), atol=1e-3, rtol=1e-3)

    def test_alpha_half_interpolates_position(self):
        """alpha=0.5 should LERP positions to midpoint."""
        rng = np.random.default_rng(42)
        old, new = self._make_pair(rng)
        result = blend_abs_tcp_actions(old, new, 0.5)
        expected_left = (old[DUAL_ABS_LEFT_POS_IDXS] + new[DUAL_ABS_LEFT_POS_IDXS]) / 2
        torch.testing.assert_close(result[DUAL_ABS_LEFT_POS_IDXS], expected_left, atol=ATOL, rtol=1e-5)


# =============================================================================
# 3. ActionChunkBlender — Temporal Ensemble (overlap > 0)
# =============================================================================


class TestActionChunkBlenderTemporalEnsemble:
    """Test blending when chunk_size > n_action_steps."""

    def _make_constant_chunk(self, pos_val: float, chunk_size: int) -> list[torch.Tensor]:
        """Make a chunk where all actions have the same position."""
        identity_q = torch.tensor([0.0, 0.0, 0.0, 1.0])
        return [
            _make_abs_tcp_action(
                [pos_val, pos_val, pos_val], identity_q,
                [pos_val, pos_val, pos_val], identity_q,
            )
            for _ in range(chunk_size)
        ]

    def test_first_chunk_no_blending(self):
        """First chunk should pass through without blending."""
        blender = ActionChunkBlender(chunk_size=16, n_action_steps=8)
        chunk = self._make_constant_chunk(1.0, 16)
        blender.on_new_chunk(chunk)

        for _ in range(8):
            action = blender.pop_action()
            assert action.shape == (1, 16)
            torch.testing.assert_close(
                action[0, DUAL_ABS_LEFT_POS_IDXS],
                torch.tensor([1.0, 1.0, 1.0]),
                atol=ATOL, rtol=1e-5,
            )

    def test_second_chunk_blends_overlap(self):
        """Second chunk should blend the overlap zone with the first chunk's tail."""
        blender = ActionChunkBlender(chunk_size=16, n_action_steps=8)

        # First chunk: positions at 1.0
        chunk1 = self._make_constant_chunk(1.0, 16)
        blender.on_new_chunk(chunk1)
        for _ in range(8):
            blender.pop_action()

        # Second chunk: positions at 3.0
        chunk2 = self._make_constant_chunk(3.0, 16)
        blender.on_new_chunk(chunk2)

        # The first 8 actions of chunk2 should be blended with the tail of chunk1
        # overlap = 16 - 8 = 8
        # alpha for step k = (k+1) / (8+1) = (k+1)/9
        for k in range(8):
            action = blender.pop_action()
            alpha = (k + 1) / 9.0
            expected_pos = (1 - alpha) * 1.0 + alpha * 3.0
            torch.testing.assert_close(
                action[0, DUAL_ABS_LEFT_POS_IDXS],
                torch.full((3,), expected_pos),
                atol=1e-4, rtol=1e-4,
            )

    def test_overlap_count(self):
        blender = ActionChunkBlender(chunk_size=16, n_action_steps=8)
        assert blender.overlap == 8

        blender2 = ActionChunkBlender(chunk_size=30, n_action_steps=30)
        assert blender2.overlap == 0

    def test_stores_tail(self):
        """After on_new_chunk, the blender should store the tail for next blend."""
        blender = ActionChunkBlender(chunk_size=10, n_action_steps=6)
        # overlap = 4
        chunk = self._make_constant_chunk(2.0, 10)
        blender.on_new_chunk(chunk)
        assert blender._prev_chunk_tail is not None
        assert len(blender._prev_chunk_tail) == 4

    def test_quaternion_slerp_in_overlap(self):
        """Overlap blending should use SLERP for quaternions."""
        rng = np.random.default_rng(42)
        blender = ActionChunkBlender(chunk_size=4, n_action_steps=2)
        # overlap = 2

        q0 = torch.tensor([0.0, 0.0, 0.0, 1.0])  # identity
        q1 = _random_quat(rng)

        chunk1 = [_make_abs_tcp_action([0, 0, 0], q0, [0, 0, 0], q0) for _ in range(4)]
        blender.on_new_chunk(chunk1)
        blender.pop_action()
        blender.pop_action()

        chunk2 = [_make_abs_tcp_action([0, 0, 0], q1, [0, 0, 0], q1) for _ in range(4)]
        blender.on_new_chunk(chunk2)

        # Step 0 of chunk2: alpha = 1/3 → SLERP between identity and q1
        a0 = blender.pop_action()
        expected = slerp_quat(q0, q1, 1.0 / 3.0)
        dot = torch.abs(torch.dot(a0[0, DUAL_ABS_LEFT_QUAT_IDXS], expected))
        torch.testing.assert_close(dot, torch.tensor(1.0), atol=1e-4, rtol=1e-4)


# =============================================================================
# 4. ActionChunkBlender — Offset-Decay (overlap == 0)
# =============================================================================


class TestActionChunkBlenderOffsetDecay:
    """Test blending when chunk_size == n_action_steps (no overlap)."""

    def test_first_chunk_no_decay(self):
        """First chunk should pass through without decay (no last_sent_action)."""
        blender = ActionChunkBlender(chunk_size=8, n_action_steps=8)
        identity_q = torch.tensor([0.0, 0.0, 0.0, 1.0])
        chunk = [
            _make_abs_tcp_action([1, 1, 1], identity_q, [1, 1, 1], identity_q)
            for _ in range(8)
        ]
        blender.on_new_chunk(chunk)

        action = blender.pop_action()
        torch.testing.assert_close(
            action[0, DUAL_ABS_LEFT_POS_IDXS],
            torch.tensor([1.0, 1.0, 1.0]),
            atol=ATOL, rtol=1e-5,
        )

    def test_second_chunk_decays_toward_last_sent(self):
        """Second chunk should blend early steps toward the last-sent action."""
        blender = ActionChunkBlender(chunk_size=8, n_action_steps=8, decay_steps=4)
        identity_q = torch.tensor([0.0, 0.0, 0.0, 1.0])

        # First chunk at position 1.0
        chunk1 = [
            _make_abs_tcp_action([1, 1, 1], identity_q, [1, 1, 1], identity_q)
            for _ in range(8)
        ]
        blender.on_new_chunk(chunk1)
        for _ in range(8):
            blender.pop_action()  # last_sent_action will be at [1,1,1]

        # Second chunk at position 5.0
        chunk2 = [
            _make_abs_tcp_action([5, 5, 5], identity_q, [5, 5, 5], identity_q)
            for _ in range(8)
        ]
        blender.on_new_chunk(chunk2)

        # decay_steps=4, alpha = (k+1)/(4+1)
        # Step 0: alpha = 1/5 = 0.2  → pos = 0.8*1 + 0.2*5 = 1.8
        a0 = blender.pop_action()
        torch.testing.assert_close(
            a0[0, DUAL_ABS_LEFT_POS_IDXS],
            torch.full((3,), 1.8),
            atol=1e-4, rtol=1e-4,
        )

        # Step 1: alpha = 2/5 = 0.4  → pos = 0.6*1 + 0.4*5 = 2.6
        a1 = blender.pop_action()
        torch.testing.assert_close(
            a1[0, DUAL_ABS_LEFT_POS_IDXS],
            torch.full((3,), 2.6),
            atol=1e-4, rtol=1e-4,
        )

    def test_actions_beyond_decay_window_unmodified(self):
        """Steps beyond the decay window should be unmodified."""
        blender = ActionChunkBlender(chunk_size=8, n_action_steps=8, decay_steps=2)
        identity_q = torch.tensor([0.0, 0.0, 0.0, 1.0])

        chunk1 = [_make_abs_tcp_action([0, 0, 0], identity_q, [0, 0, 0], identity_q) for _ in range(8)]
        blender.on_new_chunk(chunk1)
        for _ in range(8):
            blender.pop_action()

        chunk2 = [_make_abs_tcp_action([10, 10, 10], identity_q, [10, 10, 10], identity_q) for _ in range(8)]
        blender.on_new_chunk(chunk2)

        # Pop past the 2 decay steps
        blender.pop_action()
        blender.pop_action()

        # Step 2 onward should be unmodified
        a2 = blender.pop_action()
        torch.testing.assert_close(
            a2[0, DUAL_ABS_LEFT_POS_IDXS],
            torch.tensor([10.0, 10.0, 10.0]),
            atol=ATOL, rtol=1e-5,
        )


# =============================================================================
# 5. Reset and Edge Cases
# =============================================================================


class TestActionChunkBlenderReset:
    """Test reset and edge-case behaviour."""

    def test_reset_clears_state(self):
        """After reset, blender should behave as fresh."""
        blender = ActionChunkBlender(chunk_size=4, n_action_steps=2)
        identity_q = torch.tensor([0.0, 0.0, 0.0, 1.0])
        chunk = [_make_abs_tcp_action([1, 1, 1], identity_q, [1, 1, 1], identity_q) for _ in range(4)]
        blender.on_new_chunk(chunk)
        blender.pop_action()

        blender.reset()

        assert blender._current_chunk is None
        assert blender._prev_chunk_tail is None
        assert blender._last_sent_action is None
        assert blender._step_in_chunk == 0

    def test_pop_action_fails_without_chunk(self):
        """pop_action before on_new_chunk should raise."""
        blender = ActionChunkBlender(chunk_size=4, n_action_steps=2)
        with pytest.raises(AssertionError):
            blender.pop_action()

    def test_pop_action_fails_when_exhausted(self):
        """pop_action beyond chunk size should raise."""
        blender = ActionChunkBlender(chunk_size=2, n_action_steps=2)
        identity_q = torch.tensor([0.0, 0.0, 0.0, 1.0])
        chunk = [_make_abs_tcp_action([0, 0, 0], identity_q, [0, 0, 0], identity_q) for _ in range(2)]
        blender.on_new_chunk(chunk)
        blender.pop_action()
        blender.pop_action()
        with pytest.raises(AssertionError, match="Chunk exhausted"):
            blender.pop_action()

    def test_has_chunk_property(self):
        """has_chunk should reflect whether actions are available."""
        blender = ActionChunkBlender(chunk_size=2, n_action_steps=2)
        assert not blender.has_chunk

        identity_q = torch.tensor([0.0, 0.0, 0.0, 1.0])
        chunk = [_make_abs_tcp_action([0, 0, 0], identity_q, [0, 0, 0], identity_q) for _ in range(2)]
        blender.on_new_chunk(chunk)
        assert blender.has_chunk

        blender.pop_action()
        assert blender.has_chunk

        blender.pop_action()
        assert not blender.has_chunk

    def test_last_sent_action_tracking(self):
        """pop_action should update _last_sent_action."""
        blender = ActionChunkBlender(chunk_size=2, n_action_steps=2)
        identity_q = torch.tensor([0.0, 0.0, 0.0, 1.0])
        chunk = [
            _make_abs_tcp_action([1, 2, 3], identity_q, [4, 5, 6], identity_q),
            _make_abs_tcp_action([7, 8, 9], identity_q, [10, 11, 12], identity_q),
        ]
        blender.on_new_chunk(chunk)

        blender.pop_action()
        torch.testing.assert_close(
            blender._last_sent_action[DUAL_ABS_LEFT_POS_IDXS],
            torch.tensor([1.0, 2.0, 3.0]),
            atol=ATOL, rtol=1e-5,
        )

        blender.pop_action()
        torch.testing.assert_close(
            blender._last_sent_action[DUAL_ABS_LEFT_POS_IDXS],
            torch.tensor([7.0, 8.0, 9.0]),
            atol=ATOL, rtol=1e-5,
        )

    def test_continuous_multi_chunk_blending(self):
        """Verify smooth continuity across three consecutive chunks."""
        blender = ActionChunkBlender(chunk_size=6, n_action_steps=4)
        # overlap = 2, alpha = [1/3, 2/3]
        identity_q = torch.tensor([0.0, 0.0, 0.0, 1.0])

        # Chunk positions: 0.0, 10.0, 20.0
        for chunk_val in [0.0, 10.0, 20.0]:
            chunk = [
                _make_abs_tcp_action(
                    [chunk_val, 0, 0], identity_q, [chunk_val, 0, 0], identity_q
                )
                for _ in range(6)
            ]
            blender.on_new_chunk(chunk)

            actions = []
            for _ in range(4):
                a = blender.pop_action()
                actions.append(a[0, 0].item())  # left_x

        # The final 4 actions (chunk=20.0) should be blended with tail of chunk=10.0
        # overlap steps: k=0 → alpha=1/3, blended = (2/3)*10 + (1/3)*20 = 13.33
        #                k=1 → alpha=2/3, blended = (1/3)*10 + (2/3)*20 = 16.67
        # Non-overlap steps: 20.0, 20.0
        final_actions = actions[-4:]
        assert abs(final_actions[0] - 40 / 3) < 0.1  # ≈13.33
        assert abs(final_actions[1] - 50 / 3) < 0.1  # ≈16.67
        assert abs(final_actions[2] - 20.0) < ATOL
        assert abs(final_actions[3] - 20.0) < ATOL
