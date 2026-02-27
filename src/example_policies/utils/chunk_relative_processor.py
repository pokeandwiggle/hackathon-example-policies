# Copyright 2025 Poke & Wiggle GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chunk-relative UMI-delta ProcessorStep (TRI LBM style, arXiv:2507.05331).

At training time, converts absolute TCP actions (16-dim) to chunk-relative
UMI-delta actions (20-dim).  Each action step k in the predicted horizon is
expressed relative to the TCP pose at the *start of the chunk* (the most recent
observation), NOT relative to the TCP at timestep t+k.

This matches TRI's LBM paper §4.4.2: "We represent robot actions in end-effector
space.  Each robot action is the relative pose expressed in the frame of
observation at timestep t."

Layout:
    Input  (abs TCP, 16-dim): [L_pos(3), L_quat(4), R_pos(3), R_quat(4), grip_L, grip_R]
    Output (UMI delta, 20-dim): [L_dpos(3), L_rot6d(6), R_dpos(3), R_rot6d(6), grip_L, grip_R]

At inference time this step is a no-op (no action key in the transition).
The reverse operation (UMI delta → absolute TCP) is handled by
``ActionTranslator._umi_delta_tcp`` in the deployment code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry

from ..data_ops.utils.rotation_6d import (
    compute_relative_transform_6d_torch,
    quat_to_6d_torch,
)
from .action_order import (
    DUAL_ABS_LEFT_POS_IDXS,
    DUAL_ABS_LEFT_QUAT_IDXS,
    DUAL_ABS_RIGHT_POS_IDXS,
    DUAL_ABS_RIGHT_QUAT_IDXS,
    UMI_ACTION_DIM,
    UMI_LEFT_GRIPPER_IDX,
    UMI_LEFT_POS_IDXS,
    UMI_LEFT_ROT6D_IDXS,
    UMI_RIGHT_GRIPPER_IDX,
    UMI_RIGHT_POS_IDXS,
    UMI_RIGHT_ROT6D_IDXS,
)
from .constants import OBSERVATION_STATE

# Absolute TCP action dimension: pos(3) + quat(4) per arm + 2 grippers = 16
ABS_TCP_ACTION_DIM = 16
# Gripper indices in the 16-dim absolute TCP layout
ABS_TCP_LEFT_GRIPPER_IDX = 14
ABS_TCP_RIGHT_GRIPPER_IDX = 15


def abs_tcp_to_chunk_relative_umi_delta(
    abs_actions: Tensor,
    ref_pos_left: Tensor,
    ref_rot6d_left: Tensor,
    ref_pos_right: Tensor,
    ref_rot6d_right: Tensor,
) -> Tensor:
    """Convert absolute TCP actions to chunk-relative UMI-delta actions.

    All action steps are expressed relative to the *same* reference TCP (the
    observation at the start of the chunk).

    Args:
        abs_actions: Absolute TCP actions, shape ``(..., H, 16)``.
        ref_pos_left: Left arm reference position, shape ``(..., 3)``.
        ref_rot6d_left: Left arm reference 6D rotation, shape ``(..., 6)``.
        ref_pos_right: Right arm reference position, shape ``(..., 3)``.
        ref_rot6d_right: Right arm reference 6D rotation, shape ``(..., 6)``.

    Returns:
        UMI-delta actions, shape ``(..., H, 20)``.
    """
    leading = abs_actions.shape[:-2]
    H = abs_actions.shape[-2]

    umi = abs_actions.new_zeros(*leading, H, UMI_ACTION_DIM)

    # Extract targets from absolute actions (per step in horizon)
    tgt_pos_l = abs_actions[..., DUAL_ABS_LEFT_POS_IDXS]  # (..., H, 3)
    tgt_quat_l = abs_actions[..., DUAL_ABS_LEFT_QUAT_IDXS]  # (..., H, 4)
    tgt_pos_r = abs_actions[..., DUAL_ABS_RIGHT_POS_IDXS]  # (..., H, 3)
    tgt_quat_r = abs_actions[..., DUAL_ABS_RIGHT_QUAT_IDXS]  # (..., H, 4)

    # Convert target quaternions to 6D rotation
    tgt_rot6d_l = quat_to_6d_torch(tgt_quat_l)  # (..., H, 6)
    tgt_rot6d_r = quat_to_6d_torch(tgt_quat_r)  # (..., H, 6)

    # Broadcast reference poses across the horizon dimension
    # ref_pos_left: (..., 3) → (..., 1, 3)
    ref_pos_l = ref_pos_left.unsqueeze(-2).expand_as(tgt_pos_l)
    ref_r6d_l = ref_rot6d_left.unsqueeze(-2).expand_as(tgt_rot6d_l)
    ref_pos_r = ref_pos_right.unsqueeze(-2).expand_as(tgt_pos_r)
    ref_r6d_r = ref_rot6d_right.unsqueeze(-2).expand_as(tgt_rot6d_r)

    # Compute chunk-relative deltas
    delta_pos_l, delta_rot6d_l = compute_relative_transform_6d_torch(
        ref_pos_l, ref_r6d_l, tgt_pos_l, tgt_rot6d_l
    )
    delta_pos_r, delta_rot6d_r = compute_relative_transform_6d_torch(
        ref_pos_r, ref_r6d_r, tgt_pos_r, tgt_rot6d_r
    )

    # Assemble 20-dim UMI delta
    umi[..., UMI_LEFT_POS_IDXS] = delta_pos_l
    umi[..., UMI_LEFT_ROT6D_IDXS] = delta_rot6d_l
    umi[..., UMI_RIGHT_POS_IDXS] = delta_pos_r
    umi[..., UMI_RIGHT_ROT6D_IDXS] = delta_rot6d_r
    umi[..., UMI_LEFT_GRIPPER_IDX] = abs_actions[..., ABS_TCP_LEFT_GRIPPER_IDX]
    umi[..., UMI_RIGHT_GRIPPER_IDX] = abs_actions[..., ABS_TCP_RIGHT_GRIPPER_IDX]

    return umi


@dataclass
@ProcessorStepRegistry.register(name="abs_tcp_to_chunk_relative_processor")
class AbsTcpToChunkRelativeStep(ProcessorStep):
    """Convert absolute TCP actions → chunk-relative UMI-delta at training time.

    During preprocessing the ``ACTION`` tensor is transformed from 16-dim
    absolute TCP to 20-dim chunk-relative UMI-delta, using the current
    observation TCP as the reference frame.

    During inference (no action in the transition) this is a no-op.

    Args:
        obs_tcp_left_pos_indices: Indices into ``observation.state`` for
            the left arm TCP position (3 consecutive values).
        obs_tcp_left_quat_indices: Indices for left arm TCP quaternion
            (4 consecutive xyzw values).
        obs_tcp_right_pos_indices: Indices for right arm TCP position.
        obs_tcp_right_quat_indices: Indices for right arm TCP quaternion.
    """

    obs_tcp_left_pos_indices: list[int] = field(default_factory=list)
    obs_tcp_left_quat_indices: list[int] = field(default_factory=list)
    obs_tcp_right_pos_indices: list[int] = field(default_factory=list)
    obs_tcp_right_quat_indices: list[int] = field(default_factory=list)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = dict(transition)

        action = new_transition.get(TransitionKey.ACTION)
        if action is None or not isinstance(action, torch.Tensor):
            return new_transition  # inference pre-process or missing action

        # Verify this is an absolute TCP action (16-dim last axis)
        if action.shape[-1] != ABS_TCP_ACTION_DIM:
            return new_transition

        # ── Extract reference TCP from observation.state ────────────
        obs_dict = new_transition.get(TransitionKey.OBSERVATION, {})
        obs_state = obs_dict.get(OBSERVATION_STATE)
        if obs_state is None:
            raise RuntimeError(
                "AbsTcpToChunkRelativeStep requires observation.state in the "
                "transition but it was not found."
            )

        # obs_state shape:
        #   training:  (B, n_obs_steps, state_dim)  or  (n_obs_steps, state_dim)
        #   inference: (B, state_dim)  or  (state_dim,)
        # We want the LAST observation step (most recent).
        if obs_state.ndim >= 3:
            # (B, n_obs_steps, D) → take last obs step
            ref_state = obs_state[:, -1, :]  # (B, D)
        elif obs_state.ndim == 2:
            # Could be (n_obs_steps, D) unbatched or (B, D) inference
            # If action is 3-dim (B, H, 16), obs_state must be batched
            if action.ndim == 3:
                ref_state = obs_state  # (B, D) — single obs step
            else:
                ref_state = obs_state[-1, :]  # (n_obs_steps, D) → last
        else:
            ref_state = obs_state  # (D,)

        # Extract TCP quaternion poses from observation.state
        ref_pos_l = ref_state[..., self.obs_tcp_left_pos_indices]  # (..., 3)
        ref_quat_l = ref_state[..., self.obs_tcp_left_quat_indices]  # (..., 4)
        ref_pos_r = ref_state[..., self.obs_tcp_right_pos_indices]  # (..., 3)
        ref_quat_r = ref_state[..., self.obs_tcp_right_quat_indices]  # (..., 4)

        # Convert reference quaternions to 6D rotations
        ref_rot6d_l = quat_to_6d_torch(ref_quat_l)  # (..., 6)
        ref_rot6d_r = quat_to_6d_torch(ref_quat_r)  # (..., 6)

        # Convert abs TCP → chunk-relative UMI delta
        umi_action = abs_tcp_to_chunk_relative_umi_delta(
            action, ref_pos_l, ref_rot6d_l, ref_pos_r, ref_rot6d_r
        )

        new_transition[TransitionKey.ACTION] = umi_action
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Update action shape from 16-dim (abs TCP) to 20-dim (UMI delta)."""
        action_feats = features.get(PipelineFeatureType.ACTION, {})
        if "action" in action_feats:
            old_feat = action_feats["action"]
            action_feats["action"] = PolicyFeature(
                type=old_feat.type,
                shape=[UMI_ACTION_DIM],
            )
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "obs_tcp_left_pos_indices": self.obs_tcp_left_pos_indices,
            "obs_tcp_left_quat_indices": self.obs_tcp_left_quat_indices,
            "obs_tcp_right_pos_indices": self.obs_tcp_right_pos_indices,
            "obs_tcp_right_quat_indices": self.obs_tcp_right_quat_indices,
        }

    def state_dict(self) -> dict[str, Tensor]:
        # No learnable state; indices are config-only
        return {}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        pass
