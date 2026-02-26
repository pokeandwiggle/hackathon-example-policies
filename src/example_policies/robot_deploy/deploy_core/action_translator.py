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


import torch
from torch.nn import functional as F

from ...data_ops.utils.geometric import add_delta_quaternion, axis_angle_to_quat_torch
from ...data_ops.utils.rotation_6d import (
    compose_transform_6d_torch,
    quat_to_6d_torch,
    rotation_6d_to_quat_torch,
)
from ...utils.action_order import (
    DUAL_ABS_LEFT_POS_IDXS,
    DUAL_ABS_LEFT_QUAT_IDXS,
    DUAL_ABS_RIGHT_POS_IDXS,
    DUAL_ABS_RIGHT_QUAT_IDXS,
    DUAL_DELTA_LEFT_POS_IDXS,
    DUAL_DELTA_LEFT_ROT_IDXS,
    DUAL_DELTA_RIGHT_POS_IDXS,
    DUAL_DELTA_RIGHT_ROT_IDXS,
    GET_LEFT_GRIPPER_IDX,
    GET_RIGHT_GRIPPER_IDX,
    UMI_LEFT_GRIPPER_IDX,
    UMI_LEFT_POS_IDXS,
    UMI_LEFT_ROT6D_IDXS,
    UMI_RIGHT_GRIPPER_IDX,
    UMI_RIGHT_POS_IDXS,
    UMI_RIGHT_ROT6D_IDXS,
    ActionMode,
)
from ...utils.constants import OBSERVATION_STATE

_TCP_DELTA_SPECS = (
    {
        "abs_pos": DUAL_ABS_LEFT_POS_IDXS,
        "abs_quat": DUAL_ABS_LEFT_QUAT_IDXS,
        "delta_pos": DUAL_DELTA_LEFT_POS_IDXS,
        "delta_aa": DUAL_DELTA_LEFT_ROT_IDXS,
    },
    {
        "abs_pos": DUAL_ABS_RIGHT_POS_IDXS,
        "abs_quat": DUAL_ABS_RIGHT_QUAT_IDXS,
        "delta_pos": DUAL_DELTA_RIGHT_POS_IDXS,
        "delta_aa": DUAL_DELTA_RIGHT_ROT_IDXS,
    },
)


class ActionTranslator:
    def __init__(self, cfg):
        self.last_action = None
        self.action_mode: ActionMode = ActionMode.parse_action_mode(cfg)
        self._state_feature_names = []
        if getattr(cfg, "metadata", None):
            self._state_feature_names = cfg.metadata["features"][OBSERVATION_STATE][
                "names"
            ]

        # Precompute indices for absolute tcp + grippers inside observation.state
        self.state_info_idxs = self.compute_state_info_indices(self.action_mode)

    def compute_state_info_indices(self, action_mode: ActionMode):
        required_names = []
        if action_mode in (
            ActionMode.TCP, ActionMode.DELTA_TCP, ActionMode.TELEOP, ActionMode.UMI_DELTA_TCP,
        ):
            required_names.extend([f"tcp_left_pos_{i}" for i in "xyz"])
            required_names.extend([f"tcp_left_quat_{i}" for i in "xyzw"])
            required_names.extend([f"tcp_right_pos_{i}" for i in "xyz"])
            required_names.extend([f"tcp_right_quat_{i}" for i in "xyzw"])
        elif action_mode in (ActionMode.JOINT, ActionMode.DELTA_JOINT):
            required_names.extend([f"joint_pos_left_{i}" for i in range(7)])
            required_names.extend([f"joint_pos_right_{i}" for i in range(7)])
        else:
            raise RuntimeError(f"Unknown action mode: {action_mode}")

        indices = []
        for name in required_names:
            # Legacy fallback: if names not found, return None
            if name not in self._state_feature_names:
                print(f"[ActionTranslator] Missing state feature: {name}")
                return None
            indices.append(self._state_feature_names.index(name))
        return indices

    def translate(self, action: torch.Tensor, observation: dict) -> torch.Tensor:
        """
        Dispatch to the correct transformation based on inferred action mode.
        """
        left_gripper_idx = GET_LEFT_GRIPPER_IDX(self.action_mode)
        right_gripper_idx = GET_RIGHT_GRIPPER_IDX(self.action_mode)

        action = action.clone()
        action[:, left_gripper_idx] = 1.0 - action[:, left_gripper_idx]
        action[:, right_gripper_idx] = 1.0 - action[:, right_gripper_idx]

        # Invert gripper action values to match the robot's open/close convention:
        # incoming actions use "close=1", but robot expects "open=1"
        if self.action_mode == ActionMode.UMI_DELTA_TCP:
            return self._umi_delta_tcp(action, observation)
        if self.action_mode == ActionMode.DELTA_TCP:
            return self._delta_tcp(action, observation)
        if self.action_mode in (ActionMode.TCP, ActionMode.TELEOP):
            return self._absolute_tcp(action)
        if self.action_mode == ActionMode.DELTA_JOINT:
            return self._delta_joint(action, observation)
        if self.action_mode == ActionMode.JOINT:
            return self._absolute_joint(action)
        raise RuntimeError(f"Unknown action mode: {self.action_mode}")

    def _init_last_action_from_observation(self, observation: dict, device):
        state = observation[OBSERVATION_STATE]
        if self.state_info_idxs is None:
            # Fallback legacy: trust leading slice
            last_action = state[:, :14]
        else:
            last_action = state[:, self.state_info_idxs]
        last_action = last_action.to(device)
        return last_action

    def _delta_tcp(self, delta_action: torch.Tensor, observation: dict) -> torch.Tensor:
        """
        Convert delta TCP (pos + axis-angle deltas + grippers) into absolute TCP action.
        Simplified: loop over left/right specs, no repeated boilerplate.
        """
        if self.last_action is None:
            self.last_action = self._init_last_action_from_observation(
                observation, delta_action.device
            )

        abs_pose = self.last_action.clone()

        for spec in _TCP_DELTA_SPECS:
            # Position
            abs_pose[0, spec["abs_pos"]] += delta_action[0, spec["delta_pos"]]

            # Orientation (compose properly instead of linear add for robustness)
            last_q = abs_pose[0, spec["abs_quat"]]
            delta_rv = delta_action[0, spec["delta_aa"]]
            # Convert rotvec to quaternion and multiply using centralized utilities
            delta_q = axis_angle_to_quat_torch(delta_rv)
            new_q = add_delta_quaternion(last_q, delta_q)
            abs_pose[0, spec["abs_quat"]] = new_q

        # Build full 16-dim absolute action tensor
        abs_action = torch.zeros(
            (1, 16), device=delta_action.device, dtype=abs_pose.dtype
        )
        abs_action[:, :14] = abs_pose

        # Grippers (copy from delta action; indices refer to full layout)
        abs_left_gripper_idx = GET_LEFT_GRIPPER_IDX(
            ActionMode.get_absolute_mode(self.action_mode)
        )
        delta_left_gripper_idx = GET_LEFT_GRIPPER_IDX(self.action_mode)
        abs_right_gripper_idx = GET_RIGHT_GRIPPER_IDX(
            ActionMode.get_absolute_mode(self.action_mode)
        )
        delta_right_gripper_idx = GET_RIGHT_GRIPPER_IDX(self.action_mode)

        abs_action[0, abs_left_gripper_idx] = delta_action[0, delta_left_gripper_idx]
        abs_action[0, abs_right_gripper_idx] = delta_action[0, delta_right_gripper_idx]

        # Normalize / fix quaternion signs
        abs_action = self._absolute_tcp(abs_action)

        # Update pose cache (pose only)
        self.last_action = abs_action[:, :14].clone()
        return abs_action

    def _umi_delta_tcp(
        self, umi_action: torch.Tensor, observation: dict
    ) -> torch.Tensor:
        """Convert UMI-delta actions (relative pos + 6D rotation) to absolute TCP.

        UMI-delta actions encode position and orientation relative to the current
        TCP pose.  This composes them back to absolute TCP using the TCP state
        from the observation.

        Args:
            umi_action: (1, 20) UMI-delta action.
            observation: Must contain ``observation.state`` with TCP poses.

        Returns:
            (1, 16) absolute TCP action: [L_pos(3), L_quat(4), R_pos(3), R_quat(4), grip_L, grip_R].
        """
        # Extract current TCP state from observation
        ref_state = self._init_last_action_from_observation(
            observation, umi_action.device
        )
        # ref_state shape: (1, 14): [L_pos(3), L_quat(4), R_pos(3), R_quat(4)]

        abs_action = torch.zeros(
            (1, 16), device=umi_action.device, dtype=umi_action.dtype
        )

        # --- Left arm ---
        ref_pos_l = ref_state[0, DUAL_ABS_LEFT_POS_IDXS]
        ref_quat_l = ref_state[0, DUAL_ABS_LEFT_QUAT_IDXS]
        ref_rot6d_l = quat_to_6d_torch(ref_quat_l)

        delta_pos_l = umi_action[0, UMI_LEFT_POS_IDXS]
        delta_rot6d_l = umi_action[0, UMI_LEFT_ROT6D_IDXS]

        abs_pos_l, abs_rot6d_l = compose_transform_6d_torch(
            ref_pos_l, ref_rot6d_l, delta_pos_l, delta_rot6d_l
        )
        abs_quat_l = rotation_6d_to_quat_torch(abs_rot6d_l)

        # --- Right arm ---
        ref_pos_r = ref_state[0, DUAL_ABS_RIGHT_POS_IDXS]
        ref_quat_r = ref_state[0, DUAL_ABS_RIGHT_QUAT_IDXS]
        ref_rot6d_r = quat_to_6d_torch(ref_quat_r)

        delta_pos_r = umi_action[0, UMI_RIGHT_POS_IDXS]
        delta_rot6d_r = umi_action[0, UMI_RIGHT_ROT6D_IDXS]

        abs_pos_r, abs_rot6d_r = compose_transform_6d_torch(
            ref_pos_r, ref_rot6d_r, delta_pos_r, delta_rot6d_r
        )
        abs_quat_r = rotation_6d_to_quat_torch(abs_rot6d_r)

        # --- Assemble 16-dim absolute action ---
        abs_action[0, DUAL_ABS_LEFT_POS_IDXS] = abs_pos_l
        abs_action[0, DUAL_ABS_LEFT_QUAT_IDXS] = abs_quat_l
        abs_action[0, DUAL_ABS_RIGHT_POS_IDXS] = abs_pos_r
        abs_action[0, DUAL_ABS_RIGHT_QUAT_IDXS] = abs_quat_r

        # Grippers
        abs_left_gripper_idx = GET_LEFT_GRIPPER_IDX(
            ActionMode.get_absolute_mode(self.action_mode)
        )
        abs_right_gripper_idx = GET_RIGHT_GRIPPER_IDX(
            ActionMode.get_absolute_mode(self.action_mode)
        )
        abs_action[0, abs_left_gripper_idx] = umi_action[0, UMI_LEFT_GRIPPER_IDX]
        abs_action[0, abs_right_gripper_idx] = umi_action[0, UMI_RIGHT_GRIPPER_IDX]

        # Normalize quaternions
        abs_action = self._absolute_tcp(abs_action)
        return abs_action

    def _absolute_tcp(self, action: torch.Tensor) -> torch.Tensor:
        action = _normalize_quats(action.clone())
        return action

    def _absolute_joint(self, action: torch.Tensor) -> torch.Tensor:
        return action

    def _delta_joint(self, action, observation):
        if self.last_action is None:
            self.last_action = self._init_last_action_from_observation(
                observation, action.device
            )

        abs_action = action.clone()
        abs_action[:, :14] += self.last_action
        self.last_action = abs_action[:, :14].clone()
        return abs_action


def _normalize_quats(action: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    action[:, DUAL_ABS_LEFT_QUAT_IDXS] = F.normalize(
        action[:, DUAL_ABS_LEFT_QUAT_IDXS], p=2, dim=-1, eps=eps
    )
    action[:, DUAL_ABS_RIGHT_QUAT_IDXS] = F.normalize(
        action[:, DUAL_ABS_RIGHT_QUAT_IDXS], p=2, dim=-1, eps=eps
    )
    return action
