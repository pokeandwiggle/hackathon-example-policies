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

from example_policies import data_constants as dc

from ..data_ops.utils.geometric import axis_angle_to_quat_torch, quat_mul_torch
from .utils.action_mode import ActionMode

_TCP_DELTA_SPECS = (
    {
        "abs_pos": dc.DUAL_LEFT_POS_IDXS,
        "abs_quat": dc.DUAL_LEFT_QUAT_IDXS,
        "delta_pos": dc.DUAL_DELTA_LEFT_POS_IDXS,
        "delta_aa": dc.DUAL_DELTA_LEFT_ROT_IDXS,
    },
    {
        "abs_pos": dc.DUAL_RIGHT_POS_IDXS,
        "abs_quat": dc.DUAL_RIGHT_QUAT_IDXS,
        "delta_pos": dc.DUAL_DELTA_RIGHT_POS_IDXS,
        "delta_aa": dc.DUAL_DELTA_RIGHT_ROT_IDXS,
    },
)


class ActionTranslator:
    def __init__(self, cfg):
        self.last_action = None
        self.action_mode: ActionMode | None = ActionMode.parse_action_mode(cfg)
        self._state_feature_names = []
        if getattr(cfg, "metadata", None):
            self._state_feature_names = cfg.metadata["features"]["observation.state"][
                "names"
            ]

        # Precompute indices for absolute tcp + grippers inside observation.state
        self.state_info_idxs = self.compute_state_info_indices(self.action_mode)

    def compute_state_info_indices(self, action_mode: ActionMode):
        required_names = []
        if action_mode in (ActionMode.ABS_TCP, ActionMode.DELTA_TCP):
            required_names.extend([f"tcp_left_pos_{i}" for i in "xyz"])
            required_names.extend([f"tcp_left_quat_{i}" for i in "xyzw"])
            required_names.extend([f"tcp_right_pos_{i}" for i in "xyz"])
            required_names.extend([f"tcp_right_quat_{i}" for i in "xyzw"])
        elif action_mode in (ActionMode.ABS_JOINT, ActionMode.DELTA_JOINT):
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
        action = action.clone()
        action[:, dc.LEFT_GRIPPER_IDX] = 1.0 - action[:, dc.LEFT_GRIPPER_IDX]
        action[:, dc.RIGHT_GRIPPER_IDX] = 1.0 - action[:, dc.RIGHT_GRIPPER_IDX]

        # Invert gripper action values to match the robot's open/close convention:
        # incoming actions use "close=1", but robot expects "open=1"
        if self.action_mode == ActionMode.DELTA_TCP:
            return self._delta_tcp(action, observation)
        if self.action_mode == ActionMode.ABS_TCP:
            return self._absolute_tcp(action)
        if self.action_mode == ActionMode.DELTA_JOINT:
            return self._delta_joint(action, observation)
        if self.action_mode == ActionMode.ABS_JOINT:
            return self._absolute_joint(action)
        raise RuntimeError(f"Unknown action mode: {self.action_mode}")

    def _init_last_action_from_observation(self, observation: dict, device):
        state = observation["observation.state"]
        if self.state_info_idxs is None:
            # Fallback legacy: trust leading slice
            last_action = state[:, :14]
        else:
            last_action = state[:, self.state_info_idxs]
        last_action = last_action.to(device)
        return last_action

    def _delta_tcp(self, action: torch.Tensor, observation: dict) -> torch.Tensor:
        """
        Convert delta TCP (pos + axis-angle deltas + grippers) into absolute TCP action.
        Simplified: loop over left/right specs, no repeated boilerplate.
        """
        if self.last_action is None:
            self.last_action = self._init_last_action_from_observation(
                observation, action.device
            )

        abs_pose = self.last_action.clone()

        for spec in _TCP_DELTA_SPECS:
            # Position
            abs_pose[0, spec["abs_pos"]] += action[0, spec["delta_pos"]]

            # Orientation (compose properly instead of linear add for robustness)
            last_q = abs_pose[0, spec["abs_quat"]]
            delta_rv = action[0, spec["delta_aa"]]
            # Convert rotvec to quaternion and multiply using centralized utilities
            delta_q = axis_angle_to_quat_torch(delta_rv)
            new_q = quat_mul_torch(last_q, delta_q)
            abs_pose[0, spec["abs_quat"]] = new_q

        # Build full 16-dim absolute action tensor
        full_abs = torch.zeros((1, 16), device=action.device, dtype=abs_pose.dtype)
        full_abs[:, :14] = abs_pose
        # Grippers (copy from delta action; indices in dc.* refer to full layout)
        full_abs[0, dc.LEFT_GRIPPER_IDX] = action[0, dc.LEFT_GRIPPER_IDX]
        full_abs[0, dc.RIGHT_GRIPPER_IDX] = action[0, dc.RIGHT_GRIPPER_IDX]

        # Normalize / fix quaternion signs
        full_abs = self._absolute_tcp(full_abs)

        # Update pose cache (pose only)
        self.last_action = full_abs[:, :14].clone()
        return full_abs

    def _absolute_tcp(self, action: torch.Tensor) -> torch.Tensor:
        action = _normalize_quats(action.clone())
        # action = _positive_quats(action)
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
    action[:, dc.DUAL_LEFT_QUAT_IDXS] = F.normalize(
        action[:, dc.DUAL_LEFT_QUAT_IDXS], p=2, dim=-1, eps=eps
    )
    action[:, dc.DUAL_RIGHT_QUAT_IDXS] = F.normalize(
        action[:, dc.DUAL_RIGHT_QUAT_IDXS], p=2, dim=-1, eps=eps
    )
    return action


def _positive_quats(action: torch.Tensor) -> torch.Tensor:
    # Ensure quaternion w component is non-negative
    if action[0, dc.DUAL_LEFT_QUAT_IDXS][-1] < 0:
        action[:, dc.DUAL_LEFT_QUAT_IDXS] = -action[:, dc.DUAL_LEFT_QUAT_IDXS]
    if action[0, dc.DUAL_RIGHT_QUAT_IDXS][-1] < 0:
        action[:, dc.DUAL_RIGHT_QUAT_IDXS] = -action[:, dc.DUAL_RIGHT_QUAT_IDXS]
    return action
