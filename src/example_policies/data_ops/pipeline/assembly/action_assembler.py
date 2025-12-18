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

import dataclasses

import numpy as np

from ...config.pipeline_config import ActionMode, PipelineConfig
from ...utils import delta_ops

# Absolute (non-delta) action sources
ABS_SPECS = {
    ActionMode.TCP: ("des_tcp_left", "des_tcp_right"),
    ActionMode.TELEOP: ("des_tcp_left", "des_tcp_right"),
    ActionMode.JOINT: ("des_joint_left", "des_joint_right"),
}
# Delta specs: (left_key, right_key, delta_fn)
DELTA_SPECS = {
    ActionMode.DELTA_TCP: (
        "des_tcp_left",
        "des_tcp_right",
        delta_ops.pos_quat_delta,
    ),
    ActionMode.DELTA_JOINT: (
        "des_joint_left",
        "des_joint_right",
        delta_ops.joint_delta,
    ),
}


@dataclasses.dataclass
class LastCommand:
    left: np.ndarray
    right: np.ndarray


class ActionAssembler:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def reset(self):
        pass

    def assemble(self, parsed_frame: dict, last_abs_command: LastCommand | None):
        action_level = self.config.action_level

        # Invert grippers to match training convention
        grip_l = 1.0 - parsed_frame["des_gripper_left"][0]
        grip_r = 1.0 - parsed_frame["des_gripper_right"][0]

        left_action = None
        right_action = None

        if action_level in ABS_SPECS:
            left_key, right_key = ABS_SPECS[action_level]
            left_abs = parsed_frame[left_key]
            right_abs = parsed_frame[right_key]
            left_action, right_action = left_abs, right_abs
            new_last = LastCommand(left=left_abs.copy(), right=right_abs.copy())

        elif action_level in DELTA_SPECS:
            left_key, right_key, delta_fn = DELTA_SPECS[action_level]
            left_abs = parsed_frame[left_key]
            right_abs = parsed_frame[right_key]

            # Initialize history with first absolute pose
            if last_abs_command is None:
                last_abs_command = LastCommand(left=left_abs, right=right_abs)

            # Compute step-to-step deltas (previous -> current)
            left_action = delta_fn(last_abs_command.left, left_abs)
            right_action = delta_fn(last_abs_command.right, right_abs)

            # UPDATE history to current absolute
            new_last = LastCommand(left=left_abs.copy(), right=right_abs.copy())
        else:
            raise NotImplementedError(f"Unsupported action level: {action_level}")

        action_parts = [left_action, right_action, [grip_l, grip_r]]

        # Termination signal
        if self.config.requires_termination_signal():
            action_parts.append([0])

        action_vec = np.concatenate(action_parts, dtype=np.float32)

        return {"action": action_vec}, new_last
