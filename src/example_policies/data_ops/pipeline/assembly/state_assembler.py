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

import numpy as np

from ...config.pipeline_config import PipelineConfig
from .action_assembler import LastCommand


class StateAssembler:
    """Assembles the final observation vector from parsed frame data."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def reset(self):
        # Pseudo Assembler Interface for Future
        pass

    def assemble(self, parsed_frame: dict, last_action: LastCommand) -> dict:
        state_components = []
        if self.config.include_joint_positions:
            state_components.append(parsed_frame["joint_data"]["position"])
        if self.config.include_joint_velocities:
            state_components.append(parsed_frame["joint_data"]["velocity"])
        if self.config.include_joint_efforts:
            state_components.append(parsed_frame["joint_data"]["effort"])

        if self.config.include_tcp_poses:
            state_components.append(parsed_frame["actual_tcp_left"])
            state_components.append(parsed_frame["actual_tcp_right"])

        # gripper_state layout: [left_joints..., right_joints...]
        # For Panda: [left_finger1, left_finger2, right_finger1, right_finger2]
        gs = parsed_frame["gripper_state"]
        from example_policies.utils.state_builder import GripperType

        raw_left_n = 2 if self.config.left_gripper == GripperType.PANDA else 6
        raw_right_n = 2 if self.config.right_gripper == GripperType.PANDA else 6
        left_raw = gs[:raw_left_n]
        right_raw = gs[raw_left_n : raw_left_n + raw_right_n]

        if self.config.use_single_gripper_value:
            # Sum both finger-joint positions into a single gripper-width scalar.
            left_gs = np.array([left_raw.sum()], dtype=gs.dtype)
            right_gs = np.array([right_raw.sum()], dtype=gs.dtype)
        else:
            left_gs = left_raw
            right_gs = right_raw
        state_components.append(left_gs)
        state_components.append(right_gs)

        if self.config.include_last_command:
            state_components.append(
                np.concatenate([last_action.left, last_action.right])
            )

        return {
            "observation.state": np.concatenate(state_components).astype(np.float32)
        }
