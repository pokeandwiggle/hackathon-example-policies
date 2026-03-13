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
from example_policies.utils.gripper import (
    robotiq_width_from_knuckle,
)
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
        # After joint reordering:
        #   Panda:   [left_finger1, left_finger2, right_finger1, right_finger2]
        #   Robotiq: [left_knuckle, right_knuckle]
        # We always convert to 1 width-in-meters value per side.
        gs = parsed_frame["gripper_state"]
        from example_policies.utils.state_builder import GripperType

        match self.config.left_gripper:
            case GripperType.PANDA:
                left_n = 2
            case GripperType.ROBOTIQ:
                left_n = 1
            case _:
                raise ValueError(
                    f"Unsupported left gripper type: {self.config.left_gripper}"
                )
        match self.config.right_gripper:
            case GripperType.PANDA:
                right_n = 2
            case GripperType.ROBOTIQ:
                right_n = 1
            case _:
                raise ValueError(
                    f"Unsupported right gripper type: {self.config.right_gripper}"
                )
        left_raw = gs[:left_n]
        right_raw = gs[left_n : left_n + right_n]

        # Panda: width = sum of both finger joint positions (metres)
        # Robotiq: convert knuckle position (rad) to width (metres)
        match self.config.left_gripper:
            case GripperType.PANDA:
                left_width = float(left_raw.sum())
            case GripperType.ROBOTIQ:
                left_width = robotiq_width_from_knuckle(float(left_raw[0]))
            case _:
                raise ValueError(
                    f"Unsupported left gripper type: {self.config.left_gripper}"
                )

        match self.config.right_gripper:
            case GripperType.PANDA:
                right_width = float(right_raw.sum())
            case GripperType.ROBOTIQ:
                right_width = robotiq_width_from_knuckle(float(right_raw[0]))
            case _:
                raise ValueError(
                    f"Unsupported right gripper type: {self.config.right_gripper}"
                )

        state_components.append(np.array([left_width], dtype=np.float32))
        state_components.append(np.array([right_width], dtype=np.float32))

        if self.config.include_last_command:
            state_components.append(
                np.concatenate([last_action.left, last_action.right])
            )

        return {
            "observation.state": np.concatenate(state_components).astype(np.float32)
        }
