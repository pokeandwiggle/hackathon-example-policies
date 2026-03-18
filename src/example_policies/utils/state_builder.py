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

"""Shared logic for building state feature specifications.

This module consolidates the logic for determining the order and structure of
observation.state features. Both the data pipeline (pipeline_config.py) and
deployment (observation_builder.py) use this to ensure consistency.
"""

from dataclasses import dataclass
from enum import Enum


class GripperType(Enum):
    """Type of gripper being used."""

    PANDA = "panda"
    ROBOTIQ = "robotiq"


class LastCommandStyle(Enum):
    """Naming convention for last-command features."""

    LAST_TCP = "last_tcp"        # last_tcp_left_pos_x (semantic names)
    LAST_COMMAND = "last_command"  # last_command_left_0  (indexed names)


@dataclass
class StateFeatureSpec:
    """Specification for what features to include in observation.state and their order."""

    # Joint state components
    include_joint_positions: bool = False
    include_joint_velocities: bool = False
    include_joint_efforts: bool = False

    # TCP pose
    include_tcp_poses: bool = True

    # Gripper types
    left_gripper: GripperType = GripperType.PANDA
    right_gripper: GripperType = GripperType.PANDA

    # Last command (for temporal context in delta policies)
    include_last_command: bool = False
    last_command_style: LastCommandStyle = LastCommandStyle.LAST_TCP

    @property
    def include_joint_state(self) -> bool:
        """Whether any joint state information is included."""
        return (
            self.include_joint_positions
            or self.include_joint_velocities
            or self.include_joint_efforts
        )

    def get_feature_names(self) -> list[str]:
        """Get the ordered list of feature names for observation.state.

        This defines the canonical order that must be used consistently
        during data collection and deployment.

        Returns:
            List of feature names in the order they appear in the state vector
        """
        state_names = []

        # Joint positions (14 elements: 7 left + 7 right)
        if self.include_joint_positions:
            state_names.extend([f"joint_pos_left_{i}" for i in range(7)])
            state_names.extend([f"joint_pos_right_{i}" for i in range(7)])

        # Joint velocities (14 elements: 7 left + 7 right)
        if self.include_joint_velocities:
            state_names.extend([f"joint_vel_left_{i}" for i in range(7)])
            state_names.extend([f"joint_vel_right_{i}" for i in range(7)])

        # Joint efforts (14 elements: 7 left + 7 right)
        if self.include_joint_efforts:
            state_names.extend([f"joint_eff_left_{i}" for i in range(7)])
            state_names.extend([f"joint_eff_right_{i}" for i in range(7)])

        # TCP poses (14 elements: 2 arms × 7 [xyz + xyzw])
        if self.include_tcp_poses:
            state_names.extend([f"tcp_left_pos_{i}" for i in "xyz"])
            state_names.extend([f"tcp_left_quat_{i}" for i in "xyzw"])
            state_names.extend([f"tcp_right_pos_{i}" for i in "xyz"])
            state_names.extend([f"tcp_right_quat_{i}" for i in "xyzw"])

        # Left gripper (1 width value; name encodes gripper type)
        match self.left_gripper:
            case GripperType.ROBOTIQ:
                state_names.append("robotiq_left")
            case GripperType.PANDA:
                state_names.append("gripper_left")
            case _:
                raise ValueError(f"Unsupported left gripper type: {self.left_gripper}")

        # Right gripper (1 width value; name encodes gripper type)
        match self.right_gripper:
            case GripperType.ROBOTIQ:
                state_names.append("robotiq_right")
            case GripperType.PANDA:
                state_names.append("gripper_right")
            case _:
                raise ValueError(
                    f"Unsupported right gripper type: {self.right_gripper}"
                )

        # Last command (14 elements: same layout as TCP poses)
        if self.include_last_command:
            if self.last_command_style == LastCommandStyle.LAST_TCP:
                state_names.extend([f"last_tcp_left_pos_{i}" for i in "xyz"])
                state_names.extend([f"last_tcp_left_quat_{i}" for i in "xyzw"])
                state_names.extend([f"last_tcp_right_pos_{i}" for i in "xyz"])
                state_names.extend([f"last_tcp_right_quat_{i}" for i in "xyzw"])
            else:
                state_names.extend([f"last_command_left_{i}" for i in range(7)])
                state_names.extend([f"last_command_right_{i}" for i in range(7)])

        return state_names

    @classmethod
    def from_feature_names(cls, feature_names: list[str]) -> "StateFeatureSpec":
        """Reverse-engineer the spec from a list of feature names.

        This is useful for deployment when loading a trained model that has
        metadata about what features it expects.

        Args:
            feature_names: List of feature names from metadata

        Returns:
            StateFeatureSpec that would produce these features
        """
        spec = cls()

        # Detect what features are present
        spec.include_joint_positions = any(
            "joint_pos_" in name for name in feature_names
        )
        spec.include_joint_velocities = any(
            "joint_vel_" in name for name in feature_names
        )
        spec.include_joint_efforts = any("joint_eff_" in name for name in feature_names)
        spec.include_tcp_poses = any("tcp_" in name for name in feature_names)
        has_last_command = any("last_command_" in name for name in feature_names)
        has_last_tcp = any("last_tcp_" in name for name in feature_names)
        spec.include_last_command = has_last_command or has_last_tcp
        if has_last_tcp:
            spec.last_command_style = LastCommandStyle.LAST_TCP
        elif has_last_command:
            spec.last_command_style = LastCommandStyle.LAST_COMMAND

        # Detect gripper types from feature names.
        # Current format: Robotiq → "robotiq_left", Panda → "gripper_left".
        # Legacy Robotiq format used "robotiq_left_0..5" (also matched here).
        if any(name.startswith("robotiq_left") for name in feature_names):
            spec.left_gripper = GripperType.ROBOTIQ
        else:
            spec.left_gripper = GripperType.PANDA

        if any(name.startswith("robotiq_right") for name in feature_names):
            spec.right_gripper = GripperType.ROBOTIQ
        else:
            spec.right_gripper = GripperType.PANDA

        return spec
