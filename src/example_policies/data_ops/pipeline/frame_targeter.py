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

from typing import List

import numpy as np

from ..config import pipeline_config
from ..config.dataset_type import DatasetType
from .frame_buffer import FrameBuffer
from .frame_parser import FrameParser


class FrameTargeter:
    """
    Determines the category of a data frame based on its velocity profile.
    This class maintains state across frames to detect pauses or sustained
    gripper activity.
    """

    def __init__(
        self,
        cfg: pipeline_config.PipelineConfig,
    ):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.pause_detection_counter = self.cfg.max_pause_frames
        self.gripper_stationary_counter = self.cfg.max_pause_frames
        self._prior_gripper = None

    def _is_paused(self, joint_velocity: np.ndarray) -> bool:
        """Checks if the robot is in a paused state."""
        if np.sum(np.abs(joint_velocity)) < self.cfg.pause_velocity:
            self.pause_detection_counter += 1
        else:
            self.pause_detection_counter = 0
        return self.pause_detection_counter >= self.cfg.max_pause_frames

    def _is_gripper_stationary(self, gripper_states: np.ndarray) -> bool:
        """Checks if the gripper has been stationary for a configured duration."""
        if self._prior_gripper is None:
            self._prior_gripper = gripper_states
            # Assume stationary on the first frame
            return True

        delta = np.abs(gripper_states - self._prior_gripper)
        self._prior_gripper = gripper_states

        if np.any(delta > 1e-4):
            # Gripper moved, reset counter
            self.gripper_stationary_counter = 0
        else:
            # Gripper is still, increment counter
            self.gripper_stationary_counter += 1

        return self.gripper_stationary_counter >= self.cfg.max_pause_frames

    def determine_targets(
        self, frame_buffer: FrameBuffer, frame_assembler: FrameParser
    ) -> List[DatasetType]:
        """
        Determines the target dataset(s) for a given frame based on its state.
        The logic is prioritized: pause state overrides all others.
        """
        assert frame_buffer.is_complete(), "Frame buffer is not complete"
        joint_velocity, gripper_states = frame_assembler.parse_velocities(frame_buffer)

        # Check for stationary state first for pause detection
        is_gripper_still = self._is_gripper_stationary(gripper_states)
        is_robot_paused = self._is_paused(joint_velocity)

        if is_robot_paused and is_gripper_still:
            return [DatasetType.PAUSE]

        # Check for active gripper movement for speed boost logic
        if not is_gripper_still:
            targets = [DatasetType.NO_SPEED_BOOST]
            if self.gripper_stationary_counter % self.cfg.boost_factor == 0:
                targets.append(DatasetType.MAIN)
            return targets

        return [DatasetType.MAIN]
