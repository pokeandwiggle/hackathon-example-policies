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
        self.gripper_active_counter = self.cfg.grace_period_frames
        self_prior_gripper = None

    def _is_paused(self, joint_velocity: np.ndarray) -> bool:
        """Checks if the robot is in a paused state."""
        if np.sum(np.abs(joint_velocity)) < self.cfg.pause_velocity:
            self.pause_detection_counter += 1
        else:
            self.pause_detection_counter = 0
        return self.pause_detection_counter >= self.cfg.max_pause_frames

    def _is_gripper_active(self, gripper_states: np.ndarray) -> bool:
        """
        Checks if the gripper is active (i.e., moving slowly or not at all).
        This state is used to determine if speed-boosted subsampling should apply.
        """
        if self_prior_gripper is None:
            self_prior_gripper = gripper_states
            return True
        delta = np.abs(gripper_states - self_prior_gripper)
        self_prior_gripper = gripper_states
        return np.any(delta > 0.01)

    def determine_targets(
        self, frame_buffer: FrameBuffer, frame_assembler: FrameParser
    ) -> List[DatasetType]:
        """
        Determines the target dataset(s) for a given frame based on its state.
        The logic is prioritized: pause state overrides all others.
        """
        assert frame_buffer.is_complete(), "Frame buffer is not complete"
        joint_velocity, gripper_states = frame_assembler.parse_velocities(frame_buffer)

        gripper_active = self._is_gripper_active(gripper_states)

        if self._is_paused(joint_velocity) and not gripper_active:
            return [DatasetType.PAUSE]

        if gripper_active:
            targets = [DatasetType.NO_SPEED_BOOST]
            if self.gripper_active_counter % self.cfg.boost_factor == 0:
                targets.append(DatasetType.MAIN)
            return targets

        return [DatasetType.MAIN]
