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


from ..config.pipeline_config import PipelineConfig
from ..config.rosbag_topics import RosTopicEnum
from ..utils.geometric import continuous_quat
from ...utils.embodiment import EmbodimentJointConfig
from . import message_parsers as rmp
from .frame_buffer import FrameBuffer
from .message_parsers import ImageParser


class FrameParser:
    """
    Parser that converts a raw FrameBuffer into a structured dictionary
    of absolute data values with quaternion continuity tracking.

    Tracks previous quaternion values to ensure smooth transitions between
    frames, preventing jumps when rotations pass through w ≈ 0 configurations.
    Only tracks quaternions for features that are enabled in the config.
    """

    def __init__(
        self,
        config: PipelineConfig,
        embodiment: EmbodimentJointConfig | None = None,
    ):
        self.config = config
        self.embodiment = embodiment
        self._prev_actual_tcp_left = None
        self._prev_actual_tcp_right = None
        self._prev_des_tcp_left = None
        self._prev_des_tcp_right = None
        self._image_parsers: dict[str, ImageParser] = {}

    def reset(self):
        """Reset quaternion tracking state for a new episode."""
        self._prev_actual_tcp_left = None
        self._prev_actual_tcp_right = None
        self._prev_des_tcp_left = None
        self._prev_des_tcp_right = None
        for p in self._image_parsers.values():
            p.reset()

    def parse_velocities(self, frame_buffer: FrameBuffer):
        """Parses only the joint velocities for quick pause detection."""
        assert frame_buffer.is_complete(), "Frame buffer is not complete"
        msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.ACTUAL_JOINT_STATE)
        _, joint_velocity, gripper_states = rmp.parse_joints(
            self.config, msg_data, schema_name, embodiment=self.embodiment
        )
        return joint_velocity, gripper_states

    def parse_frame(self, frame_buffer: FrameBuffer) -> dict | None:
        """Parses a complete frame buffer into a structured dictionary.

        Returns None if any image cannot be decoded (e.g. AV1 P-frame
        received before the first keyframe).
        """
        assert frame_buffer.is_complete(), "Frame buffer is not complete"

        parsed_frame = {}
        parsed_frame.update(self._parse_state(frame_buffer))
        parsed_frame.update(self._parse_desired(frame_buffer))

        images = self._parse_images(frame_buffer)
        if any(v is None for v in images.values()):
            return None
        parsed_frame.update(images)

        return parsed_frame

    def _parse_state(self, frame_buffer) -> dict:
        state_frame = {}
        msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.ACTUAL_JOINT_STATE)
        joint_data, _, gripper_state = rmp.parse_joints(
            self.config, msg_data, schema_name, embodiment=self.embodiment
        )
        state_frame["joint_data"] = joint_data
        state_frame["gripper_state"] = gripper_state

        if self.config.requires_tcp_poses():
            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.ACTUAL_TCP_LEFT)
            raw_pose = rmp.parse_pose(self.config, msg_data, schema_name)
            state_frame["actual_tcp_left"] = continuous_quat(
                raw_pose, self._prev_actual_tcp_left
            )
            self._prev_actual_tcp_left = state_frame["actual_tcp_left"].copy()

            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.ACTUAL_TCP_RIGHT)
            raw_pose = rmp.parse_pose(self.config, msg_data, schema_name)
            state_frame["actual_tcp_right"] = continuous_quat(
                raw_pose, self._prev_actual_tcp_right
            )
            self._prev_actual_tcp_right = state_frame["actual_tcp_right"].copy()
        return state_frame

    def _parse_desired(self, frame_buffer) -> dict:
        desired_frame = {}
        msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DES_GRIPPER_LEFT)
        desired_frame["des_gripper_left"] = rmp.parse_gripper(
            self.config, msg_data, schema_name
        )

        msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DES_GRIPPER_RIGHT)
        desired_frame["des_gripper_right"] = rmp.parse_gripper(
            self.config, msg_data, schema_name
        )

        if self.config.is_tcp_action():
            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DES_TCP_LEFT)
            raw_pose = rmp.parse_desired_tcp(self.config, msg_data, schema_name)
            desired_frame["des_tcp_left"] = continuous_quat(
                raw_pose, self._prev_des_tcp_left
            )
            self._prev_des_tcp_left = desired_frame["des_tcp_left"].copy()

            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DES_TCP_RIGHT)
            raw_pose = rmp.parse_desired_tcp(self.config, msg_data, schema_name)
            desired_frame["des_tcp_right"] = continuous_quat(
                raw_pose, self._prev_des_tcp_right
            )
            self._prev_des_tcp_right = desired_frame["des_tcp_right"].copy()

        elif self.config.is_joint_action():
            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DES_JOINT_LEFT)
            desired_frame["des_joint_left"] = rmp.parse_target(
                self.config,
                msg_data,
                schema_name,
                "left",
                embodiment=self.embodiment,
            )

            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DES_JOINT_RIGHT)
            desired_frame["des_joint_right"] = rmp.parse_target(
                self.config,
                msg_data,
                schema_name,
                "right",
                embodiment=self.embodiment,
            )
        else:
            raise NotImplementedError(
                f"Action level {self.config.action_level} not implemented"
            )

        return desired_frame

    def _parse_images(self, frame_buffer) -> dict:
        """Assembles the dictionary of image futures."""
        images = {}
        # Static camera is always included
        msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.RGB_STATIC_IMAGE)
        images["observation.images.rgb_static"] = self._image_parser(
            "rgb_static"
        ).parse(msg_data, schema_name)

        if self.config.include_rgb_images:
            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.RGB_LEFT_IMAGE)
            images["observation.images.rgb_left"] = self._image_parser(
                "rgb_left"
            ).parse(msg_data, schema_name)

            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.RGB_RIGHT_IMAGE)
            images["observation.images.rgb_right"] = self._image_parser(
                "rgb_right"
            ).parse(msg_data, schema_name)

        if self.config.include_depth_images:
            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DEPTH_LEFT_IMAGE)
            images["observation.images.depth_left"] = self._image_parser(
                "depth_left"
            ).parse(msg_data, schema_name)

            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DEPTH_RIGHT_IMAGE)
            images["observation.images.depth_right"] = self._image_parser(
                "depth_right"
            ).parse(msg_data, schema_name)
        return images

    def _image_parser(self, key: str) -> ImageParser:
        if key not in self._image_parsers:
            self._image_parsers[key] = ImageParser(self.config)
        return self._image_parsers[key]
