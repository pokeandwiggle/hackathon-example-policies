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
from ...utils.action_order import ActionMode
from ..config.rosbag_topics import RosTopicEnum
from ..utils import message_parsers as rmp
from ..utils.geometric import positive_quat
from .frame_buffer import FrameBuffer


class FrameParser:
    """
    A stateless parser that converts a raw FrameBuffer into a structured
    dictionary of absolute data values.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def parse_velocities(self, frame_buffer: FrameBuffer):
        """Parses only the joint velocities for quick pause detection."""
        assert frame_buffer.is_complete(), "Frame buffer is not complete"
        msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.ACTUAL_JOINT_STATE)
        _, joint_velocity, gripper_states = rmp.parse_joints(
            self.config, msg_data, schema_name
        )
        return joint_velocity, gripper_states

    def parse_frame(self, frame_buffer: FrameBuffer) -> dict:
        """Parses a complete frame buffer into a structured dictionary."""
        assert frame_buffer.is_complete(), "Frame buffer is not complete"

        parsed_frame = {}
        parsed_frame.update(self._parse_state(frame_buffer))
        parsed_frame.update(self._parse_desired(frame_buffer))
        parsed_frame.update(self._parse_images(frame_buffer))

        return parsed_frame

    def _parse_state(self, frame_buffer) -> dict:
        state_frame = {}
        msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.ACTUAL_JOINT_STATE)
        joint_data, _, gripper_state = rmp.parse_joints(
            self.config, msg_data, schema_name
        )
        state_frame["joint_data"] = joint_data
        state_frame["gripper_state"] = gripper_state

        if self.config.include_tcp_poses or self.config.action_level in [
            ActionMode.TCP,
            ActionMode.TELEOP,
            ActionMode.DELTA_TCP,
        ]:
            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.ACTUAL_TCP_LEFT)
            raw_pose = rmp.parse_pose(self.config, msg_data, schema_name)
            state_frame["actual_tcp_left"] = positive_quat(raw_pose)

            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.ACTUAL_TCP_RIGHT)
            raw_pose = rmp.parse_pose(self.config, msg_data, schema_name)
            state_frame["actual_tcp_right"] = positive_quat(raw_pose)
        return state_frame

    def _parse_desired(self, frame_buffer) -> dict:
        desired_frame = {}
        msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DES_GRIPPER_LEFT)
        desired_frame["des_gripper_left"] = rmp.parse_array(
            self.config, msg_data, schema_name
        )

        msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DES_GRIPPER_RIGHT)
        desired_frame["des_gripper_right"] = rmp.parse_array(
            self.config, msg_data, schema_name
        )

        if self.config.action_level in [
            ActionMode.TCP,
            ActionMode.TELEOP,
            ActionMode.DELTA_TCP,
        ]:
            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DES_TCP_LEFT)
            raw_pose = rmp.parse_desired_tcp(self.config, msg_data, schema_name)
            desired_frame["des_tcp_left"] = positive_quat(raw_pose)

            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DES_TCP_RIGHT)
            raw_pose = rmp.parse_desired_tcp(self.config, msg_data, schema_name)
            desired_frame["des_tcp_right"] = positive_quat(raw_pose)

        elif self.config.action_level in [ActionMode.JOINT, ActionMode.DELTA_JOINT]:
            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DES_JOINT_LEFT)
            desired_frame["des_joint_left"] = rmp.parse_joint_waypoint(
                self.config, msg_data, schema_name, "left"
            )

            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DES_JOINT_RIGHT)
            desired_frame["des_joint_right"] = rmp.parse_joint_waypoint(
                self.config, msg_data, schema_name, "right"
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
        images["observation.images.rgb_static"] = rmp.parse_image(
            self.config, msg_data, schema_name
        )

        if self.config.include_rgb_images:
            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.RGB_LEFT_IMAGE)
            images["observation.images.rgb_left"] = rmp.parse_image(
                self.config, msg_data, schema_name
            )

            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.RGB_RIGHT_IMAGE)
            images["observation.images.rgb_right"] = rmp.parse_image(
                self.config, msg_data, schema_name
            )

        if self.config.include_depth_images:
            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DEPTH_LEFT_IMAGE)
            images["observation.images.depth_left"] = rmp.parse_image(
                self.config, msg_data, schema_name
            )

            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DEPTH_RIGHT_IMAGE)
            images["observation.images.depth_right"] = rmp.parse_image(
                self.config, msg_data, schema_name
            )

        if self.config.include_depth_images:
            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DEPTH_LEFT_IMAGE)
            images["observation.images.depth_left"] = rmp.parse_image(
                self.config, msg_data, schema_name
            )

            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DEPTH_RIGHT_IMAGE)
            images["observation.images.depth_right"] = rmp.parse_image(
                self.config, msg_data, schema_name
            )

        if self.config.include_depth_images:
            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DEPTH_LEFT_IMAGE)
            images["observation.images.depth_left"] = rmp.parse_image(
                self.config, msg_data, schema_name
            )

            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DEPTH_RIGHT_IMAGE)
            images["observation.images.depth_right"] = rmp.parse_image(
                self.config, msg_data, schema_name
            )
        return images
