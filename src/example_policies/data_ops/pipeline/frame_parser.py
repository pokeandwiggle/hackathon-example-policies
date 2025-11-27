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


from ...utils.action_order import ActionMode
from ..config.pipeline_config import PipelineConfig
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

        # Parse joint velocities from both robots
        msg_data_left, schema_name_left = frame_buffer.get_msg(
            RosTopicEnum.ACTUAL_JOINT_LEFT
        )
        _, joint_velocity_left = rmp.parse_joints(
            self.config, msg_data_left, schema_name_left
        )

        msg_data_right, schema_name_right = frame_buffer.get_msg(
            RosTopicEnum.ACTUAL_JOINT_RIGHT
        )
        _, joint_velocity_right = rmp.parse_joints(
            self.config, msg_data_right, schema_name_right
        )

        # Combine velocities
        import numpy as np

        joint_velocity = np.concatenate([joint_velocity_left, joint_velocity_right])
        gripper_states = np.concatenate([[0, 0],[0, 0]])

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
        """Parse robot state from separate joint topics."""
        state_frame = {}

        # Parse joint states from both robots separately
        msg_data_left, schema_name_left = frame_buffer.get_msg(
            RosTopicEnum.ACTUAL_JOINT_LEFT
        )
        joint_data_left, _ = rmp.parse_joints(
            self.config, msg_data_left, schema_name_left
        )

        msg_data_right, schema_name_right = frame_buffer.get_msg(
            RosTopicEnum.ACTUAL_JOINT_RIGHT
        )
        joint_data_right, _ = rmp.parse_joints(
            self.config, msg_data_right, schema_name_right
        )

        gripper_state_left_msg = frame_buffer.get_msg(RosTopicEnum.DES_GRIPPER_LEFT)
        gripper_state_left = rmp.parse_gripper_state(
            self.config, gripper_state_left_msg[0], gripper_state_left_msg[1]
        )

        gripper_state_right_msg = frame_buffer.get_msg(RosTopicEnum.DES_GRIPPER_RIGHT)
        gripper_state_right = rmp.parse_gripper_state(
            self.config, gripper_state_right_msg[0], gripper_state_right_msg[1]
        )

        # Combine joint data from both robots
        import numpy as np

        joint_data = {}
        for key in joint_data_left.keys():
            joint_data[key] = np.concatenate(
                [joint_data_left[key], joint_data_right[key]]
            )

        state_frame["joint_data"] = joint_data
        state_frame["gripper_state"] = np.concatenate(
            [gripper_state_left, gripper_state_right]
        )

        # TCP poses are always parsed (required for current setup)
        msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.ACTUAL_TCP_LEFT)
        raw_pose = rmp.parse_pose(self.config, msg_data, schema_name)
        state_frame["actual_tcp_left"] = positive_quat(raw_pose)

        msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.ACTUAL_TCP_RIGHT)
        raw_pose = rmp.parse_pose(self.config, msg_data, schema_name)
        state_frame["actual_tcp_right"] = positive_quat(raw_pose)
        
        return state_frame

    def _parse_desired(self, frame_buffer) -> dict:
        """Parse desired commands from available ROS topics."""
        desired_frame = {}
        
        # Parse gripper commands (always available)
        msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DES_GRIPPER_LEFT)
        desired_frame["des_gripper_left"] = rmp.parse_array(
            self.config, msg_data, schema_name
        )

        msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DES_GRIPPER_RIGHT)
        desired_frame["des_gripper_right"] = rmp.parse_array(
            self.config, msg_data, schema_name
        )

        # Only TCP-based control modes are supported
        if self.config.action_level in [
            ActionMode.TCP,
            ActionMode.TELEOP, 
            ActionMode.DELTA_TCP,
        ]:
            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DES_TCP_LEFT)
            raw_pose = rmp.parse_desired_tcp(self.config, msg_data, schema_name)
            desired_frame["des_tcp_left"] = raw_pose

            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.DES_TCP_RIGHT)
            raw_pose = rmp.parse_desired_tcp(self.config, msg_data, schema_name)
            desired_frame["des_tcp_right"] = raw_pose
        else:
            raise NotImplementedError(
                f"Action level {self.config.action_level} not supported. "
                "Only TCP, DELTA_TCP, and TELEOP modes are available."
            )

        return desired_frame

    def _parse_images(self, frame_buffer) -> dict:
        """Parse images from available camera topics."""
        images = {}
        
        # Static camera (always included)
        # msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.RGB_STATIC_IMAGE)
        # images["observation.images.rgb_static"] = rmp.parse_image(
        #     self.config, msg_data, schema_name
        # )

        # Optional wrist RGB cameras
        if self.config.include_rgb_images:
            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.RGB_LEFT_IMAGE)
            images["observation.images.rgb_left"] = rmp.parse_image(
                self.config, msg_data, schema_name
            )

            msg_data, schema_name = frame_buffer.get_msg(RosTopicEnum.RGB_RIGHT_IMAGE)
            images["observation.images.rgb_right"] = rmp.parse_image(
                self.config, msg_data, schema_name
            )

        return images
