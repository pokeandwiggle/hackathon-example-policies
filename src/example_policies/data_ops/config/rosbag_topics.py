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

from enum import Enum
from typing import Any


class RosSchemaEnum(Enum):
    JOINT = "sensor_msgs/msg/JointState"
    IMAGE = "sensor_msgs/msg/CompressedImage"
    TRANSFORM = "geometry_msgs/msg/Transform"
    POSE = "geometry_msgs/msg/PoseStamped"
    ARRAY = "std_msgs/msg/Float64MultiArray"

    FLOAT32 = "std_msgs/msg/Float32"

    JOINT_WAYPOINT = "trajectory_msgs/msg/JointTrajectory"

    # Currently unused topics
    STRING = "std_msgs/msg/String"
    CAMERA_INFO = "sensor_msgs/msg/CameraInfo"
    REALSENSE_EXTRINSICS = "realsense2_camera_msgs/msg/Extrinsics"

    POSE_TWIST = "teleop_controller_msgs/msg/PoseTwist"

    TWIST_STAMPED = "geometry_msgs/msg/TwistStamped"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        full_path = f"{cls.__module__}.{cls.__qualname__}"
        raise ValueError(f"{value} is not a known {full_path} Enum")


class RosTopicEnum(Enum):
    ANNOTATION = "/annotation"

    # Left camera topics
    RGB_LEFT_IMAGE = "/wrist_camera_left/D405/color/image_rect_raw/compressed"
    RGB_RIGHT_IMAGE = "/wrist_camera_right/D405/color/image_rect_raw/compressed"

    # Static camera topics
    RGB_STATIC_IMAGE = "/zed/zed_node/left/image_rect_color/compressed"

    # Desired Gripper Action
    DES_GRIPPER_LEFT = "/robotiq_gripper/left/f_30hz/robotiq_2f_gripper/confidence_command"
    DES_GRIPPER_RIGHT = (
        "/robotiq_gripper/right/f_30hz/robotiq_2f_gripper/confidence_command"
    )

    # Desired TCP Action
    DES_TCP_LEFT = "/franka_robot/left/f_30hz/teleop/twist_stamped"
    DES_TCP_RIGHT = "/franka_robot/right/f_30hz/teleop/twist_stamped"

    # Actual States
    ACTUAL_JOINT_LEFT = "/left/franka_robot_state_broadcaster/measured_joint_states"
    ACTUAL_TCP_LEFT = "/left/franka_robot_state_broadcaster/current_pose"
    ACTUAL_JOINT_RIGHT = "/right/franka_robot_state_broadcaster/measured_joint_states"
    ACTUAL_TCP_RIGHT = "/right/franka_robot_state_broadcaster/current_pose"

    LEFT_GRIPPER_DIST = (
        "/robotiq_gripper/left/f_30hz/robotiq_2f_gripper/finger_distance_mm"
    )
    RIGHT_GRIPPER_DIST = (
        "/robotiq_gripper/right/f_30hz/robotiq_2f_gripper/finger_distance_mm"
    )

    @classmethod
    def _missing_(cls, value: object) -> Any:
        full_path = f"{cls.__module__}.{cls.__qualname__}"
        raise ValueError(f"{value} is not a known {full_path} Enum")
