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

    JOINT_TARGET = "trajectory_msgs/msg/JointTrajectory"

    # Currently unused topics
    STRING = "std_msgs/msg/String"
    CAMERA_INFO = "sensor_msgs/msg/CameraInfo"
    REALSENSE_EXTRINSICS = "realsense2_camera_msgs/msg/Extrinsics"

    POSE_TWIST = "teleop_controller_msgs/msg/PoseTwist"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        full_path = f"{cls.__module__}.{cls.__qualname__}"
        raise ValueError(f"{value} is not a known {full_path} Enum")


class RosTopicEnum(Enum):
    ANNOTATION = "/annotation"

    # Left camera topics
    DEPTH_LEFT_INFO = "/cam_left/aligned_depth_to_color/camera_info"
    DEPTH_LEFT_IMAGE = "/cam_left/aligned_depth_to_color/image_compressed"
    DEPTH_LEFT_EXTRINSICS = "/cam_left/extrinsics/depth_to_color"
    RGB_LEFT_INFO = "/cam_left/color/camera_info"
    RGB_LEFT_IMAGE = "/cam_left/color/image_rect_compressed"

    # Right camera topics
    DEPTH_RIGHT_INFO = "/cam_right/aligned_depth_to_color/camera_info"
    DEPTH_RIGHT_IMAGE = "/cam_right/aligned_depth_to_color/image_compressed"
    DEPTH_RIGHT_EXTRINSICS = "/cam_right/extrinsics/depth_to_color"
    RGB_RIGHT_INFO = "/cam_right/color/camera_info"
    RGB_RIGHT_IMAGE = "/cam_right/color/image_rect_compressed"

    # Static camera topics
    RGB_STATIC_INFO = "/cam_static/color/camera_info"
    RGB_STATIC_IMAGE = "/cam_static/color/image_rect_compressed"

    # Desired Gripper Action
    DES_GRIPPER_LEFT = "/desired_gripper_values_left"
    DES_GRIPPER_RIGHT = "/desired_gripper_values_right"

    # Desired Teleop Pose
    DES_TELEOP_LEFT = "/cartesian_waypoint_left"
    DES_TELEOP_RIGHT = "/cartesian_waypoint_right"

    # Desired TCP Action
    DES_TCP_LEFT = "/cartesian_target_left"
    DES_TCP_RIGHT = "/cartesian_target_right"

    # Desired Joint Action
    DES_JOINT_LEFT = "/joint_target_left"
    DES_JOINT_RIGHT = "/joint_target_right"

    # Actual States
    ACTUAL_JOINT_STATE = "/joint_states"
    ACTUAL_TCP_LEFT = "/panda_left/tcp"
    ACTUAL_TCP_RIGHT = "/panda_right/tcp"

    # VR
    VR_CAMERA_POSE = "/vr_camera"
    VR_CAMERA_SCALE = "/vr_camera_scale"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        full_path = f"{cls.__module__}.{cls.__qualname__}"
        raise ValueError(f"{value} is not a known {full_path} Enum")
