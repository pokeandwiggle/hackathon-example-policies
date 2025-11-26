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

from typing import List, Optional

import numpy as np
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

# Create a typestore for ROS 2 Humble
_typestore = get_typestore(Stores.ROS2_HUMBLE)


def register_types(typemap):
    """
    Compatibility wrapper for older rosbags API.

    In old versions this lived in rosbags.typesys and used a global
    typestore. We mimic that by registering everything on our
    ROS2_HUMBLE typestore.
    """
    _typestore.register(typemap)


def deserialize_cdr(rawdata, typename, typestore=None):
    """
    Compatibility wrapper for older rosbags.serde.deserialize_cdr.

    Newer rosbags exposes deserialization as a method on the typestore.
    """
    ts = typestore or _typestore
    return ts.deserialize_cdr(rawdata, typename)


from ..config.pipeline_config import GripperType, PipelineConfig
from ..config.rosbag_topics import RosSchemaEnum
from ..utils import geometric
from .image_processor import process_image_bytes

# --- Constants for Joint Parsing (defined once for performance) ---
_LEFT_ARM = [f"panda_left_joint{i}" for i in range(1, 8)]
_RIGHT_ARM = [f"panda_right_joint{i}" for i in range(1, 8)]
_LEFT_PANDA_GRIPPER = [f"panda_left_finger_joint{i}" for i in range(1, 3)]
_RIGHT_PANDA_GRIPPER = [f"panda_right_finger_joint{i}" for i in range(1, 3)]

_ROBOTIQ_JOINTS = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]

_LEFT_ROBOTIQ_GRIPPER = ["panda_left_" + joint for joint in _ROBOTIQ_JOINTS]

_RIGHT_ROBOTIQ_GRIPPER = ["panda_right_" + joint for joint in _ROBOTIQ_JOINTS]

CANONICAL_ARM_JOINTS = _LEFT_ARM + _RIGHT_ARM
ARM_JOINT_COUNT = len(_LEFT_ARM) + len(_RIGHT_ARM)  # Should be 14

POSE_TWIST_MSG_DEF = """
std_msgs/Header header
geometry_msgs/Pose pose
geometry_msgs/Twist twist
"""

# Register custom type with rosbags
types = get_types_from_msg(POSE_TWIST_MSG_DEF, "teleop_controller_msgs/msg/PoseTwist")
register_types(types)


def create_joint_order(cfg: PipelineConfig):
    joint_order = CANONICAL_ARM_JOINTS.copy()

    if cfg.left_gripper == GripperType.PANDA:
        joint_order += _LEFT_PANDA_GRIPPER
    elif cfg.left_gripper == GripperType.ROBOTIQ:
        joint_order += _LEFT_ROBOTIQ_GRIPPER
    else:
        raise ValueError(f"Unsupported left gripper type: {cfg.left_gripper}")

    if cfg.right_gripper == GripperType.PANDA:
        joint_order += _RIGHT_PANDA_GRIPPER
    elif cfg.right_gripper == GripperType.ROBOTIQ:
        joint_order += _RIGHT_ROBOTIQ_GRIPPER
    else:
        raise ValueError(f"Unsupported right gripper type: {cfg.right_gripper}")
    return joint_order


def _joint_reorder_indices(
    cfg: PipelineConfig, names: list[str], joint_order: Optional[List[str]] = None
) -> List[int]:
    if not joint_order:
        joint_order = create_joint_order(cfg)

    # Create a mapping from the message's joint order to our canonical order.
    name_to_idx = {name: i for i, name in enumerate(names)}
    reorder_indices = [name_to_idx[name] for name in joint_order]
    return reorder_indices


def parse_joints(cfg: PipelineConfig, msg_data, schema_name: RosSchemaEnum):
    """
    Parses a JointState message, efficiently reordering the joints into a
    canonical format using pre-computed constants and NumPy indexing.
    """
    assert schema_name == RosSchemaEnum.JOINT, f"Unexpected joint schema: {schema_name}"
    joint_msg = deserialize_cdr(msg_data, schema_name.value)
    reorder_indices = _joint_reorder_indices(cfg, joint_msg.name)

    positions = np.array(joint_msg.position, dtype=np.float32)[reorder_indices]
    velocities = np.array(joint_msg.velocity, dtype=np.float32)[reorder_indices]
    efforts = np.array(joint_msg.effort, dtype=np.float32)[reorder_indices]

    gripper_state = positions[ARM_JOINT_COUNT:]
    joint_velocity_full = velocities  # Full Velocity Vector for Pause Detection

    joint_data = {}
    if cfg.include_joint_states:
        if cfg.include_joint_positions:
            joint_data["position"] = positions[:ARM_JOINT_COUNT]
        if cfg.include_joint_velocities:
            joint_data["velocity"] = velocities[:ARM_JOINT_COUNT]
        if cfg.include_joint_efforts:
            joint_data["effort"] = efforts[:ARM_JOINT_COUNT]

    return joint_data, joint_velocity_full, gripper_state


def parse_image(
    cfg: PipelineConfig, msg_data, schema_name: RosSchemaEnum
) -> np.ndarray:
    assert (
        schema_name == RosSchemaEnum.IMAGE
    ), f"Unexpected RGB image schema: {schema_name}"
    img_msg = deserialize_cdr(msg_data, schema_name.value)
    img_bytes = img_msg.data
    is_depth = "compressedDepth" in img_msg.format

    img = process_image_bytes(
        img_bytes,
        cfg.image_resolution[0],
        cfg.image_resolution[1],
        is_depth,
    )

    return img


def parse_desired_tcp(
    cfg: PipelineConfig, msg_data, schema_name: RosSchemaEnum
) -> np.ndarray:
    assert schema_name in [
        RosSchemaEnum.POSE_TWIST,
        RosSchemaEnum.ARRAY,
    ], f"Unexpected pose schema: {schema_name}"
    if schema_name == RosSchemaEnum.ARRAY:
        return parse_array(cfg, msg_data, schema_name)[:7]

    pose_msg = deserialize_cdr(msg_data, schema_name.value)
    pose = pose_msg.pose
    return _create_pose_array(pose.position, pose.orientation)


def parse_array(
    cfg: PipelineConfig, msg_data, schema_name: RosSchemaEnum
) -> np.ndarray:
    assert schema_name == RosSchemaEnum.ARRAY, f"Unexpected array schema: {schema_name}"
    array_msg = deserialize_cdr(msg_data, schema_name.value)
    return np.array(array_msg.data, dtype=np.float64)


def parse_joint_waypoint(
    cfg: PipelineConfig, msg_data, schema_name: RosSchemaEnum, side: str
) -> np.ndarray:
    assert (
        schema_name == RosSchemaEnum.JOINT_WAYPOINT
    ), f"Unexpected joint waypoint schema: {schema_name}"
    traj_msg = deserialize_cdr(msg_data, schema_name.value)

    assert len(traj_msg.points) == 1, "Expected exactly one trajectory point."

    side_order = {"left": _LEFT_ARM, "right": _RIGHT_ARM}

    reorder_indices = _joint_reorder_indices(
        cfg, traj_msg.joint_names, side_order[side]
    )
    positions = np.array(traj_msg.points[0].positions, dtype=np.float32)[
        reorder_indices
    ]
    return positions


def parse_pose(cfg: PipelineConfig, msg_data, schema_name: RosSchemaEnum) -> np.ndarray:
    assert schema_name in [
        RosSchemaEnum.POSE,
        RosSchemaEnum.TRANSFORM,
    ], f"Unexpected pose schema: {schema_name}"
    pose_msg = deserialize_cdr(msg_data, schema_name.value)

    if schema_name == RosSchemaEnum.POSE:
        coord_pos = pose_msg.position
        coord_ori = pose_msg.orientation
    elif schema_name == RosSchemaEnum.TRANSFORM:
        coord_pos = pose_msg.translation
        coord_ori = pose_msg.rotation
    else:
        raise ValueError(f"Unsupported schema for pose: {schema_name}")

    return _create_pose_array(coord_pos, coord_ori)


def _create_pose_array(coord_pos, coord_ori) -> np.ndarray:
    pose_array = np.array(
        [
            coord_pos.x,
            coord_pos.y,
            coord_pos.z,
            coord_ori.x,
            coord_ori.y,
            coord_ori.z,
            coord_ori.w,
        ],
        dtype=np.float32,
    )
    pose_array = geometric.positive_quat(pose_array)
    return pose_array
