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
from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_types_from_msg, register_types

from ...utils.state_order import (
    ARM_JOINT_COUNT,
    LEFT_ARM,
    RIGHT_ARM,
    _joint_reorder_indices,
)
from ..config.pipeline_config import PipelineConfig
from ..config.rosbag_topics import RosSchemaEnum
from ..utils import geometric
from ..utils.image_processor import process_image_bytes

POSE_TWIST_MSG_DEF = """
std_msgs/Header header
geometry_msgs/Pose pose
geometry_msgs/Twist twist
"""

# Register custom type with rosbags
types = get_types_from_msg(POSE_TWIST_MSG_DEF, "teleop_controller_msgs/msg/PoseTwist")
register_types(types)


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
    assert schema_name == RosSchemaEnum.IMAGE, (
        f"Unexpected RGB image schema: {schema_name}"
    )
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
    assert schema_name == RosSchemaEnum.JOINT_WAYPOINT, (
        f"Unexpected joint waypoint schema: {schema_name}"
    )
    traj_msg = deserialize_cdr(msg_data, schema_name.value)

    assert len(traj_msg.points) == 1, "Expected exactly one trajectory point."

    side_order = {"left": LEFT_ARM, "right": RIGHT_ARM}

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
