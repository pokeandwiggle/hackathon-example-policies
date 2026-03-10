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

from ...utils.embodiment import EmbodimentJointConfig, get_joint_config
from ...utils.state_order import _joint_reorder_indices
from ..config.pipeline_config import PipelineConfig
from ..config.rosbag_topics import RosSchemaEnum
from ..utils import geometric
from ..utils.image_processor import process_image_bytes, process_video_bytes

_DEFAULT_EMBODIMENT = get_joint_config("dual_panda_wall")

POSE_TWIST_MSG_DEF = """
std_msgs/Header header
geometry_msgs/Pose pose
geometry_msgs/Twist twist
"""

COMPRESSED_VIDEO_MSG_DEF = """
builtin_interfaces/Time timestamp
string frame_id
uint8[] data
string format
"""

GRIPPER_VALUES_MSG_DEF = """
std_msgs/Header header
float64 width
float64 speed
float64 force
"""

# Register custom types with rosbags
types = get_types_from_msg(POSE_TWIST_MSG_DEF, "teleop_controller_msgs/msg/PoseTwist")
register_types(types)
types = get_types_from_msg(
    COMPRESSED_VIDEO_MSG_DEF, "foxglove_msgs/msg/CompressedVideo"
)
register_types(types)
types = get_types_from_msg(
    GRIPPER_VALUES_MSG_DEF, "teleop_controller_msgs/msg/GripperValues"
)
register_types(types)


def parse_joints(
    cfg: PipelineConfig,
    msg_data,
    schema_name: RosSchemaEnum,
    embodiment: EmbodimentJointConfig | None = None,
):
    """
    Parses a JointState message, reordering joints into canonical format.
    Joint order is derived dynamically from the embodiment config via
    _joint_reorder_indices.
    """
    assert schema_name == RosSchemaEnum.JOINT, f"Unexpected joint schema: {schema_name}"
    joint_msg = deserialize_cdr(msg_data, schema_name.value)
    if embodiment is None:
        embodiment = _DEFAULT_EMBODIMENT
    reorder_indices = _joint_reorder_indices(cfg, joint_msg.name, embodiment=embodiment)

    positions = np.array(joint_msg.position, dtype=np.float32)[reorder_indices]
    velocities = np.array(joint_msg.velocity, dtype=np.float32)[reorder_indices]
    efforts = np.array(joint_msg.effort, dtype=np.float32)[reorder_indices]

    arm_joint_count = embodiment.arm_joint_count
    gripper_state = positions[arm_joint_count:]
    joint_velocity_full = velocities  # Full Velocity Vector for Pause Detection

    joint_data = {}
    if cfg.include_joint_states:
        if cfg.include_joint_positions:
            joint_data["position"] = positions[:arm_joint_count]
        if cfg.include_joint_velocities:
            joint_data["velocity"] = velocities[:arm_joint_count]
        if cfg.include_joint_efforts:
            joint_data["effort"] = efforts[:arm_joint_count]

    return joint_data, joint_velocity_full, gripper_state


class ImageParser:
    """Stateful image parser that reuses a video decoder across frames."""

    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg
        self._codec = None

    def reset(self):
        self._codec = None

    def parse(self, msg_data, schema_name: RosSchemaEnum) -> np.ndarray | None:
        assert schema_name in (RosSchemaEnum.IMAGE, RosSchemaEnum.COMPRESSED_VIDEO), (
            f"Unexpected RGB image schema: {schema_name}"
        )

        # Pre-decoded numpy array from decode_video_topics() — return directly
        if isinstance(msg_data, np.ndarray):
            return msg_data

        if schema_name == RosSchemaEnum.COMPRESSED_VIDEO:
            video_msg = deserialize_cdr(msg_data, schema_name.value)
            if self._codec is None:
                import av

                self._codec = av.codec.CodecContext.create("libdav1d", "r")
            return process_video_bytes(
                bytes(video_msg.data),
                self._cfg.image_resolution[0],
                self._cfg.image_resolution[1],
                self._codec,
            )

        img_msg = deserialize_cdr(msg_data, schema_name.value)
        return process_image_bytes(
            img_msg.data,
            self._cfg.image_resolution[0],
            self._cfg.image_resolution[1],
            is_depth="compressedDepth" in img_msg.format,
        )


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


def parse_gripper(
    cfg: PipelineConfig, msg_data, schema_name: RosSchemaEnum
) -> np.ndarray:
    """Parse gripper values, supporting both legacy ARRAY and new GRIPPER_VALUES schemas."""
    if schema_name == RosSchemaEnum.ARRAY:
        return parse_array(cfg, msg_data, schema_name)

    assert schema_name == RosSchemaEnum.GRIPPER_VALUES, (
        f"Unexpected gripper schema: {schema_name}"
    )
    gripper_msg = deserialize_cdr(msg_data, schema_name.value)
    return np.array([gripper_msg.width], dtype=np.float64)


def parse_target(
    cfg: PipelineConfig,
    msg_data,
    schema_name: RosSchemaEnum,
    side: str,
    embodiment: EmbodimentJointConfig | None = None,
) -> np.ndarray:
    assert schema_name == RosSchemaEnum.JOINT_TARGET, (
        f"Unexpected joint target schema: {schema_name}"
    )
    traj_msg = deserialize_cdr(msg_data, schema_name.value)

    assert len(traj_msg.points) == 1, "Expected exactly one trajectory point."

    if embodiment is None:
        embodiment = _DEFAULT_EMBODIMENT
    side_order = {
        "left": embodiment.left_arm_joints(),
        "right": embodiment.right_arm_joints(),
    }

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
        RosSchemaEnum.TRANSFORM_STAMPED,
    ], f"Unexpected pose schema: {schema_name}"
    pose_msg = deserialize_cdr(msg_data, schema_name.value)

    if schema_name == RosSchemaEnum.POSE:
        coord_pos = pose_msg.position
        coord_ori = pose_msg.orientation
    elif schema_name == RosSchemaEnum.TRANSFORM_STAMPED:
        coord_pos = pose_msg.transform.translation
        coord_ori = pose_msg.transform.rotation
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
