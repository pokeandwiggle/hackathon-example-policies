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

"""Utilities for extracting sensor timestamps from ROS messages."""

from rosbags.serde import deserialize_cdr

from ..config.rosbag_topics import RosSchemaEnum


def extract_sensor_timestamp(
    msg_data: bytes, schema_name: RosSchemaEnum | str
) -> float | None:
    """Extract sensor timestamp from a ROS message header.

    Parses the message and looks for a 'header', 'stamp', or 'timestamp'
    field containing the sensor timestamp.

    Args:
        msg_data: Raw message bytes
        schema_name: ROS schema enum or string name for the message type

    Returns:
        Timestamp in seconds (float), or None if message has no header.

    Message types with headers:
        - sensor_msgs/JointState: has header
        - sensor_msgs/CompressedImage: has header
        - geometry_msgs/PoseStamped: has header
        - geometry_msgs/TransformStamped: has header
        - teleop_controller_msgs/PoseTwist: has header
        - teleop_controller_msgs/GripperValues: has header

    Message types with 'timestamp' field:
        - foxglove_msgs/CompressedVideo: has timestamp (not header)

    Message types WITHOUT headers (returns None):
        - geometry_msgs/Transform: no header (v1.0 tcp_pose recordings)
        - std_msgs/Float64MultiArray: no header (schema v1.0 gripper values)
    """
    name = schema_name.value if isinstance(schema_name, RosSchemaEnum) else schema_name
    msg = deserialize_cdr(msg_data, name)

    # Most ROS messages with timestamps have a 'header' field
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        stamp = msg.header.stamp
        return stamp.sec + stamp.nanosec * 1e-9

    # Some messages have a direct 'stamp' field (without header)
    if hasattr(msg, "stamp"):
        stamp = msg.stamp
        return stamp.sec + stamp.nanosec * 1e-9

    # foxglove_msgs/CompressedVideo uses 'timestamp' instead of 'stamp'
    if hasattr(msg, "timestamp") and hasattr(msg.timestamp, "sec"):
        stamp = msg.timestamp
        return stamp.sec + stamp.nanosec * 1e-9

    # No timestamp available
    return None
