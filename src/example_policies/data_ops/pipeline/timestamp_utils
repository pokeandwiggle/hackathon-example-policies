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


def extract_sensor_timestamp(msg_data: bytes, schema_name: RosSchemaEnum) -> float | None:
    """Extract sensor timestamp from a ROS message header.

    Parses the message and looks for a 'header' or 'stamp' field containing
    the sensor timestamp.

    Args:
        msg_data: Raw message bytes
        schema_name: ROS schema enum for the message type

    Returns:
        Timestamp in seconds (float), or None if message has no header.

    Message types with headers:
        - sensor_msgs/JointState: has header
        - sensor_msgs/CompressedImage: has header
        - geometry_msgs/PoseStamped: has header
        - teleop_controller_msgs/PoseTwist: has header

    Message types WITHOUT headers (returns None):
        - geometry_msgs/Transform: no header
        - std_msgs/Float64MultiArray: no header
    """
    msg = deserialize_cdr(msg_data, schema_name.value)

    # Most ROS messages with timestamps have a 'header' field
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        stamp = msg.header.stamp
        return stamp.sec + stamp.nanosec * 1e-9

    # Some messages have a direct 'stamp' field
    if hasattr(msg, "stamp"):
        stamp = msg.stamp
        return stamp.sec + stamp.nanosec * 1e-9

    # For PoseStamped, the stamp is directly on the message
    if hasattr(msg, "pose") and hasattr(msg, "header"):
        stamp = msg.header.stamp
        return stamp.sec + stamp.nanosec * 1e-9

    # No timestamp available
    return None
