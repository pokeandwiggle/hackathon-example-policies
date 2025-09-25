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

from typing import Any, List

from ..config.pipeline_config import ActionLevel, PipelineConfig
from ..config.rosbag_topics import RosSchemaEnum, RosTopicEnum


class FrameBuffer:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.listen_topics = _build_required_attributes(config)
        self.buffer = {}

        self.reset()

    def reset(self):
        """Reset all buffered states to None."""
        self.buffer = {}
        for topic in self.listen_topics:
            self.buffer[topic] = None

    def is_complete(self) -> bool:
        """Check if all required state information has been received."""
        return all(self.buffer[topic] is not None for topic in self.listen_topics)

    def add_msg(self, topic: RosTopicEnum, schema_name: str, msg_data: Any):
        """Add a message to the buffer."""

        topic = RosTopicEnum(topic)
        self.buffer[topic] = (msg_data, RosSchemaEnum(schema_name))

    def get_msg(self, topic: RosTopicEnum) -> Any:
        return self.buffer[topic]

    def get_topic_names(self):
        """Get the names of all topics being listened to."""
        return [topic.value for topic in self.listen_topics]


def _build_required_attributes(config: PipelineConfig) -> List[RosTopicEnum]:
    """Build list of required attributes based on configuration."""
    listen_topics = []
    listen_topics.append(RosTopicEnum.ACTUAL_JOINT_STATE)
    listen_topics.append(RosTopicEnum.RGB_STATIC_IMAGE)
    listen_topics.extend(
        [RosTopicEnum.DES_GRIPPER_LEFT, RosTopicEnum.DES_GRIPPER_RIGHT]
    )

    if config.action_level in [ActionLevel.TCP, ActionLevel.DELTA_TCP]:
        listen_topics.extend([RosTopicEnum.DES_TCP_LEFT, RosTopicEnum.DES_TCP_RIGHT])
    elif config.action_level in [ActionLevel.JOINT, ActionLevel.DELTA_JOINT]:
        listen_topics.extend(
            [RosTopicEnum.DES_JOINT_LEFT, RosTopicEnum.DES_JOINT_RIGHT]
        )
    elif config.action_level == ActionLevel.TELEOP:
        listen_topics.extend(
            [RosTopicEnum.DES_TELEOP_LEFT, RosTopicEnum.DES_TELEOP_RIGHT]
        )

    if config.include_tcp_poses:
        listen_topics.extend(
            [RosTopicEnum.ACTUAL_TCP_LEFT, RosTopicEnum.ACTUAL_TCP_RIGHT]
        )

    if config.include_rgb_images:
        listen_topics.extend(
            [RosTopicEnum.RGB_LEFT_IMAGE, RosTopicEnum.RGB_RIGHT_IMAGE]
        )

    if config.include_depth_images:
        listen_topics.extend(
            [RosTopicEnum.DEPTH_LEFT_IMAGE, RosTopicEnum.DEPTH_RIGHT_IMAGE]
        )

    return listen_topics
