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

from collections import deque

import av
import numpy as np

from ..config.pipeline_config import PipelineConfig
from ..config.rosbag_topics import RosTopicEnum
from ..utils import message_parsers as rmp
from .message_buffer import MessageBuffer


class VideoDecodeBuffer:
    """
    Decodes independent video streams and synchronizes them by order.

    This class buffers incoming messages and internally buffers decoded frames
    from multiple, independent video streams. It yields a synchronized snapshot
    (original message buffer + images) only when a frame is available from
    all active streams, ensuring data alignment.
    """

    def __init__(self, cfg: PipelineConfig):
        self.config = cfg
        self._active_topics_and_keys = self._get_active_topics()
        self.av_decoders = {}
        self.buffer_queue = deque()
        self.decoded_frames = {
            topic: deque() for topic, key in self._active_topics_and_keys
        }

    def reset(self):
        """Resets the decoders and clears all internal buffers."""
        self.av_decoders = {}
        self.buffer_queue.clear()
        for queue in self.decoded_frames.values():
            queue.clear()

    def add_and_decode(self, msg_buffer: MessageBuffer):
        """
        Adds a message buffer, decodes frames, and yields synchronized snapshots.

        Args:
            msg_buffer: The MessageBuffer instance representing a snapshot in time.

        Yields:
            A tuple containing the original MessageBuffer and a dictionary of
            decoded images for that snapshot, once all streams are synchronized.
        """
        self.buffer_queue.append(msg_buffer)
        self._parse_and_buffer_images(msg_buffer)

        # Yield as many complete snapshots as possible
        while self._is_snapshot_ready():
            buf = self.buffer_queue.popleft()
            image_dict = {
                key: self.decoded_frames[topic].popleft()
                for topic, key in self._active_topics_and_keys
            }
            yield buf, image_dict

    def _get_active_topics(self):
        """Determines which video topics to process based on config."""
        topics = [RosTopicEnum.RGB_STATIC_IMAGE]
        keys = ["observation.images.rgb_static"]

        if self.config.include_rgb_images:
            topics.append(RosTopicEnum.RGB_LEFT_IMAGE)
            topics.append(RosTopicEnum.RGB_RIGHT_IMAGE)
            keys.extend(["observation.images.rgb_left", "observation.images.rgb_right"])
        return list(zip(topics, keys))

    def _is_snapshot_ready(self) -> bool:
        """Checks if there are enough buffers and decoded frames to form a complete snapshot."""
        if not self.buffer_queue:
            return False
        # Check if every active topic has at least one frame in its queue
        return all(
            self.decoded_frames[topic] for topic, k in self._active_topics_and_keys
        )

    def _parse_and_buffer_images(self, msg_buffer: MessageBuffer):
        """
        Attempts to decode images and adds them to their respective topic queues.
        """
        for topic, key in self._active_topics_and_keys:
            msg_data, schema_name = msg_buffer.get_msg(topic)

            image = rmp.parse_image(
                self.config,
                msg_data,
                schema_name,
                decoder=self._get_decoder(topic),
            )
            if image is not None:
                self.decoded_frames[topic].append(image)

    def _get_decoder(self, topic: RosTopicEnum) -> av.CodecContext:
        """Retrieves or creates a video decoder for a given topic."""
        if topic not in self.av_decoders:
            self.av_decoders[topic] = av.CodecContext.create("h264", "r")
        return self.av_decoders[topic]
