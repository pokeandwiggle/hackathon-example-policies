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

"""Frame synchronization for multi-topic message alignment.

This module provides sensor-timestamp-based synchronization using synthetic
timestamps at a fixed frequency. Messages from all topics are matched to each
synthetic timestamp using nearest-neighbor search.
"""

import bisect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

from mcap.reader import make_reader

from ..config.pipeline_config import PipelineConfig
from ..config.rosbag_topics import RosSchemaEnum, RosTopicEnum
from .timestamp_utils import extract_sensor_timestamp


# Legacy topic name mappings for backwards compatibility with older MCAP files
# Maps old topic names -> current RosTopicEnum
LEGACY_TOPIC_MAPPING: dict[str, RosTopicEnum] = {
    # Old desired TCP pose topics (kept for reference, but not used for sync)
    "/desired_pose_left": RosTopicEnum.DES_TCP_LEFT,
    "/desired_pose_right": RosTopicEnum.DES_TCP_RIGHT,
    # Old teleop pose topics
    "/teleop_desired_pose_left": RosTopicEnum.DES_TELEOP_LEFT,
    "/teleop_desired_pose_right": RosTopicEnum.DES_TELEOP_RIGHT,
    # Old joint waypoint topics
    "/left_desired_joint_waypoint": RosTopicEnum.DES_JOINT_LEFT,
    "/right_desired_joint_waypoint": RosTopicEnum.DES_JOINT_RIGHT,
}


@dataclass
class TimestampedMessage:
    """A message with its sensor timestamp."""

    timestamp: float  # seconds
    data: bytes | Any  # bytes for raw messages, np.ndarray for pre-decoded video
    schema_name: RosSchemaEnum


class FrameSynchronizer:
    """Synchronizes messages using synthetic timestamps at a fixed frequency.

    This class implements a two-pass approach:
    1. Ingest: Read all messages from an episode and store with sensor timestamps
    2. Generate: For each synthetic timestamp, find nearest message from each topic

    Messages from slower topics may be reused for multiple frames.
    If any topic has no message within the tolerance for any frame, the entire
    episode is skipped with a warning.

    Args:
        config: Pipeline configuration (uses config.target_fps for output frequency)
        tolerance_ms: Maximum allowed time difference in milliseconds.
            If None, defaults to 50% of frame interval (1000/target_fps * 0.5)
    """

    def __init__(
        self,
        config: PipelineConfig,
        tolerance_ms: float | None = None,
        causal: bool = True,
    ):
        self.config = config
        self.target_frequency = config.target_fps
        self.causal = causal

        # Default tolerance to 100% of frame interval (100ms at 10Hz)
        if tolerance_ms is None:
            tolerance_ms = 1000.0 / self.target_frequency
        self.tolerance = tolerance_ms / 1000.0  # Convert to seconds

        # Build list of required topics
        self.required_topics = self._build_required_topics()

        # Storage: topic -> sorted list of TimestampedMessage
        self.messages: dict[RosTopicEnum, list[TimestampedMessage]] = {}

        # For binary search: topic -> sorted list of timestamps
        self.timestamps: dict[RosTopicEnum, list[float]] = {}

        self._reset_storage()

    def _reset_storage(self) -> None:
        """Reset message storage for a new episode."""
        self.messages = {topic: [] for topic in self.required_topics}
        self.timestamps = {topic: [] for topic in self.required_topics}
        self._invalid_episode = False
        # Absolute timestamp where synced frames begin (set during generate_synced_frames)
        self._sync_episode_start: float | None = None
        # Raw first-timestamps before video decoding drops pre-keyframe frames
        self._raw_first_timestamps: dict[RosTopicEnum, float | None] = {
            topic: None for topic in self.required_topics
        }

    def _build_required_topics(self) -> list[RosTopicEnum]:
        """Build list of required topics based on config."""
        # always required
        topics = [
            RosTopicEnum.ACTUAL_JOINT_STATE,
            RosTopicEnum.DES_GRIPPER_RIGHT,
            RosTopicEnum.DES_GRIPPER_LEFT,
        ]

        # Observations
        # Optional RGB images
        if self.config.include_rgb_images:
            topics.extend([RosTopicEnum.RGB_RIGHT_IMAGE, RosTopicEnum.RGB_LEFT_IMAGE, RosTopicEnum.RGB_STATIC_IMAGE])
        # Optional depth images
        if self.config.include_depth_images:
            topics.extend(
                [RosTopicEnum.DEPTH_RIGHT_IMAGE, RosTopicEnum.DEPTH_LEFT_IMAGE]
            )

        # States
        if self.config.requires_tcp_poses():
            topics.extend([RosTopicEnum.ACTUAL_TCP_RIGHT, RosTopicEnum.ACTUAL_TCP_LEFT])

        # Actions
        if self.config.is_tcp_action():
            topics.extend([RosTopicEnum.DES_TCP_RIGHT, RosTopicEnum.DES_TCP_LEFT])
        elif self.config.is_joint_action():
            topics.extend([RosTopicEnum.DES_JOINT_RIGHT, RosTopicEnum.DES_JOINT_LEFT])

        return topics

    def get_topic_names(self) -> list[str]:
        """Get the string names of all required topics, including legacy names."""
        names = [topic.value for topic in self.required_topics]
        # Also include legacy topic names for backwards compatibility
        for legacy_name, topic_enum in LEGACY_TOPIC_MAPPING.items():
            if topic_enum in self.required_topics and legacy_name not in names:
                names.append(legacy_name)
        return names

    def _resolve_topic(self, channel_topic: str) -> RosTopicEnum | None:
        """Resolve a channel topic name to RosTopicEnum, handling legacy names."""
        # Try direct match first
        try:
            return RosTopicEnum(channel_topic)
        except ValueError:
            pass
        # Check legacy mapping
        if channel_topic in LEGACY_TOPIC_MAPPING:
            return LEGACY_TOPIC_MAPPING[channel_topic]
        return None

    def ingest_episode(self, episode_path: Path) -> None:
        """First pass: read all messages and store with sensor timestamps.

        Args:
            episode_path: Path to the MCAP episode file
        """
        # Clear previous episode data
        self._reset_storage()

        topic_names = self.get_topic_names()

        with open(episode_path, "rb") as f:
            reader = make_reader(f)

            topics_using_logtime: set[str] = set()

            for schema, channel, message in reader.iter_messages(topics=topic_names):
                topic = self._resolve_topic(channel.topic)
                if topic is None:
                    continue  # Skip unknown topics
                schema_name = RosSchemaEnum(schema.name)

                # Extract sensor timestamp from message header
                sensor_ts = extract_sensor_timestamp(message.data, schema_name)

                # Fall back to log_time if no sensor timestamp
                if sensor_ts is None:
                    sensor_ts = message.log_time * 1e-9
                    topics_using_logtime.add(channel.topic)

                # Store message
                ts_msg = TimestampedMessage(
                    timestamp=sensor_ts,
                    data=message.data,
                    schema_name=schema_name,
                )
                self.messages[topic].append(ts_msg)

            # Report topics that used log_time fallback
            if topics_using_logtime:
                logger.debug(
                    "Using log_time for topics without sensor timestamps: %s",
                    sorted(topics_using_logtime),
                )

        # Sort by timestamp (might not be strictly ordered in MCAP file)
        for topic in self.required_topics:
            if self.messages[topic]:
                self.messages[topic].sort(key=lambda msg: msg.timestamp)
                self.timestamps[topic] = [msg.timestamp for msg in self.messages[topic]]
                self._raw_first_timestamps[topic] = self.timestamps[topic][0]

    def decode_video_topics(self, width: int, height: int) -> None:
        """Pre-decode all AV1 video frames sequentially for each video topic.

        AV1 is stateful: P-frames reference previous frames, so the codec must
        see every frame in order.  The synchronizer subsamples 30 fps streams to
        10 Hz *after* ingestion, so if we only decoded the subsampled frames the
        codec would miss keyframes and most P-frames would fail.

        This method decodes ALL ingested messages for each COMPRESSED_VIDEO topic,
        replaces ``TimestampedMessage.data`` with the decoded + resized numpy
        array, and removes messages that failed to decode (mid-GOP P-frames
        before the first keyframe).
        """
        import av
        from ..utils.image_processor import decode_av1_frame, resize_and_normalize
        import cv2

        video_topics = [
            topic for topic in self.required_topics
            if self.messages[topic]
            and self.messages[topic][0].schema_name == RosSchemaEnum.COMPRESSED_VIDEO
        ]

        if not video_topics:
            return

        from rosbags.serde import deserialize_cdr

        for topic in video_topics:
            codec = av.codec.CodecContext.create("libdav1d", "r")
            decoded_messages: list[TimestampedMessage] = []
            total = len(self.messages[topic])
            failed = 0

            for msg in self.messages[topic]:
                video_msg = deserialize_cdr(msg.data, msg.schema_name.value)
                bgr = decode_av1_frame(bytes(video_msg.data), codec)
                if bgr is None:
                    failed += 1
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                arr = resize_and_normalize(rgb, width, height, is_depth=False)
                decoded_messages.append(TimestampedMessage(
                    timestamp=msg.timestamp,
                    data=arr,
                    schema_name=msg.schema_name,
                ))

            self.messages[topic] = decoded_messages
            self.timestamps[topic] = [m.timestamp for m in decoded_messages]
            logger.debug(
                "%s: decoded %d/%d AV1 frames (%d skipped before first keyframe)",
                topic.value, total - failed, total, failed,
            )

    def validate_episode(self) -> tuple[bool, dict]:
        """Validate episode data and compute synchronization statistics.

        Checks:
        - If episode was marked invalid during ingestion
        - If any required topics have no messages
        - If all sensors are using the same clock

        Returns:
            Tuple of (is_valid, stats_dict) where stats_dict contains:
            - target_frequency, tolerance_ms, message_counts per topic
            - frequency stats per topic
        """
        stats = {
            "target_frequency": self.target_frequency,
            "tolerance_ms": self.tolerance * 1000,
            "message_counts": {},
            "frequency_stats": {},
            "valid": False,
            "skip_reason": None,
        }

        if self._invalid_episode:
            stats["skip_reason"] = "invalid during ingestion"
            return False, stats

        missing_topics = []
        first_timestamps = []

        for topic in self.required_topics:
            timestamps = self.timestamps[topic]
            count = len(timestamps)
            stats["message_counts"][topic.value] = count

            if timestamps:
                first_timestamps.append(timestamps[0])

            if count == 0:
                missing_topics.append(topic)
                logger.debug("%s: NO MESSAGES", topic.value)
                continue

            if count == 1:
                logger.debug("%s: 1 message (cannot compute frequency)", topic.value)
                continue

            # Compute frequency statistics
            intervals = [
                timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
            ]
            avg_interval = sum(intervals) / len(intervals)
            avg_freq = 1.0 / avg_interval if avg_interval > 0 else 0.0

            min_interval = min(intervals)
            max_interval = max(intervals)

            # Deviation from average interval (as percentage)
            min_dev = ((min_interval - avg_interval) / avg_interval * 100) if avg_interval > 0 else 0.0
            max_dev = ((max_interval - avg_interval) / avg_interval * 100) if avg_interval > 0 else 0.0

            # Also show actual frequency range
            max_freq = 1.0 / min_interval if min_interval > 0 else float('inf')
            min_freq = 1.0 / max_interval if max_interval > 0 else 0.0

            stats["frequency_stats"][topic.value] = {
                "avg_hz": avg_freq,
                "min_hz": min_freq,
                "max_hz": max_freq,
                "interval_dev_min_pct": min_dev,
                "interval_dev_max_pct": max_dev,
            }

            logger.debug(
                "%s: %d msgs, avg %.1fHz (range: %.1f-%.1fHz), "
                "interval deviation: %+.1f%% to %+.1f%%",
                topic.value, count, avg_freq, min_freq, max_freq, min_dev, max_dev,
            )

        if missing_topics:
            stats["skip_reason"] = f"missing messages for topics: {[t.value for t in missing_topics]}"
            logger.warning("Skipping episode - %s", stats["skip_reason"])
            return False, stats

        # Check if all sensors are using the same clock by comparing first timestamps.
        # Use raw (pre-decode) first timestamps so that AV1 keyframe skipping
        # doesn't inflate the spread — video frames before the first keyframe
        # are dropped during decode_video_topics(), which can shift the first
        # decoded timestamp forward by ~1 s and falsely trigger this check.
        raw_firsts = [
            ts for ts in self._raw_first_timestamps.values() if ts is not None
        ]
        if len(raw_firsts) >= 2:
            clock_spread = max(raw_firsts) - min(raw_firsts)
            max_clock_spread = 1.0  # Maximum allowed spread in seconds
            stats["clock_spread_s"] = clock_spread
            if clock_spread > max_clock_spread:
                stats["skip_reason"] = f"sensors appear to use different clocks (spread: {clock_spread:.2f}s)"
                logger.warning("Skipping episode - %s", stats["skip_reason"])
                return False, stats

        # Compute usable time range stats
        if first_timestamps:
            last_timestamps = [ts[-1] for ts in self.timestamps.values() if ts]
            earliest_start = min(first_timestamps)
            latest_start = max(first_timestamps)
            earliest_end = min(last_timestamps)
            latest_end = max(last_timestamps)
            stats["start_offset_s"] = latest_start - earliest_start  # Data lost at beginning
            stats["end_offset_s"] = latest_end - earliest_end  # Data lost at end
            stats["usable_duration_s"] = earliest_end - latest_start

        stats["valid"] = True
        return True, stats

    def find_nearest(
        self, topic: RosTopicEnum, target_time: float
    ) -> TimestampedMessage | None:
        """Find the message nearest to target_time using binary search.

        Always returns the closest message, even if it's been used before.
        Returns None only if:
        - No messages exist for this topic, OR
        - The closest message is beyond the tolerance threshold

        Args:
            topic: The topic to search
            target_time: The target timestamp in seconds

        Returns:
            The nearest TimestampedMessage, or None if beyond tolerance
        """
        msg, _ = self.find_nearest_with_gap(topic, target_time)
        return msg

    def find_nearest_with_gap(
        self, topic: RosTopicEnum, target_time: float
    ) -> tuple[TimestampedMessage | None, float | None]:
        """Find the message nearest to target_time and return the time gap.

        Args:
            topic: The topic to search
            target_time: The target timestamp in seconds

        Returns:
            Tuple of (nearest TimestampedMessage or None, gap in seconds or None)
        """
        timestamps = self.timestamps[topic]
        messages = self.messages[topic]

        if not timestamps:
            return None, None

        # Binary search for insertion point
        idx = bisect.bisect_left(timestamps, target_time)

        if self.causal:
            # Causal mode: only consider messages at or before target_time
            # bisect_left gives us the first index >= target_time
            # So idx is an exact match, idx-1 is the last message before target_time
            candidates = []
            if idx < len(timestamps) and timestamps[idx] == target_time:
                candidates.append((idx, 0.0))  # exact match
            if idx > 0:
                candidates.append((idx - 1, target_time - timestamps[idx - 1]))
        else:
            # Default: check both past and future neighbors to find closest
            candidates = []
            if idx > 0:
                candidates.append((idx - 1, abs(timestamps[idx - 1] - target_time)))
            if idx < len(timestamps):
                candidates.append((idx, abs(timestamps[idx] - target_time)))

        if not candidates:
            return None, None

        # Pick the closest one
        best_idx, best_diff = min(candidates, key=lambda x: x[1])

        # Only reject if beyond tolerance
        if best_diff > self.tolerance:
            return None, best_diff

        return messages[best_idx], best_diff

    def generate_synced_frames(self) -> Iterator[dict[RosTopicEnum, TimestampedMessage]]:
        """Generate synchronized frames at fixed intervals using synthetic timestamps.

        Creates frames at uniform time intervals, finding the nearest message from
        each topic for each synthetic timestamp. This produces consistent timing
        that matches how policies run at inference time.

        Yields:
            Dict mapping RosTopicEnum to TimestampedMessage for each synced frame
        """
        # Validate episode
        is_valid, stats = self.validate_episode()
        logger.debug("Stats: %s", stats)
        if not is_valid:
            return

        # Determine time range where ALL topics have data
        start_times = [ts[0] for ts in self.timestamps.values() if ts]
        end_times = [ts[-1] for ts in self.timestamps.values() if ts]

        if not start_times or not end_times:
            logger.warning("No messages found for any topic")
            return

        # Start: when all topics have started (max of firsts)
        # End: when any topic ends (min of lasts)
        episode_start = max(start_times)
        episode_end = min(end_times)
        self._sync_episode_start = episode_start

        if episode_start >= episode_end:
            logger.warning("No overlapping time range for all topics")
            return

        interval = 1.0 / self.target_frequency

        # Generate synthetic timestamps at fixed intervals
        target_timestamps: list[float] = []
        t = episode_start
        while t <= episode_end:
            target_timestamps.append(t)
            t += interval

        # Build frames at each synthetic timestamp
        all_frames: list[dict[RosTopicEnum, TimestampedMessage]] = []
        for target_time in target_timestamps:
            frame: dict[RosTopicEnum, TimestampedMessage] = {}

            for topic in self.required_topics:
                msg, gap = self.find_nearest_with_gap(topic, target_time)

                if msg is None:
                    gap_ms = gap * 1000 if gap is not None else float("inf")
                    exceeded_ms = gap_ms - (self.tolerance * 1000)
                    logger.warning(
                        "Skipping entire episode - topic '%s' has no message within "
                        "%.1fms tolerance of timestamp %.3fs "
                        "(nearest was %.1fms away, exceeded by %.1fms)",
                        topic.value, self.tolerance * 1000, target_time, gap_ms, exceeded_ms,
                    )
                    return

                frame[topic] = msg

            all_frames.append(frame)

        # Yield all validated frames
        yield from all_frames

        # Log summary
        duration = episode_end - episode_start
        logger.debug(
            "Sync stats: %d frames at %.1fHz over %.2fs",
            len(all_frames), self.target_frequency, duration,
        )

    def get_sync_start_offset(self) -> float | None:
        """Get the absolute timestamp where synced frames begin.

        This is the max of all topics' first timestamps — the point where all
        sensors have started providing data. Used to relate annotation timestamps
        (relative to MCAP episode metadata start) to synced frame indices.

        Returns:
            Absolute Unix timestamp (seconds) of the first synced frame,
            or None if generate_synced_frames hasn't been called or yielded
            no frames (e.g. validation failure, no overlap).
        """
        return self._sync_episode_start
