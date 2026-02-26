"""Tests for causal (past-only) sync mode in FrameSynchronizer.

Verifies that when causal=True, find_nearest_with_gap only returns messages
with timestamp <= target_time, never future messages.
"""

import pytest
from unittest.mock import MagicMock

from example_policies.data_ops.pipeline.frame_synchronizer import (
    FrameSynchronizer,
    TimestampedMessage,
)
from example_policies.data_ops.config.rosbag_topics import RosSchemaEnum, RosTopicEnum


def make_msg(ts: float) -> TimestampedMessage:
    """Create a TimestampedMessage with the given timestamp."""
    return TimestampedMessage(timestamp=ts, data=b"", schema_name=RosSchemaEnum.JOINT_STATE)


TOPIC = RosTopicEnum.ACTUAL_JOINT_STATE


def _make_synchronizer(causal: bool, tolerance_s: float = 1.0) -> FrameSynchronizer:
    """Create a FrameSynchronizer with minimal config for unit testing."""
    config = MagicMock()
    config.target_fps = 10
    config.include_rgb_images = False
    config.include_depth_images = False
    config.requires_tcp_poses = MagicMock(return_value=False)
    config.is_tcp_action = MagicMock(return_value=False)
    config.is_joint_action = MagicMock(return_value=False)

    sync = FrameSynchronizer(config, tolerance_ms=tolerance_s * 1000, causal=causal)
    return sync


def _inject_messages(sync: FrameSynchronizer, topic: RosTopicEnum, timestamps: list[float]):
    """Inject pre-sorted messages into the synchronizer for a topic."""
    msgs = [make_msg(ts) for ts in timestamps]
    sync.messages[topic] = msgs
    sync.timestamps[topic] = timestamps


# ─── Core causal behavior ────────────────────────────────────────────────


class TestCausalFindNearest:
    """Test that causal mode only looks at past/present messages."""

    def test_causal_picks_past_not_future(self):
        """When target is between two messages, causal picks the earlier one."""
        sync = _make_synchronizer(causal=True)
        #                      past    future
        # timestamps:          1.0     3.0
        # target:                 2.0
        _inject_messages(sync, TOPIC, [1.0, 3.0])

        msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)

        assert msg is not None
        assert msg.timestamp == 1.0, f"Causal should pick past msg (1.0), got {msg.timestamp}"
        assert gap == pytest.approx(1.0)  # 2.0 - 1.0

    def test_non_causal_picks_future_when_closer(self):
        """Non-causal should pick the closer message, even if it's in the future."""
        sync = _make_synchronizer(causal=False)
        # timestamps:          1.0     2.5
        # target:                 2.0
        _inject_messages(sync, TOPIC, [1.0, 2.5])

        msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)

        assert msg is not None
        assert msg.timestamp == 2.5, f"Non-causal should pick closer future msg (2.5), got {msg.timestamp}"
        assert gap == pytest.approx(0.5)

    def test_causal_ignores_closer_future_message(self):
        """Even when future message is much closer, causal still picks the past one."""
        sync = _make_synchronizer(causal=True)
        # timestamps:          1.0           2.01
        # target:                    2.0
        # Future (2.01) is 0.01s away; past (1.0) is 1.0s away
        _inject_messages(sync, TOPIC, [1.0, 2.01])

        msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)

        assert msg is not None
        assert msg.timestamp == 1.0, f"Causal must pick past (1.0), not closer future (2.01), got {msg.timestamp}"
        assert gap == pytest.approx(1.0)


# ─── Exact matches ──────────────────────────────────────────────────────


class TestCausalExactMatch:
    """Test exact timestamp matches in causal mode."""

    def test_causal_exact_match(self):
        """Causal mode should return exact matches (gap = 0)."""
        sync = _make_synchronizer(causal=True)
        _inject_messages(sync, TOPIC, [1.0, 2.0, 3.0])

        msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)

        assert msg is not None
        assert msg.timestamp == 2.0
        assert gap == pytest.approx(0.0)

    def test_causal_exact_match_at_first_timestamp(self):
        """Causal mode handles exact match at the very first message."""
        sync = _make_synchronizer(causal=True)
        _inject_messages(sync, TOPIC, [5.0, 6.0, 7.0])

        msg, gap = sync.find_nearest_with_gap(TOPIC, 5.0)

        assert msg is not None
        assert msg.timestamp == 5.0
        assert gap == pytest.approx(0.0)

    def test_causal_exact_match_at_last_timestamp(self):
        """Causal mode handles exact match at the very last message."""
        sync = _make_synchronizer(causal=True)
        _inject_messages(sync, TOPIC, [5.0, 6.0, 7.0])

        msg, gap = sync.find_nearest_with_gap(TOPIC, 7.0)

        assert msg is not None
        assert msg.timestamp == 7.0
        assert gap == pytest.approx(0.0)


# ─── Edge cases ──────────────────────────────────────────────────────────


class TestCausalEdgeCases:
    """Test edge cases for causal sync."""

    def test_causal_target_before_all_messages(self):
        """No past messages exist → should return None."""
        sync = _make_synchronizer(causal=True)
        _inject_messages(sync, TOPIC, [5.0, 6.0, 7.0])

        msg, gap = sync.find_nearest_with_gap(TOPIC, 4.0)

        assert msg is None, "No past messages exist, should return None"

    def test_non_causal_target_before_all_messages(self):
        """Non-causal can still pick the nearest future message."""
        sync = _make_synchronizer(causal=False)
        _inject_messages(sync, TOPIC, [5.0, 6.0, 7.0])

        msg, gap = sync.find_nearest_with_gap(TOPIC, 4.0)

        assert msg is not None
        assert msg.timestamp == 5.0

    def test_causal_target_after_all_messages(self):
        """Target is after all messages → should pick the last one."""
        sync = _make_synchronizer(causal=True, tolerance_s=10.0)
        _inject_messages(sync, TOPIC, [1.0, 2.0, 3.0])

        msg, gap = sync.find_nearest_with_gap(TOPIC, 8.0)

        assert msg is not None
        assert msg.timestamp == 3.0
        assert gap == pytest.approx(5.0)

    def test_causal_single_message_before_target(self):
        """Single message that's before target."""
        sync = _make_synchronizer(causal=True)
        _inject_messages(sync, TOPIC, [1.0])

        msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)

        assert msg is not None
        assert msg.timestamp == 1.0
        assert gap == pytest.approx(1.0)

    def test_causal_single_message_after_target(self):
        """Single message that's after target → no past messages."""
        sync = _make_synchronizer(causal=True)
        _inject_messages(sync, TOPIC, [5.0])

        msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)

        assert msg is None, "Only future messages exist, causal should return None"

    def test_causal_empty_messages(self):
        """No messages at all."""
        sync = _make_synchronizer(causal=True)
        _inject_messages(sync, TOPIC, [])

        msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)

        assert msg is None
        assert gap is None


# ─── Tolerance enforcement ───────────────────────────────────────────────


class TestCausalTolerance:
    """Test that tolerance is enforced correctly in causal mode."""

    def test_causal_within_tolerance(self):
        """Past message within tolerance → returned."""
        sync = _make_synchronizer(causal=True, tolerance_s=0.5)
        _inject_messages(sync, TOPIC, [1.7])

        msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)

        assert msg is not None
        assert msg.timestamp == 1.7
        assert gap == pytest.approx(0.3)

    def test_causal_beyond_tolerance(self):
        """Past message beyond tolerance → None returned."""
        sync = _make_synchronizer(causal=True, tolerance_s=0.5)
        _inject_messages(sync, TOPIC, [1.0])

        msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)

        assert msg is None, "Past message is 1.0s away, tolerance is 0.5s"
        assert gap == pytest.approx(1.0)  # gap is still reported

    def test_causal_ignores_future_within_tolerance(self):
        """Future message within tolerance should still be ignored in causal mode."""
        sync = _make_synchronizer(causal=True, tolerance_s=0.5)
        # Only future message exists, and it's within tolerance
        _inject_messages(sync, TOPIC, [2.1])

        msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)

        assert msg is None, "Future message should be ignored even if within tolerance"

    def test_non_causal_picks_future_within_tolerance(self):
        """Non-causal should pick a future message that's within tolerance."""
        sync = _make_synchronizer(causal=False, tolerance_s=0.5)
        _inject_messages(sync, TOPIC, [2.1])

        msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)

        assert msg is not None
        assert msg.timestamp == 2.1


# ─── Gap values ──────────────────────────────────────────────────────────


class TestCausalGapValues:
    """Test that gap values are always positive (never negative) in causal mode."""

    def test_causal_gap_is_always_positive(self):
        """Gap in causal mode should be target - past_timestamp (always >= 0)."""
        sync = _make_synchronizer(causal=True, tolerance_s=10.0)
        _inject_messages(sync, TOPIC, [1.0, 2.0, 3.0, 4.0, 5.0])

        for target in [1.5, 2.5, 3.5, 4.5, 5.5]:
            msg, gap = sync.find_nearest_with_gap(TOPIC, target)
            if msg is not None:
                assert gap >= 0, f"Gap should be >= 0 for target={target}, got {gap}"
                assert msg.timestamp <= target, (
                    f"Causal message at {msg.timestamp} is after target {target}!"
                )


# ─── Systematic comparison ───────────────────────────────────────────────


class TestCausalVsNonCausal:
    """Systematic comparison: causal should never return a future message."""

    def test_sweep_causal_never_returns_future(self):
        """Sweep target across a range; causal must never return a future timestamp."""
        sync = _make_synchronizer(causal=True, tolerance_s=10.0)
        timestamps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        _inject_messages(sync, TOPIC, timestamps)

        # Sweep in fine increments
        for i in range(101):
            target = i * 0.01  # 0.00, 0.01, ..., 1.00
            msg, gap = sync.find_nearest_with_gap(TOPIC, target)

            if msg is not None:
                assert msg.timestamp <= target, (
                    f"CAUSAL VIOLATION: target={target}, msg.timestamp={msg.timestamp}"
                )
            else:
                # msg is None → no past message within tolerance, or no past message at all
                # Verify: all messages must be strictly after target
                past_msgs = [t for t in timestamps if t <= target]
                if past_msgs:
                    # There ARE past messages but they're beyond tolerance
                    closest_past = max(past_msgs)
                    assert target - closest_past > sync.tolerance, (
                        f"Past msg at {closest_past} is within tolerance of target {target} "
                        f"but find_nearest returned None"
                    )

    def test_sweep_non_causal_picks_closest(self):
        """Non-causal should always pick closest regardless of direction."""
        sync = _make_synchronizer(causal=False, tolerance_s=10.0)
        timestamps = [0.0, 0.5, 1.0]
        _inject_messages(sync, TOPIC, timestamps)

        # Target 0.3: closer to 0.5 (future, gap=0.2) than to 0.0 (past, gap=0.3)
        msg, gap = sync.find_nearest_with_gap(TOPIC, 0.3)
        assert msg is not None
        assert msg.timestamp == 0.5  # picks closer future msg

        # Same scenario in causal mode: must pick 0.0
        sync_causal = _make_synchronizer(causal=True, tolerance_s=10.0)
        _inject_messages(sync_causal, TOPIC, timestamps)

        msg, gap = sync_causal.find_nearest_with_gap(TOPIC, 0.3)
        assert msg is not None
        assert msg.timestamp == 0.0  # must pick past msg


# ─── Preference when both exact and past exist ──────────────────────────


class TestCausalPreference:
    """When exact match and past candidate both exist, exact should win."""

    def test_exact_match_preferred_over_past(self):
        """If target exactly matches a timestamp, pick it (gap=0) not the one before."""
        sync = _make_synchronizer(causal=True)
        _inject_messages(sync, TOPIC, [1.0, 2.0, 3.0])

        msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)

        assert msg.timestamp == 2.0
        assert gap == pytest.approx(0.0)
