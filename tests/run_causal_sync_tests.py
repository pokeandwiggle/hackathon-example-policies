#!/usr/bin/env python3
"""Run causal sync tests without pytest dependency."""

import sys
import traceback
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, "src")

from example_policies.data_ops.pipeline.frame_synchronizer import (
    FrameSynchronizer,
    TimestampedMessage,
)
from example_policies.data_ops.config.rosbag_topics import RosSchemaEnum, RosTopicEnum


def make_msg(ts: float) -> TimestampedMessage:
    return TimestampedMessage(timestamp=ts, data=b"", schema_name=RosSchemaEnum.JOINT)


TOPIC = RosTopicEnum.ACTUAL_JOINT_STATE


def _make_synchronizer(causal: bool, tolerance_s: float = 1.0) -> FrameSynchronizer:
    config = MagicMock()
    config.target_fps = 10
    config.include_rgb_images = False
    config.include_depth_images = False
    config.requires_tcp_poses = MagicMock(return_value=False)
    config.is_tcp_action = MagicMock(return_value=False)
    config.is_joint_action = MagicMock(return_value=False)
    return FrameSynchronizer(config, tolerance_ms=tolerance_s * 1000, causal=causal)


def _inject_messages(sync, topic, timestamps):
    msgs = [make_msg(ts) for ts in timestamps]
    sync.messages[topic] = msgs
    sync.timestamps[topic] = timestamps


passed = 0
failed = 0
errors = []


def run_test(name, func):
    global passed, failed
    try:
        func()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:
        failed += 1
        errors.append((name, e))
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()


# ─── Core causal behavior ────────────────────────────────────────────


def test_causal_picks_past_not_future():
    sync = _make_synchronizer(causal=True)
    _inject_messages(sync, TOPIC, [1.0, 3.0])
    msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)
    assert msg is not None, "Should find a past message"
    assert msg.timestamp == 1.0, f"Causal should pick past msg (1.0), got {msg.timestamp}"
    assert abs(gap - 1.0) < 1e-9, f"Gap should be 1.0, got {gap}"


def test_non_causal_picks_future_when_closer():
    sync = _make_synchronizer(causal=False)
    _inject_messages(sync, TOPIC, [1.0, 2.5])
    msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)
    assert msg is not None
    assert msg.timestamp == 2.5, f"Non-causal should pick closer future msg (2.5), got {msg.timestamp}"
    assert abs(gap - 0.5) < 1e-9


def test_causal_ignores_closer_future_message():
    sync = _make_synchronizer(causal=True)
    _inject_messages(sync, TOPIC, [1.0, 2.01])
    msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)
    assert msg is not None
    assert msg.timestamp == 1.0, f"Causal must pick past (1.0), not closer future (2.01), got {msg.timestamp}"


# ─── Exact matches ──────────────────────────────────────────────────


def test_causal_exact_match():
    sync = _make_synchronizer(causal=True)
    _inject_messages(sync, TOPIC, [1.0, 2.0, 3.0])
    msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)
    assert msg is not None
    assert msg.timestamp == 2.0
    assert abs(gap) < 1e-9


def test_causal_exact_match_at_first():
    sync = _make_synchronizer(causal=True)
    _inject_messages(sync, TOPIC, [5.0, 6.0, 7.0])
    msg, gap = sync.find_nearest_with_gap(TOPIC, 5.0)
    assert msg is not None
    assert msg.timestamp == 5.0
    assert abs(gap) < 1e-9


def test_causal_exact_match_at_last():
    sync = _make_synchronizer(causal=True)
    _inject_messages(sync, TOPIC, [5.0, 6.0, 7.0])
    msg, gap = sync.find_nearest_with_gap(TOPIC, 7.0)
    assert msg is not None
    assert msg.timestamp == 7.0
    assert abs(gap) < 1e-9


# ─── Edge cases ──────────────────────────────────────────────────────


def test_causal_target_before_all_messages():
    sync = _make_synchronizer(causal=True)
    _inject_messages(sync, TOPIC, [5.0, 6.0, 7.0])
    msg, gap = sync.find_nearest_with_gap(TOPIC, 4.0)
    assert msg is None, f"No past messages, should return None but got ts={msg.timestamp}"


def test_non_causal_target_before_all_messages():
    sync = _make_synchronizer(causal=False)
    _inject_messages(sync, TOPIC, [5.0, 6.0, 7.0])
    msg, gap = sync.find_nearest_with_gap(TOPIC, 4.0)
    assert msg is not None
    assert msg.timestamp == 5.0


def test_causal_target_after_all():
    sync = _make_synchronizer(causal=True, tolerance_s=10.0)
    _inject_messages(sync, TOPIC, [1.0, 2.0, 3.0])
    msg, gap = sync.find_nearest_with_gap(TOPIC, 8.0)
    assert msg is not None
    assert msg.timestamp == 3.0
    assert abs(gap - 5.0) < 1e-9


def test_causal_single_msg_before():
    sync = _make_synchronizer(causal=True)
    _inject_messages(sync, TOPIC, [1.0])
    msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)
    assert msg is not None
    assert msg.timestamp == 1.0


def test_causal_single_msg_after():
    sync = _make_synchronizer(causal=True)
    _inject_messages(sync, TOPIC, [5.0])
    msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)
    assert msg is None, f"Only future msgs, causal should return None, got ts={msg.timestamp}"


def test_causal_empty():
    sync = _make_synchronizer(causal=True)
    _inject_messages(sync, TOPIC, [])
    msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)
    assert msg is None
    assert gap is None


# ─── Tolerance ───────────────────────────────────────────────────────


def test_causal_within_tolerance():
    sync = _make_synchronizer(causal=True, tolerance_s=0.5)
    _inject_messages(sync, TOPIC, [1.7])
    msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)
    assert msg is not None
    assert msg.timestamp == 1.7
    assert abs(gap - 0.3) < 1e-9


def test_causal_beyond_tolerance():
    sync = _make_synchronizer(causal=True, tolerance_s=0.5)
    _inject_messages(sync, TOPIC, [1.0])
    msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)
    assert msg is None, f"Past msg 1.0s away, tolerance 0.5s, should return None"
    assert abs(gap - 1.0) < 1e-9


def test_causal_ignores_future_within_tolerance():
    sync = _make_synchronizer(causal=True, tolerance_s=0.5)
    _inject_messages(sync, TOPIC, [2.1])
    msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)
    assert msg is None, f"Future msg should be ignored even if within tolerance, got ts={msg.timestamp}"


def test_non_causal_picks_future_within_tolerance():
    sync = _make_synchronizer(causal=False, tolerance_s=0.5)
    _inject_messages(sync, TOPIC, [2.1])
    msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)
    assert msg is not None
    assert msg.timestamp == 2.1


# ─── Gap sign ────────────────────────────────────────────────────────


def test_causal_gap_always_positive():
    sync = _make_synchronizer(causal=True, tolerance_s=10.0)
    _inject_messages(sync, TOPIC, [1.0, 2.0, 3.0, 4.0, 5.0])
    for target in [1.5, 2.5, 3.5, 4.5, 5.5]:
        msg, gap = sync.find_nearest_with_gap(TOPIC, target)
        if msg is not None:
            assert gap >= 0, f"Gap should be >= 0 for target={target}, got {gap}"
            assert msg.timestamp <= target, f"Causal msg at {msg.timestamp} is after target {target}"


# ─── Sweep test ──────────────────────────────────────────────────────


def test_sweep_causal_never_returns_future():
    sync = _make_synchronizer(causal=True, tolerance_s=10.0)
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    _inject_messages(sync, TOPIC, timestamps)

    for i in range(101):
        target = i * 0.01
        msg, gap = sync.find_nearest_with_gap(TOPIC, target)
        if msg is not None:
            assert msg.timestamp <= target + 1e-9, (
                f"CAUSAL VIOLATION: target={target}, msg.timestamp={msg.timestamp}"
            )


def test_comparison_causal_vs_non_causal():
    timestamps = [0.0, 0.5, 1.0]

    # Non-causal: target 0.3, closer to 0.5 (future) than 0.0 (past)
    sync_nc = _make_synchronizer(causal=False, tolerance_s=10.0)
    _inject_messages(sync_nc, TOPIC, timestamps)
    msg, _ = sync_nc.find_nearest_with_gap(TOPIC, 0.3)
    assert msg.timestamp == 0.5, f"Non-causal should pick 0.5, got {msg.timestamp}"

    # Causal: same scenario, must pick 0.0
    sync_c = _make_synchronizer(causal=True, tolerance_s=10.0)
    _inject_messages(sync_c, TOPIC, timestamps)
    msg, _ = sync_c.find_nearest_with_gap(TOPIC, 0.3)
    assert msg.timestamp == 0.0, f"Causal should pick 0.0, got {msg.timestamp}"


def test_exact_preferred_over_past():
    sync = _make_synchronizer(causal=True)
    _inject_messages(sync, TOPIC, [1.0, 2.0, 3.0])
    msg, gap = sync.find_nearest_with_gap(TOPIC, 2.0)
    assert msg.timestamp == 2.0
    assert abs(gap) < 1e-9


# ─── Run all tests ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Running causal sync tests")
    print("=" * 60)

    tests = [
        # Core behavior
        ("causal_picks_past_not_future", test_causal_picks_past_not_future),
        ("non_causal_picks_future_when_closer", test_non_causal_picks_future_when_closer),
        ("causal_ignores_closer_future_message", test_causal_ignores_closer_future_message),
        # Exact matches
        ("causal_exact_match", test_causal_exact_match),
        ("causal_exact_match_at_first", test_causal_exact_match_at_first),
        ("causal_exact_match_at_last", test_causal_exact_match_at_last),
        # Edge cases
        ("causal_target_before_all_messages", test_causal_target_before_all_messages),
        ("non_causal_target_before_all_messages", test_non_causal_target_before_all_messages),
        ("causal_target_after_all", test_causal_target_after_all),
        ("causal_single_msg_before", test_causal_single_msg_before),
        ("causal_single_msg_after", test_causal_single_msg_after),
        ("causal_empty", test_causal_empty),
        # Tolerance
        ("causal_within_tolerance", test_causal_within_tolerance),
        ("causal_beyond_tolerance", test_causal_beyond_tolerance),
        ("causal_ignores_future_within_tolerance", test_causal_ignores_future_within_tolerance),
        ("non_causal_picks_future_within_tolerance", test_non_causal_picks_future_within_tolerance),
        # Gap sign
        ("causal_gap_always_positive", test_causal_gap_always_positive),
        # Sweep
        ("sweep_causal_never_returns_future", test_sweep_causal_never_returns_future),
        ("comparison_causal_vs_non_causal", test_comparison_causal_vs_non_causal),
        ("exact_preferred_over_past", test_exact_preferred_over_past),
    ]

    for name, func in tests:
        run_test(name, func)

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)
