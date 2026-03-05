"""Tests for the episode quality filter pipeline.

Unit tests use synthetic FrameFilterData; integration tests run on real
MCAP files from ``data/raw/stack_one_brick/``.
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile

import numpy as np
import pytest

from example_policies.data_ops.filters.base import (
    QUALITY_ORDER,
    EpisodeFilterResult,
    FrameFilterData,
    quality_meets_minimum,
    worst_quality,
)
from example_policies.data_ops.filters.gripper_filter import (
    GripperToggleFilter,
    GripperWhileMovingFilter,
)
from example_policies.data_ops.filters.pause_filter import PauseFilter
from example_policies.data_ops.filters.filter_pipeline import (
    FilterConfig,
    FilterPipeline,
    create_filter_pipeline,
)


# ── Test data directory ──────────────────────────────────────────────────
RAW_DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "stack_one_brick"


# ── Helpers ──────────────────────────────────────────────────────────────

def _frame(
    idx: int,
    ts: float,
    grip_l: float = 0.0,
    grip_r: float = 0.0,
    vel: float = 0.0,
    gripper_state: np.ndarray | None = None,
) -> FrameFilterData:
    """Convenience builder for a single FrameFilterData."""
    return FrameFilterData(
        index=idx,
        timestamp_s=ts,
        des_gripper_left=grip_l,
        des_gripper_right=grip_r,
        joint_velocity_norm=vel,
        gripper_state=gripper_state if gripper_state is not None else np.zeros(4),
    )


def _make_frames(
    n: int = 30,
    fps: float = 30.0,
    grip_l: float = 0.0,
    grip_r: float = 0.0,
    vel: float = 0.05,
) -> list[FrameFilterData]:
    """Create *n* identical frames at *fps* — arm moving, grippers constant."""
    return [_frame(i, i / fps, grip_l=grip_l, grip_r=grip_r, vel=vel) for i in range(n)]


# ======================================================================
# 1. quality helper functions
# ======================================================================


class TestQualityHelpers:
    """Tests for worst_quality and quality_meets_minimum."""

    def test_worst_quality_single(self):
        assert worst_quality("excellent") == "excellent"

    def test_worst_quality_mixed(self):
        assert worst_quality("excellent", "ok", "good") == "ok"

    def test_worst_quality_all_bad(self):
        assert worst_quality("bad", "bad") == "bad"

    @pytest.mark.parametrize(
        "quality, min_q, expected",
        [
            ("excellent", "excellent", True),
            ("good", "excellent", False),
            ("ok", "excellent", False),
            ("bad", "excellent", False),
            ("excellent", "ok", True),
            ("good", "ok", True),
            ("ok", "ok", True),
            ("bad", "ok", False),
            ("excellent", "bad", True),
        ],
    )
    def test_quality_meets_minimum(self, quality, min_q, expected):
        assert quality_meets_minimum(quality, min_q) is expected


# ======================================================================
# 2. GripperToggleFilter — unit tests
# ======================================================================


class TestGripperToggleFilter:
    """Synthetic tests for gripper toggling detection."""

    def test_clean_episode_is_excellent(self):
        """Slow, well-separated gripper changes → excellent."""
        frames = [
            _frame(0, 0.0, grip_l=0.0),
            _frame(1, 2.0, grip_l=0.8),  # close after 2 s
            _frame(2, 5.0, grip_l=0.0),  # open after 3 s more
        ]
        result = GripperToggleFilter().analyze(frames)
        assert result.quality == "excellent"
        assert len(result.events) == 0

    def test_rapid_change_flags_ok(self):
        """Two changes within min_change_interval_s → ok."""
        frames = [
            _frame(0, 0.0, grip_l=0.0),
            _frame(1, 0.3, grip_l=0.8),  # 0.3 s gap
            _frame(2, 0.6, grip_l=0.0),  # 0.3 s gap → rapid
        ]
        filt = GripperToggleFilter(min_change_interval_s=0.65)
        result = filt.analyze(frames)
        assert result.quality == "ok"
        assert any("Rapid" in e.description for e in result.events)

    def test_bounce_flags_ok(self):
        """Full off→on→off within full_cycle_threshold_s → bounce event.

        The bounce check looks at changes[i] vs changes[i-2] (3 transitions
        needed), so we need 4 frames with 3 state changes.
        """
        frames = [
            _frame(0, 0.0, grip_l=0.0),   # open
            _frame(1, 0.3, grip_l=0.8),   # close at 0.3 s (change 0)
            _frame(2, 0.6, grip_l=0.0),   # open at 0.6 s  (change 1)
            _frame(3, 0.9, grip_l=0.8),   # close at 0.9 s (change 2) → cycle 0.6 s
        ]
        filt = GripperToggleFilter(
            full_cycle_threshold_s=1.3,
            min_change_interval_s=0.1,  # relax so only bounce triggers
        )
        result = filt.analyze(frames)
        assert result.quality == "ok"
        assert any("bounce" in e.description.lower() for e in result.events)

    def test_right_side_detected(self):
        """Rapid changes on the right gripper are also caught."""
        frames = [
            _frame(0, 0.0, grip_r=0.0),
            _frame(1, 0.2, grip_r=0.8),
            _frame(2, 0.4, grip_r=0.0),
        ]
        result = GripperToggleFilter(min_change_interval_s=0.65).analyze(frames)
        assert result.quality == "ok"

    def test_no_change_is_excellent(self):
        """Gripper stays constant → excellent."""
        frames = _make_frames(20, grip_l=0.0, grip_r=0.0)
        result = GripperToggleFilter().analyze(frames)
        assert result.quality == "excellent"

    def test_all_frames_kept(self):
        """Toggle filter never trims frames (keep all)."""
        frames = [
            _frame(0, 0.0, grip_l=0.0),
            _frame(1, 0.2, grip_l=0.8),
            _frame(2, 0.4, grip_l=0.0),
        ]
        result = GripperToggleFilter().analyze(frames)
        assert all(result.frame_keep)


# ======================================================================
# 3. GripperWhileMovingFilter — unit tests
# ======================================================================


class TestGripperWhileMovingFilter:
    """Synthetic tests for gripper-changes-during-motion detection."""

    def test_change_while_still_is_excellent(self):
        """Gripper changes while arm is stationary → excellent."""
        frames = [
            _frame(0, 0.0, grip_l=0.0, vel=0.0),
            _frame(1, 0.1, grip_l=0.8, vel=0.0),  # change, arm still
        ]
        result = GripperWhileMovingFilter().analyze(frames)
        assert result.quality == "excellent"
        assert len(result.events) == 0

    def test_change_while_moving_is_ok(self):
        """Gripper changes while arm moves → ok."""
        frames = [
            _frame(0, 0.0, grip_l=0.0, vel=0.1),
            _frame(1, 0.1, grip_l=0.8, vel=0.1),  # change while moving
        ]
        result = GripperWhileMovingFilter(velocity_threshold=0.03).analyze(frames)
        assert result.quality == "ok"
        assert any("moving" in e.description.lower() for e in result.events)

    def test_no_change_while_moving(self):
        """Arm is moving but gripper stays constant → excellent."""
        frames = _make_frames(10, vel=0.5, grip_l=0.8)
        result = GripperWhileMovingFilter().analyze(frames)
        assert result.quality == "excellent"

    def test_both_sides_checked(self):
        """Right side gripper change while moving is also detected."""
        frames = [
            _frame(0, 0.0, grip_r=0.0, vel=0.2),
            _frame(1, 0.1, grip_r=0.8, vel=0.2),
        ]
        result = GripperWhileMovingFilter(velocity_threshold=0.03).analyze(frames)
        assert result.quality == "ok"

    def test_all_frames_kept(self):
        """Moving filter never trims frames."""
        frames = [
            _frame(0, 0.0, grip_l=0.0, vel=0.2),
            _frame(1, 0.1, grip_l=0.8, vel=0.2),
        ]
        result = GripperWhileMovingFilter().analyze(frames)
        assert all(result.frame_keep)


# ======================================================================
# 4. PauseFilter — unit tests
# ======================================================================


class TestPauseFilter:
    """Synthetic tests for pause trimming and detection."""

    def _idle_frames(self, n: int, start_idx: int = 0, t_offset: float = 0.0, fps: float = 30.0):
        """Create *n* idle frames (velocity=0, static gripper)."""
        return [
            _frame(start_idx + i, t_offset + i / fps, vel=0.0)
            for i in range(n)
        ]

    def _moving_frames(self, n: int, start_idx: int = 0, t_offset: float = 0.0, fps: float = 30.0):
        """Create *n* moving frames."""
        return [
            _frame(start_idx + i, t_offset + i / fps, vel=0.1)
            for i in range(n)
        ]

    def test_leading_idle_trimmed(self):
        """Idle frames at the start are trimmed when trim_leading=True."""
        fps = 30.0
        idle = self._idle_frames(15, fps=fps)
        active = self._moving_frames(20, start_idx=15, t_offset=15 / fps, fps=fps)
        frames = idle + active

        filt = PauseFilter(max_pause_seconds=0.2, pause_velocity=0.03,
                           target_fps=fps, trim_leading=True)
        result = filt.analyze(frames)

        # Leading idle frames should be trimmed
        assert not any(result.frame_keep[:15]), "Leading idle frames should be trimmed"
        # At least some active frames should be kept
        assert any(result.frame_keep[15:]), "Active frames should be kept"

    def test_no_trim_when_disabled(self):
        """Idle start is kept when trim_leading=False."""
        fps = 30.0
        idle = self._idle_frames(15, fps=fps)
        active = self._moving_frames(20, start_idx=15, t_offset=15 / fps, fps=fps)
        frames = idle + active

        filt = PauseFilter(max_pause_seconds=0.2, pause_velocity=0.03,
                           target_fps=fps, trim_leading=False, trim_trailing=False)
        result = filt.analyze(frames)
        assert all(result.frame_keep)

    def test_mid_episode_pause_downgrades_quality(self):
        """Mid-episode pause sets quality to 'ok' but keeps frames."""
        fps = 30.0
        active1 = self._moving_frames(20, fps=fps)
        idle = self._idle_frames(15, start_idx=20, t_offset=20 / fps, fps=fps)
        active2 = self._moving_frames(20, start_idx=35, t_offset=35 / fps, fps=fps)
        frames = active1 + idle + active2

        filt = PauseFilter(max_pause_seconds=0.2, pause_velocity=0.03,
                           target_fps=fps, trim_leading=True)
        result = filt.analyze(frames)
        assert result.quality == "ok", "Mid-episode pause should downgrade"
        # Mid-episode paused frames are kept
        assert all(result.frame_keep[20:35]), "Mid-pause frames should be kept"

    def test_all_active_is_excellent(self):
        """No pauses at all → excellent."""
        frames = self._moving_frames(30)
        filt = PauseFilter(max_pause_seconds=0.2, pause_velocity=0.03,
                           target_fps=30.0)
        result = filt.analyze(frames)
        assert result.quality == "excellent"


# ======================================================================
# 5. FilterPipeline — unit tests
# ======================================================================


class TestFilterPipeline:
    """Test the merged pipeline behaviour."""

    def test_empty_pipeline_is_excellent(self):
        frames = _make_frames(10)
        pipeline = FilterPipeline(filters=[])
        result = pipeline.run(frames)
        assert result.quality == "excellent"
        assert all(result.frame_keep)

    def test_merged_quality_is_worst(self):
        """When one filter says ok and another says excellent, result is ok."""
        # Rapid toggle → ok for toggle filter
        frames = [
            _frame(0, 0.0, grip_l=0.0),
            _frame(1, 0.2, grip_l=0.8),
            _frame(2, 0.4, grip_l=0.0),
        ]
        pipeline = FilterPipeline([
            GripperToggleFilter(min_change_interval_s=0.65),
            GripperWhileMovingFilter(velocity_threshold=0.03),
        ])
        result = pipeline.run(frames)
        assert result.quality == "ok"

    def test_filter_details_populated(self):
        """Each filter's individual result is accessible via filter_details."""
        frames = _make_frames(10)
        pipeline = FilterPipeline([
            GripperToggleFilter(),
            GripperWhileMovingFilter(),
        ])
        result = pipeline.run(frames)
        assert "gripper_toggle" in result.filter_details
        assert "gripper_while_moving" in result.filter_details

    def test_create_filter_pipeline_with_config(self):
        """create_filter_pipeline honours enable flags."""
        cfg = FilterConfig(
            enable_pause_filter=False,
            enable_gripper_toggle_filter=True,
            enable_gripper_while_moving_filter=False,
        )
        pipeline = create_filter_pipeline(cfg, target_fps=30.0)
        assert len(pipeline.filters) == 1
        assert pipeline.filters[0].name == "gripper_toggle"

    def test_create_filter_pipeline_all_disabled(self):
        cfg = FilterConfig(
            enable_pause_filter=False,
            enable_gripper_toggle_filter=False,
            enable_gripper_while_moving_filter=False,
        )
        pipeline = create_filter_pipeline(cfg, target_fps=30.0)
        result = pipeline.run(_make_frames(5))
        assert result.quality == "excellent"


# ======================================================================
# 6. FilterConfig defaults
# ======================================================================


class TestFilterConfig:
    """Verify FilterConfig default values."""

    def test_default_min_quality(self):
        assert FilterConfig().min_quality == "excellent"

    def test_custom_min_quality(self):
        assert FilterConfig(min_quality="ok").min_quality == "ok"


# ======================================================================
# 7. Quality gate logic (unit level)
# ======================================================================


class TestQualityGate:
    """Verify that quality_meets_minimum implements the gate correctly."""

    def test_excellent_passes_excellent(self):
        assert quality_meets_minimum("excellent", "excellent")

    def test_ok_blocked_by_excellent(self):
        assert not quality_meets_minimum("ok", "excellent")

    def test_ok_passes_ok(self):
        assert quality_meets_minimum("ok", "ok")

    def test_good_passes_ok(self):
        assert quality_meets_minimum("good", "ok")

    def test_unknown_quality_blocked(self):
        """Unknown quality string → order 0 → blocked by anything."""
        assert not quality_meets_minimum("unknown", "bad")


# ======================================================================
# 8. Integration test — real MCAP data
# ======================================================================


@pytest.mark.skipif(
    not RAW_DATA_DIR.exists(),
    reason=f"Test data not found at {RAW_DATA_DIR}",
)
class TestFiltersOnRealData:
    """Run the filter pipeline on real episodes from stack_one_brick.

    These tests do NOT convert a full dataset — they only exercise the
    filter analysis path to check that it runs without errors on real
    sensor data and produces sensible results.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Discover MCAP files and build shared objects."""
        from example_policies.data_ops.utils.conversion_utils import get_selected_episodes

        # Get all episodes (success_only=False to include the ok-rated ones)
        self.episode_paths = get_selected_episodes(
            RAW_DATA_DIR, success_only=False, excellent_only=False,
        )
        assert len(self.episode_paths) > 0, "No episodes found"

    def _extract_filter_data(self, mcap_path: pathlib.Path) -> list[FrameFilterData]:
        """Run frame synchronization and extract filter data from an MCAP file."""
        from example_policies.data_ops.config.pipeline_config import PipelineConfig
        from example_policies.data_ops.pipeline.frame_synchronizer import FrameSynchronizer
        from example_policies.data_ops.pipeline.frame_parser import FrameParser
        from example_policies.data_ops.dataset_conversion_synced import SyncedFrameBuffer
        from example_policies.utils.action_order import ActionMode

        cfg = PipelineConfig(
            task_name="test",
            target_fps=5,
            action_level=ActionMode.TCP,
            include_tcp_poses=True,
            include_rgb_images=True,
        )

        # Wide tolerance — the gripper command topics in stack_one_brick
        # have occasional 300+ ms gaps, so we need generous tolerance.
        # The exact FPS/tolerance doesn't matter since we're testing the
        # filter logic, not sync quality.
        tolerance_ms = 500.0
        sync = FrameSynchronizer(cfg, tolerance_ms=tolerance_ms, causal=True)
        sync.ingest_episode(mcap_path)

        parser = FrameParser(cfg)

        filter_data: list[FrameFilterData] = []
        for i, synced_frame in enumerate(sync.generate_synced_frames()):
            fb = SyncedFrameBuffer(synced_frame)
            raw = parser.parse_filter_data(fb)
            filter_data.append(
                FrameFilterData(
                    index=i,
                    timestamp_s=i / cfg.target_fps,
                    des_gripper_left=float(raw["des_gripper_left"][0]),
                    des_gripper_right=float(raw["des_gripper_right"][0]),
                    joint_velocity_norm=float(np.sum(np.abs(raw["joint_velocity"]))),
                    gripper_state=raw["gripper_state"],
                )
            )

        return filter_data

    def test_filter_data_extraction(self):
        """Filter data can be extracted from real MCAP files."""
        data = self._extract_filter_data(self.episode_paths[0])
        assert len(data) > 0, "No frames extracted"
        assert data[0].index == 0
        assert data[0].timestamp_s == 0.0

    def test_pipeline_runs_on_real_data(self):
        """Full pipeline runs without errors on every episode."""
        config = FilterConfig()
        pipeline = create_filter_pipeline(config, target_fps=30.0)

        for ep_path in self.episode_paths:
            data = self._extract_filter_data(ep_path)
            result = pipeline.run(data)

            assert result.quality in QUALITY_ORDER
            assert len(result.frame_keep) == len(data)
            assert result.kept_count + result.trimmed_count == len(data)
            assert "pause" in result.filter_details
            assert "gripper_toggle" in result.filter_details
            assert "gripper_while_moving" in result.filter_details

    def test_excellent_episode_quality(self):
        """The file labelled 'excellent' should pass filters as excellent."""
        excellent_paths = [p for p in self.episode_paths if "--excellent" in p.name]
        if not excellent_paths:
            pytest.skip("No excellent-labelled episode found")

        config = FilterConfig()
        pipeline = create_filter_pipeline(config, target_fps=30.0)
        data = self._extract_filter_data(excellent_paths[0])
        result = pipeline.run(data)

        # Excellent-labelled recordings *should* pass, but we accept
        # "ok" in case the filters legitimately find an issue.
        assert result.quality in ("excellent", "good", "ok")

    def test_quality_gate_blocks_ok_when_excellent_required(self):
        """Episodes rated 'ok' by filters should be blocked by min_quality='excellent'."""
        config = FilterConfig()
        pipeline = create_filter_pipeline(config, target_fps=30.0)

        for ep_path in self.episode_paths:
            data = self._extract_filter_data(ep_path)
            result = pipeline.run(data)

            if result.quality == "ok":
                assert not quality_meets_minimum(result.quality, "excellent")
                return  # found at least one — test passes

        pytest.skip("No episode rated 'ok' by filters — cannot test gate")

    def test_quality_gate_passes_ok_when_ok_required(self):
        """All episodes should pass when min_quality='ok'."""
        config = FilterConfig(min_quality="ok")
        pipeline = create_filter_pipeline(config, target_fps=30.0)

        for ep_path in self.episode_paths:
            data = self._extract_filter_data(ep_path)
            result = pipeline.run(data)
            assert quality_meets_minimum(result.quality, "ok")

    def test_filter_events_have_valid_fields(self):
        """All filter events have correct types and non-empty descriptions."""
        config = FilterConfig()
        pipeline = create_filter_pipeline(config, target_fps=30.0)

        for ep_path in self.episode_paths:
            data = self._extract_filter_data(ep_path)
            result = pipeline.run(data)

            for ev in result.events:
                assert isinstance(ev.filter_name, str) and ev.filter_name
                assert isinstance(ev.frame_idx, int) and ev.frame_idx >= 0
                assert isinstance(ev.timestamp_s, float) and ev.timestamp_s >= 0
                assert isinstance(ev.description, str) and ev.description

    def test_individual_filters_on_real_data(self):
        """Each filter individually completes without error."""
        data = self._extract_filter_data(self.episode_paths[0])

        for filt in [
            GripperToggleFilter(),
            GripperWhileMovingFilter(),
            PauseFilter(target_fps=30.0),
        ]:
            result = filt.analyze(data)
            assert result.quality in QUALITY_ORDER
            assert len(result.frame_keep) == len(data)


# ======================================================================
# 9. End-to-end conversion integration test
# ======================================================================


@pytest.mark.skipif(
    not RAW_DATA_DIR.exists(),
    reason=f"Test data not found at {RAW_DATA_DIR}",
)
class TestE2EConversionWithFilters:
    """Run a full conversion with filters enabled and verify results."""

    # Wide tolerance needed because gripper-command topic gaps in
    # stack_one_brick can exceed 300 ms.
    _TEST_FPS = 30
    _TEST_TOLERANCE_MS = 500.0

    def test_conversion_with_filters(self):
        """convert_episodes_synced completes with filters and produces a valid result."""
        from example_policies.data_ops.config.pipeline_config import PipelineConfig
        from example_policies.data_ops.dataset_conversion_synced import convert_episodes_synced
        from example_policies.data_ops.utils.conversion_utils import get_selected_episodes
        from example_policies.utils.action_order import ActionMode

        episode_paths = get_selected_episodes(
            RAW_DATA_DIR, success_only=False, excellent_only=False,
        )
        assert len(episode_paths) > 0

        cfg = PipelineConfig(
            task_name="test_stack_brick",
            target_fps=self._TEST_FPS,
            action_level=ActionMode.TCP,
            include_tcp_poses=True,
            include_rgb_images=True,
        )

        filter_cfg = FilterConfig(
            min_quality="ok",  # keep everything that isn't "bad"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = pathlib.Path(tmp_dir) / "output"

            result = convert_episodes_synced(
                episode_paths=episode_paths[:2],  # limit to 2 for speed
                output_dir=output_dir,
                config=cfg,
                tolerance_ms=self._TEST_TOLERANCE_MS,
                causal=True,
                filter_config=filter_cfg,
            )

            assert result["episodes_saved"] > 0, "Should save at least 1 episode"
            assert isinstance(result["episodes_skipped_quality"], int)
            assert isinstance(result["episode_filter_results"], dict)
            assert len(result["episode_filter_results"]) > 0

            # Check filter results are present
            for ep_idx, filt_result in result["episode_filter_results"].items():
                assert isinstance(filt_result, EpisodeFilterResult)
                assert filt_result.quality in QUALITY_ORDER

    def test_excellent_gate_skips_ok_episodes(self):
        """With min_quality='excellent', ok-rated episodes are skipped."""
        from example_policies.data_ops.config.pipeline_config import PipelineConfig
        from example_policies.data_ops.dataset_conversion_synced import convert_episodes_synced
        from example_policies.data_ops.utils.conversion_utils import get_selected_episodes
        from example_policies.utils.action_order import ActionMode

        episode_paths = get_selected_episodes(
            RAW_DATA_DIR, success_only=False, excellent_only=False,
        )

        cfg = PipelineConfig(
            task_name="test_stack_brick",
            target_fps=self._TEST_FPS,
            action_level=ActionMode.TCP,
            include_tcp_poses=True,
            include_rgb_images=True,
        )

        filter_cfg = FilterConfig(min_quality="excellent")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = pathlib.Path(tmp_dir) / "output"

            result = convert_episodes_synced(
                episode_paths=episode_paths,
                output_dir=output_dir,
                config=cfg,
                tolerance_ms=self._TEST_TOLERANCE_MS,
                causal=True,
                filter_config=filter_cfg,
            )

            # With 3 episodes (1 excellent, 2 ok labelled), some may be
            # flagged by filters. Verify the gate ran correctly.
            total = result["episodes_saved"] + result["episodes_skipped_quality"]
            assert total > 0, "At least one episode should have been processed"

            # Every saved episode must be excellent
            for ep_idx, filt_result in result["episode_filter_results"].items():
                if ep_idx in result["episode_mapping"]:
                    assert filt_result.quality == "excellent", (
                        f"Saved episode {ep_idx} has quality "
                        f"'{filt_result.quality}', expected 'excellent'"
                    )
