#!/usr/bin/env -S nix run ./nix#bazelisk -- run //example_policies:dataset_conversion_synced --

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

"""Sensor-timestamp synchronized dataset conversion.

This script converts MCAP episodes to LeRobot dataset format using
sensor timestamps for synchronization instead of log_time (arrival order).

Key features:
- Generates synthetic timestamps at a fixed target frequency
- Finds nearest messages from all topics within a tolerance window
- Messages from slower topics may be reused for multiple frames

For log_time based conversion, use dataset_conversion.py instead.
"""

import dataclasses
import enum
import pathlib
import time

import draccus
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from example_policies.data_ops.config import pipeline_config
from example_policies.data_ops.config.dataset_type import DatasetType
from example_policies.data_ops.config.rosbag_topics import RosTopicEnum
from example_policies.data_ops.filters import (
    FilterConfig,
    FilterPipeline,
    FrameFilterData,
    create_filter_pipeline,
)
from example_policies.data_ops.filters.base import quality_meets_minimum
from example_policies.data_ops.pipeline.frame_assembler import FrameAssembler
from example_policies.data_ops.pipeline.frame_parser import FrameParser
from example_policies.data_ops.pipeline.frame_synchronizer import (
    FrameSynchronizer,
    TimestampedMessage,
)
from example_policies.data_ops.pipeline.frame_targeter import FrameTargeter
from example_policies.data_ops.utils.conversion_utils import (
    AnnotationExtractor,
    extract_robot_type_from_mcap,
    get_selected_episodes,
    save_metadata,
    validate_input_dir,
)


class SyncedFrameBuffer:
    """Adapter to make synced frame dict compatible with FrameParser.

    FrameParser expects a FrameBuffer-like object with get_msg() method.
    This class wraps the synced frame dictionary to provide that interface.
    """

    def __init__(self, synced_frame: dict[RosTopicEnum, TimestampedMessage]):
        self.synced_frame = synced_frame

    def get_msg(self, topic: RosTopicEnum) -> tuple[bytes, any]:
        """Get message data and schema for a topic."""
        ts_msg = self.synced_frame[topic]
        return (ts_msg.data, ts_msg.schema_name)

    def is_complete(self) -> bool:
        """Always returns True since synced frames are pre-validated."""
        return True


class SyncedEpisodeConverter:
    """Episode converter using synthetic timestamp synchronization.

    This converter uses sensor timestamps to synchronize messages across
    topics at a fixed output frequency (config.target_fps).
    """

    def __init__(
        self,
        output_dir: pathlib.Path,
        config: pipeline_config.PipelineConfig,
        features: dict,
        tolerance_ms: float,
        robot_type: str = "panda_bimanual",
        causal: bool = False,
        filter_config: FilterConfig | None = None,
    ):
        self.config = config
        self.output_dir = output_dir
        self.tolerance_ms = tolerance_ms

        # Sync components
        self.frame_synchronizer = FrameSynchronizer(
            config, tolerance_ms=tolerance_ms, causal=causal
        )

        # Parsing and assembly (reuse existing pipeline)
        self.frame_parser = FrameParser(config)
        self.frame_assembler = FrameAssembler(config)
        self.frame_targeter = FrameTargeter(config)

        # Filter pipeline (new — replaces FrameTargeter when set)
        self.filter_pipeline: FilterPipeline | None = None
        self.filter_config: FilterConfig | None = filter_config
        if filter_config is not None:
            self.filter_pipeline = create_filter_pipeline(
                filter_config, target_fps=config.target_fps
            )

        # Dataset writer
        self.dataset = LeRobotDataset.create(
            repo_id="local_only",
            fps=config.target_fps,
            root=output_dir,
            robot_type=robot_type,
            use_videos=True,
            image_writer_threads=16,
            image_writer_processes=8,
            features=features,
        )

        # Tracking
        self.episode_counter = 0
        self.episode_mapping: dict[int, str] = {}
        self.blacklist: list[int] = []
        self.episodes_skipped_quality: int = 0
        self.start_time = time.time()

        # Frame mapping for annotation remapping (set after each process_episode)
        self._last_raw_frame_kept: list[bool] = []
        self._last_sync_abs_start_s: float = 0.0

        # Per-episode filter results (populated when filter_pipeline is used)
        self.episode_filter_results: dict[int, "EpisodeFilterResult"] = {}

    def reset_episode_state(self) -> None:
        """Reset state for new episode."""
        self.frame_assembler.reset()
        self.frame_targeter.reset()

    def process_episode(
        self, episode_path: pathlib.Path, episode_idx: int
    ) -> bool:
        """Process an episode using sensor-timestamp synchronization.

        Args:
            episode_path: Path to the MCAP episode file
            episode_idx: Index of the episode

        Returns:
            True if episode was saved, False otherwise
        """
        self.reset_episode_state()

        # Pass 1: Ingest all messages
        print("  Ingesting messages...")
        self.frame_synchronizer.ingest_episode(episode_path)

        if self.filter_pipeline is not None:
            return self._process_with_filters(episode_path, episode_idx)
        else:
            return self._process_legacy(episode_path, episode_idx)

    # ------------------------------------------------------------------
    # Legacy path (FrameTargeter-based pause skipping)
    # ------------------------------------------------------------------

    def _process_legacy(
        self, episode_path: pathlib.Path, episode_idx: int
    ) -> bool:
        """Original per-frame FrameTargeter approach."""
        saved_frames = 0
        skipped_pauses = 0
        raw_frame_kept: list[bool] = []
        for synced_frame in self.frame_synchronizer.generate_synced_frames():
            frame_buffer = SyncedFrameBuffer(synced_frame)

            target_datasets = self.frame_targeter.determine_targets(
                frame_buffer, self.frame_parser
            )
            if DatasetType.MAIN not in target_datasets:
                skipped_pauses += 1
                raw_frame_kept.append(False)
                continue

            parsed = self.frame_parser.parse_frame(frame_buffer)
            assembled = self.frame_assembler.assemble(parsed)

            assembled["task"] = self.config.task_name
            self.dataset.add_frame(assembled)
            saved_frames += 1
            raw_frame_kept.append(True)

        print(f"  Saved {saved_frames} frames (skipped {skipped_pauses} pauses)")

        self._last_raw_frame_kept = raw_frame_kept
        self._last_sync_abs_start_s = self.frame_synchronizer.get_sync_start_offset()
        return self._finalize_episode(episode_path, saved_frames)

    # ------------------------------------------------------------------
    # New path (FilterPipeline-based)
    # ------------------------------------------------------------------

    def _process_with_filters(
        self, episode_path: pathlib.Path, episode_idx: int
    ) -> bool:
        """Two-pass approach: analyse all frames, then write kept frames."""
        assert self.filter_pipeline is not None

        # Collect all synced frames
        synced_frames = list(
            self.frame_synchronizer.generate_synced_frames()
        )

        # Extract lightweight filter data from each frame
        filter_data: list[FrameFilterData] = []
        target_fps = self.config.target_fps
        for i, synced_frame in enumerate(synced_frames):
            fb = SyncedFrameBuffer(synced_frame)
            raw = self.frame_parser.parse_filter_data(fb)
            filter_data.append(
                FrameFilterData(
                    index=i,
                    timestamp_s=i / target_fps,
                    des_gripper_left=float(raw["des_gripper_left"][0]),
                    des_gripper_right=float(raw["des_gripper_right"][0]),
                    joint_velocity_norm=float(
                        np.sum(np.abs(raw["joint_velocity"]))
                    ),
                    gripper_state=raw["gripper_state"],
                )
            )

        # Run filter pipeline
        filter_result = self.filter_pipeline.run(filter_data)

        # Report filter events
        if filter_result.events:
            print(f"  Filter events ({len(filter_result.events)}):")
            for ev in filter_result.events:
                print(f"    [{ev.filter_name}] {ev.description}")
        print(f"  Episode quality: {filter_result.quality}")

        # Quality gate: skip episode entirely if below min_quality
        min_q = self.filter_config.min_quality if self.filter_config else "ok"
        if not quality_meets_minimum(filter_result.quality, min_q):
            print(
                f"  Skipping episode (quality '{filter_result.quality}' "
                f"< min '{min_q}')"
            )
            # Store result even for skipped episodes so stats are available
            self.episode_filter_results[self.episode_counter] = filter_result
            self._last_raw_frame_kept = [False] * len(synced_frames)
            self.episodes_skipped_quality += 1
            return False

        # Write kept frames (full parse + assemble)
        saved_frames = 0
        raw_frame_kept: list[bool] = []
        for i, synced_frame in enumerate(synced_frames):
            if not filter_result.should_keep(i):
                raw_frame_kept.append(False)
                continue

            fb = SyncedFrameBuffer(synced_frame)
            parsed = self.frame_parser.parse_frame(fb)
            assembled = self.frame_assembler.assemble(parsed)
            assembled["task"] = self.config.task_name
            self.dataset.add_frame(assembled)
            saved_frames += 1
            raw_frame_kept.append(True)

        trimmed = filter_result.trimmed_count
        print(
            f"  Saved {saved_frames} frames "
            f"(trimmed {trimmed}, total synced {len(synced_frames)})"
        )

        self._last_raw_frame_kept = raw_frame_kept
        self._last_sync_abs_start_s = self.frame_synchronizer.get_sync_start_offset()

        # Store filter result for this episode
        ep_idx = self.episode_counter
        self.episode_filter_results[ep_idx] = filter_result

        return self._finalize_episode(episode_path, saved_frames)

    def _finalize_episode(
        self, episode_path: pathlib.Path, saved_frames: int
    ) -> bool:
        """Shared tail logic for both legacy and filter paths."""
        if saved_frames == 0:
            return False

        self.dataset.save_episode()
        self.episode_mapping[self.episode_counter] = str(episode_path)

        min_frames = self.config.min_episode_seconds * self.config.target_fps
        if saved_frames < min_frames:
            print(f"  Episode too short ({saved_frames} frames), adding to blacklist")
            self.blacklist.append(self.episode_counter)

        self.episode_counter += 1
        return True

    def get_total_time(self) -> float:
        """Get total elapsed time."""
        return time.time() - self.start_time


def convert_episodes_synced(
    episode_paths: list[pathlib.Path],
    output_dir: pathlib.Path,
    config: pipeline_config.PipelineConfig,
    tolerance_ms: float,
    with_annotations: bool = False,
    causal: bool = False,
    filter_config: FilterConfig | None = None,
) -> dict:
    """Convert episodes using sensor-timestamp synchronization.

    Args:
        episode_paths: List of paths to .mcap episode files
        output_dir: Output directory for converted dataset
        config: Pipeline configuration (uses config.target_fps for output frequency)
        tolerance_ms: Maximum time difference for sync (milliseconds).
        with_annotations: If True, extract segment annotations from /annotation topic.
        causal: If True, only use past messages (timestamp <= target) for sync.
            This matches real-time inference where future data is unavailable.
        filter_config: Optional :class:`FilterConfig` for episode quality
            filters.  When *None*, the legacy ``FrameTargeter``-based
            pause handling is used.

    Returns:
        Dict with keys: episode_mapping, blacklist, episodes_saved, total_time,
        episode_filter_results, and optionally annotation_extractor.
    """
    features = pipeline_config.build_features(config)

    # Extract robot type from first episode
    robot_type = "panda_bimanual"
    if episode_paths:
        robot_type = extract_robot_type_from_mcap(episode_paths[0])
        print(f"  Detected robot type: {robot_type}")

    converter = SyncedEpisodeConverter(
        output_dir,
        config,
        features,
        tolerance_ms=tolerance_ms,
        robot_type=robot_type,
        causal=causal,
        filter_config=filter_config,
    )

    # Initialize annotation extractor if requested
    annotation_extractor = AnnotationExtractor(fps=config.target_fps) if with_annotations else None

    for ep_idx, episode_path in enumerate(episode_paths):
        print(f"Processing {episode_path}...")

        try:
            saved = converter.process_episode(episode_path, ep_idx)

            # Extract annotations after successful conversion
            if annotation_extractor and saved:
                dataset_ep_idx = converter.episode_counter - 1
                annotation_extractor.extract_from_mcap(
                    episode_path,
                    dataset_ep_idx,
                    raw_frame_kept=converter._last_raw_frame_kept,
                    sync_abs_start_s=converter._last_sync_abs_start_s,
                )

        except Exception as e:
            print(
                f"Skipping faulty file: {episode_path} due to {type(e).__name__}: {e}"
            )
            import traceback

            traceback.print_exc()
            continue

    # Finalize the dataset - required to close parquet writers and write footer metadata
    print("Finalizing dataset...")
    converter.dataset.finalize()

    save_metadata(
        output_dir,
        converter.episode_mapping,
        converter.blacklist,
        config,
    )

    result = {
        "episode_mapping": converter.episode_mapping,
        "blacklist": converter.blacklist,
        "episodes_saved": len(converter.episode_mapping),
        "episodes_skipped_quality": converter.episodes_skipped_quality,
        "total_time": converter.get_total_time(),
        "dataset": converter.dataset,  # Return dataset for push_to_hub
        "episode_filter_results": converter.episode_filter_results,
    }

    if annotation_extractor:
        result["annotation_extractor"] = annotation_extractor

    return result


def print_conversion_result(result: dict) -> None:
    """Print conversion statistics."""
    print("\nConversion complete!")
    print(f"  - Episodes saved: {result['episodes_saved']}")
    print(f"  - Blacklisted episodes: {len(result['blacklist'])}")
    skipped_q = result.get("episodes_skipped_quality", 0)
    if skipped_q:
        print(f"  - Skipped by quality gate: {skipped_q}")
    print(f"  - Total time: {result['total_time']:.2f}s")

    # Print filter quality summary if available
    filter_results = result.get("episode_filter_results", {})
    if filter_results:
        from collections import Counter

        qualities = Counter(
            fr.quality for fr in filter_results.values()
        )
        print("\n  Episode quality breakdown:")
        for q in ["excellent", "good", "ok", "bad"]:
            if q in qualities:
                print(f"    - {q}: {qualities[q]}")

        total_events = sum(
            len(fr.events) for fr in filter_results.values()
        )
        if total_events:
            print(f"  Total filter events: {total_events}")



@dataclasses.dataclass
class ScriptArgs:
    """Arguments for synced conversion script."""

    episodes_dir: pathlib.Path = pathlib.Path("./data")
    output: pathlib.Path = pathlib.Path("./output")

    # Sync-specific parameters
    tolerance_ms: float | None = None  # Auto-computed from target_fps if None
    max_episodes: int | None = None  # Stop after N episodes (None = no limit)
    causal: bool = True  # Only use past messages for sync (no future lookahead)

    # Episode filtering
    success_only: bool = True  # Include only successful episodes (based on MCAP metadata)
    excellent_only: bool = True  # Include only episodes with 'excellent' quality

    # --- Episode quality filters ----------------------------------------
    enable_filters: bool = False  # Enable the new filter pipeline (replaces FrameTargeter)
    trim_leading_pauses: bool = True  # Trim idle frames at episode start
    trim_trailing_pauses: bool = False  # Trim idle frames at episode end
    gripper_command_threshold: float = 0.5  # Threshold for open/closed classification
    full_cycle_threshold_s: float = 1.3  # Max off→on→off cycle time
    min_change_interval_s: float = 0.65  # Min time between gripper changes
    moving_velocity_threshold: float = 0.03  # Velocity norm for "arm is moving"
    min_quality: str = "excellent"  # Minimum episode quality to include (excellent/good/ok/bad)

    # Annotation extraction (opt-in)
    with_annotations: bool = False  # Extract segment annotations from /annotation topic
    skip_failed_segments: bool = False  # Skip frames from failed segments (requires --with-annotations)

    # HuggingFace Hub upload
    push_to_hub: bool = False  # Push dataset to HuggingFace Hub after conversion
    repo_id: str | None = None  # HuggingFace repo ID (e.g., 'user/dataset-name')


@dataclasses.dataclass
class SyncedConfig(ScriptArgs, pipeline_config.PipelineConfig):
    """Configuration for synced conversion.

    Inherits from ScriptArgs and PipelineConfig to include all parameters.
    """

    def to_dict(self):
        data = dataclasses.asdict(self)
        for key, value in data.items():
            if isinstance(value, pathlib.Path):
                data[key] = str(value)
            if isinstance(value, enum.Enum):
                data[key] = value.value
        return data


def main():
    """
    Sensor-timestamp synchronized conversion.

    Example usage:
        python dataset_conversion_synced.py \\
            --episodes-dir /path/to/episodes \\
            --output /path/to/output \\
            --target-fps 10 \\
            --tolerance-ms 40 \\
            --action-level DELTA_TCP

        # With annotation extraction:
        python dataset_conversion_synced.py \\
            --episodes-dir /path/to/episodes \\
            --output /path/to/output \\
            --with-annotations

    Use --help to see all available options and their descriptions.
    """
    config = draccus.parse(config_class=SyncedConfig)

    validate_input_dir(config.episodes_dir)

    # Compute tolerance if not specified
    tolerance_ms = config.tolerance_ms
    if tolerance_ms is None:
        tolerance_ms = 1000.0 / config.target_fps

    print(f"Converting episodes from: {config.episodes_dir}")
    print(f"Output directory: {config.output}")
    print("Sync config:")
    print(f"  - Target FPS: {config.target_fps}Hz")
    print(f"  - Tolerance: {tolerance_ms:.1f}ms {'(auto)' if config.tolerance_ms is None else ''}")
    if config.causal:
        print("  - Causal sync: enabled (past-only, no future lookahead)")
    if config.with_annotations:
        print("  - Annotations: enabled")
        if config.skip_failed_segments:
            print("  - Skip failed segments: enabled")
    if config.push_to_hub:
        print(f"  - Push to hub: {config.repo_id or 'repo_id required!'}")
    print("Pipeline config:")
    print(f"  - Action level: {config.action_level}")
    print(f"  - Image resolution: {config.image_resolution}")
    print(f"  - Task: {config.task_name}")
    # Validate push_to_hub requires repo_id
    if config.push_to_hub and not config.repo_id:
        raise ValueError("--repo_id is required when using --push_to_hub")

    episode_paths = get_selected_episodes(
        config.episodes_dir,
        success_only=config.success_only,
        excellent_only=config.excellent_only,
    )
    if not config.success_only:
        print("  - Including all episodes (success_only=False)")
    elif not config.excellent_only:
        print("  - Including ok/good/excellent episodes (excellent_only=False)")
    if config.max_episodes is not None:
        episode_paths = episode_paths[:config.max_episodes]
        print(f"  - Limiting to first {config.max_episodes} episodes")

    result = convert_episodes_synced(
        episode_paths,
        config.output,
        config,
        tolerance_ms=tolerance_ms,
        with_annotations=config.with_annotations,
        causal=config.causal,
        filter_config=FilterConfig(
            max_pause_seconds=config.max_pause_seconds,
            pause_velocity=config.pause_velocity,
            trim_leading_pauses=config.trim_leading_pauses,
            trim_trailing_pauses=config.trim_trailing_pauses,
            gripper_command_threshold=config.gripper_command_threshold,
            full_cycle_threshold_s=config.full_cycle_threshold_s,
            min_change_interval_s=config.min_change_interval_s,
            moving_velocity_threshold=config.moving_velocity_threshold,
            min_quality=config.min_quality,
        ) if config.enable_filters else None,
    )

    # Save annotations if extracted
    if config.with_annotations:
        extractor = result.get("annotation_extractor")
        if extractor is None or extractor.total_segments == 0:
            raise RuntimeError(
                "--with_annotations was set but no segment annotations were found in any episode. "
                "Ensure the MCAP files contain segment metadata "
                "('pw_episode_info' in new format, or 'segments' + 'episode_rating' in old format). "
                "If annotations are not required, remove the --with_annotations flag."
            )

        print("\nSaving annotations...")
        extractor.save(config.output)

        # Print annotation stats
        print(f"  - Episodes with annotations: {extractor.episodes_with_annotations}")
        print(f"  - Total segments: {extractor.total_segments}")

    print_conversion_result(result)

    # Push to HuggingFace Hub if requested
    if config.push_to_hub:
        print(f"\nPushing dataset to HuggingFace Hub: {config.repo_id}")
        dataset = result["dataset"]
        # Update repo_id for push
        dataset.repo_id = config.repo_id
        dataset.push_to_hub(tags=["LeRobot"], license="apache-2.0")
        print(f"Dataset uploaded to: https://huggingface.co/datasets/{config.repo_id}")


if __name__ == "__main__":
    main()
