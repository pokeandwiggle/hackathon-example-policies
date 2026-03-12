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

Example usage:
    python dataset_conversion_synced.py \
        --episodes_dir /data/raw/my_task/operator_name \
        --task_name "My Task" \
        --tolerance_ms 200.0 \
        --api_filter "my_api_filter_from_poker"

Use --help to see all available options and their descriptions.
"""

import dataclasses
import enum
import logging
import os
import pathlib
import sys
import time

import draccus
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from example_policies.data_ops.config import pipeline_config
from example_policies.data_ops.config.dataset_type import DatasetType
from example_policies.data_ops.config.rosbag_topics import RosTopicEnum
from example_policies.data_ops.pipeline.frame_assembler import FrameAssembler
from example_policies.data_ops.pipeline.frame_parser import FrameParser
from example_policies.data_ops.pipeline.frame_synchronizer import (
    FrameSynchronizer,
    TimestampedMessage,
)
from example_policies.data_ops.pipeline.frame_targeter import FrameTargeter
from example_policies.data_ops.utils.conversion_utils import (
    AnnotationExtractor,
    extract_embodiment_metadata,
    extract_robot_type_from_mcap,
    get_selected_episodes,
    save_metadata,
    validate_input_dir,
)
from example_policies.utils.embodiment import (
    get_joint_config,
    gripper_type_from_end_effector,
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
        causal: bool = True,
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
        self.start_time = time.time()

        # Frame mapping for annotation remapping (set after each process_episode)
        self._last_raw_frame_kept: list[bool] = []
        self._last_sync_abs_start_s: float = 0.0

    def reset_episode_state(self) -> None:
        """Reset state for new episode."""
        self.frame_assembler.reset()
        self.frame_targeter.reset()

    def process_episode(
        self, episode_path: pathlib.Path, episode_idx: int
    ) -> tuple[bool, str | None]:
        """Process an episode using sensor-timestamp synchronization.

        Args:
            episode_path: Path to the MCAP episode file
            episode_idx: Index of the episode

        Returns:
            Tuple of (saved, skip_reason). saved is True if episode was
            saved, False otherwise.  skip_reason is None when saved,
            or a human-readable explanation when skipped.
        """
        # Extract per-episode embodiment from MCAP metadata
        emb_meta = extract_embodiment_metadata(episode_path)
        embodiment = None
        episode_cfg = self.config
        if emb_meta is not None:
            embodiment = get_joint_config(emb_meta["embodiment_name"])
            gripper_overrides = {}
            ee_left = emb_meta.get("end_effector_left")
            if ee_left is not None:
                gripper_overrides["left_gripper"] = gripper_type_from_end_effector(ee_left)
            ee_right = emb_meta.get("end_effector_right")
            if ee_right is not None:
                gripper_overrides["right_gripper"] = gripper_type_from_end_effector(ee_right)
            if gripper_overrides:
                episode_cfg = dataclasses.replace(self.config, **gripper_overrides)

        self.frame_parser = FrameParser(episode_cfg, embodiment=embodiment)
        self.reset_episode_state()

        # Pass 1: Ingest all messages
        self.frame_synchronizer.ingest_episode(episode_path)

        # Pass 1.5: Pre-decode all AV1 video frames sequentially
        w, h = self.config.image_resolution
        self.frame_synchronizer.decode_video_topics(w, h)

        # Pass 2: Generate synchronized frames
        # Track which raw frame indices were kept vs skipped for annotation remapping
        saved_frames = 0
        skipped_pauses = 0
        skipped_decode = 0
        raw_frame_kept: list[bool] = []  # True if frame was kept, False if skipped
        for synced_frame in self.frame_synchronizer.generate_synced_frames():
            # Wrap synced frame for compatibility with FrameParser
            frame_buffer = SyncedFrameBuffer(synced_frame)

            # Check for pause (skip if robot is idle)
            target_datasets = self.frame_targeter.determine_targets(
                frame_buffer, self.frame_parser
            )
            if DatasetType.MAIN not in target_datasets:
                skipped_pauses += 1
                raw_frame_kept.append(False)
                continue

            # Parse and assemble using existing pipeline
            parsed = self.frame_parser.parse_frame(frame_buffer)
            if parsed is None:
                skipped_decode += 1
                raw_frame_kept.append(False)
                continue
            assembled = self.frame_assembler.assemble(parsed)

            # Add to dataset (v3.0 API: task goes in frame dict, not as kwarg)
            assembled["task"] = self.config.task_name
            self.dataset.add_frame(assembled)
            saved_frames += 1
            raw_frame_kept.append(True)

        # Store frame mapping for annotation remapping
        # sync_abs_start_s: absolute timestamp of first synced frame
        self._last_raw_frame_kept = raw_frame_kept
        self._last_sync_abs_start_s = self.frame_synchronizer.get_sync_start_offset()

        if saved_frames == 0:
            return False, self.frame_synchronizer.skip_reason or "no frames produced"

        # Save episode
        self.dataset.save_episode()

        # Track episode
        self.episode_mapping[self.episode_counter] = str(episode_path)

        # Check if episode is too short
        min_frames = self.config.min_episode_seconds * self.config.target_fps
        if saved_frames < min_frames:
            self.blacklist.append(self.episode_counter)

        self.episode_counter += 1
        return True, None

    def get_total_time(self) -> float:
        """Get total elapsed time."""
        return time.time() - self.start_time


def convert_episodes_synced(
    episode_paths: list[pathlib.Path],
    output_dir: pathlib.Path,
    config: pipeline_config.PipelineConfig,
    tolerance_ms: float,
    with_annotations: bool = False,
    causal: bool = True,
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

    Returns:
        Dict with keys: episode_mapping, blacklist, episodes_saved, total_time,
        and optionally annotation_extractor if with_annotations=True.
    """
    features = pipeline_config.build_features(config)

    # Extract robot type from first episode
    robot_type = "panda_bimanual"
    if episode_paths:
        robot_type = extract_robot_type_from_mcap(episode_paths[0])

    converter = SyncedEpisodeConverter(
        output_dir,
        config,
        features,
        tolerance_ms=tolerance_ms,
        robot_type=robot_type,
        causal=causal,
    )

    # Initialize annotation extractor if requested
    annotation_extractor = AnnotationExtractor(fps=config.target_fps) if with_annotations else None

    # Track per-episode results for summary
    episode_results: list[dict] = []
    total_episodes = len(episode_paths)

    for ep_idx, episode_path in enumerate(episode_paths):
        ep_name = episode_path.stem
        print(f"  [{ep_idx + 1}/{total_episodes}] {ep_name} ... ", end="", flush=True)

        try:
            saved, skip_reason = converter.process_episode(episode_path, ep_idx)
            if saved:
                print("done")
                episode_results.append({"path": ep_name, "status": "ok"})
            else:
                print(f"skipped ({skip_reason})")
                episode_results.append({"path": ep_name, "status": "no_frames", "reason": skip_reason})

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
            print(f"error ({type(e).__name__}: {e})")
            episode_results.append({"path": ep_name, "status": "error", "reason": f"{type(e).__name__}: {e}"})
            continue

    # Finalize the dataset - required to close parquet writers and write footer metadata
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
        "total_time": converter.get_total_time(),
        "dataset": converter.dataset,  # Return dataset for push_to_hub
    }

    if annotation_extractor:
        result["annotation_extractor"] = annotation_extractor

    result["episode_results"] = episode_results

    return result


def print_conversion_result(result: dict) -> None:
    """Print conversion summary."""
    ep_results = result.get("episode_results", [])
    ok = [r for r in ep_results if r["status"] == "ok"]
    no_frames = [r for r in ep_results if r["status"] == "no_frames"]
    errors = [r for r in ep_results if r["status"] == "error"]
    blacklisted = result.get("blacklist", [])

    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"  Total episodes processed : {len(ep_results)}")
    print(f"  Successfully converted   : {len(ok)}")
    print(f"  Blacklisted (too short)  : {len(blacklisted)}")
    if no_frames:
        print(f"  Skipped (no frames)      : {len(no_frames)}")
        for r in no_frames:
            print(f"    - {r['path']}: {r.get('reason', 'unknown')}")
    if errors:
        print(f"  Errors                   : {len(errors)}")
        for r in errors:
            print(f"    - {r['path']}: {r['reason']}")
    print(f"  Time elapsed             : {result['total_time']:.1f}s")
    print("=" * 60)


@dataclasses.dataclass
class ScriptArgs:
    """Arguments for synced conversion script."""

    episodes_dir: pathlib.Path = pathlib.Path("./data")
    output: pathlib.Path | None = None  # Auto-generated as /data/lerobot/<task>_<operator> if None
    delete_existing: bool = True  # Delete existing output directory before conversion

    # Sync-specific parameters
    tolerance_ms: float | None = None  # Auto-computed from target_fps if None
    max_episodes: int | None = None  # Stop after N episodes (None = no limit)
    causal: bool = True  # Only use past messages for sync (no future lookahead)

    # Episode filtering
    success_only: bool = True  # Include only successful episodes (based on MCAP metadata)
    excellent_only: bool = True  # Include only episodes with 'excellent' quality
    complete_subtasks_only: bool = False  # Include only episodes where all subtasks have a successful segment
    api_filter: str | None = None  # Platform API filter query string (overrides success/excellent/subtask filters)

    # Annotation extraction (opt-in)
    with_annotations: bool = False  # Extract segment annotations from /annotation topic
    skip_failed_segments: bool = False  # Skip frames from failed segments (requires --with-annotations)

    # HuggingFace Hub upload
    push_to_hub: bool = True  # Push dataset to HuggingFace Hub after conversion
    private: bool = True  # Create private dataset repos on HuggingFace Hub by default
    repo_id: str | None = None  # HuggingFace repo ID; auto-generated as <hub_org>/<output-dir-name> if None
    hub_org: str = "pokeandwiggle"  # HuggingFace organization for auto-generated repo_id


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
    # ------------------------------------------------------------------
    # Suppress noisy C-library stderr (libdav1d, ffmpeg mp4 muxer, SVT-AV1).
    # These write directly to file-descriptor 2, so Python-level log
    # settings can't touch them.  We redirect fd 2 to /dev/null and re-
    # point sys.stderr at the *original* fd so Python logging, warnings,
    # and tracebacks still reach the terminal.
    # ------------------------------------------------------------------
    _orig_stderr_fd = os.dup(2)
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 2)
    os.close(_devnull_fd)
    sys.stderr = os.fdopen(_orig_stderr_fd, "w", closefd=True)

    # Suppress HuggingFace datasets progress bars (Map: 100%|███…)
    try:
        import datasets as _ds
        _ds.disable_progress_bars()
    except Exception:
        pass
    logging.getLogger("datasets").setLevel(logging.WARNING)

    config = draccus.parse(config_class=SyncedConfig)

    validate_input_dir(config.episodes_dir)

    # Auto-generate output dir from episodes_dir: /data/lerobot/<task>_<operator>
    if config.output is None:
        operator_name = config.episodes_dir.name
        task_name = config.episodes_dir.parent.name
        config.output = pathlib.Path("/data/lerobot") / f"{task_name}_{operator_name}"

    # Delete existing output directory if requested
    if config.delete_existing and config.output.exists():
        import shutil
        shutil.rmtree(config.output)

    # Compute tolerance if not specified
    tolerance_ms = config.tolerance_ms
    if tolerance_ms is None:
        tolerance_ms = 1000.0 / config.target_fps

    # Auto-generate repo_id from output directory name (already includes operator)
    if config.push_to_hub and not config.repo_id:
        config.repo_id = f"{config.hub_org}/{config.output.name}"

    episode_paths = get_selected_episodes(
        config.episodes_dir,
        success_only=config.success_only,
        excellent_only=config.excellent_only,
        complete_subtasks_only=config.complete_subtasks_only,
        api_filter=config.api_filter,
    )
    if config.max_episodes is not None:
        episode_paths = episode_paths[: config.max_episodes]

    # Compact header
    print(f"Converting {len(episode_paths)} episodes from {config.episodes_dir}")
    print(f"  -> {config.output}  |  {config.target_fps}Hz  |  tol={tolerance_ms:.0f}ms  |  {config.action_level.value}")
    if config.push_to_hub:
        print(f"  -> HF: {config.repo_id}")

    result = convert_episodes_synced(
        episode_paths,
        config.output,
        config,
        tolerance_ms=tolerance_ms,
        with_annotations=config.with_annotations,
        causal=config.causal,
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
        extractor.save(config.output)

    print_conversion_result(result)

    # Push to HuggingFace Hub if requested
    if config.push_to_hub:
        print(f"Pushing to HuggingFace Hub: {config.repo_id} ...")
        dataset = result["dataset"]
        dataset.repo_id = config.repo_id
        dataset.push_to_hub(
            tags=["LeRobot"],
            private=config.private,
            upload_large_folder=True,
        )
        print(f"Uploaded: https://huggingface.co/datasets/{config.repo_id}")


if __name__ == "__main__":
    main()
