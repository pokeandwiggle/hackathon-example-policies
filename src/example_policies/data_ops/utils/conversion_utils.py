"""Shared utility functions for dataset conversion."""

import json
import logging
import pathlib
import re
import shutil

from mcap.reader import make_reader
from pydantic import ValidationError

from example_policies.utils.constants import (
    BLACKLIST_FILE,
    EPISODE_MAPPING_FILE,
    META_DIR,
    PIPELINE_CONFIG_FILE,
)

logger = logging.getLogger(__name__)


def _extract_episode_metadata(metadata_record) -> dict | None:
    """Extract episode metadata from an MCAP metadata record.

    Supports four formats (priority order):
    1. "pw_episode_info" - new format (1.0+) with JSON data field
    2. "episode_info" - previous format with JSON data containing episode.quality
    3. "episode_rating" - legacy format with flat k/v pairs
    4. "recording_info" - legacy format with flat k/v pairs

    Returns:
        Dict with 'rating' key, or None if not found
    """
    if metadata_record.name == "pw_episode_info":
        schema_version = metadata_record.metadata.get("schema_version")
        try:
            if schema_version in ("1.0", "1.1"):
                from schemas.pw_episode_info_1_0 import RecorderInfo
            elif schema_version == "2.0":
                from schemas.pw_episode_info_2_0 import RecorderInfo
            else:
                raise ValueError(f"Unsupported schema_version: {schema_version}")
        except ImportError:
            # schemas package not available — fall back to JSON parsing
            data_str = metadata_record.metadata.get("data")
            if data_str:
                try:
                    info = json.loads(data_str)
                    episode = info.get("episode", {})
                    rating = episode.get("rating")
                    return {
                        "rating": rating,
                        "embodiment_name": info.get("embodiment", {}).get("name"),
                        "end_effector_left": info.get("embodiment", {}).get("end_effector_left"),
                        "end_effector_right": info.get("embodiment", {}).get("end_effector_right"),
                    }
                except json.JSONDecodeError:
                    pass
            return None
        data_str = metadata_record.metadata.get("data")
        if data_str:
            try:
                info = RecorderInfo.model_validate_json(data_str)
                return {
                    "rating": info.episode.rating.value,
                    "embodiment_name": info.embodiment.name,
                    "end_effector_left": info.embodiment.end_effector_left,
                    "end_effector_right": info.embodiment.end_effector_right,
                }
            except ValidationError as e:
                logger.warning("pw_episode_info failed validation: %s", e)
                return None
    elif metadata_record.name == "episode_info":
        data_str = metadata_record.metadata.get("data")
        if data_str:
            try:
                info = json.loads(data_str)
                quality = info.get("episode", {}).get("quality")
                if quality:
                    return {"rating": quality}
            except json.JSONDecodeError:
                pass
    elif metadata_record.name in ("episode_rating", "recording_info"):
        # Legacy format - derive rating from quality/task_successful
        quality = metadata_record.metadata.get("quality")
        task_successful = (
            metadata_record.metadata.get("task_successful", "").lower() == "true"
        )
        if quality in ("excellent", "ok", "failed"):
            return {"rating": quality}
        if task_successful:
            return {"rating": quality if quality in ("excellent", "ok") else "ok"}
        return {"rating": "failed"}
    return None


# Annotation file constants
LEROBOT_ANNOTATIONS_FILE = "lerobot_annotations.json"
SUBTASKS_FILE = "subtasks.parquet"

# Default robot type if mount cannot be determined
DEFAULT_ROBOT_TYPE = "dual_panda"


def extract_robot_type_from_mcap(episode_path: pathlib.Path) -> str:
    """Extract robot type from MCAP metadata.

    Checks two sources in priority order:
    1. pw_episode_info metadata → embodiment.name (newer format)
    2. robot_description metadata → URDF xacro source comment (legacy format)

    Args:
        episode_path: Path to the MCAP episode file

    Returns:
        Robot type string (e.g., 'dual_panda_wall', 'dual_fr3_pedestal')
    """
    try:
        with open(episode_path, "rb") as f:
            reader = make_reader(f)
            for metadata_record in reader.iter_metadata():
                # Prefer structured embodiment info from pw_episode_info
                meta = _extract_episode_metadata(metadata_record)
                if meta is not None and meta.get("embodiment_name"):
                    return meta["embodiment_name"]

                # Fall back to parsing URDF xacro comment
                if metadata_record.name == "robot_description":
                    urdf = metadata_record.metadata.get("urdf", "")
                    match = re.search(
                        r"(dual_(?:panda|fr3)_\w+)\.urdf\.xacro", urdf
                    )
                    if match:
                        return match.group(1)
    except (OSError, ValueError) as e:
        print(f"  Warning: Failed to extract robot type from {episode_path}: {e}")

    return DEFAULT_ROBOT_TYPE


def extract_embodiment_metadata(episode_path: pathlib.Path) -> dict | None:
    """Extract embodiment metadata from an MCAP episode file.

    Returns:
        Dict with 'embodiment_name', 'end_effector_left', 'end_effector_right',
        or None if no embodiment info is available.
    """
    try:
        with open(episode_path, "rb") as f:
            reader = make_reader(f)
            for metadata_record in reader.iter_metadata():
                meta = _extract_episode_metadata(metadata_record)
                if meta is not None and "embodiment_name" in meta:
                    return {
                        "embodiment_name": meta["embodiment_name"],
                        "end_effector_left": meta.get("end_effector_left"),
                        "end_effector_right": meta.get("end_effector_right"),
                    }
    except (OSError, ValueError, KeyError) as e:
        logger.warning("Error reading embodiment from %s: %s", episode_path, e)
    return None


class AnnotationExtractor:
    """Extracts and converts segment annotations from MCAP files.

    This class handles the full pipeline of:
    1. Extracting segment annotations from MCAP metadata (segments record)
    2. Converting to LeRobot-Annotate compatible format
    3. Saving lerobot_annotations.json and subtasks.parquet

    Usage:
        extractor = AnnotationExtractor(fps=10.0)

        # During episode processing loop:
        extractor.extract_from_mcap(episode_path, episode_idx=0)
        extractor.extract_from_mcap(episode_path, episode_idx=1)

        # After all episodes:
        extractor.save(output_dir)
    """

    def __init__(self, fps: float):
        """Initialize the annotation extractor.

        Args:
            fps: Target FPS of the dataset (for potential frame index conversion)
        """
        self.fps = fps
        self.episode_annotations: dict[int, list[dict]] = {}

    def _parse_iso_timestamp(self, start_time_str: str) -> float:
        """Parse ISO format timestamp to Unix timestamp in seconds.

        Handles nanosecond precision by truncating to microseconds.
        Supports both '+00:00' timezone suffix and 'Z' suffix.

        Args:
            start_time_str: ISO format timestamp (e.g., '2026-02-05T16:10:21.582082182+00:00'
                            or '2026-02-11T11:20:46.642772353Z')

        Returns:
            Unix timestamp in seconds
        """
        from datetime import datetime

        # Handle 'Z' suffix (UTC) - convert to +00:00 format
        if start_time_str.endswith('Z'):
            # Format: 2026-02-11T11:20:46.642772353Z
            # Truncate nanoseconds to microseconds
            start_time_truncated = start_time_str[:26] + '+00:00'
        else:
            # Format: 2026-02-05T16:10:21.582082182+00:00
            # Truncate nanoseconds to microseconds for datetime parsing
            start_time_truncated = start_time_str[:26] + start_time_str[-6:]

        start_dt = datetime.fromisoformat(start_time_truncated)
        return start_dt.timestamp()

    def extract_from_mcap(
        self,
        episode_path: pathlib.Path,
        episode_idx: int,
        raw_frame_kept: list[bool] | None = None,
        sync_abs_start_s: float | None = None,
    ) -> list[dict]:
        """Extract segment annotations from MCAP metadata.

        Supports two formats:
        - Old format: 'segments' metadata with segment_map, 'episode_rating' for start time
        - New format: 'pw_episode_info' with segments list and episode.start_time

        If raw_frame_kept and sync_abs_start_s are provided, annotations are
        remapped to account for pauses that were skipped during conversion. This
        ensures annotation timestamps match the actual dataset frame indices.

        Args:
            episode_path: Path to the MCAP episode file
            episode_idx: Episode index in the output dataset
            raw_frame_kept: List of booleans, one per synced frame, indicating
                whether the frame was kept (True) or skipped as a pause (False).
            sync_abs_start_s: Absolute Unix timestamp of the first synced frame.
                Used together with the MCAP episode start time to compute the
                offset between raw annotation time and synced frame indices.

        Returns:
            List of segment dicts with start_seconds, end_seconds, rating, subtask_id
        """
        annotations = []
        episode_start_time = None
        segment_map = None
        pw_episode_info = None

        try:
            with open(episode_path, "rb") as f:
                reader = make_reader(f)
                for metadata_record in reader.iter_metadata():
                    # New format: pw_episode_info
                    if metadata_record.name == "pw_episode_info":
                        data_str = metadata_record.metadata.get("data")
                        if data_str:
                            pw_episode_info = json.loads(data_str)
                    # Old format: episode_rating for start time
                    elif metadata_record.name == "episode_rating":
                        start_time_str = metadata_record.metadata.get("start_time")
                        if start_time_str:
                            episode_start_time = self._parse_iso_timestamp(start_time_str)
                    # Old format: segments metadata
                    elif metadata_record.name == "segments":
                        segment_map_str = metadata_record.metadata.get("segment_map")
                        if segment_map_str:
                            segment_map = json.loads(segment_map_str)

            # Try new format first (pw_episode_info)
            if pw_episode_info is not None:
                annotations = self._extract_from_pw_episode_info(pw_episode_info, episode_path)
                # Get episode start time for remapping
                ep_start_str = pw_episode_info.get("episode", {}).get("start_time")
                if ep_start_str:
                    episode_start_time = self._parse_iso_timestamp(ep_start_str)
            # Fall back to old format
            elif episode_start_time is not None and segment_map is not None:
                annotations = self._extract_from_legacy_format(
                    segment_map, episode_start_time, episode_path
                )
            elif episode_start_time is None and segment_map is not None:
                print(f"  Warning: No episode_rating metadata found in {episode_path}")
            elif segment_map is None and episode_start_time is not None:
                print(f"  Warning: No segments metadata found in {episode_path}")
            # else: neither format found, annotations stays empty

        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"  Warning: Failed to read annotations from {episode_path}: {e}")

        # Remap annotations to account for pause removal and sync offset
        if annotations and raw_frame_kept is not None and sync_abs_start_s is not None:
            if episode_start_time is not None:
                annotations = self._remap_annotations_for_pauses(
                    annotations,
                    raw_frame_kept,
                    sync_abs_start_s=sync_abs_start_s,
                    mcap_episode_start_s=episode_start_time,
                )
            else:
                print("  Warning: Cannot remap annotations - no episode start time found")
                # Without remapping, timestamps are relative to MCAP start and would
                # be incorrect in the dataset. Discard them to avoid silent errors.
                annotations = []
        elif annotations and (raw_frame_kept is not None or sync_abs_start_s is not None):
            # Only one of the two remapping parameters was provided — this is
            # likely a caller error. Discard annotations to be safe.
            missing = "sync_abs_start_s" if sync_abs_start_s is None else "raw_frame_kept"
            print(
                f"  Warning: Cannot remap annotations - {missing} not provided. "
                "Discarding un-remapped annotations."
            )
            annotations = []

        # Store annotations for this episode
        if annotations:
            self.episode_annotations[episode_idx] = annotations
            print(f"  Extracted {len(annotations)} segment annotations")

        return annotations

    def _extract_from_pw_episode_info(
        self, pw_info: dict, episode_path: pathlib.Path
    ) -> list[dict]:
        """Extract annotations from new pw_episode_info format.

        Args:
            pw_info: Parsed pw_episode_info data
            episode_path: Path for error messages

        Returns:
            List of annotation dicts
        """
        annotations = []
        episode = pw_info.get("episode", {})
        segments = pw_info.get("segments", [])

        if not segments:
            return annotations

        # Get episode start time
        start_time_str = episode.get("start_time")
        if not start_time_str:
            print(f"  Warning: No start_time in pw_episode_info for {episode_path}")
            return annotations

        episode_start_time = self._parse_iso_timestamp(start_time_str)

        # Process segments - they have start_time and end_time as ISO strings
        for idx, seg in enumerate(segments):
            seg_start_str = seg.get("start_time")
            seg_end_str = seg.get("end_time")

            if not seg_start_str or not seg_end_str:
                continue

            seg_start = self._parse_iso_timestamp(seg_start_str)
            seg_end = self._parse_iso_timestamp(seg_end_str)

            start_seconds = round(seg_start - episode_start_time, 3)
            end_seconds = round(seg_end - episode_start_time, 3)

            annotations.append(
                {
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "rating": seg.get("rating", ""),
                    "subtask_id": seg.get("step_name", ""),  # step_name in new format
                    "annotation_id": idx,
                }
            )

        return annotations

    def _extract_from_legacy_format(
        self, segment_map: dict, episode_start_time: float, episode_path: pathlib.Path
    ) -> list[dict]:
        """Extract annotations from legacy segments format.

        Args:
            segment_map: Parsed segment_map data
            episode_start_time: Episode start time as Unix timestamp
            episode_path: Path for error messages

        Returns:
            List of annotation dicts
        """
        annotations = []
        segments = segment_map.get("segments", {})

        if not segments:
            print(f"  Warning: No segments in segment_map for {episode_path}")
            return annotations

        # Sort segments by annotation_id (stored as string keys)
        sorted_segment_ids = sorted(segments.keys(), key=int)

        # Build annotations: each segment has an end timestamp
        # Use seconds with 3 decimal places (millisecond precision) per lerobot-annotate format
        prev_seconds = 0.0  # Start from beginning of episode

        for seg_id in sorted_segment_ids:
            seg = segments[seg_id]
            timestamp_s = seg["timestamp_ms"] / 1000
            end_seconds = round(timestamp_s - episode_start_time, 3)

            annotations.append(
                {
                    "start_seconds": prev_seconds,
                    "end_seconds": end_seconds,
                    "rating": seg["rating"],
                    "subtask_id": seg["subtask_id"],
                    "annotation_id": int(seg_id),
                }
            )
            prev_seconds = end_seconds

        return annotations

    def _remap_annotations_for_pauses(
        self,
        annotations: list[dict],
        raw_frame_kept: list[bool],
        sync_abs_start_s: float,
        mcap_episode_start_s: float,
    ) -> list[dict]:
        """Remap annotation timestamps to account for sync offset and pause removal.

        Raw annotations use timestamps relative to the MCAP episode metadata start.
        The actual dataset frames start at a later time (sync offset) and skip
        paused frames. This method remaps annotation boundaries so they reference
        the correct dataset time.

        The mapping works as follows:
        1. Each raw synced frame i has raw time = sync_offset + i / fps
           (where sync_offset = sync_abs_start_s - mcap_episode_start_s)
        2. Kept frames are numbered sequentially as dataset frames 0, 1, 2, ...
        3. For an annotation boundary at raw time t, find the raw frame index,
           then look up how many frames were kept up to that point to get dataset time.

        Uses linear interpolation within frames for sub-frame precision.

        Args:
            annotations: List of annotation dicts with start_seconds / end_seconds
            raw_frame_kept: Boolean per synced frame, True if kept
            sync_abs_start_s: Absolute timestamp of the first synced frame
            mcap_episode_start_s: Absolute timestamp of MCAP episode metadata start

        Returns:
            Remapped annotations list (modified in place and returned)
        """
        import numpy as np

        if not raw_frame_kept:
            return annotations

        fps = self.fps
        n_raw_frames = len(raw_frame_kept)

        # Offset from MCAP episode start to first synced frame (in seconds)
        sync_offset = sync_abs_start_s - mcap_episode_start_s

        if sync_offset < 0:
            logger.warning(
                "Negative sync_offset (%.3fs): synced frames start before MCAP "
                "episode metadata start. Early annotations may be clamped to t=0.",
                sync_offset,
            )

        # Build prefix sum of kept frames (exclusive: prefix_kept[i] = kept frames
        # strictly before index i). Length is n_raw_frames + 1 so that
        # prefix_kept[i] and prefix_kept[i+1] are always safe for i < n_raw_frames.
        kept_arr = np.array(raw_frame_kept, dtype=np.int32)
        prefix_kept = np.zeros(n_raw_frames + 1, dtype=np.int32)
        prefix_kept[1:] = np.cumsum(kept_arr)
        total_kept = int(prefix_kept[-1])

        def raw_time_to_dataset_time(raw_time_s: float) -> float:
            """Convert raw annotation time to dataset time accounting for pauses.

            For a raw time t:
            1. Compute the fractional raw frame index: fi = (t - sync_offset) * fps
            2. Clamp to valid range [0, n_raw_frames]
            3. Interpolate the exclusive prefix-sum of kept frames at fi
            4. Dataset time = interpolated_kept_count / fps

            Using an exclusive prefix sum ensures the first kept frame maps to
            dataset time 0 (not 1/fps).
            """
            # Raw frame index (fractional)
            fi = (raw_time_s - sync_offset) * fps

            # Clamp to valid range
            if fi <= 0:
                return 0.0
            if fi >= n_raw_frames:
                return total_kept / fps

            # Integer frame index and fraction
            i = int(fi)
            frac = fi - i

            # Prefix kept count at frames i and i+1
            ck_i = prefix_kept[i]
            ck_next = prefix_kept[i + 1]

            # Linear interpolation
            interpolated = ck_i + frac * (ck_next - ck_i)
            return round(interpolated / fps, 3)

        # Remap each annotation
        for ann in annotations:
            old_start = ann["start_seconds"]
            old_end = ann["end_seconds"]
            ann["start_seconds"] = raw_time_to_dataset_time(old_start)
            ann["end_seconds"] = raw_time_to_dataset_time(old_end)

        # Drop zero-duration segments (e.g., segments entirely outside the sync window)
        original_count = len(annotations)
        annotations = [a for a in annotations if a["end_seconds"] > a["start_seconds"]]
        dropped = original_count - len(annotations)

        # Log summary
        if annotations:
            dataset_dur = total_kept / fps
            msg = (
                f"  Remapped annotations: {total_kept} kept frames "
                f"({n_raw_frames - total_kept} pauses removed), "
                f"dataset duration {dataset_dur:.3f}s"
            )
            if dropped:
                msg += f", dropped {dropped} segment(s) outside sync window"
            print(msg)

        return annotations

    def to_lerobot_format(self) -> dict:
        """Convert collected annotations to lerobot_annotations.json format.

        Returns:
            Dict in lerobot_annotations.json format compatible with LeRobot-Annotate
        """
        result = {"version": 1, "episodes": {}}

        for ep_idx, segments in self.episode_annotations.items():
            subtasks = []
            for seg in segments:
                rating = seg["rating"]
                subtask_id = seg.get("subtask_id", "")

                if rating == "failed":
                    # For failed segments, use subtask_id with _failed suffix
                    label = f"{subtask_id}_failed" if subtask_id else "failed"
                else:
                    # Use the actual subtask_id with rating suffix
                    label = f"{subtask_id}_{rating}" if subtask_id else f"unknown_{rating}"

                subtasks.append(
                    {
                        "start": seg["start_seconds"],
                        "end": seg["end_seconds"],
                        "label": label,
                    }
                )

            result["episodes"][str(ep_idx)] = {
                "subtasks": subtasks,
                "high_levels": [],  # Can be populated from episode metadata later
            }

        return result

    def save(self, output_dir: pathlib.Path) -> None:
        """Save annotations in LeRobot-Annotate compatible format.

        Writes:
        - meta/lerobot_annotations.json
        - meta/subtasks.parquet

        Args:
            output_dir: Output dataset directory
        """
        if not self.episode_annotations:
            print("  No annotations to save")
            return

        annotations_data = self.to_lerobot_format()

        # Save JSON annotations
        meta_dir = output_dir / META_DIR
        meta_dir.mkdir(parents=True, exist_ok=True)

        annotations_path = meta_dir / LEROBOT_ANNOTATIONS_FILE
        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(annotations_data, f, indent=2)
        print(f"  Saved annotations to {annotations_path}")

        # Save subtasks parquet
        self._build_subtasks_parquet(annotations_data, output_dir)

    def _build_subtasks_parquet(
        self, annotations_data: dict, output_dir: pathlib.Path
    ) -> None:
        """Build and save subtasks.parquet from annotations."""
        try:
            import pandas as pd
        except ImportError:
            print("  Warning: pandas not available, skipping subtasks.parquet")
            return

        # Collect all unique labels
        labels = set()
        for ep_data in annotations_data.get("episodes", {}).values():
            for subtask in ep_data.get("subtasks", []):
                if subtask.get("label"):
                    labels.add(subtask["label"])

        if not labels:
            print("  No subtask labels found, skipping subtasks.parquet")
            return

        # Build dataframe with sorted labels
        sorted_labels = sorted(labels)
        data = [
            {"subtask": label, "subtask_index": idx}
            for idx, label in enumerate(sorted_labels)
        ]
        df = pd.DataFrame(data)
        df = df.set_index("subtask")

        # Save to parquet
        meta_dir = output_dir / META_DIR
        subtasks_path = meta_dir / SUBTASKS_FILE
        df.to_parquet(subtasks_path, engine="pyarrow", compression="snappy")

        print(f"  Saved {len(sorted_labels)} subtasks to {subtasks_path}")

    @property
    def total_segments(self) -> int:
        """Total number of segments across all episodes."""
        return sum(len(segs) for segs in self.episode_annotations.values())

    @property
    def episodes_with_annotations(self) -> int:
        """Number of episodes that have annotations."""
        return len(self.episode_annotations)


def _extract_episode_rating(episode_path: pathlib.Path) -> dict | None:
    """Extract episode rating from an MCAP file.

    Iterates over metadata records and delegates to _extract_episode_metadata
    to parse any supported format.

    Args:
        episode_path: Path to the MCAP episode file

    Returns:
        Dict with 'rating' key, or None if not found
    """
    try:
        with open(episode_path, "rb") as f:
            reader = make_reader(f)
            for metadata_record in reader.iter_metadata():
                result = _extract_episode_metadata(metadata_record)
                if result is not None:
                    return result
    except (OSError, ValueError) as e:
        print(f"  Warning: Failed to read metadata from {episode_path}: {e}")

    return None


def get_sorted_episodes(episode_dir: pathlib.Path) -> list[pathlib.Path]:
    """Get episode paths sorted by creation time (oldest first).

    Args:
        episode_dir: Directory containing .mcap episode files

    Returns:
        List of episode paths sorted by creation date
    """
    episode_paths = list(episode_dir.rglob("*.mcap"))
    episode_paths.sort(key=lambda p: p.stat().st_ctime)
    return episode_paths


def check_subtask_completeness(episode_path: pathlib.Path) -> tuple[bool, dict]:
    """Check if all defined subtasks have at least one successful segment.

    Supports two formats:
    - Old format: 'segments' metadata with segment_map containing subtasks dict and segments dict
    - New format: 'pw_episode_info' with episode.steps list and segments list

    Args:
        episode_path: Path to the MCAP episode file

    Returns:
        Tuple of (is_complete, details) where:
        - is_complete: True if all subtasks have at least one successful segment
        - details: Dict with 'defined_subtasks', 'completed_subtasks', 'missing_subtasks'
    """
    details = {
        "defined_subtasks": set(),
        "completed_subtasks": set(),
        "missing_subtasks": set(),
    }

    successful_ratings = {"excellent", "good", "ok"}

    try:
        with open(episode_path, "rb") as f:
            reader = make_reader(f)
            for metadata_record in reader.iter_metadata():
                # New format: pw_episode_info
                if metadata_record.name == "pw_episode_info":
                    schema_version = metadata_record.metadata.get("schema_version")
                    if schema_version == "1.0":
                        from schemas.pw_episode_info_1_0 import RecorderInfo
                    else:
                        raise ValueError(f"Unsupported schema_version: {schema_version}")
                    data_str = metadata_record.metadata.get("data")
                    if not data_str:
                        continue

                    try:
                        info = RecorderInfo.model_validate_json(data_str)
                    except ValidationError as e:
                        logger.warning("pw_episode_info failed validation: %s", e)
                        continue

                    # Get defined steps (subtasks) from episode.steps
                    if info.episode.steps:
                        details["defined_subtasks"] = {step.name for step in info.episode.steps}

                    # Find which steps have successful segments
                    for seg in info.segments or []:
                        if seg.root.step_name and seg.root.rating.value in successful_ratings:
                            details["completed_subtasks"].add(seg.root.step_name)

                    # Calculate missing subtasks
                    details["missing_subtasks"] = (
                        details["defined_subtasks"] - details["completed_subtasks"]
                    )
                    break

                # Old format: segments metadata
                elif metadata_record.name == "segments":
                    segment_map_str = metadata_record.metadata.get("segment_map")
                    if not segment_map_str:
                        continue

                    segment_map = json.loads(segment_map_str)

                    # Get all defined subtasks
                    subtasks = segment_map.get("subtasks", {})
                    details["defined_subtasks"] = set(subtasks.keys())

                    # Find which subtasks have successful segments
                    segments = segment_map.get("segments", {})

                    for seg in segments.values():
                        subtask_id = seg.get("subtask_id")
                        rating = seg.get("rating", "")
                        if subtask_id and rating in successful_ratings:
                            details["completed_subtasks"].add(subtask_id)

                    # Calculate missing subtasks
                    details["missing_subtasks"] = (
                        details["defined_subtasks"] - details["completed_subtasks"]
                    )
                    break

    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"  Warning: Failed to check subtask completeness for {episode_path}: {e}")
        return False, details

    is_complete = len(details["missing_subtasks"]) == 0 and len(details["defined_subtasks"]) > 0
    return is_complete, details


def get_selected_episodes(
    episode_dir: pathlib.Path,
    success_only: bool = True,
    excellent_only: bool = True,
    complete_subtasks_only: bool = False,
    api_filter: str | None = None,
) -> list[pathlib.Path]:
    """Get episode paths sorted by creation time (oldest first), that fulfil success criteria in the MCAP file.

    When api_filter is provided, filtering is delegated to the platform API:
    only local files whose filename appears in the API response are returned.
    The success_only / excellent_only / complete_subtasks_only flags are ignored
    in that case (the API already applied the equivalent filters).

    Args:
        episode_dir: Directory containing .mcap episode files
        success_only: If True, include only successful episodes
        excellent_only: If True, include only episodes with "excellent" quality
        complete_subtasks_only: If True, include only episodes where all defined
            subtasks have at least one successful segment
        api_filter: URL query string from the platform UI (e.g.
            "task=pick_red_brick&rating=excellent&hide_ignored=true").
            When set, other filter flags are ignored.

    Returns:
        List of episode paths sorted by creation date that fulfil success criteria in the MCAP file
    """

    episode_paths = list(episode_dir.rglob("*.mcap"))
    episode_paths.sort(key=lambda p: p.stat().st_ctime)

    if api_filter is not None:
        from example_policies.data_ops.cloud.platform_api_client import fetch_episode_paths

        api_episodes = fetch_episode_paths(api_filter)
        api_filenames = {ep["object_storage_path"].rsplit("/", 1)[-1] for ep in api_episodes}
        return [p for p in episode_paths if p.name in api_filenames]

    filtered_episode_paths = []

    if success_only is True:
        for ep_path in episode_paths:
            metadata = _extract_episode_rating(ep_path)
            if metadata is None:
                continue

            rating = metadata.get("rating")
            if excellent_only is True and rating == "excellent":
                filtered_episode_paths.append(ep_path)
            elif excellent_only is False and rating in ["excellent", "good", "ok"]:
                filtered_episode_paths.append(ep_path)
    else:
        filtered_episode_paths = episode_paths

    # Apply subtask completeness filter if requested
    if complete_subtasks_only:
        complete_paths = []
        for ep_path in filtered_episode_paths:
            is_complete, details = check_subtask_completeness(ep_path)
            if is_complete:
                complete_paths.append(ep_path)
            else:
                missing = details["missing_subtasks"]
                if missing:
                    print(f"  ⚠️  Skipping {ep_path.name}: missing subtasks {sorted(missing)}")
                else:
                    print(f"  ⚠️  Skipping {ep_path.name}: no subtask definitions found")
        filtered_episode_paths = complete_paths

    return filtered_episode_paths


def _parse_episode_filename(
    file_path: pathlib.Path,
) -> tuple[str | None, str | None]:
    """Parse episode file path to extract operator and rating.

    Supports two filename formats:
    - New format: {timestamp}--{task}--{operator}--{rating}.mcap
      Operator and rating extracted from filename.
    - Old format: {parent_dir}/{task}/{operator}/{timestamp}_{success/failure}_{rating}.mcap
      Operator extracted from parent directory, rating from filename.

    Args:
        file_path: Full path to the episode file

    Returns:
        Tuple of (operator, rating), with None for fields that couldn't be parsed
    """
    filename_stem = file_path.stem
    if "--" in filename_stem:
        parts = filename_stem.split("--")
        if len(parts) == 4:
            _timestamp, _task, operator, rating = parts
            return operator, rating
    else:
        parts = filename_stem.split("_")
        if len(parts) >= 4:
            rating = parts[-1]
            operator = file_path.parent.name
            return operator, rating
    return None, None


def filter_episode_paths(
    episode_paths: list[pathlib.Path],
    operator_blacklist: list[str],
    rating_whitelist: list[str],
) -> list[pathlib.Path]:
    """Filter episode paths based on operator blacklist and rating whitelist.

    Supports two filename formats:
    - New format: {timestamp}--{task}--{operator}--{rating}.mcap
    - Old format: {parent_dir}/{task}/{operator}/{timestamp}_{success}_{rating}.mcap

    Args:
        episode_paths: List of episode file paths
        operator_blacklist: List of operator names to ignore
        rating_whitelist: List of rating values to include

    Returns:
        List of filtered episode paths
    """
    if len(operator_blacklist) == 0 and len(rating_whitelist) == 0:
        return episode_paths

    filtered_paths = []
    for ep_path in episode_paths:
        operator, rating = _parse_episode_filename(ep_path)

        if rating is None:
            continue

        operator_ok = (
            not operator_blacklist
            or operator is None
            or operator not in operator_blacklist
        )
        rating_ok = not rating_whitelist or rating in rating_whitelist

        if operator_ok and rating_ok:
            filtered_paths.append(ep_path)

    return filtered_paths


def validate_input_dir(path: pathlib.Path) -> None:
    """Validate that input directory exists.

    Args:
        path: Directory path to validate

    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    if not path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {path}")


def validate_output_dir(path: pathlib.Path, force: bool = False) -> None:
    """Validate output directory, handle --force flag.

    Args:
        path: Output directory path
        force: If True, delete existing directory

    Raises:
        FileExistsError: If directory exists and force=False
    """
    if path.exists():
        if force:
            print(
                f"Warning: Output directory {path} already exists. Overwriting due to --force."
            )
            shutil.rmtree(path)
        else:
            raise FileExistsError(
                f"Output directory already exists: {path}. Use --force to overwrite."
            )


def resolve_blacklist_path(blacklist_arg: pathlib.Path | None) -> pathlib.Path | None:
    """Resolve blacklist path with fallback logic.

    Handles both file paths and directory paths. If directory is provided,
    looks for blacklist.json in the directory or meta subdirectory.

    Args:
        blacklist_arg: Blacklist file or directory path

    Returns:
        Resolved blacklist file path, or None if not found
    """
    if blacklist_arg is None:
        return None

    if not blacklist_arg.exists():
        print(f"Warning: Specified blacklist file does not exist: {blacklist_arg}")
        return None

    if blacklist_arg.is_file():
        return blacklist_arg

    # Directory case - try to find blacklist.json
    blacklist_path = blacklist_arg / BLACKLIST_FILE
    if blacklist_path.exists():
        return blacklist_path

    blacklist_path = blacklist_arg / META_DIR / BLACKLIST_FILE
    if blacklist_path.exists():
        return blacklist_path

    print(f"Warning: No blacklist file found in directory: {blacklist_arg}")
    return None


def build_output_path(
    episodes_dir: pathlib.Path,
    output_parent: pathlib.Path | None,
    name_prefix: str | None,
    mode_suffix: str,
) -> pathlib.Path:
    """Construct output directory path.

    Args:
        episodes_dir: Input episodes directory
        output_parent: Parent directory for output (if None, uses episodes_dir/../../lerobot)
        name_prefix: Prefix for output directory name (if None, uses episodes_dir.name)
        mode_suffix: Mode string to append to directory name

    Returns:
        Constructed output directory path
    """
    if output_parent is None:
        output_parent = episodes_dir.parent.parent / "lerobot"

    if name_prefix is None:
        name_prefix = episodes_dir.name

    return output_parent / f"{name_prefix}_{mode_suffix}"


def copy_blacklist(
    output_dir: pathlib.Path, blacklist_path: pathlib.Path | None
) -> None:
    """Copy blacklist file to output directory.

    Args:
        output_dir: Output dataset directory
        blacklist_path: Source blacklist file path (can be None)
    """
    if blacklist_path is None:
        return

    if not blacklist_path.exists():
        return

    if blacklist_path.is_file():
        blacklist_dst = output_dir / META_DIR / BLACKLIST_FILE
        shutil.copy(blacklist_path, blacklist_dst)
        print(f"Copied blacklist to {blacklist_dst}")


def save_metadata(
    output_dir: pathlib.Path,
    episode_mapping: dict[int, str],
    blacklist: list[int],
    config,
) -> None:
    """Save all metadata files.

    Args:
        output_dir: Output dataset directory
        episode_mapping: Mapping of episode indices to file paths
        blacklist: List of blacklisted episode indices
        config: Pipeline configuration with to_dict() method
    """
    meta_dir = output_dir / META_DIR

    with open(meta_dir / EPISODE_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(episode_mapping, f, indent=2)

    with open(meta_dir / PIPELINE_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    with open(meta_dir / BLACKLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(blacklist, f, indent=2)

    print(f"\nMetadata saved to {meta_dir}")
