#!/usr/bin/env python3
"""Generate a multi-page dataset quality PDF report.

Standalone script that produces the exact same output as notebook
``08_dataset_frequency_analysis.ipynb``.

Usage
-----
    python src/example_policies/data_ops/review/generate_quality_report.py /data/raw/build_lego_duplo_flower/MythicToad --tolerance-ms 100.0 --no-excellent-filter

Options
-------
    --output-dir DIR      Folder for the PDF (default: notebooks/outputs)
    --target-fps N        Target recording FPS (default: 30)
    --tolerance-ms MS     Sync tolerance in ms (default: auto = 1000/target_fps)
    --max-episodes N      Limit number of episodes (default: all)
    --no-success-filter   Include non-success episodes
    --no-excellent-filter Include non-excellent episodes
"""

from __future__ import annotations

import argparse
import datetime
import gc
import logging
import pathlib
from collections import defaultdict

import av
import av.logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore

from example_policies.data_ops.config.rosbag_topics import RosTopicEnum
from example_policies.data_ops.pipeline.timestamp_utils import extract_sensor_timestamp
from example_policies.data_ops.utils.conversion_utils import get_selected_episodes
from example_policies.default_paths import PLOTS_DIR

# ── Suppress noisy AV1 decoder logs ──────────────────────────────────
logging.getLogger("libav").setLevel(logging.CRITICAL)
av.logging.set_level(av.logging.PANIC)

# ── Use non-interactive backend (no GUI needed) ──────────────────────
matplotlib.use("Agg")

# ═══════════════════════════════════════════════════════════════════════
# Style & palette (seaborn-inspired)
# ═══════════════════════════════════════════════════════════════════════
plt.rcParams.update(
    {
        # Figure
        "figure.facecolor": "#f0f0f0",
        "figure.dpi": 130,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        # Axes
        "axes.facecolor": "#eaeaf2",
        "axes.edgecolor": "white",
        "axes.linewidth": 0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "medium",
        "axes.titlepad": 10,
        # Grid
        "grid.color": "white",
        "grid.linewidth": 1.2,
        "grid.linestyle": "-",
        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        # Ticks
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "font.size": 11,
        "text.color": "#333333",
        "axes.labelcolor": "#333333",
        # Legend
        "legend.frameon": False,
        "legend.fontsize": 9,
        # Savefig
        "savefig.facecolor": "#f0f0f0",
        "savefig.edgecolor": "#f0f0f0",
    }
)

PALETTE = [
    "#4878d0",
    "#ee854a",
    "#6acc64",
    "#d65f5f",
    "#956cb4",
    "#8c613c",
    "#dc7ec0",
    "#797979",
    "#d5bb67",
    "#82c6e2",
]

# Slightly darker colours for spike-timeline (5a) — one shade richer than PALETTE
_CLR_OK = "#3a65c2"      # deeper blue (PALETTE[0] = #4878d0)
_CLR_VIOL = "#c44545"    # deeper red  (PALETTE[3] = #d65f5f)
_CLR_THRESH = "#c44545"  # threshold line (same deeper red)

# ═══════════════════════════════════════════════════════════════════════
# Topics
# ═══════════════════════════════════════════════════════════════════════
TOPICS = [
    ("Joint State", [RosTopicEnum.ACTUAL_JOINT_STATE.value]),
    ("RGB L", [RosTopicEnum.RGB_LEFT_IMAGE.value]),
    ("RGB R", [RosTopicEnum.RGB_RIGHT_IMAGE.value]),
    ("RGB S", [RosTopicEnum.RGB_STATIC_IMAGE.value]),
    ("Depth L", ["/cam_left/aligned_depth_to_color/image_compressed"]),
    ("Depth R", ["/cam_right/aligned_depth_to_color/image_compressed"]),
    ("TCP L", [RosTopicEnum.ACTUAL_TCP_LEFT.value, "/left/franka_robot_state_broadcaster/current_pose", "/panda_left/tcp"]),
    ("TCP R", [RosTopicEnum.ACTUAL_TCP_RIGHT.value, "/right/franka_robot_state_broadcaster/current_pose", "/panda_right/tcp"]),
    ("Cmd Joint L", ["/joint_target_left"]),
    ("Cmd Joint R", ["/joint_target_right"]),
    ("Cmd TCP L", ["/cartesian_target_left", "/desired_pose_twist_left"]),
    ("Cmd TCP R", ["/cartesian_target_right", "/desired_pose_twist_right"]),
    ("Cmd Gripper L", [RosTopicEnum.DES_GRIPPER_LEFT.value]),
    ("Cmd Gripper R", [RosTopicEnum.DES_GRIPPER_RIGHT.value]),
]

# ═══════════════════════════════════════════════════════════════════════
# AV1 keyframe & episode analysis
# ═══════════════════════════════════════════════════════════════════════
_COMPRESSED_VIDEO_SCHEMA = "foxglove_msgs/msg/CompressedVideo"
_typestore = get_typestore(Stores.ROS2_HUMBLE)


def _is_av1_keyframe(video_data: bytes) -> bool:
    """Check if AV1 frame data is a keyframe by parsing OBU header."""
    if len(video_data) < 2:
        return False
    obu_type = (video_data[0] >> 3) & 0x0F
    return obu_type == 1


def analyse_episode(mcap_path, topic_names_flat):
    """Return ({topic: sorted timestamps}, {topic: source_counts},
              {topic: schema_name}, {topic: drift_list},
              {topic: keyframe_timestamps}).

    Reads ALL topics from the MCAP (topic_names_flat is kept for API
    compatibility but is no longer used as a filter).
    """
    per_topic: dict[str, list[float]] = defaultdict(list)
    schema_cache: dict[str, str] = {}
    ts_source: dict[str, dict[str, int]] = defaultdict(lambda: {"sensor": 0, "log": 0})
    ts_drift: dict[str, list[float]] = defaultdict(list)
    keyframe_ts: dict[str, list[float]] = defaultdict(list)

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            t = channel.topic
            if t not in schema_cache and schema is not None:
                schema_cache[t] = schema.name

            log_ts = message.log_time * 1e-9

            ts = None
            if t in schema_cache:
                try:
                    ts = extract_sensor_timestamp(message.data, schema_cache[t])
                except Exception:
                    pass
            if ts is not None:
                ts_source[t]["sensor"] += 1
                ts_drift[t].append(abs(ts - log_ts))
            else:
                ts = log_ts
                ts_source[t]["log"] += 1

            if schema_cache.get(t) == _COMPRESSED_VIDEO_SCHEMA:
                try:
                    video_msg = _typestore.deserialize_cdr(
                        message.data, _COMPRESSED_VIDEO_SCHEMA
                    )
                    if _is_av1_keyframe(bytes(video_msg.data)):
                        keyframe_ts[t].append(ts)
                except Exception:
                    pass

            per_topic[t].append(ts)

    for v in per_topic.values():
        v.sort()
    return per_topic, dict(ts_source), dict(schema_cache), dict(ts_drift), dict(keyframe_ts)


def extract_dataset_version(mcap_path: pathlib.Path) -> str | None:
    """Extract the schema_version from an MCAP file's pw_episode_info record.

    Returns the version string (e.g. '2.0') or None if not found.
    """
    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            for metadata_record in reader.iter_metadata():
                if metadata_record.name == "pw_episode_info":
                    return metadata_record.metadata.get("schema_version")
    except (OSError, ValueError):
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a dataset quality PDF report.",
    )
    parser.add_argument(
        "dataset_path",
        type=pathlib.Path,
        help="Path to the raw dataset directory (e.g. /data/raw/task/operator)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory for the PDF (default: notebooks/outputs)",
    )
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=None,
        help="Sync tolerance in ms (default: auto = 1000/target_fps)",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--no-success-filter", action="store_true", help="Include non-success episodes"
    )
    parser.add_argument(
        "--no-excellent-filter",
        action="store_true",
        help="Include non-excellent episodes",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        metavar="VER",
        help="Only include episodes with this schema_version (e.g. 2.0)",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        metavar="PAGES",
        help="Comma-separated page numbers to include, e.g. '2' or '1,2' "
             "(1=summary, 2=drift, 3+=episodes). Default: all.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PDF resolution in DPI (default: 200, try 100 for smaller files)",
    )
    args = parser.parse_args()

    RAW_DATA_DIR = args.dataset_path.resolve()
    TARGET_FPS = args.target_fps
    TOLERANCE_MS = args.tolerance_ms
    MAX_EPISODES = args.max_episodes
    SUCCESS_ONLY = not args.no_success_filter
    EXCELLENT_ONLY = not args.no_excellent_filter

    PDF_DPI = args.dpi
    SELECTED_PAGES: set[int] | None = None
    if args.pages is not None:
        SELECTED_PAGES = {int(p.strip()) for p in args.pages.split(",")}

    actual_tolerance_ms = (
        TOLERANCE_MS if TOLERANCE_MS is not None else (1000.0 / TARGET_FPS)
    )

    TASK_NAME = RAW_DATA_DIR.parent.name
    OPERATOR_NAME = RAW_DATA_DIR.name
    DATASET_LABEL = f"{TASK_NAME}_{OPERATOR_NAME}"
    DATASET_TITLE = TASK_NAME

    # ── Output directory ──────────────────────────────────────────────
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir.resolve()
    else:
        OUTPUT_DIR = PLOTS_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset:   {RAW_DATA_DIR}")
    print(f"Task:      {TASK_NAME}")
    print(f"Operator:  {OPERATOR_NAME}")
    print(f"Target:    {TARGET_FPS} Hz")
    print(f"Tolerance: {actual_tolerance_ms:.1f} ms {'(auto)' if TOLERANCE_MS is None else ''}")
    print(f"Output:    {OUTPUT_DIR}")

    # ── Flatten topic name list ───────────────────────────────────────
    topic_names_flat: set[str] = set()
    label_to_names: dict[str, list[str]] = {}
    for label, names in TOPICS:
        topic_names_flat.update(names)
        label_to_names[label] = names

    topic_order_fwd = [label for label, _ in TOPICS]

    # ── Iterate episodes ──────────────────────────────────────────────
    episode_paths = get_selected_episodes(
        RAW_DATA_DIR,
        success_only=SUCCESS_ONLY,
        excellent_only=EXCELLENT_ONLY,
    )
    if MAX_EPISODES is not None:
        episode_paths = episode_paths[:MAX_EPISODES]

    print(f"Analysing {len(episode_paths)} episodes ...")

    # ── Extract dataset version(s) from all episodes ─────────────────
    dataset_versions: set[str] = set()
    ep_version_map: dict[int, str | None] = {}
    for i, ep_path in enumerate(episode_paths):
        v = extract_dataset_version(ep_path)
        ep_version_map[i] = v
        if v:
            dataset_versions.add(v)
    dataset_version_str = ", ".join(sorted(dataset_versions)) if dataset_versions else None
    print(f"Version:   {dataset_version_str or '(not found)'}")

    # ── Filter by version if requested ────────────────────────────
    VERSION_FILTER = args.version
    if VERSION_FILTER is not None:
        filtered = [ep for i, ep in enumerate(episode_paths) if ep_version_map.get(i) == VERSION_FILTER]
        n_before = len(episode_paths)
        episode_paths = filtered
        print(f"Filter:    v{VERSION_FILTER} → {len(episode_paths)}/{n_before} episodes")
        if not episode_paths:
            print(f"\n⚠️  No episodes match version '{VERSION_FILTER}'. Available: {dataset_version_str}")
            return
        # Update displayed version to reflect the filter
        dataset_version_str = VERSION_FILTER
    rows: list[dict] = []
    all_intervals: dict[str, list[float]] = defaultdict(list)
    # episode_timestamps uses the raw ep_idx; we will remap after filtering
    _raw_episode_timestamps: dict[int, dict[str, np.ndarray]] = {}
    _raw_episode_keyframes: dict[int, dict[str, np.ndarray]] = {}
    agg_ts_source: dict[str, dict[str, int]] = defaultdict(
        lambda: {"sensor": 0, "log": 0}
    )
    topic_schemas: dict[str, str] = {}
    agg_ts_drift: dict[str, list[float]] = defaultdict(list)
    # all raw topic names seen across all episodes (for page-1 table)
    all_mcap_topics: set[str] = set()

    for ep_idx, ep_path in enumerate(episode_paths):
        ts_data, ts_source, ep_schemas, ts_drift, kf_ts = analyse_episode(
            ep_path, topic_names_flat
        )
        ep_ts: dict[str, np.ndarray] = {}
        ep_kf: dict[str, np.ndarray] = {}

        # Collect all raw topic names present in this episode
        all_mcap_topics.update(ts_data.keys())

        # --- predefined labelled topics (for heatmap / violin / violation stats) ---
        for label, names in TOPICS:
            timestamps: list[float] = []
            matched_name = None
            for n in names:
                if n in ts_data and ts_data[n]:
                    timestamps = ts_data[n]
                    matched_name = n
                    break

            if matched_name and matched_name in ts_source:
                agg_ts_source[label]["sensor"] += ts_source[matched_name]["sensor"]
                agg_ts_source[label]["log"] += ts_source[matched_name]["log"]
            if (
                matched_name
                and matched_name in ep_schemas
                and label not in topic_schemas
            ):
                topic_schemas[label] = ep_schemas[matched_name]
            if matched_name and matched_name in ts_drift:
                agg_ts_drift[label].extend(ts_drift[matched_name])
            if matched_name and matched_name in kf_ts:
                ep_kf[label] = np.array(kf_ts[matched_name], dtype=np.float64)

            ts_arr = np.array(timestamps, dtype=np.float64)
            ep_ts[label] = ts_arr

            n_msgs = len(timestamps)
            if n_msgs < 2:
                rows.append(
                    dict(
                        episode=ep_idx,
                        topic=label,
                        n_msgs=n_msgs,
                        avg_hz=0,
                        std_hz=0,
                        min_hz=0,
                        max_hz=0,
                        duration_s=0,
                        median_interval_ms=0,
                    )
                )
                continue

            intervals = np.diff(ts_arr)
            valid = intervals[intervals > 0]
            if len(valid) == 0:
                continue

            freqs = 1.0 / valid
            all_intervals[label].extend((valid * 1000).tolist())

            rows.append(
                dict(
                    episode=ep_idx,
                    topic=label,
                    n_msgs=n_msgs,
                    avg_hz=float(np.mean(freqs)),
                    std_hz=float(np.std(freqs)),
                    min_hz=float(np.min(freqs)),
                    max_hz=float(np.max(freqs)),
                    duration_s=float(timestamps[-1] - timestamps[0]),
                    median_interval_ms=float(np.median(valid) * 1000),
                )
            )

        # Also store raw-topic arrays for the detail pages
        for raw_topic, raw_ts_list in ts_data.items():
            ts_arr_raw = np.array(raw_ts_list, dtype=np.float64)
            ep_ts[raw_topic] = ts_arr_raw
            if raw_topic in kf_ts:
                ep_kf[raw_topic] = np.array(kf_ts[raw_topic], dtype=np.float64)

        _raw_episode_timestamps[ep_idx] = ep_ts
        _raw_episode_keyframes[ep_idx] = ep_kf

        if (ep_idx + 1) % 10 == 0 or ep_idx == len(episode_paths) - 1:
            print(f"  {ep_idx + 1}/{len(episode_paths)}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("\n⚠️  No data collected — check that the dataset path exists and contains valid episodes.")
        return
    print(
        f"\nDone. {len(df)} rows collected "
        f"({df['topic'].nunique()} topics x {df['episode'].nunique()} episodes)."
    )

    # ══════════════════════════════════════════════════════════════════
    # Filter out episodes that have ANY interval above tolerance
    # ══════════════════════════════════════════════════════════════════
    # Determine which raw episodes exceed tolerance on ANY predefined topic
    _over_tolerance_raw: set[int] = set()
    for raw_ep_idx in range(len(episode_paths)):
        ep_ts_check = _raw_episode_timestamps.get(raw_ep_idx, {})
        for label in topic_order_fwd:
            ts = ep_ts_check.get(label)
            if ts is None or len(ts) < 2:
                continue
            if np.any(np.diff(ts) * 1000 > actual_tolerance_ms):
                _over_tolerance_raw.add(raw_ep_idx)
                break

    # Build remapped dicts containing ONLY passing episodes, numbered from 0
    _passing_raw_indices = [i for i in range(len(episode_paths)) if i not in _over_tolerance_raw]
    n_filtered_out = len(_over_tolerance_raw)
    print(
        f"Tolerance filter: {n_filtered_out} episodes above {actual_tolerance_ms:.1f} ms removed; "
        f"{len(_passing_raw_indices)} passing episodes kept."
    )

    episode_timestamps: dict[int, dict[str, np.ndarray]] = {}
    episode_keyframes: dict[int, dict[str, np.ndarray]] = {}
    for new_idx, raw_idx in enumerate(_passing_raw_indices):
        episode_timestamps[new_idx] = _raw_episode_timestamps[raw_idx]
        episode_keyframes[new_idx] = _raw_episode_keyframes[raw_idx]

    # ══════════════════════════════════════════════════════════════════
    # Whitelist & name map (used for table, heatmap, and violin)
    # ══════════════════════════════════════════════════════════════════
    WHITELISTED_TOPICS = [
        # State (observations)
        "/joint_states",
        "/arm_left/tcp_pose",
        "/left/franka_robot_state_broadcaster/current_pose",
        "/panda_left/tcp",
        "/arm_right/tcp_pose",
        "/right/franka_robot_state_broadcaster/current_pose",
        "/panda_right/tcp",
        # Cameras
        "/cam_left/color/image_rect_compressed",
        "/cam_left/aligned_depth_to_color/image_compressed",
        "/cam_right/color/image_rect_compressed",
        "/cam_right/aligned_depth_to_color/image_compressed",
        "/cam_static/color/image_rect_compressed",
        # Commands (actions)
        "/joint_target_left",
        "/joint_target_right",
        "/desired_pose_twist_left",
        "/cartesian_target_left",
        "/desired_pose_twist_right",
        "/cartesian_target_right",
        "/desired_gripper_values_left",
        "/desired_gripper_values_right",
    ]

    # Build reverse mapping: raw topic name → display name
    raw_topic_to_label: dict[str, str] = {}
    for _lbl, _names in TOPICS:
        for _n in _names:
            raw_topic_to_label[_n] = _lbl
    _extra_names: dict[str, str] = {
        "/cam_left/aligned_depth_to_color/image_compressed": "Depth L",
        "/cam_left/color/image_rect_compressed": "RGB L",
        "/cam_right/aligned_depth_to_color/image_compressed": "Depth R",
        "/cam_right/color/image_rect_compressed": "RGB R",
        "/cam_static/color/image_rect_compressed": "RGB S",
        "/desired_gripper_values_left": "Cmd Gripper L",
        "/desired_gripper_values_right": "Cmd Gripper R",
        "/desired_pose_twist_left": "Cmd TCP L",
        "/desired_pose_twist_right": "Cmd TCP R",
        "/joint_states": "Joint State",
        "/joint_target_left": "Cmd Joint L",
        "/joint_target_right": "Cmd Joint R",
        "/arm_left/tcp_pose": "TCP L",
        "/left/franka_robot_state_broadcaster/current_pose": "TCP L",
        "/panda_left/tcp": "TCP L",
        "/arm_right/tcp_pose": "TCP R",
        "/right/franka_robot_state_broadcaster/current_pose": "TCP R",
        "/panda_right/tcp": "TCP R",
        "/cartesian_target_left": "Cmd TCP L",
        "/cartesian_target_right": "Cmd TCP R",
    }
    for _t, _n in _extra_names.items():
        raw_topic_to_label.setdefault(_t, _n)

    whitelisted_present = [t for t in WHITELISTED_TOPICS if t in all_mcap_topics]
    # Unique display names in whitelist order (dedup for topics sharing a label)
    _seen_labels: set[str] = set()
    wl_display_labels: list[str] = []
    wl_display_topics: list[str] = []  # one representative raw topic per label
    for _t in whitelisted_present:
        _lbl = raw_topic_to_label.get(_t, _t)
        if _lbl not in _seen_labels:
            _seen_labels.add(_lbl)
            wl_display_labels.append(_lbl)
            wl_display_topics.append(_t)

    # ══════════════════════════════════════════════════════════════════
    # Heatmap data (episode × whitelisted topic)
    # ══════════════════════════════════════════════════════════════════
    n_episodes = len(episode_timestamps)
    heatmap_labels = wl_display_labels
    heatmap = np.zeros((n_episodes, len(heatmap_labels)))

    for ep_idx in range(n_episodes):
        ep_ts = episode_timestamps.get(ep_idx, {})
        for col_idx, raw_t in enumerate(wl_display_topics):
            ts = ep_ts.get(raw_t)
            if ts is None or len(ts) < 2:
                continue
            intervals_ms = np.diff(ts) * 1000
            frac = np.mean(intervals_ms > actual_tolerance_ms)
            heatmap[ep_idx, col_idx] = frac

    # Intervals for violin (all whitelisted topics from episode_timestamps)
    wl_all_intervals: dict[str, list[float]] = {}
    for _lbl, _raw_t in zip(wl_display_labels, wl_display_topics):
        _ivs: list[float] = []
        for ep_idx in range(n_episodes):
            ts = episode_timestamps.get(ep_idx, {}).get(_raw_t)
            if ts is not None and len(ts) >= 2:
                _ivs.extend((np.diff(ts) * 1000).tolist())
        if _ivs:
            wl_all_intervals[_lbl] = _ivs

    # ══════════════════════════════════════════════════════════════════
    # Violation-position data (5b)
    # ══════════════════════════════════════════════════════════════════
    N_TIME_BINS = 50
    viol_ep_indices: list[int] = []
    viol_heatmap_pct: list[np.ndarray] = []
    viol_heatmap_sec: list[np.ndarray] = []
    bins_pct = np.linspace(0, 100, N_TIME_BINS + 1)

    max_duration = 0.0
    for ep_ts in episode_timestamps.values():
        for label in topic_order_fwd:
            ts = ep_ts.get(label)
            if ts is not None and len(ts) >= 2:
                max_duration = max(max_duration, ts[-1] - ts[0])
    if max_duration <= 0:
        max_duration = 10.0
    bins_sec = np.linspace(0, max_duration * 1.02, N_TIME_BINS + 1)

    keyframe_positions_pct: dict[str, list[float]] = defaultdict(list)
    keyframe_positions_sec: dict[str, list[float]] = defaultdict(list)

    for ep_idx in sorted(episode_timestamps.keys()):
        ep_ts = episode_timestamps[ep_idx]
        ep_kf = episode_keyframes.get(ep_idx, {})
        row_pct = np.zeros(N_TIME_BINS)
        row_sec = np.zeros(N_TIME_BINS)
        has_viol = False

        for label in topic_order_fwd:
            ts = ep_ts.get(label)
            if ts is None or len(ts) < 2:
                continue
            intervals_ms = np.diff(ts) * 1000
            duration = ts[-1] - ts[0]
            if duration <= 0:
                continue
            mid_times = ts[:-1] + np.diff(ts) / 2
            elapsed = mid_times - ts[0]
            mask = intervals_ms > actual_tolerance_ms
            if mask.any():
                has_viol = True
                pos_pct = elapsed[mask] / duration * 100
                counts_pct, _ = np.histogram(pos_pct, bins=bins_pct)
                row_pct += counts_pct
                pos_sec = elapsed[mask]
                counts_sec, _ = np.histogram(pos_sec, bins=bins_sec)
                row_sec += counts_sec

            if label in ep_kf and len(ep_kf[label]) > 0:
                kf_elapsed = ep_kf[label] - ts[0]
                keyframe_positions_pct[label].extend(
                    (kf_elapsed / duration * 100).tolist()
                )
                keyframe_positions_sec[label].extend(kf_elapsed.tolist())

        if has_viol:
            viol_ep_indices.append(ep_idx)
            viol_heatmap_pct.append(row_pct)
            viol_heatmap_sec.append(row_sec)

    all_kf_pct = [v for vals in keyframe_positions_pct.values() for v in vals]
    all_kf_sec = [v for vals in keyframe_positions_sec.values() for v in vals]

    if viol_ep_indices:
        hm_pos_pct = np.array(viol_heatmap_pct)
        hm_pos_sec = np.array(viol_heatmap_sec)

    # ══════════════════════════════════════════════════════════════════
    # Verdict thresholds
    # ══════════════════════════════════════════════════════════════════
    WARN_DROP_PCT = 10.0
    FAIL_DROP_PCT = 30.0

    n_total_episodes = len(episode_timestamps)
    per_topic_viol: dict[str, dict] = {}

    for label in topic_order_fwd:
        eps_with_viol = 0
        worst_ms = 0.0
        viol_magnitudes: list[float] = []

        for ep_idx in range(n_total_episodes):
            ts = episode_timestamps.get(ep_idx, {}).get(label)
            if ts is None or len(ts) < 2:
                continue
            intervals_ms = np.diff(ts) * 1000
            mask = intervals_ms > actual_tolerance_ms
            if np.any(mask):
                eps_with_viol += 1
                viol_magnitudes.extend(intervals_ms[mask].tolist())
            if len(intervals_ms) > 0:
                worst_ms = max(worst_ms, float(np.max(intervals_ms)))

        ep_pct = (
            (eps_with_viol / n_total_episodes * 100) if n_total_episodes > 0 else 0.0
        )
        median_viol_ms = float(np.median(viol_magnitudes)) if viol_magnitudes else 0.0
        all_ivs = all_intervals.get(label, [])
        avg_interval_ms = float(np.mean(all_ivs)) if all_ivs else 0.0
        std_interval_ms = float(np.std(all_ivs)) if all_ivs else 0.0
        per_topic_viol[label] = dict(
            n_eps_viol=eps_with_viol,
            ep_pct=ep_pct,
            worst_ms=worst_ms,
            median_viol_ms=median_viol_ms,
            avg_interval_ms=avg_interval_ms,
            std_interval_ms=std_interval_ms,
        )

    worst_topic = (
        max(per_topic_viol, key=lambda k: per_topic_viol[k]["ep_pct"])
        if per_topic_viol
        else "—"
    )
    worst_topic_pct = (
        per_topic_viol[worst_topic]["ep_pct"] if worst_topic != "—" else 0.0
    )

    # Episode drop rate
    dropped_episodes: set[int] = set()
    for ep_idx in range(n_total_episodes):
        ep_ts = episode_timestamps.get(ep_idx, {})
        for label in topic_order_fwd:
            ts = ep_ts.get(label)
            if ts is None or len(ts) < 2:
                continue
            intervals_ms = np.diff(ts) * 1000
            if np.any(intervals_ms > actual_tolerance_ms):
                dropped_episodes.add(ep_idx)
                break

    n_dropped = len(dropped_episodes)
    drop_pct = (n_dropped / n_total_episodes * 100) if n_total_episodes > 0 else 0.0
    n_surviving = n_total_episodes - n_dropped

    print("\nPer-topic violations (episode-based):")
    for label in topic_order_fwd:
        v = per_topic_viol[label]
        print(
            f"  {label:<16s}  {v['n_eps_viol']:>3d} / {n_total_episodes:>3d} episodes "
            f"= {v['ep_pct']:.1f}%  worst={v['worst_ms']:.1f} ms"
        )
    print(
        f"\nEpisode drop rate: {n_dropped}/{n_total_episodes} = {drop_pct:.1f}%  "
        f"({n_surviving} surviving)"
    )

    has_unexpected_logtime = any(
        agg_ts_source[label]["log"] > 0 and agg_ts_source[label]["sensor"] > 0
        for label, _ in TOPICS
    )

    if drop_pct >= FAIL_DROP_PCT or has_unexpected_logtime:
        verdict, verdict_color = "FAIL", "#d65f5f"
    elif drop_pct >= WARN_DROP_PCT:
        verdict, verdict_color = "WARNING", "#ee854a"
    else:
        verdict, verdict_color = "PASS", "#6acc64"

    # ══════════════════════════════════════════════════════════════════
    # Build the one-pager (page 1)
    # ══════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16.53, 16))  # wide page
    fig.set_facecolor("#f0f0f0")
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[0.13, 1.0, 1.6],
        hspace=0.45,
        wspace=0.25,
        left=0.09,
        right=0.91,
        top=0.91,
        bottom=0.03,
    )

    # Title bar
    fig.suptitle(
        f"Dataset Quality Report: {DATASET_TITLE}",
        fontsize=18,
        fontweight="bold",
        y=0.97,
        color="#333333",
    )

    # Verdict badge removed — verdict shown in summary row instead

    # ── ROW 0: Summary (full width, window style) ────────────────
    ax_stats = fig.add_subplot(gs[0, :])
    ax_stats.set_facecolor("#eef6fb")
    ax_stats.axis("off")
    ax_stats.patch.set_edgecolor(PALETTE[0])
    ax_stats.patch.set_linewidth(2)

    stats_items = [
        ("Generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), None),
        ("Episodes", f"{n_total_episodes}  (filtered: {n_filtered_out})", None),
        ("Operator", f"{OPERATOR_NAME}", None),
        ("Target / Tolerance", f"{TARGET_FPS} Hz / {actual_tolerance_ms:.0f} ms", None),
        ("Worst interval", f"{max((v['worst_ms'] for v in per_topic_viol.values()), default=0):.1f} ms", None),
        ("Dataset Version", dataset_version_str or "—", None),
        ("Verdict", verdict, verdict_color),
    ]
    n_stat_cols = len(stats_items)
    for col_i, (lbl, val, val_color) in enumerate(stats_items):
        x = (col_i + 0.5) / n_stat_cols
        ax_stats.text(x, 0.80, lbl, ha="center", va="top",
                      fontsize=9, fontweight="bold", color="#555",
                      transform=ax_stats.transAxes)
        ax_stats.text(x, 0.35, val, ha="center", va="top",
                      fontsize=10, fontfamily="monospace",
                      color=val_color if val_color else "#222",
                      fontweight="bold" if val_color else "normal",
                      transform=ax_stats.transAxes)

    # ── ROW 1 LEFT: Violin (per-message frequency, all whitelisted) ─
    ax_viol = fig.add_subplot(gs[1, 0])
    wl_rev_labels = list(reversed(wl_display_labels))
    FREQ_CLIP_HZ_PDF = 1200

    pdf_inst_order: list[str] = []
    pdf_inst_data: list[np.ndarray] = []
    for label in wl_rev_labels:
        if label in wl_all_intervals and len(wl_all_intervals[label]) > 0:
            intervals_ms = np.array(wl_all_intervals[label])
            freqs = 1000.0 / intervals_ms
            freqs = freqs[freqs <= FREQ_CLIP_HZ_PDF]
            if len(freqs) > 0:
                pdf_inst_order.append(label)
                pdf_inst_data.append(freqs)

    if pdf_inst_data:
        parts = ax_viol.violinplot(
            pdf_inst_data,
            positions=range(len(pdf_inst_order)),
            vert=False,
            showmedians=True,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(PALETTE[0])
            pc.set_edgecolor("#aaaaaa")
            pc.set_linewidth(0.5)
            pc.set_alpha(0.65)
        parts["cmedians"].set_color("#333333")
        parts["cmedians"].set_linewidth(1.5)

    # Target FPS marker
    _trans_viol = blended_transform_factory(ax_viol.transData, ax_viol.transAxes)
    ax_viol.plot(
        TARGET_FPS,
        0,
        marker="|",
        markersize=8,
        markeredgewidth=1.5,
        color=PALETTE[3],
        clip_on=False,
        zorder=5,
        transform=_trans_viol,
    )
    ax_viol.annotate(
        f"{TARGET_FPS} Hz",
        xy=(TARGET_FPS, 0),
        xycoords=_trans_viol,
        xytext=(3, -6),
        textcoords="offset points",
        fontsize=8,
        color=PALETTE[3],
        fontweight="medium",
        clip_on=False,
        va="top",
        ha="left",
    )

    ax_viol.set_yticks(range(len(pdf_inst_order)))
    ax_viol.set_yticklabels(pdf_inst_order, fontsize=9)
    ax_viol.set_xlabel("Frequency (Hz)", fontsize=10)
    ax_viol.set_title(
        "Per-Message Frequency Distribution", fontsize=12, fontweight="bold", pad=8
    )
    ax_viol.tick_params(axis="x", labelsize=9)
    ax_viol.margins(y=0.05)
    ax_viol.figure.subplots_adjust()  # no-op, kept for clarity
    plt.setp(ax_viol.get_yticklabels(), ha="right")
    ax_viol.yaxis.set_tick_params(pad=4)

    # ── ROW 2: Timestamp source table (full width) ─────────────────
    ax_tbl = fig.add_subplot(gs[2, :])
    # Extend table axes left edge to near-figure-edge (wider than violin below)
    _tbl_pos = ax_tbl.get_position()
    ax_tbl.set_position([0.09, _tbl_pos.y0, 0.82, _tbl_pos.height * 0.93])
    ax_tbl.set_facecolor("#f0f0f0")
    ax_tbl.axis("off")

    # Build per-raw-topic aggregated stats from passing episodes
    raw_topic_schemas: dict[str, str] = {}
    raw_agg_source: dict[str, dict[str, int]] = defaultdict(lambda: {"sensor": 0, "log": 0})
    raw_topic_msgs: dict[str, int] = defaultdict(int)

    for ep_idx in range(n_total_episodes):
        raw_ep_path = episode_paths[_passing_raw_indices[ep_idx]]
        with open(raw_ep_path, "rb") as _f:
            _r = make_reader(_f)
            _schema_cache: dict[str, str] = {}
            for _schema, _channel, _msg in _r.iter_messages():
                _t = _channel.topic
                if _t not in _schema_cache and _schema is not None:
                    _schema_cache[_t] = _schema.name
                    raw_topic_schemas[_t] = _schema.name
                raw_topic_msgs[_t] += 1
                _log_ts = _msg.log_time * 1e-9
                if _t in _schema_cache:
                    try:
                        _sts = extract_sensor_timestamp(_msg.data, _schema_cache[_t])
                        if _sts is not None:
                            raw_agg_source[_t]["sensor"] += 1
                            continue
                    except Exception:
                        pass
                raw_agg_source[_t]["log"] += 1

    # Print topics found in MCAP but not in whitelist
    all_raw_topics_sorted = sorted(all_mcap_topics)
    extra_topics = [t for t in all_raw_topics_sorted if t not in set(WHITELISTED_TOPICS)]
    if extra_topics:
        print("\nTopics in MCAP not included in table:")
        for t in extra_topics:
            print(f"  {t}")

    tbl_ep_pcts: list[float] = []
    tbl_data = []
    for raw_topic in whitelisted_present:
        s = raw_agg_source[raw_topic]["sensor"]
        l_count = raw_agg_source[raw_topic]["log"]
        total = s + l_count
        schema = raw_topic_schemas.get(raw_topic, "—")
        if total == 0:
            src = "no data"
        elif l_count == 0:
            src = "sensor"
        elif s == 0:
            src = "log_time"
        else:
            src = f"mixed ({l_count / total * 100:.0f}% log)"
        n_msgs_total = raw_topic_msgs.get(raw_topic, 0)
        label_key = raw_topic_to_label.get(raw_topic)
        viol = per_topic_viol.get(label_key, {}) if label_key else {}
        n_ev = viol.get("n_eps_viol", 0)
        ep_p = float(viol.get("ep_pct", 0.0))
        worst_ms = viol.get("worst_ms", 0.0)
        avg_interval_ms = viol.get("avg_interval_ms", 0.0)
        std_interval_ms = viol.get("std_interval_ms", 0.0)
        if label_key and n_total_episodes > 0:
            viol_str = f"{n_ev} / {n_total_episodes} ({ep_p:.1f}%)"
            hz_str = f"{1000.0 / avg_interval_ms:.1f}" if avg_interval_ms > 0 else "—"
            mean_str = f"{avg_interval_ms:.1f}"
            std_str = f"{std_interval_ms:.1f}"
            worst_str = f"{worst_ms:.1f}"
        else:
            viol_str = "—"
            hz_str = "—"
            mean_str = "—"
            std_str = "—"
            worst_str = "—"
            ep_p = -1.0  # sentinel: no violation data available
        tbl_ep_pcts.append(ep_p)
        tbl_data.append([
            label_key or "—",
            raw_topic,
            schema.split("/")[-1],
            src,
            f"{n_msgs_total:,}",
            viol_str,
            hz_str,
            mean_str,
            std_str,
            worst_str,
        ])

    col_labels = [
        "Name",
        "Topic",
        "Schema",
        "Source",
        "Total msgs",
        "Viol. eps (%)",
        "Hz",
        "Mean gap (ms)",
        "Std (ms)",
        "Max gap (ms)",
    ]
    table = ax_tbl.table(
        cellText=tbl_data, colLabels=col_labels, cellLoc="left",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(len(col_labels))))
    table.scale(1, 1.2)

    for row_idx in range(len(tbl_data)):
        ep_p = tbl_ep_pcts[row_idx]
        for col_idx in range(len(col_labels)):
            c = table[row_idx + 1, col_idx]
            c.set_facecolor("#f8f8f8" if row_idx % 2 == 0 else "white")
            c.set_edgecolor("#dddddd")
        src_cell = table[row_idx + 1, 3]
        if "log" in tbl_data[row_idx][3]:
            src_cell.set_facecolor("#fdebd0")
        elif tbl_data[row_idx][3] == "sensor":
            src_cell.set_facecolor("#d5f5e3")
        if ep_p >= 0:
            viol_cell = table[row_idx + 1, 5]
            if ep_p >= FAIL_DROP_PCT:
                viol_cell.set_facecolor("#f5b7b1")
            elif ep_p >= WARN_DROP_PCT:
                viol_cell.set_facecolor("#fdebd0")
            else:
                viol_cell.set_facecolor("#d5f5e3")

        # Mean gap (ms) col index 7
        try:
            mean_val = float(tbl_data[row_idx][7])
            table[row_idx + 1, 7].set_facecolor("#d5f5e3" if mean_val < 35 else ("#f8f8f8" if row_idx % 2 == 0 else "white"))
        except (ValueError, TypeError):
            pass
        # Std (ms) col index 8
        try:
            std_val = float(tbl_data[row_idx][8])
            table[row_idx + 1, 8].set_facecolor("#d5f5e3" if std_val < 1.0 else ("#f8f8f8" if row_idx % 2 == 0 else "white"))
        except (ValueError, TypeError):
            pass
        # Max gap (ms) col index 9
        try:
            max_val = float(tbl_data[row_idx][9])
            table[row_idx + 1, 9].set_facecolor("#d5f5e3" if max_val < 45 else "#fdebd0")
        except (ValueError, TypeError):
            pass

    for col_idx in range(len(col_labels)):
        hdr = table[0, col_idx]
        hdr.set_facecolor(PALETTE[0])
        hdr.set_text_props(color="white", fontweight="bold")
        hdr.set_edgecolor("white")

    ax_tbl.text(
        0.5, 1.015,
        "MCAP Topics — Sources & Violations",
        ha="center", va="bottom",
        fontsize=12, fontweight="bold",
        transform=ax_tbl.transAxes,
    )

    # ── ROW 1 RIGHT: Episode × topic heatmap ────────────────────────
    ax_hm = fig.add_subplot(gs[1, 1])
    ax_hm.grid(False)
    ax_hm.set_facecolor("#eaeaf2")
    hm_vmax_pdf = float(np.max(heatmap)) if np.any(heatmap > 0) else 0.05
    heatmap_masked = np.ma.masked_where(heatmap == 0, heatmap)
    im = ax_hm.imshow(
        heatmap_masked,
        aspect="auto",
        cmap="magma_r",
        vmin=0,
        vmax=hm_vmax_pdf,
        interpolation="nearest",
        zorder=2,
    )
    ax_hm.set_xticks(range(len(heatmap_labels)))
    ax_hm.set_xticklabels(heatmap_labels, rotation=45, ha="right", fontsize=9)
    ax_hm.set_ylabel("Episode", fontsize=10)
    ax_hm.set_xticks(np.arange(-0.5, len(heatmap_labels), 1), minor=True)
    ax_hm.set_yticks(np.arange(-0.5, heatmap_masked.shape[0], 1), minor=True)
    ax_hm.grid(which="minor", color="white", linewidth=0.5, zorder=0)
    ax_hm.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax_hm, shrink=0.7, pad=0.02, label="Violation frac.")
    if not np.any(heatmap > 0):
        ax_hm.text(
            0.5, 0.5, "No violations",
            ha="center", va="center",
            fontsize=14, color="#aaaaaa",
            fontweight="medium",
            transform=ax_hm.transAxes,
        )
    ax_hm.set_title(
        f"Violation Heatmap (>{actual_tolerance_ms:.0f} ms)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    # ══════════════════════════════════════════════════════════════════
    # Save PDF
    # ══════════════════════════════════════════════════════════════════
    pdf_path = OUTPUT_DIR / f"dataset_quality_report_{DATASET_LABEL}.pdf"
    n_pages = 0
    with PdfPages(pdf_path) as pdf:
        # Page 1: one-pager summary
        if SELECTED_PAGES is None or 1 in SELECTED_PAGES:
            pdf.savefig(fig, dpi=PDF_DPI, facecolor=fig.get_facecolor())
            n_pages += 1
        plt.close(fig)

        # Pages 2+: per-episode detail timelines for all passing episodes
        for drill_idx, ep_idx in enumerate(sorted(episode_timestamps.keys())):
            ep_ts = episode_timestamps.get(ep_idx)
            if ep_ts is None:
                continue
            ep_kf = episode_keyframes.get(ep_idx, {})

            # Only show whitelisted topics in the detail view.
            _ep_present = {t for t in all_mcap_topics if t in ep_ts and len(ep_ts[t]) >= 2}
            active_raw_topics = [t for t in WHITELISTED_TOPICS if t in _ep_present]
            if not active_raw_topics:
                continue

            t0 = min(ep_ts[t][0] for t in active_raw_topics)
            t_end = max(ep_ts[t][-1] for t in active_raw_topics)
            ep_duration = t_end - t0
            n_topics = len(active_raw_topics)

            fig_ep, axes_ep = plt.subplots(
                n_topics,
                1,
                figsize=(16.53, max(11.69, n_topics * 1.1)),
                sharex=True,
                squeeze=False,
            )
            fig_ep.set_facecolor("#f0f0f0")
            # Episodes are numbered from 1 continuously
            page_ep_num = drill_idx + 1
            fig_ep.suptitle(
                f"Episode {page_ep_num}: Message Timing per Topic",
                fontsize=14,
                fontweight="bold",
                y=0.97,
            )

            for ax_row, raw_topic in enumerate(active_raw_topics):
                ax = axes_ep[ax_row, 0]
                ts = ep_ts[raw_topic]
                elapsed = ts - t0
                intervals_ms = np.diff(ts) * 1000
                mid_times = elapsed[:-1] + np.diff(elapsed) / 2

                ok = intervals_ms <= actual_tolerance_ms
                n_viol = int((~ok).sum())
                ax.scatter(
                    mid_times[ok],
                    intervals_ms[ok],
                    s=8,
                    alpha=0.55,
                    color=_CLR_OK,
                    edgecolors="none",
                    label="OK",
                    zorder=3,
                    rasterized=True,
                )
                if n_viol > 0:
                    ax.scatter(
                        mid_times[~ok],
                        intervals_ms[~ok],
                        s=28,
                        alpha=0.9,
                        color=_CLR_VIOL,
                        edgecolors="none",
                        marker="X",
                        label=f">{actual_tolerance_ms:.0f} ms  (n={n_viol})",
                        zorder=4,
                    )
                ax.axhline(
                    actual_tolerance_ms,
                    ls="--",
                    lw=1.2,
                    color=_CLR_THRESH,
                    alpha=0.5,
                )

                # Consistent y-axis: 0 to 2× tolerance so all subplots align
                ax.set_ylim(0, actual_tolerance_ms * 2)

                if raw_topic in ep_kf and len(ep_kf[raw_topic]) > 0:
                    kf_elapsed = ep_kf[raw_topic] - t0
                    for k_i, kf_t in enumerate(kf_elapsed):
                        ax.axvline(
                            kf_t,
                            ls="-",
                            lw=0.8,
                            color=PALETTE[2],
                            alpha=0.45,
                            label="Keyframe" if k_i == 0 else None,
                        )

                # Use display name + ROS topic in parentheses
                _label = raw_topic_to_label.get(raw_topic, raw_topic)
                display_name = f"{_label} ({raw_topic})"
                # Place name as a left-aligned text inside the axes to avoid overlap
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelsize=7)
                ax.text(
                    0.002, 0.97, display_name,
                    transform=ax.transAxes,
                    fontsize=8, fontweight="bold",
                    va="top", ha="left",
                    color="#333",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.7),
                )
                if ax_row == 0:
                    ax.legend(loc="upper right", fontsize=7)

            axes_ep[-1, 0].set_xlabel("Elapsed time (s)")
            axes_ep[0, 0].set_xlim(0, ep_duration)
            fig_ep.tight_layout(rect=[0.09, 0.03, 0.91, 0.97])
            page_num = 2 + drill_idx
            if SELECTED_PAGES is None or page_num in SELECTED_PAGES:
                pdf.savefig(fig_ep, dpi=min(PDF_DPI, 100), facecolor=fig_ep.get_facecolor())
                n_pages += 1
            plt.close(fig_ep)
            gc.collect()

    print(
        f"\n✅ PDF saved to: {pdf_path}  ({n_pages} pages)"
    )


if __name__ == "__main__":
    main()
