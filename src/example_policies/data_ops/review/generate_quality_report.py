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
    ("TCP L", [RosTopicEnum.ACTUAL_TCP_LEFT.value, "/left/franka_robot_state_broadcaster/current_pose", "/panda_left/tcp"]),
    ("TCP R", [RosTopicEnum.ACTUAL_TCP_RIGHT.value, "/right/franka_robot_state_broadcaster/current_pose", "/panda_right/tcp"]),
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
              {topic: keyframe_timestamps})."""
    per_topic: dict[str, list[float]] = defaultdict(list)
    schema_cache: dict[str, str] = {}
    ts_source: dict[str, dict[str, int]] = defaultdict(lambda: {"sensor": 0, "log": 0})
    ts_drift: dict[str, list[float]] = defaultdict(list)
    keyframe_ts: dict[str, list[float]] = defaultdict(list)

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            t = channel.topic
            if t not in topic_names_flat:
                continue
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
    args = parser.parse_args()

    RAW_DATA_DIR = args.dataset_path.resolve()
    TARGET_FPS = args.target_fps
    TOLERANCE_MS = args.tolerance_ms
    MAX_EPISODES = args.max_episodes
    SUCCESS_ONLY = not args.no_success_filter
    EXCELLENT_ONLY = not args.no_excellent_filter

    actual_tolerance_ms = (
        TOLERANCE_MS if TOLERANCE_MS is not None else (1000.0 / TARGET_FPS)
    )

    TASK_NAME = RAW_DATA_DIR.parent.name
    OPERATOR_NAME = RAW_DATA_DIR.name
    DATASET_LABEL = f"{TASK_NAME}_{OPERATOR_NAME}"
    DATASET_TITLE = f"{TASK_NAME} — {OPERATOR_NAME}"

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
    episode_timestamps: dict[int, dict[str, np.ndarray]] = {}
    episode_keyframes: dict[int, dict[str, np.ndarray]] = {}
    agg_ts_source: dict[str, dict[str, int]] = defaultdict(
        lambda: {"sensor": 0, "log": 0}
    )
    topic_schemas: dict[str, str] = {}
    agg_ts_drift: dict[str, list[float]] = defaultdict(list)

    for ep_idx, ep_path in enumerate(episode_paths):
        ts_data, ts_source, ep_schemas, ts_drift, kf_ts = analyse_episode(
            ep_path, topic_names_flat
        )
        ep_ts: dict[str, np.ndarray] = {}
        ep_kf: dict[str, np.ndarray] = {}

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

        episode_timestamps[ep_idx] = ep_ts
        episode_keyframes[ep_idx] = ep_kf

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
    # Heatmap data (5c: episode × topic)
    # ══════════════════════════════════════════════════════════════════
    n_episodes = len(episode_timestamps)
    heatmap_labels = topic_order_fwd
    heatmap = np.zeros((n_episodes, len(heatmap_labels)))

    for ep_idx in range(n_episodes):
        ep_ts = episode_timestamps.get(ep_idx, {})
        for col_idx, label in enumerate(heatmap_labels):
            ts = ep_ts.get(label)
            if ts is None or len(ts) < 2:
                continue
            intervals_ms = np.diff(ts) * 1000
            frac = np.mean(intervals_ms > actual_tolerance_ms)
            heatmap[ep_idx, col_idx] = frac

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
        per_topic_viol[label] = dict(
            n_eps_viol=eps_with_viol,
            ep_pct=ep_pct,
            worst_ms=worst_ms,
            median_viol_ms=median_viol_ms,
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
    fig = plt.figure(figsize=(16.53, 11.69))  # landscape A4
    fig.set_facecolor("#f0f0f0")
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1.1, 1.2, 0.9],
        hspace=0.35,
        wspace=0.3,
        left=0.06,
        right=0.96,
        top=0.90,
        bottom=0.06,
    )

    # Title bar
    fig.suptitle(
        f"Dataset Quality Report — {DATASET_TITLE}",
        fontsize=16,
        fontweight="bold",
        y=0.97,
        color="#333333",
    )
    fig.text(
        0.5,
        0.935,
        f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}  |  "
        f"{len(episode_paths)} episodes  |  {TARGET_FPS} Hz target  |  "
        f"{actual_tolerance_ms:.0f} ms tolerance"
        + (f"  |  v{dataset_version_str}" if dataset_version_str else ''),
        ha="center",
        fontsize=9,
        color="#777",
    )

    # Verdict badge
    fig.text(
        0.93,
        0.96,
        verdict,
        fontsize=14,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=verdict_color,
            edgecolor="none",
            alpha=0.9,
        ),
    )

    # ── ROW 1 LEFT: Key violation statistics ──────────────────────────
    ax_stats = fig.add_subplot(gs[0, 0])
    ax_stats.set_facecolor("#f0f0f0")
    ax_stats.axis("off")

    stats_lines = [
        (
            "Episode drop rate",
            f"{n_dropped} / {n_total_episodes}  ({drop_pct:.1f}%)",
        ),
        ("Surviving episodes", f"{n_surviving}"),
        ("Worst topic", f"{worst_topic} ({worst_topic_pct:.0f}% of episodes)"),
        (
            "Worst single interval",
            f"{max((v['worst_ms'] for v in per_topic_viol.values()), default=0):.1f} ms",
        ),
        ("Operator", f"{OPERATOR_NAME}"),
        ("Target / Tolerance", f"{TARGET_FPS} Hz / {actual_tolerance_ms:.0f} ms"),
        ("Dataset Version", dataset_version_str or "—"),
    ]

    y_pos = 0.95
    for label_txt, value_txt in stats_lines:
        ax_stats.text(
            0.05,
            y_pos,
            label_txt,
            fontsize=9,
            fontweight="medium",
            color="#555",
            transform=ax_stats.transAxes,
            va="top",
        )
        ax_stats.text(
            0.55,
            y_pos,
            value_txt,
            fontsize=9,
            fontfamily="monospace",
            color="#333",
            transform=ax_stats.transAxes,
            va="top",
        )
        y_pos -= 0.12

    ax_stats.set_title(
        "Violation Summary", fontsize=10, fontweight="medium", pad=12
    )

    # ── ROW 1 RIGHT: Violin (per-message frequency) ──────────────────
    ax_viol = fig.add_subplot(gs[0, 1])
    topic_order_rev = list(reversed(topic_order_fwd))
    FREQ_CLIP_HZ_PDF = 1200

    pdf_inst_order: list[str] = []
    pdf_inst_data: list[np.ndarray] = []
    for label in topic_order_rev:
        if label in all_intervals and len(all_intervals[label]) > 0:
            intervals_ms = np.array(all_intervals[label])
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
        fontsize=6,
        color=PALETTE[3],
        fontweight="medium",
        clip_on=False,
        va="top",
        ha="left",
    )

    ax_viol.set_yticks(range(len(pdf_inst_order)))
    ax_viol.set_yticklabels(pdf_inst_order, fontsize=7)
    ax_viol.set_xlabel("Frequency (Hz)", fontsize=8)
    ax_viol.set_title(
        "Per-Message Frequency Distribution", fontsize=10, fontweight="medium", pad=8
    )
    ax_viol.tick_params(axis="x", labelsize=7)

    # ── ROW 2 LEFT: Timestamp source table ────────────────────────────
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_tbl.set_facecolor("#f0f0f0")
    ax_tbl.axis("off")

    tbl_data = []
    for label, _ in TOPICS:
        s = agg_ts_source[label]["sensor"]
        l_count = agg_ts_source[label]["log"]
        total = s + l_count
        schema = topic_schemas.get(label, "—")
        if total == 0:
            src = "no data"
        elif l_count == 0:
            src = "sensor"
        elif s == 0:
            src = "log_time"
        else:
            src = f"mixed ({l_count / total * 100:.0f}% log)"
        drifts = agg_ts_drift.get(label, [])
        if drifts:
            d = np.array(drifts) * 1000
            drift_str = f"{np.median(d):.1f} / {np.max(d):.1f}"
        else:
            drift_str = "—"
        viol = per_topic_viol.get(label, {})
        n_ev = viol.get("n_eps_viol", 0)
        ep_p = viol.get("ep_pct", 0)
        viol_str = (
            f"{n_ev} / {n_total_episodes} ({ep_p:.0f}%)"
            if n_total_episodes > 0
            else "—"
        )
        tbl_data.append([label, schema.split("/")[-1], src, drift_str, viol_str])

    col_labels = [
        "Topic",
        "Schema",
        "Source",
        "Drift med/max (ms)",
        "Violations (eps)",
    ]
    table = ax_tbl.table(
        cellText=tbl_data, colLabels=col_labels, loc="center", cellLoc="left"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.auto_set_column_width(list(range(len(col_labels))))
    table.scale(1, 1.15)

    for row_idx in range(len(tbl_data)):
        cell = table[row_idx + 1, 4]
        ep_p = per_topic_viol.get(tbl_data[row_idx][0], {}).get("ep_pct", 0)
        if ep_p >= FAIL_DROP_PCT:
            cell.set_facecolor("#f5b7b1")
        elif ep_p >= WARN_DROP_PCT:
            cell.set_facecolor("#fdebd0")
        else:
            cell.set_facecolor("#d5f5e3")
        src_cell = table[row_idx + 1, 2]
        if "log" in tbl_data[row_idx][2]:
            src_cell.set_facecolor("#fdebd0")
        elif tbl_data[row_idx][2] == "sensor":
            src_cell.set_facecolor("#d5f5e3")
        for col_idx in range(len(col_labels)):
            c = table[row_idx + 1, col_idx]
            if col_idx not in (2, 4):
                c.set_facecolor("#f8f8f8" if row_idx % 2 == 0 else "white")
            c.set_edgecolor("#dddddd")

    for col_idx in range(len(col_labels)):
        hdr = table[0, col_idx]
        hdr.set_facecolor(PALETTE[0])
        hdr.set_text_props(color="white", fontweight="bold")
        hdr.set_edgecolor(PALETTE[0])

    ax_tbl.set_title(
        "Timestamp Sources & Violation Rates",
        fontsize=10,
        fontweight="medium",
        pad=12,
    )

    # ── ROW 2 RIGHT: Episode × topic heatmap ─────────────────────────
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
    ax_hm.set_xticklabels(heatmap_labels, rotation=45, ha="right", fontsize=7)
    ax_hm.set_ylabel("Episode", fontsize=8)
    ax_hm.set_xticks(np.arange(-0.5, len(heatmap_labels), 1), minor=True)
    ax_hm.set_yticks(np.arange(-0.5, heatmap_masked.shape[0], 1), minor=True)
    ax_hm.grid(which="minor", color="white", linewidth=0.5, zorder=0)
    ax_hm.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax_hm, shrink=0.7, pad=0.02, label="Violation frac.")
    ax_hm.set_title(
        f"Violation Heatmap (>{actual_tolerance_ms:.0f} ms)",
        fontsize=10,
        fontweight="medium",
        pad=8,
    )

    # ── ROW 3: Violation position scatter ─────────────────────────────
    ax_pos_pct = fig.add_subplot(gs[2, 0])
    ax_pos_sec = fig.add_subplot(gs[2, 1])

    pdf_viol_data: list[tuple[int, list[tuple[float, float, float]]]] = []

    for ep_idx in sorted(episode_timestamps.keys()):
        ep_ts = episode_timestamps[ep_idx]
        ep_points: list[tuple[float, float, float]] = []

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
                for p, s_val, m in zip(
                    (elapsed[mask] / duration * 100).tolist(),
                    elapsed[mask].tolist(),
                    intervals_ms[mask].tolist(),
                ):
                    ep_points.append((p, s_val, m))

        if ep_points:
            pdf_viol_data.append((ep_idx, ep_points))

    if pdf_viol_data:
        n_ve = len(pdf_viol_data)
        pdf_ve_indices: list[int] = []
        all_pct_x: list[float] = []
        all_pct_y: list[float] = []
        all_sec_x: list[float] = []
        all_sec_y: list[float] = []
        all_mag: list[float] = []

        for row, (ep_idx, points) in enumerate(pdf_viol_data):
            pdf_ve_indices.append(ep_idx)
            for p, s_val, m in points:
                all_pct_x.append(p)
                all_pct_y.append(row)
                all_sec_x.append(s_val)
                all_sec_y.append(row)
                all_mag.append(m)

        all_pct_x_a = np.array(all_pct_x)
        all_pct_y_a = np.array(all_pct_y)
        all_sec_x_a = np.array(all_sec_x)
        all_sec_y_a = np.array(all_sec_y)
        all_mag_a = np.array(all_mag)

        # Left: normalised (%)
        sc_p = ax_pos_pct.scatter(
            all_pct_x_a,
            all_pct_y_a,
            c=all_mag_a,
            cmap="magma_r",
            s=12,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.3,
            vmin=actual_tolerance_ms,
            zorder=3,
        )
        ax_pos_pct.set_yticks(range(n_ve))
        ax_pos_pct.set_yticklabels([str(i) for i in pdf_ve_indices], fontsize=5)
        ax_pos_pct.set_ylabel("Episode", fontsize=8)
        ax_pos_pct.set_xlabel("Position (%)", fontsize=8)
        ax_pos_pct.set_title(
            "Violation Position — Normalised",
            fontsize=10,
            fontweight="medium",
            pad=8,
        )
        ax_pos_pct.set_xlim(-2, 102)
        ax_pos_pct.set_ylim(n_ve - 0.5, -0.5)
        fig.colorbar(
            sc_p, ax=ax_pos_pct, shrink=0.7, pad=0.02, label="Interval (ms)"
        )

        if all_kf_pct:
            ax_pos_pct.scatter(
                all_kf_pct,
                [-0.8] * len(all_kf_pct),
                marker="|",
                s=20,
                lw=0.5,
                color=PALETTE[2],
                alpha=0.5,
                clip_on=False,
                label="Keyframes",
            )
            ax_pos_pct.legend(fontsize=6, loc="upper right")

        # Right: absolute (seconds)
        sc_s = ax_pos_sec.scatter(
            all_sec_x_a,
            all_sec_y_a,
            c=all_mag_a,
            cmap="magma_r",
            s=12,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.3,
            vmin=actual_tolerance_ms,
            zorder=3,
        )
        ax_pos_sec.set_yticks(range(n_ve))
        ax_pos_sec.set_yticklabels([str(i) for i in pdf_ve_indices], fontsize=5)
        ax_pos_sec.set_ylabel("Episode", fontsize=8)
        ax_pos_sec.set_xlabel("Position (s)", fontsize=8)
        ax_pos_sec.set_title(
            "Violation Position — Absolute",
            fontsize=10,
            fontweight="medium",
            pad=8,
        )
        ax_pos_sec.set_ylim(n_ve - 0.5, -0.5)
        fig.colorbar(
            sc_s, ax=ax_pos_sec, shrink=0.7, pad=0.02, label="Interval (ms)"
        )

        if all_kf_sec:
            ax_pos_sec.scatter(
                all_kf_sec,
                [-0.8] * len(all_kf_sec),
                marker="|",
                s=20,
                lw=0.5,
                color=PALETTE[2],
                alpha=0.5,
                clip_on=False,
            )

        ax_pos_pct.tick_params(labelsize=7)
        ax_pos_sec.tick_params(labelsize=7)
    else:
        for ax_empty in [ax_pos_pct, ax_pos_sec]:
            ax_empty.axis("off")
            ax_empty.text(
                0.5,
                0.5,
                "No violations",
                ha="center",
                va="center",
                fontsize=12,
                color="#aaa",
                transform=ax_empty.transAxes,
            )

    # ══════════════════════════════════════════════════════════════════
    # Save PDF
    # ══════════════════════════════════════════════════════════════════
    pdf_path = OUTPUT_DIR / f"dataset_quality_report_{DATASET_LABEL}.pdf"
    with PdfPages(pdf_path) as pdf:
        # Page 1: one-pager summary
        pdf.savefig(fig, dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)

        # Pages 2+: per-episode spike timelines for violating episodes
        pdf_drill_eps = sorted(dropped_episodes)
        for ep_idx in pdf_drill_eps:
            ep_ts = episode_timestamps.get(ep_idx)
            if ep_ts is None:
                continue
            ep_kf = episode_keyframes.get(ep_idx, {})

            active_topics = [
                lbl
                for lbl in topic_order_fwd
                if lbl in ep_ts and len(ep_ts[lbl]) >= 2
            ]
            if not active_topics:
                continue

            t0 = min(ep_ts[lbl][0] for lbl in active_topics)
            n_topics = len(active_topics)

            fig_ep, axes_ep = plt.subplots(
                n_topics,
                1,
                figsize=(16.53, 11.69),
                sharex=True,
                squeeze=False,
            )
            fig_ep.set_facecolor("#f0f0f0")
            fig_ep.suptitle(
                f"Episode {ep_idx} — inter-message interval vs. time",
                fontsize=14,
                fontweight="bold",
                y=0.97,
            )

            for ax_row, label in enumerate(active_topics):
                ax = axes_ep[ax_row, 0]
                ts = ep_ts[label]
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

                if label in ep_kf and len(ep_kf[label]) > 0:
                    kf_elapsed = ep_kf[label] - t0
                    for k_i, kf_t in enumerate(kf_elapsed):
                        ax.axvline(
                            kf_t,
                            ls="-",
                            lw=0.8,
                            color=PALETTE[2],
                            alpha=0.45,
                            label="Keyframe" if k_i == 0 else None,
                        )

                ax.set_ylabel(label, fontsize=9)
                if ax_row == 0:
                    ax.legend(loc="upper right", fontsize=7)

            axes_ep[-1, 0].set_xlabel("Elapsed time (s)")
            fig_ep.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig_ep, dpi=200, facecolor=fig_ep.get_facecolor())
            plt.close(fig_ep)

    n_pages = 1 + len(pdf_drill_eps)
    print(
        f"\n✅ PDF saved to: {pdf_path}  "
        f"({n_pages} pages: 1 summary + {n_pages - 1} episode drill-downs)"
    )


if __name__ == "__main__":
    main()
