"""Plot UMI-delta per-step action distributions as violin plots.

One subplot per action dimension (20 total), each showing a violin for every
action step in the chunk horizon.  Requires the actual LeRobot dataset so we
can build chunk-relative UMI-delta values from the raw data.

Usage:
  python -m example_policies.data_ops.review.plot_umi_delta_stats \\
      --dataset /path/to/lerobot_dataset --out umi_stats.png

  # Or pass horizon explicitly (default: auto-detect from stats JSON):
  python -m example_policies.data_ops.review.plot_umi_delta_stats \\
      --dataset /path/to/lerobot_dataset --horizon 144 --out umi_stats.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

STEPWISE_STATS_FILENAME = "stepwise_percentile_stats.json"

# 20-dim UMI-delta layout:
# [L_dpos(3), L_rot6d(6), R_dpos(3), R_rot6d(6), grip_L, grip_R]
DIM_LABELS = [
    "L dpos x", "L dpos y", "L dpos z",
    "L rot6d 0", "L rot6d 1", "L rot6d 2",
    "L rot6d 3", "L rot6d 4", "L rot6d 5",
    "R dpos x", "R dpos y", "R dpos z",
    "R rot6d 0", "R rot6d 1", "R rot6d 2",
    "R rot6d 3", "R rot6d 4", "R rot6d 5",
    "Grip L", "Grip R",
]

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#eeeeee",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "medium",
    "figure.dpi": 130,
})


def _detect_horizon(data_dir: Path) -> int:
    """Try to read horizon from existing stats JSON."""
    stats_path = data_dir / STEPWISE_STATS_FILENAME
    if stats_path.exists():
        data = json.loads(stats_path.read_text())
        return len(data["p_low"])
    raise ValueError(
        f"Cannot detect horizon: {stats_path} not found. Pass --horizon explicitly."
    )


def _build_chunks(data_dir: Path, horizon: int) -> np.ndarray:
    """Load dataset and build chunk-relative UMI-delta chunks.

    Returns:
        np.ndarray of shape (N_chunks, horizon, 20).
    """
    from example_policies.utils.chunk_relative_processor import (
        abs_tcp_to_chunk_relative_umi_delta,
    )
    from example_policies.data_ops.utils.rotation_6d import quat_to_6d_torch

    parquet_dir = data_dir / "data"
    all_actions: list[torch.Tensor] = []
    all_obs_states: list[torch.Tensor] = []
    episode_indices: list[int] = []

    for parquet_file in sorted(parquet_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_file)
        if "action" not in df.columns or "observation.state" not in df.columns:
            continue
        for a, s, ep_idx in zip(
            df["action"].tolist(),
            df["observation.state"].tolist(),
            df["episode_index"].tolist(),
        ):
            all_actions.append(torch.tensor(a, dtype=torch.float32))
            all_obs_states.append(torch.tensor(s, dtype=torch.float32))
            episode_indices.append(ep_idx)

    if not all_actions:
        raise RuntimeError(f"No actions found in {parquet_dir}.")

    # Group by episode
    episodes: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    for action, obs_state, ep_idx in zip(all_actions, all_obs_states, episode_indices):
        episodes[ep_idx].append((action, obs_state))

    # Read TCP indices from info.json (same as config_factory)
    info_path = data_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    state_names = info.get("features", {}).get("observation.state", {}).get("names", [])
    if not state_names:
        raise ValueError("observation.state feature names not found in info.json.")

    def _find_indices(prefix: str, count: int) -> list[int]:
        start = state_names.index(prefix)
        return list(range(start, start + count))

    obs_tcp_l_pos = _find_indices("tcp_left_pos_x", 3)
    obs_tcp_l_quat = _find_indices("tcp_left_quat_x", 4)
    obs_tcp_r_pos = _find_indices("tcp_right_pos_x", 3)
    obs_tcp_r_quat = _find_indices("tcp_right_quat_x", 4)

    chunks: list[torch.Tensor] = []
    for ep_idx in sorted(episodes.keys()):
        ep_data = episodes[ep_idx]
        for i in range(len(ep_data) - horizon + 1):
            _, ref_obs_state = ep_data[i]

            ref_pos_l = ref_obs_state[obs_tcp_l_pos]
            ref_quat_l = ref_obs_state[obs_tcp_l_quat]
            ref_pos_r = ref_obs_state[obs_tcp_r_pos]
            ref_quat_r = ref_obs_state[obs_tcp_r_quat]

            ref_rot6d_l = quat_to_6d_torch(ref_quat_l)
            ref_rot6d_r = quat_to_6d_torch(ref_quat_r)

            abs_chunk = torch.stack(
                [ep_data[i + k][0] for k in range(horizon)]
            )

            umi_chunk = abs_tcp_to_chunk_relative_umi_delta(
                abs_chunk.unsqueeze(0),
                ref_pos_l.unsqueeze(0),
                ref_rot6d_l.unsqueeze(0),
                ref_pos_r.unsqueeze(0),
                ref_rot6d_r.unsqueeze(0),
            ).squeeze(0)

            chunks.append(umi_chunk)

    print(f"Built {len(chunks)} chunks from {len(episodes)} episodes (horizon={horizon})")
    return torch.stack(chunks).numpy()  # (N, H, 20)


def _normalize_chunks(
    chunks: np.ndarray,
    stats_path: Path,
    skip_feature_indices: list[int] | None = None,
    clip_min: float = -1.5,
    clip_max: float = 1.5,
) -> np.ndarray:
    """Apply stepwise percentile normalization to chunks.

    Uses the same formula as :class:`StepwisePercentileNormalize`:
        y = clamp(2 * (x - p02) / (p98 - p02) - 1,  clip_min,  clip_max)

    Args:
        chunks: (N_chunks, horizon, action_dim) array of raw UMI-delta values.
        stats_path: Path to ``stepwise_percentile_stats.json``.
        skip_feature_indices: Feature indices to leave unnormalized (pass-through).
        clip_min: Lower clamp bound (default -1.5).
        clip_max: Upper clamp bound (default  1.5).

    Returns:
        Normalized array of the same shape.
    """
    data = json.loads(stats_path.read_text())
    p_low = np.array(data["p_low"])   # (H, D)
    p_high = np.array(data["p_high"])  # (H, D)

    H_stats, D = p_low.shape
    _, H_chunk, D_chunk = chunks.shape
    assert D_chunk == D, f"Dim mismatch: chunks {D_chunk} vs stats {D}"
    assert H_chunk <= H_stats, f"Chunk horizon {H_chunk} > stats horizon {H_stats}"

    p_low = p_low[:H_chunk]   # slice to actual horizon
    p_high = p_high[:H_chunk]

    denom = p_high - p_low
    denom[denom == 0] = 1.0  # avoid division by zero

    normed = 2.0 * (chunks - p_low[np.newaxis]) / denom[np.newaxis] - 1.0
    normed = np.clip(normed, clip_min, clip_max)

    # Restore skip features (pass-through)
    if skip_feature_indices:
        for idx in skip_feature_indices:
            normed[:, :, idx] = chunks[:, :, idx]

    return normed


def _plot_violins(
    chunks: np.ndarray,
    out_path: Path,
    dataset_label: str = "",
    step_stride: int = 1,
    colormap: str = "Blues",
    edge_color: str = "#2c3e6e",
    median_color: str = "#1a2540",
    title_prefix: str = "UMI-delta per-step distributions",
) -> None:
    """Create violin plots: one subplot per dimension, violins per action step.

    Args:
        chunks: (N_chunks, horizon, action_dim) array.
        out_path: Where to save the figure.
        dataset_label: Label for the figure title.
        step_stride: Plot every N-th step (useful for large horizons).
        colormap: Matplotlib colormap name for violin fills.
        edge_color: Edge color for violin bodies.
        median_color: Color for median lines.
        title_prefix: First part of the figure title.
    """
    n_chunks, horizon, action_dim = chunks.shape
    step_indices = list(range(0, horizon, step_stride))
    n_steps = len(step_indices)

    n_labels = min(action_dim, len(DIM_LABELS))
    labels = DIM_LABELS[:n_labels] + [f"dim {d}" for d in range(n_labels, action_dim)]

    fig, axes = plt.subplots(
        action_dim, 1,
        figsize=(max(8, n_steps * 0.12 + 2), action_dim * 1.6),
        sharex=True,
    )
    if action_dim == 1:
        axes = [axes]

    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0.4, 0.85, n_steps))

    for d, ax in enumerate(axes):
        data_per_step = [chunks[:, s, d] for s in step_indices]

        parts = ax.violinplot(
            data_per_step,
            positions=step_indices,
            showmedians=True,
            showextrema=False,
            widths=max(0.8, step_stride * 0.8),
        )

        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[j])
            pc.set_edgecolor(edge_color)
            pc.set_linewidth(0.4)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color(median_color)
        parts["cmedians"].set_linewidth(1.0)

        ax.axhline(0, ls="-", lw=0.5, color="#999999", alpha=0.5)
        ax.set_ylabel(labels[d], fontsize=8, rotation=0, ha="right", va="center")
        ax.tick_params(axis="y", labelsize=7)

    axes[-1].set_xlabel("Action step in chunk")
    axes[-1].tick_params(axis="x", labelsize=8)

    title = title_prefix
    if dataset_label:
        title += f" — {dataset_label}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot UMI-delta per-step action distributions as violins"
    )
    parser.add_argument(
        "--dataset", "--stats", type=Path, required=True,
        help="LeRobot dataset root directory",
    )
    parser.add_argument(
        "--horizon", type=int, default=None,
        help="Action chunk horizon (auto-detected from stats JSON if omitted)",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Plot every N-th step (default: 1 = all steps)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("umi_delta_stepwise_stats.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--normalized", action="store_true", default=False,
        help="Also produce a normalized plot (green violins) using stepwise percentile stats",
    )
    parser.add_argument(
        "--out-normalized", type=Path, default=None,
        help="Output PNG path for the normalized plot (default: <out>_normalized.png)",
    )
    args = parser.parse_args()

    data_dir = Path(args.dataset)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    horizon = args.horizon or _detect_horizon(data_dir)
    dataset_label = f"{data_dir.parent.name}/{data_dir.name}"

    chunks = _build_chunks(data_dir, horizon)

    # --- Unnormalized plot (blue) ---
    _plot_violins(
        chunks, args.out,
        dataset_label=dataset_label,
        step_stride=args.stride,
        colormap="Blues",
        edge_color="#2c3e6e",
        median_color="#1a2540",
        title_prefix="UMI-delta per-step distributions (unnormalized)",
    )

    # --- Normalized plot (green) ---
    if args.normalized:
        stats_path = data_dir / STEPWISE_STATS_FILENAME
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Stepwise stats not found: {stats_path}. "
                "Cannot produce normalized plot."
            )
        normed_chunks = _normalize_chunks(chunks, stats_path)
        out_norm = args.out_normalized or args.out.with_name(
            args.out.stem + "_normalized" + args.out.suffix
        )
        _plot_violins(
            normed_chunks, out_norm,
            dataset_label=dataset_label,
            step_stride=args.stride,
            colormap="Greens",
            edge_color="#1a4d2e",
            median_color="#0d2618",
            title_prefix="UMI-delta per-step distributions (normalized)",
        )


if __name__ == "__main__":
    main()
