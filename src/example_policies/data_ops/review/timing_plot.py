"""Timing-analysis plot for deployment rollouts.

Generates a 2×2 figure showing step-duration timeline, duration histogram,
inference compute time per chunk, and frequency timeline — all split by
inference vs. queue-replay steps.

Style follows the seaborn-inspired palette used in
``example_policies.data_ops.review.generate_quality_report``.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from example_policies.robot_deploy.deploy_core.inference_runner import TimingStats

# ── Palette ──────────────────────────────────────────────────────────
CLR_INF = "#4878d0"      # blue  – inference steps
CLR_QUEUE = "#6acc64"    # green – queue-replay steps
CLR_TARGET = "#d65f5f"   # red   – target / budget line
CLR_BG = "#f0f0f0"

_RC_OVERRIDES = {
    "figure.facecolor": CLR_BG,
    "axes.facecolor": "#eaeaf2",
    "axes.edgecolor": "white",
    "axes.linewidth": 0,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "font.family": "sans-serif",
    "font.size": 11,
    "savefig.facecolor": CLR_BG,
}


def save_timing_plot(
    stats: TimingStats,
    target_period: float,
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Generate and save a timing analysis plot. Returns *output_path*."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not stats.step_durations:
        return output_path

    target_hz = 1.0 / target_period
    target_ms = target_period * 1000
    all_ms = np.array(stats.step_durations) * 1000
    is_inf = (
        np.array(stats.step_is_inference, dtype=bool)
        if stats.step_is_inference
        else np.ones(len(all_ms), dtype=bool)
    )
    inf_ms = all_ms[is_inf]
    queue_ms = all_ms[~is_inf]
    step_idx = np.arange(len(all_ms))

    plt.rcParams.update(_RC_OVERRIDES)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Deployment Timing Analysis  ({len(all_ms)} steps, target {target_hz:.0f} Hz)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # ── (0,0) Step duration timeline ─────────────────────────────────
    ax = axes[0, 0]
    inf_idx = step_idx[is_inf]
    queue_idx = step_idx[~is_inf]
    ax.scatter(inf_idx, inf_ms, s=6, alpha=0.7, color=CLR_INF, label="Inference", zorder=3)
    ax.scatter(queue_idx, queue_ms, s=6, alpha=0.7, color=CLR_QUEUE, label="Queue replay", zorder=3)
    ax.axhline(target_ms, color=CLR_TARGET, ls="--", lw=1.2, label=f"Target ({target_ms:.1f} ms)")
    ax.set_xlabel("Step index")
    ax.set_ylabel("Step duration (ms)")
    ax.set_title("Step Duration Timeline")
    ax.legend(loc="upper right", fontsize=9)

    # ── (0,1) Duration histogram by type ─────────────────────────────
    ax = axes[0, 1]
    bins = np.linspace(0, min(max(all_ms) * 1.1, target_ms * 4), 60)
    if len(inf_ms) > 0:
        ax.hist(inf_ms, bins=bins, alpha=0.7, color=CLR_INF, label=f"Inference (n={len(inf_ms)})")
    if len(queue_ms) > 0:
        ax.hist(queue_ms, bins=bins, alpha=0.7, color=CLR_QUEUE, label=f"Queue (n={len(queue_ms)})")
    ax.axvline(target_ms, color=CLR_TARGET, ls="--", lw=1.2, label=f"Target ({target_ms:.1f} ms)")
    ax.set_xlabel("Step duration (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Step Duration Distribution")
    ax.legend(loc="upper right", fontsize=9)

    # ── (1,0) Inference compute time per chunk ───────────────────────
    ax = axes[1, 0]
    if stats.inference_durations:
        inf_compute_ms = np.array(stats.inference_durations) * 1000
        inf_chunk_idx = np.arange(len(inf_compute_ms))
        ax.bar(inf_chunk_idx, inf_compute_ms, color=CLR_INF, alpha=0.8, width=0.8)
        ax.axhline(target_ms, color=CLR_TARGET, ls="--", lw=1.2, label=f"Budget ({target_ms:.1f} ms)")
        ax.set_xlabel("Chunk index")
        ax.set_ylabel("Inference time (ms)")
        ax.set_title("Inference Compute Time per Chunk")
        ax.legend(loc="upper right", fontsize=9)
    else:
        ax.text(
            0.5, 0.5, "No inference data",
            transform=ax.transAxes, ha="center", va="center", fontsize=13, color="#888",
        )
        ax.set_title("Inference Compute Time per Chunk")

    # ── (1,1) Frequency timeline ─────────────────────────────────────
    ax = axes[1, 1]
    freqs = np.where(all_ms > 0, 1000.0 / all_ms, 0.0)
    inf_freqs = freqs[is_inf]
    queue_freqs = freqs[~is_inf]
    ax.scatter(inf_idx, inf_freqs, s=6, alpha=0.7, color=CLR_INF, label="Inference", zorder=3)
    ax.scatter(queue_idx, queue_freqs, s=6, alpha=0.7, color=CLR_QUEUE, label="Queue replay", zorder=3)
    ax.axhline(target_hz, color=CLR_TARGET, ls="--", lw=1.2, label=f"Target ({target_hz:.0f} Hz)")
    ax.set_xlabel("Step index")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Step Frequency Timeline")
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
