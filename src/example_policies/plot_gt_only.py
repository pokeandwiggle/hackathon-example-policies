#!/usr/bin/env python
"""Plot ground-truth actions only (no model needed).

Produces two plots:
  1. Absolute TCP (16-dim) — raw dataset actions
  2. Chunk-relative UMI-delta (20-dim) — after conversion

Usage:
    python -m example_policies.plot_gt_only \
        --dataset data/lerobot/stack_one_brick_30hz \
        --max-episodes 10 \
        --horizon 144
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from example_policies.data_ops.utils.rotation_6d import quat_to_6d_torch
from example_policies.utils.action_order import UMI_ACTION_DIM
from example_policies.utils.chunk_relative_processor import (
    abs_tcp_to_chunk_relative_umi_delta,
)


def main():
    parser = argparse.ArgumentParser(description="Plot GT actions only")
    parser.add_argument("-d", "--dataset", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=144,
                        help="Chunk horizon for UMI-delta conversion (default: 144 for 30hz)")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    dataset = LeRobotDataset(
        repo_id=args.dataset,
        root=args.dataset,
        video_backend="pyav",
    )

    # Discover feature names
    info = dataset.meta.info
    action_names = info.get("features", {}).get("action", {}).get("names", [])
    state_names = info.get("features", {}).get("observation.state", {}).get("names", [])

    def _find_idxs(names_list, prefix, count):
        start = names_list.index(prefix)
        return list(range(start, start + count))

    tcp_idx = {
        "pos_l": _find_idxs(state_names, "tcp_left_pos_x", 3),
        "quat_l": _find_idxs(state_names, "tcp_left_quat_x", 4),
        "pos_r": _find_idxs(state_names, "tcp_right_pos_x", 3),
        "quat_r": _find_idxs(state_names, "tcp_right_quat_x", 4),
    }

    # Collect data per episode
    episodes = {}
    prev_ep = -1
    chunk_start_idx = {}  # track chunk boundaries per episode

    for idx in range(len(dataset)):
        sample = dataset[idx]
        ep = int(sample["episode_index"].item())

        if args.max_episodes is not None and ep >= args.max_episodes:
            break

        if ep != prev_ep:
            prev_ep = ep
            chunk_start_idx[ep] = 0

        if ep not in episodes:
            episodes[ep] = {"actions": [], "states": [], "times": []}

        episodes[ep]["actions"].append(sample["action"].float())
        episodes[ep]["states"].append(sample["observation.state"].float())
        episodes[ep]["times"].append(float(sample["timestamp"].item()))

    # Stack
    for ep in episodes:
        episodes[ep]["actions"] = torch.stack(episodes[ep]["actions"])  # (T, D)
        episodes[ep]["states"] = torch.stack(episodes[ep]["states"])    # (T, D_state)
        episodes[ep]["times"] = np.array(episodes[ep]["times"])

    num_eps = len(episodes)
    if num_eps == 0:
        print("No episodes found.")
        return

    D = episodes[list(episodes.keys())[0]]["actions"].shape[1]

    # ── Plot 1: absolute TCP ──
    fig, axes = plt.subplots(D, 1, figsize=(12, 2 * D), sharex=True)
    if D == 1:
        axes = [axes]

    for ep in sorted(episodes):
        times = episodes[ep]["times"]
        acts = episodes[ep]["actions"].numpy()
        color = f"C{ep % 10}"
        for d in range(D):
            axes[d].plot(times, acts[:, d], color=color, alpha=0.6, linewidth=0.8)
            if ep == 0:
                label = action_names[d] if d < len(action_names) else f"dim_{d}"
                axes[d].set_ylabel(label, fontsize=7)
                axes[d].grid(True, linestyle="--", alpha=0.3)

    axes[0].set_title(f"Ground Truth — Absolute TCP ({num_eps} episodes)")
    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()

    out_dir = args.output or Path(__file__).parent.resolve()
    path_tcp = out_dir / f"gt_abs_tcp_{num_eps}ep.png"
    fig.savefig(path_tcp, dpi=150)
    plt.close(fig)
    print(f"Saved abs TCP plot → {path_tcp}")

    # ── Plot 2: chunk-relative UMI-delta ──
    umi_names = [
        "L_dx", "L_dy", "L_dz",
        "L_r6d_0", "L_r6d_1", "L_r6d_2", "L_r6d_3", "L_r6d_4", "L_r6d_5",
        "R_dx", "R_dy", "R_dz",
        "R_r6d_0", "R_r6d_1", "R_r6d_2", "R_r6d_3", "R_r6d_4", "R_r6d_5",
        "grip_L", "grip_R",
    ]
    D6 = UMI_ACTION_DIM
    horizon = args.horizon

    fig6, axes6 = plt.subplots(D6, 1, figsize=(12, 2 * D6), sharex=True)
    if D6 == 1:
        axes6 = [axes6]

    for ep in sorted(episodes):
        times = episodes[ep]["times"]
        acts = episodes[ep]["actions"]   # (T, 16)
        states = episodes[ep]["states"]  # (T, D_state)
        T = acts.shape[0]
        color = f"C{ep % 10}"

        # Convert each frame using its chunk-start reference
        umi_all = torch.zeros(T, D6)
        chunk_boundaries = []

        for t in range(T):
            # Determine chunk boundary: every `horizon` steps, or first frame
            if t % horizon == 0:
                ref_state = states[t]
                ref_pos_l = ref_state[tcp_idx["pos_l"]]
                ref_quat_l = ref_state[tcp_idx["quat_l"]]
                ref_rot6d_l = quat_to_6d_torch(ref_quat_l)
                ref_pos_r = ref_state[tcp_idx["pos_r"]]
                ref_quat_r = ref_state[tcp_idx["quat_r"]]
                ref_rot6d_r = quat_to_6d_torch(ref_quat_r)
                chunk_boundaries.append(times[t])

            # Convert this single action to UMI-delta
            act_16 = acts[t].unsqueeze(0).unsqueeze(0)  # (1,1,16)
            umi_1 = abs_tcp_to_chunk_relative_umi_delta(
                act_16, ref_pos_l, ref_rot6d_l, ref_pos_r, ref_rot6d_r,
            ).squeeze(0).squeeze(0)  # (20,)
            umi_all[t] = umi_1

        umi_np = umi_all.numpy()
        for d in range(D6):
            axes6[d].plot(times, umi_np[:, d], color=color, alpha=0.6, linewidth=0.8)
            # Chunk boundaries
            for ct in chunk_boundaries:
                axes6[d].axvline(ct, color='gray', alpha=0.2, linewidth=0.5, linestyle=':')
            if ep == 0:
                axes6[d].set_ylabel(umi_names[d], fontsize=7)
                axes6[d].grid(True, linestyle="--", alpha=0.3)

    axes6[0].set_title(
        f"Ground Truth — Chunk-Relative UMI-Delta ({num_eps} episodes, horizon={horizon})"
    )
    axes6[-1].set_xlabel("time (s)")
    plt.tight_layout()

    path_6d = out_dir / f"gt_umi_delta_{num_eps}ep.png"
    fig6.savefig(path_6d, dpi=150)
    plt.close(fig6)
    print(f"Saved UMI-delta plot → {path_6d}")


if __name__ == "__main__":
    main()
