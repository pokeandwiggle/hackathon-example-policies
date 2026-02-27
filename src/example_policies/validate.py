#!/usr/bin/env python

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

'''
nix run ./nix#bazelisk -- run //example_policies:validate -- --checkpoint <checkpoint_path> --dataset <dataset_path>
'''

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from example_policies.data_ops.utils.rotation_6d import (
    compose_transform_6d_torch,
    quat_to_6d_torch,
    rotation_6d_to_quat_torch,
)
from example_policies.robot_deploy.deploy_core.policy_loader import load_policy
from example_policies.utils.action_order import (
    ActionMode,
    DUAL_ABS_LEFT_POS_IDXS,
    DUAL_ABS_LEFT_QUAT_IDXS,
    DUAL_ABS_RIGHT_POS_IDXS,
    DUAL_ABS_RIGHT_QUAT_IDXS,
    UMI_LEFT_POS_IDXS,
    UMI_LEFT_ROT6D_IDXS,
    UMI_RIGHT_POS_IDXS,
    UMI_RIGHT_ROT6D_IDXS,
    UMI_LEFT_GRIPPER_IDX,
    UMI_RIGHT_GRIPPER_IDX,
    UMI_ACTION_DIM,
)
from example_policies.utils.chunk_relative_processor import abs_tcp_to_chunk_relative_umi_delta


def to_device_batch(batch: dict, device: torch.device, non_blocking: bool = True):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


def _match_quaternion_sign(pred_quat: torch.Tensor, gt_quat: torch.Tensor) -> torch.Tensor:
    """Flip predicted quaternion sign to match ground truth (closest hemisphere)."""
    gt = gt_quat.to(pred_quat.device)
    if torch.dot(pred_quat, gt) < 0:
        return -pred_quat
    return pred_quat


def umi_delta_to_abs_tcp(
    umi_action: torch.Tensor,
    ref_state: torch.Tensor,
    tcp_indices: dict,
) -> torch.Tensor:
    """Convert a single 20-dim UMI-delta action to 16-dim abs TCP."""
    umi = umi_action.squeeze(0) if umi_action.dim() > 1 else umi_action
    dev = umi.device
    abs_tcp = torch.zeros(16, device=dev, dtype=umi.dtype)

    for side, pos_sl, rot6d_sl, abs_pos_sl, abs_quat_sl, grip_idx in [
        ("l", UMI_LEFT_POS_IDXS, UMI_LEFT_ROT6D_IDXS,
         DUAL_ABS_LEFT_POS_IDXS, DUAL_ABS_LEFT_QUAT_IDXS, UMI_LEFT_GRIPPER_IDX),
        ("r", UMI_RIGHT_POS_IDXS, UMI_RIGHT_ROT6D_IDXS,
         DUAL_ABS_RIGHT_POS_IDXS, DUAL_ABS_RIGHT_QUAT_IDXS, UMI_RIGHT_GRIPPER_IDX),
    ]:
        ref_pos = ref_state[tcp_indices[f"pos_{side}"]].to(dev)
        ref_quat = ref_state[tcp_indices[f"quat_{side}"]].to(dev)
        ref_rot6d = quat_to_6d_torch(ref_quat)

        delta_pos = umi[pos_sl]
        delta_rot6d = umi[rot6d_sl]

        abs_pos, abs_rot6d = compose_transform_6d_torch(
            ref_pos, ref_rot6d, delta_pos, delta_rot6d,
        )
        abs_quat = rotation_6d_to_quat_torch(abs_rot6d)
        abs_quat = abs_quat / abs_quat.norm()

        abs_tcp[abs_pos_sl] = abs_pos
        abs_tcp[abs_quat_sl] = abs_quat

    abs_tcp[14] = umi[UMI_LEFT_GRIPPER_IDX]
    abs_tcp[15] = umi[UMI_RIGHT_GRIPPER_IDX]
    return abs_tcp


def parse_args():
    parser = argparse.ArgumentParser(description="Validate policy with action plot for all episodes")
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to the policy checkpoint directory",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output path for the figure (default: validation_actions_<N>_episodes.png in script directory)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=10,
        metavar="N",
        help="Maximum number of episodes to plot (default: 10)",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Select your device
    device = "cpu" if not torch.cuda.is_available() else "cuda"

    policy, cfg, preprocessor, postprocessor = load_policy(args.checkpoint)
    policy.to(device)

    # Detect UMI-delta action mode
    action_mode = ActionMode.parse_action_mode(cfg)
    is_umi_delta = action_mode == ActionMode.UMI_DELTA_TCP

    tcp_indices = None
    if is_umi_delta:
        state_names = cfg.metadata["features"]["observation.state"]["names"]

        def _find_idxs(prefix, count):
            start = state_names.index(prefix)
            return list(range(start, start + count))

        tcp_indices = {
            "pos_l": _find_idxs("tcp_left_pos_x", 3),
            "quat_l": _find_idxs("tcp_left_quat_x", 4),
            "pos_r": _find_idxs("tcp_right_pos_x", 3),
            "quat_r": _find_idxs("tcp_right_quat_x", 4),
        }
        print("UMI-delta model detected — predictions will be converted to abs TCP")

    dataset = LeRobotDataset(
        repo_id=args.dataset,
        root=args.dataset,
        video_backend="pyav",
    )

    # Dictionary to store data for each episode
    episodes_data = {}
    action_dim = None
    prev_ep = -1
    chunk_ref_state = None

    # Iterate samples directly (no DataLoader) so images stay (C, H, W)
    # matching the deployment pattern that select_action expects.
    for idx in range(len(dataset)):
        sample = dataset[idx]
        b_ep = int(sample["episode_index"].item())

        # Skip if we've reached max episodes
        if args.max_episodes is not None and b_ep >= args.max_episodes:
            break

        # Reset policy when starting a new episode
        if b_ep != prev_ep:
            policy.reset()
            prev_ep = b_ep

        # Initialize episode data structure if needed
        if b_ep not in episodes_data:
            episodes_data[b_ep] = {
                "targets": [],
                "preds": [],
                "times": [],
                "targets_6d": [],
                "preds_6d": [],
                "chunk_boundary_times": [],
            }

        sample = to_device_batch(
            sample, device, non_blocking=True
        )

        tgt = sample["action"].detach().float().view(-1)
        if action_dim is None:
            action_dim = tgt.numel()

        # Remove action from batch before preprocessing — during deployment
        # the observation dict never contains actions.
        obs = {k: v for k, v in sample.items() if k != "action"}
        
        # Detect whether a new action chunk will be predicted this step.
        _queues = getattr(policy, "_queues", None)
        is_new_chunk = (
            _queues is not None and len(_queues.get("action", [])) == 0
        )

        # Capture the raw observation.state at chunk boundaries for UMI-delta
        if is_umi_delta and is_new_chunk:
            chunk_ref_state = sample["observation.state"].detach().float().view(-1)
            t_boundary = float(sample["timestamp"].detach().cpu().item())
            episodes_data[b_ep]["chunk_boundary_times"].append(t_boundary)

        # Apply preprocessor if available (normalization)
        if preprocessor is not None:
            obs = preprocessor(obs)

        action = policy.select_action(
            obs
        )  # This is the output of the action chunk. Could be more or less.
        
        # Apply postprocessor if available (unnormalization)
        # For stepwise unnormalizers we must unnormalize the full chunk at once
        # so each timestep gets its correct per-step stats (p02[k], p98[k]).
        if postprocessor is not None:
            if is_new_chunk and _queues is not None:
                queue = policy._queues["action"]
                remaining = list(queue)
                full_chunk = torch.stack([action] + remaining, dim=1)
                full_chunk = postprocessor(full_chunk)
                action = full_chunk[:, 0]
                queue.clear()
                for t in range(1, full_chunk.shape[1]):
                    queue.append(full_chunk[:, t])
            elif not is_new_chunk and _queues is not None:
                # Already unnormalized when the chunk was processed.
                pass
            else:
                # No queue (e.g. ACT) — unnormalize individually.
                action = postprocessor(action)
        
        pred = action.detach().float().view(-1)

        # Convert UMI-delta predictions (20-dim) to absolute TCP (16-dim)
        if is_umi_delta:
            # Keep the raw UMI-delta (6D) for the second plot
            pred_6d = pred.clone().cpu()

            # Also convert GT abs TCP to 6D for comparison
            ref = chunk_ref_state
            ref_pos_l = ref[tcp_indices["pos_l"]]
            ref_quat_l = ref[tcp_indices["quat_l"]]
            ref_rot6d_l = quat_to_6d_torch(ref_quat_l)
            ref_pos_r = ref[tcp_indices["pos_r"]]
            ref_quat_r = ref[tcp_indices["quat_r"]]
            ref_rot6d_r = quat_to_6d_torch(ref_quat_r)
            tgt_16 = tgt.unsqueeze(0).unsqueeze(0)  # (1,1,16)
            tgt_6d = abs_tcp_to_chunk_relative_umi_delta(
                tgt_16, ref_pos_l, ref_rot6d_l, ref_pos_r, ref_rot6d_r,
            ).squeeze(0).squeeze(0).cpu()  # (20,)

            episodes_data[b_ep]["targets_6d"].append(tgt_6d)
            episodes_data[b_ep]["preds_6d"].append(pred_6d)

            pred = umi_delta_to_abs_tcp(pred, chunk_ref_state, tcp_indices)
            pred[DUAL_ABS_LEFT_QUAT_IDXS] = _match_quaternion_sign(
                pred[DUAL_ABS_LEFT_QUAT_IDXS], tgt[DUAL_ABS_LEFT_QUAT_IDXS]
            )
            pred[DUAL_ABS_RIGHT_QUAT_IDXS] = _match_quaternion_sign(
                pred[DUAL_ABS_RIGHT_QUAT_IDXS], tgt[DUAL_ABS_RIGHT_QUAT_IDXS]
            )

        # collect for this episode
        episodes_data[b_ep]["targets"].append(tgt.cpu())
        episodes_data[b_ep]["preds"].append(pred.cpu())

        # append the time
        t = float(sample["timestamp"].detach().cpu().item())
        episodes_data[b_ep]["times"].append(t)

    # Stack data for each episode
    for ep_idx in episodes_data:
        episodes_data[ep_idx]["targets"] = torch.stack(episodes_data[ep_idx]["targets"], dim=0)  # [T, D]
        episodes_data[ep_idx]["preds"] = torch.stack(episodes_data[ep_idx]["preds"], dim=0)  # [T, D]
        episodes_data[ep_idx]["times"] = torch.tensor(episodes_data[ep_idx]["times"])  # [T]
        if episodes_data[ep_idx]["targets_6d"]:
            episodes_data[ep_idx]["targets_6d"] = torch.stack(episodes_data[ep_idx]["targets_6d"], dim=0)
            episodes_data[ep_idx]["preds_6d"] = torch.stack(episodes_data[ep_idx]["preds_6d"], dim=0)

    num_episodes = len(episodes_data)
    if num_episodes == 0:
        print("No episodes found in dataset.")
        return

    # --- Print per-dimension metrics ---
    action_names = cfg.metadata.get("features", {}).get("action", {}).get("names", None)
    all_targets = torch.cat([episodes_data[e]["targets"] for e in sorted(episodes_data)], dim=0)
    all_preds = torch.cat([episodes_data[e]["preds"] for e in sorted(episodes_data)], dim=0)
    D_act = all_targets.shape[1]
    print(f"\n{'='*70}")
    print(f"Per-dimension MAE ({num_episodes} episodes, T={all_targets.shape[0]} total frames):")
    print(f"{'dim':>4}  {'label':>20}  {'MAE':>10}  {'GT_range':>10}  {'MAE/range':>10}")
    for d in range(D_act):
        label = action_names[d] if action_names and d < len(action_names) else f"dim_{d}"
        mae = (all_targets[:, d] - all_preds[:, d]).abs().mean().item()
        gt_range = all_targets[:, d].max().item() - all_targets[:, d].min().item()
        ratio = mae / gt_range if gt_range > 1e-8 else float('inf')
        print(f"  [{d:2d}] {label:>20}  {mae:>10.6f}  {gt_range:>10.6f}  {ratio:>10.2%}")
    print(f"{'='*70}\n")

    # Get action dimension from first episode
    first_ep = list(episodes_data.keys())[0]
    D = episodes_data[first_ep]["targets"].shape[1]

    # Create plots
    fig, axes = plt.subplots(D, 1, figsize=(10, 2.2 * D), sharex=True)
    if D == 1:
        axes = [axes]

    # Plot each episode
    for ep_idx in sorted(episodes_data.keys()):
        ep_data = episodes_data[ep_idx]
        times = ep_data["times"].numpy()
        targets = ep_data["targets"].numpy()
        preds = ep_data["preds"].numpy()

        # Use matplotlib's default color cycle for each episode
        color = f"C{ep_idx % 10}"  # Cycle through colors if more than 10 episodes

        for d in range(D):
            ax = axes[d]
            # Plot target and prediction for this episode
            ax.plot(times, targets[:, d], color=color, alpha=0.7, label=f"Ep {ep_idx} Target", linestyle='-')
            ax.plot(times, preds[:, d], color=color, alpha=0.7, label=f"Ep {ep_idx} Pred", linestyle='--')
            dim_label = action_names[d] if action_names and d < len(action_names) else f"dim {d}"
            ax.set_ylabel(dim_label, fontsize=7)
            ax.grid(True, linestyle="--", alpha=0.3)
            if d == 0:
                title = f"Episodes ({num_episodes} total): ground truth actions vs. predictions"
                if is_umi_delta:
                    title += "  (UMI-delta → abs TCP)"
                ax.set_title(title)

    axes[-1].set_xlabel("time (s)")

    # Add legend
    if num_episodes <= 3:
        axes[0].legend(loc="upper right", fontsize='small')
    else:
        # For many episodes, add a simplified legend
        handles, labels = axes[0].get_legend_handles_labels()
        # Only show first few in legend to avoid clutter
        fig.legend(handles[:6], labels[:6], loc="upper right", fontsize='small')

    plt.tight_layout(rect=[0, 0, 0.98, 0.98])

    if args.output:
        output_path = args.output
    else:
        # Try to find the actual source directory, fall back to current working directory
        # When run via Bazel, BUILD_WORKSPACE_DIRECTORY points to the actual workspace
        workspace_dir = os.environ.get('BUILD_WORKSPACE_DIRECTORY')
        if workspace_dir:
            output_dir = Path(workspace_dir) / "example_policies" / "src" / "example_policies"
        else:
            # When run directly, use the script's directory
            output_dir = Path(__file__).parent.resolve()
        
        output_path = output_dir / f"validation_actions_{args.max_episodes}_episodes.png"
    
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved abs-TCP plot with {num_episodes} episodes to {output_path}")

    # --- Second plot: 6D UMI-delta space (no quat conversion jitter) ---
    if is_umi_delta:
        umi_dim_names = [
            "L_dx", "L_dy", "L_dz",
            "L_r6d_0", "L_r6d_1", "L_r6d_2", "L_r6d_3", "L_r6d_4", "L_r6d_5",
            "R_dx", "R_dy", "R_dz",
            "R_r6d_0", "R_r6d_1", "R_r6d_2", "R_r6d_3", "R_r6d_4", "R_r6d_5",
            "grip_L", "grip_R",
        ]
        D6 = UMI_ACTION_DIM

        # Print 6D metrics
        all_t6 = torch.cat([episodes_data[e]["targets_6d"] for e in sorted(episodes_data)], dim=0)
        all_p6 = torch.cat([episodes_data[e]["preds_6d"] for e in sorted(episodes_data)], dim=0)
        print(f"\n{'='*70}")
        print(f"Per-dimension MAE in 6D UMI-delta space ({num_episodes} episodes, T={all_t6.shape[0]}):")
        print(f"{'dim':>4}  {'label':>12}  {'MAE':>10}  {'GT_range':>10}  {'MAE/range':>10}")
        for d in range(D6):
            mae = (all_t6[:, d] - all_p6[:, d]).abs().mean().item()
            gt_range = all_t6[:, d].max().item() - all_t6[:, d].min().item()
            ratio = mae / gt_range if gt_range > 1e-8 else float('inf')
            print(f"  [{d:2d}] {umi_dim_names[d]:>12}  {mae:>10.6f}  {gt_range:>10.6f}  {ratio:>10.2%}")
        print(f"{'='*70}\n")

        fig6, axes6 = plt.subplots(D6, 1, figsize=(10, 2.2 * D6), sharex=True)
        if D6 == 1:
            axes6 = [axes6]

        for ep_idx in sorted(episodes_data.keys()):
            ep_data = episodes_data[ep_idx]
            times = ep_data["times"].numpy()
            t6 = ep_data["targets_6d"].numpy()
            p6 = ep_data["preds_6d"].numpy()
            color = f"C{ep_idx % 10}"
            chunk_times = ep_data["chunk_boundary_times"]

            for d in range(D6):
                ax = axes6[d]
                ax.plot(times, t6[:, d], color=color, alpha=0.7, linestyle='-',
                        label=f"Ep {ep_idx} GT" if d == 0 else None)
                ax.plot(times, p6[:, d], color=color, alpha=0.7, linestyle='--',
                        label=f"Ep {ep_idx} Pred" if d == 0 else None)
                # Mark chunk boundaries with thin vertical lines
                for ct in chunk_times:
                    ax.axvline(ct, color='gray', alpha=0.3, linewidth=0.5, linestyle=':')
                ax.set_ylabel(umi_dim_names[d], fontsize=7)
                ax.grid(True, linestyle="--", alpha=0.3)
                if d == 0:
                    ax.set_title(f"6D UMI-delta space ({num_episodes} episodes): GT vs. predictions  (dotted lines = chunk boundaries)")

        axes6[-1].set_xlabel("time (s)")
        if num_episodes <= 3:
            axes6[0].legend(loc="upper right", fontsize='small')
        else:
            handles, labels = axes6[0].get_legend_handles_labels()
            fig6.legend(handles[:6], labels[:6], loc="upper right", fontsize='small')

        plt.tight_layout(rect=[0, 0, 0.98, 0.98])
        output_6d = output_path.parent / (output_path.stem + "_6d" + output_path.suffix)
        fig6.savefig(output_6d, dpi=150)
        plt.close(fig6)
        print(f"Saved 6D UMI-delta plot to {output_6d}")


if __name__ == "__main__":
    main()
