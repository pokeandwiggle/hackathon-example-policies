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

import argparse
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
)


def to_device_batch(batch: dict, device: torch.device, non_blocking: bool = True):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


def _match_quaternion_sign(pred_quat: torch.Tensor, gt_quat: torch.Tensor) -> torch.Tensor:
    """Flip predicted quaternion sign to match ground truth (closest hemisphere).

    Quaternions q and -q represent the same rotation.  Pick the sign of
    ``pred_quat`` closest to ``gt_quat`` (positive dot product) to avoid
    misleading jumps in validation plots.
    """
    gt = gt_quat.to(pred_quat.device)
    if torch.dot(pred_quat, gt) < 0:
        return -pred_quat
    return pred_quat


def umi_delta_to_abs_tcp(
    umi_action: torch.Tensor,
    ref_state: torch.Tensor,
    tcp_indices: dict,
) -> torch.Tensor:
    """Convert a single 20-dim UMI-delta action to 16-dim abs TCP.

    Args:
        umi_action: UMI-delta action, shape (20,) or (1, 20).
        ref_state: Raw observation.state at chunk start, shape (state_dim,).
        tcp_indices: Dict with keys pos_l, quat_l, pos_r, quat_r mapping to
            observation.state index lists.

    Returns:
        Absolute TCP action, shape (16,).
    """
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
        abs_quat = abs_quat / abs_quat.norm()  # ensure unit quaternion

        abs_tcp[abs_pos_sl] = abs_pos
        abs_tcp[abs_quat_sl] = abs_quat

    # grippers pass through
    abs_tcp[14] = umi[UMI_LEFT_GRIPPER_IDX]
    abs_tcp[15] = umi[UMI_RIGHT_GRIPPER_IDX]
    return abs_tcp


def parse_args():
    parser = argparse.ArgumentParser(description="Validate policy with action plot")
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
        "-e",
        "--episode",
        type=int,
        default=0,
        metavar="N",
        help="Episode index to compare (default: 0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output path for the figure (default: actions_episode<E>.png)",
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        default="pyav",
        choices=["torchcodec", "pyav"],
        help="Video decoding backend (default: pyav)",
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

    # For UMI-delta: extract TCP pose indices from observation.state
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
        print(f"UMI-delta model detected — predictions will be converted to abs TCP")

    dataset = LeRobotDataset(
        repo_id=args.dataset,
        root=args.dataset,
        video_backend=args.video_backend,
    )

    ep = args.episode
    targets = []
    preds = []
    times = []
    action_dim = None
    episode_started = False

    # Track the observation.state at each chunk boundary for UMI-delta
    chunk_ref_state = None

    # Iterate samples directly (no DataLoader) so images stay (C, H, W)
    # matching the deployment pattern that select_action expects.
    for idx in range(len(dataset)):
        sample = dataset[idx]
        b_ep = int(sample["episode_index"].item())

        if b_ep < ep:
            continue
        if b_ep > ep:
            break

        # Reset policy at the start of the target episode
        if not episode_started:
            policy.reset()
            episode_started = True

        sample = to_device_batch(sample, device, non_blocking=True)

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

        # Apply preprocessor if available (normalization)
        if preprocessor is not None:
            obs = preprocessor(obs)

        action = policy.select_action(obs)

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
            pred = umi_delta_to_abs_tcp(pred, chunk_ref_state, tcp_indices)
            # Match quaternion signs to avoid misleading jumps in plots
            pred[DUAL_ABS_LEFT_QUAT_IDXS] = _match_quaternion_sign(
                pred[DUAL_ABS_LEFT_QUAT_IDXS], tgt[DUAL_ABS_LEFT_QUAT_IDXS]
            )
            pred[DUAL_ABS_RIGHT_QUAT_IDXS] = _match_quaternion_sign(
                pred[DUAL_ABS_RIGHT_QUAT_IDXS], tgt[DUAL_ABS_RIGHT_QUAT_IDXS]
            )

        # collect
        targets.append(tgt.cpu())
        preds.append(pred.cpu())

        # append the time
        t = float(sample["timestamp"].detach().cpu().item())
        times.append(t)

    # stack T x D
    targets = torch.stack(targets, dim=0)  # [T, D]
    preds = torch.stack(preds, dim=0)  # [T, D]
    times = torch.tensor(times)  # [T]

    # --- Print per-dimension metrics ---
    action_names = cfg.metadata.get("features", {}).get("action", {}).get("names", None)
    print(f"\n{'='*70}")
    print(f"Per-dimension MAE (episode {ep}, T={targets.shape[0]}):")
    print(f"{'dim':>4}  {'label':>20}  {'MAE':>10}  {'GT_range':>10}  {'MAE/range':>10}")
    for d in range(targets.shape[1]):
        label = action_names[d] if action_names and d < len(action_names) else f"dim_{d}"
        mae = (targets[:, d] - preds[:, d]).abs().mean().item()
        gt_range = targets[:, d].max().item() - targets[:, d].min().item()
        ratio = mae / gt_range if gt_range > 1e-8 else float('inf')
        print(f"  [{d:2d}] {label:>20}  {mae:>10.6f}  {gt_range:>10.6f}  {ratio:>10.2%}")
    print(f"{'='*70}\n")

    T, D = targets.shape
    fig, axes = plt.subplots(D, 1, figsize=(8, 2.2 * D), sharex=True)
    if D == 1:
        axes = [axes]

    for d in range(D):
        ax = axes[d]
        ax.plot(times, targets[:, d].numpy(), label="Target")
        ax.plot(times, preds[:, d].numpy(), label="Pred")
        dim_label = action_names[d] if action_names and d < len(action_names) else f"dim {d}"
        ax.set_ylabel(dim_label, fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)
        if d == 0:
            title = f"Episode {ep}: action targets vs. predictions"
            if is_umi_delta:
                title += "  (UMI-delta → abs TCP)"
            ax.set_title(title)
    axes[-1].set_xlabel("time (s)")

    # single legend outside if many dims
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout(rect=[0, 0, 0.98, 0.98])

    output_path = args.output or Path(f"./actions_episode{args.episode}.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved continuous plot to {output_path}")


if __name__ == "__main__":
    main()
