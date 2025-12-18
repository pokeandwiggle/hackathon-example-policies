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

from example_policies.robot_deploy.deploy_core.policy_loader import load_policy


def to_device_batch(batch: dict, device: torch.device, non_blocking: bool = True):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Select your device
    device = "cpu" if not torch.cuda.is_available() else "cuda"

    policy, cfg = load_policy(args.checkpoint)
    policy.to(device)

    dataset = LeRobotDataset(
        repo_id=args.dataset,
        root=args.dataset,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=1,
        shuffle=False,  # Not shuffling, so processing sequential batches from the dataset
        pin_memory=device != "cpu",
        drop_last=False,
    )

    ep = args.episode
    targets = []
    preds = []
    times = []
    action_dim = None

    def _fmt(v: torch.Tensor, w=9, p=3):
        return f"{float(v):{w}.{p}f}"

    for batch in dataloader:
        b_ep = batch.get("episode_index")
        if b_ep is None:
            raise KeyError("Expected key 'episode_index' in batch.")
        b_ep = int(b_ep.view(-1)[0].item())

        if b_ep < ep:
            continue
        if b_ep > ep:
            break

        batch = to_device_batch(
            batch, device, non_blocking=True
        )  # Push all tensors of the batch to the GPU

        tgt = batch["action"].detach().float().view(-1)
        if action_dim is None:
            action_dim = tgt.numel()  # Only load the action dimension once
        action = policy.select_action(
            batch
        )  # This is the output of the action chunk. Could be more or less.
        pred = action.detach().float().view(-1)

        # collect
        targets.append(tgt.cpu())
        preds.append(pred.cpu())

        # append the time
        t = float(batch["timestamp"].view(-1)[0].detach().cpu().item())
        times.append(t)

    # stack T x D
    targets = torch.stack(targets, dim=0)  # [T, D]
    preds = torch.stack(preds, dim=0)  # [T, D]
    times = torch.tensor(times)  # [T]

    T, D = targets.shape
    fig, axes = plt.subplots(D, 1, figsize=(8, 2.2 * D), sharex=True)
    if D == 1:
        axes = [axes]

    for d in range(D):
        ax = axes[d]
        ax.plot(times, targets[:, d].numpy(), label="Target")
        ax.plot(times, preds[:, d].numpy(), label="Pred")
        ax.set_ylabel(f"dim {d}")
        ax.grid(True, linestyle="--", alpha=0.3)
        if d == 0:
            ax.set_title(f"Episode {ep}: action targets vs. predictions")
    axes[-1].set_xlabel("time (s)" if "timestamp" in batch else "step")

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
