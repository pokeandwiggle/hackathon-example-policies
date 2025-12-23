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

    # Dictionary to store data for each episode
    episodes_data = {}
    action_dim = None

    # Collect data for all episodes
    for batch in dataloader:
        b_ep = batch.get("episode_index")
        if b_ep is None:
            raise KeyError("Expected key 'episode_index' in batch.")
        b_ep = int(b_ep.view(-1)[0].item())

        # Skip if we've reached max episodes
        if args.max_episodes is not None and b_ep >= args.max_episodes:
            break

        # Initialize episode data structure if needed
        if b_ep not in episodes_data:
            episodes_data[b_ep] = {
                "targets": [],
                "preds": [],
                "times": [],
            }

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

        # collect for this episode
        episodes_data[b_ep]["targets"].append(tgt.cpu())
        episodes_data[b_ep]["preds"].append(pred.cpu())

        # append the time
        t = float(batch["timestamp"].view(-1)[0].detach().cpu().item())
        episodes_data[b_ep]["times"].append(t)

    # Stack data for each episode
    for ep_idx in episodes_data:
        episodes_data[ep_idx]["targets"] = torch.stack(episodes_data[ep_idx]["targets"], dim=0)  # [T, D]
        episodes_data[ep_idx]["preds"] = torch.stack(episodes_data[ep_idx]["preds"], dim=0)  # [T, D]
        episodes_data[ep_idx]["times"] = torch.tensor(episodes_data[ep_idx]["times"])  # [T]

    num_episodes = len(episodes_data)
    if num_episodes == 0:
        print("No episodes found in dataset.")
        return

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
            ax.set_ylabel(f"dim {d}")
            ax.grid(True, linestyle="--", alpha=0.3)
            if d == 0:
                ax.set_title(f"Episodes ({num_episodes} total): ground truth actions vs. predictions")

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
    print(f"Saved plot with {num_episodes} episodes to {output_path}")


if __name__ == "__main__":
    main()
