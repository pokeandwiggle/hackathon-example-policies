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
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from matplotlib.backends.backend_pdf import PdfPages


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize all episodes in a dataset with config info")
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
        help="Output path for the PDF file (default: dataset_visualization.pdf)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of episodes to plot (default: all)",
    )
    args = parser.parse_args()
    return args


def create_config_page(dataset, dataset_path):
    """Create a page with dataset configuration information."""
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Gather dataset information
    config_info = []
    config_info.append("=" * 80)
    config_info.append("DATASET CONFIGURATION")
    config_info.append("=" * 80)
    config_info.append("")
    
    # Basic info
    config_info.append(f"Dataset Path: {dataset_path}")
    config_info.append(f"Total Frames: {len(dataset)}")
    
    # Episode info
    if hasattr(dataset, 'episode_data_index'):
        num_episodes = len(dataset.episode_data_index['from'])
        config_info.append(f"Total Episodes: {num_episodes}")
        config_info.append("")
        
        # Episode lengths
        config_info.append("Episode Lengths:")
        for ep_idx in range(min(num_episodes, 20)):  # Show first 20 episodes
            start = dataset.episode_data_index['from'][ep_idx]
            end = dataset.episode_data_index['to'][ep_idx]
            length = end - start
            config_info.append(f"  Episode {ep_idx}: {length} frames")
        if num_episodes > 20:
            config_info.append(f"  ... and {num_episodes - 20} more episodes")
        config_info.append("")
    
    # Features/Keys
    if hasattr(dataset, 'features'):
        config_info.append("Dataset Features:")
        for key, feature in dataset.features.items():
            config_info.append(f"  {key}: {feature}")
        config_info.append("")
    
    # Try to load and display meta information
    meta_path = dataset_path / "meta" / "info.json"
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta_info = json.load(f)
            config_info.append("Meta Information:")
            for key, value in meta_info.items():
                if isinstance(value, dict):
                    config_info.append(f"  {key}:")
                    for k, v in value.items():
                        config_info.append(f"    {k}: {v}")
                else:
                    config_info.append(f"  {key}: {value}")
            config_info.append("")
        except Exception as e:
            config_info.append(f"Could not load meta info: {e}")
            config_info.append("")
    
    # Dataset config
    config_path = dataset_path / "meta" / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                dataset_config = json.load(f)
            config_info.append("Dataset Config:")
            config_json = json.dumps(dataset_config, indent=2)
            config_info.extend(config_json.split('\n'))
        except Exception as e:
            config_info.append(f"Could not load config: {e}")
    
    # Display text
    text = '\n'.join(config_info)
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=8,
            verticalalignment='top')
    
    return fig


def main():
    args = parse_args()

    dataset = LeRobotDataset(
        repo_id=args.dataset,
        root=args.dataset,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    # Dictionary to store data for each episode
    episodes_data = {}
    action_dim = None
    observation_keys = set()

    print("Loading dataset...")
    
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
                "actions": [],
                "times": [],
                "observations": {},
            }

        # Collect action
        action = batch["action"].detach().float().view(-1)
        if action_dim is None:
            action_dim = action.numel()
        
        episodes_data[b_ep]["actions"].append(action.cpu())

        # Collect timestamp
        t = float(batch["timestamp"].view(-1)[0].detach().cpu().item())
        episodes_data[b_ep]["times"].append(t)
        
        # Collect observations (if they are numeric)
        for key in batch.keys():
            if key.startswith("observation.") and torch.is_tensor(batch[key]):
                obs_value = batch[key].detach().float().view(-1)
                observation_keys.add(key)
                if key not in episodes_data[b_ep]["observations"]:
                    episodes_data[b_ep]["observations"][key] = []
                episodes_data[b_ep]["observations"][key].append(obs_value.cpu())

    print(f"Loaded {len(episodes_data)} episodes")

    # Stack data for each episode
    for ep_idx in episodes_data:
        episodes_data[ep_idx]["actions"] = torch.stack(episodes_data[ep_idx]["actions"], dim=0)  # [T, D]
        episodes_data[ep_idx]["times"] = torch.tensor(episodes_data[ep_idx]["times"])  # [T]
        for obs_key in episodes_data[ep_idx]["observations"]:
            episodes_data[ep_idx]["observations"][obs_key] = torch.stack(
                episodes_data[ep_idx]["observations"][obs_key], dim=0
            )

    num_episodes = len(episodes_data)
    if num_episodes == 0:
        print("No episodes found in dataset.")
        return

    # Get action dimension from first episode
    first_ep = list(episodes_data.keys())[0]
    D = episodes_data[first_ep]["actions"].shape[1]

    # Create PDF with multiple pages
    output_path = args.output or Path("./dataset_visualization.pdf")
    
    with PdfPages(output_path) as pdf:
        # Page 1: Dataset configuration
        print("Creating configuration page...")
        config_fig = create_config_page(dataset, args.dataset)
        pdf.savefig(config_fig, bbox_inches='tight')
        plt.close(config_fig)
        
        # Page 2: Action trajectories
        print("Creating action trajectories plot...")
        fig, axes = plt.subplots(D, 1, figsize=(11, 2.2 * D), sharex=True)
        if D == 1:
            axes = [axes]

        # Plot each episode's actions
        for ep_idx in sorted(episodes_data.keys()):
            ep_data = episodes_data[ep_idx]
            times = ep_data["times"].numpy()
            actions = ep_data["actions"].numpy()

            for d in range(D):
                ax = axes[d]
                ax.plot(times, actions[:, d], alpha=0.6, label=f"Episode {ep_idx}")
                ax.set_ylabel(f"Action dim {d}")
                ax.grid(True, linestyle="--", alpha=0.3)
                if d == 0:
                    ax.set_title(f"Action Trajectories - All Episodes ({num_episodes} total)")

        axes[-1].set_xlabel("Time (s)")

        # Add legend
        if num_episodes <= 10:
            for d in range(D):
                axes[d].legend(loc="best", fontsize='small', ncol=2)
        else:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles[:10], labels[:10], loc="upper right", fontsize='small')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3+: Observation trajectories (if available)
        for obs_key in sorted(observation_keys):
            print(f"Creating plot for {obs_key}...")
            # Get observation dimension
            obs_dim = episodes_data[first_ep]["observations"][obs_key].shape[1]
            
            fig, axes = plt.subplots(obs_dim, 1, figsize=(11, 2.2 * obs_dim), sharex=True)
            if obs_dim == 1:
                axes = [axes]
            
            # Plot each episode's observations
            for ep_idx in sorted(episodes_data.keys()):
                ep_data = episodes_data[ep_idx]
                times = ep_data["times"].numpy()
                obs_data = ep_data["observations"][obs_key].numpy()
                
                for d in range(obs_dim):
                    ax = axes[d]
                    ax.plot(times, obs_data[:, d], alpha=0.6, label=f"Episode {ep_idx}")
                    ax.set_ylabel(f"dim {d}")
                    ax.grid(True, linestyle="--", alpha=0.3)
                    if d == 0:
                        ax.set_title(f"{obs_key} - All Episodes ({num_episodes} total)")
            
            axes[-1].set_xlabel("Time (s)")
            
            # Add legend
            if num_episodes <= 10:
                for d in range(obs_dim):
                    axes[d].legend(loc="best", fontsize='small', ncol=2)
            else:
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles[:10], labels[:10], loc="upper right", fontsize='small')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'Dataset Visualization'
        d['Author'] = 'Poke & Wiggle'
        d['Subject'] = f'Visualization of dataset: {args.dataset}'
        d['Keywords'] = 'LeRobot Dataset Visualization'
        d['CreationDate'] = None  # Will use current date

    print(f"Saved visualization with {num_episodes} episodes to {output_path}")


if __name__ == "__main__":
    main()
