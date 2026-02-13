#!/usr/bin/env python3

# Copyright 2025 Poke & Wiggle GmbH. All rights reserved.


"""
Plot starting XY positions and first grasp positions from a LeRobot dataset.

Usage: python plot_grasp_positions.py --data-dir /path/to/dataset [--arm left|right|both]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from example_policies.robot_deploy.deploy_core.policy_loader import load_metadata
from example_policies.utils.constants import BLACKLIST_FILE, META_DIR


def find_first_grasp(actions, gripper_idx, threshold=0.5):
    """
    Find the first frame where gripper closes (grasp action).

    Args:
        actions: Array of actions for the episode
        gripper_idx: Index of the gripper action in the action vector
        threshold: Threshold for considering gripper as closed (higher = more closed)

    Returns:
        Index of first grasp, or None if no grasp found
    """
    gripper_values = actions[:, gripper_idx]

    # Find where gripper crosses threshold (closing)
    # Typically gripper value > threshold means closed
    grasp_frames = np.where(gripper_values > threshold)[0]

    if len(grasp_frames) > 0:
        return grasp_frames[0]

    # Alternative: look for significant change in gripper position
    gripper_diff = np.abs(np.diff(gripper_values))
    significant_changes = np.where(gripper_diff > 0.3)[0]

    if len(significant_changes) > 0:
        return significant_changes[0] + 1

    return None


def extract_positions(dataset_path: Path, arm: str = "both", max_episodes: int = None):
    """
    Extract starting and first grasp positions from dataset.

    Args:
        dataset_path: Path to LeRobot dataset directory
        arm: Which arm to analyze ('left', 'right', or 'both')
        max_episodes: Maximum number of episodes to process (None = all)

    Returns:
        Dictionary with positions for each arm
    """
    print(f"Loading dataset from {dataset_path}...")

    # Load metadata first (following the pattern from robot_deploy)
    metadata = load_metadata(dataset_path)

    # Get feature names from metadata
    state_names = metadata["features"]["observation.state"]["names"]
    action_names = metadata["features"]["action"]["names"]

    # Now load the dataset
    dataset = LeRobotDataset(repo_id=dataset_path.name, root=dataset_path)

    # Load blacklist if it exists
    blacklist_path = dataset_path / META_DIR / BLACKLIST_FILE
    blacklist = []
    if blacklist_path.exists():
        with open(blacklist_path, "r", encoding="utf-8") as f:
            blacklist = json.load(f)
        print(f"Loaded blacklist with {len(blacklist)} episodes")

    print("\nDataset info:")
    print(f"  Total frames: {len(dataset)}")
    print(f"  Episodes: {dataset.num_episodes}")
    print(f"  State features: {state_names}")
    print(f"  Action features: {action_names}")

    # Find indices for TCP positions in state
    tcp_left_x_idx = (
        state_names.index("tcp_left_pos_x") if "tcp_left_pos_x" in state_names else None
    )
    tcp_left_y_idx = (
        state_names.index("tcp_left_pos_y") if "tcp_left_pos_y" in state_names else None
    )
    tcp_right_x_idx = (
        state_names.index("tcp_right_pos_x")
        if "tcp_right_pos_x" in state_names
        else None
    )
    tcp_right_y_idx = (
        state_names.index("tcp_right_pos_y")
        if "tcp_right_pos_y" in state_names
        else None
    )

    # Find indices for gripper actions
    gripper_left_idx = (
        action_names.index("gripper_left") if "gripper_left" in action_names else None
    )
    gripper_right_idx = (
        action_names.index("gripper_right") if "gripper_right" in action_names else None
    )

    results = {
        "left": {"start_positions": [], "grasp_positions": [], "episode_ids": []},
        "right": {"start_positions": [], "grasp_positions": [], "episode_ids": []},
    }

    # Process each episode
    num_episodes = (
        dataset.num_episodes
        if max_episodes is None
        else min(max_episodes, dataset.num_episodes)
    )

    for ep_idx in range(num_episodes):
        if ep_idx in blacklist:
            print(f"Skipping blacklisted episode {ep_idx}")
            continue

        # Get frame range for this episode using episode_data_index
        # episode_data_index contains 'from' and 'to' indices for each episode
        from_idx = dataset.episode_data_index["from"][ep_idx].item()
        to_idx = dataset.episode_data_index["to"][ep_idx].item()

        if from_idx >= to_idx:
            continue

        # Get first frame (starting position)
        first_frame = dataset[from_idx]
        start_state = first_frame["observation.state"].numpy()

        # Get all actions and states for the episode
        actions = []
        states = []
        for frame_idx in range(from_idx, to_idx):
            frame = dataset[frame_idx]
            actions.append(frame["action"].numpy())
            states.append(frame["observation.state"].numpy())

        actions = np.array(actions)
        states = np.array(states)

        # Process left arm
        if (
            (arm == "left" or arm == "both")
            and tcp_left_x_idx is not None
            and gripper_left_idx is not None
        ):
            start_x = start_state[tcp_left_x_idx]
            start_y = start_state[tcp_left_y_idx]
            results["left"]["start_positions"].append((start_x, start_y))
            results["left"]["episode_ids"].append(ep_idx)

            # Find first grasp
            grasp_frame_idx = find_first_grasp(actions, gripper_left_idx)
            if grasp_frame_idx is not None and grasp_frame_idx < len(states):
                grasp_state = states[grasp_frame_idx]
                grasp_x = grasp_state[tcp_left_x_idx]
                grasp_y = grasp_state[tcp_left_y_idx]
                results["left"]["grasp_positions"].append((grasp_x, grasp_y))
            else:
                results["left"]["grasp_positions"].append((np.nan, np.nan))

        # Process right arm
        if (
            (arm == "right" or arm == "both")
            and tcp_right_x_idx is not None
            and gripper_right_idx is not None
        ):
            start_x = start_state[tcp_right_x_idx]
            start_y = start_state[tcp_right_y_idx]
            results["right"]["start_positions"].append((start_x, start_y))
            results["right"]["episode_ids"].append(ep_idx)

            # Find first grasp
            grasp_frame_idx = find_first_grasp(actions, gripper_right_idx)
            if grasp_frame_idx is not None and grasp_frame_idx < len(states):
                grasp_state = states[grasp_frame_idx]
                grasp_x = grasp_state[tcp_right_x_idx]
                grasp_y = grasp_state[tcp_right_y_idx]
                results["right"]["grasp_positions"].append((grasp_x, grasp_y))
            else:
                results["right"]["grasp_positions"].append((np.nan, np.nan))

        if (ep_idx + 1) % 10 == 0:
            print(f"Processed {ep_idx + 1}/{num_episodes} episodes...")

    print(f"\nProcessed {num_episodes} episodes")
    return results


def plot_positions(results, arm: str, output_path: Path):
    """
    Create scatter plot of starting and grasp positions.

    Args:
        results: Dictionary with position data
        arm: Which arm to plot ('left', 'right', or 'both')
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    arms_to_plot = ["left", "right"] if arm == "both" else [arm]

    # Color scheme: left arm (blue/red), right arm (green/orange)
    colors = {
        "left": {"start": "blue", "grasp": "red"},
        "right": {"start": "green", "grasp": "orange"},
    }

    stats_lines = []

    for arm_name in arms_to_plot:
        start_pos = np.array(results[arm_name]["start_positions"])
        grasp_pos = np.array(results[arm_name]["grasp_positions"])
        episode_ids = results[arm_name]["episode_ids"]

        if len(start_pos) == 0:
            continue

        # Remove NaN values from grasp positions
        valid_grasp = ~np.isnan(grasp_pos[:, 0])

        # Plot starting positions
        ax.scatter(
            start_pos[:, 0],
            start_pos[:, 1],
            c=colors[arm_name]["start"],
            alpha=0.6,
            s=50,
            label=f"{arm_name.capitalize()} Start",
            marker="o",
        )

        # Add episode ID labels for start positions
        for i, (x, y) in enumerate(start_pos):
            ax.annotate(
                str(episode_ids[i]),
                (x, y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                alpha=0.7,
                color=colors[arm_name]["start"],
            )

        # Plot grasp positions
        if np.any(valid_grasp):
            ax.scatter(
                grasp_pos[valid_grasp, 0],
                grasp_pos[valid_grasp, 1],
                c=colors[arm_name]["grasp"],
                alpha=0.6,
                s=50,
                label=f"{arm_name.capitalize()} Grasp",
                marker="x",
            )

            # Add episode ID labels for grasp positions
            for i, valid in enumerate(valid_grasp):
                if valid:
                    x, y = grasp_pos[i]
                    ax.annotate(
                        str(episode_ids[i]),
                        (x, y),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=7,
                        alpha=0.7,
                        color=colors[arm_name]["grasp"],
                    )

        # Collect statistics
        stats_lines.append(
            f"{arm_name.capitalize()}: {len(start_pos)} episodes, {np.sum(valid_grasp)} with grasp"
        )

    ax.set_xlabel("X Position (m)", fontsize=12)
    ax.set_ylabel("Y Position (m)", fontsize=12)

    if arm == "both":
        ax.set_title("Workspace Overview - Start and Grasp Positions", fontsize=14)
    else:
        ax.set_title(f"{arm.capitalize()} Arm - Start and Grasp Positions", fontsize=14)

    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    # Add statistics
    stats_text = "\n".join(stats_lines)
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot starting and first grasp positions from LeRobot dataset"
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=Path,
        required=True,
        help="Path to the LeRobot dataset directory",
    )
    parser.add_argument(
        "--arm",
        "-a",
        type=str,
        choices=["left", "right", "both"],
        default="both",
        help="Which arm to analyze (default: both)",
    )
    parser.add_argument(
        "--max-episodes",
        "-m",
        type=int,
        default=None,
        help="Maximum number of episodes to process (default: all)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for the plot (default: grasp_positions.png in current directory)",
    )
    parser.add_argument(
        "--gripper-threshold",
        "-t",
        type=float,
        default=0.5,
        help="Threshold for detecting gripper closure (default: 0.5)",
    )

    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: Dataset directory not found: {args.data_dir}")
        return 1

    # Set default output path if not provided
    if args.output is None:
        args.output = Path.cwd() / "grasp_positions.png"

    # Extract positions
    results = extract_positions(args.data_dir, args.arm, args.max_episodes)

    # Plot results
    plot_positions(results, args.arm, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
