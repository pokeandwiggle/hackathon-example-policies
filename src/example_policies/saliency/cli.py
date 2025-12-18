#!/usr/bin/env python
"""
Command-line interface for saliency analysis.

Usage:
    python -m example_policies.saliency.cli -c /path/to/checkpoint -d /path/to/dataset -e 0 -f 10
"""

import argparse
import logging
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from example_policies.robot_deploy.deploy_core.policy_loader import load_policy
from example_policies.saliency.core import compute_saliency
from example_policies.saliency.visualization import visualize_saliency

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def to_device_batch(batch: dict, device: torch.device, non_blocking: bool = True):
    """Move batch tensors to device."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Saliency analysis of a robot learning policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze episode 0, frame 10
  python -m example_policies.saliency.cli -c ./outputs/train_123 -d ./data -e 0 -f 10

  # Analyze with custom output path
  python -m example_policies.saliency.cli -c ./outputs/train_123 -d ./data -e 5 -f 20 -o ./saliency.png
        """,
    )
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
        help="Episode index to analyze (default: 0)",
    )
    parser.add_argument(
        "-f",
        "--frame",
        type=int,
        default=0,
        metavar="N",
        help="Frame index within the episode (default: 0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output path for the figure (default: saliency_ep<E>_frame<F>.png)",
    )
    return parser.parse_args()


def find_frame_in_dataset(dataloader, target_episode, target_frame, device):
    """
    Find a specific frame in the dataset.

    Args:
        dataloader: PyTorch DataLoader for the dataset
        target_episode: Episode index to find
        target_frame: Frame index within the episode
        device: Device to move batch to

    Returns:
        dict: The batch containing the target frame, or None if not found
    """
    current_frame_in_episode = 0

    for batch in dataloader:
        b_ep = batch.get("episode_index")
        if b_ep is None:
            raise KeyError("Expected key 'episode_index' in batch.")
        b_ep = int(b_ep.view(-1)[0].item())

        if b_ep < target_episode:
            continue
        if b_ep > target_episode:
            logging.error(f"Episode {target_episode} not found in dataset")
            return None

        # We're in the target episode
        if current_frame_in_episode < target_frame:
            current_frame_in_episode += 1
            continue
        elif current_frame_in_episode > target_frame:
            logging.error(f"Frame {target_frame} not found in episode {target_episode}")
            return None

        # Found the target frame
        logging.info(f"Processing episode {target_episode}, frame {target_frame}")
        return to_device_batch(batch, device, non_blocking=False)

    logging.error(f"Episode {target_episode} not found in dataset")
    return None


def main():
    """Main entry point for saliency analysis CLI."""
    args = parse_args()

    # Select device
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    logging.info(f"Using device: {device}")

    # Load policy
    logging.info(f"Loading policy from {args.checkpoint}")
    policy, cfg = load_policy(args.checkpoint)
    policy.to(device)

    # Load dataset
    logging.info(f"Loading dataset from {args.dataset}")
    dataset = LeRobotDataset(
        repo_id=args.dataset,
        root=args.dataset,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,  # Use 0 for gradient computation
        batch_size=1,
        shuffle=False,
        pin_memory=device != "cpu",
        drop_last=False,
    )

    # Find target frame
    logging.info(f"Searching for episode {args.episode}, frame {args.frame}...")
    batch = find_frame_in_dataset(dataloader, args.episode, args.frame, device)
    if batch is None:
        return

    # Identify image keys
    image_keys = [k for k in batch.keys() if k.startswith("observation.images.")]
    if not image_keys:
        logging.warning("No image observations found in batch")
        return

    logging.info(f"Found {len(image_keys)} image inputs: {image_keys}")

    # Store original images for visualization
    images_dict = {k: batch[k].detach().clone() for k in image_keys}

    # Compute saliency maps
    state_key = "observation.state"
    logging.info("Computing saliency maps...")
    with torch.set_grad_enabled(True):
        image_saliency_maps, state_saliency, action_output = compute_saliency(
            policy, batch, image_keys, state_key
        )

    # Visualize results
    output_path = args.output or Path(
        f"./saliency_ep{args.episode}_frame{args.frame}.png"
    )
    logging.info("Generating visualization...")
    visualize_saliency(
        images_dict,
        image_saliency_maps,
        state_saliency,
        output_path,
        args.episode,
        args.frame,
    )

    logging.info(f"Saved saliency visualization to {output_path}")


if __name__ == "__main__":
    main()
