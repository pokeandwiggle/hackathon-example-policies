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

"""Adapted from https://github.com/huggingface/lerobot"""

import argparse
from pathlib import Path
from typing import Any, Dict

# Workaround for torch environment bug
import numpy
import torch
import wandb
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def main(
    dataset_root_dir: str,
    output_dir: str = "outputs/train/custom_loop",
    training_steps: int = 5000,
    log_freq: int = 100,
    checkpoint_freq: int = 1000,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    num_workers: int = 4,
    wandb_project: str | None = None,
):
    """
    Trains a diffusion policy on a LeRobot dataset.

    Args:
        dataset_root_dir: Path to the root directory of the dataset.
        output_dir: Directory to save training outputs (checkpoints, logs).
        training_steps: Total number of training steps.
        log_freq: Frequency (in steps) for logging training loss.
        checkpoint_freq: Frequency (in steps) for saving model checkpoints.
        batch_size: Number of samples per batch.
        learning_rate: Optimizer learning rate.
        num_workers: Number of data loader workers.
        wandb_project: Name of the Weights & Biases project. If None, wandb is disabled.
        wandb_entity: Name of the Weights & Biases entity (user or team).
    """
    # Initialize Weights & Biases if a project name is provided.
    if wandb_project:
        wandb.init(project=wandb_project)

    # Create a directory to store the training checkpoints.
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # When starting from scratch, we need to specify two things before creating the policy:
    # 1. Input/output shapes: to properly size the policy's neural networks.
    # 2. Dataset stats: for normalization and denormalization of inputs/outputs.
    # `LeRobotDatasetMetadata` fetches this information from the dataset's metadata file.
    dataset_metadata = LeRobotDatasetMetadata(
        repo_id=dataset_root_dir,
        root=dataset_root_dir,
    )
    features = dataset_to_policy_features(dataset_metadata.features)

    # Define Input and Output Features for the Model
    # Actions are the outputs we want the policy to predict.
    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }

    # For this example, we select specific observation modalities as inputs.
    selected_inputs = [
        "observation.state",
        "observation.images.rgb_left",
        "observation.images.rgb_right",
    ]
    input_features = {
        key: ft
        for key, ft in features.items()
        if key not in output_features and key in selected_inputs
    }

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    cfg = DiffusionConfig(
        input_features=input_features, output_features=output_features
    )

    # We can now instantiate our policy with this config and the dataset stats.
    policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # `delta_timestamps` define the temporal context for observations and actions.
    # Each policy expects a specific number of frames for its inputs and outputs.
    # For instance, `observation_delta_indices` in the config might specify loading
    # the current frame (t=0) and the previous frame (t=-1).
    delta_timestamps = {
        feature_name: [i / 10.0 for i in cfg.observation_delta_indices]
        for feature_name in input_features.keys()
        if feature_name.startswith("observation")
    }
    delta_timestamps["action"] = [i / 10.0 for i in cfg.action_delta_indices]

    # In this case with the standard configuration for Diffusion Policy, it is equivalent to this:
    hard_coded_delta_timestamps = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.images.rgb_left": [-0.1, 0.0],
        "observation.images.rgb_right": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to supervise the policy.
        "action": [
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
        ],
    }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset(
        repo_id=dataset_root_dir,
        root=dataset_root_dir,
        delta_timestamps=delta_timestamps,
    )

    # Create the optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device != "cpu",
        drop_last=True,
    )

    # --- Training Loop ---
    step = 0
    print("Starting training loop...")
    for step in range(training_steps):
        for batch in dataloader:
            # Move batch to the training device.
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            # Forward pass to compute the loss.
            loss, _ = policy.forward(batch)

            # Backward pass and optimization step.
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log loss and save checkpoints periodically.
            if step % log_freq == 0:
                print(f"Step: {step} Loss: {loss.item():.3f}")
            if wandb_project:
                wandb.log({"loss": loss.item()}, step=step)

            if step > 0 and step % checkpoint_freq == 0:
                checkpoint_dir = output_directory / f"checkpoint_{step}"
                print(f"Saving checkpoint to {checkpoint_dir}...")
                policy.save_pretrained(checkpoint_dir)

    # --- End of Training ---
    print("Training finished.")

    # Save the final policy checkpoint.
    final_checkpoint_dir = output_directory / "final"
    print(f"Saving final checkpoint to {final_checkpoint_dir}...")
    policy.save_pretrained(final_checkpoint_dir)

    if wandb_project:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a diffusion policy.")
    parser.add_argument(
        "--dataset-root-dir",
        type=str,
        required=True,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/train/custom_loop",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=5000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "--log-freq", type=int, default=100, help="Logging frequency in steps."
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=1000,
        help="Checkpoint saving frequency in steps.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate."
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of dataloader workers."
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name.",
    )
    args = parser.parse_args()

    main(**vars(args))
