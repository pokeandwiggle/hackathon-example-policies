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

import dataclasses
import pathlib
from copy import deepcopy

import draccus

from example_policies import config_factory as cf

"""
Example usage:
    python train.py \
        --hf_repo_id pokeandwiggle/my_task \
        --policy.steps 100000 \
        --policy.horizon 96 \
        --policy.n_action_steps 48
"""


@dataclasses.dataclass
class TrainCliArgs:
    """Main training script arguments with nested policy configuration.

    All policy-specific parameters are exposed via the --policy.* namespace.
    For example:
        --policy.batch-size 32
        --policy.horizon 20
        --policy.termination-focal-loss-weight 1.0
    """

    # Path to the dataset directory containing training data.
    data_dir: pathlib.Path = None
    # HuggingFace dataset repo ID (alternative to data_dir). Downloads and caches locally.
    hf_repo_id: str | None = None
    # Include depth images in the input features.
    include_depth: bool = False
    # Nested policy configuration - exposes all DiTFlowConfig parameters
    policy: cf.DiTFlowConfig = dataclasses.field(default_factory=cf.DiTFlowConfig)

    def __post_init__(self):
        """Sync data_dir to nested policy config after parsing."""
        if self.hf_repo_id and self.data_dir:
            raise ValueError("Set only one of --data-dir or --hf-repo-id, not both.")
        if not self.hf_repo_id and not self.data_dir:
            raise ValueError("Set either --data-dir (local) or --hf-repo-id (HuggingFace).")
        if self.hf_repo_id:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            print(f"Downloading dataset '{self.hf_repo_id}' from HuggingFace Hub...")
            _hf_dataset = LeRobotDataset(repo_id=self.hf_repo_id)
            self.data_dir = pathlib.Path(_hf_dataset.root)
            print(f"Downloaded to: {self.data_dir}")
            del _hf_dataset
        self.policy.dataset_root_dir = str(self.data_dir)


def select_inputs(include_depth: bool = False):
    selected_inputs = [
        "observation.state",
        "observation.images.rgb_left",
        "observation.images.rgb_right",
        "observation.images.rgb_static",
    ]
    if include_depth:
        selected_inputs.append("observation.images.depth_left")
        selected_inputs.append("observation.images.depth_right")
    return selected_inputs


def filter_depth(cfg, include_depth: bool = False):
    selected_inputs = select_inputs(include_depth)

    input_features = deepcopy(cfg.policy.input_features)
    cfg.policy.input_features = {
        k: v for k, v in input_features.items() if k in selected_inputs
    }

    return cfg


def train_policy(cli_config: TrainCliArgs):
    """Build training configuration and start training.

    Args:
        args: Parsed CLI arguments with nested policy configuration.
    """
    # Build the LeRobot training configuration from the nested policy config.
    # The policy config already has all parameters set via CLI (e.g., --policy.batch-size).
    cfg = cli_config.policy.build()

    # Ensure a checkpoint is saved at end of training
    if cfg.save_freq > cfg.steps:
        cfg.save_freq = cfg.steps

    # Filter depth images based on include_depth flag
    cfg = filter_depth(cfg, cli_config.include_depth)

    train(cfg)


def train(cfg):
    import warnings

    warnings.filterwarnings("ignore", message=".*video decoding.*torchvision.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*No files have been modified since last commit.*")

    # Capture and disable lerobot's built-in push_to_hub — it uploads to a
    # temp dir that misses dataset_info.json.  We use upload_model() instead.
    push_to_hub = getattr(cfg.policy, "push_to_hub", False)
    repo_id = getattr(cfg.policy, "repo_id", None)
    private = getattr(cfg.policy, "private", None)
    if push_to_hub:
        cfg.policy.push_to_hub = False

    print("\nStarting training...")
    from lerobot.scripts.lerobot_train import train as lerobot_train

    lerobot_train(cfg)

    if push_to_hub and repo_id:
        from example_policies.data_ops.upload_model import upload_model

        # Default to private=True when not explicitly set
        upload_model(cfg.output_dir, repo_id=repo_id, private=private if private is not None else True)


def main():
    from example_policies.lerobot_patches import apply_patches
    apply_patches()
    cli_config = draccus.parse(config_class=TrainCliArgs)
    train_policy(cli_config)


if __name__ == "__main__":
    main()
