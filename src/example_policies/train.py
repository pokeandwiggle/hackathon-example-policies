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
from example_policies.lerobot_patches import apply_patches


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
    data_dir: pathlib.Path
    # Include depth images in the input features.
    include_depth: bool = False
    # Nested policy configuration - exposes all DiTFlowConfig parameters
    policy: cf.DiTFlowConfig = dataclasses.field(default_factory=cf.DiTFlowConfig)

    def __post_init__(self):
        """Sync data_dir to nested policy config after parsing."""
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
    # --- 2. Inspect Dataset and Select Model Inputs ---
    print("Available dataset features:")
    for name, feature in cfg.policy.input_features.items():
        print(f"  - {name}: shape={feature.shape}")

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

    # Filter depth images based on include_depth flag
    cfg = filter_depth(cfg, cli_config.include_depth)

    train(cfg)


def train(cfg):
    print("\nStarting training...")
    # import after monkey patching
    from lerobot.scripts.train import init_logging
    from lerobot.scripts.train import train as lerobot_train

    init_logging()
    lerobot_train(cfg)


def main():
    cli_config = draccus.parse(config_class=TrainCliArgs)
    train_policy(cli_config)


if __name__ == "__main__":
    apply_patches()
    main()
