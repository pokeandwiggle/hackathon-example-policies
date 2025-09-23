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
from copy import deepcopy
from pprint import pprint

# First Import Numpy. Workaround for torch / lerobot bug
import numpy

from example_policies import config_factory as cf
from example_policies.lerobot_patches import apply_patches


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


def main(
    data_dir: str,
    include_depth: bool = False,
    batch_size: int = 32,
    resume_path: str = None,
    wandb_project: str = None,
):
    # --- 1. Configure Training ---
    # This section defines the core parameters for the training run, including the model
    # architecture, dataset, and key hyperparameters.

    # cfg, input_features = cf.create_predefined_model_config(...)
    cfg = cf.dit_flow(
        data_dir,
        batch_size,
        resume_path=resume_path,
    )

    cfg.wandb.project = (
        wandb_project if wandb_project is not None else "munich_hackathon"
    )

    # Reduce System Memory
    # cfg.num_workers = 2

    cfg = filter_depth(cfg, include_depth)

    train(cfg)


def train(cfg):
    print("\nStarting training...")
    # import after monkey patching
    from lerobot.scripts.train import init_logging, train

    init_logging()
    train(cfg)


def _parse_args():
    parser = argparse.ArgumentParser(description="Training script for LeRobot policies")
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to the data directory",
    )
    parser.add_argument(
        "--include_depth", action="store_true", help="Include depth images in the input"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )

    parser.add_argument(
        "--resume", type=str, default=None, help="Path to the checkpoint directory"
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name. If not set, defaults to 'munich_hackathon'.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    apply_patches()
    args = _parse_args()
    main(
        data_dir=args.data_dir,
        include_depth=args.include_depth,
        batch_size=args.batch_size,
        resume_path=args.resume,
        wandb_project=args.wandb_project,
    )
