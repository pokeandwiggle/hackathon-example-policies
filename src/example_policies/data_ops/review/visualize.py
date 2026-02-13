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

import os

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.visualize_dataset import visualize_dataset

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize a LeRobot dataset")
    parser.add_argument(
        "root_dir",
        type=str,
        help="Path to the dataset directory or Hugging Face repo ID",
    )
    parser.add_argument(
        "-e",
        "--episode-index",
        type=int,
        default=0,
        metavar="N",
        help="Episode to visualize (default: 0)",
    )

    args = parser.parse_args()

    if os.path.isdir(args.root_dir):
        # It's a local directory
        dataset = LeRobotDataset(args.root_dir, root=args.root_dir)
    else:
        # It's not a local directory, assume it's a repo_id
        print(
            f"Warning: '{args.root_dir}' not found locally. Attempting to load from Hugging Face Hub."
        )
        dataset = LeRobotDataset(args.root_dir)

    print(f"Visualizing episode {args.episode_index}...")
    visualize_dataset(dataset, episode_index=args.episode_index)
