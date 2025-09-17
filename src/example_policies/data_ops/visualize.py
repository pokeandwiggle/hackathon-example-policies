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

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.visualize_dataset import visualize_dataset

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize a LeRobot dataset.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="If set, load the dataset from a local directory instead of the hub.",
    )

    parser.add_argument(
        "--root",
        type=str,
        help="Path to the root directory of the dataset.",
        required=False,
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        help="Hugging Face repo ID to load the dataset from. Example: 'username/dataset_name'",
        required=False,
    )

    # Episode Index. Defaults to 0 if not provided.
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode to visualize.",
    )

    args = parser.parse_args()

    if not args.local and not args.repo_id:
        raise ValueError(
            "Either --local or --repo-id (for online repositories) must be provided."
        )

    if args.local and not args.root:
        raise ValueError("--root must be provided when --local is set.")

    if args.local:
        dataset = LeRobotDataset(
            args.repo_id if args.repo_id else args.root, root=args.root
        )
    else:
        dataset = LeRobotDataset(args.repo_id)

    print(f"Visualizing episode {args.episode_index}...")
    visualize_dataset(dataset, episode_index=args.episode_index)
