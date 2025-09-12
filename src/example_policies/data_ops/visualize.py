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
        "root",
        type=str,
        help="Path to the root directory of the dataset.",
    )

    # Episode Index. Defaults to 0 if not provided.
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode to visualize.",
    )

    args = parser.parse_args()

    dataset = LeRobotDataset(args.root, root=args.root)
    visualize_dataset(dataset, episode_index=args.episode_index)
