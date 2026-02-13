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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload a LeRobot dataset to the hub.")
    parser.add_argument(
        "--repo-id",
        type=str,
        help="Hugging Face repo ID to push the dataset to. Example: 'username/dataset_name'.",
        required=True,
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Path to the root directory of the dataset.",
        required=True,
    )

    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    dataset.push_to_hub()
