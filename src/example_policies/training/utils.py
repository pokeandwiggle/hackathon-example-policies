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

import json
import os
import pathlib

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features


def create_dataset_config(
    dataset_root_dir: pathlib.Path | None = None,
    repo_id: str | None = None,
):
    assert dataset_root_dir is not None or repo_id is not None, (
        "Either data_dir or repo_id must be provided"
    )
    # get last folder name of dataset_root_dir_path. Nice Side Effect: Automatic Tag in WandB
    fake_repo_id = repo_id if repo_id is not None else dataset_root_dir.name

    if repo_id is None:
        episode_list = make_episode_white_list(dataset_root_dir)
    else:
        episode_list = None
    data_cfg = DatasetConfig(
        repo_id=fake_repo_id,
        root=dataset_root_dir,
        episodes=episode_list,
    )

    meta_data = LeRobotDatasetMetadata(
        repo_id=fake_repo_id,
        root=dataset_root_dir,
    )
    features = dataset_to_policy_features(meta_data.features)

    return data_cfg, features


def make_episode_white_list(dataset_root_dir: str | pathlib.Path):
    blacklist_path = os.path.join(dataset_root_dir, "meta", "blacklist.json")
    episodes_path = os.path.join(dataset_root_dir, "meta", "episodes.jsonl")

    if not os.path.exists(blacklist_path):
        return None

    with open(blacklist_path, "r") as f:
        # Use a set for efficient lookup of blacklisted indices.
        blacklisted_indices = set(json.load(f))

    all_episodes = []
    with open(episodes_path, "r") as f:
        # Correctly parse the .jsonl file line by line.
        for line in f:
            if line.strip():  # Ensure the line is not empty
                all_episodes.append(json.loads(line)["episode_index"])

    # Create a whitelist of episodes that are not in the blacklist.
    whitelist = [
        episode for episode in all_episodes if episode not in blacklisted_indices
    ]
    return whitelist
