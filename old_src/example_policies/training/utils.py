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

from example_policies.utils.constants import BLACKLIST_FILE, EPISODES_FILE, META_DIR


def create_dataset_config(data_dir: pathlib.Path):
    # get last folder name of dataset_root_dir_path. Nice Side Effect: Automatic Tag in WandB
    fake_repo_id = data_dir.name

    episode_list = make_episode_white_list(data_dir)
    data_cfg = DatasetConfig(repo_id=fake_repo_id, root=data_dir, episodes=episode_list)

    meta_data = LeRobotDatasetMetadata(
        repo_id=fake_repo_id,
        root=data_dir,
    )
    features = dataset_to_policy_features(meta_data.features)

    return data_cfg, features


def make_episode_white_list(dataset_root_dir: str | pathlib.Path):
    blacklist_path = os.path.join(dataset_root_dir, META_DIR, BLACKLIST_FILE)
    episodes_path = os.path.join(dataset_root_dir, META_DIR, EPISODES_FILE)

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


def shorten_name(
    full_name: str, max_word_length: int = 4, sep: str = "_", joint: str = ""
) -> str:
    """Shorten a full name by truncating each word to a maximum length.

    Args:
        full_name: The original full name string.
        max_word_length: Maximum length for each word.
        sep: Separator used to split and join words.
        joint: String used to join the shortened words.

    Returns:
        Shortened name string.
    """
    # Remove special characters
    full_name = full_name.translate(str.maketrans("", "", "[](),.- \\"))
    words = full_name.split(sep)

    shortened_words = [
        word if len(word) <= max_word_length else word[:max_word_length]
        for word in words
    ]
    shortened_name = joint.join(shortened_words)
    return shortened_name
