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
import pathlib

from lerobot.policies.pretrained import PreTrainedConfig

from example_policies.lerobot_patches import apply_patches


def get_checkpoint_path(checkpoint_path: pathlib.Path | str) -> pathlib.Path:
    """Returns the path to the checkpoint directory."""
    checkpoint_path = pathlib.Path(checkpoint_path)
    # check if checkpoint path contains config.json
    if not (checkpoint_path / "config.json").exists():
        print(
            f"Checkpoint path {checkpoint_path} does not contain config.json, extending path."
        )
        checkpoint_path_extend = (
            checkpoint_path / "checkpoints" / "last" / "pretrained_model"
        )

        if not checkpoint_path_extend.exists():
            raise FileNotFoundError(
                f"Extended checkpoint path {checkpoint_path_extend} does not exist."
            )
        checkpoint_path = checkpoint_path_extend
    return checkpoint_path


def load_metadata(dir_path: pathlib.Path) -> dict:
    """Load Metadata for a model checkpoint or a dataset dir

    Args:
        dir_path (pathlib.Path): Path to the directory of model checkpoint or dataset

    Returns:
        dict: Metadata information
    """
    meta_json = dir_path / "dataset_info.json"
    if not meta_json.exists():
        print("Did not find any dataset metadata")
        return load_dataset_info(dir_path)
    with open(meta_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata


def load_dataset_info(dir_path: pathlib.Path) -> dict:
    """Load Dataset Info from lerobot package

    Args:
        dir_path (pathlib.Path): Path to the directory of the dataset
    Returns:
        dict: Dataset Info
    """
    meta_json = dir_path / "meta" / "info.json"
    with open(meta_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata


def load_policy(checkpoint_dir: pathlib.Path):
    apply_patches()
    from lerobot.policies.factory import get_policy_class

    checkpoint_dir = get_checkpoint_path(checkpoint_dir)
    cfg = PreTrainedConfig.from_pretrained(checkpoint_dir)

    PolicyCls = get_policy_class(cfg.type)
    policy = PolicyCls.from_pretrained(checkpoint_dir)
    policy.reset()
    metadata = load_metadata(checkpoint_dir)

    setattr(cfg, "metadata", metadata)
    return policy, cfg
