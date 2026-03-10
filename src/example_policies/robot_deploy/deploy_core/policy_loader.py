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
from example_policies.utils.constants import INFO_FILE, META_DIR


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
    meta_json = dir_path / META_DIR / INFO_FILE
    with open(meta_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata


def load_policy(checkpoint_dir: pathlib.Path):
    apply_patches()
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    checkpoint_dir = get_checkpoint_path(checkpoint_dir)
    cfg = PreTrainedConfig.from_pretrained(checkpoint_dir)

    PolicyCls = get_policy_class(cfg.type)
    policy = PolicyCls.from_pretrained(checkpoint_dir)
    policy.reset()
    metadata = load_metadata(checkpoint_dir)

    # Load preprocessor and postprocessor for inference
    # Try loading from pretrained path first, fall back to creating new ones
    #
    # Override the device_processor to use the actual device (the saved config
    # may reference 'cuda' even when running inference on CPU, or vice-versa).
    device_override = {"device_processor": {"device": str(cfg.device)}}
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=checkpoint_dir,
            preprocessor_overrides=device_override,
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
    except Exception:
        # Old checkpoint without processor configs - extract stats from policy's
        # normalize_inputs/unnormalize_outputs modules (if available) and create processors
        dataset_stats = _extract_stats_from_policy(policy)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=None,
            dataset_stats=dataset_stats,
        )

    setattr(cfg, "metadata", metadata)
    return policy, cfg, preprocessor, postprocessor


def _extract_stats_from_policy(policy) -> dict | None:
    """Extract dataset stats from policy's normalize_inputs module for backward compatibility.

    Old checkpoints have stats stored in normalize_inputs.buffer_* parameters.
    This extracts them to create processors dynamically.
    """
    if not hasattr(policy, 'normalize_inputs'):
        return None

    stats = {}
    normalize = policy.normalize_inputs

    # Iterate through attributes looking for buffer_* ParameterDicts
    for attr_name in dir(normalize):
        if attr_name.startswith('buffer_'):
            # Convert buffer_observation_state back to observation.state
            key = attr_name[7:].replace('_', '.')
            buffer = getattr(normalize, attr_name)

            if hasattr(buffer, 'mean') and hasattr(buffer, 'std'):
                stats[key] = {
                    'mean': buffer['mean'].data,
                    'std': buffer['std'].data,
                }
            elif hasattr(buffer, 'min') and hasattr(buffer, 'max'):
                stats[key] = {
                    'min': buffer['min'].data,
                    'max': buffer['max'].data,
                }

    # Also get action stats from unnormalize_outputs
    if hasattr(policy, 'unnormalize_outputs'):
        unnormalize = policy.unnormalize_outputs
        for attr_name in dir(unnormalize):
            if attr_name.startswith('buffer_'):
                key = attr_name[7:].replace('_', '.')
                buffer = getattr(unnormalize, attr_name)

                if hasattr(buffer, 'mean') and hasattr(buffer, 'std'):
                    stats[key] = {
                        'mean': buffer['mean'].data,
                        'std': buffer['std'].data,
                    }
                elif hasattr(buffer, 'min') and hasattr(buffer, 'max'):
                    stats[key] = {
                        'min': buffer['min'].data,
                        'max': buffer['max'].data,
                    }

    return stats if stats else None
