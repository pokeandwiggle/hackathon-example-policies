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

"""
Monkey patches for lerobot compatibility.

Patches Status (lerobot v0.4.x from pokeandwiggle fork):
- monkey_patch_video_query: NEEDED - Preserves temporal dimension for n_obs_steps=1
- monkey_patch_policy_factory: NEEDED - Registers custom policies (ditflow, xditflow)
- monkey_patch_save_checkpoint: NEEDED - Copies dataset_info.json for deployment
"""

import pathlib
import shutil
import torch

from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets import video_utils
from lerobot.policies import factory as lerobot_factory
from lerobot.utils import train_utils

from .policies.factory import get_policy
from .utils.constants import INFO_FILE, META_DIR


def monkey_patch_video_query():
    """
    Patch _query_videos to avoid squeezing the temporal dimension when n_obs_steps=1.
    
    The original code uses `frames.squeeze(0)` which removes the temporal dimension
    when there's only 1 frame. This causes shape mismatches in the policy's forward
    pass which expects shape (B, n_obs_steps, C, H, W).
    
    The original squeeze(0) was likely intended for a different purpose but is harmful
    when n_obs_steps=1. We simply don't squeeze at all - the temporal dimension should
    always be preserved.
    """

    def patched_query_videos(
        self, query_timestamps: dict[str, list[float]], ep_idx: int
    ) -> dict[str, torch.Tensor]:
        # Get episode metadata for timestamp offset
        ep = self.meta.episodes[ep_idx]
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            # Episodes are stored sequentially on a single mp4 to reduce the number of files.
            # Thus we load the start timestamp of the episode on this mp4 and,
            # shift the query timestamp accordingly.
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]
            
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = video_utils.decode_video_frames(
                video_path, shifted_query_ts, self.tolerance_s, self.video_backend
            )
            # Don't squeeze - preserve temporal dimension for all cases
            # Original code did frames.squeeze(0) which breaks n_obs_steps=1
            item[vid_key] = frames  # Shape: (n_obs_steps, C, H, W)

        return item

    LeRobotDataset._query_videos = patched_query_videos


def monkey_patch_policy_factory():
    """
    Extend lerobot's policy factory to include custom policies.
    
    This allows training with custom policies (ditflow, xditflow) via lerobot's
    training script by falling back to our policy registry when lerobot doesn't
    recognize the policy name.
    """
    original_lerobot_factory_fn = lerobot_factory.get_policy_class

    def extended_get_policy_cls(name: str):
        try:
            return original_lerobot_factory_fn(name)
        except (NotImplementedError, ValueError):
            return get_policy(name)
        except Exception as e:
            raise e

    lerobot_factory.get_policy_class = extended_get_policy_cls


def monkey_patch_save_checkpoint():
    """
    Extend save_checkpoint to copy dataset metadata for deployment.
    
    This copies dataset_info.json to the pretrained model directory so that
    deployment code can access dataset metadata (e.g., action feature names)
    without needing the original dataset.
    """
    original_save_fn = train_utils.save_checkpoint

    def patched_save_checkpoint(
        checkpoint_dir: pathlib.Path, step: int, cfg, policy, optimizer, scheduler,
        preprocessor=None, postprocessor=None
    ):
        original_save_fn(checkpoint_dir, step, cfg, policy, optimizer, scheduler,
                        preprocessor=preprocessor, postprocessor=postprocessor)
        pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
        meta_json = pathlib.Path(cfg.dataset.root) / META_DIR / INFO_FILE
        # Copy Metadata for deployment data selection
        if meta_json.exists():
            shutil.copy(meta_json, pretrained_dir / "dataset_info.json")

    train_utils.save_checkpoint = patched_save_checkpoint


def apply_patches():
    """Apply all required monkey patches for lerobot compatibility."""
    monkey_patch_policy_factory()
    monkey_patch_video_query()
    monkey_patch_save_checkpoint()
