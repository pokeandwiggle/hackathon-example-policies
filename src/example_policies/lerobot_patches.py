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

import logging
import math
import os
import pathlib
import shutil

import torch
import torch.nn as nn
from lerobot.constants import HF_LEROBOT_HOME, PRETRAINED_MODEL_DIR
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import factory as lerobot_factory
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT
from lerobot.utils import train_utils, wandb_utils
from transformers import AutoImageProcessor, AutoModel

from .policies.factory import get_policy


def monkey_patch_dataset():
    original_fn = LeRobotDataset._get_query_indices

    def patched_get_query_indices(
        self, idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int | bool]]]:
        if self.episodes is not None:
            ep_idx = self.episodes.index(ep_idx)
        return original_fn(self, idx, ep_idx)

    LeRobotDataset._get_query_indices = patched_get_query_indices


def monkey_patch_policy_factory():
    original_lerobot_factory_fn = lerobot_factory.get_policy_class

    def extended_get_policy_cls(name: str):
        try:
            return original_lerobot_factory_fn(name)
        except NotImplementedError:
            return get_policy(name)
        except Exception as e:
            raise e

    lerobot_factory.get_policy_class = extended_get_policy_cls


def monkey_patch_save_checkpoint():
    original_save_fn = train_utils.save_checkpoint

    def patched_save_checkpoint(
        checkpoint_dir: pathlib.Path, step: int, cfg, policy, optimizer, scheduler
    ):
        original_save_fn(checkpoint_dir, step, cfg, policy, optimizer, scheduler)
        pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
        if cfg.dataset.root is None:
            root = HF_LEROBOT_HOME / cfg.dataset.repo_id
        else:
            root = pathlib.Path(cfg.dataset.root)
        meta_json = root / "meta" / "info.json"
        # Copy Metadata for deployment data selection
        shutil.copy(meta_json, pretrained_dir / "dataset_info.json")

    train_utils.save_checkpoint = patched_save_checkpoint


def monkey_patch_wandb():
    "Hacky way to resume training with allow instead of must"
    orig_class = wandb_utils.WandBLogger

    class PatchedWandBLogger(orig_class):
        """A helper class to log object using wandb."""

        def __init__(self, cfg):
            self.cfg = cfg.wandb
            self.log_dir = cfg.output_dir
            self.job_name = cfg.job_name
            self.env_fps = cfg.env.fps if cfg.env else None
            self._group = wandb_utils.cfg_to_group(cfg)

            # Set up WandB.
            os.environ["WANDB_SILENT"] = "True"
            import wandb

            wandb_run_id = (
                cfg.wandb.run_id
                if cfg.wandb.run_id
                else (
                    wandb_utils.get_wandb_run_id_from_filesystem(self.log_dir)
                    if cfg.resume
                    else None
                )
            )
            wandb.init(
                id=wandb_run_id,
                project=self.cfg.project,
                entity=self.cfg.entity,
                name=self.job_name,
                notes=self.cfg.notes,
                tags=wandb_utils.cfg_to_group(cfg, return_list=True),
                dir=self.log_dir,
                config=cfg.to_dict(),
                # TODO(rcadene): try set to True
                save_code=False,
                # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
                job_type="train_eval",
                resume="allow" if cfg.resume else None,
                mode=(
                    self.cfg.mode
                    if self.cfg.mode in ["online", "offline", "disabled"]
                    else "online"
                ),
            )
            run_id = wandb.run.id
            # NOTE: We will override the cfg.wandb.run_id with the wandb run id.
            # This is because we want to be able to resume the run from the wandb run id.
            cfg.wandb.run_id = run_id
            # Handle custom step key for rl asynchronous training.
            self._wandb_custom_step_key: set[str] | None = None
            print(
                wandb_utils.colored(
                    "Logs will be synced with wandb.", "blue", attrs=["bold"]
                )
            )
            logging.info(
                f"Track this run --> {wandb_utils.colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}"
            )
            self._wandb = wandb

    wandb_utils.WandBLogger = PatchedWandBLogger


class DINOv3BackboneWrapper(nn.Module):
    """Wrapper for DINOv3 backbone to make it compatible with ACT's expected interface."""

    def __init__(self, model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"):
        super().__init__()
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Auto-detect feature dimensions
        self.feature_dim = self.model.config.hidden_size  # 768 for base, 1024 for large
        self.patch_size = getattr(self.model.config, "patch_size", 14)
        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)

    def forward(self, x):
        """
        Convert DINOv3's patch-based features to spatial feature maps compatible with ACT.

        Args:
            x: Input tensor of shape (B, C, H, W) - already normalized by LeRobot

        Returns:
            dict with "feature_map" key containing tensor of shape (B, D, H', W')
        """
        batch_size, _, orig_h, orig_w = x.shape

        # Resize images to DINOv3's expected input size without changing normalization
        # DINOv3 expects 224x224 or 518x518 depending on the model
        target_size = self.processor.size["height"]

        if orig_h != target_size or orig_w != target_size:
            x_resized = torch.nn.functional.interpolate(
                x, size=(target_size, target_size), mode="bilinear", align_corners=False
            )
        else:
            x_resized = x

        # Forward through DINOv3 directly (inputs are already normalized by LeRobot)
        # Skip the processor since it would apply normalization again
        outputs = self.model(pixel_values=x_resized)
        features = outputs.last_hidden_state  # Shape: (B, N+1+num_register_tokens, D)

        # Remove CLS token and register tokens, keep only patch tokens
        patch_features = features[:, 1 + self.num_register_tokens :, :]  # (B, N, D)

        # Reshape to spatial format: (B, N, D) -> (B, D, H', W')
        B, N, D = patch_features.shape
        H_patches = W_patches = int(math.sqrt(N))  # Assuming square patch grid

        # Reshape to spatial feature map
        spatial_features = patch_features.transpose(1, 2).view(
            B, D, H_patches, W_patches
        )

        return {"feature_map": spatial_features}


def monkey_patch_dinov3_support():
    """Add DINOv3 backbone support to ACT."""

    # Patch ACT config validation
    original_post_init = ACTConfig.__post_init__

    def patched_post_init(self):
        # Call the original validation first (it calls super().__post_init__())
        # but temporarily bypass vision backbone check
        temp_backbone = self.vision_backbone
        if self.vision_backbone.startswith("dinov3"):
            self.vision_backbone = "resnet18"  # Temporary placeholder for validation

        # Call original post_init (this will handle all the other validations)
        try:
            original_post_init(self)
        finally:
            # Restore original backbone name
            self.vision_backbone = temp_backbone

        # Modified vision backbone validation to allow DINOv3
        if not (
            self.vision_backbone.startswith("resnet")
            or self.vision_backbone.startswith("dinov3")
        ):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants or DINOv3 models. Got {self.vision_backbone}."
            )

    ACTConfig.__post_init__ = patched_post_init

    # Patch ACT.__init__ to handle DINOv3 backbone
    original_act_init = ACT.__init__

    def patched_act_init(self, config: ACTConfig):
        # Store original backbone config for later use
        self._original_vision_backbone = config.vision_backbone

        if config.vision_backbone.startswith("dinov3"):
            # Temporarily set to a ResNet for original init, then override backbone
            config.vision_backbone = "resnet18"  # Temporary placeholder

        # Call original init
        original_act_init(self, config)

        # Override backbone if DINOv3 was requested
        if self._original_vision_backbone.startswith("dinov3"):
            if self.config.image_features:
                # Map dinov3 model names to HF model IDs
                dinov3_models = {
                    "dinov3-base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                    "dinov3-large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                    "dinov3-convnext": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
                }

                model_id = dinov3_models.get(
                    self._original_vision_backbone,
                    "facebook/dinov3-vitb16-pretrain-lvd1689m",
                )

                self.backbone = DINOv3BackboneWrapper(model_id)

                # Update the feature projection layer to match DINOv3 dimensions
                dinov3_feature_dim = self.backbone.feature_dim
                self.encoder_img_feat_input_proj = nn.Conv2d(
                    dinov3_feature_dim, config.dim_model, kernel_size=1
                )

        # Restore original config
        config.vision_backbone = self._original_vision_backbone

    ACT.__init__ = patched_act_init


def apply_patches():
    monkey_patch_policy_factory()
    monkey_patch_dataset()
    monkey_patch_save_checkpoint()
    monkey_patch_wandb()
    # monkey_patch_dinov3_support()
