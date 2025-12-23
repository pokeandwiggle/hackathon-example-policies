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

import datetime as dt
import pathlib
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig

from .robot_deploy.deploy_core.policy_loader import get_checkpoint_path
from .training.utils import create_dataset_config, shorten_name


@dataclass
class PolicyConfigBase(ABC):
    """Base class for all policy configurations."""

    # Common parameters
    dataset_root_dir: Optional[str | pathlib.Path] = None
    batch_size: int = 32
    lr: float = 1e-4
    steps: int = 100_000
    save_freq: int = 10_000
    resume_path: Optional[str] = None
    wandb_enable: bool = True
    wandb_entity: Optional[str] = None
    wandb_project: str = "lerobot"
    policy_kwargs: dict = field(default_factory=dict)
    pretrained_actions: bool = False
    build_exp_name_dir: bool = True

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the LeRobot policy model."""
        pass

    @property
    @abstractmethod
    def default_policy_kwargs(self) -> dict:
        """Default policy-specific kwargs."""
        pass

    def _build_exp_name_dir(self):
        """Build experiment name and directory based on dataset and model."""
        if not self.build_exp_name_dir:
            # Fallback to Lerobot Defaults
            return None, None

        # Shorten Dataset Name
        ds_root = pathlib.Path(self.dataset_root_dir)
        ds_name = ds_root.name
        short_ds_name = shorten_name(ds_name, max_word_length=4)

        # Timestamp
        now = dt.datetime.now()
        short_ts = f"{now:%y%m%d%H%M%S}"

        # Shorten Model Name
        model_name = self.model_name
        short_model_name = shorten_name(model_name, max_word_length=4)

        # Experiment Name & Directory relative to dataset root
        exp_name = f"{short_ds_name}_{short_model_name}_{short_ts}"
        exp_dir = ds_root.parent.parent / "models" / exp_name

        return exp_name, exp_dir

    def get_pretrained_config(self) -> Optional[PreTrainedConfig]:
        """Get pretrained config if needed."""
        if self.pretrained_actions:
            config = PreTrainedConfig.from_pretrained(f"lerobot/{self.model_name}")
            config.push_to_hub = False
            return config
        return None

    def build(self) -> TrainPipelineConfig:
        """Build the training configuration."""
        # Validate that dataset_root_dir has been set
        if self.dataset_root_dir is None or self.dataset_root_dir == "":
            raise ValueError(
                "dataset_root_dir must be set before calling build(). "
                "This should be automatically set from the --data_dir argument in train.py"
            )

        # Merge default and custom policy kwargs
        merged_policy_kwargs = {**self.default_policy_kwargs, **self.policy_kwargs}

        dataset_cfg, _ = create_dataset_config(pathlib.Path(self.dataset_root_dir))

        pretrained_config = self.get_pretrained_config()
        if pretrained_config is None:
            pretrained_config = PreTrainedConfig.get_choice_class(self.model_name)(
                push_to_hub=False, **merged_policy_kwargs
            )

        exp_name, exp_dir = self._build_exp_name_dir()

        cfg = TrainPipelineConfig(
            policy=pretrained_config,
            dataset=dataset_cfg,
            batch_size=self.batch_size,
            steps=self.steps,
            save_freq=self.save_freq,
            output_dir=exp_dir,
            job_name=exp_name,
        )

        cfg.wandb.enable = self.wandb_enable
        cfg.wandb.disable_artifact = True
        cfg.wandb.project = self.wandb_project
        cfg.wandb.entity = self.wandb_entity

        if self.lr is not None:
            cfg.policy.optimizer_lr = self.lr
            if hasattr(cfg.policy, "optimizer_lr_backbone"):
                cfg.policy.optimizer_lr_backbone = self.lr

        if self.resume_path is not None:
            resume_path = get_checkpoint_path(self.resume_path)
            cfg.resume = True
            cfg.checkpoint_dir = resume_path
            cfg.output_dir = resume_path.parent.parent.parent
            sys.argv.append(f"--config_path={resume_path / 'config.json'}")
            cfg.optimizer = cfg.policy.get_optimizer_preset()
            cfg.scheduler = cfg.policy.get_scheduler_preset()

        print("\nFinal Training Configuration (full details):")
        pprint(cfg)
        return cfg


@dataclass
class ACTConfig(PolicyConfigBase):
    """ACT Policy Configuration."""

    batch_size: int = 24
    lr: float = 2e-5
    steps: int = 800_000
    save_freq: int = 10_000

    @property
    def model_name(self) -> str:
        return "integrated_so3_act"

    @property
    def default_policy_kwargs(self) -> dict:
        return {
            "vision_backbone": "resnet34",
            "pretrained_backbone_weights": "ResNet34_Weights.IMAGENET1K_V1",
            "chunk_size": 30,
            "n_action_steps": 30,
            "latent_dim": 64,
            "n_decoder_layers": 7,
        }


@dataclass
class SmolVLAConfig(PolicyConfigBase):
    """SmolVLA Policy Configuration."""

    batch_size: int = 24
    lr: float = 1e-4
    steps: int = 100_000
    save_freq: int = 10_000

    @property
    def model_name(self) -> str:
        return "smolvla"

    @property
    def default_policy_kwargs(self) -> dict:
        return {
            "chunk_size": 20,
            "n_action_steps": 20,
        }

    def get_pretrained_config(self) -> Optional[PreTrainedConfig]:
        """Override to use smolvla_base."""
        if self.pretrained_actions:
            config = PreTrainedConfig.from_pretrained("lerobot/smolvla_base")
            config.push_to_hub = False
            return config
        return None


@dataclass
class DiffusionConfig(PolicyConfigBase):
    """Diffusion Policy Configuration."""

    batch_size: int = 96
    lr: float = 1e-4
    steps: int = 400_000
    save_freq: int = 8_000
    n_obs_steps: int = 2
    horizon: int = 32
    n_action_steps: int = 16

    @property
    def model_name(self) -> str:
        return "integrated_so3_diffusion"

    @property
    def default_policy_kwargs(self) -> dict:
        return {
            "vision_backbone": "resnet34",
            "crop_shape": (224, 224),
            "use_separate_rgb_encoder_per_camera": True,
            "down_dims": (128, 256, 512, 512),
            "kernel_size": 3,
            "n_groups": 8,
            "num_train_timesteps": 1000,
            "diffusion_step_embed_dim": 512,
            "prediction_type": "sample",
            "horizon": self.horizon,
            "n_action_steps": self.n_action_steps,
            "drop_n_last_frames": self.horizon
            - self.n_action_steps
            - self.n_obs_steps
            + 1,
        }


@dataclass
class Pi0Config(PolicyConfigBase):
    """Pi0 Policy Configuration."""

    batch_size: int = 1
    lr: float = 2.5e-5
    steps: int = 100_000
    save_freq: int = 10_000

    @property
    def model_name(self) -> str:
        return "pi0"

    @property
    def default_policy_kwargs(self) -> dict:
        return {
            "chunk_size": 30,
            "n_action_steps": 30,
        }


@dataclass
class DiTFlowConfig(PolicyConfigBase):
    """DiT Flow Policy Configuration."""

    batch_size: int = 64
    lr: float = 2e-4
    steps: int = 10_000
    save_freq: int = 5_000
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    # # Weight for SO3 Aware Trajectory integration loss. Recommended: 0.0 to disable. 0.01 to start.
    # integrated_so3_loss_weight: float = 0.0
    # # Weight for focal loss on termination signal. Recommended: 0.0 to disable. 10.0 to start.
    # termination_focal_loss_weight: float = 0.0
    # termination_focal_loss_index: int = -1

    @property
    def model_name(self) -> str:
        return "ditflow"

    @property
    def default_policy_kwargs(self) -> dict:
        return {
            "n_obs_steps": self.n_obs_steps,
            "horizon": self.horizon,
            "n_action_steps": self.n_action_steps,
            # "integrated_so3_loss_weight": self.integrated_so3_loss_weight,
            # "termination_focal_loss_weight": self.termination_focal_loss_weight,
            # "termination_focal_loss_index": self.termination_focal_loss_index,
        }
