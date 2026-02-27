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
    video_backend: str = "pyav"  # "torchcodec" or "pyav"
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

        cfg.dataset.video_backend = self.video_backend

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
class Pi0PretrainedConfig(PolicyConfigBase):
    """Pi0 Policy Configuration with Pretrained Weights.
    
    π0 (Pi-Zero) is a Vision-Language-Action (VLA) model that uses:
    - PaliGemma vision-language backbone (SigLIP + Gemma)
    - Expert Gemma layers for action prediction
    - Flow-matching for action generation
    
    This config is specifically for FINE-TUNING from the pretrained
    `lerobot/pi0` checkpoint on HuggingFace. The pretrained model was
    trained on diverse robot manipulation data.
    
    Requires:
    - CUDA GPU (model runs on GPU only)
    - ~16GB+ VRAM for batch_size=1 (RTX 5090 with 32GB is excellent)
    - Hugging Face token with access to pi0 weights
    
    Action Space:
    - Works with absolute TCP (position + quaternion) action spaces
    - Automatically pads to max_state_dim/max_action_dim (default 32)
    
    Example usage in training notebook:
        config = Pi0PretrainedConfig(pretrained_actions=True)
    """

    batch_size: int = 1  # VLMs are memory-intensive, start small
    lr: float = 2.5e-5  # Lower LR for fine-tuning pretrained models
    steps: int = 30_000
    save_freq: int = 5_000
    
    # Action chunking (at 10Hz: 16 steps = 1.6s prediction, execute 8 = 0.8s)
    chunk_size: int = 16  # How many future actions to predict
    n_action_steps: int = 8  # How many to execute before re-planning
    
    # Fine-tuning settings
    freeze_vision_encoder: bool = True  # Freeze SigLIP vision encoder (saves memory)
    train_expert_only: bool = False  # Only train action expert (fastest)
    train_state_proj: bool = True  # Train state projection layer
    
    # Flow matching
    num_steps: int = 10  # Number of denoising steps during inference
    
    # Whether to load pretrained weights from HuggingFace
    pretrained_actions: bool = True  # Default to True for this config
    
    # HuggingFace model path for pretrained weights
    pretrained_model_path: str = "lerobot/pi0"

    @property
    def model_name(self) -> str:
        return "pi0"  # Uses pi0 from local LeRobot fork

    @property
    def default_policy_kwargs(self) -> dict:
        return {
            "chunk_size": self.chunk_size,
            "n_action_steps": self.n_action_steps,
            "freeze_vision_encoder": self.freeze_vision_encoder,
            "train_expert_only": self.train_expert_only,
            "train_state_proj": self.train_state_proj,
            "num_steps": self.num_steps,
        }

    def get_pretrained_config(self) -> Optional[PreTrainedConfig]:
        """Load pretrained Pi0 config from HuggingFace.
        
        The pretrained model is at 'lerobot/pi0' on HuggingFace Hub.
        We create a fresh config with our settings and set pretrained_path
        so that make_policy() will load the weights.
        """
        if self.pretrained_actions:
            # Import the policy class to use from_pretrained
            from lerobot.policies.pi0.configuration_pi0 import PI0Config
            
            # Create fresh config with our settings (don't load from HF config.json)
            # The weights will be loaded by make_policy() when pretrained_path is set
            config = PI0Config(
                chunk_size=self.chunk_size,
                n_action_steps=self.n_action_steps,
                freeze_vision_encoder=self.freeze_vision_encoder,
                train_expert_only=self.train_expert_only,
                train_state_proj=self.train_state_proj,
                num_steps=self.num_steps,
                push_to_hub=False,
            )
            # Set pretrained_path so make_policy() loads weights from HuggingFace
            config.pretrained_path = self.pretrained_model_path
            return config
        return None


@dataclass
class Pi0FastConfig(PolicyConfigBase):
    """Pi0-FAST Policy Configuration.
    
    Pi0-FAST uses autoregressive token prediction (FAST tokenizer) instead of
    flow matching. Generally faster inference but may require more training.
    
    Key differences from Pi0:
    - Uses FAST tokenizer for action generation
    - Autoregressive decoding instead of flow-matching
    - May be faster at inference time
    
    Requires:
    - CUDA GPU
    - Hugging Face token with access to pi0fast weights
    - transformers library with PaliGemma support
    """

    batch_size: int = 1
    lr: float = 1e-4  # PI0FASTConfig default
    steps: int = 30_000
    save_freq: int = 5_000
    
    # Action chunking (at 10Hz: 10 steps = 1s prediction, execute 5 = 0.5s)
    chunk_size: int = 10  # PI0FASTConfig default
    n_action_steps: int = 5  # PI0FASTConfig default
    
    # Fine-tuning settings
    freeze_vision_encoder: bool = True

    @property
    def model_name(self) -> str:
        return "pi0fast"  # Correct name in local LeRobot fork

    @property
    def default_policy_kwargs(self) -> dict:
        return {
            "chunk_size": self.chunk_size,
            "n_action_steps": self.n_action_steps,
            "freeze_vision_encoder": self.freeze_vision_encoder,
        }

    def get_pretrained_config(self) -> Optional[PreTrainedConfig]:
        """Pi0-FAST config - creates from scratch, weights loaded separately."""
        return None


@dataclass
class DiTFlowConfig(PolicyConfigBase):
    """DiT Flow Policy Configuration.

    Set ``use_chunk_relative_actions=True`` to convert absolute TCP actions
    to chunk-relative UMI-delta (20-dim) at training time.  When enabled,
    stepwise normalization is also applied and statistics are computed from
    the dataset parquet files using the configured ``horizon``.
    """

    batch_size: int = 64
    lr: float = 2e-4
    steps: int = 10_000
    save_freq: int = 5_000
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    # Chunk-relative UMI-delta conversion (opt-in)
    use_chunk_relative_actions: bool = False

    # Image preprocessing
    crop_shape: tuple[int, int] = (224, 224)
    crop_is_random: bool = True  # Random crop during training, center crop during eval

    # Vision backbone
    pretrained_backbone_weights: str | None = "IMAGENET1K_V1"  # Pretrained ImageNet weights
    use_group_norm: bool = True  # Replace BatchNorm with GroupNorm (Stanford approach)

    @property
    def model_name(self) -> str:
        return "ditflow"

    def _is_abs_tcp_dataset(self) -> bool:
        """Check whether the dataset uses absolute TCP actions.

        Returns ``True`` if the action feature names contain "tcp_"
        prefixed names (e.g. ``tcp_left_pos_x``).
        """
        import json

        info_path = pathlib.Path(self.dataset_root_dir) / "meta" / "info.json"
        if not info_path.exists():
            return False
        info = json.loads(info_path.read_text())
        names = info.get("features", {}).get("action", {}).get("names", [])
        return any(n.startswith("tcp_") for n in names)

    def _get_obs_tcp_indices(self) -> dict[str, list[int]]:
        """Look up TCP pose indices in observation.state from info.json.

        Returns a dict with keys ``obs_tcp_{left,right}_{pos,quat}_indices``,
        each mapping to a list of integer indices into the state vector.
        """
        import json

        info_path = pathlib.Path(self.dataset_root_dir) / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        state_names = (
            info.get("features", {}).get("observation.state", {}).get("names", [])
        )
        if not state_names:
            raise ValueError(
                "observation.state feature names not found in info.json. "
                "Cannot determine TCP pose indices for chunk-relative conversion."
            )

        def _find_indices(prefix: str, count: int) -> list[int]:
            start = state_names.index(prefix)
            return list(range(start, start + count))

        return {
            "obs_tcp_left_pos_indices": _find_indices("tcp_left_pos_x", 3),
            "obs_tcp_left_quat_indices": _find_indices("tcp_left_quat_x", 4),
            "obs_tcp_right_pos_indices": _find_indices("tcp_right_pos_x", 3),
            "obs_tcp_right_quat_indices": _find_indices("tcp_right_quat_x", 4),
        }

    @property
    def default_policy_kwargs(self) -> dict:
        base = {
            "n_obs_steps": self.n_obs_steps,
            "horizon": self.horizon,
            "n_action_steps": self.n_action_steps,
            "crop_shape": self.crop_shape,
            "crop_is_random": self.crop_is_random,
            "pretrained_backbone_weights": self.pretrained_backbone_weights,
            "use_group_norm": self.use_group_norm,
        }

        if self.use_chunk_relative_actions:
            if not self._is_abs_tcp_dataset():
                raise ValueError(
                    "use_chunk_relative_actions=True requires a dataset with "
                    "absolute TCP actions (feature names starting with 'tcp_')."
                )
            # ── Absolute TCP dataset → chunk-relative conversion at training time ──
            from .utils.action_order import UMI_ROTATION_FEATURE_INDICES
            from .utils.compute_stepwise_stats import (
                compute_stepwise_stats_from_parquet,
            )

            tcp_indices = self._get_obs_tcp_indices()

            stats_path = compute_stepwise_stats_from_parquet(
                self.dataset_root_dir,
                horizon=self.horizon,
                obs_tcp_left_pos_indices=tcp_indices["obs_tcp_left_pos_indices"],
                obs_tcp_left_quat_indices=tcp_indices["obs_tcp_left_quat_indices"],
                obs_tcp_right_pos_indices=tcp_indices["obs_tcp_right_pos_indices"],
                obs_tcp_right_quat_indices=tcp_indices["obs_tcp_right_quat_indices"],
            )

            base.update(
                {
                    "use_chunk_relative_actions": True,
                    **tcp_indices,
                    "use_stepwise_normalization": True,
                    "stepwise_stats_path": stats_path,
                    "stepwise_skip_feature_indices": [], #list(UMI_ROTATION_FEATURE_INDICES),
                    "stepwise_clip_min": -1.5,
                    "stepwise_clip_max": 1.5,
                    "clip_sample_range": 1.5,
                }
            )
        return base
