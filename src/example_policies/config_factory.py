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

import pathlib
import sys
from pprint import pprint

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.transforms import ImageTransformsConfig

from .robot_deploy.policy_loader import get_checkpoint_path
from .training.utils import create_dataset_config


def create_lerobot_config(
    model_name: str,
    dataset_root_dir: str | None = None,
    repo_id: str | None = None,
    image_transforms: ImageTransformsConfig | None = None,
    pretrained_config: PreTrainedConfig | None = None,
    batch_size: int = 8,
    lr: float = None,
    steps: int = 100_000,
    enable_wandb: bool = True,
    resume_path: str = None,
    policy_kwargs: dict | None = None,
    save_freq: int = 10_000,
):
    """Create a Training Configuration for LeRobot Predefined Models

    Args:
        model_name (str): Name of LeRobot Policy Model. Examples: "act", "diffusion", "pi0", "smolvla"
        dataset_root_dir (str): Root directory of the custom dataset.
        batch_size (int, optional): Batch size for training. Defaults to 8.
        lr (float, optional): Learning rate for the optimizer. Defaults to None.
        steps (int, optional): Number of training steps. Defaults to 100000.
        enable_wandb (bool, optional): Whether to enable Weights & Biases logging. Defaults to True.

    Returns:
        _type_: _description_
    """
    assert repo_id is not None or dataset_root_dir is not None, (
        "Either repo_id or dataset_root_dir must be provided"
    )
    if policy_kwargs is None:
        policy_kwargs = {}

    dataset_root_dir = (
        pathlib.Path(dataset_root_dir) if dataset_root_dir is not None else None
    )
    dataset_cfg, features = create_dataset_config(
        dataset_root_dir=dataset_root_dir,
        repo_id=repo_id,
        image_transforms=image_transforms,
    )

    if pretrained_config is None:
        pretrained_config = PreTrainedConfig.get_choice_class(model_name)(
            push_to_hub=False, **policy_kwargs
        )

    cfg = TrainPipelineConfig(
        policy=pretrained_config,
        dataset=dataset_cfg,
        batch_size=batch_size,
        steps=steps,
        save_freq=save_freq,
    )
    cfg.wandb.enable = enable_wandb
    cfg.wandb.disable_artifact = False

    if lr is not None:
        cfg.policy.optimizer_lr = lr
        if hasattr(cfg.policy, "optimizer_lr_backbone"):
            cfg.policy.optimizer_lr_backbone = lr

    if resume_path is not None:
        resume_path = get_checkpoint_path(resume_path)
        cfg.resume = True
        cfg.checkpoint_dir = resume_path
        cfg.output_dir = resume_path.parent.parent.parent
        sys.argv.append(f"--config_path={resume_path / 'config.json'}")
        cfg.optimizer = cfg.policy.get_optimizer_preset()
        cfg.scheduler = cfg.policy.get_scheduler_preset()

    print("\nFinal Training Configuration (full details):")
    pprint(cfg)
    return cfg


def original_act_config(
    dataset_root_dir: str | None = None,
    repo_id: str | None = None,
    lr: float = 2e-5,
    batch_size: int = 24,
    resume_path: str = None,
    policy_kwargs: dict = None,
):
    assert repo_id is not None or dataset_root_dir is not None, (
        "Either repo_id or dataset_root_dir must be provided"
    )
    default_kwargs = {
        "vision_backbone": "resnet18",
        "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
        "chunk_size": 20,
        "n_action_steps": 10,
        "latent_dim": 32,
        "n_decoder_layers": 1,
    }

    if policy_kwargs is not None:
        default_kwargs.update(policy_kwargs)
    policy_kwargs = default_kwargs

    cfg = create_lerobot_config(
        # Model selection: e.g., "act", "diffusion", "pi0", "smolvla"
        model_name="act",
        # Path to the LeRobot dataset directory
        dataset_root_dir=dataset_root_dir,
        repo_id=repo_id,
        # Training hyperparameters
        batch_size=batch_size,
        lr=lr,
        steps=800_000,
        save_freq=5000,
        # Enable Weights & Biases for experiment tracking
        enable_wandb=True,
        resume_path=resume_path,
        policy_kwargs=policy_kwargs,
    )
    return cfg


def act_config(
    repo_id: str | None = None,
    dataset_root_dir: str | None = None,
    image_transforms: ImageTransformsConfig | None = None,
    batch_size: int = 8,
    resume_path: str = None,
    policy_kwargs: dict = None,
):
    assert repo_id is not None or dataset_root_dir is not None, (
        "Either repo_id or dataset_root_dir must be provided"
    )
    default_kwargs = {
        "vision_backbone": "resnet34",
        "pretrained_backbone_weights": "ResNet34_Weights.IMAGENET1K_V1",
        "chunk_size": 30,
        "n_action_steps": 30,
        "latent_dim": 64,
        "n_decoder_layers": 7,
    }

    if policy_kwargs is not None:
        default_kwargs.update(policy_kwargs)
    policy_kwargs = default_kwargs

    cfg = create_lerobot_config(
        # Model selection: e.g., "act", "diffusion", "pi0", "smolvla"
        model_name="integrated_so3_act",
        # Path to the LeRobot dataset directory
        dataset_root_dir=dataset_root_dir,
        repo_id=repo_id,
        image_transforms=image_transforms,
        # Training hyperparameters
        batch_size=batch_size,
        lr=2e-5,
        steps=800_000,
        save_freq=5_000,
        # Enable Weights & Biases for experiment tracking
        enable_wandb=True,
        resume_path=resume_path,
        policy_kwargs=policy_kwargs,
    )
    return cfg


def smolvla_config(
    dataset_root_dir: str,
    batch_size: int = 24,
    resume_path: str = None,
    policy_kwargs: dict = None,
    pretrained_actions: bool = False,
):
    default_kwargs = {
        "chunk_size": 20,
        "n_action_steps": 20,
    }

    if policy_kwargs is not None:
        default_kwargs.update(policy_kwargs)
    policy_kwargs = default_kwargs

    policy = None
    if pretrained_actions:
        policy = PreTrainedConfig.from_pretrained("lerobot/smolvla_base")
        policy.push_to_hub = False

    cfg = create_lerobot_config(
        # Model selection: e.g., "act", "diffusion", "pi0", "smolvla"
        model_name="smolvla",
        dataset_root_dir=dataset_root_dir,
        pretrained_config=policy,
        # Training hyperparameters
        batch_size=batch_size,
        lr=1e-4,
        steps=100_000,
        save_freq=10_000,
        # Enable Weights & Biases for experiment tracking
        enable_wandb=True,
        resume_path=resume_path,
        policy_kwargs=policy_kwargs,
    )
    return cfg


def diffusion_config(
    dataset_root_dir: str,
    batch_size: int = 96,
    resume_path: str = None,
    policy_kwargs: dict = None,
):
    # Diffusion Policy settings:
    n_obs_steps: int = 2
    horizon: int = 32
    n_action_steps: int = 16

    default_kwargs = {
        "vision_backbone": "resnet34",
        # "pretrained_backbone_weights": "ResNet34_Weights.IMAGENET1K_V1",
        "crop_shape": (224, 224),
        "use_separate_rgb_encoder_per_camera": True,
        "down_dims": (128, 256, 512, 512),
        "kernel_size": 3,
        "n_groups": 8,
        "num_train_timesteps": 1000,
        "diffusion_step_embed_dim": 512,
        "prediction_type": "sample",
        # "n_obs_steps": n_obs_steps,
        "horizon": horizon,
        "n_action_steps": n_action_steps,
        "drop_n_last_frames": horizon - n_action_steps - n_obs_steps + 1,
    }

    if policy_kwargs is not None:
        default_kwargs.update(policy_kwargs)
    policy_kwargs = default_kwargs

    cfg = create_lerobot_config(
        # Model selection: e.g., "act", "diffusion", "pi0", "smolvla"
        model_name="integrated_so3_diffusion",
        # Path to the LeRobot dataset directory
        dataset_root_dir=dataset_root_dir,
        # Training hyperparameters
        batch_size=batch_size,
        lr=1e-4,
        steps=400_000,
        save_freq=8_000,
        # Enable Weights & Biases for experiment tracking
        enable_wandb=True,
        resume_path=resume_path,
        # Additional Policy Keywords
        policy_kwargs=policy_kwargs,
    )
    return cfg


def pi0_config(
    dataset_root_dir: str,
    batch_size: int = 1,
    resume_path: str = None,
    pretrained_actions: bool = False,
):
    policy = None
    if pretrained_actions:
        policy = PreTrainedConfig.from_pretrained("lerobot/pi0")
        policy.push_to_hub = False
    cfg = create_lerobot_config(
        # Model selection: e.g., "act", "diffusion", "pi0", "smolvla"
        model_name="pi0",
        dataset_root_dir=dataset_root_dir,
        pretrained_config=policy,
        # Training hyperparameters
        batch_size=batch_size,
        lr=2.5e-5,
        steps=100_000,
        save_freq=10_000,
        # Enable Weights & Biases for experiment tracking
        enable_wandb=True,
        resume_path=resume_path,
        # Additional Policy Keywords
        policy_kwargs={
            "chunk_size": 30,
            "n_action_steps": 30,
        },
    )

    return cfg
