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
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from .robot_deploy.policy_loader import get_checkpoint_path
from .training.utils import create_dataset_config


def get_dataset_info(dataset_root_dir: str) -> dict:
    """Get information about a LeRobot dataset.
    
    Args:
        dataset_root_dir: Root directory of the dataset.
        
    Returns:
        Dictionary containing:
            - total_frames: Total number of frames in the dataset
            - total_episodes: Total number of episodes
            - fps: Frames per second
    """
    data_dir = pathlib.Path(dataset_root_dir)
    fake_repo_id = data_dir.name
    
    metadata = LeRobotDatasetMetadata(
        repo_id=fake_repo_id,
        root=data_dir,
    )
    
    return {
        "total_frames": metadata.total_frames,
        "total_episodes": metadata.total_episodes,
        "fps": metadata.fps,
    }


def epochs_to_steps(epochs: int, dataset_size: int, batch_size: int) -> int:
    """Convert number of epochs to training steps.
    
    Args:
        epochs: Number of epochs to train.
        dataset_size: Total number of samples in the dataset.
        batch_size: Batch size for training.
        
    Returns:
        Number of training steps.
    """
    steps_per_epoch = dataset_size // batch_size
    return epochs * steps_per_epoch


def create_lerobot_config(
    model_name: str,
    dataset_root_dir: str,
    pretrained_config: PreTrainedConfig | None = None,
    batch_size: int = 8,
    lr: float = None,
    steps: int = None,
    epochs: int = None,
    enable_wandb: bool = True,
    resume_path: str = None,
    policy_kwargs: dict | None = None,
    save_freq_epochs: int = 100,
):
    """Create a Training Configuration for LeRobot Predefined Models

    Args:
        model_name (str): Name of LeRobot Policy Model. Examples: "act", "diffusion", "pi0", "smolvla"
        dataset_root_dir (str): Root directory of the custom dataset.
        batch_size (int, optional): Batch size for training. Defaults to 8.
        lr (float, optional): Learning rate for the optimizer. Defaults to None.
        steps (int, optional): Number of training steps. Defaults to None.
        epochs (int, optional): Number of training epochs. If provided, overrides steps. Defaults to None.
        enable_wandb (bool, optional): Whether to enable Weights & Biases logging. Defaults to True.
        resume_path (str, optional): Path to checkpoint to resume from. Defaults to None.
        policy_kwargs (dict, optional): Additional policy configuration. Defaults to None.
        save_freq_epochs (int, optional): Save checkpoint every N epochs. Defaults to 100.

    Returns:
        TrainPipelineConfig: The training configuration.
        
    Note:
        Either `steps` or `epochs` must be provided. If both are provided, `epochs` takes precedence.
    """
    if policy_kwargs is None:
        policy_kwargs = {}

    dataset_cfg, features = create_dataset_config(pathlib.Path(dataset_root_dir))
    
    # Get dataset info for epoch calculations
    dataset_info = get_dataset_info(dataset_root_dir)
    dataset_size = dataset_info["total_frames"]
    steps_per_epoch = dataset_size // batch_size
    
    # Calculate save_freq in steps from epochs
    save_freq = save_freq_epochs * steps_per_epoch
    
    # Calculate steps from epochs if provided
    if epochs is not None:
        training_steps = epochs_to_steps(epochs, dataset_size, batch_size)
        print(f"\n📊 Training by epochs:")
        print(f"   - Dataset size: {dataset_size} frames")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Epochs: {epochs}")
        print(f"   - Calculated steps: {training_steps}")
        print(f"   - Save every: {save_freq_epochs} epochs ({save_freq} steps)")
    elif steps is not None:
        training_steps = steps
        print(f"\n📊 Training by steps:")
        print(f"   - Steps: {training_steps}")
        print(f"   - Save every: {save_freq_epochs} epochs ({save_freq} steps)")
    else:
        # Default to 200 epochs
        default_epochs = 200
        training_steps = epochs_to_steps(default_epochs, dataset_size, batch_size)
        print(f"\n📊 Training with default epochs:")
        print(f"   - Dataset size: {dataset_size} frames")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Epochs: {default_epochs} (default)")
        print(f"   - Calculated steps: {training_steps}")
        print(f"   - Save every: {save_freq_epochs} epochs ({save_freq} steps)")

    if pretrained_config is None:
        pretrained_config = PreTrainedConfig.get_choice_class(model_name)(
            push_to_hub=False, **policy_kwargs
        )

    cfg = TrainPipelineConfig(
        policy=pretrained_config,
        dataset=dataset_cfg,
        batch_size=batch_size,
        steps=training_steps,
        save_freq=save_freq,
    )
    cfg.wandb.enable = enable_wandb
    cfg.wandb.disable_artifact = True

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


def act_config(
    dataset_root_dir: str,
    batch_size: int = 24,
    epochs: int = 200,
    resume_path: str = None,
    policy_kwargs: dict = None,
):

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
        # Training hyperparameters
        batch_size=batch_size,
        lr=2e-5,
        epochs=epochs,
        save_freq_epochs=100,
        # Enable Weights & Biases for experiment tracking
        enable_wandb=True,
        resume_path=resume_path,
        policy_kwargs=policy_kwargs,
    )
    return cfg


def smolvla_config(
    dataset_root_dir: str,
    batch_size: int = 24,
    epochs: int = 200,
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
        epochs=epochs,
        save_freq_epochs=100,
        # Enable Weights & Biases for experiment tracking
        enable_wandb=True,
        resume_path=resume_path,
        policy_kwargs=policy_kwargs,
    )
    return cfg


def diffusion_config(
    dataset_root_dir: str,
    batch_size: int = 96,
    epochs: int = 200,
    resume_path: str = None,
    policy_kwargs: dict = None,
):
    # Diffusion Policy settings:
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    default_kwargs = {
        "vision_backbone": "resnet18",
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
        epochs=epochs,
        save_freq_epochs=100,
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
    epochs: int = 200,
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
        epochs=epochs,
        save_freq_epochs=100,
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


def dit_flow_config(
    dataset_root_dir: str,
    batch_size: int = 64,
    epochs: int = 200,
    resume_path: str = None,
    policy_kwargs: dict = None,
):
    # Diffusion Policy settings:
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    default_kwargs = {}

    if policy_kwargs is not None:
        default_kwargs.update(policy_kwargs)
    policy_kwargs = default_kwargs

    cfg = create_lerobot_config(
        # Model selection: e.g., "act", "diffusion", "pi0", "smolvla"
        model_name="ditflow",
        # Path to the LeRobot dataset directory
        dataset_root_dir=dataset_root_dir,
        # Training hyperparameters
        batch_size=batch_size,
        lr=2e-4,
        epochs=epochs,
        save_freq_epochs=100,
        # Enable Weights & Biases for experiment tracking
        enable_wandb=True,
        resume_path=resume_path,
        # Additional Policy Keywords
        policy_kwargs=policy_kwargs,
    )
    return cfg


def beso_config(
    dataset_root_dir: str,
    batch_size: int = 96,
    epochs: int = 200,
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
        model_name="beso",
        # Path to the LeRobot dataset directory
        # pretrained_config=beso_cfg,
        dataset_root_dir=dataset_root_dir,
        # Training hyperparameters
        batch_size=batch_size,
        lr=1e-4,
        epochs=epochs,
        save_freq_epochs=100,
        # Enable Weights & Biases for experiment tracking
        enable_wandb=True,
        resume_path=resume_path,
        # Additional Policy Keywords
        policy_kwargs=policy_kwargs,
    )
    return cfg
