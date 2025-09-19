from lerobot.datasets.transforms import ImageTransformsConfig

from example_policies import lerobot_patches
from example_policies.config_factory import (
    act_config,
    diffusion_config,
    original_act_config,
    smolvla_config,
)
from example_policies.train import train

lerobot_patches.apply_patches()

# repo_id = "jccj/mh2_step_1_and_2"
dataset_root_dir = "data/lerobot/step_1_2_and_3"
cfg = original_act_config(
    # repo_id=repo_id,
    dataset_root_dir=dataset_root_dir,
    batch_size=16,
)

cfg.policy.pretrained_path = (
    "outputs/train/2025-09-18/23-01-44_act/checkpoints/last/pretrained_model/"
)


train(cfg)
