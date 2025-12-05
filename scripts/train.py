# Set environment variables BEFORE any imports
import os
import warnings

os.environ["LEROBOT_VIDEO_BACKEND"] = "pyav"

# Suppress torchvision video deprecation warning
warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities.*")

from example_policies import lerobot_patches

lerobot_patches.apply_patches()

import pathlib

# TODO: Set the path to your converted dataset directory.
DATA_DIR = pathlib.Path("/home/yizhang/Projects/hackathon-ki-fabrik/data/stack_red_blocks_85")


# Select one of the following configurations
from example_policies.config_factory import diffusion_config, dit_flow_config, dit_flow_image_config

cfg = dit_flow_image_config(DATA_DIR, enable_wandb=False)

# Disable multiprocessing if there are issues with dataloader workers
# cfg.num_workers = 0

# cfg.log_freq = 1
cfg.save_freq = 5000
cfg.steps = 40000


cfg.policy.optimizer_lr = 2e-4

cfg.job_name = "ditflow_image_stack_red_blocks_85_40000"
cfg.output_dir = pathlib.Path("/home/yizhang/Projects/hackathon-ki-fabrik/outputs/ditflow_image_stack_red_blocks_85_40000")
cfg.wandb.enable = True
cfg.wandb.project = "paper"
cfg.wandb.entity = "470620104-technical-university-of-munich"


from example_policies.train import train

# Set video backend in config
cfg.dataset.video_backend = "pyav"

train(cfg)
