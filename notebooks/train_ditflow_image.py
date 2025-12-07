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
DATA_DIR = pathlib.Path("./sort_red_blocks_100")

# Select one of the following configurations
from example_policies.config_factory import diffusion_config, dit_flow_config, dit_flow_image_config

# cfg = dit_flow_config(DATA_DIR, enable_wandb=False)
resume_path = "/root/autodl-tmp/outputs/ditflow_image_sort_red_blocks_100_long/checkpoints/last/pretrained_model/"
cfg = dit_flow_image_config(DATA_DIR, enable_wandb=False, resume_path=resume_path)

# Disable multiprocessing if there are issues with dataloader workers
cfg.num_workers = 16

# cfg.steps = 

cfg.job_name = "ditflow_image_sort_red_blocks_100_long" # "ditflow_sort_red_blocks_80"
cfg.output_dir = pathlib.Path("/root/autodl-tmp/outputs/ditflow_image_sort_red_blocks_100_long/")
cfg.save_freq = 5000
# cfg.steps = 20000
# cfg.log_freq = 100
# cfg.save_freq = 5
# cfg.steps = 10
cfg.log_freq = 100

cfg.batch_size = 128

cfg.wandb.enable = True
cfg.wandb.project = "paper"
cfg.wandb.entity = "470620104-technical-university-of-munich"

# print(cfg.to_dict())

from example_policies.train import train

# Set video backend in config
cfg.dataset.video_backend = "pyav"

train(cfg)