#!/usr/bin/env python
# coding: utf-8

# # 💿 Dataset Conversion
#
# This notebook converts raw robot recordings (`.mcap` files) into the LeRobot format required for training.
#
# The process involves:
# 1.  **Exploring** the available raw data.
# 2.  **Configuring** the dataset parameters (e.g., observations, actions).
# 3.  **Running** the conversion script.

# ---
# ## 1. Explore Raw Data
#
# First, let's list the available raw data directories. Each directory contains a set of `.mcap` files from different teleoperation sessions.

# In[ ]:


# ---
# ## 2. Configure Conversion
#
# Now, specify the input and output paths and define the dataset's structure.
#
# > **Action Required:** Update `RAW_DATA_DIR` and `OUTPUT_DIR` below.

# In[ ]:


import pathlib

from example_policies.data_ops.config.pipeline_config import ActionLevel, PipelineConfig

# --- Paths ---
# TODO: Set the input directory containing your .mcap files.
RAW_DATA_DIR = pathlib.Path("data/wandb/filtered_artifacts")

# TODO: Set your desired output directory name.
OUTPUT_DIR = pathlib.Path("data/lerobot/filtered_dataset")

# --- Configuration ---
# TODO: A descriptive label for the task, used for VLA-style text conditioning.
TASK_LABEL = "step_1"

cfg = PipelineConfig(
    include_joint_positions=True,
    task_name=TASK_LABEL,
    # Observation features to include in the dataset.
    include_tcp_poses=True,
    include_rgb_images=True,
    include_depth_images=True,
    # Action representation. DELTA_TCP is a good default.
    action_level=ActionLevel.DELTA_TCP,
    image_resolution=(512, 512),
    # Subsampling and filtering. These are task-dependent.
    target_fps=10,
    max_pause_seconds=60 * 10,
    min_episode_seconds=1,
    save_normal=False,
    save_pauses=False,
    pause_velocity=-1,
)

print(f"Input path:  {RAW_DATA_DIR}")
print(f"Output path: {OUTPUT_DIR}")

# ---
# ## 3. Run Conversion
#
# This cell executes the conversion process. It may take a while depending on the size of your data. You will see progress updates printed below.

# In[ ]:


from example_policies import lerobot_patches
from example_policies.data_ops.dataset_conversion import convert_episodes

lerobot_patches.apply_patches()

convert_episodes(RAW_DATA_DIR, OUTPUT_DIR, cfg)

# ---
# ## ✅ Done!
#
# Your new dataset is ready at the output path you specified. You can now proceed to the next notebook to train a policy.
