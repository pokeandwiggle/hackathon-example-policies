#!/usr/bin/env python
# coding: utf-8

# # 🚀 Model Training
#
# This notebook guides you through the process of training a policy on your converted dataset.
#
# The process is broken down into a few simple steps:
# 1.  **Setup**: Apply necessary patches to the `lerobot` library.
# 2.  **Dataset**: Specify the path to your training data.
# 3.  **Configuration**: Select a model architecture and its hyperparameters.
# 4.  **Training**: Launch the training process.

# ---
# ### 1. Setup
#
# First, apply our custom patches to the `lerobot` library. This only needs to be done once per session.

# In[1]:


from example_policies import lerobot_patches

lerobot_patches.apply_patches()

# ---
# ### 2. Select Dataset
#
# > **Action Required:** Update `DATA_DIR` to point to the dataset you created in the previous notebook.

# In[3]:


import pathlib

# TODO: Set the path to your converted dataset directory.
DATA_DIR = pathlib.Path("../data/my_awesome_dataset")

# ---
# ### 3. Select Model Configuration
#
# We provide several pre-made configurations as a starting point. Uncomment the model you wish to use. You can also adjust parameters like `batch_size` as needed.

# In[ ]:


# Select one of the following configurations
from example_policies.config_factory import act_config, diffusion_config, smolvla_config

repo_id = "jccj/mh2_step_1"
cfg = act_config(
    repo_id=repo_id,
)

# You can specify additional keywords by looking at the lerobot configuration code, e.g. `lerobot.policies.act.configuration_act`
# and then adapt the code cell accordingly:
# ```python
# cfg = act_config(DATA_DIR, policy_kwargs={
#     optimizer_lr=1e-5
# })
# ```

# ---
# ### 4. Start Training
#
# This cell will start the training process. Metrics and logs will be streamed to the console, and if you have configured it, to Weights & Biases.

# In[ ]:


from example_policies.train import train

train(cfg)

#

#
