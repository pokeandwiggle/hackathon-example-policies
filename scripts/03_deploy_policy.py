import pathlib

import wandb
from example_policies.robot_deploy import policy_loader
from example_policies.robot_deploy.deploy import deploy_policy

#!/usr/bin/env python
# coding: utf-8

# # 🚀 Deploy a Trained Policy
#
# This notebook guides you through deploying a trained policy to a physical robot.
#
# ### Process:
# 1.  **Configure**: Set the path to your trained model and the robot's server address.
# 2.  **Load**: The `policy_loader` automatically loads the model and its training configuration.
# 3.  **Deploy**: The `deploy_policy` function starts the inference loop and sends commands to the robot.
#
# The deployment script automatically handles details like the action space (`tcp`, `joint`, etc.) based on the loaded training configuration.

# ## 1. Configuration
#
# First, specify the necessary parameters for deployment. **You must edit these values.**

# In[ ]:


# TODO: Change to the directory containing your trained policy checkpoint.
# Example: "outputs/2025-09-14/12-00-00"
CHECKPOINT_DIR = pathlib.Path("outputs/vastai/080000/pretrained_model")
wandb_checkpoint_path = None
# data/output/checkpoints/last/pretrained_model
last_checkpoint_path = pathlib.Path(CHECKPOINT_DIR) / "checkpoints" / "last"
if wandb_checkpoint_path:
    run = wandb.init()
    artifact = run.use_artifact("jc-cj/uncategorized/060000:v0", type="dataset")
    artifact.download(root=str(last_checkpoint_path))

# TODO: Change to the robot's IP address.
SERVER_ENDPOINT = "192.168.0.207:50051"

# Inference frequency in Hz. Higher values result in smoother but potentially faster movements.
INFERENCE_FREQUENCY_HZ: float = 5.0

print(f"Attempting to load policy from: {CHECKPOINT_DIR}")
print(f"Robot server endpoint: {SERVER_ENDPOINT}")
print(f"Inference frequency: {INFERENCE_FREQUENCY_HZ} Hz")

# ## 2. Load the Policy
#
# Now, we load the policy from the specified checkpoint directory. The loader will find the latest checkpoint and its corresponding configuration file.

# In[ ]:


policy, cfg = policy_loader.load_policy(CHECKPOINT_DIR)

print("✅ Policy loaded successfully!")

# ## 3. (Optional) Modify Policy Attributes
#
# Before deployment, you can override policy attributes for experimentation. For example, you might want to adjust the action chunking (`n_action_steps`) to see how it affects robot behavior.

# In[ ]:


# Uncomment and modify the lines below to change policy attributes.
# For available options, refer to the lerobot policy's config documentation.

# Change the device on the config, not the policy!!
cfg.device = "cuda"
policy.to(cfg.device)  # or "cpu"
# policy.n_action_steps = 15  # Number of actions to predict in each forward pass

# print(f"Action steps set to: {policy.n_action_steps}")

# ## 4. Deploy to Robot
#
# Finally, execute the cell below to start sending commands to the robot.
#
# ⚠️ **Warning**: This will move the physical robot. Ensure the robot has a clear and safe workspace.

# In[ ]:


deploy_policy(policy, cfg, hz=INFERENCE_FREQUENCY_HZ, server=SERVER_ENDPOINT)
