from lerobot.datasets.transforms import ImageTransformsConfig

from example_policies import lerobot_patches
from example_policies.config_factory import (
    act_config,
    diffusion_config,
    original_act_config,
    smolvla_config,
)

lerobot_patches.apply_patches()

repo_id = "jccj/mh2_step_3"
cfg = original_act_config(
    repo_id=repo_id,
    batch_size=16,
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
