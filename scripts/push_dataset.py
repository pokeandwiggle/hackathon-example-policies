from lerobot.datasets.lerobot_dataset import LeRobotDataset

from example_policies import lerobot_patches

lerobot_patches.apply_patches()


repo_id = "jccj/mh2_step_2_pre_clean"
dataset = LeRobotDataset(
    repo_id=repo_id, root="data/lerobot/step_2_pre_clean"
).push_to_hub()
