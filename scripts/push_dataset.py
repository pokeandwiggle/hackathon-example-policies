from lerobot.datasets.lerobot_dataset import LeRobotDataset

from example_policies import lerobot_patches

lerobot_patches.apply_patches()


repo_id = "jccj/mh2_step_1"
dataset = LeRobotDataset(
    repo_id=repo_id, root="data/lerobot/filtered_dataset"
).push_to_hub()
