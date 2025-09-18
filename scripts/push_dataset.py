from lerobot.datasets.lerobot_dataset import LeRobotDataset

from example_policies import lerobot_patches

lerobot_patches.apply_patches()


repo_id = "jccj/mh2_step_1_and_2"
dataset = LeRobotDataset(
    repo_id=repo_id,
    root="data/lerobot/step_1_and_2",
).push_to_hub()
