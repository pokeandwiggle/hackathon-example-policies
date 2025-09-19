# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import time

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
)
from lerobot.policies.factory import make_policy
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    # so100_follower,
)
# from lerobot.scripts.rl.gym_manipulator import make_robot_env
from example_policies.rl.gym_manipulator import make_robot_env
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    # so101_leader,  # noqa: F401
)
from example_policies.rl.robot import RobotIO  # noqa: F401
from lerobot.utils.robot_utils import busy_wait

logging.basicConfig(level=logging.INFO)


def eval_policy(env, policy, n_episodes, fps=None):
    sum_reward_episode = []
    for _ in range(n_episodes):
        policy.reset()
        obs, _ = env.reset()
        episode_reward = 0.0
        while True:
            start_time = time.perf_counter()
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break

            if fps is not None:
                dt_time = time.perf_counter() - start_time
                busy_wait(1 / fps - dt_time)

        sum_reward_episode.append(episode_reward)

    logging.info(f"Success after 20 steps {sum_reward_episode}")
    logging.info(f"success rate {sum(sum_reward_episode) / len(sum_reward_episode)}")


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    env_cfg = cfg.env
    env = make_robot_env(env_cfg)
    # dataset_cfg = cfg.dataset
    # dataset = LeRobotDataset(repo_id=dataset_cfg.repo_id)
    # dataset_meta = dataset.meta

    if env_cfg.pretrained_policy_name_or_path is not None:
        cfg.policy.pretrained_path = env_cfg.pretrained_policy_name_or_path
    else:
        # Construct path to the last checkpoint directory
        checkpoint_dir = os.path.join(cfg.output_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
        logging.info(f"Loading training state from {checkpoint_dir}")

        pretrained_policy_name_or_path = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR)
        cfg.policy.pretrained_path = pretrained_policy_name_or_path

    logging.info(f"Using pretrained policy from {cfg.policy.pretrained_path}")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        # ds_meta=dataset_meta,
    )
    # policy.from_pretrained(env_cfg.pretrained_policy_name_or_path)
    policy.eval()

    eval_policy(env, policy=policy, n_episodes=10, fps=env_cfg.fps)


if __name__ == "__main__":
    main()
