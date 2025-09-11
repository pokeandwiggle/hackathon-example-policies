# Copyright 2025 Poke & Wiggle GmbH. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
from pathlib import Path

import grpc

# Lerobot Environment Bug
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ex_pol.robot_deploy.action_translator import ActionTranslator
from ex_pol.robot_deploy.debug_helpers.utils import print_info
from ex_pol.robot_deploy.policy_loader import load_policy
from ex_pol.robot_deploy.robot_io.robot_interface import RobotInterface
from ex_pol.robot_deploy.robot_io.robot_service import (
    robot_service_pb2,
    robot_service_pb2_grpc,
)


def inference_loop(
    data_dir: Path, checkpoint_dir: Path, service_stub: robot_service_pb2_grpc.RobotServiceStub, ep_index: int = 0
):

    policy, cfg = load_policy(checkpoint_dir)
    robot_interface = RobotInterface(service_stub, cfg)
    model_to_action_trans = ActionTranslator(cfg)

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset(
        repo_id=data_dir.name,
        root=data_dir,
        episodes=[ep_index],
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=1,
        shuffle=False,
        drop_last=True,
    )

    step = 0
    done = False

    iterator = iter(dataloader)

    batch = next(iterator)
    state = batch["observation.state"]
    action = np.concatenate([state[0, :14].cpu().numpy(), [0, 0]]).astype(np.float32)

    # add batch axis
    action = action[None, :]

    observation = None
    while not observation:
        observation = robot_interface.get_observation("cpu")
        time.sleep(0.1)

    print_info(step, observation, action)

    input("Press Enter to move robot to start...")

    robot_interface.send_action(torch.from_numpy(action))

    input("Press Enter to continue...")
    # Inference Loop
    print("Starting inference loop...")
    hz = 1.0
    period = 1.0 / hz
    while not done:
        start_time = time.time()
        observation = robot_interface.get_observation("cpu")

        if observation:
            batch = next(iterator)

            action = batch["action"]

            action = model_to_action_trans.translate(action, observation)
            print_info(step, observation, action)

            robot_interface.send_action(action)
            # policy._queues["action"].clear()

        # wait for execution to finish
        elapsed_time = time.time() - start_time
        sleep_duration = period - elapsed_time
        print(sleep_duration)
        # wait for input
        # input("Press Enter to continue...")
        time.sleep(max(0.0, sleep_duration))

        step += 1


def main():
    parser = argparse.ArgumentParser(description="Robot service client")
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to the data directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the policy checkpoint directory.",
    )

    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="Robot service server address (default: localhost:50051)",
    )

    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to run (default: 0)",
    )

    args = parser.parse_args()

    channel = grpc.insecure_channel(args.server)
    stub = robot_service_pb2_grpc.RobotServiceStub(channel)
    try:
        inference_loop(args.data_dir, args.checkpoint, stub, args.episode)
    except Exception as e:
        print(f"Error occurred: {e}")
        raise e
    finally:
        channel.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()
