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
from rich import print

from example_policies.robot_deploy.action_translator import ActionTranslator
from example_policies.robot_deploy.policy_loader import load_policy
from example_policies.robot_deploy.robot_io.robot_interface import (
    RobotClient,
    RobotInterface,
)
from example_policies.robot_deploy.robot_io.robot_service import (
    robot_service_pb2,
    robot_service_pb2_grpc,
)
from example_policies.robot_deploy.utils import print_info
from rich import print


def inference_loop(
    policy,
    cfg,
    data_dir: Path,
    hz: float,
    service_stub: robot_service_pb2_grpc.RobotServiceStub,
    controller=None,
    ep_index: int = 0,
    ask_for_input: bool = False,
):
    """Run policy inference using all live robot observations (state + images).

    Args:
        policy: The policy model to run inference with.
        cfg: Policy configuration.
        data_dir (Path): Path to the dataset directory (used only for timing/pacing).
        hz (float): Frequency to run the policy.
        service_stub: gRPC service stub.
        controller: Robot controller type.
        ep_index (int): Episode index to run.
        ask_for_input (bool): Whether to ask for user input at each action.
    """
    if controller is None:
        controller = RobotClient.CART_WAYPOINT

    print(f"The config is: {cfg}")

    fake_repo_id = data_dir.name

    # Load dataset (only for timing/frame count)
    dataset = LeRobotDataset(
        repo_id=fake_repo_id,
        root=data_dir,
        episodes=[ep_index],
        video_backend="pyav",
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=1,
        shuffle=False,
        drop_last=True,
    )

    robot_interface = RobotInterface(service_stub, cfg)
    model_to_action_trans = ActionTranslator(cfg)
    dbg_printer = print_info.InfoPrinter(cfg)

    # Wait for robot to be ready
    observation = None
    while not observation:
        observation = robot_interface.get_observation(cfg.device)
        time.sleep(0.1)

    print(f"Robot observation: {observation.keys()}")
    input("Press Enter to move robot to start...")
    robot_interface.move_home()

    # Dataset observation
    iterator = iter(dataloader)
    batch = next(iterator)

    dataset_observation = {
        key: value.to(cfg.device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

    print(f"Dataset observation: {dataset_observation.keys()}")

    input(
        "Press Enter to start policy inference with full robot observations (state + images from robot)..."
    )

    step = 0
    done = False

    # Inference Loop
    print("Starting policy inference loop with full robot observations...")
    period = 1.0 / hz

    while not done:
        start_time = time.time()

        # Get current robot observation (use this for everything)
        robot_observation = robot_interface.get_observation(cfg.device)

        if robot_observation is None:
            print(
                "[yellow]Warning: Failed to get robot observation, skipping step[/yellow]"
            )
            time.sleep(0.1)
            continue

        # Use full robot observation (both state and images)
        observation = dataset_observation.copy()

        # Replace all image observations with robot images
        image_keys = [key for key in robot_observation.keys() if "image" in key.lower()]
        for img_key in image_keys:
            print(f"Found image key: {img_key}")
            if img_key in robot_observation:
                print(f"Replacing observation key '{img_key}' with robot image")
                observation[img_key] = robot_observation[img_key]

        # Debug: Compare dataset state vs robot state
        dataset_state = dataset_observation["observation.state"].cpu().numpy().squeeze()
        robot_state = robot_observation["observation.state"].cpu().numpy().squeeze()

        print(f"Dataset state: {type(dataset_state)} , len={dataset_state.shape[0]}")
        print(f"Robot state:   {type(robot_state)}, len={robot_state.shape[0]}")
        print(f"State difference (robot - dataset): {robot_state - dataset_state}")
        # observation["observation.state"] = robot_observation["observation.state"]

        if ask_for_input:
            input("Press Enter to send next action...")

        # Predict the next action using the policy with full robot observation
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Translate action to robot coordinates
        # action = model_to_action_trans.translate(action, observation)

        print(f"\n=== POLICY OUTPUT FOR FULL ROBOT OBSERVATION ===")
        # dbg_printer.print(step, dataset_observation, action, raw_action=False)
        # dbg_printer.print(step, robot_observation, action, raw_action=False)

        # Send action to robot
        robot_interface.send_action(
            action,
            model_to_action_trans.action_mode,
            controller,
        )

        # Wait for execution to finish
        elapsed_time = time.time() - start_time
        sleep_duration = period - elapsed_time

        print(f"Sleep duration: {sleep_duration:.3f} s")
        time.sleep(max(0.0, sleep_duration))

        step += 1


def main():
    parser = argparse.ArgumentParser(
        description="Run policy inference with full robot observations (state + images) on real robot"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the policy checkpoint directory.",
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to the dataset directory (used only for frame count/timing)",
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
    parser.add_argument(
        "--hz",
        type=float,
        default=10.0,
        help="Frequency to run the policy (default: 10.0 Hz)",
    )
    parser.add_argument(
        "--step-by-step",
        action="store_true",
        help="Ask for user input at each action (default: False)",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="Do not display images (default: False)",
    )

    args = parser.parse_args()

    # Select device
    device = "cpu" if not torch.cuda.is_available() else "cuda"

    # Load policy
    policy, cfg = load_policy(args.checkpoint)
    policy.to(device)

    # Connect to robot service
    channel = grpc.insecure_channel(args.server)
    stub = robot_service_pb2_grpc.RobotServiceStub(channel)

    try:
        inference_loop(
            policy=policy,
            cfg=cfg,
            data_dir=args.data_dir,
            hz=args.hz,
            service_stub=stub,
            ep_index=args.episode,
            ask_for_input=args.step_by_step,
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        raise e
    finally:
        channel.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()
