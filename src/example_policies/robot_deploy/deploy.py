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
from example_policies.robot_deploy.utils.action_mode import ActionMode


def inference_loop(
    policy,
    cfg,
    hz: float,
    service_stub: robot_service_pb2_grpc.RobotServiceStub,
    controller=None,
):
    if controller is None:
        controller = RobotClient.CART_WAYPOINT

    print(f"The config is: {cfg}")

    robot_interface = RobotInterface(service_stub, cfg)
    model_to_action_trans = ActionTranslator(cfg)
    dbg_printer = print_info.InfoPrinter(cfg)

    robot_interface.move_home()

    step = 0
    done = False

    # Inference Loop
    print("Starting inference loop...")
    period = 1.0 / hz

    while not done:
        start_time = time.time()
        observation = robot_interface.get_observation(cfg.device, show=False)

        if observation:
            # Predict the next action with respect to the current observation
            print(observation["observation.state"])
            with torch.inference_mode():
                action = policy.select_action(observation)

            action = model_to_action_trans.translate(action, observation)
            break

            # print(f"\n=== ABSOLUTE ROBOT COMMANDS ===")
            dbg_printer.print(step, observation, action, raw_action=False)

            robot_interface.send_action(
                action,
                model_to_action_trans.action_mode,
                controller,
            )
            # policy._queues["action"].clear()

        # wait for execution to finish
        elapsed_time = time.time() - start_time
        sleep_duration = period - elapsed_time
        # print(sleep_duration)
        # wait for input
        # input("Press Enter to continue...")
        time.sleep(max(0.0, sleep_duration))

        step += 1


def main():
    parser = argparse.ArgumentParser(description="Robot service client")
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
    args = parser.parse_args()

    # Select your device
    device = "cpu" if not torch.cuda.is_available() else "cuda"

    policy, cfg = load_policy(args.checkpoint)
    policy.to(device)
    deploy_policy(policy, cfg, 10, args.server)


def deploy_policy(policy, cfg, hz: float, server: str, controller=None):
    if controller is None:
        controller = RobotClient.CART_WAYPOINT
    channel = grpc.insecure_channel(server)
    stub = robot_service_pb2_grpc.RobotServiceStub(channel)
    try:
        inference_loop(policy, cfg, hz, stub, controller)
    except Exception as e:
        print(f"Error occurred: {e}")
        raise e
    finally:
        channel.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()
