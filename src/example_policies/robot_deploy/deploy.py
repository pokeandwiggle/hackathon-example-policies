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

from example_policies.robot_deploy.action_translator import ActionTranslator
from example_policies.robot_deploy.debug_helpers.utils import print_info
from example_policies.robot_deploy.policy_loader import load_policy
from example_policies.robot_deploy.robot_io.robot_interface import RobotInterface
from example_policies.robot_deploy.robot_io.robot_service import (
    robot_service_pb2,
    robot_service_pb2_grpc,
)
from example_policies.robot_deploy.action_translator import ActionMode


def inference_loop(
    policy, cfg, hz: float, service_stub: robot_service_pb2_grpc.RobotServiceStub
):

    robot_interface = RobotInterface(service_stub, cfg)
    model_to_action_trans = ActionTranslator(cfg)

    step = 0
    done = False

    # Inference Loop
    print("Starting inference loop...")
    period = 1.0 / hz

    # Initial preparation for queued execution. 
    prepare_request = robot_service_pb2.PrepareExecutionRequest()
    # If check which action mode is used
    action_mode = model_to_action_trans.action_mode
    if action_mode in (ActionMode.DELTA_TCP, ActionMode.ABS_TCP):
        prepare_request.execution_mode = (robot_service_pb2.ExecutionMode.EXECUTION_MODE_CARTESIAN_TARGET_QUEUE)
    # If joint direct targets are used
    elif action_mode in (ActionMode.DELTA_JOINT, ActionMode.ABS_JOINT):
        prepare_request.execution_mode = (robot_service_pb2.ExecutionMode.EXECUTION_MODE_JOINT_TARGET)
    else:
        raise ValueError(f"Unknown model to action mode: {action_mode}")
    service_stub.PrepareExecution(prepare_request)


    while not done:
        start_time = time.time()
        print(policy.config.input_features)
        observation = robot_interface.get_observation(cfg.device, show=False)

        if observation:
            # Predict the next action with respect to the current observation
            with torch.inference_mode():
                action = policy.select_action(observation)
                print(f"\n=== RAW MODEL PREDICTION ===")
                print_info(step, observation, action)
                print()
            action = model_to_action_trans.translate(action, observation)

            print(f"\n=== ABSOLUTE ROBOT COMMANDS ===")
            print_info(step, observation, action)

            robot_interface.send_action(action, action_mode)
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

    deploy_policy(policy, cfg, hz=1.5, server=args.server)


def deploy_policy(policy, cfg, hz: float, server: str):
    channel = grpc.insecure_channel(server)
    stub = robot_service_pb2_grpc.RobotServiceStub(channel)
    try:
        inference_loop(policy, cfg, hz, stub)
    except Exception as e:
        print(f"Error occurred: {e}")
        raise e
    finally:
        channel.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()
