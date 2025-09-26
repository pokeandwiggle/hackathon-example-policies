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
import numpy as np
import torch

from example_policies.robot_deploy.robot_io.robot_interface import RobotInterface
from example_policies.robot_deploy.robot_io.robot_service import robot_service_pb2_grpc
from example_policies.robot_deploy.utils.action_mode import ActionMode


def replay_npy_actions(
    npy_file: Path,
    service_stub: robot_service_pb2_grpc.RobotServiceStub,
    replay_frequency: float = 10.0,
    ask_for_input: bool = False,
    action_mode: ActionMode = ActionMode.ABS_JOINT,
):
    """Replay actions from NPY file on the robot.

    Args:
        npy_file (Path): Path to the NPY file with shape (361, 1, 16).
        service_stub: gRPC service stub.
        replay_frequency (float): Frequency to replay the data.
        ask_for_input (bool): Whether to ask for user input at each action.
        action_mode (ActionMode): Type of action (ABS_JOINT, DELTA_JOINT, etc.)
    """
    # Load the NPY file
    print(f"Loading actions from: {npy_file}")
    actions = np.load(npy_file)
    print(f"Loaded actions with shape: {actions.shape}")

    if len(actions.shape) != 3 or actions.shape[1] != 1:
        print(f"Warning: Expected shape (N, 1, 16), got {actions.shape}")
        print("Reshaping to (N, 1, 16)...")
        if len(actions.shape) == 2:
            actions = actions[:, None, :]  # Add batch dimension

    # Initialize robot interface (pass None for cfg since we don't need observations)
    robot_interface = RobotInterface(service_stub, cfg=None)

    # Move to home position
    input("Press Enter to move robot to home...")
    robot_interface.move_home()

    input(f"Press Enter to start replaying {len(actions)} actions...")

    # Replay actions
    print(f"Starting action replay with mode: {action_mode}")
    period = 1.0 / replay_frequency

    for step, action in enumerate(actions):
        start_time = time.time()

        # Convert to torch tensor (action should be [1, 16])
        action_tensor = torch.from_numpy(action).float()

        if ask_for_input:
            input(f"Press Enter to send action {step + 1}/{len(actions)}...")

        print(
            f"Step {step + 1}/{len(actions)}: Sending action {action_tensor.squeeze().numpy()[:4]}... (showing first 4 values)"
        )

        try:
            # Send action using the existing RobotInterface
            robot_interface.send_action(action_tensor, action_mode)
        except Exception as e:
            print(f"Failed to send action at step {step + 1}: {e}")
            user_input = input("Continue? (y/n): ")
            if user_input.lower() != "y":
                break

        # Wait for next action
        elapsed_time = time.time() - start_time
        sleep_duration = period - elapsed_time

        if sleep_duration > 0:
            time.sleep(sleep_duration)
        else:
            print(
                f"Warning: Action took {elapsed_time:.3f}s (longer than period {period:.3f}s)"
            )

    print("Finished replaying all actions")


def main():
    parser = argparse.ArgumentParser(
        description="Simple robot action replay from NPY file"
    )
    parser.add_argument(
        "npy_file",
        type=Path,
        help="Path to the NPY file containing actions",
    )
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="Robot service server address (default: localhost:50051)",
    )
    parser.add_argument(
        "--replay-frequency",
        type=float,
        default=10.0,
        help="Frequency to replay the data (default: 10.0 Hz)",
    )
    parser.add_argument(
        "--ask-for-input",
        action="store_true",
        help="Ask for user input at each action (default: False)",
    )
    parser.add_argument(
        "--action-mode",
        choices=["abs_joint", "delta_joint", "abs_tcp", "delta_tcp"],
        default="abs_joint",
        help="Action mode (default: abs_joint)",
    )

    args = parser.parse_args()

    # Convert string to ActionMode enum
    action_mode_map = {
        "abs_joint": ActionMode.ABS_JOINT,
        "delta_joint": ActionMode.DELTA_JOINT,
        "abs_tcp": ActionMode.ABS_TCP,
        "delta_tcp": ActionMode.DELTA_TCP,
    }
    action_mode = action_mode_map[args.action_mode]

    if not args.npy_file.exists():
        print(f"Error: NPY file not found: {args.npy_file}")
        return

    # Connect to robot service
    print(f"Connecting to robot service at {args.server}")
    channel = grpc.insecure_channel(args.server)
    stub = robot_service_pb2_grpc.RobotServiceStub(channel)

    try:
        replay_npy_actions(
            args.npy_file,
            stub,
            replay_frequency=args.replay_frequency,
            ask_for_input=args.ask_for_input,
            action_mode=action_mode,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
        raise e
    finally:
        channel.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()
