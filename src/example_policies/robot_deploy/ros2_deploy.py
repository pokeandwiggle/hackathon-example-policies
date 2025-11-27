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
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch

from example_policies import data_constants as dc

from example_policies.robot_deploy.action_translator import ActionTranslator
from example_policies.robot_deploy.debug_helpers import sensor_stream as dbg_sensors
from example_policies.robot_deploy.policy_loader import load_policy
from example_policies.robot_deploy.robot_io.observation_builder import ObservationBuilder
from example_policies.robot_deploy.robot_io.ros2_client import ROS2RobotClient
from example_policies.robot_deploy.utils import print_info
from example_policies.robot_deploy.utils.action_mode import ActionMode


class ROS2RobotInterface:
    """Handles communication and data conversion with the robot via ROS2."""

    def __init__(self, ros2_client: ROS2RobotClient, cfg):
        self.client = ros2_client
        self.observation_builder = ObservationBuilder(cfg)
        self.robot_names = None
        self.last_command = None

    def get_observation(self, device, show=False):
        """Gets the current observation from the robot."""
        snapshot_response, self.robot_names = self.client.get_snapshot()

        if show:
            dbg_sensors.show_response(snapshot_response)
            cv2.waitKey(1)

        obs = self.observation_builder.get_observation(
            snapshot_response, self.last_command, device
        )
        return obs

    def send_action(
        self,
        action: torch.Tensor,
        action_mode: ActionMode,
        ctrl_mode: str = ROS2RobotClient.CART_WAYPOINT,
    ):
        """Sends a predicted action to the robot via ROS2."""
        numpy_action = action.squeeze(0).to("cpu").numpy()

        cart_targets = _build_cart_target_dict(numpy_action)

        for target in cart_targets:
            self.client._send_cartesian_command(target)

    def move_home(self):
        """Sends a command to move the robot to its home position."""
        self.client.send_move_home()


def _build_cart_target_dict(np_action: np.ndarray) -> list[dict]:
    """
    Creates cartesian target dictionaries from action array for ROS2 client.

    Args:
        np_action: Action array containing left and right arm poses and grippers

    Returns:
        List of dictionaries with target information for left and right robots
    """
    targets = []

    # Left robot
    left_pose = np_action[dc.LEFT_ARM]
    targets.append(
        {
            "robot_name": "left",
            "twist": {
                "linear": {
                    "x": float(left_pose[0]),
                    "y": float(left_pose[1]),
                    "z": float(left_pose[2]),
                },
                "angular": {
                    "x": float(left_pose[3]),
                    "y": float(left_pose[4]),
                    "z": float(left_pose[5]),
                },
            },
            "gripper": float(np_action[dc.LEFT_GRIPPER_IDX]),
        }
    )

    # Right robot
    right_pose = np_action[dc.RIGHT_ARM]
    targets.append(
        {
            "robot_name": "right",
            "twist": {
                "linear": {
                    "x": float(right_pose[0]),
                    "y": float(right_pose[1]),
                    "z": float(right_pose[2]),
                },
                "angular": {
                    "x": float(right_pose[3]),
                    "y": float(right_pose[4]),
                    "z": float(right_pose[5]),
                },
            },
            "gripper": float(np_action[dc.RIGHT_GRIPPER_IDX]),
        }
    )

    return targets

def inference_loop(
    policy,
    cfg,
    hz: float,
    ros2_client: ROS2RobotClient,
    controller=None,
):
    """
    Main inference loop that runs the policy and sends commands via ROS2.

    Args:
        policy: The loaded policy model
        cfg: Configuration object
        hz: Control frequency in Hz
        ros2_client: ROS2RobotClient instance
        controller: Controller mode (CART_WAYPOINT, CART_DIRECT, etc.)
    """
    if controller is None:
        controller = ROS2RobotClient.CART_WAYPOINT

    robot_interface = ROS2RobotInterface(ros2_client, cfg)
    model_to_action_trans = ActionTranslator(cfg)
    dbg_printer = print_info.InfoPrinter(cfg)

    step = 0
    done = False

    # Inference Loop
    print("Starting ROS2 inference loop...")
    period = 1.0 / hz

    # ROS2 spinning in background thread
    spin_thread = threading.Thread(target=lambda: rclpy.spin(ros2_client), daemon=True)
    spin_thread.start()

    # Wait for ROS2 topics to connect and start receiving data
    print("Waiting for ROS2 topics to become available...")
    time.sleep(2.0)
    print("Starting inference...")

    try:
        while not done:
            start_time = time.time()
            observation = robot_interface.get_observation(cfg.device, show=False)

            if observation:
                # Predict the next action with respect to the current observation
                with torch.inference_mode():
                    action = policy.select_action(observation)
                    print("\n=== RAW MODEL PREDICTION ===")
                    dbg_printer.print(step, observation, action, raw_action=True)
                    print()
                #action = model_to_action_trans.translate(action, observation)
                #action[:, dc.LEFT_GRIPPER_IDX] = 1.0 - action[:, dc.LEFT_GRIPPER_IDX]
                #action[:, dc.RIGHT_GRIPPER_IDX] = 1.0 - action[:, dc.RIGHT_GRIPPER_IDX]

                print("\n=== ABSOLUTE ROBOT COMMANDS ===")
                dbg_printer.print(step, observation, action, raw_action=False)

                robot_interface.send_action(
                   action, model_to_action_trans.action_mode, controller
                )

                if step >= 1:
                    done = True

            # Wait for execution to finish
            elapsed_time = time.time() - start_time
            sleep_duration = period - elapsed_time
            print(f"Sleep duration: {sleep_duration:.4f}s")
            time.sleep(max(0.0, sleep_duration))

            step += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("Stopping inference loop...")


def main():
    """Main entry point for ROS2-based robot deployment."""
    parser = argparse.ArgumentParser(description="ROS2 robot service client")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the policy checkpoint directory.",
    )
    parser.add_argument(
        "--node-name",
        default="robot_deploy_client",
        help="ROS2 node name (default: robot_deploy_client)",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=10.0,
        help="Control frequency in Hz (default: 10.0)",
    )
    args = parser.parse_args()

    # Select your device
    device = "cpu" if not torch.cuda.is_available() else "cuda"

    # Load policy
    policy, cfg = load_policy(args.checkpoint)
    policy.to(device)

    # Deploy via ROS2
    deploy_policy(policy, cfg, args.hz, args.node_name)


def deploy_policy(
    policy, cfg, hz: float, node_name: str = "robot_deploy_client", controller=None
):
    """
    Deploy policy using ROS2 communication.

    Args:
        policy: The loaded policy model
        cfg: Configuration object
        hz: Control frequency in Hz
        node_name: Name for the ROS2 node
        controller: Controller mode (optional)
    """
    if controller is None:
        controller = ROS2RobotClient.CART_WAYPOINT

    # Initialize ROS2
    if not rclpy.ok():
        rclpy.init()

    # Create ROS2 client
    ros2_client = ROS2RobotClient(node_name)

    try:
        inference_loop(policy, cfg, hz, ros2_client, controller)
    except Exception as e:
        print(f"Error occurred: {e}")
        raise e
    finally:
        ros2_client.shutdown()
        rclpy.shutdown()
        print("ROS2 connection closed.")


if __name__ == "__main__":
    main()
