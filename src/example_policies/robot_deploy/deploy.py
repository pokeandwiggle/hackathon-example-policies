#!/usr/bin/env python

import torch

from example_policies.robot_deploy.deploy_argument_parser import DeployArgumentParser
from example_policies.robot_deploy.deploy_core.deployment_structures import (
    InferenceConfig,
)
from example_policies.robot_deploy.deploy_core.inference_runner import InferenceRunner
from example_policies.robot_deploy.deploy_core.policy_manager import PolicyManager
from example_policies.robot_deploy.robot_io.connection import RobotConnection
from example_policies.robot_deploy.robot_io.robot_interface import (
    RobotClient,
    RobotInterface,
)
from example_policies.utils.embodiment import get_joint_config

# Home joint configurations for different robot mounts
# Order: left_joint1..7, right_joint1..7 (canonical arm joints)
HOME_JOINT_ANGLES = {
    "wall": [
        # left arm
        -0.0,
        -1.6,
        0.4,
        -2.38,
        1.7,
        2.2,
        -1.5,
        # right arm
        0.0,
        -1.6,
        -0.4,
        -2.38,
        -1.7,
        2.2,
        -0.0,
    ],
    "table": [
        # left arm
        0.0063710,
        -0.3389045,
        0.4691349,
        -2.5093593,
        -0.2406735,
        2.3181225,
        -1.1294540,
        # right arm
        -0.0140383,
        -0.3523653,
        -0.4208789,
        -2.5604102,
        0.3912430,
        2.3928931,
        -0.7659729,
    ],
}

# Gripper open width in meters
GRIPPER_OPEN_WIDTH = 0.08
GRIPPER_SPEED = 0.1  # m/s
GRIPPER_FORCE = 50.0  # N


def move_home(robot_interface: RobotInterface, mount: str = "wall") -> None:
    """Move the robot to home pose and open grippers.

    Args:
        robot_interface: The robot interface to use for commands
        mount: Robot mount type ("table" or "wall")
    """
    import numpy as np

    angles = HOME_JOINT_ANGLES.get(mount)
    if angles is None:
        raise ValueError(
            f"Unknown mount type: {mount}. Valid options: {list(HOME_JOINT_ANGLES.keys())}"
        )

    joint_angles = np.array(angles)

    print(f"Moving to home pose ({mount} mount)...")
    robot_interface.move_to_joint_goal(joint_angles)

    print("Opening grippers...")
    robot_interface.set_gripper_state(
        "left", GRIPPER_OPEN_WIDTH, GRIPPER_SPEED, GRIPPER_FORCE
    )
    robot_interface.set_gripper_state(
        "right", GRIPPER_OPEN_WIDTH, GRIPPER_SPEED, GRIPPER_FORCE
    )

    print("Robot at home position with grippers open.")


def main():
    parser = DeployArgumentParser.create_single_policy_parser()
    args = parser.parse_args()

    if args.move_home and args.mount is None:
        parser.error("--mount is required when --move-home is set")

    # Load policy
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    policy_bundle = PolicyManager.load_single(args.checkpoint, device)

    # Setup inference configuration
    config = InferenceConfig(
        hz=args.hertz,
        device=device,
        controller=RobotClient.CART_WAYPOINT,
    )

    # Run inference loop
    print("Starting single-policy inference loop...")
    with RobotConnection(args.robot_server) as stub:
        embodiment = get_joint_config(args.embodiment) if args.embodiment else None
        if embodiment is not None:
            policy_bundle.config.embodiment = embodiment
        robot_interface = RobotInterface(stub, policy_bundle.config, embodiment)

        # Move to home position if requested
        if args.move_home:
            move_home(robot_interface, args.mount)
            print("Press Enter to start inference...")
            input()

        runner = InferenceRunner(robot_interface, config)

        while True:
            runner.run_step(policy_bundle)


if __name__ == "__main__":
    main()
