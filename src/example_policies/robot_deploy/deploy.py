#!/usr/bin/env python

from pathlib import Path
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
from example_policies.utils.state_order import CANONICAL_ARM_JOINTS

# Home pose configs for different robot mounts
HOME_CONFIGS = {
    "table": Path(__file__).parent / "panda_tum_config" / "dual_panda_table.yaml",
    "wall": Path(__file__).parent / "panda_tum_config" / "dual_panda_wall.yaml",
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
    import yaml

    config_path = HOME_CONFIGS.get(mount)
    if config_path is None:
        raise ValueError(f"Unknown mount type: {mount}. Valid options: {list(HOME_CONFIGS.keys())}")

    if not config_path.exists():
        raise FileNotFoundError(f"Home config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate presence of home_joint_configuration in the YAML
    if "home_joint_configuration" not in config:
        raise KeyError(
            f"Missing 'home_joint_configuration' in home config YAML: {config_path}"
        )
    home_config = config["home_joint_configuration"]

    # Validate that all expected joint names are present
    missing_joints = [joint_name for joint_name in CANONICAL_ARM_JOINTS if joint_name not in home_config]
    if missing_joints:
        raise KeyError(
            f"Missing joint configuration(s) {missing_joints} in 'home_joint_configuration' "
            f"for config file: {config_path}"
        )

    # Extract joint angles in canonical order (left arm first, then right arm)
    joint_angles = np.array([home_config[joint_name] for joint_name in CANONICAL_ARM_JOINTS])

    print(f"Moving to home pose ({mount} mount)...")
    robot_interface.move_to_joint_goal(joint_angles, CANONICAL_ARM_JOINTS)

    print("Opening grippers...")
    robot_interface.set_gripper_state("left", GRIPPER_OPEN_WIDTH, GRIPPER_SPEED, GRIPPER_FORCE)
    robot_interface.set_gripper_state("right", GRIPPER_OPEN_WIDTH, GRIPPER_SPEED, GRIPPER_FORCE)

    print("Robot at home position with grippers open.")


def main():
    parser = DeployArgumentParser.create_single_policy_parser()
    args = parser.parse_args()

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
        robot_interface = RobotInterface(stub, policy_bundle.config)

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
