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
        runner = InferenceRunner(robot_interface, config)

        while True:
            runner.run_step(policy_bundle)


if __name__ == "__main__":
    main()
