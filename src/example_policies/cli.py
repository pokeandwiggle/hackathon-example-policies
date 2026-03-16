#!/usr/bin/env python
"""Unified CLI for example_policies with subcommands."""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: example_policies <command> [args...]")
        print()
        print("Commands:")
        print("  convert            Convert MCAP episodes to LeRobot dataset")
        print("  deploy             Run policy inference on robot")
        print("  deploy-loop        Deploy policy in a move-home loop (with optional recording)")
        print("  pre-deploy-check   Compare dataset first-frame vs live camera")
        print("  train              Train a policy")
        print("  validate           Validate trained policy with plots")
        print("  review             Review dataset")
        print("  sensor-stream      Stream sensor data for debugging")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = sys.argv[1:]  # Shift argv so subcommand sees correct args

    if command == "convert":
        from example_policies.data_ops.dataset_conversion_synced import (
            main as convert_main,
        )

        convert_main()
    elif command == "deploy":
        from example_policies.robot_deploy.deploy import main as deploy_main

        deploy_main()
    elif command == "deploy-loop":
        from example_policies.robot_deploy.deploy_loop import (
            main as deploy_loop_main,
        )

        deploy_loop_main()
    elif command == "pre-deploy-check":
        from example_policies.robot_deploy.pre_deploy_check import (
            main as pre_deploy_check_main,
        )

        pre_deploy_check_main()
    elif command == "train":
        from example_policies.train import main as train_main

        train_main()
    elif command == "validate":
        from example_policies.validate_with_plot import main as validate_main

        validate_main()
    elif command == "review":
        from example_policies.data_ops.review.review_dataset import (
            main as review_main,
        )

        review_main()
    elif command == "sensor-stream":
        from example_policies.robot_deploy.debug_helpers.sensor_stream import (
            main as sensor_main,
        )

        sensor_main()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
