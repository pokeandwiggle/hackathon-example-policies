#!/usr/bin/env python

import torch

from example_policies.robot_deploy.deploy_argument_parser import DeployArgumentParser
from example_policies.robot_deploy.deploy import move_home
from example_policies.utils.embodiment import get_joint_config
from example_policies.robot_deploy.deploy_core.deployment_structures import (
    InferenceConfig,
)
from example_policies.robot_deploy.deploy_core.inference_runner import InferenceRunner
from example_policies.robot_deploy.deploy_core.policy_manager import PolicyManager
from example_policies.robot_deploy.deploy_core.policy_selector import PolicySelector
from example_policies.robot_deploy.deploy_core.policy_switcher import PolicySwitcher
from example_policies.robot_deploy.robot_io.connection import RobotConnection
from example_policies.robot_deploy.robot_io.robot_interface import (
    RobotClient,
    RobotInterface,
)


def main():
    parser = DeployArgumentParser.create_multi_policy_parser()
    args = parser.parse_args()

    if args.move_home and args.mount is None:
        parser.error("--mount is required when --move-home is set")

    # Load policies
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    policies = PolicyManager.load_multiple(args.checkpoints, device)

    # Setup inference configuration
    config = InferenceConfig(
        hz=args.hertz,
        device=device,
        controller=RobotClient.CART_WAYPOINT,
    )

    # Run multi-policy inference loop
    print(f"Starting multi-policy inference loop with {len(policies)} policies")
    if args.prompt_interval > 0:
        print(f"Prompting for policy selection every {args.prompt_interval} steps")
    else:
        print("Periodic prompts disabled (only termination signals will trigger)")

    with RobotConnection(args.robot_server) as stub:
        embodiment = get_joint_config(args.embodiment) if args.embodiment else None
        if embodiment is not None:
            for pb in policies:
                pb.config.embodiment = embodiment
        robot_interface = RobotInterface(stub, policies[0].config, embodiment)

        # Move to home position if requested
        if args.move_home:
            move_home(robot_interface, args.mount)
            print("Press Enter to start inference...")
            input()

        runner = InferenceRunner(robot_interface, config)
        switcher = PolicySwitcher(policies)

        while True:
            # Check for scheduled prompt
            if switcher.should_prompt(args.prompt_interval):
                selected_idx = PolicySelector.prompt_for_selection(
                    switcher.policies,
                    switcher.current_idx,
                    switcher.global_step,
                    switcher.policy_step,
                    f"Interval reached ({args.prompt_interval} steps)",
                )
                if switcher.switch_to(selected_idx):
                    runner.reset()
                    print(f"\n{'=' * 60}")
                    print(
                        f"SWITCHED TO POLICY {switcher.current_idx} ({switcher.current_policy.name})"
                    )
                    print(f"{'=' * 60}\n")

            # Run one step
            policy_bundle = switcher.current_policy
            print(
                f"[Policy {switcher.current_idx}: {policy_bundle.name}] "
                f"Global Step {switcher.global_step}, Policy Step {switcher.policy_step}"
            )

            termination_signal = runner.run_step(policy_bundle)

            # Check termination signal
            if termination_signal is not None and termination_signal > 0.5:
                print(f"\n{'!' * 60}")
                print(f"TERMINATION SIGNAL DETECTED (value: {termination_signal:.4f})")
                print(f"{'!' * 60}")
                selected_idx = PolicySelector.prompt_for_selection(
                    switcher.policies,
                    switcher.current_idx,
                    switcher.global_step,
                    switcher.policy_step,
                    "Termination signal triggered",
                )
                if switcher.switch_to(selected_idx):
                    runner.reset()
                    print(f"\n{'=' * 60}")
                    print(
                        f"SWITCHED TO POLICY {switcher.current_idx} ({switcher.current_policy.name})"
                    )
                    print(f"{'=' * 60}\n")

            switcher.increment_steps()


if __name__ == "__main__":
    main()
