#!/usr/bin/env python
"""Deploy a trained policy in a move-home → record loop.

Runs NUM_ROLLOUTS cycles of:
  1. Move home (grippers open, consistent start pose — not recorded)
  2. Wait for Enter to begin
  3. Deploy policy until Ctrl-C (recorded as one episode)
  4. Rate the rollout: press s (success) or f (failure)
  5. Episode is saved

After all rollouts the dataset is finalized and optionally pushed to
HuggingFace Hub.

Usage:
    example_policies deploy-record \\
        --checkpoint /data/models/my_policy \\
        --robot-server 10.0.0.1:50051 \\
        --mount wall \\
        --num-rollouts 10 \\
        --output /data/rollout_recordings/run_1

    # Or with a HuggingFace model:
    example_policies deploy-record \\
        --hf-repo-id pokeandwiggle/my_model \\
        --robot-server 10.0.0.1:50051 \\
        --mount pedestal \\
        --num-rollouts 5 \\
        --push-to-hub
"""

import argparse
import pathlib
import shutil

import torch

from example_policies.robot_deploy.deploy import move_home
from example_policies.robot_deploy.deploy_core.deployment_structures import (
    InferenceConfig,
)
from example_policies.robot_deploy.deploy_core.inference_runner import InferenceRunner
from example_policies.robot_deploy.deploy_core.policy_manager import PolicyManager
from example_policies.robot_deploy.deploy_core.rollout_recorder import RolloutRecorder
from example_policies.robot_deploy.robot_io.connection import RobotConnection
from example_policies.robot_deploy.robot_io.robot_interface import (
    RobotClient,
    RobotInterface,
)
from example_policies.utils.embodiment import get_joint_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy policy with rollout recording",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Model source (mutually exclusive) ---
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "-c", "--checkpoint",
        type=pathlib.Path,
        metavar="PATH",
        help="Path to a local policy checkpoint directory",
    )
    model_group.add_argument(
        "--hf-repo-id",
        type=str,
        metavar="REPO",
        help="HuggingFace model repo ID (e.g. pokeandwiggle/my_model)",
    )

    # --- Robot ---
    parser.add_argument(
        "-s", "--robot-server",
        default="localhost:50051",
        metavar="ADDR",
        help="Robot service server address (default: localhost:50051)",
    )
    parser.add_argument(
        "-z", "--hertz",
        type=float,
        default=30.0,
        metavar="HZ",
        help="Inference frequency in Hz (default: 30.0)",
    )
    parser.add_argument(
        "--mount",
        type=str,
        choices=["table", "wall", "pedestal"],
        required=True,
        help="Robot mount type for home pose",
    )
    parser.add_argument(
        "--embodiment",
        type=str,
        default=None,
        help="Embodiment name override (e.g. dual_fr3_pedestal)",
    )

    # --- Recording ---
    parser.add_argument(
        "-o", "--output",
        type=pathlib.Path,
        default=pathlib.Path("/data/rollout_recordings/recording_1"),
        metavar="DIR",
        help="Output directory for the recorded dataset (default: /data/rollout_recordings/recording_1)",
    )
    parser.add_argument(
        "-n", "--num-rollouts",
        type=int,
        default=10,
        metavar="N",
        help="Number of rollouts to record (default: 10)",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="policy_rollout",
        help="Task name stored in the dataset (default: policy_rollout)",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep existing output directory (default: delete and recreate)",
    )

    # --- HuggingFace upload ---
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the recorded dataset to HuggingFace Hub after all rollouts",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=None,
        metavar="REPO",
        help="HuggingFace dataset repo ID for upload (auto-generated if omitted)",
    )
    parser.add_argument(
        "--hub-org",
        type=str,
        default="pokeandwiggle",
        help="HuggingFace organization for auto-generated repo ID (default: pokeandwiggle)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the uploaded dataset public (default: private)",
    )

    # --- HuggingFace model download ---
    parser.add_argument(
        "--hf-download-dir",
        type=pathlib.Path,
        default=pathlib.Path("/data/models"),
        metavar="DIR",
        help="Local directory for downloaded HuggingFace models (default: /data/models)",
    )

    return parser


def _resolve_checkpoint(args: argparse.Namespace) -> pathlib.Path:
    """Resolve checkpoint path from local path or HuggingFace download."""
    if args.checkpoint:
        return args.checkpoint

    from huggingface_hub import snapshot_download

    local_dir = args.hf_download_dir / args.hf_repo_id.replace("/", "_")
    print(f"Downloading model '{args.hf_repo_id}' from HuggingFace Hub...")
    path = pathlib.Path(snapshot_download(repo_id=args.hf_repo_id, local_dir=local_dir))
    print(f"Downloaded to: {path}")
    return path


def _push_dataset(recorder: RolloutRecorder, args: argparse.Namespace) -> None:
    """Push the recorded dataset to HuggingFace Hub."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Build repo ID
    repo_id = args.hub_repo_id
    if repo_id is None:
        # Auto-generate: org/output-dir-name_<success_rate>
        base_name = args.output.name
        rate_pct = int(recorder.success_rate * 100)
        n_success = sum(1 for o in recorder.outcomes if o == "success")
        n_total = len(recorder.outcomes)
        repo_id = f"{args.hub_org}/{base_name}_{rate_pct}pct_{n_success}of{n_total}"

    private = not args.public

    print(f"\nPushing to HuggingFace Hub: {repo_id} ...")
    dataset = LeRobotDataset(repo_id=repo_id, root=args.output)
    dataset.push_to_hub(
        tags=["LeRobot"],
        private=private,
        upload_large_folder=True,
    )
    print(f"Uploaded: https://huggingface.co/datasets/{repo_id}")


def main():
    args = build_parser().parse_args()

    # --- Resolve checkpoint ---
    checkpoint_dir = _resolve_checkpoint(args)

    # --- Load policy ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_bundle = PolicyManager.load_single(checkpoint_dir, device)
    print(f"Policy loaded on {device}")

    # --- Prepare output directory ---
    if args.output.exists() and not args.keep_existing:
        shutil.rmtree(args.output)
        print(f"Deleted existing dataset at {args.output}")

    # --- Create recorder ---
    recorder = RolloutRecorder.from_policy_bundle(
        output_dir=args.output,
        policy_bundle=policy_bundle,
        fps=int(args.hertz),
        task_name=args.task_name,
    )

    # --- Print config summary ---
    print()
    print(f"Checkpoint:  {checkpoint_dir}")
    print(f"Robot:       {args.robot_server}")
    print(f"Mount:       {args.mount}")
    print(f"Frequency:   {args.hertz} Hz")
    print(f"Output:      {args.output}")
    print(f"Rollouts:    {args.num_rollouts}")
    if args.push_to_hub:
        print(f"Push to Hub: yes ({args.hub_org})")
    print()

    # --- Setup inference ---
    config = InferenceConfig(
        hz=args.hertz,
        device=device,
        controller=RobotClient.CART_WAYPOINT,
    )

    with RobotConnection(args.robot_server) as stub:
        embodiment = get_joint_config(args.embodiment) if args.embodiment else None
        if embodiment is not None:
            policy_bundle.config.embodiment = embodiment
        robot_interface = RobotInterface(stub, policy_bundle.config, embodiment)
        runner = InferenceRunner(robot_interface, config, verbose=False)

        for rollout_idx in range(args.num_rollouts):
            # --- Move home (not recorded) ---
            print(f"\n{'=' * 60}")
            print(f"  Rollout {rollout_idx + 1}/{args.num_rollouts} — moving home")
            print(f"{'=' * 60}")
            move_home(robot_interface, mount=args.mount)

            input("Press Enter to start deployment (Ctrl-C to stop)...")

            # --- Deploy + record ---
            runner.step = 0
            recorder.start_episode()

            try:
                while True:
                    result = runner.run_step_recorded(policy_bundle)
                    recorder.record_step(result)
            except KeyboardInterrupt:
                print("\nRollout stopped.")

            # --- Rate the rollout ---
            while True:
                rating = input("Rate this rollout — (s)uccess or (f)ailure: ").strip().lower()
                if rating in ("s", "f"):
                    break
                print("  Please enter 's' or 'f'.")

            outcome = "success" if rating == "s" else "failure"
            recorder.end_episode(outcome=outcome)

        # --- Final move home ---
        print(f"\n{'=' * 60}")
        print("  Moving home (final)")
        print(f"{'=' * 60}")
        move_home(robot_interface, mount=args.mount)

    # --- Finalize ---
    recorder.close()

    n_success = sum(1 for o in recorder.outcomes if o == "success")
    n_total = len(recorder.outcomes)
    print(f"\nAll {args.num_rollouts} rollouts complete.")
    print(f"Success rate: {n_success}/{n_total} ({int(recorder.success_rate * 100)}%)")
    print(f"Dataset saved to: {args.output}")

    # --- Upload ---
    if args.push_to_hub:
        _push_dataset(recorder, args)


if __name__ == "__main__":
    main()
