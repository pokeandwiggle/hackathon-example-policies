#!/usr/bin/env python
"""Deploy a trained policy in a move-home → deploy loop.

Runs NUM_ROLLOUTS cycles of:
  1. Move home (grippers open, consistent start pose)
  2. Wait for Enter to begin
  3. Deploy policy until Ctrl-C
  4. (If --record) Rate the rollout, save, and upload to HuggingFace

Recording is optional — pass --record to capture rollouts into a
LeRobot v3.0 dataset and auto-upload to HuggingFace Hub.

Usage:
    # Deploy only (no recording):
    example_policies deploy-loop \
        --hf-repo-id pokeandwiggle/my_model \
        --robot-server 192.168.0.101:50051 \
        --mount pedestal \
        --num-rollouts 10 \
        --n-action-steps 48

    # Deploy with recording (auto-uploads to HF):
    example_policies deploy-loop \
        --hf-repo-id pokeandwiggle/my_model \
        --robot-server 192.168.0.101:50051 \
        --mount pedestal \
        --n-action-steps 48 \
        --num-rollouts 10 \
        --record
"""

import argparse
import datetime
import pathlib
import shutil

import torch

from example_policies.default_paths import MODELS_DIR, PLOTS_DIR, ROLLOUT_RECORDINGS_DIR
from example_policies.robot_deploy.deploy import _MOUNT_EMBODIMENT, move_home
from example_policies.robot_deploy.deploy_core.action_chunk_blender import (
    ActionChunkBlender,
)
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


def _model_name(args: argparse.Namespace) -> str:
    """Derive a short model name from checkpoint path or HF repo ID."""
    if args.hf_repo_id:
        # "pokeandwiggle/my_model" → "my_model"
        return args.hf_repo_id.split("/")[-1]
    # /data/models/my_policy → "my_policy"
    return args.checkpoint.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy policy in a move-home loop (optionally with recording)",
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
        help="Robot mount type (determines home pose and joint names)",
    )

    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        metavar="N",
        help="Override the number of action steps executed per chunk (default: from model config)",
    )
    parser.add_argument(
        "--controller",
        type=str,
        choices=["waypoint", "direct"],
        default="waypoint",
        help="Cartesian controller mode: waypoint (default) or direct",
    )

    # --- Loop ---
    parser.add_argument(
        "-n", "--num-rollouts",
        type=int,
        default=10,
        metavar="N",
        help="Number of rollouts to run (default: 10)",
    )

    # --- Recording (opt-in; implies push-to-hub) ---
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record rollouts and upload to HuggingFace Hub",
    )
    parser.add_argument(
        "-o", "--output",
        type=pathlib.Path,
        default=None,
        metavar="DIR",
        help="Output directory for recorded dataset (default: /data/rollout_recordings/<model_name>)",
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
    parser.add_argument(
        "--hub-org",
        type=str,
        default="pokeandwiggle",
        help="HuggingFace organization for the uploaded dataset (default: pokeandwiggle)",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Record locally but skip HuggingFace upload",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix appended to recording folder and HF repo name (e.g. '_blabbla')",
    )

    # --- Temporal ensemble (opt-in) ---
    parser.add_argument(
        "--temporal-ensemble",
        action="store_true",
        help="Enable temporal-ensemble blending to smooth action chunk boundaries",
    )
    parser.add_argument(
        "--decay-steps",
        type=int,
        default=8,
        metavar="N",
        help="Number of offset-decay blending steps when chunk_size == n_action_steps (default: 8)",
    )

    return parser


def _resolve_checkpoint(args: argparse.Namespace) -> pathlib.Path:
    """Resolve checkpoint path from local path or HuggingFace download."""
    if args.checkpoint:
        return args.checkpoint

    from huggingface_hub import snapshot_download

    download_dir = MODELS_DIR
    local_dir = download_dir / args.hf_repo_id.replace("/", "_")
    print(f"Downloading model '{args.hf_repo_id}' from HuggingFace Hub...")
    path = pathlib.Path(snapshot_download(repo_id=args.hf_repo_id, local_dir=local_dir))
    print(f"Downloaded to: {path}")
    return path


def _push_dataset(recorder, args: argparse.Namespace, model_name: str) -> None:
    """Push the recorded dataset to HuggingFace Hub (always private)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Auto-generate repo ID: org/eval_<model_name>_<success_rate>
    rate_pct = int(recorder.success_rate * 100)
    n_success = sum(1 for o in recorder.outcomes if o == "success")
    n_total = len(recorder.outcomes)
    repo_id = f"{args.hub_org}/eval_{model_name}{args.suffix}_{rate_pct}pct_{n_success}of{n_total}"

    print(f"\nPushing to HuggingFace Hub: {repo_id} ...")
    dataset = LeRobotDataset(repo_id=repo_id, root=args.output)
    dataset.push_to_hub(
        tags=["LeRobot"],
        private=True,
        upload_large_folder=True,
    )
    print(f"Uploaded: https://huggingface.co/datasets/{repo_id}")


def main():
    args = build_parser().parse_args()

    model_name = _model_name(args)

    # Auto-generate output path if not specified
    if args.record and args.output is None:
        args.output = ROLLOUT_RECORDINGS_DIR / f"{model_name}{args.suffix}"

    # --- Resolve checkpoint ---
    checkpoint_dir = _resolve_checkpoint(args)

    # --- Load policy ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_bundle = PolicyManager.load_single(checkpoint_dir, device)
    print(f"Policy loaded on {device}")

    # --- Override n_action_steps if requested ---
    if args.n_action_steps is not None:
        old = policy_bundle.config.n_action_steps
        policy_bundle.config.n_action_steps = args.n_action_steps
        print(f"Overriding n_action_steps: {old} → {args.n_action_steps}")

    # --- Derive embodiment from mount ---
    embodiment_name = _MOUNT_EMBODIMENT[args.mount]
    embodiment = get_joint_config(embodiment_name)

    # --- Create recorder (if recording) ---
    recorder = None
    if args.record:
        from example_policies.robot_deploy.deploy_core.rollout_recorder import RolloutRecorder

        if args.output.exists() and not args.keep_existing:
            shutil.rmtree(args.output)
            print(f"Deleted existing dataset at {args.output}")

        recorder = RolloutRecorder.from_policy_bundle(
            output_dir=args.output,
            policy_bundle=policy_bundle,
            fps=int(args.hertz),
            task_name=args.task_name,
        )

    # --- Print config summary ---
    print()
    print(f"Model:       {model_name} ({checkpoint_dir})")
    print(f"Robot:       {args.robot_server}")
    print(f"Mount:       {args.mount} ({embodiment_name})")
    print(f"Frequency:   {args.hertz} Hz")
    print(f"Rollouts:    {args.num_rollouts}")
    if recorder:
        print(f"Recording:   {args.output}")
        if not args.no_push:
            print(f"Upload:      enabled (to {args.hub_org})")
    else:
        print(f"Recording:   off")
    print()

    # --- Setup inference ---
    _CONTROLLER_MAP = {
        "waypoint": RobotClient.CART_WAYPOINT,
        "direct": RobotClient.CART_DIRECT,
    }
    config = InferenceConfig(
        hz=args.hertz,
        device=device,
        controller=_CONTROLLER_MAP[args.controller],
    )

    # --- Optional temporal-ensemble blender ---
    blender = None
    if args.temporal_ensemble:
        chunk_size = getattr(policy_bundle.config, "chunk_size", None) or getattr(
            policy_bundle.config, "horizon", None
        )
        n_action_steps = policy_bundle.config.n_action_steps
        if chunk_size is None:
            print("Warning: cannot determine chunk_size from config; disabling temporal ensemble.")
        else:
            blender = ActionChunkBlender(
                chunk_size=chunk_size,
                n_action_steps=n_action_steps,
                decay_steps=args.decay_steps,
            )
            overlap = blender.overlap
            mode = "temporal-ensemble" if overlap > 0 else "offset-decay"
            print(f"Temporal ensemble:  {mode} (chunk={chunk_size}, execute={n_action_steps}, overlap={overlap})")

    with RobotConnection(args.robot_server) as stub:
        policy_bundle.config.embodiment = embodiment
        robot_interface = RobotInterface(stub, policy_bundle.config, embodiment)
        runner = InferenceRunner(robot_interface, config, verbose=False, blender=blender)

        for rollout_idx in range(args.num_rollouts):
            # --- Move home ---
            print(f"\n{'=' * 60}")
            print(f"  Rollout {rollout_idx + 1}/{args.num_rollouts} — moving home")
            print(f"{'=' * 60}")
            move_home(robot_interface, mount=args.mount)

            input("Press Enter to start deployment (Ctrl-C to stop)...")

            # --- Deploy (+ optionally record) ---
            runner.reset()
            if recorder:
                recorder.start_episode()

            try:
                while True:
                    if recorder:
                        result = runner.run_step_recorded(policy_bundle)
                        recorder.record_step(result)
                    else:
                        runner.run_step(policy_bundle)
            except KeyboardInterrupt:
                print("\nRollout stopped.")
                runner.print_timing_summary()
                # Save timing plot
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = PLOTS_DIR / "deploy_timing" / f"deploy_timing_{model_name}_r{rollout_idx+1}_{ts}.png"
                runner.save_timing_plot(plot_path)
                print(f"Timing plot saved to: {plot_path}")

            # --- Rate the rollout (only when recording) ---
            if recorder:
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
    print(f"\nAll {args.num_rollouts} rollouts complete.")

    if recorder:
        recorder.close()
        n_success = sum(1 for o in recorder.outcomes if o == "success")
        n_total = len(recorder.outcomes)
        print(f"Success rate: {n_success}/{n_total} ({int(recorder.success_rate * 100)}%)")
        print(f"Dataset saved to: {args.output}")

        if not args.no_push:
            _push_dataset(recorder, args, model_name)


if __name__ == "__main__":
    main()
