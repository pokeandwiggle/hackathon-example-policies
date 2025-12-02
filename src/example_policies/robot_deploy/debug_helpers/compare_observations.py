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
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rich import print
from rich.table import Table

from example_policies.robot_deploy.policy_loader import load_policy
from example_policies.robot_deploy.robot_io.robot_interface import RobotInterface
from example_policies.robot_deploy.robot_io.robot_service import (
    robot_service_pb2_grpc,
)


def compare_observations(
    cfg,
    data_dir: Path,
    service_stub: robot_service_pb2_grpc.RobotServiceStub,
    ep_index: int = 0,
    frame_index: int = 0,
):
    """Compare dataset observation with current robot observation.

    Args:
        cfg: Policy configuration.
        data_dir (Path): Path to the dataset directory.
        service_stub: gRPC service stub.
        ep_index (int): Episode index to load.
        frame_index (int): Frame index within the episode to compare.
    """
    print(f"[bold cyan]Loading dataset from: {data_dir}[/bold cyan]")
    print(f"Episode: {ep_index}, Frame: {frame_index}\n")

    fake_repo_id = data_dir.name

    # Load dataset
    dataset = LeRobotDataset(
        repo_id=fake_repo_id,
        root=data_dir,
        episodes=[ep_index],
        video_backend="pyav",
    )

    # Get dataset observation
    print("[yellow]Loading observation from dataset...[/yellow]")
    dataset_obs = dataset[frame_index]

    # Initialize robot interface
    robot_interface = RobotInterface(service_stub, cfg)

    # Get current robot observation
    print("[yellow]Getting current observation from robot...[/yellow]")
    robot_obs = None
    while robot_obs is None:
        robot_obs = robot_interface.get_observation(cfg.device)
        time.sleep(0.1)

    print("[green]Observations loaded successfully![/green]\n")

    # Compare observations
    print("COMPARISON")

    # # Compare state observations
    # if "observation.state" in dataset_obs and "observation.state" in robot_obs:
    #     print("[bold cyan]State Comparison:[/bold cyan]")
    #     dataset_state = dataset_obs["observation.state"].cpu().numpy().squeeze()
    #     robot_state = robot_obs["observation.state"].cpu().numpy().squeeze()
    #
    #     state_names = cfg.input_features["observation.state"]
    #     print(state_names)
    #
    #     # Create comparison table
    #     table = Table(title="State Values Comparison")
    #     table.add_column("Feature", style="cyan")
    #     table.add_column("Dataset", style="green")
    #     table.add_column("Robot", style="yellow")
    #     table.add_column("Difference", style="red")
    #     table.add_column("Abs Diff", style="magenta")
    #
    #     for i, name in enumerate(state_names):
    #         dataset_val = dataset_state[i]
    #         robot_val = robot_state[i]
    #         diff = robot_val - dataset_val
    #         abs_diff = abs(diff)
    #
    #         table.add_row(
    #             name,
    #             f"{dataset_val:.6f}",
    #             f"{robot_val:.6f}",
    #             f"{diff:.6f}",
    #             f"{abs_diff:.6f}",
    #         )
    #
    #     print(table)
    #
    #     # Summary statistics
    #     print("\n[bold cyan]State Statistics:[/bold cyan]")
    #     print(
    #         f"  Mean Absolute Difference: {np.mean(np.abs(robot_state - dataset_state)):.6f}"
    #     )
    #     print(
    #         f"  Max Absolute Difference:  {np.max(np.abs(robot_state - dataset_state)):.6f}"
    #     )
    #     print(
    #         f"  RMS Difference:           {np.sqrt(np.mean((robot_state - dataset_state) ** 2)):.6f}"
    #     )
    #     print()

    # Compare image observations
    image_keys = [key for key in dataset_obs.keys() if "image" in key.lower()]

    if image_keys:
        print("[bold cyan]Image Observations Comparison:[/bold cyan]")
        for img_key in image_keys:
            print(f"\nComparing image key: [yellow]{img_key}[/yellow]")
            if img_key in dataset_obs and img_key in robot_obs:
                dataset_img = dataset_obs[img_key].cpu().numpy()
                robot_img = robot_obs[img_key].cpu().squeeze().numpy()

                print(f"\n  [yellow]{img_key}:[/yellow]")
                print(f"    Dataset shape: {dataset_img.shape}")
                print(f"    Robot shape:   {robot_img.shape}")

                # Print individual camera statistics
                print(f"\n    [cyan]Dataset Image Statistics:[/cyan]")
                print(f"      Min:    {np.min(dataset_img):.4f}")
                print(f"      Max:    {np.max(dataset_img):.4f}")
                print(f"      Mean:   {np.mean(dataset_img):.4f}")
                print(f"      Std:    {np.std(dataset_img):.4f}")
                print(f"      Median: {np.median(dataset_img):.4f}")

                print(f"\n    [cyan]Robot Image Statistics:[/cyan]")
                print(f"      Min:    {np.min(robot_img):.4f}")
                print(f"      Max:    {np.max(robot_img):.4f}")
                print(f"      Mean:   {np.mean(robot_img):.4f}")
                print(f"      Std:    {np.std(robot_img):.4f}")
                print(f"      Median: {np.median(robot_img):.4f}")

                if dataset_img.shape == robot_img.shape:
                    diff = robot_img - dataset_img
                    print(f"\n    [cyan]Comparison Metrics:[/cyan]")
                    print(f"      Mean pixel difference:     {np.mean(np.abs(diff)):.4f}")
                    print(f"      Max pixel difference:      {np.max(np.abs(diff)):.4f}")
                    print(
                        f"      RMS pixel difference:      {np.sqrt(np.mean(diff**2)):.4f}"
                    )
                    print(
                        f"      Correlation coefficient:   {np.corrcoef(dataset_img.flatten(), robot_img.flatten())[0, 1]:.4f}"
                    )
                else:
                    print("    [red]ERROR: Shape mismatch![/red]")

    # Compare all other keys
    print("\n[bold cyan]Other Observation Keys:[/bold cyan]")
    all_keys = set(dataset_obs.keys()) | set(robot_obs.keys())
    state_and_image_keys = {"observation.state"} | set(image_keys)
    other_keys = all_keys - state_and_image_keys

    if other_keys:
        table = Table(title="Other Features")
        table.add_column("Key", style="cyan")
        table.add_column("In Dataset", style="green")
        table.add_column("In Robot", style="yellow")
        table.add_column("Match", style="magenta")

        for key in sorted(other_keys):
            in_dataset = key in dataset_obs
            in_robot = key in robot_obs

            if in_dataset and in_robot:
                dataset_val = dataset_obs[key]
                robot_val = robot_obs[key]

                if isinstance(dataset_val, torch.Tensor) and isinstance(
                    robot_val, torch.Tensor
                ):
                    match = torch.allclose(dataset_val, robot_val, rtol=1e-5, atol=1e-5)
                    match_str = "✓" if match else "✗"
                else:
                    match_str = "?" if dataset_val == robot_val else "✗"
            else:
                match_str = "N/A"

            table.add_row(
                key,
                "✓" if in_dataset else "✗",
                "✓" if in_robot else "✗",
                match_str,
            )

        print(table)

    # Summary
    print("SUMMARY")
    print(f"Dataset keys:      {sorted(dataset_obs.keys())}")
    print(f"Robot keys:        {sorted(robot_obs.keys())}")
    print(
        f"Common keys:       {sorted(set(dataset_obs.keys()) & set(robot_obs.keys()))}"
    )
    print(
        f"Dataset-only keys: {sorted(set(dataset_obs.keys()) - set(robot_obs.keys()))}"
    )
    print(
        f"Robot-only keys:   {sorted(set(robot_obs.keys()) - set(dataset_obs.keys()))}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare dataset observation with current robot observation"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the policy checkpoint directory.",
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="Robot service server address (default: localhost:50051)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to load (default: 0)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index within the episode to compare (default: 0)",
    )

    args = parser.parse_args()

    # Load policy to get config
    policy, cfg = load_policy(args.checkpoint)

    channel = grpc.insecure_channel(args.server)
    stub = robot_service_pb2_grpc.RobotServiceStub(channel)

    try:
        compare_observations(
            cfg,
            args.data_dir,
            stub,
            args.episode,
            args.frame,
        )
    except Exception as e:
        print(f"[red]Error occurred: {e}[/red]")
        raise e
    finally:
        channel.close()
        print("\n[green]Connection closed.[/green]")


if __name__ == "__main__":
    main()
