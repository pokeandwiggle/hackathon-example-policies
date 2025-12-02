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
from PIL import Image
from rich import print

from example_policies.robot_deploy.policy_loader import load_policy
from example_policies.robot_deploy.robot_io.robot_interface import RobotInterface
from example_policies.robot_deploy.robot_io.robot_service import (
    robot_service_pb2_grpc,
)


def capture_and_save_images(
    cfg,
    service_stub: robot_service_pb2_grpc.RobotServiceStub,
    output_dir: Path,
):
    """Capture images from robot and save them as JPG files.

    Args:
        cfg: Policy configuration.
        service_stub: gRPC service stub.
        output_dir (Path): Directory to save the captured images.
    """
    print("[bold cyan]Capturing images from robot...[/bold cyan]\n")

    # Initialize robot interface
    robot_interface = RobotInterface(service_stub, cfg)

    # Get current robot observation
    print("[yellow]Getting current observation from robot...[/yellow]")
    robot_obs = None
    while robot_obs is None:
        robot_obs = robot_interface.get_observation(cfg.device)
        time.sleep(0.1)

    print("[green]Observation received![/green]\n")

    # Find all image keys in the observation
    image_keys = [key for key in robot_obs.keys() if "image" in key.lower()]

    if not image_keys:
        print("[red]No image observations found in robot observation![/red]")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bold cyan]Found {len(image_keys)} image(s). Saving to: {output_dir}[/bold cyan]\n")

    # Process and save each image
    for img_key in image_keys:
        print(f"Processing image: [yellow]{img_key}[/yellow]")

        # Get the image tensor
        img_tensor = robot_obs[img_key]

        # Convert to numpy array
        if isinstance(img_tensor, torch.Tensor):
            img_array = img_tensor.cpu().squeeze().numpy()
        else:
            img_array = np.array(img_tensor).squeeze()

        print(f"  Shape: {img_array.shape}")

        # Handle different image formats
        # Assuming format is (C, H, W) or (H, W, C)
        if img_array.ndim == 3:
            # If channels first (C, H, W), transpose to (H, W, C)
            if img_array.shape[0] in [1, 3, 4]:  # Common channel counts
                img_array = np.transpose(img_array, (1, 2, 0))

        # Normalize to 0-255 range if needed
        if img_array.dtype == np.float32 or img_array.dtype == np.float64:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        elif img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)

        # Remove single channel dimension if grayscale
        if img_array.shape[-1] == 1:
            img_array = img_array.squeeze(-1)

        # Create PIL Image and save
        img = Image.fromarray(img_array)

        # Generate filename from image key
        safe_key = img_key.replace("observation.", "").replace(".", "_")
        output_path = output_dir / f"{safe_key}.jpg"

        img.save(output_path, "JPEG", quality=95)
        print(f"  Saved to: [green]{output_path}[/green]")

    print(f"\n[bold green]Successfully saved {len(image_keys)} image(s)![/bold green]")


def main():
    parser = argparse.ArgumentParser(
        description="Capture images from robot and save as JPG files"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the policy checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./robot_images"),
        help="Directory to save captured images (default: ./robot_images)",
    )
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="Robot service server address (default: localhost:50051)",
    )

    args = parser.parse_args()

    # Load policy to get config
    print(f"[cyan]Loading policy from: {args.checkpoint}[/cyan]")
    policy, cfg = load_policy(args.checkpoint)

    channel = grpc.insecure_channel(args.server)
    stub = robot_service_pb2_grpc.RobotServiceStub(channel)

    try:
        capture_and_save_images(cfg, stub, args.output)
    except Exception as e:
        print(f"[red]Error occurred: {e}[/red]")
        raise e
    finally:
        channel.close()
        print("\n[green]Connection closed.[/green]")


if __name__ == "__main__":
    main()
