#!/usr/bin/env python
"""Pre-deployment camera check: compare dataset first-frame vs live camera.

Connects to the robot, grabs the current head-cam image, and overlays it
with the first frame of either a single episode or the mean of all episode
first-frames from the training dataset.  Three visualisations are produced:

  1. **Side-by-side** – dataset frame (left) vs live frame (right)
  2. **Blended overlay** – 50/50 alpha blend so you can judge alignment
  3. **Pixel-wise std-diff** – bright regions = large difference

The plot is displayed (if a display is available) and saved to
``outputs/pre_deploy_check_<timestamp>.png``.

Usage:
    # Local dataset:
    example_policies pre-deploy-check \\
        --dataset /data/lerobot/stack_one_brick \\
        --robot-server 192.168.0.101:50051

    # Dataset from HuggingFace Hub:
    example_policies pre-deploy-check \\
        --hf-repo-id pokeandwiggle/stack_one_brick \\
        --robot-server 192.168.0.101:50051

    # Check a specific episode's first frame instead of the mean:
    example_policies pre-deploy-check \\
        --dataset /data/lerobot/stack_one_brick \\
        --robot-server 192.168.0.101:50051 \\
        --episode 3

    # Override which camera to compare (default: rgb_static):
    example_policies pre-deploy-check \\
        --dataset /data/lerobot/stack_one_brick \\
        --robot-server 192.168.0.101:50051 \\
        --camera rgb_left
"""

from __future__ import annotations

import argparse
import os
import pathlib
from datetime import datetime

import cv2
import grpc
import matplotlib

# Use non-interactive backend when no display is available (SSH, CI, …)
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from example_policies.data_ops.utils import image_processor
from example_policies.default_paths import DATASETS_DIR
from example_policies.robot_deploy.robot_io.robot_service import (
    robot_service_pb2,
    robot_service_pb2_grpc,
)


# ── helpers ──────────────────────────────────────────────────────────

def _grab_live_image(
    stub: robot_service_pb2_grpc.RobotServiceStub,
    camera_key: str,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Fetch one frame from the robot for *camera_key*.

    Returns an RGB float32 array normalised to [0, 1], shape (H, W, 3).
    """
    # camera_key is e.g. "rgb_static" – the gRPC snapshot exposes cameras
    # keyed by ROS frame name:  cam_{side}_color_optical_frame
    # where side ∈ {left, right, static}.
    side = camera_key.replace("rgb_", "").replace("depth_", "")
    is_depth = camera_key.startswith("depth_")
    modality = "depth" if is_depth else "color"
    ros_cam_name = f"cam_{side}_{modality}_optical_frame"

    snapshot_req = robot_service_pb2.GetStateRequest()
    snapshot_resp = stub.GetState(snapshot_req)
    state = snapshot_resp.current_state

    if ros_cam_name not in state.cameras:
        available = list(state.cameras.keys())
        raise RuntimeError(
            f"Camera '{ros_cam_name}' not found on robot. Available: {available}"
        )

    cam_data = state.cameras[ros_cam_name]
    img = image_processor.process_image_bytes(
        cam_data.data,
        width=target_w,
        height=target_h,
        is_depth=is_depth,
    )
    return img  # float32, [0, 1], (H, W, 3)


def _resolve_dataset_dir(
    dataset_dir: pathlib.Path | None,
    hf_repo_id: str | None,
) -> pathlib.Path:
    """Return a local dataset directory, downloading from HF if needed."""
    if dataset_dir is not None:
        return dataset_dir

    if hf_repo_id is None:
        raise ValueError("Provide either --dataset or --hf-repo-id.")

    from huggingface_hub import snapshot_download

    # Download into the standard datasets directory:
    #   /data/lerobot/<org>_<dataset_name>
    local_dir = DATASETS_DIR / hf_repo_id.replace("/", "_")
    print(f"Downloading dataset '{hf_repo_id}' from HuggingFace Hub …")
    path = pathlib.Path(
        snapshot_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            local_dir=local_dir,
        )
    )
    print(f"  Downloaded to: {path}")
    return path


def _load_dataset_first_frame(
    dataset_dir: pathlib.Path,
    camera_key: str,
    episode: int | None,
) -> np.ndarray:
    """Load the first frame from the dataset.

    If *episode* is ``None`` the mean of the first frames across **all**
    episodes is returned (same idea as notebook 07).  Otherwise only the
    specified episode's first frame is returned.

    Returns an RGB uint8 array, shape (H, W, 3).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    obs_key = f"observation.images.{camera_key}"

    dataset = LeRobotDataset(
        repo_id=dataset_dir.name,
        root=dataset_dir,
        video_backend="pyav",
    )

    # Validate that the camera key exists
    if obs_key not in dataset.meta.features:
        available = [k for k in dataset.meta.features if k.startswith("observation.images.")]
        raise RuntimeError(
            f"Camera key '{obs_key}' not found in dataset. Available: {available}"
        )

    total_episodes = dataset.meta.total_episodes

    if episode is not None:
        if episode < 0 or episode >= total_episodes:
            raise ValueError(
                f"Episode {episode} out of range (dataset has {total_episodes} episodes)"
            )
        episodes_to_load = [episode]
    else:
        episodes_to_load = list(range(total_episodes))

    frames: list[np.ndarray] = []
    for ep_idx in episodes_to_load:
        ep_meta = dataset.meta.episodes[ep_idx]
        global_start = ep_meta["dataset_from_index"]
        sample = dataset[global_start]
        img_tensor = sample[obs_key]
        if isinstance(img_tensor, torch.Tensor):
            img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            img = np.array(img_tensor)
        frames.append(img)

    if len(frames) == 1:
        return frames[0]

    # Mean across all first frames
    stacked = np.stack(frames, axis=0).astype(np.float64)
    mean_img = np.mean(stacked, axis=0).astype(np.uint8)
    return mean_img


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 [0, 255]."""
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.0:
            return (img * 255).astype(np.uint8)
    return img.astype(np.uint8)


# ── main logic ───────────────────────────────────────────────────────

def run_check(
    dataset_dir: pathlib.Path,
    server_address: str,
    camera_key: str = "rgb_static",
    episode: int | None = None,
    output_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    """Run the pre-deploy camera comparison.

    Returns the path to the saved comparison image.
    """
    if output_dir is None:
        output_dir = pathlib.Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load dataset first frame(s) ────────────────────────────
    ep_label = f"episode {episode}" if episode is not None else "mean of all episodes"
    print(f"Loading dataset first frame ({ep_label}) …")
    dataset_img = _load_dataset_first_frame(dataset_dir, camera_key, episode)
    h, w = dataset_img.shape[:2]
    print(f"  Dataset image shape: {dataset_img.shape}")

    # ── 2. Grab live image from robot ─────────────────────────────
    print(f"Connecting to robot at {server_address} …")
    channel = grpc.insecure_channel(server_address)
    stub = robot_service_pb2_grpc.RobotServiceStub(channel)

    print(f"Grabbing live image (camera: {camera_key}) …")
    live_img = _grab_live_image(stub, camera_key, target_h=h, target_w=w)
    live_img = _to_uint8(live_img)
    channel.close()
    print(f"  Live image shape: {live_img.shape}")

    # ── 3. Compute comparison metrics ─────────────────────────────
    ds_f = dataset_img.astype(np.float64)
    lv_f = live_img.astype(np.float64)

    abs_diff = np.abs(ds_f - lv_f)
    mean_diff = abs_diff.mean()
    max_diff = abs_diff.max()

    # Per-pixel L2 across channels, then normalise for visualisation
    pixel_diff = np.sqrt(np.sum((ds_f - lv_f) ** 2, axis=-1))
    pixel_diff_norm = (pixel_diff / pixel_diff.max() * 255).astype(np.uint8) if pixel_diff.max() > 0 else pixel_diff.astype(np.uint8)

    # ── 4. Build comparison figure ────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: dataset frame
    axes[0, 0].imshow(dataset_img)
    axes[0, 0].set_title(f"Dataset ({ep_label})", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # Top-right: live frame
    axes[0, 1].imshow(live_img)
    axes[0, 1].set_title("Live Camera", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    # Bottom-left: blended overlay (50/50)
    blended = (0.5 * ds_f + 0.5 * lv_f).astype(np.uint8)
    axes[1, 0].imshow(blended)
    axes[1, 0].set_title("Overlay (50/50 blend)", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    # Bottom-right: pixel-wise difference heatmap
    im = axes[1, 1].imshow(pixel_diff_norm, cmap="hot")
    axes[1, 1].set_title(
        f"Pixel Difference (mean={mean_diff:.1f}, max={max_diff:.0f})",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 1].axis("off")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Pre-Deploy Camera Check — {camera_key}\n"
        f"Dataset: {dataset_dir.name}  |  Robot: {server_address}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    # ── 5. Save & show ────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"pre_deploy_check_{timestamp}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison to {save_path}")

    # Print a quick numeric summary
    print(f"\n{'─' * 50}")
    print(f"  Camera:        {camera_key}")
    print(f"  Mean abs diff: {mean_diff:.2f}  (out of 255)")
    print(f"  Max abs diff:  {max_diff:.0f}")
    print(f"  RMSE:          {np.sqrt(np.mean((ds_f - lv_f) ** 2)):.2f}")
    print(f"{'─' * 50}")

    if mean_diff < 15:
        print("  ✅ Images look very similar — good to deploy!")
    elif mean_diff < 30:
        print("  ⚠️  Moderate difference — double-check lighting / positioning.")
    else:
        print("  ❌ Large difference — scene may have changed significantly!")

    try:
        plt.show()
    except Exception:
        pass  # headless environment

    return save_path


# ── CLI ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare dataset first-frame vs live camera before deploying",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset source — local path or HuggingFace repo (mutually exclusive)
    ds_group = parser.add_mutually_exclusive_group(required=True)
    ds_group.add_argument(
        "-d", "--dataset",
        type=pathlib.Path,
        metavar="PATH",
        help="Path to a local LeRobot dataset directory",
    )
    ds_group.add_argument(
        "--hf-repo-id",
        type=str,
        metavar="REPO",
        help="HuggingFace dataset repo ID (e.g. pokeandwiggle/stack_one_brick). "
             "Downloaded to /data/lerobot/<org>_<name> automatically.",
    )

    parser.add_argument(
        "-s", "--robot-server",
        default="localhost:50051",
        metavar="ADDR",
        help="Robot service server address (default: localhost:50051)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="rgb_static",
        metavar="KEY",
        help="Camera key to compare, e.g. rgb_static, rgb_left, rgb_right (default: rgb_static)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        metavar="N",
        help="Compare against a specific episode's first frame (default: mean of all)",
    )
    parser.add_argument(
        "-o", "--output",
        type=pathlib.Path,
        default=None,
        metavar="DIR",
        help="Output directory for the comparison image (default: outputs/)",
    )
    return parser


def main():
    args = build_parser().parse_args()

    dataset_dir = _resolve_dataset_dir(args.dataset, args.hf_repo_id)

    run_check(
        dataset_dir=dataset_dir,
        server_address=args.robot_server,
        camera_key=args.camera,
        episode=args.episode,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
