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
    example_policies pre-deploy-check \
        --dataset /data/lerobot/stack_one_brick \
        --robot-server 192.168.0.101:50051

    # Dataset from HuggingFace Hub:
    example_policies pre-deploy-check \
        --hf-repo-id pokeandwiggle/stack_one_brick \
        --robot-server 192.168.0.101:50051

    # Check a specific episode's first frame instead of the mean:
    example_policies pre-deploy-check \
        --dataset /data/lerobot/stack_one_brick \
        --robot-server 192.168.0.101:50051 \
        --episode 3

    # Override which camera to compare (default: rgb_static):
    example_policies pre-deploy-check \
        --dataset /data/lerobot/stack_one_brick \
        --robot-server 192.168.0.101:50051 \
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
from example_policies.default_paths import DATASETS_DIR, PLOTS_DIR
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
        output_dir = PLOTS_DIR / "pre_deploy_check"
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
    rmse = np.sqrt(np.mean((ds_f - lv_f) ** 2))

    # Per-pixel L2 across channels, then normalise for visualisation
    pixel_diff = np.sqrt(np.sum((ds_f - lv_f) ** 2, axis=-1))
    pixel_diff_norm = (pixel_diff / pixel_diff.max() * 255).astype(np.uint8) if pixel_diff.max() > 0 else pixel_diff.astype(np.uint8)

    # ── 3b. SSIM (structural similarity) ──────────────────────────
    # Computed on greyscale — measures structural changes, robust to
    # uniform brightness shifts.  Returns a per-pixel SSIM map too.
    ds_grey = cv2.cvtColor(dataset_img, cv2.COLOR_RGB2GRAY)
    lv_grey = cv2.cvtColor(live_img, cv2.COLOR_RGB2GRAY)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    _WIN = 11  # window size (must be odd)

    mu_ds = cv2.GaussianBlur(ds_grey.astype(np.float64), (_WIN, _WIN), 1.5)
    mu_lv = cv2.GaussianBlur(lv_grey.astype(np.float64), (_WIN, _WIN), 1.5)
    mu_ds_sq = mu_ds ** 2
    mu_lv_sq = mu_lv ** 2
    mu_ds_lv = mu_ds * mu_lv

    sigma_ds_sq = cv2.GaussianBlur((ds_grey.astype(np.float64)) ** 2, (_WIN, _WIN), 1.5) - mu_ds_sq
    sigma_lv_sq = cv2.GaussianBlur((lv_grey.astype(np.float64)) ** 2, (_WIN, _WIN), 1.5) - mu_lv_sq
    sigma_ds_lv = cv2.GaussianBlur(
        ds_grey.astype(np.float64) * lv_grey.astype(np.float64), (_WIN, _WIN), 1.5
    ) - mu_ds_lv

    ssim_map = ((2 * mu_ds_lv + C1) * (2 * sigma_ds_lv + C2)) / \
               ((mu_ds_sq + mu_lv_sq + C1) * (sigma_ds_sq + sigma_lv_sq + C2))
    ssim_score = float(ssim_map.mean())

    # Invert for visualisation: bright = low similarity (potential movement)
    ssim_diff_vis = ((1.0 - ssim_map) * 255).clip(0, 255).astype(np.uint8)

    # ── 3c. Edge-based structural diff ────────────────────────────
    # Canny edges are invariant to brightness — differences reveal
    # geometric changes (moved objects, shifted surfaces).
    edges_ds = cv2.Canny(ds_grey, 50, 150)
    edges_lv = cv2.Canny(lv_grey, 50, 150)

    # XOR highlights edges present in one image but not the other
    edge_diff = cv2.bitwise_xor(edges_ds, edges_lv)
    # Dilate for visibility
    edge_diff_vis = cv2.dilate(edge_diff, np.ones((3, 3), np.uint8), iterations=1)
    edge_change_pct = float(np.count_nonzero(edge_diff) / max(np.count_nonzero(edges_ds | edges_lv), 1) * 100)

    # ── 4. Build comparison figure (seaborn-inspired) ───────────────
    _BG = "#f0f0f0"
    _TEXT = "#333333"
    _SUBTITLE = "#777777"
    _ACCENT = "#4878d0"
    _WARN = "#ee854a"
    _FAIL = "#d65f5f"
    _PASS = "#6acc64"

    plt.rcParams.update({
        "figure.facecolor": _BG,
        "axes.facecolor": _BG,
        "text.color": _TEXT,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "savefig.facecolor": _BG,
        "savefig.edgecolor": _BG,
    })

    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = plt.figure(figsize=(14, 16))
    fig.set_facecolor(_BG)

    # Simple 3×2 grid — colorbars are attached to their image axes
    # via make_axes_locatable so they always match image height.
    gs = GridSpec(3, 2, figure=fig,
                  hspace=0.14, wspace=0.08,
                  left=0.03, right=0.97, top=0.92, bottom=0.03)

    # ── Titles ────────────────────────────────────────────────────
    fig.text(0.5, 0.975,
             f"Pre-Deploy Camera Check — {camera_key}",
             ha="center", fontsize=15, fontweight="bold", color=_TEXT)
    fig.text(0.5, 0.955,
             f"Dataset: {dataset_dir.name}  |  Robot: {server_address}",
             ha="center", fontsize=10, color=_SUBTITLE)

    # Verdict — incorporate SSIM and edge change %
    # SSIM: 1.0 = identical, <0.85 = significant structural change
    # Edge change: >30% = significant geometric change
    if ssim_score < 0.80 or edge_change_pct > 40:
        _verdict, _vcol = "FAIL", _FAIL
    elif ssim_score < 0.90 or edge_change_pct > 25 or mean_diff >= 30:
        _verdict, _vcol = "WARNING", _WARN
    elif mean_diff < 15 and ssim_score >= 0.95:
        _verdict, _vcol = "PASS", _PASS
    else:
        _verdict, _vcol = "WARNING", _WARN

    fig.text(0.95, 0.975, _verdict, fontsize=13, fontweight="bold",
             color="white", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=_vcol,
                       edgecolor="none", alpha=0.9))

    # Helper: add a panel with attached colorbar
    def _add_image_panel(gs_pos, img, title, cmap=None, add_cbar=False):
        ax = fig.add_subplot(gs_pos)
        ax.set_facecolor(_BG)
        kw = dict(aspect="equal")
        if cmap:
            kw["cmap"] = cmap
        im_obj = ax.imshow(img, **kw)
        ax.set_title(title, fontsize=11, fontweight="medium",
                     color=_TEXT, pad=8)
        ax.axis("off")
        # Thin border around image
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#cccccc")
            spine.set_linewidth(0.8)
        if add_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.06)
            cbar = fig.colorbar(im_obj, cax=cax)
            cbar.ax.tick_params(labelsize=8, colors=_SUBTITLE)
            cbar.outline.set_linewidth(0.5)
        return ax

    # Row 1: dataset vs live
    _add_image_panel(gs[0, 0], dataset_img,
                     f"Dataset ({ep_label})")
    _add_image_panel(gs[0, 1], live_img,
                     "Live Camera")

    # Row 2: overlay and pixel diff
    blended = (0.5 * ds_f + 0.5 * lv_f).astype(np.uint8)
    _add_image_panel(gs[1, 0], blended,
                     "Overlay (50/50 blend)")
    _add_image_panel(gs[1, 1], pixel_diff_norm,
                     f"Pixel Difference  —  mean {mean_diff:.1f}  |  max {max_diff:.0f}  |  RMSE {rmse:.1f}",
                     cmap="magma", add_cbar=True)

    # Row 3: SSIM diff and edge diff
    _add_image_panel(gs[2, 0], ssim_diff_vis,
                     f"SSIM Structural Diff  —  SSIM = {ssim_score:.3f}"
                     f"  ({'good' if ssim_score >= 0.95 else 'moderate' if ssim_score >= 0.85 else 'low'})",
                     cmap="magma", add_cbar=True)

    # Edge diff: show as coloured overlay (cyan = dataset-only, red = live-only)
    edge_rgb = np.zeros((*edge_diff.shape, 3), dtype=np.uint8)
    edge_rgb[..., 0] = np.where(edges_lv > 0, 200, 0)                      # red = live-only
    edge_rgb[..., 1] = np.where(edges_ds > 0, 200, 0)                      # green = dataset-only
    edge_rgb[edges_ds & edges_lv > 0] = [100, 100, 100]                    # grey = shared
    edge_rgb[edge_diff > 0] = np.where(                                     # highlight XOR
        edges_lv[edge_diff > 0, None] > 0, [255, 80, 80], [80, 220, 220]
    )
    _add_image_panel(gs[2, 1], edge_rgb,
                     f"Edge Diff (Canny XOR)  —  {edge_change_pct:.1f}% edges changed"
                     f"  (cyan=dataset, red=live)")

    # ── 5. Save & show ────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"pre_deploy_check_{timestamp}.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved comparison to {save_path}")

    # Print a quick numeric summary
    print(f"\n{'─' * 60}")
    print(f"  Camera:         {camera_key}")
    print(f"  Mean abs diff:  {mean_diff:.2f}  (out of 255)")
    print(f"  Max abs diff:   {max_diff:.0f}")
    print(f"  RMSE:           {rmse:.2f}")
    print(f"  SSIM:           {ssim_score:.4f}  (1.0 = identical)")
    print(f"  Edge change:    {edge_change_pct:.1f}%  (Canny XOR)")
    print(f"{'─' * 60}")

    if _verdict == "PASS":
        print("  ✅ Scene looks consistent — good to deploy!")
    elif _verdict == "WARNING":
        print("  ⚠️  Structural changes detected — check object positions & lighting.")
    else:
        print("  ❌ Significant scene change — objects may have moved!")

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
