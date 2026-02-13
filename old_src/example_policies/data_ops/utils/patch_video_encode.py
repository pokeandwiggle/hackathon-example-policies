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

import functools
import glob
import logging
from pathlib import Path

import av

# Import the original module from lerobot
import lerobot.datasets.video_utils
import torch
from PIL import Image

"""HARDWARE ENCODING DOES NOT SEEM TO BE WORTH IT REGARDING QUALITY / SPEED"""


@functools.wraps(lerobot.datasets.video_utils.encode_video_frames)
def encode_video_frames_gpu_simple(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    # All other parameters are removed and hardcoded below.
    # We only keep the essential `overwrite` flag.
    overwrite: bool = False,
) -> None:
    """
    (Monkey-patched for simple, high-performance NVIDIA GPU encoding)
    Encodes video frames using av1_nvenc with hardcoded quality settings
    equivalent to the original function's defaults for machine learning.
    """
    # --- Hardcoded settings for optimal GPU performance and quality ---
    gpu_vcodec = "av1_nvenc"
    pix_fmt = "yuv420p"  # Most compatible pixel format
    log_level = av.logging.ERROR

    video_options = {
        # `g=2`: Group of Pictures. Kept from the original ML-tuned settings.
        "g": "2",
        "bf": "0",
        # `cq=32`: Constant Quality. Chosen to be visually similar to `libsvtav1`'s `crf=30`.
        # This is the most important quality parameter.
        "cq": "20",
        # `preset=p5`: A good balance between encoding speed and quality for NVENC.
        # Options range from p1 (fastest) to p7 (best quality).
        "preset": "p7",
        "multipass": "qres",
    }
    # --- End of hardcoded settings ---

    logging.info(
        f"Using simplified GPU encoder '{gpu_vcodec}' with options: {video_options}"
    )

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)
    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    template = "frame_" + ("[0-9]" * 6) + ".png"
    input_list = sorted(
        glob.glob(str(imgs_dir / template)),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )

    if len(input_list) == 0:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")
    dummy_image = Image.open(input_list[0])
    width, height = dummy_image.size

    if log_level is not None:
        logging.getLogger("libav").setLevel(log_level)

    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(gpu_vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        for input_data in input_list:
            input_image = Image.open(input_data).convert("RGB")
            input_frame = av.VideoFrame.from_image(input_image)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


def patch_video_encoding():
    if torch.cuda.is_available():
        logging.info("Patching lerobot.datasets.video_utils.encode_video_frames...")
        lerobot.datasets.video_utils.encode_video_frames = (
            encode_video_frames_gpu_simple
        )
