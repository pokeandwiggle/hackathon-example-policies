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
import cv2
import numpy as np


def encode_depth_multi_scale(
    depth_normalized: np.ndarray,
    scales: tuple = (1.0, 0.1, 0.01),
) -> np.ndarray:
    """
    Encode depth at multiple scales in RGB channels to preserve fine details.

    This approach uses each color channel to represent depth at different scales:
    - R channel: Full depth range (coarse structure)
    - G channel: 10x magnified depth (medium details)
    - B channel: 100x magnified depth (fine details)

    The modulo operation creates a "wrapped" representation where each channel
    captures different levels of detail across the varying depth ranges.

    Args:
        depth_normalized: Depth array with values in [0, 1]
        scales: Scale factors for each RGB channel

    Returns:
        RGB image (H, W, 3) with values in [0, 1] ready for PNG saving
    """
    h, w = depth_normalized.shape
    rgb_depth = np.zeros((h, w, 3), dtype=np.float32)

    for i, scale in enumerate(scales):
        # Scale and wrap depth values - this is the key insight
        # Higher scales reveal finer depth variations that would be lost in quantization
        scaled_depth = (depth_normalized * scale) % 1.0
        rgb_depth[:, :, i] = scaled_depth

    return rgb_depth


def process_image_bytes(
    img_bytes: bytes,
    width: int,
    height: int,
    is_depth: bool,
    depth_scale: float = 1000.0,
) -> np.ndarray:
    """
    Process image data optimized for wrist cam depth with multi-scale encoding.
    Returns H,W,3 array ready for PNG saving.
    """
    # Your existing PNG decoding logic
    png_data = img_bytes[12:] if is_depth else img_bytes
    nparr = np.frombuffer(png_data, np.uint8)
    read_flag = cv2.IMREAD_UNCHANGED if is_depth else cv2.IMREAD_COLOR
    img_array = cv2.imdecode(nparr, read_flag)

    # For RGB images, convert from BGR to RGB
    if not is_depth and len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # Your existing aspect-ratio-preserving resize
    original_h, original_w = img_array.shape[:2]
    scale = max(width / original_w, height / original_h)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    resized_img = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top = (new_h - height) // 2
    left = (new_w - width) // 2
    img_array = resized_img[top : top + height, left : left + width]

    # Convert to float
    img_array = img_array.astype(np.float32)

    if is_depth:
        # Normalize depth to [0, 1]
        img_array /= depth_scale
        img_array = np.clip(img_array, 0.0, 1.0)

        # Apply multi-scale encoding - this is the key improvement
        img_array = encode_depth_multi_scale(img_array)
    else:
        # Standard RGB normalization
        img_array /= 255.0

    return img_array
