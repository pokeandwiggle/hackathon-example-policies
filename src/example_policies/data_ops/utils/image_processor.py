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

import io

import cv2
import numpy as np
from PIL import Image, ImageOps


def process_image_bytes(
    img_bytes: bytes,
    width: int,
    height: int,
    is_depth: bool,
    depth_scale: float = 1000,
) -> np.ndarray:
    """
    Process image data for model input using OpenCV for performance.
    This function performs a "fit" operation: it resizes the image to fit
    within the target dimensions while maintaining aspect ratio, then crops
    the center to match the target dimensions exactly.
    """
    png_data = img_bytes[12:] if is_depth else img_bytes

    # Decode the image from bytes using OpenCV
    nparr = np.frombuffer(png_data, np.uint8)
    read_flag = cv2.IMREAD_UNCHANGED if is_depth else cv2.IMREAD_COLOR
    img_array = cv2.imdecode(nparr, read_flag)

    # For RGB images, convert from BGR to RGB
    if not is_depth and len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # --- Start of Aspect-Ratio-Preserving Crop and Resize ---
    original_h, original_w = img_array.shape[:2]
    target_h, target_w = height, width

    # Calculate the scaling factor, taking the larger of the two to ensure the image covers the target area
    scale = max(target_w / original_w, target_h / original_h)

    # Calculate new dimensions after scaling
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    # Resize the image with the new dimensions
    resized_img = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Calculate coordinates for center cropping
    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2

    # Crop the resized image to the final target size
    img_array = resized_img[top : top + target_h, left : left + target_w]
    # --- End of Aspect-Ratio-Preserving Crop and Resize ---

    # Convert to float and normalize
    img_array = img_array.astype(np.float32)
    if is_depth:
        img_array /= depth_scale
    else:
        img_array /= 255.0

    # Ensure the image has 3 channels if it's grayscale
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    return img_array
