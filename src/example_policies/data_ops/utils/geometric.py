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

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def quat_to_axis_angle(quat):
    """Convert a quaternion to axis-angle representation.

    Args:
        quat (np.ndarray): Quaternion [x, y, z, w].

    Returns:
        np.ndarray: Axis-angle representation [x, y, z] where magnitude is angle.
    """
    r = R.from_quat(quat)
    return r.as_rotvec()


def quaternion_to_delta_axis_angle(q_actual, q_desired):
    """
    Convert two quaternions to delta axis-angle representation.
    Fast and simple implementation using scipy.

    Args:
        q_actual: Current quaternion [x, y, z, w]
        q_desired: Desired quaternion [x, y, z, w]

    Returns:
        delta_axis_angle: [x, y, z] axis-angle representation (angle encoded in magnitude)
    """
    # Create rotations and compute delta
    r_actual = R.from_quat(q_actual)
    r_desired = R.from_quat(q_desired)
    r_delta = r_desired * r_actual.inv()

    return r_delta.as_rotvec()


def axis_angle_to_quaternion_xyzw(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle representation to quaternion (xyzw format).

    Args:
        axis_angle (np.ndarray): Axis-angle representation [x, y, z] where magnitude is angle.

    Returns:
        np.ndarray: Quaternion representation [x, y, z, w].
    """
    # Convert to rotation and then quaternion
    r = R.from_rotvec(axis_angle)
    quat_xyzw = r.as_quat()  # scipy returns [x, y, z, w]

    return quat_xyzw


def positive_quat(pose: np.ndarray) -> np.ndarray:
    """Ensures positive quaternion

    Args:
        pose (np.ndarray): Can either be shape 4 (xyzw) or 7 (xyz xyzw)

    Returns:
        np.ndarray: pose array with positive omega
    """

    if pose[-1] < 0:
        pose[-4:] = -1 * pose[-4:]
    return pose


def axis_angle_to_quat_torch(aa: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Convert axis-angle (...,3) to quaternions (...,4) in xyzw format."""
    angle = aa.norm(dim=-1, keepdim=True)

    # Create quat tensor with correct shape
    quat = torch.zeros(*aa.shape[:-1], 4, device=aa.device, dtype=aa.dtype)

    # Handle small angles separately to avoid numerical issues
    small_angle_mask = (angle < eps).squeeze(-1)
    large_angle_mask = ~small_angle_mask

    if torch.any(large_angle_mask):
        # For large angles, use the standard conversion
        axis = aa[large_angle_mask] / angle[large_angle_mask]
        half = 0.5 * angle[large_angle_mask]
        sin_half = torch.sin(half)
        quat[large_angle_mask, :3] = axis * sin_half
        quat[large_angle_mask, 3] = torch.cos(half).squeeze(-1)

    if torch.any(small_angle_mask):
        # For small angles, use Taylor series approximation
        # sin(θ/2) ≈ θ/2, cos(θ/2) ≈ 1
        # This gives us quat = [aa/2, 1] for small angles
        quat[small_angle_mask, :3] = aa[small_angle_mask] * 0.5
        quat[small_angle_mask, 3] = 1.0

    return quat


def quat_mul_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions (...,4) in xyzw format.
    Computes q1 * q2.
    """
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dim=-1,
    )
