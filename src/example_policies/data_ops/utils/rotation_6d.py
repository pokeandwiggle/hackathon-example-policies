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

"""6D rotation representation utilities (numpy + torch).

Implements the continuous 6D rotation representation from:
    Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019.

The 6D representation consists of the first two columns of the 3x3 rotation matrix,
flattened to a 6-element vector. Recovery of the full rotation matrix uses
Gram-Schmidt orthogonalization.

This representation is used by TRI's LBM and avoids discontinuities present in
Euler angles and quaternions, making it well-suited for learning-based methods.
The values are naturally bounded in [-1, 1], so percentile normalization can be
skipped for these features.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


# =============================================================================
# NumPy implementations (data pipeline)
# =============================================================================


def rotation_matrix_to_6d_np(rot_matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to 6D representation.

    Takes the first two columns of the rotation matrix and flattens them.

    Args:
        rot_matrix: Rotation matrix of shape (..., 3, 3).

    Returns:
        6D rotation vector of shape (..., 6).
    """
    # Take first two columns: col0 (3,) and col1 (3,)
    col0 = rot_matrix[..., :, 0]  # (..., 3)
    col1 = rot_matrix[..., :, 1]  # (..., 3)
    return np.concatenate([col0, col1], axis=-1)  # (..., 6)


def rotation_6d_to_matrix_np(r6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to 3x3 rotation matrix via Gram-Schmidt.

    Args:
        r6d: 6D rotation vector of shape (..., 6).

    Returns:
        Rotation matrix of shape (..., 3, 3).
    """
    a1 = r6d[..., :3]  # (..., 3)
    a2 = r6d[..., 3:6]  # (..., 3)

    # Gram-Schmidt: normalize first vector
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)

    # Project a2 onto b1 and subtract
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)

    # Third column via cross product
    b3 = np.cross(b1, b2)

    # Stack into rotation matrix
    rot_matrix = np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)
    return rot_matrix


def quat_to_6d_np(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion (xyzw) to 6D rotation representation.

    Args:
        quat_xyzw: Quaternion in xyzw format, shape (4,) or (N, 4).

    Returns:
        6D rotation vector, shape (6,) or (N, 6).
    """
    single = quat_xyzw.ndim == 1
    if single:
        quat_xyzw = quat_xyzw[np.newaxis, :]

    rot = R.from_quat(quat_xyzw)  # scipy expects xyzw
    matrices = rot.as_matrix()  # (N, 3, 3)
    r6d = rotation_matrix_to_6d_np(matrices)

    if single:
        return r6d[0]
    return r6d


def rotation_6d_to_quat_np(r6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to quaternion (xyzw).

    Args:
        r6d: 6D rotation vector, shape (6,) or (N, 6).

    Returns:
        Quaternion in xyzw format, shape (4,) or (N, 4).
    """
    single = r6d.ndim == 1
    if single:
        r6d = r6d[np.newaxis, :]

    matrices = rotation_6d_to_matrix_np(r6d)  # (N, 3, 3)
    rot = R.from_matrix(matrices)
    quats = rot.as_quat()  # xyzw

    if single:
        return quats[0]
    return quats


def compute_relative_pose_6d_np(
    ref_pos: np.ndarray,
    ref_quat_xyzw: np.ndarray,
    target_pos: np.ndarray,
    target_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """Compute a relative pose (position delta + 6D rotation delta).

    Computes the pose of `target` expressed in the frame of `ref`:
        delta_pos = target_pos - ref_pos
        delta_R   = R_ref^{-1} @ R_target  →  6D representation

    Args:
        ref_pos: Reference position (3,).
        ref_quat_xyzw: Reference quaternion (4,) in xyzw.
        target_pos: Target position (3,).
        target_quat_xyzw: Target quaternion (4,) in xyzw.

    Returns:
        9-element array: [delta_xyz(3), delta_rot6d(6)].
    """
    delta_pos = target_pos - ref_pos

    r_ref = R.from_quat(ref_quat_xyzw)
    r_target = R.from_quat(target_quat_xyzw)
    r_delta = r_ref.inv() * r_target
    delta_6d = rotation_matrix_to_6d_np(r_delta.as_matrix())

    return np.concatenate([delta_pos, delta_6d])


# =============================================================================
# PyTorch implementations (training / inference)
# =============================================================================


def rotation_matrix_to_6d_torch(rot_matrix: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to 6D representation.

    Args:
        rot_matrix: Rotation matrix of shape (..., 3, 3).

    Returns:
        6D rotation vector of shape (..., 6).
    """
    col0 = rot_matrix[..., :, 0]  # (..., 3)
    col1 = rot_matrix[..., :, 1]  # (..., 3)
    return torch.cat([col0, col1], dim=-1)  # (..., 6)


def rotation_6d_to_matrix_torch(r6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix via Gram-Schmidt.

    Args:
        r6d: 6D rotation vector of shape (..., 6).

    Returns:
        Rotation matrix of shape (..., 3, 3).
    """
    a1 = r6d[..., :3]
    a2 = r6d[..., 3:6]

    # Gram-Schmidt
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.linalg.cross(b1, b2)

    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)


def quat_to_6d_torch(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (xyzw) to 6D rotation representation.

    Args:
        quat_xyzw: Quaternion in xyzw format, shape (..., 4).

    Returns:
        6D rotation vector, shape (..., 6).
    """
    # Build rotation matrix from quaternion (xyzw), row-major order
    x, y, z, w = quat_xyzw.unbind(-1)

    rot = torch.stack(
        [
            1 - 2 * (y * y + z * z),  # R[0,0]
            2 * (x * y - w * z),       # R[0,1]
            2 * (x * z + w * y),       # R[0,2]
            2 * (x * y + w * z),       # R[1,0]
            1 - 2 * (x * x + z * z),  # R[1,1]
            2 * (y * z - w * x),       # R[1,2]
            2 * (x * z - w * y),       # R[2,0]
            2 * (y * z + w * x),       # R[2,1]
            1 - 2 * (x * x + y * y),  # R[2,2]
        ],
        dim=-1,
    ).reshape(*quat_xyzw.shape[:-1], 3, 3)

    return rotation_matrix_to_6d_torch(rot)


def rotation_6d_to_quat_torch(r6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to quaternion (xyzw).

    Uses Gram-Schmidt to recover the rotation matrix, then extracts quaternion.

    Args:
        r6d: 6D rotation vector, shape (..., 6).

    Returns:
        Quaternion in xyzw format, shape (..., 4).
    """
    rot = rotation_6d_to_matrix_torch(r6d)  # (..., 3, 3)
    return _matrix_to_quat_torch(rot)


def compute_relative_transform_6d_torch(
    ref_pos: torch.Tensor,
    ref_rot6d: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot6d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute relative transform in 6D rotation space.

    Args:
        ref_pos: Reference position (..., 3).
        ref_rot6d: Reference 6D rotation (..., 6).
        target_pos: Target position (..., 3).
        target_rot6d: Target 6D rotation (..., 6).

    Returns:
        Tuple of (delta_pos (..., 3), delta_rot6d (..., 6)).
    """
    delta_pos = target_pos - ref_pos

    R_ref = rotation_6d_to_matrix_torch(ref_rot6d)
    R_target = rotation_6d_to_matrix_torch(target_rot6d)
    R_delta = R_ref.transpose(-2, -1) @ R_target  # R_ref^T @ R_target
    delta_rot6d = rotation_matrix_to_6d_torch(R_delta)

    return delta_pos, delta_rot6d


def compose_transform_6d_torch(
    ref_pos: torch.Tensor,
    ref_rot6d: torch.Tensor,
    delta_pos: torch.Tensor,
    delta_rot6d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose a relative transform with a reference pose to get absolute pose.

    Inverse of compute_relative_transform_6d_torch:
        abs_pos = ref_pos + delta_pos
        abs_R   = R_ref @ R_delta

    Args:
        ref_pos: Reference position (..., 3).
        ref_rot6d: Reference 6D rotation (..., 6).
        delta_pos: Relative position (..., 3).
        delta_rot6d: Relative 6D rotation (..., 6).

    Returns:
        Tuple of (abs_pos (..., 3), abs_rot6d (..., 6)).
    """
    abs_pos = ref_pos + delta_pos

    R_ref = rotation_6d_to_matrix_torch(ref_rot6d)
    R_delta = rotation_6d_to_matrix_torch(delta_rot6d)
    R_abs = R_ref @ R_delta
    abs_rot6d = rotation_matrix_to_6d_torch(R_abs)

    return abs_pos, abs_rot6d


# =============================================================================
# Internal helpers
# =============================================================================


def _matrix_to_quat_torch(rot: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion (xyzw format).

    Numerically stable method that picks the largest diagonal element.

    Args:
        rot: Rotation matrix, shape (..., 3, 3).

    Returns:
        Quaternion in xyzw format, shape (..., 4).
    """
    batch_shape = rot.shape[:-2]
    m = rot.reshape(-1, 3, 3)
    batch_size = m.shape[0]

    # Shepperd's method
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    quat = torch.zeros(batch_size, 4, device=rot.device, dtype=rot.dtype)

    # Case 1: trace > 0
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2  # s = 4w
    mask = trace > 0
    if mask.any():
        quat[mask, 3] = 0.25 * s[mask]
        quat[mask, 0] = (m[mask, 2, 1] - m[mask, 1, 2]) / s[mask]
        quat[mask, 1] = (m[mask, 0, 2] - m[mask, 2, 0]) / s[mask]
        quat[mask, 2] = (m[mask, 1, 0] - m[mask, 0, 1]) / s[mask]

    # Case 2: m[0,0] is largest diagonal
    mask2 = (~mask) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    if mask2.any():
        s2 = torch.sqrt(torch.clamp(1.0 + m[mask2, 0, 0] - m[mask2, 1, 1] - m[mask2, 2, 2], min=1e-10)) * 2
        quat[mask2, 0] = 0.25 * s2
        quat[mask2, 1] = (m[mask2, 0, 1] + m[mask2, 1, 0]) / s2
        quat[mask2, 2] = (m[mask2, 0, 2] + m[mask2, 2, 0]) / s2
        quat[mask2, 3] = (m[mask2, 2, 1] - m[mask2, 1, 2]) / s2

    # Case 3: m[1,1] is largest diagonal
    mask3 = (~mask) & (~mask2) & (m[:, 1, 1] > m[:, 2, 2])
    if mask3.any():
        s3 = torch.sqrt(torch.clamp(1.0 + m[mask3, 1, 1] - m[mask3, 0, 0] - m[mask3, 2, 2], min=1e-10)) * 2
        quat[mask3, 0] = (m[mask3, 0, 1] + m[mask3, 1, 0]) / s3
        quat[mask3, 1] = 0.25 * s3
        quat[mask3, 2] = (m[mask3, 1, 2] + m[mask3, 2, 1]) / s3
        quat[mask3, 3] = (m[mask3, 0, 2] - m[mask3, 2, 0]) / s3

    # Case 4: m[2,2] is largest diagonal
    mask4 = (~mask) & (~mask2) & (~mask3)
    if mask4.any():
        s4 = torch.sqrt(torch.clamp(1.0 + m[mask4, 2, 2] - m[mask4, 0, 0] - m[mask4, 1, 1], min=1e-10)) * 2
        quat[mask4, 0] = (m[mask4, 0, 2] + m[mask4, 2, 0]) / s4
        quat[mask4, 1] = (m[mask4, 1, 2] + m[mask4, 2, 1]) / s4
        quat[mask4, 2] = 0.25 * s4
        quat[mask4, 3] = (m[mask4, 1, 0] - m[mask4, 0, 1]) / s4

    return quat.reshape(*batch_shape, 4)
