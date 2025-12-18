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

import torch
from torch import nn

from ...utils.action_order import (
    DUAL_ABS_LEFT_POS_IDXS,
    DUAL_ABS_LEFT_QUAT_IDXS,
    DUAL_ABS_RIGHT_POS_IDXS,
    DUAL_ABS_RIGHT_QUAT_IDXS,
    DUAL_DELTA_LEFT_POS_IDXS,
    DUAL_DELTA_LEFT_ROT_IDXS,
    DUAL_DELTA_RIGHT_POS_IDXS,
    DUAL_DELTA_RIGHT_ROT_IDXS,
    GET_LEFT_GRIPPER_IDX,
    GET_RIGHT_GRIPPER_IDX,
    ActionMode,
)


def quat_geodesic_angle(q_pred, q_gt, eps=1e-5):
    """Calculates the geodesic angle between two quaternions, handling antipodal symmetry."""
    # Normalize quaternions to prevent numerical issues
    q_pred = q_pred / (q_pred.norm(dim=-1, keepdim=True) + eps)
    q_gt = q_gt / (q_gt.norm(dim=-1, keepdim=True) + eps)
    # Dot product, handling antipodal symmetry
    dot = (q_pred * q_gt).sum(-1).abs().clamp(max=1.0 - eps)
    return torch.square(2.0 * torch.acos(dot))  # radians


class PoseLoss(nn.Module):
    """
    Calculates a composite loss for a dual-arm action, including position (MSE),
    orientation (geodesic), and gripper (L1) components.
    """

    # Define slices for action tensor components
    LEFT_POS_IDXS = DUAL_ABS_LEFT_POS_IDXS
    LEFT_QUAT_IDXS = DUAL_ABS_LEFT_QUAT_IDXS
    RIGHT_POS_IDXS = DUAL_ABS_RIGHT_POS_IDXS
    RIGHT_QUAT_IDXS = DUAL_ABS_RIGHT_QUAT_IDXS
    LEFT_GRIPPER_IDX = GET_LEFT_GRIPPER_IDX(ActionMode.ABS_TCP)
    RIGHT_GRIPPER_IDX = GET_RIGHT_GRIPPER_IDX(ActionMode.ABS_TCP)

    def __init__(self, pos_weight=1.0, quat_weight=1.0, grip_weight=1.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.quat_weight = quat_weight
        self.grip_weight = grip_weight

    def _calculate_arm_loss(self, predicted, target, pos_idxs, quat_idxs):
        """Helper to calculate position and quaternion loss for a single arm."""
        pos_mse = nn.functional.mse_loss(
            predicted[:, :, pos_idxs], target[:, :, pos_idxs], reduction="none"
        ).mean(dim=-1)

        quat_geodesic = quat_geodesic_angle(
            predicted[:, :, quat_idxs], target[:, :, quat_idxs]
        )
        return pos_mse, quat_geodesic

    def forward(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Calculates the total loss and a dictionary of sub-losses for logging.
        Returns a loss tensor of shape (batch, sequence).
        """
        has_gripper = predicted.shape[-1] == 16

        # Calculate losses for each arm
        left_pos_mse, left_quat_geo = self._calculate_arm_loss(
            predicted, target, self.LEFT_POS_IDXS, self.LEFT_QUAT_IDXS
        )
        right_pos_mse, right_quat_geo = self._calculate_arm_loss(
            predicted, target, self.RIGHT_POS_IDXS, self.RIGHT_QUAT_IDXS
        )

        # Combine position and quaternion losses
        pos_loss = left_pos_mse + right_pos_mse
        quat_loss = left_quat_geo + right_quat_geo

        # Calculate gripper loss
        if has_gripper:
            left_grip_l1 = nn.functional.l1_loss(
                predicted[:, :, self.LEFT_GRIPPER_IDX],
                target[:, :, self.LEFT_GRIPPER_IDX],
                reduction="none",
            )
            right_grip_l1 = nn.functional.l1_loss(
                predicted[:, :, self.RIGHT_GRIPPER_IDX],
                target[:, :, self.RIGHT_GRIPPER_IDX],
                reduction="none",
            )
            grip_loss = left_grip_l1 + right_grip_l1
        else:
            grip_loss = torch.tensor(0.0, device=predicted.device)

        # Combine all losses with their respective weights
        total_loss = (
            self.pos_weight * pos_loss
            + self.quat_weight * quat_loss
            + self.grip_weight * grip_loss
        )

        # Populate dictionary for logging
        loss_dict = {
            "pos_loss": pos_loss.mean().item(),
            "quat_loss": quat_loss.mean().item(),
            "grip_loss": grip_loss.mean().item() if has_gripper else 0.0,
        }

        return total_loss.unsqueeze(-1), loss_dict


class IntegratedDeltaPoseLoss(PoseLoss):
    """
    A variant of PoseLoss that computes loss based on integrated deltas
    between consecutive time steps, rather than absolute poses.

    For position we sum deltas; for orientation we compose incremental rotations.
    Gripper values are assumed ABSOLUTE per timestep; we take the last value.
    """

    DELTA_LEFT_POS_IDXS = DUAL_DELTA_LEFT_POS_IDXS
    DELTA_LEFT_ROT_IDXS = DUAL_DELTA_LEFT_ROT_IDXS
    DELTA_RIGHT_POS_IDXS = DUAL_DELTA_RIGHT_POS_IDXS
    DELTA_RIGHT_ROT_IDXS = DUAL_DELTA_RIGHT_ROT_IDXS

    def forward(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Calculates loss by first integrating predicted and target delta trajectories
        into absolute trajectories, then computing a pose loss on them.
        """
        predicted_abs_traj = self.integrate_trajectory_sequence(predicted)
        target_abs_traj = self.integrate_trajectory_sequence(target)
        loss, loss_dict = super().forward(predicted_abs_traj, target_abs_traj)

        return loss, loss_dict

    def integrate_trajectory(
        self, trajectory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate delta trajectory to absolute poses.

        Args:
            trajectory: (B, T, 14) delta actions (pos/axis-angle/pos/axis-angle[/absolute grippers])
        Returns:
            tuple of (abs_traj, final_pose):
                abs_traj: (B, T, 16) absolute poses for all timesteps
                final_pose: (B, 1, 16) final absolute pose (last timestep)
        """
        # Reuse the new per-step integration and pick the last step
        abs_traj = self.integrate_trajectory_sequence(trajectory)  # (B,T,16)
        final_pose = abs_traj[:, -1:, :]  # (B,1,16)
        return abs_traj, final_pose

    def integrate_trajectory_sequence(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Integrate a sequence of delta actions into absolute poses for every timestep.

        Args:
            trajectory: (B, T, 14) delta actions
                Layout (first 12 dims always delta):
                  0:3   left  dpos
                  3:6   left  d(axis-angle)
                  6:9   right dpos
                  9:12  right d(axis-angle)
                  12    (optional) left  gripper ABS
                  13    (optional) right gripper ABS
        Returns:
            abs_traj: (B, T, 16) absolute TCP poses + grippers for each step
        """
        B, T, D = trajectory.shape
        device = trajectory.device
        dtype = trajectory.dtype

        # --- Positions: cumulative sum of deltas ---
        left_pos_cum = torch.cumsum(
            trajectory[:, :, self.DELTA_LEFT_POS_IDXS], dim=1
        )  # (B,T,3)
        right_pos_cum = torch.cumsum(
            trajectory[:, :, self.DELTA_RIGHT_POS_IDXS], dim=1
        )  # (B,T,3)

        # --- Rotations: compose delta axis-angle increments into cumulative quats ---
        left_rot_deltas = trajectory[:, :, self.DELTA_LEFT_ROT_IDXS]  # (B,T,3)
        right_rot_deltas = trajectory[:, :, self.DELTA_RIGHT_ROT_IDXS]

        # Convert each delta axis-angle to incremental quaternion
        left_q_inc = self._axis_angle_to_quat(left_rot_deltas)  # (B,T,4)
        right_q_inc = self._axis_angle_to_quat(right_rot_deltas)

        # Initialize identity quaternions
        ident = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype).view(1, 1, 4)
        left_q_list = []
        right_q_list = []
        qL = ident.repeat(B, 1, 1)  # (B,1,4)
        qR = ident.repeat(B, 1, 1)
        for t in range(T):
            # Compose: q_new = q_prev * q_inc_t
            qL = self._quat_mul(qL[:, 0], left_q_inc[:, t]).unsqueeze(1)
            qR = self._quat_mul(qR[:, 0], right_q_inc[:, t]).unsqueeze(1)
            left_q_list.append(qL)
            right_q_list.append(qR)
        left_quat_cum = torch.cat(left_q_list, dim=1)  # (B,T,4)
        right_quat_cum = torch.cat(right_q_list, dim=1)

        # --- Grippers: copy absolute value per timestep if present, else zeros ---
        has_grip = D >= 14
        if has_grip:
            left_grip_seq = trajectory[:, :, 12]
            right_grip_seq = trajectory[:, :, 13]
        else:
            left_grip_seq = torch.zeros(B, T, device=device, dtype=dtype)
            right_grip_seq = torch.zeros(B, T, device=device, dtype=dtype)

        # --- Assemble absolute trajectory tensor ---
        abs_traj = torch.zeros(B, T, 16, device=device, dtype=dtype)
        abs_traj[:, :, self.LEFT_POS_IDXS] = left_pos_cum
        abs_traj[:, :, self.LEFT_QUAT_IDXS] = left_quat_cum
        abs_traj[:, :, self.RIGHT_POS_IDXS] = right_pos_cum
        abs_traj[:, :, self.RIGHT_QUAT_IDXS] = right_quat_cum
        abs_traj[:, :, self.LEFT_GRIPPER_IDX] = left_grip_seq
        abs_traj[:, :, self.RIGHT_GRIPPER_IDX] = right_grip_seq
        return abs_traj

    # --- Rotation helpers ---
    def _axis_angle_to_quat(self, aa: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Convert axis-angle (B,T,3) to quaternions (B,T,4)."""
        angle = aa.norm(dim=-1, keepdim=True)  # (B,T,1)
        axis = aa / (angle + eps)
        half = 0.5 * angle
        sin_half = torch.sin(half)
        quat = torch.zeros(*aa.shape[:-1], 4, device=aa.device, dtype=aa.dtype)
        quat[..., :3] = axis * sin_half
        quat[..., 3] = torch.cos(half).squeeze(-1)
        return quat

    def _quat_mul(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Hamilton product of two quaternions (…,4)."""
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

    def _compose_axis_angle_sequence(self, rot_deltas: torch.Tensor) -> torch.Tensor:
        """Compose sequence of axis-angle rotations into a single quaternion (B,4)."""
        B, T, _ = rot_deltas.shape
        q_inc = self._axis_angle_to_quat(rot_deltas)  # (B,T,4)
        q_cur = (
            torch.tensor([0, 0, 0, 1], dtype=rot_deltas.dtype, device=rot_deltas.device)
            .view(1, 1, 4)
            .repeat(B, 1, 1)
        )
        for t in range(T):
            q_cur = self._quat_mul(q_cur[:, 0], q_inc[:, t]).unsqueeze(1)
        return q_cur[:, 0]
