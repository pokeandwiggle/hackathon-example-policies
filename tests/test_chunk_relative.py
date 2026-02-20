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

"""Tests for chunk-relative UMI-delta conversion (TRI LBM style).

Covers:
    1. abs_tcp_to_chunk_relative_umi_delta (pure function)
       - Identity: action = current TCP → zero position delta + identity rotation
       - Known translation
       - Known rotation
       - Grippers pass through unchanged
       - Batch / horizon broadcasting
    2. AbsTcpToChunkRelativeStep (ProcessorStep)
       - Transforms action in transition
       - No-op when no action present (inference)
       - transform_features updates action shape 16 → 20
    3. Round-trip: abs TCP → chunk-relative → compose back → abs TCP
    4. "Wider spread for future actions" property (TRI LBM §4.4.2)
"""

import math

import pytest
import torch
from scipy.spatial.transform import Rotation as R

from example_policies.data_ops.utils.rotation_6d import (
    compose_transform_6d_torch,
    quat_to_6d_torch,
    rotation_6d_to_quat_torch,
)
from example_policies.utils.action_order import (
    DUAL_ABS_LEFT_POS_IDXS,
    DUAL_ABS_LEFT_QUAT_IDXS,
    DUAL_ABS_RIGHT_POS_IDXS,
    DUAL_ABS_RIGHT_QUAT_IDXS,
    UMI_LEFT_GRIPPER_IDX,
    UMI_LEFT_POS_IDXS,
    UMI_LEFT_ROT6D_IDXS,
    UMI_RIGHT_GRIPPER_IDX,
    UMI_RIGHT_POS_IDXS,
    UMI_RIGHT_ROT6D_IDXS,
)
from example_policies.utils.chunk_relative_processor import (
    ABS_TCP_ACTION_DIM,
    ABS_TCP_LEFT_GRIPPER_IDX,
    ABS_TCP_RIGHT_GRIPPER_IDX,
    AbsTcpToChunkRelativeStep,
    abs_tcp_to_chunk_relative_umi_delta,
)
from example_policies.utils.constants import OBSERVATION_STATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Observation.state layout: [L_pos(3), L_quat(4), R_pos(3), R_quat(4)] = 14
OBS_TCP_LEFT_POS_INDICES = [0, 1, 2]
OBS_TCP_LEFT_QUAT_INDICES = [3, 4, 5, 6]
OBS_TCP_RIGHT_POS_INDICES = [7, 8, 9]
OBS_TCP_RIGHT_QUAT_INDICES = [10, 11, 12, 13]


def _make_abs_tcp_action(
    left_pos=(0.5, 0.0, 0.3),
    left_quat=(0.0, 0.0, 0.0, 1.0),
    right_pos=(-0.5, 0.0, 0.3),
    right_quat=(0.0, 0.0, 0.0, 1.0),
    gripper_l=0.5,
    gripper_r=0.5,
) -> torch.Tensor:
    """Create a 16-dim absolute TCP action."""
    action = torch.zeros(ABS_TCP_ACTION_DIM)
    action[DUAL_ABS_LEFT_POS_IDXS] = torch.tensor(left_pos)
    action[DUAL_ABS_LEFT_QUAT_IDXS] = torch.tensor(left_quat)
    action[DUAL_ABS_RIGHT_POS_IDXS] = torch.tensor(right_pos)
    action[DUAL_ABS_RIGHT_QUAT_IDXS] = torch.tensor(right_quat)
    action[ABS_TCP_LEFT_GRIPPER_IDX] = gripper_l
    action[ABS_TCP_RIGHT_GRIPPER_IDX] = gripper_r
    return action


def _make_obs_state(
    left_pos=(0.5, 0.0, 0.3),
    left_quat=(0.0, 0.0, 0.0, 1.0),
    right_pos=(-0.5, 0.0, 0.3),
    right_quat=(0.0, 0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Create a 14-dim observation state with TCP poses."""
    state = torch.zeros(14)
    state[OBS_TCP_LEFT_POS_INDICES] = torch.tensor(left_pos)
    state[OBS_TCP_LEFT_QUAT_INDICES] = torch.tensor(left_quat)
    state[OBS_TCP_RIGHT_POS_INDICES] = torch.tensor(right_pos)
    state[OBS_TCP_RIGHT_QUAT_INDICES] = torch.tensor(right_quat)
    return state


def _identity_rot6d() -> torch.Tensor:
    """Identity rotation as 6D: first two columns of I₃."""
    return torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _euler_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> tuple:
    """Convert Euler angles (rad) to quaternion in xyzw convention."""
    r = R.from_euler("xyz", [roll, pitch, yaw])
    q = r.as_quat()  # xyzw
    return tuple(q.tolist())


# ===========================================================================
# 1. Pure function tests: abs_tcp_to_chunk_relative_umi_delta
# ===========================================================================


class TestAbsTcpToChunkRelativeFunction:
    """Tests for the pure conversion function."""

    def test_identity_produces_zero_delta(self):
        """When target == reference, delta should be zero pos + identity rot."""
        ref_pos = torch.tensor([0.5, 0.0, 0.3])
        ref_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
        ref_rot6d = quat_to_6d_torch(ref_quat)

        # Action at the same pose as the reference
        action = _make_abs_tcp_action(
            left_pos=(0.5, 0.0, 0.3), left_quat=(0.0, 0.0, 0.0, 1.0),
            right_pos=(-0.5, 0.0, 0.3), right_quat=(0.0, 0.0, 0.0, 1.0),
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 16)

        umi = abs_tcp_to_chunk_relative_umi_delta(
            action,
            ref_pos.unsqueeze(0), ref_rot6d.unsqueeze(0),
            torch.tensor([-0.5, 0.0, 0.3]).unsqueeze(0),
            quat_to_6d_torch(torch.tensor([0.0, 0.0, 0.0, 1.0])).unsqueeze(0),
        )  # (1, 1, 20)

        assert umi.shape == (1, 1, 20)
        # Position deltas should be zero
        torch.testing.assert_close(
            umi[0, 0, UMI_LEFT_POS_IDXS], torch.zeros(3), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            umi[0, 0, UMI_RIGHT_POS_IDXS], torch.zeros(3), atol=1e-6, rtol=1e-6
        )
        # Rotation deltas should be identity
        torch.testing.assert_close(
            umi[0, 0, UMI_LEFT_ROT6D_IDXS], _identity_rot6d(), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            umi[0, 0, UMI_RIGHT_ROT6D_IDXS], _identity_rot6d(), atol=1e-6, rtol=1e-6
        )

    def test_known_translation_delta(self):
        """A pure translation should produce matching position delta, identity rotation."""
        ref_pos_l = torch.tensor([0.5, 0.0, 0.3])
        ref_quat_l = torch.tensor([0.0, 0.0, 0.0, 1.0])
        ref_rot6d_l = quat_to_6d_torch(ref_quat_l)

        # Target is 0.1m forward in x
        action = _make_abs_tcp_action(
            left_pos=(0.6, 0.0, 0.3), left_quat=(0.0, 0.0, 0.0, 1.0),
            right_pos=(-0.5, 0.0, 0.3), right_quat=(0.0, 0.0, 0.0, 1.0),
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 16)

        umi = abs_tcp_to_chunk_relative_umi_delta(
            action,
            ref_pos_l.unsqueeze(0), ref_rot6d_l.unsqueeze(0),
            torch.tensor([-0.5, 0.0, 0.3]).unsqueeze(0),
            quat_to_6d_torch(torch.tensor([0.0, 0.0, 0.0, 1.0])).unsqueeze(0),
        )

        # Left arm should have dx=0.1, dy=0, dz=0
        torch.testing.assert_close(
            umi[0, 0, UMI_LEFT_POS_IDXS],
            torch.tensor([0.1, 0.0, 0.0]),
            atol=1e-5, rtol=1e-5
        )
        # Left rotation should be identity
        torch.testing.assert_close(
            umi[0, 0, UMI_LEFT_ROT6D_IDXS], _identity_rot6d(), atol=1e-5, rtol=1e-5
        )

    def test_known_rotation_delta(self):
        """A pure rotation produces zero pos delta and matching rot 6D delta."""
        ref_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])  # identity
        ref_rot6d = quat_to_6d_torch(ref_quat)

        # Target: 90° around Z axis
        tgt_quat = _euler_to_quat_xyzw(0, 0, math.pi / 2)
        action = _make_abs_tcp_action(
            left_pos=(0.5, 0.0, 0.3), left_quat=tgt_quat,
            right_pos=(-0.5, 0.0, 0.3), right_quat=(0.0, 0.0, 0.0, 1.0),
        ).unsqueeze(0).unsqueeze(0)

        umi = abs_tcp_to_chunk_relative_umi_delta(
            action,
            torch.tensor([0.5, 0.0, 0.3]).unsqueeze(0),
            ref_rot6d.unsqueeze(0),
            torch.tensor([-0.5, 0.0, 0.3]).unsqueeze(0),
            ref_rot6d.unsqueeze(0),
        )

        # Position delta should be zero (same position)
        torch.testing.assert_close(
            umi[0, 0, UMI_LEFT_POS_IDXS], torch.zeros(3), atol=1e-5, rtol=1e-5
        )
        # The rotation delta should be non-trivial
        rot_delta = umi[0, 0, UMI_LEFT_ROT6D_IDXS]
        assert not torch.allclose(rot_delta, _identity_rot6d(), atol=1e-3)

    def test_grippers_pass_through(self):
        """Gripper values should be copied unchanged."""
        action = _make_abs_tcp_action(gripper_l=0.7, gripper_r=0.2)
        action = action.unsqueeze(0).unsqueeze(0)  # (1, 1, 16)

        ref_rot6d = quat_to_6d_torch(torch.tensor([0.0, 0.0, 0.0, 1.0]))
        umi = abs_tcp_to_chunk_relative_umi_delta(
            action,
            torch.tensor([0.5, 0.0, 0.3]).unsqueeze(0),
            ref_rot6d.unsqueeze(0),
            torch.tensor([-0.5, 0.0, 0.3]).unsqueeze(0),
            ref_rot6d.unsqueeze(0),
        )

        assert umi[0, 0, UMI_LEFT_GRIPPER_IDX].item() == pytest.approx(0.7)
        assert umi[0, 0, UMI_RIGHT_GRIPPER_IDX].item() == pytest.approx(0.2)

    def test_batch_and_horizon(self):
        """Should handle (B, H, 16) batched input."""
        B, H = 4, 8
        actions = torch.randn(B, H, ABS_TCP_ACTION_DIM)
        # Ensure quaternions are valid
        actions[..., DUAL_ABS_LEFT_QUAT_IDXS] = torch.nn.functional.normalize(
            actions[..., DUAL_ABS_LEFT_QUAT_IDXS], dim=-1
        )
        actions[..., DUAL_ABS_RIGHT_QUAT_IDXS] = torch.nn.functional.normalize(
            actions[..., DUAL_ABS_RIGHT_QUAT_IDXS], dim=-1
        )

        ref_rot6d = quat_to_6d_torch(torch.tensor([0.0, 0.0, 0.0, 1.0])).expand(B, 6)
        ref_pos = torch.zeros(B, 3)

        umi = abs_tcp_to_chunk_relative_umi_delta(
            actions, ref_pos, ref_rot6d, ref_pos, ref_rot6d
        )

        assert umi.shape == (B, H, 20)


# ===========================================================================
# 2. Round-trip test: abs TCP → chunk-relative → compose back → abs TCP
# ===========================================================================


class TestChunkRelativeRoundTrip:
    """Verify that converting to chunk-relative and composing back recovers the original."""

    def test_roundtrip_identity_ref(self):
        """Round-trip with identity reference pose."""
        ref_pos = torch.tensor([0.5, 0.0, 0.3])
        ref_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
        ref_rot6d = quat_to_6d_torch(ref_quat)

        tgt_pos = torch.tensor([0.6, 0.1, 0.35])
        tgt_quat = torch.tensor(list(_euler_to_quat_xyzw(0.1, 0.2, 0.3)))

        action = _make_abs_tcp_action(
            left_pos=tuple(tgt_pos.tolist()),
            left_quat=tuple(tgt_quat.tolist()),
            right_pos=(-0.5, 0.0, 0.3),
            right_quat=(0.0, 0.0, 0.0, 1.0),
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 16)

        umi = abs_tcp_to_chunk_relative_umi_delta(
            action,
            ref_pos.unsqueeze(0), ref_rot6d.unsqueeze(0),
            torch.tensor([-0.5, 0.0, 0.3]).unsqueeze(0),
            ref_rot6d.unsqueeze(0),
        )

        # Compose back: abs = ref + delta
        delta_pos_l = umi[0, 0, UMI_LEFT_POS_IDXS]
        delta_rot6d_l = umi[0, 0, UMI_LEFT_ROT6D_IDXS]
        recovered_pos, recovered_rot6d = compose_transform_6d_torch(
            ref_pos, ref_rot6d, delta_pos_l, delta_rot6d_l
        )
        recovered_quat = rotation_6d_to_quat_torch(recovered_rot6d)

        torch.testing.assert_close(recovered_pos, tgt_pos, atol=1e-5, rtol=1e-5)
        # Quaternion sign ambiguity
        if torch.dot(recovered_quat, tgt_quat) < 0:
            recovered_quat = -recovered_quat
        torch.testing.assert_close(recovered_quat, tgt_quat, atol=1e-5, rtol=1e-5)

    def test_roundtrip_non_identity_ref(self):
        """Round-trip with a rotated reference pose."""
        ref_quat = torch.tensor(list(_euler_to_quat_xyzw(0.3, -0.2, 0.5)))
        ref_pos = torch.tensor([0.4, 0.1, 0.5])
        ref_rot6d = quat_to_6d_torch(ref_quat)

        tgt_quat = torch.tensor(list(_euler_to_quat_xyzw(0.6, 0.1, -0.3)))
        tgt_pos = torch.tensor([0.7, -0.2, 0.4])

        action = _make_abs_tcp_action(
            left_pos=tuple(tgt_pos.tolist()),
            left_quat=tuple(tgt_quat.tolist()),
            right_pos=(-0.4, 0.1, 0.5),
            right_quat=tuple(ref_quat.tolist()),
        ).unsqueeze(0).unsqueeze(0)

        umi = abs_tcp_to_chunk_relative_umi_delta(
            action,
            ref_pos.unsqueeze(0), ref_rot6d.unsqueeze(0),
            torch.tensor([-0.4, 0.1, 0.5]).unsqueeze(0),
            ref_rot6d.unsqueeze(0),
        )

        delta_pos_l = umi[0, 0, UMI_LEFT_POS_IDXS]
        delta_rot6d_l = umi[0, 0, UMI_LEFT_ROT6D_IDXS]
        recovered_pos, recovered_rot6d = compose_transform_6d_torch(
            ref_pos, ref_rot6d, delta_pos_l, delta_rot6d_l
        )
        recovered_quat = rotation_6d_to_quat_torch(recovered_rot6d)

        torch.testing.assert_close(recovered_pos, tgt_pos, atol=1e-5, rtol=1e-5)
        if torch.dot(recovered_quat, tgt_quat) < 0:
            recovered_quat = -recovered_quat
        torch.testing.assert_close(recovered_quat, tgt_quat, atol=1e-4, rtol=1e-4)

    def test_roundtrip_batched_horizon(self):
        """Round-trip for a full batch with multiple horizon steps."""
        B, H = 2, 4
        torch.manual_seed(42)

        # Random reference poses
        ref_pos_l = torch.randn(B, 3)
        ref_quat_l = torch.nn.functional.normalize(torch.randn(B, 4), dim=-1)
        ref_rot6d_l = quat_to_6d_torch(ref_quat_l)
        ref_pos_r = torch.randn(B, 3)
        ref_quat_r = torch.nn.functional.normalize(torch.randn(B, 4), dim=-1)
        ref_rot6d_r = quat_to_6d_torch(ref_quat_r)

        # Random absolute actions
        actions = torch.randn(B, H, ABS_TCP_ACTION_DIM)
        actions[..., DUAL_ABS_LEFT_QUAT_IDXS] = torch.nn.functional.normalize(
            actions[..., DUAL_ABS_LEFT_QUAT_IDXS], dim=-1
        )
        actions[..., DUAL_ABS_RIGHT_QUAT_IDXS] = torch.nn.functional.normalize(
            actions[..., DUAL_ABS_RIGHT_QUAT_IDXS], dim=-1
        )

        # Forward: abs → chunk-relative
        umi = abs_tcp_to_chunk_relative_umi_delta(
            actions, ref_pos_l, ref_rot6d_l, ref_pos_r, ref_rot6d_r
        )
        assert umi.shape == (B, H, 20)

        # Inverse: compose back
        for b in range(B):
            for h in range(H):
                # Left arm
                rec_pos_l, rec_rot6d_l = compose_transform_6d_torch(
                    ref_pos_l[b], ref_rot6d_l[b],
                    umi[b, h, UMI_LEFT_POS_IDXS],
                    umi[b, h, UMI_LEFT_ROT6D_IDXS],
                )
                torch.testing.assert_close(
                    rec_pos_l, actions[b, h, DUAL_ABS_LEFT_POS_IDXS],
                    atol=1e-4, rtol=1e-4,
                )

                rec_quat_l = rotation_6d_to_quat_torch(rec_rot6d_l)
                orig_quat_l = actions[b, h, DUAL_ABS_LEFT_QUAT_IDXS]
                if torch.dot(rec_quat_l, orig_quat_l) < 0:
                    rec_quat_l = -rec_quat_l
                torch.testing.assert_close(rec_quat_l, orig_quat_l, atol=1e-4, rtol=1e-4)

                # Grippers
                assert umi[b, h, UMI_LEFT_GRIPPER_IDX].item() == pytest.approx(
                    actions[b, h, ABS_TCP_LEFT_GRIPPER_IDX].item(), abs=1e-6
                )
                assert umi[b, h, UMI_RIGHT_GRIPPER_IDX].item() == pytest.approx(
                    actions[b, h, ABS_TCP_RIGHT_GRIPPER_IDX].item(), abs=1e-6
                )


# ===========================================================================
# 3. ProcessorStep tests
# ===========================================================================


class TestAbsTcpToChunkRelativeStep:
    """Tests for the ProcessorStep wrapper."""

    def _make_step(self):
        return AbsTcpToChunkRelativeStep(
            obs_tcp_left_pos_indices=OBS_TCP_LEFT_POS_INDICES,
            obs_tcp_left_quat_indices=OBS_TCP_LEFT_QUAT_INDICES,
            obs_tcp_right_pos_indices=OBS_TCP_RIGHT_POS_INDICES,
            obs_tcp_right_quat_indices=OBS_TCP_RIGHT_QUAT_INDICES,
        )

    def test_transforms_action_shape(self):
        """Action should go from (B, H, 16) → (B, H, 20)."""
        from lerobot.processor.core import TransitionKey

        step = self._make_step()
        B, H = 2, 4
        actions = torch.randn(B, H, ABS_TCP_ACTION_DIM)
        actions[..., DUAL_ABS_LEFT_QUAT_IDXS] = torch.nn.functional.normalize(
            actions[..., DUAL_ABS_LEFT_QUAT_IDXS], dim=-1
        )
        actions[..., DUAL_ABS_RIGHT_QUAT_IDXS] = torch.nn.functional.normalize(
            actions[..., DUAL_ABS_RIGHT_QUAT_IDXS], dim=-1
        )

        obs_state = torch.zeros(B, 2, 14)  # (B, n_obs_steps, 14)
        obs_state[:, -1, OBS_TCP_LEFT_POS_INDICES] = torch.tensor([0.5, 0.0, 0.3])
        obs_state[:, -1, OBS_TCP_LEFT_QUAT_INDICES] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        obs_state[:, -1, OBS_TCP_RIGHT_POS_INDICES] = torch.tensor([-0.5, 0.0, 0.3])
        obs_state[:, -1, OBS_TCP_RIGHT_QUAT_INDICES] = torch.tensor([0.0, 0.0, 0.0, 1.0])

        transition = {
            TransitionKey.OBSERVATION: {OBSERVATION_STATE: obs_state},
            TransitionKey.ACTION: actions,
        }

        result = step(transition)
        assert result[TransitionKey.ACTION].shape == (B, H, 20)

    def test_noop_without_action(self):
        """During inference pre-processing, no action key → no-op."""
        from lerobot.processor.core import TransitionKey

        step = self._make_step()
        transition = {
            TransitionKey.OBSERVATION: {OBSERVATION_STATE: torch.randn(1, 14)},
        }
        result = step(transition)
        assert TransitionKey.ACTION not in result

    def test_noop_non_abs_tcp_action(self):
        """If action is already 20-dim (UMI delta), should pass through."""
        from lerobot.processor.core import TransitionKey

        step = self._make_step()
        action = torch.randn(1, 4, 20)  # already UMI delta
        transition = {
            TransitionKey.OBSERVATION: {OBSERVATION_STATE: torch.randn(1, 2, 14)},
            TransitionKey.ACTION: action,
        }
        result = step(transition)
        # Should be unchanged (20-dim doesn't match 16-dim check)
        torch.testing.assert_close(result[TransitionKey.ACTION], action)

    def test_transform_features_updates_shape(self):
        """transform_features should change action shape from [16] to [20]."""
        from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature

        step = self._make_step()
        features = {
            PipelineFeatureType.ACTION: {
                "action": PolicyFeature(type=FeatureType.ACTION, shape=[16]),
            },
            PipelineFeatureType.OBSERVATION: {
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=[14]),
            },
        }
        result = step.transform_features(features)
        assert result[PipelineFeatureType.ACTION]["action"].shape == [20]


# ===========================================================================
# 4. "Wider spread for future actions" test (arXiv:2507.05331 §4.4.2)
# ===========================================================================


class TestWiderSpreadProperty:
    """Verify that chunk-relative deltas have monotonically increasing spread
    for actions further into the future, matching TRI's observation."""

    def test_future_actions_have_wider_spread(self):
        """Position delta variance should increase with horizon step index.

        We simulate a simple random-walk trajectory (realistic for robot arms)
        and verify that chunk-relative deltas have increasing variance for
        later timesteps.
        """
        torch.manual_seed(123)
        N_episodes = 50
        ep_length = 100
        H = 16

        # Simulate random-walk trajectories
        all_chunks = []
        for _ in range(N_episodes):
            # Start from a random pose
            pos_l = torch.tensor([0.5, 0.0, 0.3])
            pos_r = torch.tensor([-0.5, 0.0, 0.3])
            quat_identity = torch.tensor([0.0, 0.0, 0.0, 1.0])

            positions = []
            for t in range(ep_length):
                # Small random walk
                pos_l = pos_l + torch.randn(3) * 0.005
                pos_r = pos_r + torch.randn(3) * 0.005
                action = _make_abs_tcp_action(
                    left_pos=tuple(pos_l.tolist()),
                    left_quat=tuple(quat_identity.tolist()),
                    right_pos=tuple(pos_r.tolist()),
                    right_quat=tuple(quat_identity.tolist()),
                )
                positions.append(action)

            # Build chunks
            ref_rot6d = quat_to_6d_torch(quat_identity)
            for i in range(len(positions) - H + 1):
                chunk = torch.stack(positions[i:i + H]).unsqueeze(0)  # (1, H, 16)
                ref_pos_l = positions[i][DUAL_ABS_LEFT_POS_IDXS].unsqueeze(0)
                ref_pos_r = positions[i][DUAL_ABS_RIGHT_POS_IDXS].unsqueeze(0)

                umi = abs_tcp_to_chunk_relative_umi_delta(
                    chunk,
                    ref_pos_l, ref_rot6d.unsqueeze(0),
                    ref_pos_r, ref_rot6d.unsqueeze(0),
                )
                all_chunks.append(umi.squeeze(0))  # (H, 20)

        all_chunks_t = torch.stack(all_chunks)  # (N, H, 20)

        # Compute per-timestep variance for left arm position delta
        left_pos_deltas = all_chunks_t[:, :, UMI_LEFT_POS_IDXS]  # (N, H, 3)
        per_step_var = left_pos_deltas.var(dim=0).sum(dim=-1)  # (H,)

        # Variance should be monotonically increasing (with some tolerance for noise)
        for k in range(1, H):
            assert per_step_var[k] > per_step_var[k - 1] * 0.95, (
                f"Variance at step {k} ({per_step_var[k]:.6f}) should be greater "
                f"than step {k-1} ({per_step_var[k-1]:.6f})"
            )

        # First step should have minimal variance (identity or near-zero delta)
        assert per_step_var[0] < per_step_var[-1] * 0.1, (
            "First step variance should be much smaller than last step variance"
        )
