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

"""Tests for UMI-delta action space and stepwise percentile normalizer.

Covers:
    1. 6D rotation roundtrips (numpy + torch)
    2. Stepwise percentile normalizer
    3. Action mode parsing and index correctness
"""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from example_policies.data_ops.utils.rotation_6d import (
    compute_relative_pose_6d_np,
    compute_relative_transform_6d_torch,
    compose_transform_6d_torch,
    quat_to_6d_np,
    quat_to_6d_torch,
    rotation_6d_to_matrix_np,
    rotation_6d_to_matrix_torch,
    rotation_6d_to_quat_np,
    rotation_6d_to_quat_torch,
    rotation_matrix_to_6d_np,
    rotation_matrix_to_6d_torch,
)
from example_policies.utils.stepwise_normalize import (
    StepwisePercentileNormalize,
    StepwisePercentileUnnormalize,
    compute_stepwise_percentile_stats,
    make_stepwise_normalizer_pair,
)
from example_policies.utils.action_order import (
    ActionMode,
    UMI_ACTION_DIM,
    UMI_LEFT_GRIPPER_IDX,
    UMI_LEFT_POS_IDXS,
    UMI_LEFT_ROT6D_IDXS,
    UMI_RIGHT_GRIPPER_IDX,
    UMI_RIGHT_POS_IDXS,
    UMI_RIGHT_ROT6D_IDXS,
    UMI_ROTATION_FEATURE_INDICES,
    GET_LEFT_GRIPPER_IDX,
    GET_RIGHT_GRIPPER_IDX,
)

ATOL = 1e-5


# =============================================================================
# 1. 6D Rotation Tests
# =============================================================================


class TestRotation6DNumpy:
    """Test 6D rotation utilities (numpy / data pipeline)."""

    def test_matrix_to_6d_roundtrip_identity(self):
        """Identity matrix → 6D → matrix should recover identity."""
        eye = np.eye(3)
        r6d = rotation_matrix_to_6d_np(eye)
        assert r6d.shape == (6,)
        recovered = rotation_6d_to_matrix_np(r6d)
        np.testing.assert_allclose(recovered, eye, atol=ATOL)

    def test_matrix_to_6d_roundtrip_random(self):
        """Random rotation matrices should roundtrip through 6D."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            rot = R.random(random_state=rng).as_matrix()
            r6d = rotation_matrix_to_6d_np(rot)
            recovered = rotation_6d_to_matrix_np(r6d)
            np.testing.assert_allclose(recovered, rot, atol=ATOL)

    def test_quat_to_6d_roundtrip(self):
        """Quaternion → 6D → quaternion should recover the same rotation."""
        rng = np.random.default_rng(123)
        for _ in range(20):
            q = R.random(random_state=rng).as_quat()  # xyzw
            r6d = quat_to_6d_np(q)
            assert r6d.shape == (6,)
            q_rec = rotation_6d_to_quat_np(r6d)
            # Quaternions are equivalent up to sign
            dot = np.abs(np.dot(q, q_rec))
            np.testing.assert_allclose(dot, 1.0, atol=ATOL)

    def test_6d_values_bounded(self):
        """6D representation values should be in [-1, 1] for valid rotations."""
        rng = np.random.default_rng(7)
        for _ in range(50):
            rot = R.random(random_state=rng).as_matrix()
            r6d = rotation_matrix_to_6d_np(rot)
            assert np.all(r6d >= -1.0 - 1e-7)
            assert np.all(r6d <= 1.0 + 1e-7)

    def test_batch_quat_to_6d(self):
        """Batch quaternion conversion should work."""
        rng = np.random.default_rng(99)
        quats = R.random(10, random_state=rng).as_quat()  # (10, 4)
        r6d = quat_to_6d_np(quats)
        assert r6d.shape == (10, 6)
        quats_rec = rotation_6d_to_quat_np(r6d)
        dots = np.abs(np.sum(quats * quats_rec, axis=-1))
        np.testing.assert_allclose(dots, np.ones(10), atol=ATOL)

    def test_relative_pose_identity_ref(self):
        """Relative pose with identity reference should equal the target pose."""
        ref_pos = np.zeros(3)
        ref_quat = np.array([0.0, 0.0, 0.0, 1.0])  # identity
        target_pos = np.array([1.0, 2.0, 3.0])
        target_quat = R.from_euler("xyz", [30, 45, 60], degrees=True).as_quat()

        result = compute_relative_pose_6d_np(ref_pos, ref_quat, target_pos, target_quat)
        assert result.shape == (9,)

        # Position delta should equal target position
        np.testing.assert_allclose(result[:3], target_pos, atol=ATOL)

        # Rotation delta should equal target rotation (since ref is identity)
        expected_6d = quat_to_6d_np(target_quat)
        np.testing.assert_allclose(result[3:], expected_6d, atol=ATOL)

    def test_relative_pose_self_reference(self):
        """Relative pose to self should be zero delta position and identity rotation."""
        rng = np.random.default_rng(42)
        pos = rng.standard_normal(3)
        quat = R.random(random_state=rng).as_quat()

        result = compute_relative_pose_6d_np(pos, quat, pos, quat)

        # Position delta should be zero
        np.testing.assert_allclose(result[:3], np.zeros(3), atol=ATOL)

        # Rotation delta should be identity (6D of identity matrix)
        identity_6d = rotation_matrix_to_6d_np(np.eye(3))
        np.testing.assert_allclose(result[3:], identity_6d, atol=ATOL)


class TestRotation6DTorch:
    """Test 6D rotation utilities (torch / training)."""

    def test_matrix_to_6d_roundtrip_identity(self):
        eye = torch.eye(3)
        r6d = rotation_matrix_to_6d_torch(eye)
        assert r6d.shape == (6,)
        recovered = rotation_6d_to_matrix_torch(r6d)
        torch.testing.assert_close(recovered, eye, atol=ATOL, rtol=1e-5)

    def test_matrix_to_6d_roundtrip_random(self):
        rng = np.random.default_rng(55)
        for _ in range(20):
            rot_np = R.random(random_state=rng).as_matrix()
            rot = torch.from_numpy(rot_np).float()
            r6d = rotation_matrix_to_6d_torch(rot)
            recovered = rotation_6d_to_matrix_torch(r6d)
            torch.testing.assert_close(recovered, rot, atol=ATOL, rtol=1e-5)

    def test_quat_to_6d_roundtrip(self):
        rng = np.random.default_rng(77)
        for _ in range(20):
            q_np = R.random(random_state=rng).as_quat()
            q = torch.from_numpy(q_np).float()
            r6d = quat_to_6d_torch(q)
            assert r6d.shape == (6,)
            q_rec = rotation_6d_to_quat_torch(r6d)
            dot = torch.abs(torch.dot(q, q_rec))
            torch.testing.assert_close(dot, torch.tensor(1.0), atol=ATOL, rtol=1e-5)

    def test_batch_quat_to_6d(self):
        rng = np.random.default_rng(33)
        quats_np = R.random(10, random_state=rng).as_quat()
        quats = torch.from_numpy(quats_np).float()
        r6d = quat_to_6d_torch(quats)
        assert r6d.shape == (10, 6)
        quats_rec = rotation_6d_to_quat_torch(r6d)
        dots = torch.abs(torch.sum(quats * quats_rec, dim=-1))
        torch.testing.assert_close(dots, torch.ones(10), atol=ATOL, rtol=1e-5)

    def test_numpy_torch_consistency(self):
        """NumPy and Torch implementations should produce the same results."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            rot_np = R.random(random_state=rng).as_matrix()
            rot_torch = torch.from_numpy(rot_np).float()

            r6d_np = rotation_matrix_to_6d_np(rot_np)
            r6d_torch = rotation_matrix_to_6d_torch(rot_torch).numpy()
            np.testing.assert_allclose(r6d_np, r6d_torch, atol=ATOL)

    def test_relative_transform_roundtrip(self):
        """compute + compose should roundtrip to the original absolute pose."""
        rng = np.random.default_rng(42)
        ref_pos = torch.randn(3)
        ref_rot6d = quat_to_6d_torch(torch.from_numpy(R.random(random_state=rng).as_quat()).float())
        target_pos = torch.randn(3)
        target_rot6d = quat_to_6d_torch(torch.from_numpy(R.random(random_state=rng).as_quat()).float())

        delta_pos, delta_rot6d = compute_relative_transform_6d_torch(
            ref_pos, ref_rot6d, target_pos, target_rot6d
        )
        rec_pos, rec_rot6d = compose_transform_6d_torch(
            ref_pos, ref_rot6d, delta_pos, delta_rot6d
        )

        torch.testing.assert_close(rec_pos, target_pos, atol=ATOL, rtol=1e-5)
        torch.testing.assert_close(rec_rot6d, target_rot6d, atol=ATOL, rtol=1e-5)

    def test_relative_transform_batch(self):
        """Batched relative transform should work."""
        B = 5
        rng = np.random.default_rng(11)
        ref_pos = torch.randn(B, 3)
        ref_rot6d = quat_to_6d_torch(
            torch.from_numpy(R.random(B, random_state=rng).as_quat()).float()
        )
        target_pos = torch.randn(B, 3)
        target_rot6d = quat_to_6d_torch(
            torch.from_numpy(R.random(B, random_state=rng).as_quat()).float()
        )

        delta_pos, delta_rot6d = compute_relative_transform_6d_torch(
            ref_pos, ref_rot6d, target_pos, target_rot6d
        )
        rec_pos, rec_rot6d = compose_transform_6d_torch(
            ref_pos, ref_rot6d, delta_pos, delta_rot6d
        )

        torch.testing.assert_close(rec_pos, target_pos, atol=ATOL, rtol=1e-5)
        torch.testing.assert_close(rec_rot6d, target_rot6d, atol=1e-4, rtol=1e-4)


# =============================================================================
# 2. Stepwise Percentile Normalizer Tests
# =============================================================================


class TestStepwisePercentileNormalize:
    """Test the stepwise percentile normalizer forward pass."""

    def _make_stats(self, horizon=5, action_dim=20) -> tuple[torch.Tensor, torch.Tensor]:
        """Create simple p02/p98 stats for testing."""
        p02 = torch.zeros(horizon, action_dim)
        p98 = torch.ones(horizon, action_dim) * 10.0
        return p02, p98

    def test_midpoint_maps_to_zero(self):
        """Midpoint of [p02, p98] should normalise to 0."""
        p02, p98 = self._make_stats(horizon=4, action_dim=6)
        norm = StepwisePercentileNormalize(p02, p98)

        x = (p02 + p98) / 2  # midpoint → (4, 6)
        y = norm(x)

        torch.testing.assert_close(y, torch.zeros_like(y), atol=1e-6, rtol=1e-6)

    def test_p02_maps_to_neg_one(self):
        """x = p02 should map to -1."""
        p02, p98 = self._make_stats(horizon=4, action_dim=6)
        norm = StepwisePercentileNormalize(p02, p98)

        y = norm(p02.clone())
        torch.testing.assert_close(y, torch.full_like(y, -1.0), atol=1e-5, rtol=1e-5)

    def test_p98_maps_to_pos_one(self):
        """x = p98 should map to +1."""
        p02, p98 = self._make_stats(horizon=4, action_dim=6)
        norm = StepwisePercentileNormalize(p02, p98)

        y = norm(p98.clone())
        torch.testing.assert_close(y, torch.full_like(y, 1.0), atol=1e-5, rtol=1e-5)

    def test_clipping(self):
        """Values far outside [p02, p98] should be clamped to [-1.5, 1.5]."""
        p02, p98 = self._make_stats(horizon=4, action_dim=6)
        norm = StepwisePercentileNormalize(p02, p98)

        x = torch.full((4, 6), 100.0)  # way above p98
        y = norm(x)
        assert y.max() <= 1.5 + 1e-7

        x = torch.full((4, 6), -100.0)  # way below p02
        y = norm(x)
        assert y.min() >= -1.5 - 1e-7

    def test_custom_clip_range(self):
        """Custom clip_min/clip_max should be respected."""
        p02, p98 = self._make_stats(horizon=4, action_dim=6)
        norm = StepwisePercentileNormalize(p02, p98, clip_min=-2.0, clip_max=2.0)

        x = torch.full((4, 6), 100.0)
        y = norm(x)
        assert y.max() <= 2.0 + 1e-7

    def test_skip_rotation_features(self):
        """Features listed in skip_feature_indices should pass through unchanged."""
        horizon, dim = 4, 20
        p02 = torch.zeros(horizon, dim)
        p98 = torch.ones(horizon, dim) * 10.0
        skip = UMI_ROTATION_FEATURE_INDICES  # indices 3-8 and 12-17

        norm = StepwisePercentileNormalize(p02, p98, skip_feature_indices=skip)

        x = torch.full((horizon, dim), 5.0)
        y = norm(x)

        # Rotation features should be unchanged
        for idx in skip:
            torch.testing.assert_close(y[:, idx], x[:, idx])

        # Non-rotation features should be different (normalised)
        non_skip = [i for i in range(dim) if i not in skip]
        for idx in non_skip:
            assert not torch.allclose(y[:, idx], x[:, idx]), f"Feature {idx} was not normalised"

    def test_batch_dimension(self):
        """Should handle (B, H, D) input correctly."""
        p02, p98 = self._make_stats(horizon=4, action_dim=6)
        norm = StepwisePercentileNormalize(p02, p98)

        B = 8
        x = torch.rand(B, 4, 6) * 10.0
        y = norm(x)
        assert y.shape == (B, 4, 6)

    def test_2d_input_preserves_shape(self):
        """(H, D) input (no batch dim) should return (H, D)."""
        p02, p98 = self._make_stats(horizon=4, action_dim=6)
        norm = StepwisePercentileNormalize(p02, p98)

        x = torch.rand(4, 6) * 10.0
        y = norm(x)
        assert y.shape == (4, 6)

    def test_per_step_stats(self):
        """Different timestep indices should use different stats."""
        p02 = torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        p98 = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])

        norm = StepwisePercentileNormalize(p02, p98)

        # x = midpoint for each step
        x = torch.tensor([[5.0, 5.0], [15.0, 15.0], [25.0, 25.0]])
        y = norm(x)
        torch.testing.assert_close(y, torch.zeros(3, 2), atol=1e-6, rtol=1e-6)


class TestStepwisePercentileUnnormalize:
    """Test the stepwise percentile unnormalizer."""

    def test_forward_inverse_roundtrip(self):
        """Normalise → unnormalise should recover the original values."""
        horizon, dim = 5, 8
        p02 = torch.randn(horizon, dim)
        p98 = p02 + torch.rand(horizon, dim) * 5 + 1  # ensure p98 > p02

        normalizer = StepwisePercentileNormalize(p02, p98)
        unnormalizer = StepwisePercentileUnnormalize(p02, p98)

        x = p02 + torch.rand(2, horizon, dim) * (p98 - p02)  # values within [p02, p98]
        y = normalizer(x)
        x_rec = unnormalizer(y)

        torch.testing.assert_close(x_rec, x, atol=1e-4, rtol=1e-4)

    def test_roundtrip_with_skip_features(self):
        """Roundtrip with skip features should work correctly."""
        horizon, dim = 4, 20
        p02 = torch.randn(horizon, dim)
        p98 = p02 + torch.rand(horizon, dim) * 5 + 1
        skip = UMI_ROTATION_FEATURE_INDICES

        normalizer = StepwisePercentileNormalize(p02, p98, skip_feature_indices=skip)
        unnormalizer = StepwisePercentileUnnormalize(p02, p98, skip_feature_indices=skip)

        x = p02 + torch.rand(3, horizon, dim) * (p98 - p02)
        y = normalizer(x)
        x_rec = unnormalizer(y)

        torch.testing.assert_close(x_rec, x, atol=1e-4, rtol=1e-4)

    def test_2d_shape_preserved(self):
        """(H, D) input should return (H, D)."""
        horizon, dim = 4, 6
        p02 = torch.zeros(horizon, dim)
        p98 = torch.ones(horizon, dim) * 10.0
        unnorm = StepwisePercentileUnnormalize(p02, p98)

        y = torch.zeros(horizon, dim)  # normalised midpoint
        x = unnorm(y)
        assert x.shape == (horizon, dim)

    def test_zero_maps_to_midpoint(self):
        """y = 0 should map back to midpoint of [p02, p98]."""
        horizon, dim = 4, 6
        p02 = torch.zeros(horizon, dim)
        p98 = torch.ones(horizon, dim) * 10.0
        unnorm = StepwisePercentileUnnormalize(p02, p98)

        y = torch.zeros(horizon, dim)
        x = unnorm(y)
        expected = (p02 + p98) / 2
        torch.testing.assert_close(x, expected, atol=1e-5, rtol=1e-5)


class TestComputeStepwisePercentileStats:
    """Test the statistics computation utility."""

    def _make_fake_dataset(self, n_samples, horizon, action_dim, rng=None):
        """Create a list-like fake dataset returning action chunks."""
        if rng is None:
            rng = np.random.default_rng(42)

        data = []
        for _ in range(n_samples):
            action = torch.from_numpy(rng.standard_normal((horizon, action_dim))).float()
            data.append({"action": action})

        class FakeDataset:
            def __init__(self, items):
                self._items = items

            def __len__(self):
                return len(self._items)

            def __getitem__(self, idx):
                return self._items[idx]

        return FakeDataset(data)

    def test_output_shape(self):
        """Stats should have shape (horizon, action_dim)."""
        ds = self._make_fake_dataset(50, horizon=8, action_dim=20)
        stats = compute_stepwise_percentile_stats(ds, horizon=8)

        assert stats["p_low"].shape == (8, 20)
        assert stats["p_high"].shape == (8, 20)

    def test_p_low_less_than_p_high(self):
        """p02 should be less than p98 for non-degenerate data."""
        ds = self._make_fake_dataset(100, horizon=5, action_dim=10)
        stats = compute_stepwise_percentile_stats(ds, horizon=5)

        assert torch.all(stats["p_low"] < stats["p_high"])

    def test_uniform_data_expected_percentiles(self):
        """With uniform [0, 1] data, p02 ≈ 0.02, p98 ≈ 0.98."""
        n_samples = 5000
        horizon, dim = 4, 3
        rng = np.random.default_rng(123)

        items = []
        for _ in range(n_samples):
            action = torch.from_numpy(rng.uniform(0, 1, (horizon, dim))).float()
            items.append({"action": action})

        class DS:
            def __len__(self):
                return len(items)
            def __getitem__(self, idx):
                return items[idx]

        stats = compute_stepwise_percentile_stats(DS(), horizon=horizon)

        # With 5000 samples, percentiles should be fairly close
        torch.testing.assert_close(
            stats["p_low"], torch.full((horizon, dim), 0.02), atol=0.02, rtol=0.5
        )
        torch.testing.assert_close(
            stats["p_high"], torch.full((horizon, dim), 0.98), atol=0.02, rtol=0.5
        )

    def test_horizon_inferred(self):
        """Horizon should be inferred from first sample when not provided."""
        ds = self._make_fake_dataset(10, horizon=6, action_dim=4)
        stats = compute_stepwise_percentile_stats(ds)

        assert stats["p_low"].shape == (6, 4)


class TestMakeStepwiseNormalizerPair:
    """Test the convenience factory."""

    def test_from_stats(self):
        """Create pair from pre-computed stats."""
        p02 = torch.zeros(5, 10)
        p98 = torch.ones(5, 10) * 10.0
        stats = {"p_low": p02, "p_high": p98}

        normalizer, unnormalizer = make_stepwise_normalizer_pair(stats=stats)

        assert isinstance(normalizer, StepwisePercentileNormalize)
        assert isinstance(unnormalizer, StepwisePercentileUnnormalize)

    def test_from_dataset(self):
        """Create pair from dataset."""
        rng = np.random.default_rng(42)
        items = [
            {"action": torch.from_numpy(rng.standard_normal((5, 10))).float()}
            for _ in range(20)
        ]

        class DS:
            def __len__(self):
                return len(items)
            def __getitem__(self, idx):
                return items[idx]

        normalizer, unnormalizer = make_stepwise_normalizer_pair(dataset=DS())
        assert isinstance(normalizer, StepwisePercentileNormalize)

    def test_raises_without_dataset_or_stats(self):
        """Should raise ValueError if neither dataset nor stats provided."""
        with pytest.raises(ValueError, match="Provide either"):
            make_stepwise_normalizer_pair()

    def test_skip_indices_propagated(self):
        """Skip indices should be propagated to both modules."""
        p02 = torch.zeros(5, 20)
        p98 = torch.ones(5, 20) * 10.0
        stats = {"p_low": p02, "p_high": p98}
        skip = UMI_ROTATION_FEATURE_INDICES

        normalizer, unnormalizer = make_stepwise_normalizer_pair(
            stats=stats, skip_feature_indices=skip
        )

        # Verify mask is correct: rotation features → False, others → True
        for idx in skip:
            assert not normalizer.normalize_mask[idx].item()
            assert not unnormalizer.normalize_mask[idx].item()

        non_skip = [i for i in range(20) if i not in skip]
        for idx in non_skip:
            assert normalizer.normalize_mask[idx].item()
            assert unnormalizer.normalize_mask[idx].item()


# =============================================================================
# 3. Action Mode and Index Tests
# =============================================================================


class TestActionModeUMIDelta:
    """Test UMI_DELTA_TCP action mode parsing and indices."""

    def test_parse_umi_delta_tcp(self):
        """Should correctly parse action mode from feature names."""
        cfg = Mock()
        cfg.output_features = {"action": Mock()}
        cfg.output_features["action"].shape = [20]
        cfg.metadata = {
            "features": {
                "action": {
                    "names": [
                        "umi_delta_tcp_left_xyz_0",
                        "umi_delta_tcp_left_xyz_1",
                        "umi_delta_tcp_left_xyz_2",
                        "umi_delta_tcp_left_rot6d_0",
                        "umi_delta_tcp_left_rot6d_1",
                        "umi_delta_tcp_left_rot6d_2",
                        "umi_delta_tcp_left_rot6d_3",
                        "umi_delta_tcp_left_rot6d_4",
                        "umi_delta_tcp_left_rot6d_5",
                        "umi_delta_tcp_right_xyz_0",
                        "umi_delta_tcp_right_xyz_1",
                        "umi_delta_tcp_right_xyz_2",
                        "umi_delta_tcp_right_rot6d_0",
                        "umi_delta_tcp_right_rot6d_1",
                        "umi_delta_tcp_right_rot6d_2",
                        "umi_delta_tcp_right_rot6d_3",
                        "umi_delta_tcp_right_rot6d_4",
                        "umi_delta_tcp_right_rot6d_5",
                        "left_gripper",
                        "right_gripper",
                    ]
                }
            }
        }

        mode = ActionMode.parse_action_mode(cfg)
        assert mode == ActionMode.UMI_DELTA_TCP

    def test_umi_delta_not_confused_with_delta_tcp(self):
        """UMI delta should not be parsed as regular delta TCP."""
        cfg = Mock()
        cfg.output_features = {"action": Mock()}
        cfg.output_features["action"].shape = [20]
        cfg.metadata = {
            "features": {
                "action": {
                    "names": ["umi_delta_tcp_left_xyz_0"] + ["x"] * 19,
                }
            }
        }

        mode = ActionMode.parse_action_mode(cfg)
        assert mode == ActionMode.UMI_DELTA_TCP
        assert mode != ActionMode.DELTA_TCP

    def test_delta_tcp_still_works(self):
        """Regular delta TCP should still parse correctly."""
        cfg = Mock()
        cfg.output_features = {"action": Mock()}
        cfg.output_features["action"].shape = [14]
        cfg.metadata = {
            "features": {
                "action": {
                    "names": [
                        "delta_tcp_left_pos_x",
                    ] + ["x"] * 13,
                }
            }
        }

        mode = ActionMode.parse_action_mode(cfg)
        assert mode == ActionMode.DELTA_TCP

    def test_absolute_mode(self):
        """UMI_DELTA_TCP should map to TCP in absolute mode."""
        abs_mode = ActionMode.get_absolute_mode(ActionMode.UMI_DELTA_TCP)
        assert abs_mode == ActionMode.TCP

    def test_umi_action_dim(self):
        """UMI_ACTION_DIM should be 20."""
        assert UMI_ACTION_DIM == 20

    def test_umi_index_layout(self):
        """UMI index constants should cover all 20 dims without overlap."""
        all_indices = set()

        for idx in range(UMI_LEFT_POS_IDXS.start, UMI_LEFT_POS_IDXS.stop):
            all_indices.add(idx)
        for idx in range(UMI_LEFT_ROT6D_IDXS.start, UMI_LEFT_ROT6D_IDXS.stop):
            all_indices.add(idx)
        all_indices.add(UMI_LEFT_GRIPPER_IDX)

        for idx in range(UMI_RIGHT_POS_IDXS.start, UMI_RIGHT_POS_IDXS.stop):
            all_indices.add(idx)
        for idx in range(UMI_RIGHT_ROT6D_IDXS.start, UMI_RIGHT_ROT6D_IDXS.stop):
            all_indices.add(idx)
        all_indices.add(UMI_RIGHT_GRIPPER_IDX)

        assert all_indices == set(range(UMI_ACTION_DIM))

    def test_rotation_feature_indices(self):
        """UMI_ROTATION_FEATURE_INDICES should be exactly the rot6d slices."""
        expected = list(range(3, 9)) + list(range(12, 18))
        assert UMI_ROTATION_FEATURE_INDICES == expected
        assert len(UMI_ROTATION_FEATURE_INDICES) == 12

    def test_gripper_idx_functions(self):
        """Gripper index functions should return correct values for UMI mode."""
        assert GET_LEFT_GRIPPER_IDX(ActionMode.UMI_DELTA_TCP) == 18
        assert GET_RIGHT_GRIPPER_IDX(ActionMode.UMI_DELTA_TCP) == 19
