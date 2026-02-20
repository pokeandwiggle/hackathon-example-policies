#!/usr/bin/env python
"""End-to-end test of the UMI-delta + stepwise normalizer pipeline with real data.

This test uses the converted absolute TCP dataset (data/e2e_test_abs-default/)
to verify:
  1. Dataset conversion produced valid absolute TCP data (16-dim actions, 18-dim state)
  2. Stepwise stats computation from parquet (chunk-relative UMI-delta)
  3. Full policy construction with auto-detected chunk-relative + stepwise norm
  4. Forward pass (training loss computation) with real observations
  5. Inference (action generation) and round-trip normalization sanity
  6. Chunk-relative conversion produces sensible deltas (near-zero for t=0)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
import torch

# ── Locate the dataset ──────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # pokeandwiggle/
DATASET_DIR = REPO_ROOT / "data" / "e2e_test_abs-default"

pytestmark = pytest.mark.skipif(
    not DATASET_DIR.exists(),
    reason=f"Real-data dataset not found at {DATASET_DIR}. "
    "Run the conversion first: python -m example_policies.data_ops.dataset_conversion_synced ...",
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_parquet() -> pd.DataFrame:
    """Load the parquet data from the converted dataset."""
    parquet_files = sorted((DATASET_DIR / "data").rglob("*.parquet"))
    assert len(parquet_files) > 0, "No parquet files found"
    return pd.read_parquet(parquet_files[0])


def _load_info() -> dict:
    info_path = DATASET_DIR / "meta" / "info.json"
    return json.loads(info_path.read_text())


# ════════════════════════════════════════════════════════════════════════════
# Test 1: Dataset structure validation
# ════════════════════════════════════════════════════════════════════════════

class TestDatasetStructure:
    """Verify the converted dataset has the expected absolute TCP format."""

    def test_info_json_exists(self):
        assert (DATASET_DIR / "meta" / "info.json").exists()

    def test_action_is_16dim_abs_tcp(self):
        info = _load_info()
        action_feat = info["features"]["action"]
        assert action_feat["shape"] == [16], f"Expected 16-dim actions, got {action_feat['shape']}"
        names = action_feat["names"]
        assert any(n.startswith("tcp_") for n in names), f"Action names should start with tcp_: {names}"

    def test_state_is_18dim(self):
        info = _load_info()
        state_feat = info["features"]["observation.state"]
        assert state_feat["shape"] == [18], f"Expected 18-dim state, got {state_feat['shape']}"

    def test_parquet_has_data(self):
        df = _load_parquet()
        assert len(df) > 100, f"Expected >100 frames, got {len(df)}"
        assert "action" in df.columns
        assert "observation.state" in df.columns

    def test_action_values_are_reasonable(self):
        """Absolute TCP positions should be in a reasonable range (meters)."""
        df = _load_parquet()
        actions = np.array(df["action"].tolist())
        # Position values (first 3 dims per arm) should be in [-2, 2] meters
        pos_left = actions[:, :3]
        pos_right = actions[:, 7:10]
        assert np.all(np.abs(pos_left) < 2.0), "Left arm positions out of range"
        assert np.all(np.abs(pos_right) < 2.0), "Right arm positions out of range"

    def test_quaternions_are_unit(self):
        """Quaternion components should form unit quaternions."""
        df = _load_parquet()
        actions = np.array(df["action"].tolist())
        quat_left = actions[:, 3:7]
        quat_right = actions[:, 10:14]
        norms_left = np.linalg.norm(quat_left, axis=1)
        norms_right = np.linalg.norm(quat_right, axis=1)
        np.testing.assert_allclose(norms_left, 1.0, atol=1e-3, err_msg="Left arm quats not unit")
        np.testing.assert_allclose(norms_right, 1.0, atol=1e-3, err_msg="Right arm quats not unit")


# ════════════════════════════════════════════════════════════════════════════
# Test 2: Stepwise stats computation
# ════════════════════════════════════════════════════════════════════════════

class TestStepwiseStatsComputation:
    """Verify chunk-relative UMI-delta conversion and stepwise stats."""

    def test_compute_stepwise_stats(self):
        """Compute stepwise stats from real parquet data."""
        from example_policies.utils.compute_stepwise_stats import (
            compute_stepwise_stats_from_parquet,
        )
        from example_policies.utils.stepwise_processor import load_stepwise_stats

        horizon = 16
        stats_path = compute_stepwise_stats_from_parquet(
            DATASET_DIR,
            horizon=horizon,
            obs_tcp_left_pos_indices=[0, 1, 2],
            obs_tcp_left_quat_indices=[3, 4, 5, 6],
            obs_tcp_right_pos_indices=[7, 8, 9],
            obs_tcp_right_quat_indices=[10, 11, 12, 13],
            force=True,
        )
        assert pathlib.Path(stats_path).exists()

        stats = load_stepwise_stats(stats_path)
        p_low = stats["p_low"]
        p_high = stats["p_high"]

        assert p_low.shape == (horizon, 20), f"Expected (16, 20), got {p_low.shape}"
        assert p_high.shape == (horizon, 20), f"Expected (16, 20), got {p_high.shape}"

    def test_stepwise_stats_show_fan_out(self):
        """Actions further into the future should have wider spread (the whole
        point of chunk-relative + stepwise normalization)."""
        from example_policies.utils.stepwise_processor import load_stepwise_stats

        stats_path = DATASET_DIR / "stepwise_percentile_stats.json"
        if not stats_path.exists():
            pytest.skip("Stepwise stats not computed yet")

        stats = load_stepwise_stats(stats_path)
        p_low = stats["p_low"]
        p_high = stats["p_high"]
        spread = p_high - p_low  # (H, 20)

        # Position features (indices 0,1,2 for left; 9,10,11 for right)
        pos_indices = [0, 1, 2, 9, 10, 11]
        pos_spread = spread[:, pos_indices]  # (H, 6)

        # Average spread per timestep across position dims
        avg_spread_per_step = pos_spread.mean(dim=1)  # (H,)

        # Spread at step 0 should be near zero (delta from self)
        assert avg_spread_per_step[0] < avg_spread_per_step[-1], (
            f"Spread at t=0 ({avg_spread_per_step[0]:.4f}) should be less than "
            f"at t=-1 ({avg_spread_per_step[-1]:.4f})"
        )

        # Spread should generally increase with horizon step
        # (not strictly monotonic, but last half should be wider than first half)
        first_half_mean = avg_spread_per_step[: len(avg_spread_per_step) // 2].mean()
        second_half_mean = avg_spread_per_step[len(avg_spread_per_step) // 2 :].mean()
        assert second_half_mean > first_half_mean, (
            f"Second half spread ({second_half_mean:.4f}) should exceed "
            f"first half ({first_half_mean:.4f})"
        )

    def test_chunk_relative_first_step_near_zero(self):
        """The first action in a chunk should be nearly identity (delta from self)."""
        from example_policies.utils.chunk_relative_processor import (
            abs_tcp_to_chunk_relative_umi_delta,
        )
        from example_policies.data_ops.utils.rotation_6d import quat_to_6d_torch

        df = _load_parquet()
        obs_states = torch.tensor(np.array(df["observation.state"].tolist()), dtype=torch.float32)
        actions = torch.tensor(np.array(df["action"].tolist()), dtype=torch.float32)

        # Take first 16 frames as a chunk
        horizon = 16
        abs_chunk = actions[:horizon].unsqueeze(0)  # (1, 16, 16)

        ref_state = obs_states[0]
        ref_pos_l = ref_state[:3].unsqueeze(0)
        ref_quat_l = ref_state[3:7].unsqueeze(0)
        ref_pos_r = ref_state[7:10].unsqueeze(0)
        ref_quat_r = ref_state[10:14].unsqueeze(0)

        ref_rot6d_l = quat_to_6d_torch(ref_quat_l)
        ref_rot6d_r = quat_to_6d_torch(ref_quat_r)

        umi_chunk = abs_tcp_to_chunk_relative_umi_delta(
            abs_chunk, ref_pos_l, ref_rot6d_l, ref_pos_r, ref_rot6d_r
        )

        # The first action's position delta should be small (within a few cm)
        first_action = umi_chunk[0, 0]  # (20,)
        pos_delta_left = first_action[:3]
        pos_delta_right = first_action[9:12]

        assert torch.abs(pos_delta_left).max() < 0.1, (
            f"First action left pos delta too large: {pos_delta_left}"
        )
        assert torch.abs(pos_delta_right).max() < 0.1, (
            f"First action right pos delta too large: {pos_delta_right}"
        )


# ════════════════════════════════════════════════════════════════════════════
# Test 3: Policy construction with real dataset
# ════════════════════════════════════════════════════════════════════════════

class TestPolicyConstruction:
    """Verify auto-detection of abs TCP and full policy build."""

    def test_config_factory_detects_abs_tcp(self):
        """DiTFlowConfig should auto-detect abs TCP actions."""
        from example_policies.config_factory import DiTFlowConfig as FactoryConfig

        cfg = FactoryConfig()
        cfg.dataset_root_dir = str(DATASET_DIR)
        assert cfg._is_abs_tcp_dataset(), "Should detect abs TCP dataset"

    def test_config_factory_finds_tcp_indices(self):
        from example_policies.config_factory import DiTFlowConfig as FactoryConfig

        cfg = FactoryConfig()
        cfg.dataset_root_dir = str(DATASET_DIR)
        indices = cfg._get_obs_tcp_indices()
        assert indices["obs_tcp_left_pos_indices"] == [0, 1, 2]
        assert indices["obs_tcp_left_quat_indices"] == [3, 4, 5, 6]
        assert indices["obs_tcp_right_pos_indices"] == [7, 8, 9]
        assert indices["obs_tcp_right_quat_indices"] == [10, 11, 12, 13]

    def test_config_factory_enables_chunk_relative_and_stepwise(self):
        from example_policies.config_factory import DiTFlowConfig as FactoryConfig

        cfg = FactoryConfig()
        cfg.dataset_root_dir = str(DATASET_DIR)
        kwargs = cfg.default_policy_kwargs
        assert kwargs["use_chunk_relative_actions"] is True
        assert kwargs["use_stepwise_normalization"] is True
        assert "stepwise_stats_path" in kwargs
        assert kwargs["clip_sample_range"] == 1.5


# ════════════════════════════════════════════════════════════════════════════
# Test 4: Full forward pass with real data
# ════════════════════════════════════════════════════════════════════════════

class TestForwardPass:
    """Run a training forward pass with real observations and actions."""

    @pytest.fixture
    def policy_and_processors(self):
        """Build a DiTFlow policy configured for the real dataset."""
        from example_policies.config_factory import DiTFlowConfig as FactoryConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.datasets.utils import dataset_to_policy_features
        from lerobot.datasets.factory import resolve_delta_timestamps
        from lerobot.configs.types import FeatureType

        # Create policy config via the factory (auto-detects abs TCP)
        factory_cfg = FactoryConfig()
        factory_cfg.dataset_root_dir = str(DATASET_DIR)
        factory_cfg.wandb_enable = False

        # Build train pipeline config
        train_cfg = factory_cfg.build()
        policy_config = train_cfg.policy

        # Populate features (same as lerobot's make_policy does)
        meta = LeRobotDatasetMetadata(repo_id=DATASET_DIR.name, root=DATASET_DIR)
        features = dataset_to_policy_features(meta.features)
        if not policy_config.output_features:
            policy_config.output_features = {
                key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
            }
        if not policy_config.input_features:
            policy_config.input_features = {
                key: ft for key, ft in features.items() if key not in policy_config.output_features
            }

        dataset_stats = meta.stats

        # Create the policy
        from example_policies.policies.models.dit_flow.modeling_dit_flow import DiTFlowPolicy
        from example_policies.policies.models.dit_flow.processor_dit_flow import (
            make_ditflow_pre_post_processors,
        )

        policy = DiTFlowPolicy(config=policy_config, dataset_stats=dataset_stats)

        # Build pre/post processors
        preprocessor, postprocessor = make_ditflow_pre_post_processors(
            config=policy_config,
            dataset_stats=dataset_stats,
        )

        # Load dataset with proper delta timestamps (windowing)
        delta_timestamps = resolve_delta_timestamps(policy_config, meta)
        dataset = LeRobotDataset(
            repo_id=DATASET_DIR.name,
            root=DATASET_DIR,
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
        )

        return policy, preprocessor, postprocessor, dataset, policy_config

    def test_training_forward_pass(self, policy_and_processors):
        """Run one training step with real data and check loss is finite."""
        policy, preprocessor, postprocessor, dataset, config = policy_and_processors

        # Use a DataLoader (batch_size=1) to get the batch dimension,
        # just like real training does. AddBatchDimensionActionStep only
        # unsqueezes 1D tensors; with windowed (2D) actions, the DataLoader
        # collation is what provides the leading batch dim.
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        batch = next(iter(loader))

        assert "observation.state" in batch
        assert "action" in batch

        # Preprocess
        processed = preprocessor(batch)

        # Check that after preprocessing, action is 20-dim (chunk-relative)
        action = processed["action"]
        assert action.ndim == 3, f"Expected (B, H, D), got shape {action.shape}"
        assert action.shape[-1] == 20, (
            f"After chunk-relative conversion, action should be 20-dim, got {action.shape[-1]}"
        )
        assert action.shape[1] == config.horizon, (
            f"Horizon mismatch: action has {action.shape[1]} steps, config expects {config.horizon}"
        )

        # Run forward pass
        policy.train()
        loss, _ = policy(processed)
        assert loss is not None
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
        print(f"\n  Training loss with real data: {loss.item():.4f}")

    def test_inference_generates_actions(self, policy_and_processors):
        """Run inference and check output action shape.

        The select_action path is designed for online deployment: it receives one
        observation frame at a time and uses internal queues to build the temporal
        window.  So we simulate a single env step by extracting the *latest*
        observation frame (index -1 along the n_obs_steps axis) and only feeding
        observation keys (no action).
        """
        policy, preprocessor, postprocessor, dataset, config = policy_and_processors
        policy.eval()
        policy.reset()

        # Get a raw sample with windowing so we can extract single frames
        sample = dataset[10]

        # Build a single-frame observation dict (simulates one env step).
        # The preprocessor needs un-windowed tensors: (D,) for state, (C, H, W) for images.
        obs_batch = {}
        for k, v in sample.items():
            if k == "action" or k == "action_is_pad":
                continue
            if k == "observation.state":
                # Take the latest frame: (n_obs_steps, D) → (D,)
                obs_batch[k] = v[-1]
            elif k.startswith("observation.images."):
                if k.endswith("_is_pad"):
                    continue
                # Take the latest frame: (n_obs_steps, C, H, W) → (C, H, W)
                obs_batch[k] = v[-1]

        # Add batch dim (required by the policy)
        obs_batch = {k: v.unsqueeze(0) for k, v in obs_batch.items()}

        # Run inference (feed the same frame n_obs_steps times to fill queues)
        with torch.no_grad():
            action = policy.select_action(obs_batch)

        assert action is not None
        # The model produces chunk-relative 20-dim actions
        assert action.shape[-1] == 20, f"Expected 20-dim action, got shape {action.shape}"
        print(f"\n  Generated action shape: {action.shape}")
        assert torch.isfinite(action).all(), "Generated actions contain non-finite values"


# ════════════════════════════════════════════════════════════════════════════
# Test 5: Normalization round-trip
# ════════════════════════════════════════════════════════════════════════════

class TestNormalizationRoundTrip:
    """Verify that normalize → unnormalize recovers the original values."""

    def test_stepwise_normalize_unnormalize_roundtrip(self):
        """Apply stepwise norm then unnorm and check recovery."""
        from example_policies.utils.stepwise_processor import load_stepwise_stats
        from example_policies.utils.stepwise_normalize import (
            StepwisePercentileNormalize,
            StepwisePercentileUnnormalize,
        )
        from example_policies.utils.action_order import UMI_ROTATION_FEATURE_INDICES

        stats_path = DATASET_DIR / "stepwise_percentile_stats.json"
        if not stats_path.exists():
            pytest.skip("Stepwise stats not found")

        stats = load_stepwise_stats(stats_path)
        p02 = stats["p_low"]
        p98 = stats["p_high"]

        skip_indices = list(UMI_ROTATION_FEATURE_INDICES)

        normalizer = StepwisePercentileNormalize(
            p02=p02,
            p98=p98,
            skip_feature_indices=skip_indices,
            clip_min=-1.5,
            clip_max=1.5,
        )
        unnormalizer = StepwisePercentileUnnormalize(
            p02=p02,
            p98=p98,
            skip_feature_indices=skip_indices,
        )

        # Create a realistic UMI-delta action chunk
        from example_policies.utils.chunk_relative_processor import (
            abs_tcp_to_chunk_relative_umi_delta,
        )
        from example_policies.data_ops.utils.rotation_6d import quat_to_6d_torch

        df = _load_parquet()
        obs_states = torch.tensor(np.array(df["observation.state"].tolist()), dtype=torch.float32)
        actions = torch.tensor(np.array(df["action"].tolist()), dtype=torch.float32)

        horizon = 16
        # Use frame 100 as reference to avoid edge effects
        start = 100
        abs_chunk = actions[start : start + horizon].unsqueeze(0)  # (1, H, 16)

        ref_state = obs_states[start]
        ref_pos_l = ref_state[:3].unsqueeze(0)
        ref_quat_l = ref_state[3:7].unsqueeze(0)
        ref_pos_r = ref_state[7:10].unsqueeze(0)
        ref_quat_r = ref_state[10:14].unsqueeze(0)
        ref_rot6d_l = quat_to_6d_torch(ref_quat_l)
        ref_rot6d_r = quat_to_6d_torch(ref_quat_r)

        umi_chunk = abs_tcp_to_chunk_relative_umi_delta(
            abs_chunk, ref_pos_l, ref_rot6d_l, ref_pos_r, ref_rot6d_r
        ).squeeze(0)  # (H, 20)

        # Normalize
        normalized = normalizer(umi_chunk)
        # Check normalized non-rotation features are in [-1.5, 1.5]
        non_rot_mask = normalizer.normalize_mask  # (20,) bool
        non_rot_values = normalized[:, non_rot_mask]
        assert non_rot_values.min() >= -1.5 - 1e-6, f"Min value {non_rot_values.min()} below clip"
        assert non_rot_values.max() <= 1.5 + 1e-6, f"Max value {non_rot_values.max()} above clip"

        # Rotation features should pass through unchanged
        rot_indices = skip_indices
        torch.testing.assert_close(
            normalized[:, rot_indices],
            umi_chunk[:, rot_indices],
            msg="Rotation features should be unchanged by normalization",
        )

        # Unnormalize
        recovered = unnormalizer(normalized)

        # Non-clipped values should round-trip exactly
        # (Clipped values may lose information — that's expected)
        torch.testing.assert_close(
            recovered[:, rot_indices],
            umi_chunk[:, rot_indices],
            msg="Rotation features should round-trip exactly",
        )

        # For position features within the normal range, round-trip should be close
        # We can't guarantee exact round-trip for clipped values, but for typical
        # motions the error should be small
        pos_indices = [0, 1, 2, 9, 10, 11]
        pos_error = (recovered[:, pos_indices] - umi_chunk[:, pos_indices]).abs()
        mean_error = pos_error.mean()
        print(f"\n  Position round-trip mean error: {mean_error:.6f}")
        # Reasonable threshold — if things are badly broken this will fail
        assert mean_error < 0.05, f"Position round-trip error too large: {mean_error:.6f}"
