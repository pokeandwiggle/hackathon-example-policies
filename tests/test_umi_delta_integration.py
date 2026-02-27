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

"""Integration tests for UMI-delta pipeline wiring.

Covers:
    1. StepwiseNormalizerProcessorStep / StepwiseUnnormalizerProcessorStep
       - state_dict roundtrip
       - action normalization / observation passthrough
    2. ActionTranslator._umi_delta_tcp
       - identity delta produces original pose
       - known delta translations
    3. DiTFlowConfig stepwise wiring
       - normalization_mapping auto-set to IDENTITY for actions
       - clip_sample_range auto-adjusted
    4. Processor pipeline construction
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from example_policies.utils.stepwise_normalize import (
    StepwisePercentileNormalize,
    StepwisePercentileUnnormalize,
)
from example_policies.utils.stepwise_processor import (
    STEPWISE_STATS_FILENAME,
    StepwiseNormalizerProcessorStep,
    StepwiseUnnormalizerProcessorStep,
    load_stepwise_stats,
    save_stepwise_stats,
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
    UMI_RIGHT_ROT6D_IDXS,
    UMI_ROTATION_FEATURE_INDICES,
    ActionMode,
)
from example_policies.utils.constants import OBSERVATION_STATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_random_stats(horizon: int = 4, action_dim: int = 20):
    """Create synthetic p_low / p_high for testing."""
    p_low = torch.randn(horizon, action_dim) - 1.0
    p_high = p_low + torch.rand(horizon, action_dim) * 2.0 + 0.5  # ensure positive spread
    return {"p_low": p_low, "p_high": p_high}


def _make_identity_umi_delta(gripper_l: float = 0.5, gripper_r: float = 0.5) -> torch.Tensor:
    """Create a UMI-delta action that represents the identity transform (no movement)."""
    action = torch.zeros(1, 20)
    # Position deltas = 0 (already zero)
    # Rotation 6D identity = first two columns of I_3: [1,0,0, 0,1,0]
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    action[0, UMI_LEFT_ROT6D_IDXS] = identity_6d
    action[0, UMI_RIGHT_ROT6D_IDXS] = identity_6d
    action[0, UMI_LEFT_GRIPPER_IDX] = gripper_l
    action[0, UMI_RIGHT_GRIPPER_IDX] = gripper_r
    return action


def _make_observation_with_tcp(
    left_pos=(0.5, 0.0, 0.3),
    left_quat=(0.0, 0.0, 0.0, 1.0),
    right_pos=(-0.5, 0.0, 0.3),
    right_quat=(0.0, 0.0, 0.0, 1.0),
) -> dict:
    """Create an observation dict with a 14-dim TCP state."""
    state = torch.zeros(1, 14)
    state[0, DUAL_ABS_LEFT_POS_IDXS] = torch.tensor(left_pos)
    state[0, DUAL_ABS_LEFT_QUAT_IDXS] = torch.tensor(left_quat)
    state[0, DUAL_ABS_RIGHT_POS_IDXS] = torch.tensor(right_pos)
    state[0, DUAL_ABS_RIGHT_QUAT_IDXS] = torch.tensor(right_quat)
    return {OBSERVATION_STATE: state}


# ===========================================================================
# 1. StepwiseProcessorStep tests
# ===========================================================================


class TestStepwiseNormalizerProcessorStep:
    """Tests for``StepwiseNormalizerProcessorStep``."""

    def test_state_dict_roundtrip(self):
        """state_dict → load_state_dict recovers normalizer exactly."""
        stats = _make_random_stats(horizon=4, action_dim=20)
        step = StepwiseNormalizerProcessorStep(
            p_low=stats["p_low"],
            p_high=stats["p_high"],
            skip_feature_indices=UMI_ROTATION_FEATURE_INDICES,
        )

        sd = step.state_dict()
        assert "p02" in sd
        assert "p98" in sd
        assert "normalize_mask" in sd

        # Create new step without stats and load
        step2 = StepwiseNormalizerProcessorStep(
            skip_feature_indices=UMI_ROTATION_FEATURE_INDICES,
        )
        assert step2._normalizer is None
        step2.load_state_dict(sd)
        assert step2._normalizer is not None
        torch.testing.assert_close(step2._normalizer.p02, step._normalizer.p02)
        torch.testing.assert_close(step2._normalizer.p98, step._normalizer.p98)
        torch.testing.assert_close(
            step2._normalizer.normalize_mask, step._normalizer.normalize_mask
        )

    def test_normalizes_action_in_transition(self):
        """ProcessorStep normalises the ACTION key and leaves OBSERVATION untouched."""
        from lerobot.processor.core import TransitionKey

        stats = _make_random_stats(horizon=4, action_dim=8)
        step = StepwiseNormalizerProcessorStep(
            p_low=stats["p_low"],
            p_high=stats["p_high"],
            skip_feature_indices=[],
        )

        obs = torch.randn(1, 32)  # arbitrary observation
        action = torch.randn(1, 4, 8)  # (B, H, D)

        transition = {
            TransitionKey.OBSERVATION: {OBSERVATION_STATE: obs.clone()},
            TransitionKey.ACTION: action.clone(),
        }

        result = step(transition)

        # Observation should be unchanged
        torch.testing.assert_close(
            result[TransitionKey.OBSERVATION][OBSERVATION_STATE], obs
        )
        # Action should be different (normalised)
        assert not torch.allclose(result[TransitionKey.ACTION], action)
        # Action should be within clip range
        assert result[TransitionKey.ACTION].min() >= -1.5
        assert result[TransitionKey.ACTION].max() <= 1.5

    def test_no_stats_passthrough(self):
        """Without stats, normalizer passes actions through unchanged."""
        from lerobot.processor.core import TransitionKey

        step = StepwiseNormalizerProcessorStep()
        action = torch.randn(1, 4, 8)
        transition = {TransitionKey.ACTION: action.clone()}
        result = step(transition)
        torch.testing.assert_close(result[TransitionKey.ACTION], action)


class TestStepwiseUnnormalizerProcessorStep:
    """Tests for ``StepwiseUnnormalizerProcessorStep``."""

    def test_state_dict_roundtrip(self):
        stats = _make_random_stats(horizon=4, action_dim=20)
        step = StepwiseUnnormalizerProcessorStep(
            p_low=stats["p_low"],
            p_high=stats["p_high"],
            skip_feature_indices=UMI_ROTATION_FEATURE_INDICES,
        )
        sd = step.state_dict()
        step2 = StepwiseUnnormalizerProcessorStep(
            skip_feature_indices=UMI_ROTATION_FEATURE_INDICES,
        )
        step2.load_state_dict(sd)
        torch.testing.assert_close(step2._unnormalizer.p02, step._unnormalizer.p02)
        torch.testing.assert_close(step2._unnormalizer.p98, step._unnormalizer.p98)

    def test_normalize_unnormalize_roundtrip(self):
        """Normalise → unnormalise recovers original within tolerance."""
        from lerobot.processor.core import TransitionKey

        stats = _make_random_stats(horizon=4, action_dim=8)
        norm_step = StepwiseNormalizerProcessorStep(
            p_low=stats["p_low"],
            p_high=stats["p_high"],
            skip_feature_indices=[],
        )
        unnorm_step = StepwiseUnnormalizerProcessorStep(
            p_low=stats["p_low"],
            p_high=stats["p_high"],
            skip_feature_indices=[],
        )

        # Use values within the p_low..p_high range to avoid clipping
        action = stats["p_low"] + (stats["p_high"] - stats["p_low"]) * torch.rand(4, 8)
        action = action.unsqueeze(0)  # (1, H, D)

        transition = {TransitionKey.ACTION: action.clone()}
        normalised = norm_step(transition)
        restored = unnorm_step(normalised)

        torch.testing.assert_close(
            restored[TransitionKey.ACTION], action, atol=1e-5, rtol=1e-5
        )


# ===========================================================================
# 2. Stats file utilities
# ===========================================================================


class TestStepwiseStatsIO:
    """Tests for save/load of stepwise percentile stats JSON."""

    def test_save_load_roundtrip(self, tmp_path: Path):
        stats = _make_random_stats(horizon=4, action_dim=20)
        save_stepwise_stats(stats, tmp_path)
        loaded = load_stepwise_stats(tmp_path)
        torch.testing.assert_close(loaded["p_low"], stats["p_low"])
        torch.testing.assert_close(loaded["p_high"], stats["p_high"])

    def test_save_to_file_path(self, tmp_path: Path):
        stats = _make_random_stats(horizon=2, action_dim=5)
        file_path = tmp_path / "custom_stats.json"
        save_stepwise_stats(stats, file_path)
        assert file_path.exists()
        loaded = load_stepwise_stats(file_path)
        torch.testing.assert_close(loaded["p_low"], stats["p_low"])

    def test_saved_json_is_valid(self, tmp_path: Path):
        stats = _make_random_stats(horizon=3, action_dim=4)
        save_stepwise_stats(stats, tmp_path)
        with open(tmp_path / STEPWISE_STATS_FILENAME) as f:
            data = json.load(f)
        assert "p_low" in data
        assert "p_high" in data
        assert len(data["p_low"]) == 3
        assert len(data["p_low"][0]) == 4


# ===========================================================================
# 3. ActionTranslator UMI-delta
# ===========================================================================


class TestActionTranslatorUMIDelta:
    """Tests for ActionTranslator._umi_delta_tcp."""

    def _make_translator(self):
        """Create an ActionTranslator configured for UMI_DELTA_TCP."""
        # Build a minimal cfg mock
        cfg = MagicMock()
        cfg.metadata = {
            "features": {
                OBSERVATION_STATE: {
                    "names": (
                        [f"tcp_left_pos_{c}" for c in "xyz"]
                        + [f"tcp_left_quat_{c}" for c in "xyzw"]
                        + [f"tcp_right_pos_{c}" for c in "xyz"]
                        + [f"tcp_right_quat_{c}" for c in "xyzw"]
                    )
                }
            }
        }
        # Patch output_shapes to 20-dim action  and action_type
        cfg.output_shapes = {"action": [20]}
        cfg.action_type = "umi_delta_tcp"

        from example_policies.robot_deploy.deploy_core.action_translator import (
            ActionTranslator,
        )

        translator = ActionTranslator.__new__(ActionTranslator)
        translator.last_action = None
        translator.action_mode = ActionMode.UMI_DELTA_TCP
        translator._state_feature_names = cfg.metadata["features"][OBSERVATION_STATE][
            "names"
        ]
        translator.state_info_idxs = translator.compute_state_info_indices(
            ActionMode.UMI_DELTA_TCP
        )
        return translator

    def test_identity_delta_preserves_pose(self):
        """Identity UMI-delta (zero pos, identity rot6d) → same absolute pose."""
        translator = self._make_translator()

        left_pos = (0.5, 0.1, 0.3)
        # scipy <1.13 doesn't support scalar_last kwarg; default is xyzw (scalar last)
        left_quat = tuple(R.from_euler("xyz", [10, 20, 30], degrees=True).as_quat())
        right_pos = (-0.5, -0.1, 0.3)
        right_quat = tuple(R.from_euler("xyz", [5, -10, 15], degrees=True).as_quat())

        obs = _make_observation_with_tcp(left_pos, left_quat, right_pos, right_quat)
        umi_action = _make_identity_umi_delta(gripper_l=0.7, gripper_r=0.3)

        result = translator._umi_delta_tcp(umi_action, obs)

        # Result should be (1, 16)
        assert result.shape == (1, 16)

        # Positions should match the reference
        torch.testing.assert_close(
            result[0, DUAL_ABS_LEFT_POS_IDXS],
            torch.tensor(left_pos),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            result[0, DUAL_ABS_RIGHT_POS_IDXS],
            torch.tensor(right_pos),
            atol=1e-5,
            rtol=1e-5,
        )

        # Quaternions should be close (up to sign)
        ref_left_q = torch.tensor(left_quat, dtype=torch.float32)
        res_left_q = result[0, DUAL_ABS_LEFT_QUAT_IDXS]
        # Handle quaternion double-cover
        cos_angle = torch.abs(torch.dot(ref_left_q, res_left_q))
        assert cos_angle > 0.999, f"Left quat mismatch: cos_angle={cos_angle}"

        ref_right_q = torch.tensor(right_quat, dtype=torch.float32)
        res_right_q = result[0, DUAL_ABS_RIGHT_QUAT_IDXS]
        cos_angle = torch.abs(torch.dot(ref_right_q, res_right_q))
        assert cos_angle > 0.999, f"Right quat mismatch: cos_angle={cos_angle}"

    def test_known_translation_delta(self):
        """A pure position delta should shift the absolute position accordingly."""
        translator = self._make_translator()

        obs = _make_observation_with_tcp(
            left_pos=(0.5, 0.0, 0.3),
            left_quat=(0.0, 0.0, 0.0, 1.0),
            right_pos=(-0.5, 0.0, 0.3),
            right_quat=(0.0, 0.0, 0.0, 1.0),
        )

        umi_action = _make_identity_umi_delta()
        # Add a translation delta to the left arm
        umi_action[0, UMI_LEFT_POS_IDXS] = torch.tensor([0.1, 0.2, -0.05])

        result = translator._umi_delta_tcp(umi_action, obs)

        expected_left_pos = torch.tensor([0.6, 0.2, 0.25])
        torch.testing.assert_close(
            result[0, DUAL_ABS_LEFT_POS_IDXS],
            expected_left_pos,
            atol=1e-5,
            rtol=1e-5,
        )
        # Right arm should remain unchanged
        torch.testing.assert_close(
            result[0, DUAL_ABS_RIGHT_POS_IDXS],
            torch.tensor([-0.5, 0.0, 0.3]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_output_quaternions_are_normalised(self):
        """The output quaternions should all have unit norm."""
        translator = self._make_translator()
        obs = _make_observation_with_tcp()
        umi_action = _make_identity_umi_delta()

        result = translator._umi_delta_tcp(umi_action, obs)

        left_q_norm = result[0, DUAL_ABS_LEFT_QUAT_IDXS].norm()
        right_q_norm = result[0, DUAL_ABS_RIGHT_QUAT_IDXS].norm()
        assert abs(left_q_norm.item() - 1.0) < 1e-5
        assert abs(right_q_norm.item() - 1.0) < 1e-5

    def test_gripper_values_pass_through(self):
        """Gripper values from UMI action should appear in the output."""
        translator = self._make_translator()
        obs = _make_observation_with_tcp()
        umi_action = _make_identity_umi_delta(gripper_l=0.8, gripper_r=0.2)

        result = translator._umi_delta_tcp(umi_action, obs)

        # In the 16-dim absolute TCP output, grippers are at indices 14, 15
        assert abs(result[0, 14].item() - 0.8) < 1e-6
        assert abs(result[0, 15].item() - 0.2) < 1e-6


# ===========================================================================
# 4. DiTFlowConfig stepwise wiring
# ===========================================================================


class TestDiTFlowConfigStepwise:
    """Tests for stepwise normalization config fields."""

    def test_stepwise_sets_action_identity(self):
        """When use_stepwise_normalization=True, ACTION norm mode → IDENTITY."""
        from lerobot.configs.types import NormalizationMode

        from example_policies.policies.models.dit_flow.configuration_dit_flow import (
            DiTFlowConfig,
        )

        config = DiTFlowConfig(use_stepwise_normalization=True)
        assert config.normalization_mapping["ACTION"] == NormalizationMode.IDENTITY
        # Observations should remain untouched
        assert config.normalization_mapping["VISUAL"] == NormalizationMode.MEAN_STD
        assert config.normalization_mapping["STATE"] == NormalizationMode.MIN_MAX

    def test_stepwise_adjusts_clip_sample_range(self):
        """clip_sample_range should auto-adjust to 1.5 when stepwise is on."""
        from example_policies.policies.models.dit_flow.configuration_dit_flow import (
            DiTFlowConfig,
        )

        config = DiTFlowConfig(use_stepwise_normalization=True)
        assert config.clip_sample_range == 1.5

    def test_stepwise_custom_clip_range_not_overridden(self):
        """If user explicitly sets clip_sample_range, it shouldn't be overridden."""
        from example_policies.policies.models.dit_flow.configuration_dit_flow import (
            DiTFlowConfig,
        )

        config = DiTFlowConfig(
            use_stepwise_normalization=True,
            clip_sample_range=2.0,
        )
        assert config.clip_sample_range == 2.0

    def test_default_has_no_stepwise(self):
        """Default DiTFlowConfig should not use stepwise normalization."""
        from lerobot.configs.types import NormalizationMode

        from example_policies.policies.models.dit_flow.configuration_dit_flow import (
            DiTFlowConfig,
        )

        config = DiTFlowConfig()
        assert config.use_stepwise_normalization is False
        assert config.normalization_mapping["ACTION"] == NormalizationMode.MIN_MAX
        assert config.clip_sample_range == 1.0


# ===========================================================================
# 5. Processor pipeline wiring
# ===========================================================================


class TestProcessorPipelineWiring:
    """Test that make_ditflow_pre_post_processors correctly wires stepwise steps."""

    def test_standard_pipeline_has_no_stepwise(self):
        """Without stepwise, pipeline should not contain stepwise steps."""
        from example_policies.policies.models.dit_flow.configuration_dit_flow import (
            DiTFlowConfig,
        )
        from example_policies.policies.models.dit_flow.processor_dit_flow import (
            make_ditflow_pre_post_processors,
        )

        config = DiTFlowConfig()
        pre, post = make_ditflow_pre_post_processors(config)
        step_types = [type(s).__name__ for s in pre.steps]
        assert "StepwiseNormalizerProcessorStep" not in step_types

        post_step_types = [type(s).__name__ for s in post.steps]
        assert "StepwiseUnnormalizerProcessorStep" not in post_step_types

    def test_stepwise_pipeline_has_stepwise_steps(self):
        """With stepwise enabled, pipeline should contain stepwise steps."""
        from example_policies.policies.models.dit_flow.configuration_dit_flow import (
            DiTFlowConfig,
        )
        from example_policies.policies.models.dit_flow.processor_dit_flow import (
            make_ditflow_pre_post_processors,
        )

        config = DiTFlowConfig(
            use_stepwise_normalization=True,
            stepwise_skip_feature_indices=UMI_ROTATION_FEATURE_INDICES,
        )
        pre, post = make_ditflow_pre_post_processors(config)

        step_types = [type(s).__name__ for s in pre.steps]
        assert "StepwiseNormalizerProcessorStep" in step_types

        post_step_types = [type(s).__name__ for s in post.steps]
        assert "StepwiseUnnormalizerProcessorStep" in post_step_types

    def test_stepwise_pipeline_with_stats_file(self, tmp_path: Path):
        """Pipeline should load stats from a file when path is provided."""
        from example_policies.policies.models.dit_flow.configuration_dit_flow import (
            DiTFlowConfig,
        )
        from example_policies.policies.models.dit_flow.processor_dit_flow import (
            make_ditflow_pre_post_processors,
        )

        stats = _make_random_stats(horizon=4, action_dim=20)
        stats_file = tmp_path / STEPWISE_STATS_FILENAME
        save_stepwise_stats(stats, stats_file)

        config = DiTFlowConfig(
            use_stepwise_normalization=True,
            stepwise_stats_path=str(stats_file),
            stepwise_skip_feature_indices=UMI_ROTATION_FEATURE_INDICES,
        )
        pre, post = make_ditflow_pre_post_processors(config)

        # Find the stepwise normalizer step and check it has stats
        stepwise_step = None
        for s in pre.steps:
            if isinstance(s, StepwiseNormalizerProcessorStep):
                stepwise_step = s
                break
        assert stepwise_step is not None
        assert stepwise_step._normalizer is not None
        torch.testing.assert_close(stepwise_step._normalizer.p02, stats["p_low"])
        torch.testing.assert_close(stepwise_step._normalizer.p98, stats["p_high"])

    def test_stepwise_normalizer_step_ordering(self):
        """Stepwise normalizer should come after standard normalizer in preprocessor."""
        from lerobot.processor import NormalizerProcessorStep

        from example_policies.policies.models.dit_flow.configuration_dit_flow import (
            DiTFlowConfig,
        )
        from example_policies.policies.models.dit_flow.processor_dit_flow import (
            make_ditflow_pre_post_processors,
        )

        config = DiTFlowConfig(use_stepwise_normalization=True)
        pre, _ = make_ditflow_pre_post_processors(config)

        norm_idx = None
        stepwise_idx = None
        for i, s in enumerate(pre.steps):
            if isinstance(s, NormalizerProcessorStep):
                norm_idx = i
            if isinstance(s, StepwiseNormalizerProcessorStep):
                stepwise_idx = i

        assert norm_idx is not None, "NormalizerProcessorStep not found"
        assert stepwise_idx is not None, "StepwiseNormalizerProcessorStep not found"
        assert norm_idx < stepwise_idx, (
            f"Standard normalizer (idx={norm_idx}) should come before "
            f"stepwise normalizer (idx={stepwise_idx})"
        )


# ===========================================================================
# 5. Checkpoint save / load roundtrip
# ===========================================================================


class TestCheckpointRoundtrip:
    """Verify that stepwise processor state survives save_pretrained → from_pretrained."""

    def _make_preprocessor_pipeline(self, stats):
        """Build a minimal preprocessor pipeline with a stepwise normalizer step."""
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME

        step = StepwiseNormalizerProcessorStep(
            p_low=stats["p_low"],
            p_high=stats["p_high"],
            skip_feature_indices=list(UMI_ROTATION_FEATURE_INDICES),
            clip_min=-1.5,
            clip_max=1.5,
        )
        return PolicyProcessorPipeline(
            steps=[step],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        )

    def _make_postprocessor_pipeline(self, stats):
        """Build a minimal postprocessor pipeline with a stepwise unnormalizer step."""
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            policy_action_to_transition,
            transition_to_policy_action,
        )
        from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME

        step = StepwiseUnnormalizerProcessorStep(
            p_low=stats["p_low"],
            p_high=stats["p_high"],
            skip_feature_indices=list(UMI_ROTATION_FEATURE_INDICES),
        )
        return PolicyProcessorPipeline(
            steps=[step],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

    def test_preprocessor_save_load_roundtrip(self, tmp_path: Path):
        """Stepwise normalizer stats should survive preprocessor checkpoint roundtrip."""
        from lerobot.processor import PolicyProcessorPipeline

        stats = _make_random_stats(horizon=4, action_dim=20)
        pipeline = self._make_preprocessor_pipeline(stats)

        config_filename = "policy_preprocessor.json"
        pipeline.save_pretrained(tmp_path, config_filename=config_filename)

        loaded = PolicyProcessorPipeline.from_pretrained(
            tmp_path, config_filename=config_filename
        )

        # Find the stepwise step in the loaded pipeline
        stepwise_steps = [
            s for s in loaded.steps
            if isinstance(s, StepwiseNormalizerProcessorStep)
        ]
        assert len(stepwise_steps) == 1, "Expected exactly one StepwiseNormalizerProcessorStep"
        loaded_step = stepwise_steps[0]

        assert loaded_step._normalizer is not None, "Normalizer should be rebuilt after load"
        torch.testing.assert_close(loaded_step._normalizer.p02, stats["p_low"])
        torch.testing.assert_close(loaded_step._normalizer.p98, stats["p_high"])

    def test_postprocessor_save_load_roundtrip(self, tmp_path: Path):
        """Stepwise unnormalizer stats should survive postprocessor checkpoint roundtrip."""
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            policy_action_to_transition,
            transition_to_policy_action,
        )

        stats = _make_random_stats(horizon=4, action_dim=20)
        pipeline = self._make_postprocessor_pipeline(stats)

        config_filename = "policy_postprocessor.json"
        pipeline.save_pretrained(tmp_path, config_filename=config_filename)

        loaded = PolicyProcessorPipeline.from_pretrained(
            tmp_path,
            config_filename=config_filename,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

        stepwise_steps = [
            s for s in loaded.steps
            if isinstance(s, StepwiseUnnormalizerProcessorStep)
        ]
        assert len(stepwise_steps) == 1, "Expected exactly one StepwiseUnnormalizerProcessorStep"
        loaded_step = stepwise_steps[0]

        assert loaded_step._unnormalizer is not None, "Unnormalizer should be rebuilt after load"
        torch.testing.assert_close(loaded_step._unnormalizer.p02, stats["p_low"])
        torch.testing.assert_close(loaded_step._unnormalizer.p98, stats["p_high"])

    def test_config_preserved_after_roundtrip(self, tmp_path: Path):
        """Step config (skip_feature_indices, clip bounds) should survive roundtrip."""
        from lerobot.processor import PolicyProcessorPipeline

        stats = _make_random_stats(horizon=4, action_dim=20)
        pipeline = self._make_preprocessor_pipeline(stats)

        config_filename = "policy_preprocessor.json"
        pipeline.save_pretrained(tmp_path, config_filename=config_filename)

        loaded = PolicyProcessorPipeline.from_pretrained(
            tmp_path, config_filename=config_filename
        )

        loaded_step = [
            s for s in loaded.steps
            if isinstance(s, StepwiseNormalizerProcessorStep)
        ][0]

        assert loaded_step.skip_feature_indices == list(UMI_ROTATION_FEATURE_INDICES)
        assert loaded_step.clip_min == -1.5
        assert loaded_step.clip_max == 1.5

    def test_normalize_mask_preserved(self, tmp_path: Path):
        """The normalize_mask tensor should match after save/load."""
        from lerobot.processor import PolicyProcessorPipeline

        stats = _make_random_stats(horizon=4, action_dim=20)
        pipeline = self._make_preprocessor_pipeline(stats)

        # Grab original mask
        orig_step = pipeline.steps[0]
        assert isinstance(orig_step, StepwiseNormalizerProcessorStep)
        orig_mask = orig_step._normalizer.normalize_mask.clone()

        config_filename = "policy_preprocessor.json"
        pipeline.save_pretrained(tmp_path, config_filename=config_filename)

        loaded = PolicyProcessorPipeline.from_pretrained(
            tmp_path, config_filename=config_filename
        )
        loaded_step = [
            s for s in loaded.steps
            if isinstance(s, StepwiseNormalizerProcessorStep)
        ][0]

        torch.testing.assert_close(
            loaded_step._normalizer.normalize_mask, orig_mask
        )

    def test_saved_files_exist(self, tmp_path: Path):
        """save_pretrained should create a JSON config and a .safetensors state file."""
        stats = _make_random_stats(horizon=4, action_dim=20)
        pipeline = self._make_preprocessor_pipeline(stats)

        config_filename = "policy_preprocessor.json"
        pipeline.save_pretrained(tmp_path, config_filename=config_filename)

        # Check config file
        config_path = tmp_path / config_filename
        assert config_path.exists(), f"Config file not found: {config_path}"

        with open(config_path) as f:
            config = json.load(f)
        assert "steps" in config
        assert len(config["steps"]) == 1
        assert "state_file" in config["steps"][0]

        # Check state file
        state_path = tmp_path / config["steps"][0]["state_file"]
        assert state_path.exists(), f"State file not found: {state_path}"
