"""Tests for use_single_gripper_value feature.

Verifies that:
- StateFeatureSpec produces 16-dim state (1 gripper per side) or 18-dim (2 per side)
- from_feature_names correctly round-trips both modes
- StateAssembler slices the raw gripper_state appropriately
- PipelineConfig passes the flag through to build_features
"""

import numpy as np

from example_policies.data_ops.config.pipeline_config import (
    PipelineConfig,
    build_features,
)
from example_policies.data_ops.pipeline.assembly.action_assembler import LastCommand
from example_policies.data_ops.pipeline.assembly.state_assembler import StateAssembler
from example_policies.utils.state_builder import GripperType, StateFeatureSpec


# ── StateFeatureSpec ──────────────────────────────────────────────────────────


class TestStateFeatureSpec:
    def test_legacy_produces_18_dim(self):
        spec = StateFeatureSpec(use_single_gripper_value=False)
        names = spec.get_feature_names()
        assert len(names) == 18
        assert "gripper_left_0" in names
        assert "gripper_left_1" in names
        assert "gripper_right_0" in names
        assert "gripper_right_1" in names

    def test_single_produces_16_dim(self):
        spec = StateFeatureSpec(use_single_gripper_value=True)
        names = spec.get_feature_names()
        assert len(names) == 16
        assert "gripper_left" in names
        assert "gripper_right" in names
        assert "gripper_left_0" not in names
        assert "gripper_right_0" not in names

    def test_from_feature_names_legacy_round_trip(self):
        spec = StateFeatureSpec(use_single_gripper_value=False)
        names = spec.get_feature_names()
        rt = StateFeatureSpec.from_feature_names(names)
        assert rt.use_single_gripper_value is False
        assert rt.get_feature_names() == names

    def test_from_feature_names_single_round_trip(self):
        spec = StateFeatureSpec(use_single_gripper_value=True)
        names = spec.get_feature_names()
        rt = StateFeatureSpec.from_feature_names(names)
        assert rt.use_single_gripper_value is True
        assert rt.get_feature_names() == names

    def test_robotiq_ignores_single_flag(self):
        """Robotiq grippers always produce 6 values per side regardless."""
        spec = StateFeatureSpec(
            left_gripper=GripperType.ROBOTIQ,
            right_gripper=GripperType.ROBOTIQ,
            use_single_gripper_value=True,
        )
        names = spec.get_feature_names()
        robotiq_names = [n for n in names if "robotiq_" in n]
        # 6 per side = 12 total
        assert len(robotiq_names) == 12


# ── StateAssembler ────────────────────────────────────────────────────────────


def _make_parsed_frame():
    tcp_left = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    tcp_right = np.array([4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    gripper_state = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    return {
        "actual_tcp_left": tcp_left,
        "actual_tcp_right": tcp_right,
        "gripper_state": gripper_state,
    }


class TestStateAssembler:
    def test_legacy_assembles_18_dim(self):
        cfg = PipelineConfig(use_single_gripper_value=False)
        asm = StateAssembler(cfg)
        last_cmd = LastCommand(
            left=np.zeros(7, dtype=np.float32), right=np.zeros(7, dtype=np.float32)
        )
        result = asm.assemble(_make_parsed_frame(), last_cmd)
        state = result["observation.state"]
        assert len(state) == 18
        # Gripper part: all 4 finger joint values
        np.testing.assert_array_almost_equal(state[14:], [0.01, 0.02, 0.03, 0.04])

    def test_single_assembles_16_dim(self):
        cfg = PipelineConfig(use_single_gripper_value=True)
        asm = StateAssembler(cfg)
        last_cmd = LastCommand(
            left=np.zeros(7, dtype=np.float32), right=np.zeros(7, dtype=np.float32)
        )
        result = asm.assemble(_make_parsed_frame(), last_cmd)
        state = result["observation.state"]
        assert len(state) == 16
        # Gripper part: sum of both finger joints per arm
        np.testing.assert_array_almost_equal(state[14:], [0.03, 0.07])


# ── build_features ────────────────────────────────────────────────────────────


class TestBuildFeatures:
    def test_legacy_features_18_dim(self):
        cfg = PipelineConfig(use_single_gripper_value=False)
        feats = build_features(cfg)
        assert feats["observation.state"]["shape"] == (18,)

    def test_single_features_16_dim(self):
        cfg = PipelineConfig(use_single_gripper_value=True)
        feats = build_features(cfg)
        assert feats["observation.state"]["shape"] == (16,)
        names = feats["observation.state"]["names"]
        assert "gripper_left" in names
        assert "gripper_right" in names
