"""Tests for unified gripper width representation.

Verifies that:
- StateFeatureSpec always produces 1 gripper width per side (16-dim state)
- from_feature_names correctly round-trips for both Panda and Robotiq
- StateAssembler converts raw joint data to width (metres)
- PipelineConfig passes gripper types through to build_features
"""

import numpy as np

from example_policies.data_ops.config.pipeline_config import (
    PipelineConfig,
    build_features,
)
from example_policies.data_ops.pipeline.assembly.action_assembler import LastCommand
from example_policies.data_ops.pipeline.assembly.state_assembler import StateAssembler
from example_policies.utils.gripper import (
    ROBOTIQ_CLOSED_POSITION_RAD,
    ROBOTIQ_MAX_WIDTH_M,
    robotiq_width_from_knuckle,
)
from example_policies.utils.state_builder import GripperType, StateFeatureSpec


# ── StateFeatureSpec ──────────────────────────────────────────────────────────


class TestStateFeatureSpec:
    def test_panda_produces_16_dim(self):
        spec = StateFeatureSpec()
        names = spec.get_feature_names()
        assert len(names) == 16
        assert "gripper_left" in names
        assert "gripper_right" in names

    def test_robotiq_produces_16_dim(self):
        spec = StateFeatureSpec(
            left_gripper=GripperType.ROBOTIQ,
            right_gripper=GripperType.ROBOTIQ,
        )
        names = spec.get_feature_names()
        assert len(names) == 16
        assert "robotiq_left" in names
        assert "robotiq_right" in names
        # Should NOT have old robotiq_left_N multi-joint names
        assert not any("robotiq_left_" in n for n in names)

    def test_from_feature_names_round_trip(self):
        spec = StateFeatureSpec()
        names = spec.get_feature_names()
        rt = StateFeatureSpec.from_feature_names(names)
        assert rt.get_feature_names() == names

    def test_from_feature_names_robotiq_round_trip(self):
        spec = StateFeatureSpec(
            left_gripper=GripperType.ROBOTIQ,
            right_gripper=GripperType.ROBOTIQ,
        )
        names = spec.get_feature_names()
        rt = StateFeatureSpec.from_feature_names(names)
        assert rt.get_feature_names() == names


# ── Robotiq conversion ────────────────────────────────────────────────────────


class TestRobotiqConversion:
    def test_fully_open(self):
        """Knuckle at 0 rad → max width (0.085 m)."""
        assert robotiq_width_from_knuckle(0.0) == pytest.approx(ROBOTIQ_MAX_WIDTH_M)

    def test_fully_closed(self):
        """Knuckle at 0.7929 rad → 0 m."""
        assert robotiq_width_from_knuckle(ROBOTIQ_CLOSED_POSITION_RAD) == pytest.approx(
            0.0
        )

    def test_halfway(self):
        half_pos = ROBOTIQ_CLOSED_POSITION_RAD / 2.0
        expected = ROBOTIQ_MAX_WIDTH_M / 2.0
        assert robotiq_width_from_knuckle(half_pos) == pytest.approx(expected)


# ── StateAssembler ────────────────────────────────────────────────────────────


def _make_parsed_frame(gripper_state=None):
    tcp_left = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    tcp_right = np.array([4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if gripper_state is None:
        gripper_state = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    return {
        "actual_tcp_left": tcp_left,
        "actual_tcp_right": tcp_right,
        "gripper_state": gripper_state,
    }


class TestStateAssembler:
    def test_panda_assembles_16_dim(self):
        cfg = PipelineConfig(include_last_command=False)
        asm = StateAssembler(cfg)
        last_cmd = LastCommand(
            left=np.zeros(7, dtype=np.float32), right=np.zeros(7, dtype=np.float32)
        )
        result = asm.assemble(_make_parsed_frame(), last_cmd)
        state = result["observation.state"]
        assert len(state) == 16
        # Gripper widths: sum(0.01, 0.02) and sum(0.03, 0.04)
        np.testing.assert_array_almost_equal(state[14:], [0.03, 0.07])

    def test_robotiq_assembles_16_dim(self):
        cfg = PipelineConfig(
            left_gripper=GripperType.ROBOTIQ,
            right_gripper=GripperType.ROBOTIQ,
            include_last_command=False,
        )
        asm = StateAssembler(cfg)
        last_cmd = LastCommand(
            left=np.zeros(7, dtype=np.float32), right=np.zeros(7, dtype=np.float32)
        )
        # Robotiq: gripper_state = [left_knuckle, right_knuckle]
        knuckle_left = 0.0  # fully open
        knuckle_right = ROBOTIQ_CLOSED_POSITION_RAD  # fully closed
        gs = np.array([knuckle_left, knuckle_right], dtype=np.float32)
        result = asm.assemble(_make_parsed_frame(gripper_state=gs), last_cmd)
        state = result["observation.state"]
        assert len(state) == 16
        np.testing.assert_array_almost_equal(
            state[14:], [ROBOTIQ_MAX_WIDTH_M, 0.0], decimal=5
        )


# ── build_features ────────────────────────────────────────────────────────────


class TestBuildFeatures:
    def test_panda_features_16_dim(self):
        cfg = PipelineConfig(include_last_command=False)
        feats = build_features(cfg)
        assert feats["observation.state"]["shape"] == (16,)
        names = feats["observation.state"]["names"]
        assert "gripper_left" in names
        assert "gripper_right" in names

    def test_robotiq_features_16_dim(self):
        cfg = PipelineConfig(
            left_gripper=GripperType.ROBOTIQ,
            right_gripper=GripperType.ROBOTIQ,
            include_last_command=False,
        )
        feats = build_features(cfg)
        assert feats["observation.state"]["shape"] == (16,)
        names = feats["observation.state"]["names"]
        assert "robotiq_left" in names
        assert "robotiq_right" in names
