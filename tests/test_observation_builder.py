"""Tests for ObservationBuilder embodiment parameterization."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from example_policies.robot_deploy.robot_io.observation_builder import (
    ObservationBuilder,
)
from example_policies.utils.embodiment import get_joint_config
from example_policies.utils.state_builder import GripperType


def _make_joint(position=0.0, velocity=0.0, effort=0.0):
    return SimpleNamespace(position=position, velocity=velocity, effort=effort)


def _make_snapshot(joint_names, joint_values=None):
    """Build a minimal snapshot with the given joint names."""
    if joint_values is None:
        joint_values = {name: float(i) for i, name in enumerate(joint_names)}
    joints = {name: _make_joint(position=val) for name, val in joint_values.items()}
    return SimpleNamespace(joints=joints)


def _make_cfg(metadata=None, embodiment=None):
    """Build a minimal cfg namespace for ObservationBuilder."""
    return SimpleNamespace(
        metadata=metadata,
        input_features={},
        embodiment=embodiment,
    )


class TestGetJointStateEmbodiment:
    """_get_joint_state should use embodiment-derived canonical arm joints."""

    def test_panda_embodiment_uses_panda_joint_names(self):
        panda_cfg = get_joint_config("dual_panda_wall")
        cfg = _make_cfg(embodiment=panda_cfg)
        builder = ObservationBuilder(cfg)
        builder.state_spec.include_joint_positions = True

        panda_joints = panda_cfg.canonical_arm_joints()
        snapshot = _make_snapshot(panda_joints)

        result = builder._get_joint_state(snapshot)
        assert len(result) == 14

    def test_fr3_embodiment_uses_fr3_joint_names(self):
        fr3_cfg = get_joint_config("dual_fr3_pedestal")
        cfg = _make_cfg(embodiment=fr3_cfg)
        builder = ObservationBuilder(cfg)
        builder.state_spec.include_joint_positions = True

        fr3_joints = fr3_cfg.canonical_arm_joints()
        snapshot = _make_snapshot(fr3_joints)

        result = builder._get_joint_state(snapshot)
        assert len(result) == 14

    def test_fr3_reads_correct_values(self):
        fr3_cfg = get_joint_config("dual_fr3_pedestal")
        cfg = _make_cfg(embodiment=fr3_cfg)
        builder = ObservationBuilder(cfg)
        builder.state_spec.include_joint_positions = True

        fr3_joints = fr3_cfg.canonical_arm_joints()
        values = {name: float(i) * 0.1 for i, name in enumerate(fr3_joints)}
        snapshot = _make_snapshot(fr3_joints, values)

        result = builder._get_joint_state(snapshot)
        expected = np.array([float(i) * 0.1 for i in range(14)], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_default_embodiment_uses_panda(self):
        """When no embodiment is provided, should use Panda defaults."""
        cfg = _make_cfg()
        builder = ObservationBuilder(cfg)
        builder.state_spec.include_joint_positions = True

        panda_cfg = get_joint_config("dual_panda_wall")
        panda_joints = panda_cfg.canonical_arm_joints()
        snapshot = _make_snapshot(panda_joints)

        result = builder._get_joint_state(snapshot)
        assert len(result) == 14


class TestGetGripperStateEmbodiment:
    """_get_gripper_state should use embodiment-derived gripper joint names."""

    def test_panda_grippers_uses_panda_finger_joints(self):
        panda_cfg = get_joint_config("dual_panda_wall")
        cfg = _make_cfg(embodiment=panda_cfg)
        builder = ObservationBuilder(cfg)

        gripper_joints = (
            panda_cfg.left_panda_gripper_joints()
            + panda_cfg.right_panda_gripper_joints()
        )
        snapshot = _make_snapshot(gripper_joints)

        result = builder._get_gripper_state(snapshot)
        # Always 2 values: 1 width per side
        assert len(result) == 2

    def test_fr3_panda_grippers_uses_fr3_finger_joints(self):
        fr3_cfg = get_joint_config("dual_fr3_pedestal")
        cfg = _make_cfg(embodiment=fr3_cfg)
        builder = ObservationBuilder(cfg)

        gripper_joints = (
            fr3_cfg.left_panda_gripper_joints() + fr3_cfg.right_panda_gripper_joints()
        )
        snapshot = _make_snapshot(gripper_joints)

        result = builder._get_gripper_state(snapshot)
        # Always 2 values: 1 width per side
        assert len(result) == 2

    def test_fr3_panda_grippers_reads_correct_values(self):
        fr3_cfg = get_joint_config("dual_fr3_pedestal")
        cfg = _make_cfg(embodiment=fr3_cfg)
        builder = ObservationBuilder(cfg)

        gripper_joints = (
            fr3_cfg.left_panda_gripper_joints() + fr3_cfg.right_panda_gripper_joints()
        )
        values = {name: float(i) * 0.01 for i, name in enumerate(gripper_joints)}
        snapshot = _make_snapshot(gripper_joints, values)

        result = builder._get_gripper_state(snapshot)
        # Width = sum of both finger joints per side:
        #   left: 0.00 + 0.01 = 0.01, right: 0.02 + 0.03 = 0.05
        expected = np.array([0.01, 0.05], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_robotiq_grippers_uses_robotiq_joint_names(self):
        """When state_spec says Robotiq, should use robotiq gripper joints."""
        panda_cfg = get_joint_config("dual_panda_wall")
        cfg = _make_cfg(embodiment=panda_cfg)
        builder = ObservationBuilder(cfg)
        builder.state_spec.left_gripper = GripperType.ROBOTIQ
        builder.state_spec.right_gripper = GripperType.ROBOTIQ

        gripper_joints = (
            panda_cfg.left_robotiq_gripper_joints()
            + panda_cfg.right_robotiq_gripper_joints()
        )
        snapshot = _make_snapshot(gripper_joints)

        result = builder._get_gripper_state(snapshot)
        # Always 2 values: 1 width per side
        assert len(result) == 2

    def test_default_embodiment_uses_panda_finger_joints(self):
        """When no embodiment is provided, should use Panda defaults."""
        cfg = _make_cfg()
        builder = ObservationBuilder(cfg)

        panda_cfg = get_joint_config("dual_panda_wall")
        gripper_joints = (
            panda_cfg.left_panda_gripper_joints()
            + panda_cfg.right_panda_gripper_joints()
        )
        snapshot = _make_snapshot(gripper_joints)

        result = builder._get_gripper_state(snapshot)
        # Always 2 values: 1 width per side
        assert len(result) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
