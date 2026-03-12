"""Tests for embodiment configuration."""

import sys

import pytest

from example_policies.utils.embodiment import (
    get_joint_config,
    gripper_type_from_end_effector,
)
from example_policies.utils.state_builder import GripperType


class TestGetJointConfig:
    """Tests for get_joint_config lookup."""

    def test_dual_panda_wall(self):
        cfg = get_joint_config("dual_panda_wall")
        assert cfg.left_arm_prefix == "panda_left"
        assert cfg.right_arm_prefix == "panda_right"
        assert cfg.joint_stem == "joint"

    def test_dual_panda_table(self):
        cfg = get_joint_config("dual_panda_table")
        assert cfg.left_arm_prefix == "panda_left"
        assert cfg.right_arm_prefix == "panda_right"
        assert cfg.joint_stem == "joint"

    def test_dual_fr3_pedestal(self):
        cfg = get_joint_config("dual_fr3_pedestal")
        assert cfg.left_arm_prefix == "left"
        assert cfg.right_arm_prefix == "right"
        assert cfg.joint_stem == "fr3v2_joint"

    def test_unknown_embodiment_raises(self):
        with pytest.raises(ValueError, match="Unknown embodiment"):
            get_joint_config("unknown_robot")


class TestEmbodimentJointConfigNames:
    """Tests for joint name generation methods."""

    def test_panda_left_arm_joints(self):
        cfg = get_joint_config("dual_panda_wall")
        expected = [f"panda_left_joint{i}" for i in range(1, 8)]
        assert cfg.left_arm_joints() == expected

    def test_panda_right_arm_joints(self):
        cfg = get_joint_config("dual_panda_wall")
        expected = [f"panda_right_joint{i}" for i in range(1, 8)]
        assert cfg.right_arm_joints() == expected

    def test_panda_canonical_arm_joints(self):
        cfg = get_joint_config("dual_panda_wall")
        expected = [f"panda_left_joint{i}" for i in range(1, 8)] + [
            f"panda_right_joint{i}" for i in range(1, 8)
        ]
        assert cfg.canonical_arm_joints() == expected

    def test_fr3_left_arm_joints(self):
        cfg = get_joint_config("dual_fr3_pedestal")
        expected = [f"left_fr3v2_joint{i}" for i in range(1, 8)]
        assert cfg.left_arm_joints() == expected

    def test_fr3_right_arm_joints(self):
        cfg = get_joint_config("dual_fr3_pedestal")
        expected = [f"right_fr3v2_joint{i}" for i in range(1, 8)]
        assert cfg.right_arm_joints() == expected

    def test_fr3_canonical_arm_joints(self):
        cfg = get_joint_config("dual_fr3_pedestal")
        expected = [f"left_fr3v2_joint{i}" for i in range(1, 8)] + [
            f"right_fr3v2_joint{i}" for i in range(1, 8)
        ]
        assert cfg.canonical_arm_joints() == expected

    def test_panda_left_panda_gripper_joints(self):
        cfg = get_joint_config("dual_panda_wall")
        expected = ["panda_left_finger_joint1", "panda_left_finger_joint2"]
        assert cfg.left_panda_gripper_joints() == expected

    def test_panda_right_panda_gripper_joints(self):
        cfg = get_joint_config("dual_panda_wall")
        expected = ["panda_right_finger_joint1", "panda_right_finger_joint2"]
        assert cfg.right_panda_gripper_joints() == expected

    def test_panda_left_robotiq_gripper_joints(self):
        cfg = get_joint_config("dual_panda_wall")
        expected = [
            "panda_left_robotiq_85_left_knuckle_joint",
        ]
        assert cfg.left_robotiq_gripper_joints() == expected

    def test_fr3_left_robotiq_gripper_joints(self):
        cfg = get_joint_config("dual_fr3_pedestal")
        expected = [
            "left_robotiq_85_left_knuckle_joint",
        ]
        assert cfg.left_robotiq_gripper_joints() == expected

    def test_fr3_right_robotiq_gripper_joints(self):
        cfg = get_joint_config("dual_fr3_pedestal")
        expected = [
            "right_robotiq_85_left_knuckle_joint",
        ]
        assert cfg.right_robotiq_gripper_joints() == expected


class TestGripperTypeFromEndEffector:
    """Tests for gripper_type_from_end_effector helper."""

    def test_franka_hand(self):
        assert gripper_type_from_end_effector("franka_hand") == GripperType.PANDA

    def test_robotiq_2f_85(self):
        assert gripper_type_from_end_effector("robotiq_2f_85") == GripperType.ROBOTIQ

    def test_unknown_end_effector_raises(self):
        with pytest.raises(ValueError, match="Unknown end effector"):
            gripper_type_from_end_effector("unknown_gripper")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
