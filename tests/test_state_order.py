"""Tests for state_order joint ordering with embodiment configuration."""

import sys
from types import SimpleNamespace

import pytest

from example_policies.utils.embodiment import get_joint_config
from example_policies.utils.state_builder import GripperType
from example_policies.utils.state_order import (
    create_joint_order,
    _joint_reorder_indices,
)


def _make_cfg(left_gripper=GripperType.PANDA, right_gripper=GripperType.PANDA):
    return SimpleNamespace(left_gripper=left_gripper, right_gripper=right_gripper)


class TestCreateJointOrder:
    """Tests for create_joint_order with embodiment config."""

    def test_panda_both_panda_grippers(self):
        cfg = _make_cfg()
        emb = get_joint_config("dual_panda_wall")
        order = create_joint_order(cfg, emb)
        assert len(order) == 14 + 2 + 2  # 14 arm + 2 left gripper + 2 right gripper
        assert order[:7] == [f"panda_left_joint{i}" for i in range(1, 8)]
        assert order[7:14] == [f"panda_right_joint{i}" for i in range(1, 8)]
        assert order[14] == "panda_left_finger_joint1"
        assert order[15] == "panda_left_finger_joint2"
        assert order[16] == "panda_right_finger_joint1"
        assert order[17] == "panda_right_finger_joint2"

    def test_fr3_both_robotiq_grippers(self):
        cfg = _make_cfg(GripperType.ROBOTIQ, GripperType.ROBOTIQ)
        emb = get_joint_config("dual_fr3_pedestal")
        order = create_joint_order(cfg, emb)
        assert len(order) == 14 + 1 + 1
        assert order[:7] == [f"left_fr3v2_joint{i}" for i in range(1, 8)]
        assert order[7:14] == [f"right_fr3v2_joint{i}" for i in range(1, 8)]
        assert order[14] == "left_robotiq_85_left_knuckle_joint"
        assert order[15] == "right_robotiq_85_left_knuckle_joint"

    def test_backward_compat_without_embodiment(self):
        """Calling without embodiment config defaults to Panda."""
        cfg = _make_cfg()
        order = create_joint_order(cfg)
        assert order[:7] == [f"panda_left_joint{i}" for i in range(1, 8)]
        assert order[7:14] == [f"panda_right_joint{i}" for i in range(1, 8)]


class TestJointReorderIndices:
    """Tests for _joint_reorder_indices with embodiment config."""

    def test_panda_reorder(self):
        cfg = _make_cfg()
        emb = get_joint_config("dual_panda_wall")
        names = (
            [f"panda_right_joint{i}" for i in range(1, 8)]
            + [f"panda_left_joint{i}" for i in range(1, 8)]
            + ["panda_right_finger_joint1", "panda_right_finger_joint2"]
            + ["panda_left_finger_joint1", "panda_left_finger_joint2"]
        )
        indices = _joint_reorder_indices(cfg, names, embodiment=emb)
        joint_order = create_joint_order(cfg, emb)
        reordered = [names[i] for i in indices]
        assert reordered == joint_order

    def test_fr3_reorder(self):
        cfg = _make_cfg(GripperType.ROBOTIQ, GripperType.ROBOTIQ)
        emb = get_joint_config("dual_fr3_pedestal")
        joint_order = create_joint_order(cfg, emb)
        import random

        shuffled = joint_order.copy()
        random.Random(42).shuffle(shuffled)
        indices = _joint_reorder_indices(cfg, shuffled, embodiment=emb)
        reordered = [shuffled[i] for i in indices]
        assert reordered == joint_order

    def test_backward_compat_without_embodiment(self):
        cfg = _make_cfg()
        names = (
            [f"panda_left_joint{i}" for i in range(1, 8)]
            + [f"panda_right_joint{i}" for i in range(1, 8)]
            + ["panda_left_finger_joint1", "panda_left_finger_joint2"]
            + ["panda_right_finger_joint1", "panda_right_finger_joint2"]
        )
        indices = _joint_reorder_indices(cfg, names)
        reordered = [names[i] for i in indices]
        expected_order = create_joint_order(cfg)
        assert reordered == expected_order


class TestArmJointCount:
    def test_arm_joint_count_is_14(self):
        for name in ("dual_panda_wall", "dual_panda_table", "dual_fr3_pedestal"):
            emb = get_joint_config(name)
            assert emb.arm_joint_count == 14


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
