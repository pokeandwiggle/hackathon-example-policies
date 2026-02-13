from typing import List, Optional

from example_policies.data_ops.config.pipeline_config import GripperType, PipelineConfig

# --- Constants for Joint Parsing (defined once for performance) ---
LEFT_ARM = [f"panda_left_joint{i}" for i in range(1, 8)]
RIGHT_ARM = [f"panda_right_joint{i}" for i in range(1, 8)]
_LEFT_PANDA_GRIPPER = [f"panda_left_finger_joint{i}" for i in range(1, 3)]
_RIGHT_PANDA_GRIPPER = [f"panda_right_finger_joint{i}" for i in range(1, 3)]

_ROBOTIQ_JOINTS = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]

_LEFT_ROBOTIQ_GRIPPER = ["panda_left_" + joint for joint in _ROBOTIQ_JOINTS]

_RIGHT_ROBOTIQ_GRIPPER = ["panda_right_" + joint for joint in _ROBOTIQ_JOINTS]

CANONICAL_ARM_JOINTS = LEFT_ARM + RIGHT_ARM
ARM_JOINT_COUNT = len(LEFT_ARM) + len(RIGHT_ARM)  # Should be 14


def create_joint_order(cfg: PipelineConfig):
    joint_order = CANONICAL_ARM_JOINTS.copy()

    if cfg.left_gripper == GripperType.PANDA:
        joint_order += _LEFT_PANDA_GRIPPER
    elif cfg.left_gripper == GripperType.ROBOTIQ:
        joint_order += _LEFT_ROBOTIQ_GRIPPER
    else:
        raise ValueError(f"Unsupported left gripper type: {cfg.left_gripper}")

    if cfg.right_gripper == GripperType.PANDA:
        joint_order += _RIGHT_PANDA_GRIPPER
    elif cfg.right_gripper == GripperType.ROBOTIQ:
        joint_order += _RIGHT_ROBOTIQ_GRIPPER
    else:
        raise ValueError(f"Unsupported right gripper type: {cfg.right_gripper}")
    return joint_order


def _joint_reorder_indices(
    cfg: PipelineConfig, names: list[str], joint_order: Optional[List[str]] = None
) -> List[int]:
    if not joint_order:
        joint_order = create_joint_order(cfg)

    # Create a mapping from the message's joint order to our canonical order.
    name_to_idx = {name: i for i, name in enumerate(names)}
    reorder_indices = [name_to_idx[name] for name in joint_order]
    return reorder_indices
