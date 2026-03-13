from typing import List, Optional

from example_policies.data_ops.config.pipeline_config import GripperType, PipelineConfig
from example_policies.utils.embodiment import EmbodimentJointConfig, get_joint_config

_DEFAULT_EMBODIMENT = get_joint_config("dual_panda_wall")


def create_joint_order(
    cfg: PipelineConfig,
    embodiment: Optional[EmbodimentJointConfig] = None,
) -> List[str]:
    if embodiment is None:
        embodiment = _DEFAULT_EMBODIMENT

    joint_order = embodiment.canonical_arm_joints()

    match cfg.left_gripper:
        case GripperType.PANDA:
            joint_order += embodiment.left_panda_gripper_joints()
        case GripperType.ROBOTIQ:
            joint_order += embodiment.left_robotiq_gripper_joints()
        case _:
            raise ValueError(f"Unsupported left gripper type: {cfg.left_gripper}")

    match cfg.right_gripper:
        case GripperType.PANDA:
            joint_order += embodiment.right_panda_gripper_joints()
        case GripperType.ROBOTIQ:
            joint_order += embodiment.right_robotiq_gripper_joints()
        case _:
            raise ValueError(f"Unsupported right gripper type: {cfg.right_gripper}")

    return joint_order


def _joint_reorder_indices(
    cfg: PipelineConfig,
    names: list[str],
    joint_order: Optional[List[str]] = None,
    embodiment: Optional[EmbodimentJointConfig] = None,
) -> List[int]:
    if not joint_order:
        joint_order = create_joint_order(cfg, embodiment)

    name_to_idx = {name: i for i, name in enumerate(names)}
    reorder_indices = [name_to_idx[name] for name in joint_order]
    return reorder_indices
