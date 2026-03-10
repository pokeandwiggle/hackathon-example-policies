from dataclasses import dataclass

from example_policies.utils.state_builder import GripperType

_ROBOTIQ_JOINT_STEMS = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]


@dataclass(frozen=True)
class EmbodimentJointConfig:
    left_arm_prefix: str
    right_arm_prefix: str
    joint_stem: str

    def left_arm_joints(self) -> list[str]:
        return [f"{self.left_arm_prefix}_{self.joint_stem}{i}" for i in range(1, 8)]

    def right_arm_joints(self) -> list[str]:
        return [f"{self.right_arm_prefix}_{self.joint_stem}{i}" for i in range(1, 8)]

    def canonical_arm_joints(self) -> list[str]:
        return self.left_arm_joints() + self.right_arm_joints()

    @property
    def arm_joint_count(self) -> int:
        return len(self.canonical_arm_joints())

    def left_panda_gripper_joints(self) -> list[str]:
        return [f"{self.left_arm_prefix}_finger_joint{i}" for i in range(1, 3)]

    def right_panda_gripper_joints(self) -> list[str]:
        return [f"{self.right_arm_prefix}_finger_joint{i}" for i in range(1, 3)]

    def left_robotiq_gripper_joints(self) -> list[str]:
        return [f"{self.left_arm_prefix}_{j}" for j in _ROBOTIQ_JOINT_STEMS]

    def right_robotiq_gripper_joints(self) -> list[str]:
        return [f"{self.right_arm_prefix}_{j}" for j in _ROBOTIQ_JOINT_STEMS]


_EMBODIMENT_CONFIGS: dict[str, EmbodimentJointConfig] = {
    "dual_panda_wall": EmbodimentJointConfig(
        left_arm_prefix="panda_left",
        right_arm_prefix="panda_right",
        joint_stem="joint",
    ),
    "dual_panda_table": EmbodimentJointConfig(
        left_arm_prefix="panda_left",
        right_arm_prefix="panda_right",
        joint_stem="joint",
    ),
    "dual_fr3_pedestal": EmbodimentJointConfig(
        left_arm_prefix="left",
        right_arm_prefix="right",
        joint_stem="fr3v2_joint",
    ),
}


def get_joint_config(embodiment_name: str) -> EmbodimentJointConfig:
    config = _EMBODIMENT_CONFIGS.get(embodiment_name)
    if config is None:
        known = ", ".join(sorted(_EMBODIMENT_CONFIGS.keys()))
        raise ValueError(
            f"Unknown embodiment '{embodiment_name}'. Known embodiments: {known}"
        )
    return config


_END_EFFECTOR_TO_GRIPPER: dict[str, GripperType] = {
    "franka_hand": GripperType.PANDA,
    "robotiq_2f_85": GripperType.ROBOTIQ,
}


def gripper_type_from_end_effector(end_effector: str) -> GripperType:
    gripper_type = _END_EFFECTOR_TO_GRIPPER.get(end_effector)
    if gripper_type is None:
        known = ", ".join(sorted(_END_EFFECTOR_TO_GRIPPER.keys()))
        raise ValueError(
            f"Unknown end effector '{end_effector}'. Known end effectors: {known}"
        )
    return gripper_type
