"""Utilities for action order and indexing in policy models."""

from enum import Enum

from .constants import ACTION

# =============================================================================
# Action Mode Constants
# =============================================================================


class ActionMode(Enum):
    """Unified action mode enum.

    Note: TCP and JOINT refer to absolute positions (legacy: ABS_TCP, ABS_JOINT)
            DELTA_TCP and DELTA_JOINT refer to delta/relative positions
            TELEOP is an alias for TCP used in teleoperation contexts
    """

    TCP = "tcp"  # Absolute TCP pose (formerly abs_tcp)
    DELTA_TCP = "delta_tcp"
    JOINT = "joint"  # Absolute joint positions (formerly abs_joint)
    DELTA_JOINT = "delta_joint"
    TELEOP = "teleop"  # Alias for TCP in teleoperation contexts

    # Legacy aliases for backward compatibility
    ABS_TCP = "tcp"
    ABS_JOINT = "joint"

    @staticmethod
    def parse_action_mode(cfg):
        action_shape = cfg.output_features[ACTION].shape[0]

        # Fallback for early legacy models
        if not getattr(cfg, "metadata", None):
            action_mode = ActionMode.DELTA_TCP if action_shape == 14 else ActionMode.TCP
            return action_mode

        names: list[str] = cfg.metadata["features"][ACTION]["names"]

        if any("delta_tcp" in n for n in names):
            action_mode = ActionMode.DELTA_TCP
        elif any(n.startswith("tcp_") for n in names):
            action_mode = ActionMode.TCP
        elif any("delta_joint" in n for n in names):
            action_mode = ActionMode.DELTA_JOINT
        elif any(n.startswith("joint_") for n in names):
            action_mode = ActionMode.JOINT
        else:
            # Fallback heuristic
            action_mode = ActionMode.DELTA_TCP if action_shape == 14 else ActionMode.TCP
        return action_mode

    @staticmethod
    def get_absolute_mode(action_mode: "ActionMode") -> "ActionMode":
        """Get the absolute variant of the given action mode."""
        if action_mode in (ActionMode.DELTA_TCP, ActionMode.TCP, ActionMode.TELEOP):
            return ActionMode.TCP
        elif action_mode in (ActionMode.DELTA_JOINT, ActionMode.JOINT):
            return ActionMode.JOINT
        else:
            raise RuntimeError(f"Unknown action mode: {action_mode}")


# =============================================================================
# Action and State Array Indices
# =============================================================================

# Single-arm action indices (7-element: pos + quat) or (6-element: pos + rot)
ACTION_ARRAY_POS_IDXS = slice(0, 3)
ACTION_ARRAY_QUAT_IDXS = slice(3, 7)
ACTION_ARRAY_ROT_IDXS = slice(3, 6)

# Dual-arm absolute pose indices (16-element: left_pos + left_quat + right_pos + right_quat + grippers)
DUAL_ABS_LEFT_POS_IDXS = slice(0, 3)
DUAL_ABS_LEFT_QUAT_IDXS = slice(3, 7)
LEFT_ARM = slice(DUAL_ABS_LEFT_POS_IDXS.start, DUAL_ABS_LEFT_QUAT_IDXS.stop)

DUAL_ABS_RIGHT_POS_IDXS = slice(7, 10)
DUAL_ABS_RIGHT_QUAT_IDXS = slice(10, 14)
RIGHT_ARM = slice(DUAL_ABS_RIGHT_POS_IDXS.start, DUAL_ABS_RIGHT_QUAT_IDXS.stop)

# Dual-arm delta pose indices (14-element: left_dpos + left_drot + right_dpos + right_drot + grippers)
DUAL_DELTA_LEFT_POS_IDXS = slice(0, 3)
DUAL_DELTA_LEFT_ROT_IDXS = slice(3, 6)
DUAL_DELTA_RIGHT_POS_IDXS = slice(6, 9)
DUAL_DELTA_RIGHT_ROT_IDXS = slice(9, 12)


def GET_LEFT_GRIPPER_IDX(action_mode: ActionMode) -> int:
    if action_mode == ActionMode.DELTA_TCP:
        return DUAL_DELTA_RIGHT_ROT_IDXS.stop
    else:
        return DUAL_ABS_RIGHT_QUAT_IDXS.stop


def GET_RIGHT_GRIPPER_IDX(action_mode: ActionMode) -> int:
    return GET_LEFT_GRIPPER_IDX(action_mode) + 1


def GET_TERMINATION_IDX(action_mode: ActionMode) -> int:
    return GET_RIGHT_GRIPPER_IDX(action_mode) + 1
