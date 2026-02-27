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
            UMI_DELTA_TCP uses the chunk-relative action representation with
            6D rotation (all actions relative to the observation TCP at chunk
            start).  This is set at training time by AbsTcpToChunkRelativeStep
            and used at deployment time by ActionTranslator._umi_delta_tcp.
    """

    TCP = "tcp"  # Absolute TCP pose (formerly abs_tcp)
    DELTA_TCP = "delta_tcp"
    JOINT = "joint"  # Absolute joint positions (formerly abs_joint)
    DELTA_JOINT = "delta_joint"
    TELEOP = "teleop"  # Alias for TCP in teleoperation contexts
    UMI_DELTA_TCP = "umi_delta_tcp"  # UMI-style relative actions with 6D rotation

    # Legacy aliases for backward compatibility
    ABS_TCP = "tcp"
    ABS_JOINT = "joint"

    @staticmethod
    def parse_action_mode(cfg):
        action_shape = cfg.output_features[ACTION].shape[0]

        # Check config flag first — most reliable for chunk-relative models
        # where the dataset has abs TCP names/shape but the model outputs
        # UMI-delta (20-dim) actions.
        if getattr(cfg, "use_chunk_relative_actions", False):
            return ActionMode.UMI_DELTA_TCP

        # Fallback for early legacy models
        if not getattr(cfg, "metadata", None):
            action_mode = ActionMode.DELTA_TCP if action_shape == 14 else ActionMode.TCP
            return action_mode

        names: list[str] = cfg.metadata["features"][ACTION]["names"]

        if any("umi_delta_tcp" in n for n in names):
            action_mode = ActionMode.UMI_DELTA_TCP
        elif action_shape == UMI_ACTION_DIM and any(n.startswith("tcp_") for n in names):
            # Pre-converted UMI-delta dataset: shape is 20 but metadata
            # still has tcp_* names (from the original dataset info copied
            # into the checkpoint by monkey_patch_save_checkpoint).
            action_mode = ActionMode.UMI_DELTA_TCP
        elif any("delta_tcp" in n for n in names):
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
        if action_mode in (
            ActionMode.DELTA_TCP,
            ActionMode.TCP,
            ActionMode.TELEOP,
            ActionMode.UMI_DELTA_TCP,
        ):
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

# UMI-delta 6D rotation indices (20-element: left_dpos(3) + left_rot6d(6) + right_dpos(3) + right_rot6d(6) + grippers(2))
UMI_LEFT_POS_IDXS = slice(0, 3)
UMI_LEFT_ROT6D_IDXS = slice(3, 9)
UMI_RIGHT_POS_IDXS = slice(9, 12)
UMI_RIGHT_ROT6D_IDXS = slice(12, 18)
UMI_LEFT_GRIPPER_IDX = 18
UMI_RIGHT_GRIPPER_IDX = 19
UMI_ACTION_DIM = 20

# Indices for 6D rotation features within the 20-dim UMI action (used to skip normalization)
UMI_ROTATION_FEATURE_INDICES = list(range(3, 9)) + list(range(12, 18))


def GET_LEFT_GRIPPER_IDX(action_mode: ActionMode) -> int:
    if action_mode == ActionMode.UMI_DELTA_TCP:
        return UMI_LEFT_GRIPPER_IDX
    elif action_mode == ActionMode.DELTA_TCP:
        return DUAL_DELTA_RIGHT_ROT_IDXS.stop
    else:
        return DUAL_ABS_RIGHT_QUAT_IDXS.stop


def GET_RIGHT_GRIPPER_IDX(action_mode: ActionMode) -> int:
    return GET_LEFT_GRIPPER_IDX(action_mode) + 1


def GET_TERMINATION_IDX(action_mode: ActionMode) -> int:
    return GET_RIGHT_GRIPPER_IDX(action_mode) + 1
