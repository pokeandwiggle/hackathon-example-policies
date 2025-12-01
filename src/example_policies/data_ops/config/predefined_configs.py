"""Predefined pipeline configurations for common use cases."""

import enum

from example_policies.data_ops.config.pipeline_config import PipelineConfig
from example_policies.utils.action_order import ActionMode


class PredefinedConfigEnum(enum.Enum):
    """Predefined configuration modes."""

    ABS_DEFAULT = "abs-default"
    ABS_NOSTATE = "abs-nostate"
    ABS_FORCE = "abs-force"
    DELTA_DEFAULT = "delta-default"
    DELTA_NOSTATE = "delta-nostate"
    DELTA_FORCE = "delta-force"


# Predefined configuration registry - only differences from defaults
PREDEFINED_CONFIGS = {
    PredefinedConfigEnum.ABS_DEFAULT: {
        "include_tcp_poses": True,
        "include_last_command": False,
        "action_level": ActionMode.ABS_TCP,
    },
    PredefinedConfigEnum.ABS_NOSTATE: {
        "include_tcp_poses": False,
        "include_last_command": False,
        "action_level": ActionMode.ABS_TCP,
    },
    PredefinedConfigEnum.ABS_FORCE: {
        "include_tcp_poses": True,
        "include_last_command": False,
        "include_joint_efforts": True,
        "action_level": ActionMode.ABS_TCP,
    },
    PredefinedConfigEnum.DELTA_DEFAULT: {
        "include_tcp_poses": True,
        "include_last_command": True,
        "action_level": ActionMode.DELTA_TCP,
    },
    PredefinedConfigEnum.DELTA_NOSTATE: {
        "include_tcp_poses": False,
        "include_last_command": False,
        "action_level": ActionMode.DELTA_TCP,
    },
    PredefinedConfigEnum.DELTA_FORCE: {
        "include_tcp_poses": True,
        "include_last_command": True,
        "include_joint_efforts": True,
        "action_level": ActionMode.DELTA_TCP,
    },
}


def create_pipeline_config(
    mode: PredefinedConfigEnum, task_name: str
) -> PipelineConfig:
    """Create pipeline config from predefined mode.

    Args:
        mode: Predefined configuration mode
        task_name: Task name for the dataset

    Returns:
        PipelineConfig with settings for the specified mode

    Raises:
        ValueError: If mode is not in PREDEFINED_CONFIGS
    """
    if mode not in PREDEFINED_CONFIGS:
        raise ValueError(f"Unknown mode: {mode}")

    config_kwargs = {
        "task_name": task_name,
    }

    # Apply mode-specific overrides
    config_kwargs.update(PREDEFINED_CONFIGS[mode])

    return PipelineConfig(**config_kwargs)
