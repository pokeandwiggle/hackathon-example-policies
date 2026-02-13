#!/usr/bin/env python

"""Simple dataset conversion with predefined configurations.

This script provides an easy interface for converting MCAP episodes to LeRobot
dataset format using predefined configurations for common use cases.

For advanced control over all pipeline parameters, use dataset_conversion.py.
"""

import dataclasses
import pathlib

import draccus

from example_policies.data_ops.config.predefined_configs import (
    PredefinedConfigEnum,
    create_pipeline_config,
)
from example_policies.data_ops.dataset_conversion import (
    convert_episodes,
    print_conversion_result,
)
from example_policies.data_ops.utils.conversion_utils import (
    build_output_path,
    copy_blacklist,
    filter_episode_paths,
    get_sorted_episodes,
    resolve_blacklist_path,
    validate_input_dir,
    validate_output_dir,
)


@dataclasses.dataclass
class SimpleArgs:
    """Arguments for simple conversion with predefined configurations."""

    # Directory containing the raw mcap files to be processed.
    episodes_dir: pathlib.Path
    # Predefined configuration mode for processing.
    # Options: abs-default, abs-nostate, abs-force, delta-default, delta-nostate, delta-force
    mode: PredefinedConfigEnum = PredefinedConfigEnum.ABS_DEFAULT
    # The name of the task to be embedded in the converted data.
    task_name: str = "Move the Object"
    # If True, overwrite the output directory if it already exists.
    force: bool = False
    blacklist: pathlib.Path | None = None
    # Optional: Parent Directory where the lerobot dataset directory will be created.
    output_dir: pathlib.Path | None = None
    # Optional prefix for output directory name. If not specified, uses input directory name.
    name_prefix: str | None = None
    # Optional: List of operator names to ignore during conversion.
    operator_blacklist: list[str] = dataclasses.field(default_factory=list)
    # Optional: List of state names to include during conversion.
    rating_whitelist: list[str] = dataclasses.field(default_factory=list)


def main():
    """
    Simple conversion mode with predefined configurations.
    Use --help to see all available options.
    """
    config = draccus.parse(config_class=SimpleArgs)

    # Build output directory
    output_dir = build_output_path(
        config.episodes_dir,
        config.output_dir,
        config.name_prefix,
        config.mode.value,
    )

    # Validate directories
    validate_input_dir(config.episodes_dir)
    validate_output_dir(output_dir, force=config.force)

    # Resolve blacklist path
    blacklist_path = resolve_blacklist_path(config.blacklist)

    # Create pipeline config from mode
    pipeline_config = create_pipeline_config(config.mode, config.task_name)

    print("-" * 40)
    print(f"Input Dir: \t {config.episodes_dir}")
    print(f"Output Dir: \t {output_dir}")
    print(f"Mode: \t\t {config.mode.value}")
    print(f"Task: \t\t {config.task_name}")
    print("-" * 40)

    episode_paths = get_sorted_episodes(config.episodes_dir)
    episode_paths = filter_episode_paths(
        episode_paths,
        config.operator_blacklist,
        config.rating_whitelist,
    )

    result = convert_episodes(episode_paths, output_dir, pipeline_config)

    # Copy blacklist if provided
    copy_blacklist(output_dir, blacklist_path)

    print_conversion_result(result)


if __name__ == "__main__":
    main()
