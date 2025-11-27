#!/usr/bin/env python

# Copyright 2025 Poke & Wiggle GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Advanced dataset conversion with full pipeline configuration control.

This script provides complete control over all pipeline configuration parameters
for converting MCAP episodes to LeRobot dataset format.

For simplified conversion with predefined configurations, use simple_conversion.py.
"""

import dataclasses
import enum
import pathlib

import draccus
from mcap.reader import NonSeekingReader

from example_policies.data_ops.config import pipeline_config
from example_policies.data_ops.pipeline.episode_converter import EpisodeConverter
from example_policies.data_ops.pipeline.post_lerobot_ops import PostLerobotPipeline
from example_policies.data_ops.utils.conversion_utils import (
    get_selected_episodes,
    get_sorted_episodes,
    save_metadata,
    validate_input_dir,
)


def convert_episodes(
    episode_paths: list[pathlib.Path],
    output_dir: pathlib.Path,
    config: pipeline_config.PipelineConfig,
) -> dict:
    """Convert episodes to the LeRobot dataset format.

    Args:
        episode_paths: List of paths to .mcap episode files
        output_dir: Output directory for converted dataset
        config: Pipeline configuration

    Returns:
        Dict with keys: episode_mapping, blacklist, episodes_saved, total_time
    """
    features = pipeline_config.build_features(config)
    converter = EpisodeConverter(output_dir, config, features)
    post_pipeline = PostLerobotPipeline(config)

    for ep_idx, episode_path in enumerate(episode_paths):
        print(f"Processing {episode_path}...")

        converter.reset_episode_state()

        with open(episode_path, "rb") as f:
            reader = NonSeekingReader(f, record_size_limit=None)

            for schema, channel, message in reader.iter_messages(
                topics=converter.frame_buffer.get_topic_names()
            ):
                print(channel)
                converter.process_message(channel.topic, schema.name, message.data)

        converter.finalize_episode(ep_idx, episode_path)


    save_metadata(
        output_dir,
        converter.episode_mapping,
        converter.blacklist,
        config,
    )

    # Only run post-processing if episodes were successfully converted
    if len(converter.episode_mapping) > 0:
        post_pipeline.process_lerobot(output_dir)
    else:
        print("\nNo episodes were successfully converted.")

    return {
        "episode_mapping": converter.episode_mapping,
        "blacklist": converter.blacklist,
        "episodes_saved": len(converter.episode_mapping),
        "total_time": converter.get_total_time(),
    }


def print_conversion_result(result: dict) -> None:
    """Print conversion statistics.

    Args:
        result: Dict with episode_mapping, blacklist, episodes_saved, total_time
    """
    print("\nConversion complete!")
    print(f"  - Episodes saved: {result['episodes_saved']}")
    print(f"  - Blacklisted episodes: {len(result['blacklist'])}")
    print(f"  - Total time: {result['total_time']:.2f}s")


@dataclasses.dataclass
class ScriptArgs:
    """Arguments for advanced conversion script."""

    episodes_dir: pathlib.Path = pathlib.Path("./data")
    output: pathlib.Path = pathlib.Path("./output")


@dataclasses.dataclass
class AdvancedConfig(ScriptArgs, pipeline_config.PipelineConfig):
    """Configuration for advanced conversion.

    Inherits from ScriptArgs and PipelineConfig to include all parameters.
    """

    def to_dict(self):
        data = dataclasses.asdict(self)
        for key, value in data.items():
            if isinstance(value, pathlib.Path):
                data[key] = str(value)
            if isinstance(value, enum.Enum):
                data[key] = value.value
        return data


def main():
    """
    Advanced conversion mode with full pipeline configuration control.

    Example usage:
        python dataset_conversion.py --episodes-dir /path/to/episodes --output /path/to/output \\
            --action-level DELTA_TCP --target-fps 15 --include-joint-efforts

    Use --help to see all available options and their descriptions.
    """
    config = draccus.parse(config_class=AdvancedConfig)

    validate_input_dir(config.episodes_dir)

    print(f"Converting episodes from: {config.episodes_dir}")
    print(f"Output directory: {config.output}")
    print("Pipeline config summary:")
    print(f"  - Action level: {config.action_level}")
    print(f"  - Image resolution: {config.image_resolution}")
    print(f"  - Target FPS: {config.target_fps}")
    print(f"  - Task: {config.task_name}")

    episode_paths = get_sorted_episodes(config.episodes_dir)
    print(episode_paths)
    result = convert_episodes(episode_paths, config.output, config)
    print_conversion_result(result)


if __name__ == "__main__":
    main()
