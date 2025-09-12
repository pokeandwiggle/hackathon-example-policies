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

import argparse
import json
import pathlib
import time

# Workaround for torch / lerobot bug
import numpy as np
from mcap.reader import NonSeekingReader

from example_policies.data_ops.config import argparse_pipeline_config, pipeline_config
from example_policies.data_ops.pipeline.dataset_writer import DatasetWriter
from example_policies.data_ops.pipeline.message_buffer import MessageBuffer


def convert_episodes(
    episode_dir: pathlib.Path,
    output_dir: pathlib.Path,
    config: pipeline_config.PipelineConfig,
):
    """Convert the episodes to the LeRobot dataset format."""
    features = pipeline_config.build_features(config)
    frame_buffer = MessageBuffer(config)

    dataset_manager = DatasetWriter(output_dir, features, config)

    episode_paths = list(episode_dir.rglob("*.mcap"))
    # Sort by creation date (oldest first)
    episode_paths.sort(key=lambda p: p.stat().st_ctime)

    now = time.time()

    actual_episode_counter = 0
    episode_counter_path_dict = {}

    global_start = time.time()
    frame_start = global_start
    global_frames = 0

    blacklist = []

    for ep_idx, episode_path in enumerate(episode_paths):
        print(f"Processing {episode_path}...")

        try:
            # Use MCAP reader for .mcap files
            with open(episode_path, "rb") as f:
                reader = NonSeekingReader(f, record_size_limit=None)
                seen_frames = config.subsample_offset
                saved_frames = 0

                # Iterate through messages with automatic deserialization
                for schema, channel, message in reader.iter_messages(
                    topics=frame_buffer.get_topic_names()
                ):
                    topic = channel.topic
                    msg_data = message.data
                    schema_name = schema.name

                    frame_buffer.add_msg(topic, schema_name, msg_data)

                    if not frame_buffer.is_complete():
                        continue

                    seen_frames += 1
                    global_frames += 1

                    perform_save = dataset_manager.add_frame(frame_buffer)
                    frame_buffer.reset()

                    if not perform_save:
                        continue
                    saved_frames += 1

                    # Print status every second
                    if saved_frames % config.capture_frequency == 0:
                        frame_end = time.time()
                        print(
                            f"  - Seen / Saved: {seen_frames} / {saved_frames} in {frame_end - frame_start:.2f}s |  Total Time: {frame_end - global_start:.2f}s | FPS: {global_frames / (frame_end - global_start):.2f} ",
                            end="\r",
                        )
                        frame_start = frame_end
                print()

                # Blacklist short episodes
                if saved_frames > 0:
                    print(f"Saving {episode_path} processed with {seen_frames} frames.")
                    perform_save = dataset_manager.save_episode(ep_idx)
                    if perform_save:
                        episode_counter_path_dict[actual_episode_counter] = str(
                            episode_path
                        )
                        actual_episode_counter += 1

                        if saved_frames < config.min_episode_frames:
                            print(
                                f"Episode too short ({saved_frames} frames), Adding to Blacklist."
                            )
                            blacklist.append(actual_episode_counter)
        except Exception as e:
            print(
                f"Skipping faulty file: {episode_path} due to {type(e).__name__}: {e}"
            )
            raise e
            continue

    with open(output_dir / "meta" / "episode_mapping.json", "w", encoding="utf-8") as f:
        json.dump(episode_counter_path_dict, f, indent=2)

    with open(output_dir / "meta" / "pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    with open(output_dir / "meta" / "blacklist.json", "w", encoding="utf-8") as f:
        json.dump(blacklist, f, indent=2)


def save_episode(dataset, episode_idx, pause_dataset=None):
    """Save the current episode to the dataset."""
    dataset.save_episode(episode_idx)
    if pause_dataset:
        pause_dataset.save_episode(episode_idx)


def main():
    parser = argparse.ArgumentParser(
        description="Convert ROS2 bags to LeRobot v2 dataset format with configurable features."
    )
    parser.add_argument(
        "episodes_dir",
        type=pathlib.Path,
        help="Path to the directory with rosbag2 files.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Path to save the dataset directory.",
    )

    # Add Pipeline Config fields to argparse.
    config, args = argparse_pipeline_config.parse_pipeline_config_from_args(parser)

    if not args.episodes_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.episodes_dir}")

    print(f"Converting with config: {config}")

    convert_episodes(args.episodes_dir, args.output, config)


if __name__ == "__main__":
    main()
