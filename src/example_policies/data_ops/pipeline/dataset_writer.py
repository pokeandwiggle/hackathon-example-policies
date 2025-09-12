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

import pathlib
from typing import Dict

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ..config import pipeline_config
from ..config.dataset_type import DatasetType
from .frame_assembler import FrameAssembler
from .frame_parser import FrameParser
from .frame_targeter import FrameTargeter
from .message_buffer import MessageBuffer
from .video_decode_buffer import VideoDecodeBuffer


class DatasetWriter:
    """Manages one primary and multiple secondary LeRobot datasets."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        features: dict,
        cfg: pipeline_config.PipelineConfig,
    ):
        self.cfg = cfg
        self.frame_parser = FrameParser(cfg)
        self.video_decode_buffer = VideoDecodeBuffer(cfg)
        self.frame_targeter = FrameTargeter(cfg)
        self.frame_assembler = FrameAssembler(cfg)
        self.datasets: Dict[DatasetType, LeRobotDataset] = {}
        self.dataset_frame_counter: Dict[DatasetType, int] = {
            DatasetType.MAIN: 0,
            DatasetType.PAUSE: 0,
            DatasetType.NO_SPEED_BOOST: 0,
        }

        # Create the main dataset
        self.datasets[DatasetType.MAIN] = LeRobotDataset.create(
            repo_id="local_only",
            fps=30,
            root=output_dir,
            use_videos=True,
            image_writer_threads=16,
            image_writer_processes=8,
            features=features,
        )

        # Conditionally create secondary datasets
        if self.cfg.save_pauses:
            self.datasets[DatasetType.PAUSE] = self._create_secondary_dataset(
                output_dir, features, DatasetType.PAUSE.value
            )

        if self.cfg.save_normal:
            self.datasets[DatasetType.NO_SPEED_BOOST] = self._create_secondary_dataset(
                output_dir, features, DatasetType.NO_SPEED_BOOST.value
            )

    def reset(self):
        """Resets the dataset manager to its initial state."""
        self.dataset_frame_counter = {
            DatasetType.MAIN: 0,
            DatasetType.PAUSE: 0,
            DatasetType.NO_SPEED_BOOST: 0,
        }
        self.frame_parser.reset()
        self.frame_targeter.reset()
        self.frame_assembler.reset()

    def _create_secondary_dataset(
        self, output_dir: pathlib.Path, features: dict, suffix: str
    ) -> LeRobotDataset:
        """Helper to create a secondary dataset with a specific suffix."""
        return LeRobotDataset.create(
            repo_id="local_only",
            fps=30,
            root=output_dir.with_name(f"{output_dir.name}_{suffix}"),
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=2,
            features=features,
        )

    def add_frame(self, msg_buffer: MessageBuffer) -> bool:

        for synced_buffer, image_dict in self.video_decode_buffer.add_and_decode(
            msg_buffer
        ):
            self.process_frame(synced_buffer, image_dict)

    def process_frame(self, msg_buffer: MessageBuffer, image_dict: dict) -> bool:
        """Adds a frame to the appropriate dataset(s) based on its classification."""
        target_datasets = self.frame_targeter.determine_targets(
            msg_buffer, self.frame_parser
        )

        frame = None
        performed_save = False

        for target in target_datasets:
            if target in self.datasets:
                # Lazily parse and assemble the frame only if needed
                if frame is None:
                    frame_dict = self.frame_parser.parse_frame(msg_buffer)
                    frame_dict.update(image_dict)
                    frame_dict = self.frame_assembler.assemble(frame_dict)

                self.datasets[target].add_frame(frame_dict, task=self.cfg.task_name)
                self.dataset_frame_counter[target] += 1
                performed_save = True
        return performed_save

    def save_episode(self, episode_idx: int):
        """Saves the current episode for all managed datasets."""
        performed_save = False
        for dataset_type, dataset in self.datasets.items():
            if self.dataset_frame_counter[dataset_type] > 0:
                dataset.save_episode()
                performed_save = True
            self.dataset_frame_counter[dataset_type] = 0
            print(f"Saved episode {episode_idx} for {dataset_type.value} dataset.")
        self.reset()
        return performed_save

    def _clear_episode(self):
        "LeRobot Clear Episode Buffer is Buggy. DO NOT USE."
        """Clears the current episode for all managed datasets."""
        for dataset in self.datasets.values():
            dataset.clear_episode_buffer()
        self.reset()
