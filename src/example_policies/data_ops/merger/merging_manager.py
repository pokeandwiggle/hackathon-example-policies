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
import shutil

from ...utils.constants import EPISODE_DIR, META_DIR, VIDEO_DIR
from .lerobot_pseudo_dataset import LerobotPseudoDataset
from .meta_manager import MetaManager


class MergingManager:
    def __init__(self, output_path: pathlib.Path):
        self.output_path = output_path
        self.create_new_dataset_structure()

        self.episode_counter = 0
        self.dataset_episode_offset = 0
        self.dataset_frame_offset = 0

        self.task_map = {}

        self.meta_manager = MetaManager()

    def create_new_dataset_structure(self):
        if self.output_path.exists():
            raise FileExistsError(f"Output path {self.output_path} already exists.")
        self.output_path.mkdir(parents=True, exist_ok=False)
        (self.output_path / EPISODE_DIR).mkdir(parents=True, exist_ok=False)
        (self.output_path / META_DIR).mkdir(parents=True, exist_ok=False)
        (self.output_path / VIDEO_DIR).mkdir(parents=True, exist_ok=False)

    def add_dataset(self, dataset: LerobotPseudoDataset):
        self.create_video_dirs(dataset)
        self.add_tasks(dataset)
        frame_offset = 0
        for ep in dataset.meta.episodes:
            frame_offset += self.add_episode_parquet(ep, dataset)
            self.add_videos(ep, dataset)

        self.add_meta(dataset)
        self.dataset_frame_offset += frame_offset
        self.dataset_episode_offset += len(dataset.meta.episodes)

        self.check_consistency()

    def check_consistency(self):
        meta_episodes = len(self.meta_manager.episodes)
        if meta_episodes != self.dataset_episode_offset:
            raise ValueError(
                f"Meta episodes {meta_episodes} does not match dataset episode offset {self.dataset_episode_offset}"
            )

        meta_total_episodes = self.meta_manager.info["total_episodes"]
        if meta_total_episodes != self.dataset_episode_offset:
            raise ValueError(
                f"Meta total episodes {meta_total_episodes} does not match dataset frame offset {self.dataset_frame_offset}"
            )

        meta_total_frames = self.meta_manager.info["total_frames"]
        if meta_total_frames != self.dataset_frame_offset:
            raise ValueError(
                f"Meta total frames {meta_total_frames} does not match dataset frame offset {self.dataset_frame_offset}"
            )

    def add_episode_parquet(
        self, episode_meta_dict: dict, dataset: LerobotPseudoDataset
    ):
        orig_ep_idx = episode_meta_dict["episode_index"]
        ep_parquet = dataset.read_episode_parquet(orig_ep_idx)
        ep_parquet["episode_index"] += self.dataset_episode_offset
        ep_parquet["index"] += self.dataset_frame_offset

        ep_parquet["task_index"] = ep_parquet["task_index"].apply(
            lambda x: self._update_task_idx(x, dataset)
        )

        ep_parquet.to_parquet(
            self.output_path
            / EPISODE_DIR
            / f"episode_{orig_ep_idx + self.dataset_episode_offset:06d}.parquet"
        )

        return len(ep_parquet)

    def add_tasks(self, dataset: LerobotPseudoDataset):
        for task_name, task_id in dataset.task_2_id.items():
            if task_name not in self.task_map:
                new_id = len(self.task_map)
                self.task_map[task_name] = new_id

    def create_video_dirs(self, dataset: LerobotPseudoDataset):
        video_paths = dataset.video_paths
        for camera in video_paths:
            cam_name = camera.name
            (self.output_path / VIDEO_DIR / cam_name).mkdir(parents=True, exist_ok=True)

    def add_videos(self, episode_meta_dict: dict, dataset: LerobotPseudoDataset):
        orig_ep_idx = episode_meta_dict["episode_index"]
        video_paths = dataset.video_paths

        for camera in video_paths:
            cam_name = camera.name

            src = camera / f"episode_{orig_ep_idx:06d}.mp4"
            dst = (
                self.output_path
                / VIDEO_DIR
                / cam_name
                / f"episode_{orig_ep_idx + self.dataset_episode_offset:06d}.mp4"
            )

            # Copy (not move); overwrite if already present.
            shutil.copy2(src, dst)

    def add_meta(self, dataset: LerobotPseudoDataset):
        self.meta_manager.add_meta(
            dataset.meta, self.dataset_episode_offset, self.task_map
        )
        self.meta_manager.save(self.output_path)

    def _update_task_idx(self, old_id: int, dataset: LerobotPseudoDataset):
        return self.task_map[dataset.id_2_task[old_id]]
