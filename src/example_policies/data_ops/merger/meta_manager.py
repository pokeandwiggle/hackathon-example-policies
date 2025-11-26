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

from __future__ import annotations

import json
import pathlib
from typing import List

from . import constants as c


class MetaManager:
    def __init__(self):
        self.blacklist: List[int] = []
        self.episode_mapping: dict = {}
        self.stats: List[dict] = []
        self.episodes: List[dict] = []
        self.pipeline_config: dict = {}
        self.tasks: List[dict] = []

        self.info: dict = {}

    def load_from_files(self, dataset_dir: pathlib.Path):
        meta_path = dataset_dir / c.META_DIR
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta directory {meta_path} does not exist.")

        self.blacklist = load_json(meta_path / c.BLACKLIST_FILE)
        self.episode_mapping = load_json(meta_path / c.EPISODE_MAPPING_FILE)
        self.stats = load_jsonl(meta_path / c.STATS_FILE)
        self.episodes = load_jsonl(meta_path / c.EPISODES_FILE)
        self.pipeline_config = load_json(meta_path / c.PIPELINE_CONFIG_FILE)
        self.tasks = load_jsonl(meta_path / c.TASKS_FILE)

        self.info = load_json(meta_path / c.INFO_FILE)

    def save(self, dataset_dir: pathlib.Path):
        meta_path = dataset_dir / c.META_DIR
        meta_path.mkdir(parents=True, exist_ok=True)

        write_json(self.blacklist, meta_path / c.BLACKLIST_FILE)
        write_json(self.episode_mapping, meta_path / c.EPISODE_MAPPING_FILE)
        write_jsonl(self.stats, meta_path / c.STATS_FILE)
        write_jsonl(self.episodes, meta_path / c.EPISODES_FILE)
        write_json(self.pipeline_config, meta_path / c.PIPELINE_CONFIG_FILE)
        write_jsonl(self.tasks, meta_path / c.TASKS_FILE)
        write_json(self.info, meta_path / c.INFO_FILE)

    def add_meta(self, dataset_meta: MetaManager, episode_offset: int, task_map: dict):

        self._extend_blacklist(dataset_meta, episode_offset)
        self._extend_episode_mapping(dataset_meta, episode_offset)
        self._extend_stats(dataset_meta, episode_offset)
        self._extend_episodes(dataset_meta, episode_offset)
        self._extend_pipeline_config(dataset_meta)
        self._extend_tasks(task_map)

        # Update Info Last
        self._extend_info(dataset_meta)

    def _extend_blacklist(self, dataset_meta: MetaManager, episode_offset: int):
        for bl_idx in dataset_meta.blacklist:
            self.blacklist.append(bl_idx + episode_offset)

    def _extend_episode_mapping(self, dataset_meta: MetaManager, episode_offset: int):
        for idx, path in dataset_meta.episode_mapping.items():
            idx = int(idx)
            self.episode_mapping[idx + episode_offset] = path

    def _extend_stats(self, dataset_meta: MetaManager, episode_offset: int):
        for stat_dict in dataset_meta.stats:
            stat_dict["episode_index"] += episode_offset
            self.stats.append(stat_dict)

    def _extend_episodes(self, dataset_meta: MetaManager, episode_offset: int):
        for episode in dataset_meta.episodes:
            episode["episode_index"] += episode_offset
            self.episodes.append(episode)

    def _extend_pipeline_config(self, dataset_meta: MetaManager):
        self.pipeline_config.update(dataset_meta.pipeline_config)

    def _extend_tasks(self, task_map: dict):
        task_list = []
        for task_name, task_idx in sorted(task_map.items(), key=lambda x: x[1]):
            task_list.append({"task_index": task_idx, "task": task_name})
        self.tasks = task_list

    def _extend_info(self, dataset_meta: MetaManager):
        if not self.info:
            self.info = dataset_meta.info
        else:
            self.info["total_episodes"] += dataset_meta.info["total_episodes"]
            self.info["total_frames"] += dataset_meta.info["total_frames"]
            self.info["total_videos"] += dataset_meta.info["total_videos"]
            self.info["total_tasks"] = len(self.tasks)
            self.info["splits"]["train"] = f"0:{self.info['total_episodes']}"


def load_json(path: pathlib.Path) -> dict:
    # Handle JSONL
    if path.suffix == ".jsonl":
        return load_jsonl(path)

    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path: pathlib.Path) -> dict:
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_json(data: dict, path: pathlib.Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def write_jsonl(data: List[dict], path: pathlib.Path) -> None:
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
