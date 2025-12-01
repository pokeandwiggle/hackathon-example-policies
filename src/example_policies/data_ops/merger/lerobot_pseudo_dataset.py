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

import pandas as pd

from ...utils.constants import EPISODE_DIR, VIDEO_DIR
from .meta_manager import MetaManager


class LerobotPseudoDataset:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root

        self.meta = MetaManager()

        self.task_2_id: Dict[str, int] = {}
        self.id_2_task: Dict[int, str] = {}

        self.video_paths = []

        self.read_meta_info()
        self.build_task_map()
        self.read_video_paths()

    def read_meta_info(self):
        self.meta.load_from_files(self.root)

    def build_task_map(self):
        for task in self.meta.tasks:
            name = task["task"]
            idx = task["task_index"]
            self.task_2_id[name] = idx
            self.id_2_task[idx] = name

    def read_video_paths(self):
        video_path = self.root / VIDEO_DIR
        self.video_paths = [p for p in video_path.iterdir() if p.is_dir()]

    def read_episode_parquet(self, ep_idx: int):
        return pd.read_parquet(
            self.root / EPISODE_DIR / f"episode_{ep_idx:06d}.parquet"
        )
