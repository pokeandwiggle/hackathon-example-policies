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
import pathlib
from typing import List

from example_policies.data_ops.merger.lerobot_pseudo_dataset import LerobotPseudoDataset
from example_policies.data_ops.merger.merging_manager import MergingManager


def merge_datasets(dataset_paths: List[pathlib.Path], output_path: pathlib.Path):
    out_dataset = MergingManager(output_path)

    for data_path in dataset_paths:
        dataset = LerobotPseudoDataset(data_path)
        dataset.read_meta_info()
        out_dataset.add_dataset(dataset)


if __name__ == "__main__":
    # input a list of paths
    parser = argparse.ArgumentParser(description="Merge LeRobot data")
    parser.add_argument("paths", nargs="+", help="List of paths to merge")

    # Single Output path
    parser.add_argument("--output", type=str, help="Path to the output file")

    args = parser.parse_args()
    dataset_paths = [pathlib.Path(p) for p in args.paths]
    output_path = pathlib.Path(args.output)
    merge_datasets(dataset_paths, output_path)
