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

"""Shared constants used across data_ops and robot_deploy packages."""

# =============================================================================
# Dataset Directory and File Paths
# =============================================================================

# Directory paths
EPISODE_DIR = "data/chunk-000"
META_DIR = "meta"
VIDEO_DIR = "videos/chunk-000"

# Metadata file names
BLACKLIST_FILE = "blacklist.json"
EPISODE_MAPPING_FILE = "episode_mapping.json"
STATS_FILE = "episodes_stats.jsonl"
EPISODES_FILE = "episodes.jsonl"
INFO_FILE = "info.json"
PIPELINE_CONFIG_FILE = "pipeline_config.json"
TASKS_FILE = "tasks.jsonl"

# =============================================================================
# Feature Keys
# =============================================================================

# Observation feature keys
OBSERVATION_STATE = "observation.state"
OBSERVATION_IMAGES_RGB_LEFT = "observation.images.rgb_left"
OBSERVATION_IMAGES_RGB_RIGHT = "observation.images.rgb_right"
OBSERVATION_IMAGES_RGB_STATIC = "observation.images.rgb_static"
OBSERVATION_IMAGES_DEPTH_LEFT = "observation.images.depth_left"
OBSERVATION_IMAGES_DEPTH_RIGHT = "observation.images.depth_right"

# Action feature key
ACTION = "action"
