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

"""Episode quality filters for the synced conversion pipeline.

Filters analyze frame sequences to assess episode quality and optionally
trim leading/trailing pauses. Available filters:

- **PauseFilter**: Trims leading pauses, flags mid-episode pauses.
- **GripperToggleFilter**: Detects rapid gripper on/off toggling.
- **GripperWhileMovingFilter**: Detects gripper commands during arm motion.

Episodes whose quality falls below ``FilterConfig.min_quality`` are
excluded from the dataset entirely.

Usage::

    from example_policies.data_ops.filters import FilterConfig

    filter_config = FilterConfig(
        trim_leading_pauses=True,
        full_cycle_threshold_s=1.3,
        min_change_interval_s=0.65,
    )

    result = convert_episodes_synced(
        ...,
        filter_config=filter_config,
    )
"""

from .base import EpisodeFilterResult, FilterEvent, FrameFilterData
from .filter_pipeline import FilterConfig, FilterPipeline, create_filter_pipeline

__all__ = [
    "FilterConfig",
    "FilterPipeline",
    "EpisodeFilterResult",
    "FilterEvent",
    "FrameFilterData",
    "create_filter_pipeline",
]
