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

"""Filter pipeline — runs all enabled filters and merges results.

Provides :class:`FilterConfig` (a flat dataclass for easy notebook
configuration) and :func:`create_filter_pipeline` to build a
:class:`FilterPipeline` from a config.
"""

from __future__ import annotations

import dataclasses

from .base import (
    EpisodeFilter,
    EpisodeFilterResult,
    FrameFilterData,
    SingleFilterResult,
    worst_quality,
)
from .gripper_filter import GripperToggleFilter, GripperWhileMovingFilter
from .pause_filter import PauseFilter


@dataclasses.dataclass
class FilterConfig:
    """Flat configuration for all episode quality filters.

    Pass an instance of this to :func:`convert_episodes_synced` via
    the ``filter_config`` parameter.  When ``None`` is passed (the
    default), the legacy ``FrameTargeter``-based pause handling is
    used instead.

    Example::

        from example_policies.data_ops.filters import FilterConfig

        fc = FilterConfig(
            trim_leading_pauses=True,
            full_cycle_threshold_s=1.3,
            min_change_interval_s=0.65,
        )
    """

    # --- Pause filter ---------------------------------------------------
    enable_pause_filter: bool = True
    max_pause_seconds: float = 0.2
    pause_velocity: float = 0.03
    trim_leading_pauses: bool = True
    trim_trailing_pauses: bool = False

    # --- Gripper toggle filter ------------------------------------------
    enable_gripper_toggle_filter: bool = True
    gripper_command_threshold: float = 0.5
    full_cycle_threshold_s: float = 1.3
    min_change_interval_s: float = 0.65

    # --- Gripper-while-moving filter ------------------------------------
    enable_gripper_while_moving_filter: bool = True
    moving_velocity_threshold: float = 0.03

    # --- Quality gate ---------------------------------------------------
    # Minimum episode quality to include in the dataset.  Episodes rated
    # below this threshold are excluded entirely (not written).
    # Set to "excellent" to keep only perfect episodes, "ok" to keep
    # anything that isn't "bad", etc.
    min_quality: str = "excellent"


class FilterPipeline:
    """Runs a list of :class:`EpisodeFilter` instances and merges results.

    The merged :class:`EpisodeFilterResult`:

    * **quality** — worst quality across all filters.
    * **frame_keep** — per-frame AND of all filter keep decisions.
    * **events** — union of all filter events (sorted by timestamp).
    """

    def __init__(self, filters: list[EpisodeFilter]):
        self.filters = filters

    def run(self, frames: list[FrameFilterData]) -> EpisodeFilterResult:
        """Run every filter and return the merged result."""
        n = len(frames)

        if not self.filters:
            return EpisodeFilterResult(
                quality="excellent",
                frame_keep=[True] * n,
                events=[],
                filter_details={},
            )

        results: dict[str, SingleFilterResult] = {}
        for filt in self.filters:
            results[filt.name] = filt.analyze(frames)

        # Merge quality (worst wins)
        merged_quality = worst_quality(
            *(r.quality for r in results.values())
        )

        # Merge frame_keep (AND)
        merged_keep = [True] * n
        for r in results.values():
            for i in range(n):
                if not r.frame_keep[i]:
                    merged_keep[i] = False

        # Merge events (sorted by timestamp, then frame index)
        all_events = []
        for r in results.values():
            all_events.extend(r.events)
        all_events.sort(key=lambda e: (e.timestamp_s, e.frame_idx))

        return EpisodeFilterResult(
            quality=merged_quality,
            frame_keep=merged_keep,
            events=all_events,
            filter_details=results,
        )


def create_filter_pipeline(
    config: FilterConfig,
    target_fps: float,
) -> FilterPipeline:
    """Build a :class:`FilterPipeline` from a :class:`FilterConfig`.

    Args:
        config: Flat filter configuration.
        target_fps: Output frame rate (needed for pause-frame conversion).
    """
    filters: list[EpisodeFilter] = []

    if config.enable_pause_filter:
        filters.append(
            PauseFilter(
                max_pause_seconds=config.max_pause_seconds,
                pause_velocity=config.pause_velocity,
                target_fps=target_fps,
                trim_leading=config.trim_leading_pauses,
                trim_trailing=config.trim_trailing_pauses,
            )
        )

    if config.enable_gripper_toggle_filter:
        filters.append(
            GripperToggleFilter(
                gripper_threshold=config.gripper_command_threshold,
                full_cycle_threshold_s=config.full_cycle_threshold_s,
                min_change_interval_s=config.min_change_interval_s,
            )
        )

    if config.enable_gripper_while_moving_filter:
        filters.append(
            GripperWhileMovingFilter(
                gripper_threshold=config.gripper_command_threshold,
                velocity_threshold=config.moving_velocity_threshold,
            )
        )

    return FilterPipeline(filters)
