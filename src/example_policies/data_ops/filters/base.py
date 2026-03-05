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

"""Base types and abstract interface for episode quality filters."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod

import numpy as np


# Quality hierarchy (higher = better).
QUALITY_ORDER: dict[str, int] = {
    "excellent": 4,
    "good": 3,
    "ok": 2,
    "bad": 1,
}


def worst_quality(*qualities: str) -> str:
    """Return the worst (lowest-ranked) quality among the inputs."""
    return min(qualities, key=lambda q: QUALITY_ORDER.get(q, 0))


def quality_meets_minimum(quality: str, min_quality: str) -> bool:
    """Return ``True`` if *quality* is at least as good as *min_quality*."""
    return QUALITY_ORDER.get(quality, 0) >= QUALITY_ORDER.get(min_quality, 0)


@dataclasses.dataclass
class FrameFilterData:
    """Lightweight per-frame data extracted for filtering.

    This is intentionally minimal — only fields required by the
    currently implemented filters are included.
    """

    index: int
    timestamp_s: float  # seconds from first synced frame
    des_gripper_left: float  # raw gripper command value
    des_gripper_right: float
    joint_velocity_norm: float  # sum of |joint velocities|
    gripper_state: np.ndarray  # actual gripper joint positions


@dataclasses.dataclass
class FilterEvent:
    """A concrete issue detected by a filter."""

    filter_name: str
    frame_idx: int
    timestamp_s: float
    description: str


@dataclasses.dataclass
class SingleFilterResult:
    """Output of a single :class:`EpisodeFilter`."""

    quality: str  # "excellent", "good", "ok", "bad"
    frame_keep: list[bool]  # per-frame keep/trim decision
    events: list[FilterEvent]

    @property
    def kept_count(self) -> int:
        return sum(self.frame_keep)

    @property
    def trimmed_count(self) -> int:
        return len(self.frame_keep) - self.kept_count


@dataclasses.dataclass
class EpisodeFilterResult:
    """Merged result of running all filters on an episode.

    Attributes:
        quality: Worst quality across all filters.
        frame_keep: Per-frame AND of all filter keep decisions.
        events: Union of events from all filters.
        filter_details: Per-filter name → SingleFilterResult.
    """

    quality: str
    frame_keep: list[bool]
    events: list[FilterEvent]
    filter_details: dict[str, SingleFilterResult]

    def should_keep(self, frame_idx: int) -> bool:
        """Whether the frame at *frame_idx* should be written."""
        return self.frame_keep[frame_idx]

    @property
    def kept_count(self) -> int:
        return sum(self.frame_keep)

    @property
    def trimmed_count(self) -> int:
        return len(self.frame_keep) - self.kept_count


class EpisodeFilter(ABC):
    """Abstract base class for episode quality filters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable filter identifier."""
        ...

    @abstractmethod
    def analyze(self, frames: list[FrameFilterData]) -> SingleFilterResult:
        """Analyze a full episode and return a filter result.

        Args:
            frames: Ordered sequence of per-frame filter data.

        Returns:
            A :class:`SingleFilterResult` with quality, per-frame keep
            decisions, and any detected events.
        """
        ...
