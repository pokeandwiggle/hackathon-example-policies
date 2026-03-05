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

"""Pause filter — trims leading pauses and flags mid-episode pauses.

Replaces the old ``FrameTargeter``-based pause handling with more
nuanced behaviour:

* **Leading pauses** (robot idle at the start of the episode) are
  trimmed — the corresponding frames are not written to the dataset.
* **Trailing pauses** (idle at the end) can optionally be trimmed.
* **Mid-episode pauses** are *kept* (frames are written) but the
  episode quality is downgraded to ``"ok"``.

Pause detection replicates the logic of :class:`FrameTargeter`:
a frame is considered *paused* when the joint-velocity norm stays
below ``pause_velocity`` **and** the gripper joints are stationary,
both for at least ``max_pause_frames`` consecutive frames.

Note that ``max_pause_frames`` is derived as
``max_pause_seconds * target_fps`` (same as ``PipelineConfig``).
"""

from __future__ import annotations

import numpy as np

from .base import (
    EpisodeFilter,
    FilterEvent,
    FrameFilterData,
    SingleFilterResult,
)


class PauseFilter(EpisodeFilter):
    """Trims leading pauses and flags mid-episode pauses.

    Args:
        max_pause_seconds: Duration (in seconds) of inactivity before a
            segment is considered a pause.
        pause_velocity: Joint-velocity norm below which the arm is
            considered stationary.
        target_fps: Output frame rate — used to convert seconds to frames.
        trim_leading: If ``True``, leading pause frames are removed.
        trim_trailing: If ``True``, trailing pause frames are removed.
        gripper_change_threshold: Minimum change in any gripper joint
            between consecutive frames to count as "moving".
    """

    def __init__(
        self,
        max_pause_seconds: float = 0.2,
        pause_velocity: float = 0.03,
        target_fps: float = 10.0,
        trim_leading: bool = True,
        trim_trailing: bool = False,
        gripper_change_threshold: float = 1e-4,
    ):
        self.max_pause_seconds = max_pause_seconds
        self.pause_velocity = pause_velocity
        self.target_fps = target_fps
        self.trim_leading = trim_leading
        self.trim_trailing = trim_trailing
        self.gripper_change_threshold = gripper_change_threshold
        self.max_pause_frames = int(max_pause_seconds * target_fps)

    @property
    def name(self) -> str:
        return "pause"

    # ------------------------------------------------------------------
    # Pause detection (mirrors FrameTargeter logic)
    # ------------------------------------------------------------------

    def _detect_pauses(self, frames: list[FrameFilterData]) -> list[bool]:
        """Return a boolean mask: ``True`` where a frame is paused.

        Uses counter-based detection identical to ``FrameTargeter``:
        counters start at ``max_pause_frames`` so the very first idle
        frame is immediately classified as paused.
        """
        n = len(frames)
        is_paused = [False] * n

        # Counters start at threshold so initial idle → immediate pause
        vel_counter = self.max_pause_frames
        grip_counter = self.max_pause_frames
        prev_gripper: np.ndarray | None = None

        for i, f in enumerate(frames):
            # --- velocity check ---
            if f.joint_velocity_norm < self.pause_velocity:
                vel_counter += 1
            else:
                vel_counter = 0

            robot_paused = vel_counter >= self.max_pause_frames

            # --- gripper stationary check ---
            if prev_gripper is None:
                gripper_still = True
            else:
                delta = np.abs(f.gripper_state - prev_gripper)
                if np.any(delta > self.gripper_change_threshold):
                    grip_counter = 0
                else:
                    grip_counter += 1
                gripper_still = grip_counter >= self.max_pause_frames
            prev_gripper = f.gripper_state.copy()

            is_paused[i] = robot_paused and gripper_still

        return is_paused

    # ------------------------------------------------------------------

    def analyze(self, frames: list[FrameFilterData]) -> SingleFilterResult:
        n = len(frames)
        if n == 0:
            return SingleFilterResult(quality="excellent", frame_keep=[], events=[])

        is_paused = self._detect_pauses(frames)
        frame_keep = [True] * n
        events: list[FilterEvent] = []

        # --- Leading pause trimming ---
        leading_end = 0
        if self.trim_leading:
            for i in range(n):
                if is_paused[i]:
                    leading_end = i + 1
                else:
                    break
            for i in range(leading_end):
                frame_keep[i] = False
            if leading_end > 0:
                events.append(
                    FilterEvent(
                        filter_name=self.name,
                        frame_idx=0,
                        timestamp_s=frames[0].timestamp_s,
                        description=(
                            f"Trimmed {leading_end} leading pause frames "
                            f"({leading_end / self.target_fps:.2f}s)"
                        ),
                    )
                )

        # --- Trailing pause trimming ---
        trailing_start = n
        if self.trim_trailing:
            for i in range(n - 1, -1, -1):
                if is_paused[i]:
                    trailing_start = i
                else:
                    break
            # Don't trim if the entire episode is paused (already handled
            # by leading trim or should be caught as zero-frame episode)
            if trailing_start < n and trailing_start > leading_end:
                for i in range(trailing_start, n):
                    frame_keep[i] = False
                events.append(
                    FilterEvent(
                        filter_name=self.name,
                        frame_idx=trailing_start,
                        timestamp_s=frames[trailing_start].timestamp_s,
                        description=(
                            f"Trimmed {n - trailing_start} trailing pause frames "
                            f"({(n - trailing_start) / self.target_fps:.2f}s)"
                        ),
                    )
                )

        # --- Mid-episode pauses → quality downgrade ---
        active_start = leading_end
        active_end = trailing_start
        mid_pause_count = sum(
            is_paused[i] for i in range(active_start, active_end)
        )

        quality = "excellent"
        if mid_pause_count > 0:
            quality = "ok"
            # Find contiguous pause segments for reporting
            in_pause = False
            pause_start_idx = 0
            for i in range(active_start, active_end):
                if is_paused[i] and not in_pause:
                    in_pause = True
                    pause_start_idx = i
                elif not is_paused[i] and in_pause:
                    in_pause = False
                    pause_len = i - pause_start_idx
                    events.append(
                        FilterEvent(
                            filter_name=self.name,
                            frame_idx=pause_start_idx,
                            timestamp_s=frames[pause_start_idx].timestamp_s,
                            description=(
                                f"Mid-episode pause: {pause_len} frames "
                                f"({pause_len / self.target_fps:.2f}s) "
                                f"at t={frames[pause_start_idx].timestamp_s:.2f}s"
                            ),
                        )
                    )
            # Close an open segment at the boundary
            if in_pause:
                pause_len = active_end - pause_start_idx
                events.append(
                    FilterEvent(
                        filter_name=self.name,
                        frame_idx=pause_start_idx,
                        timestamp_s=frames[pause_start_idx].timestamp_s,
                        description=(
                            f"Mid-episode pause: {pause_len} frames "
                            f"({pause_len / self.target_fps:.2f}s) "
                            f"at t={frames[pause_start_idx].timestamp_s:.2f}s"
                        ),
                    )
                )

        return SingleFilterResult(
            quality=quality,
            frame_keep=frame_keep,
            events=events,
        )
