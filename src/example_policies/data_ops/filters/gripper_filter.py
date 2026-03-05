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

"""Gripper quality filters — detects problematic gripper command patterns.

Two independent checks, each implemented as a separate
:class:`EpisodeFilter`:

**GripperToggleFilter**
    Flags episodes where the gripper toggles open/closed too quickly:

    1. Any gripper command change within ``min_change_interval_s`` of
       the previous change.
    2. A full off→on→off (or on→off→on) cycle within
       ``full_cycle_threshold_s``.

**GripperWhileMovingFilter**
    Flags episodes where a gripper command change occurs while the arm
    joints are still in motion (joint-velocity norm >
    ``velocity_threshold``).

Both filters downgrade the episode quality to ``"ok"`` when triggered.
"""

from __future__ import annotations

from .base import (
    EpisodeFilter,
    FilterEvent,
    FrameFilterData,
    SingleFilterResult,
)


# ======================================================================
# Gripper toggle filter
# ======================================================================


class GripperToggleFilter(EpisodeFilter):
    """Detects episodes with rapid gripper command toggling.

    Args:
        gripper_threshold: Command value above which the gripper is
            considered "closed" (below → "open").
        full_cycle_threshold_s: Maximum duration for an off→on→off
            (or on→off→on) cycle to be flagged.
        min_change_interval_s: Minimum time between any two
            consecutive gripper state changes.
    """

    def __init__(
        self,
        gripper_threshold: float = 0.5,
        full_cycle_threshold_s: float = 1.3,
        min_change_interval_s: float = 0.65,
    ):
        self.gripper_threshold = gripper_threshold
        self.full_cycle_threshold_s = full_cycle_threshold_s
        self.min_change_interval_s = min_change_interval_s

    @property
    def name(self) -> str:
        return "gripper_toggle"

    # ------------------------------------------------------------------

    def _classify(self, value: float) -> str:
        """Return ``"closed"`` or ``"open"`` for a raw command value."""
        return "closed" if value > self.gripper_threshold else "open"

    def _check_side(
        self,
        frames: list[FrameFilterData],
        side: str,
    ) -> list[FilterEvent]:
        """Analyze one gripper side and return detected events."""
        events: list[FilterEvent] = []

        # Collect state-change timestamps
        changes: list[tuple[int, float, str]] = []  # (frame_idx, time_s, new_state)
        prev_state: str | None = None

        for f in frames:
            val = f.des_gripper_left if side == "left" else f.des_gripper_right
            state = self._classify(val)
            if prev_state is not None and state != prev_state:
                changes.append((f.index, f.timestamp_s, state))
            prev_state = state

        if len(changes) < 2:
            return events

        # --- Check 1: rapid consecutive changes ---
        for i in range(1, len(changes)):
            dt = changes[i][1] - changes[i - 1][1]
            if dt < self.min_change_interval_s:
                events.append(
                    FilterEvent(
                        filter_name=self.name,
                        frame_idx=changes[i][0],
                        timestamp_s=changes[i][1],
                        description=(
                            f"Rapid gripper change ({side}): "
                            f"{changes[i-1][2]}→{changes[i][2]} after {dt:.2f}s "
                            f"(< {self.min_change_interval_s}s)"
                        ),
                    )
                )

        # --- Check 2: full-cycle bounce (state returns within threshold) ---
        for i in range(2, len(changes)):
            cycle_dt = changes[i][1] - changes[i - 2][1]
            # State at changes[i] matches changes[i-2] because states alternate.
            if cycle_dt < self.full_cycle_threshold_s:
                events.append(
                    FilterEvent(
                        filter_name=self.name,
                        frame_idx=changes[i][0],
                        timestamp_s=changes[i][1],
                        description=(
                            f"Gripper bounce ({side}): "
                            f"{changes[i-2][2]}→{changes[i-1][2]}→{changes[i][2]} "
                            f"in {cycle_dt:.2f}s (< {self.full_cycle_threshold_s}s)"
                        ),
                    )
                )

        return events

    # ------------------------------------------------------------------

    def analyze(self, frames: list[FrameFilterData]) -> SingleFilterResult:
        events: list[FilterEvent] = []
        events.extend(self._check_side(frames, "left"))
        events.extend(self._check_side(frames, "right"))

        quality = "ok" if events else "excellent"
        frame_keep = [True] * len(frames)

        return SingleFilterResult(
            quality=quality,
            frame_keep=frame_keep,
            events=events,
        )


# ======================================================================
# Gripper-while-moving filter
# ======================================================================


class GripperWhileMovingFilter(EpisodeFilter):
    """Detects gripper command changes that occur during arm motion.

    Args:
        gripper_threshold: Command value above which the gripper is
            considered "closed".
        velocity_threshold: Minimum joint-velocity norm to consider the
            arm "in motion".
    """

    def __init__(
        self,
        gripper_threshold: float = 0.5,
        velocity_threshold: float = 0.03,
    ):
        self.gripper_threshold = gripper_threshold
        self.velocity_threshold = velocity_threshold

    @property
    def name(self) -> str:
        return "gripper_while_moving"

    # ------------------------------------------------------------------

    def _classify(self, value: float) -> str:
        return "closed" if value > self.gripper_threshold else "open"

    def _check_side(
        self,
        frames: list[FrameFilterData],
        side: str,
    ) -> list[FilterEvent]:
        events: list[FilterEvent] = []
        if len(frames) < 2:
            return events

        for i in range(1, len(frames)):
            prev_val = (
                frames[i - 1].des_gripper_left
                if side == "left"
                else frames[i - 1].des_gripper_right
            )
            curr_val = (
                frames[i].des_gripper_left
                if side == "left"
                else frames[i].des_gripper_right
            )

            prev_state = self._classify(prev_val)
            curr_state = self._classify(curr_val)

            if prev_state != curr_state:
                # Gripper command changed — check if arm is moving
                vel = frames[i].joint_velocity_norm
                if vel > self.velocity_threshold:
                    events.append(
                        FilterEvent(
                            filter_name=self.name,
                            frame_idx=frames[i].index,
                            timestamp_s=frames[i].timestamp_s,
                            description=(
                                f"Gripper change while moving ({side}): "
                                f"{prev_state}→{curr_state} at vel={vel:.4f} "
                                f"(> {self.velocity_threshold})"
                            ),
                        )
                    )

        return events

    # ------------------------------------------------------------------

    def analyze(self, frames: list[FrameFilterData]) -> SingleFilterResult:
        events: list[FilterEvent] = []
        events.extend(self._check_side(frames, "left"))
        events.extend(self._check_side(frames, "right"))

        quality = "ok" if events else "excellent"
        frame_keep = [True] * len(frames)

        return SingleFilterResult(
            quality=quality,
            frame_keep=frame_keep,
            events=events,
        )
