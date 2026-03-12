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

"""Shared Robotiq 2F-85 gripper conversion utilities.

These constants and helpers are used by both the data pipeline
(state_assembler) and the deployment path (observation_builder).
"""

import math

# Robotiq 2F-85 joint-to-width conversion constants.
# Matches the inverse of the C++ controller formula:
#   position = (1 - (width - min_width) / range) * kGripperClosedPosition
ROBOTIQ_CLOSED_POSITION_RAD = 0.7929
ROBOTIQ_MAX_WIDTH_M = 0.085


def robotiq_width_from_knuckle(position: float) -> float:
    """Convert Robotiq left_knuckle_joint position (rad) to width (m).

    The raw inverse-kinematics mapping can produce values slightly outside the
    physical range due to sensor noise or calibration drift; we clamp to
    [0.0, ROBOTIQ_MAX_WIDTH_M] and treat NaNs as fully closed (0.0).
    """
    width = (1.0 - position / ROBOTIQ_CLOSED_POSITION_RAD) * ROBOTIQ_MAX_WIDTH_M

    if math.isnan(width):
        return 0.0

    return max(0.0, min(width, ROBOTIQ_MAX_WIDTH_M))
