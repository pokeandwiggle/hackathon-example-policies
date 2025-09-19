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

from enum import Enum


class ActionMode(Enum):
    ABS_TCP = "abs_tcp"
    DELTA_TCP = "delta_tcp"
    ABS_JOINT = "abs_joint"
    DELTA_JOINT = "delta_joint"

    @classmethod
    def parse_action_mode(cls, cfg):
        action_shape = cfg.output_features["action"].shape[0]

        # Fallback for early legacy models
        if not getattr(cfg, "metadata", None):
            action_mode = (
                ActionMode.DELTA_TCP if action_shape == 14 else ActionMode.ABS_TCP
            )
            return action_mode

        names: list[str] = cfg.metadata["features"]["action"]["names"]

        if any("delta_tcp" in n for n in names):
            action_mode = ActionMode.DELTA_TCP
        elif any(n.startswith("tcp_") for n in names):
            action_mode = ActionMode.ABS_TCP
        elif any("delta_joint" in n for n in names):
            action_mode = ActionMode.DELTA_JOINT
        elif any(n.startswith("joint_") for n in names):
            action_mode = ActionMode.ABS_JOINT
        else:
            # Fallback heuristic
            action_mode = (
                ActionMode.DELTA_TCP if action_shape == 14 else ActionMode.ABS_TCP
            )
        return action_mode
