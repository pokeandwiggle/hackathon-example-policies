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

ACTION_ARRAY_POS_IDXS = slice(0, 3)
ACTION_ARRAY_QUAT_IDXS = slice(3, 7)
ACTION_ARRAY_ROT_IDXS = slice(3, 6)


DUAL_LEFT_POS_IDXS = slice(0, 3)
DUAL_LEFT_QUAT_IDXS = slice(3, 7)
LEFT_ARM = slice(DUAL_LEFT_POS_IDXS.start, DUAL_LEFT_QUAT_IDXS.stop)

DUAL_RIGHT_POS_IDXS = slice(7, 10)
DUAL_RIGHT_QUAT_IDXS = slice(10, 14)
RIGHT_ARM = slice(DUAL_RIGHT_POS_IDXS.start, DUAL_RIGHT_QUAT_IDXS.stop)

LEFT_GRIPPER_IDX = -3
RIGHT_GRIPPER_IDX = -2

DUAL_DELTA_LEFT_POS_IDXS = slice(0, 3)
DUAL_DELTA_LEFT_ROT_IDXS = slice(3, 6)
DUAL_DELTA_RIGHT_POS_IDXS = slice(6, 9)
DUAL_DELTA_RIGHT_ROT_IDXS = slice(9, 12)
