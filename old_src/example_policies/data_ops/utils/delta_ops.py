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

import numpy as np

from example_policies import data_constants as dc

from .geometric import quaternion_to_delta_axis_angle


def pos_quat_delta(last_pos_quat, des_pos_quat):
    pos_idx = dc.ACTION_ARRAY_POS_IDXS
    quat_idx = dc.ACTION_ARRAY_QUAT_IDXS
    axis_angle_idx = dc.ACTION_ARRAY_ROT_IDXS

    delta = np.zeros(6, dtype=np.float32)
    delta[pos_idx] = des_pos_quat[pos_idx] - last_pos_quat[pos_idx]
    delta[axis_angle_idx] = quaternion_to_delta_axis_angle(
        last_pos_quat[quat_idx], des_pos_quat[quat_idx]
    )
    return delta


def joint_delta(last_joint, des_joint):
    return des_joint - last_joint
