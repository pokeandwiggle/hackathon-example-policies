# Copyright 2025 Nur Muhammad Mahi Shafiullah,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""XDiTFlow - Extended DiTFlow with additional loss function configurations."""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

from .configuration_dit_flow import DiTFlowConfig


@PreTrainedConfig.register_subclass("xditflow")
@dataclass
class XDiTFlowConfig(DiTFlowConfig):
    """XDiTFlow configuration with custom loss function parameters.

    This configuration extends the base DiTFlowConfig to add support for custom loss functions
    while keeping the original implementation pure.

    Additional Args:
        integrated_so3_loss_weight: Weight for the integrated SO3 pose loss. Set to 0.0 to disable.
            This loss applies quaternion-aware pose loss on integrated delta actions.
            Recommended starting value: 0.01
        termination_focal_loss_weight: Weight for the focal loss on termination signals. Set to 0.0 to disable.
            This loss applies focal loss to a binary termination signal within the action vector.
            Recommended starting value: 10.0
        termination_focal_loss_index: Index of the termination signal in the action vector.
            Default is -1 (last dimension).
    """

    # Custom loss configuration
    integrated_so3_loss_weight: float = 0.0
    termination_focal_loss_weight: float = 0.0
    termination_focal_loss_index: int = -1
