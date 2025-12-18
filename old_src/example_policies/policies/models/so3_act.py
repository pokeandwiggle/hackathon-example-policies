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

import torch
import torch.nn.functional as F
from lerobot.policies.act.configuration_act import ACTConfig, PreTrainedConfig
from lerobot.policies.act.modeling_act import ACT, ACTION, OBS_IMAGES, ACTPolicy

from example_policies import data_constants as dc

from ..factory import register_policy
from ..losses.pose_loss import IntegratedDeltaPoseLoss, PoseLoss


# Register with LeRobot
@PreTrainedConfig.register_subclass("so3_act")
@PreTrainedConfig.register_subclass("integrated_so3_act")
class SO3ACTConfig(ACTConfig):
    loss_pos_weight: float = 1.0
    loss_quat_weight: float = 1.0
    loss_grip_weight: float = 1.0


@register_policy(name="so3_act")
class SO3ACTPolicy(ACTPolicy):
    def __init__(
        self,
        config: SO3ACTConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__(config, dataset_stats)
        self.loss_fn = PoseLoss(
            config.loss_pos_weight,
            config.loss_quat_weight,
            config.loss_grip_weight,
        )
        self.model = SO3ACT(config)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        batch = self.normalize_targets(batch)
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        action_loss, loss_dict = self.loss_fn(batch[ACTION], actions_hat)
        action_loss = (action_loss * ~batch["action_is_pad"].unsqueeze(-1)).mean()

        loss_dict["action_loss"] = action_loss.item()
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
            mean_kld = (
                (
                    -0.5
                    * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())
                )
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = action_loss + mean_kld * self.config.kl_weight
        else:
            loss = action_loss

        return loss, loss_dict


class SO3ACT(ACT):
    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | tuple[None, None]]:
        action, vae_tuple = super().forward(batch)
        action_normalized = action.clone()

        LEFT_QUAT_IDXS = dc.DUAL_LEFT_QUAT_IDXS
        RIGHT_QUAT_IDXS = dc.DUAL_RIGHT_QUAT_IDXS

        # normalize to unit quaternions
        left_quat_pred = action[:, :, LEFT_QUAT_IDXS]
        right_quat_pred = action[:, :, RIGHT_QUAT_IDXS]

        action_normalized[:, :, LEFT_QUAT_IDXS] = F.normalize(
            left_quat_pred, p=2, dim=-1
        )
        action_normalized[:, :, RIGHT_QUAT_IDXS] = F.normalize(
            right_quat_pred, p=2, dim=-1
        )

        return action_normalized, vae_tuple


@register_policy(name="integrated_so3_act")
class IntegratedSO3ACTPolicy(SO3ACTPolicy):
    def __init__(
        self,
        config: SO3ACTConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__(config, dataset_stats)
        self.loss_fn = IntegratedDeltaPoseLoss(
            pos_weight=config.loss_pos_weight,
            quat_weight=config.loss_quat_weight,
            grip_weight=config.loss_grip_weight,
        )
        self.model = ACT(config)
