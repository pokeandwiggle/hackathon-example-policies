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
from lerobot.policies.diffusion.configuration_diffusion import (
    DiffusionConfig,
    PreTrainedConfig,
)
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionModel,
    DiffusionPolicy,
)

from ..factory import register_policy
from ..losses.pose_loss import IntegratedDeltaPoseLoss, PoseLoss


# Register with LeRobot
@PreTrainedConfig.register_subclass("so3_diffusion")
@PreTrainedConfig.register_subclass("integrated_so3_diffusion")
class SO3DiffusionConfig(DiffusionConfig):
    loss_pos_weight: float = 1.0
    loss_quat_weight: float = 1.0
    loss_grip_weight: float = 1.0


@register_policy(name="so3_diffusion")
class SO3DiffusionPolicy(DiffusionPolicy):
    def __init__(
        self,
        config: SO3DiffusionConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__(config, dataset_stats)
        print("SOE3")

        self.diffusion = SO3DiffusionModel(config)


class SO3DiffusionModel(DiffusionModel):
    def __init__(self, config: SO3DiffusionConfig):
        super().__init__(config)
        self.pose_loss = PoseLoss(
            pos_weight=config.loss_pos_weight,
            quat_weight=config.loss_quat_weight,
            grip_weight=config.loss_grip_weight,
        )

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch["action"]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
            loss = F.mse_loss(pred, target, reduction="none")
        elif self.config.prediction_type == "sample":
            target = batch["action"]
            loss, loss_dict = self.pose_loss(pred, target)
        else:
            raise ValueError(
                f"Unsupported prediction type {self.config.prediction_type}"
            )

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()


@register_policy(name="integrated_so3_diffusion")
class IntegratedDeltaSO3DiffusionPolicy(SO3DiffusionPolicy):
    def __init__(
        self,
        config: SO3DiffusionConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__(config, dataset_stats)
        print("INTEGRATED")

        self.diffusion = IntegratedDeltaSO3DiffusionModel(config)


class IntegratedDeltaSO3DiffusionModel(SO3DiffusionModel):
    def __init__(self, config: SO3DiffusionConfig):
        super().__init__(config)
        self.pose_loss = IntegratedDeltaPoseLoss(
            pos_weight=config.loss_pos_weight,
            quat_weight=config.loss_quat_weight,
            grip_weight=config.loss_grip_weight,
        )
