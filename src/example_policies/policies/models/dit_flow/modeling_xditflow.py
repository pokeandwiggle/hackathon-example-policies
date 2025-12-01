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
"""XDiTFlow - Extended DiTFlow model with additional loss functions."""

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.policies.pretrained import PreTrainedPolicy

from ...factory import register_policy
from ...losses.focal_loss import FocalTerminationLoss
from ...losses.pose_loss import IntegratedDeltaPoseLoss
from .configuration_xditflow import XDiTFlowConfig
from .modeling_dit_flow import DiTFlowModel, DiTFlowPolicy


@register_policy(name="xditflow")
class XDiTFlowPolicy(DiTFlowPolicy):
    """
    XDiTFlow Policy with custom loss functions.

    This policy extends the base DiTFlowPolicy to use XDiTFlowModel which
    supports additional loss functions while keeping the original implementation pure.
    """

    config_class = XDiTFlowConfig
    name = "xditflow"

    def __init__(
        self,
        config: XDiTFlowConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        """
        Args:
            config: XDiTFlow policy configuration class instance.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        # Call PreTrainedPolicy.__init__ directly to avoid DiTFlowPolicy.__init__
        PreTrainedPolicy.__init__(self, config)

        config.validate_features()
        self.config = config

        # Set up normalization (same as base class)
        from lerobot.policies.normalize import Normalize, Unnormalize

        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        # Use the custom model instead of the base model
        self.dit_flow = XDiTFlowModel(config, self.unnormalize_outputs)

        self.reset()

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        batch = self.normalize_targets(batch)
        loss, loss_dict = self.dit_flow.compute_loss(batch)
        return loss, loss_dict


class XDiTFlowModel(DiTFlowModel):
    """
    XDiTFlow Model with custom loss functions.

    This model extends the base DiTFlowModel to add support for:
    - IntegratedDeltaPoseLoss: Quaternion-aware pose loss for delta actions
    - FocalTerminationLoss: Focal loss for binary termination signals

    The custom losses are conditionally activated based on their weights in the config.
    """

    def __init__(self, config: XDiTFlowConfig, action_unnormalizer=None):
        # Call parent __init__ to set up the base model
        super().__init__(config)

        # Initialize custom loss modules if their weights are > 0
        if config.integrated_so3_loss_weight > 0.0:
            self.integrated_so3_loss = IntegratedDeltaPoseLoss()

        if config.termination_focal_loss_weight > 0.0:
            self.termination_focal_loss = FocalTerminationLoss(
                focal_idx=config.termination_focal_loss_index
            )
        self.unnormalize_outputs = action_unnormalizer

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """
        Extended loss computation that adds custom losses to the base MSE loss.

        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }

        Returns:
            Combined loss scalar = MSE loss + weighted custom losses
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
        noise = self.velocity_net.sample_noise(trajectory.shape[0], trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = self.noise_distribution.sample((trajectory.shape[0],)).to(
            trajectory.device
        )
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = (1 - timesteps[:, None, None]) * noise + timesteps[
            :, None, None
        ] * trajectory

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.velocity_net(
            noisy_actions=noisy_trajectory, time=timesteps, global_cond=global_cond
        )
        target = trajectory - noise
        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        # Compute mean MSE loss
        mse_loss = loss.mean()

        full_loss = mse_loss
        loss_dict = {"mse_loss": mse_loss.item()}

        ### Custom Losses ###
        # Add integrated SO3 loss if configured
        if self.config.integrated_so3_loss_weight > 0.0:
            pred_traj = noise + pred
            integrated_so3_loss_tensor, _ = self.integrated_so3_loss(
                pred_traj,
                trajectory,
            )
            # Apply same masking as MSE loss if configured
            if self.config.do_mask_loss_for_padding:
                integrated_so3_loss_tensor = (
                    integrated_so3_loss_tensor * in_episode_bound.unsqueeze(-1)
                )
            integrated_so3_loss_scalar = integrated_so3_loss_tensor.mean()
            full_loss = (
                full_loss
                + self.config.integrated_so3_loss_weight * integrated_so3_loss_scalar
            )
            loss_dict["integrated_so3_loss"] = integrated_so3_loss_scalar.item()

        # Add termination focal loss if configured
        if self.config.termination_focal_loss_weight > 0.0:
            pred_traj = noise + pred

            unnormalized_pred = self.unnormalize_outputs({"action": pred_traj})[
                "action"
            ]
            unnormalized_target = self.unnormalize_outputs({"action": trajectory})[
                "action"
            ]

            focal_term_loss_tensor, _ = self.termination_focal_loss(
                unnormalized_pred,
                unnormalized_target,
            )

            # Apply same masking as MSE loss if configured
            if self.config.do_mask_loss_for_padding:
                focal_term_loss_tensor = (
                    focal_term_loss_tensor * in_episode_bound.unsqueeze(-1)
                )
            focal_term_loss_scalar = focal_term_loss_tensor.mean()
            full_loss = (
                full_loss
                + self.config.termination_focal_loss_weight * focal_term_loss_scalar
            )
            loss_dict["focal_termination_loss"] = focal_term_loss_scalar.item()

        return full_loss, loss_dict
