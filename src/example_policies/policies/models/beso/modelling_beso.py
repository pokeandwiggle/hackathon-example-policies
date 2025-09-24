#!/usr/bin/env python

# Copyright 2025 Intuitive Robots Lab, KIT,
# Columbia Artificial Intelligence, Robotics Lab,
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

from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from torch import Tensor, nn

from ...factory import register_policy
from .beso_transformer_backbone import Noise_Dec_only
from .configuration_beso import BesoConfig
from .utils import append_dims, get_sigmas_exponential, make_sample_density, sample_ddim


@register_policy(name="beso")
class BesoPolicy(PreTrainedPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://huggingface.co/papers/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = BesoConfig
    name = "beso"

    def __init__(
        self,
        config: BesoConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        config.n_action_steps = 16
        self.config = config
        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_inputs = Unnormalize(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.step_counter = 0
        self.display = True
        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None
        # self.
        self.diffusion = BesoModel(config)
        self.enable_saliency_maps = False  # set this when you want gradients & overlays

        self.reset()
        self.new_traj = False

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }

        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(
                maxlen=self.config.n_obs_steps
            )

    # @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        # stack n latest observations from the queue
        batch = {
            k: torch.stack(list(self._queues[k]), dim=1)
            for k in batch
            if k in self._queues
        }
        actions = self.diffusion.generate_actions(batch)
        # TODO(rcadene): make above methods return output dictionary?
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    # @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        # in select_action(...)
        if ACTION in batch:
            batch.pop(ACTION)
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        self.step_counter += 1
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )

        batch = self.normalize_targets(batch)
        loss = self.diffusion.compute_loss(batch)
        # no output_dict so returning None
        return loss, None

    def _saliency_on_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (1, C, H, W) in [0,1] on the encoder's device. Returns (1,1,H,W) in [0,1].
        Enhancements: percentile clip + gamma to boost visibility.
        """
        enc = self.diffusion.rgb_encoder
        if isinstance(enc, nn.ModuleList):
            enc = enc[0]
        enc.eval()

        dev = next(enc.parameters()).device
        with torch.inference_mode(False):
            x = x.to(device=dev).detach().requires_grad_(True)
            feat = enc(x)
            score = (feat**2).mean()

            enc.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad.zero_()
            score.backward()

            sal = x.grad.detach().abs().max(dim=1, keepdim=True)[0]  # (1,1,H,W)

            # --- Visibility boosts ---
            # percentile clip (robust to outliers)
            flat = sal.view(-1)
            p_hi = torch.quantile(flat, 0.98).clamp_min(1e-12)
            sal = sal / p_hi
            sal = sal.clamp_(0, 1)

            # gamma < 1: brighten low/mid signals; try 0.5–0.7
            gamma = 0.6
            sal = sal.pow(gamma)

            return sal

    @staticmethod
    def _overlay_saliency(
        rgb: torch.Tensor, sal: torch.Tensor, max_alpha: float = 0.75
    ) -> np.ndarray:
        """
        rgb: (1,3,H,W) in [0,1]
        sal: (1,1,H,W) in [0,1]  — already contrast-boosted
        Returns HxWx3 numpy: bright red saliency overlaid with per-pixel alpha.
        """
        rgb_np = rgb[0].permute(1, 2, 0).detach().cpu().numpy()
        sal_np = sal[0, 0].detach().cpu().numpy()  # HxW

        # Red mask; no colormap dependency
        red = np.zeros_like(rgb_np)
        red[..., 0] = 1.0  # pure red

        # Per-pixel alpha = scaled saliency
        alpha = (sal_np * max_alpha)[..., None]  # HxWx1

        out = (1 - alpha) * rgb_np + alpha * red
        return np.clip(out, 0.0, 1.0)

    @torch.no_grad()
    def _split_img_feats_per_cam(self, img_feats: torch.Tensor) -> list[torch.Tensor]:
        """
        img_feats: (B, S, n_cam*D) -> list of length n_cam, each (B, S, D)
        """
        n_cam = self.diffusion.num_cameras
        if img_feats is None or n_cam == 0:
            return []
        D = img_feats.shape[-1] // n_cam
        return [img_feats[..., i * D : (i + 1) * D] for i in range(n_cam)]

    def modality_importance_over_time(
        self, batch: dict[str, Tensor], score_mode: str = "action_l2"
    ) -> dict:
        """
        Returns dict of { 'state': (S,), 'cam_0': (S,), 'cam_1': (S,), ... } normalized to sum to 1 per-step and overall curve.
        score_mode: "action_l2" or "supervised_loss"
        """
        # 1) Build normalized+stacked batch like in forward
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack(
                [batch[k].unsqueeze(0) for k in self.config.image_features], dim=-4
            )

        # 2) Enable grads on inputs that produce encoder features
        #    Practical: grads wrt feature tensors captured in BesoModel via return_feature_parts
        self.diffusion.return_feature_parts = True

        # 3) One forward to produce outputs and a scalar score
        #    We need gradients, so no @torch.no_grad()
        #    Use the same path as training to define a clean scalar:
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = self.config.horizon

        # Create a fake target to reuse compute_loss if dataset batch missing targets
        use_supervised = (score_mode == "supervised_loss") and (
            "action" in batch and "action_is_pad" in batch
        )

        if use_supervised:
            # compute supervised diffusion loss (already a scalar)
            loss = self.diffusion.compute_loss(
                self.normalize_targets(batch)
            )  # compute_loss expects normalized targets
            score = loss
        else:
            # derive a scalar from predicted action chunk at current obs step
            feats = self.diffusion._prepare_global_conditioning(
                batch
            )  # captures last_* feats
            B = batch[OBS_STATE].shape[0]
            actions = self.diffusion.conditional_sample(B, global_cond=feats)
            # take actions at current time index (last obs) for a stable scalar
            t = n_obs_steps - 1
            a_now = actions[:, t]  # (B, action_dim)
            score = (a_now**2).mean()

        # 4) Backprop to features
        self.diffusion.zero_grad(set_to_none=True)
        score.backward()

        # 5) Pull grads on captured features and aggregate per time step
        state_feats = self.diffusion.last_state_feats  # (B, S, D_s)
        img_feats = self.diffusion.last_img_feats  # (B, S, n_cam*D) or None

        out = {}
        if state_feats is not None and state_feats.grad is not None:
            g = state_feats.grad.detach().abs()  # (B,S,D_s)
            out["state"] = g.mean(dim=(0, 2)).cpu().numpy()  # (S,)

        cams = self._split_img_feats_per_cam(img_feats)
        for i, cam_feats in enumerate(cams):
            if cam_feats is not None and cam_feats.grad is not None:
                g = cam_feats.grad.detach().abs()  # (B,S,D)
                out[f"cam_{i}"] = g.mean(dim=(0, 2)).cpu().numpy()  # (S,)

        # 6) Normalize curves to sum to 1 per time step (comparability)
        if len(out) > 0:
            import numpy as np

            keys = list(out.keys())
            S = out[keys[0]].shape[0]
            M = np.stack([out[k] for k in keys], axis=0)  # (M,S)
            denom = M.sum(axis=0, keepdims=True) + 1e-12
            M_norm = M / denom
            for i, k in enumerate(keys):
                out[k] = M_norm[i]

        self.diffusion.return_feature_parts = False
        self.diffusion.last_state_feats = None
        self.diffusion.last_img_feats = None
        return out


class BesoModel(nn.Module):
    def __init__(self, config: BesoConfig):
        super().__init__()
        self.config = config
        self.sigma_data = 0.5
        self.sigma_max = 80
        self.sigma_min = 0.001
        self.act_seq_len = config.horizon
        self.sampling_steps = 8
        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            # This assumes all encoders have the same feature_dim
            # A dummy encoder is instantiated to get the feature_dim
            dummy_encoder = BesoRgbEncoder(config)
            global_cond_dim += dummy_encoder.feature_dim * num_images

            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [BesoRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
            else:
                self.rgb_encoder = BesoRgbEncoder(config)

        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.dit_backbone = Noise_Dec_only(
            state_dim=global_cond_dim,
            action_dim=self.config.action_feature.shape[0],
            goal_dim=0,
            device=config.device,
            goal_conditioned=False,
            embed_dim=config.embed_dim,
            embed_pdrob=0,
            goal_seq_len=0,
            obs_seq_len=self.config.n_obs_steps,
            action_seq_len=self.config.horizon,
            # linear_output=False,
            use_ada_conditioning=False,
            diffusion_type="beso",  # ddpm, beso or rf,
            use_pos_emb=False,
        )
        # self.device =  get_device_from_parameters(self)
        self.device = config.device

        ### IMPORTANCE MAPS
        # in BesoModel.__init__
        self.return_feature_parts = False
        self.last_state_feats = None
        self.last_img_feats = None
        self.num_cameras = len(config.image_features) if config.image_features else 0
        self.img_feat_dim = None  # we’ll set it on first forward

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        actions = (
            torch.randn(
                size=(
                    batch_size,
                    self.config.horizon,
                    self.config.action_feature.shape[0],
                ),
                # size=(batch_size, 8, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            * self.sigma_max
        )

        input_state = global_cond
        sigmas = get_sigmas_exponential(
            self.sampling_steps, self.sigma_min, self.sigma_max, self.device
        )
        actions = sample_ddim(self, input_state, actions, None, sigmas)

        return actions

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]

        # 1) state as-is (FIX)
        state_feats = batch[OBS_STATE]  # (B, S, state_dim) - use actual state data

        global_cond_feats = [state_feats]

        # 2) images -> encoder -> concat cameras
        img_features = None
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(
                    batch["observation.images"], "b s n ... -> n (b s) ..."
                )
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(
                            self.rgb_encoder, images_per_camera, strict=True
                        )
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list,
                    "(n b s) ... -> b s (n ...)",
                    b=batch_size,
                    s=n_obs_steps,
                )
            else:
                shared = self.rgb_encoder(
                    einops.rearrange(
                        batch["observation.images"], "b s n ... -> (b s n) ..."
                    )
                )
                # Fix: Provide all three dimensions explicitly
                img_features = einops.rearrange(
                    shared,
                    "(b s n) d -> b s (n d)",
                    b=batch_size,
                    s=n_obs_steps,
                    n=self.num_cameras,
                )
            global_cond_feats.append(img_features)
            if self.img_feat_dim is None and img_features is not None:
                self.img_feat_dim = (
                    img_features.shape[-1] // self.num_cameras
                )  # per-camera D

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        feats = torch.cat(
            global_cond_feats, dim=-1
        )  # Remove the incorrect flatten and unsqueeze

        if self.return_feature_parts:
            self.last_state_feats = state_feats
            self.last_img_feats = img_features  # (B, S, n_cam * D) or None

        return feats

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps
        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)
        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)
        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
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
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        sigmas = make_sample_density("loglogistic", self.sigma_max, self.sigma_min)(
            shape=(len(trajectory),), device=self.device
        ).to(self.device)
        # ->

        c_skip, c_out, c_in = [
            append_dims(x, trajectory.ndim) for x in self.get_scalings(sigmas)
        ]
        noised_input = trajectory + noise * append_dims(sigmas, trajectory.ndim)
        model_output = self.dit_backbone(global_cond, noised_input * c_in, None, sigmas)
        target = (trajectory - c_skip * noised_input) / c_out

        loss = F.mse_loss(model_output, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)
        loss = loss.mean()
        return loss

    def get_scalings(self, sigma):
        """
        Compute the scalings for the denoising process.

        Args:
            sigma: The input sigma.
        Returns:
            The computed scalings for skip connections, output, and input.
        """
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def forward(self, state, action, goal, sigma):
        c_skip, c_out, c_in = [
            append_dims(x, action.ndim) for x in self.get_scalings(sigma)
        ]
        return (
            self.dit_backbone(state, action * c_in, goal, sigma) * c_out
            + action * c_skip
        )

    # def get_target(self, state, action, goal, noise, sigma, **kwargs):


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://huggingface.co/papers/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class BesoRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: BesoConfig):
        super().__init__()
        # Set up optional preprocessing.
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(
                    config.crop_shape
                )
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )

        # Note: we have a check in the config class to make sure all images have the same shape.
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = (
            config.crop_shape if config.crop_shape is not None else images_shape[1:]
        )
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(
            feature_map_shape, num_kp=config.spatial_softmax_num_keypoints
        )
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # If we're going to crop, first up/downscale to 256x256 so the crop has enough context.
        if self.do_crop:
            # Bilinear resize on a batch tensor; preserves value range [0,1].
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)

        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(
        predicate(m) for _, m in root_module.named_modules(remove_duplicate=True)
    )
    return root_module
