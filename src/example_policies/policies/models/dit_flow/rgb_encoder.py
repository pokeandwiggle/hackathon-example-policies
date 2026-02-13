"""Custom RGB encoder that supports pretrained ImageNet weights + GroupNorm replacement.

This follows the Stanford Diffusion Policy approach (real-stanford/diffusion_policy):
1. Load pretrained ResNet with ImageNet weights
2. Replace all BatchNorm2d → GroupNorm (num_groups = num_features // 16)

The conv weights are preserved; only the normalization layer parameters are re-initialized.
GroupNorm(weight=1, bias=0) starts as near-identity, so the pretrained features are
largely intact and the GroupNorm params fine-tune quickly during training.
"""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor

from lerobot.policies.utils import get_output_shape


class SpatialSoftmax(nn.Module):
    """Spatial Soft Argmax operation.

    From "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    Ported from the robomimic / LeRobot implementation.
    """

    def __init__(self, input_shape: list[int], num_kp: int | None = None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w),
            np.linspace(-1.0, 1.0, self._in_h),
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)
        return feature_keypoints


def _replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """Replace submodules matching predicate with the result of func.

    Same implementation as LeRobot's _replace_submodules.
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
    # Verify that all matching modules have been replaced.
    assert not any(
        predicate(m) for _, m in root_module.named_modules(remove_duplicate=True)
    ), "Some BatchNorm2d modules were not replaced!"
    return root_module


class PretrainedGroupNormRgbEncoder(nn.Module):
    """RGB encoder with pretrained ResNet backbone + GroupNorm replacement.

    This implements the Stanford Diffusion Policy approach:
    1. Load ResNet with pretrained ImageNet weights (preserving conv feature extractors)
    2. Replace all BatchNorm2d with GroupNorm (num_groups = num_features // 16)
    3. Pool with SpatialSoftmax → Linear → ReLU to get a 1D feature vector

    The GroupNorm replacement uses the same formula as the Stanford repo:
        nn.GroupNorm(num_groups=C // 16, num_channels=C)

    For ResNet18 this gives:
        layer1 (64 ch)  → 4 groups
        layer2 (128 ch) → 8 groups
        layer3 (256 ch) → 16 groups
        layer4 (512 ch) → 32 groups
    """

    def __init__(self, config):
        super().__init__()

        # Set up optional preprocessing (crop).
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Step 1: Load pretrained backbone
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Strip avgpool + fc, keep through layer4 feature maps
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

        # Step 2: Replace BatchNorm2d → GroupNorm (Stanford approach)
        if config.use_group_norm:
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features,
                ),
            )

        # Step 3: Set up pooling and final layers
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
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
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x
