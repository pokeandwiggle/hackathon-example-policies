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

"""Stepwise percentile normalizer inspired by TRI's LBM paper (arXiv:2507.05331, §4.4.2).

Unlike the standard MIN_MAX or MEAN_STD normalization which uses a single set of statistics
across all timestep indices, this normalizer computes per-timestep-index (per position in
the action chunk) percentile statistics (p2 and p98) and normalizes to [-1.5, 1.5]:

    y = clamp(2 * (x - p02) / (p98 - p02) - 1,  -1.5,  1.5)

6D rotation features are excluded from normalization since they are already bounded in [-1, 1]
by construction (they represent orthonormal basis vectors).

This module provides:
    - ``StepwisePercentileNormalize``  — forward normalizer (nn.Module)
    - ``StepwisePercentileUnnormalize`` — inverse normalizer (nn.Module)
    - ``compute_stepwise_percentile_stats`` — utility to compute p02/p98 from a dataset
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_LOW_PERCENTILE: float = 0.02
DEFAULT_HIGH_PERCENTILE: float = 0.98
DEFAULT_CLIP_MIN: float = -1.5
DEFAULT_CLIP_MAX: float = 1.5


# ──────────────────────────────────────────────────────────────────────────────
# Forward Normalizer
# ──────────────────────────────────────────────────────────────────────────────


class StepwisePercentileNormalize(nn.Module):
    """Per-timestep-index percentile normalizer (TRI LBM style).

    Stores p02 and p98 statistics with shape ``(horizon, action_dim)`` and applies:

        y = clamp(2 * (x - p02) / (p98 - p02) - 1,  clip_min,  clip_max)

    Features whose indices are listed in *skip_feature_indices* are passed through
    unchanged — this is intended for 6D rotation features which are inherently
    bounded in [-1, 1].

    Args:
        p02: 2nd-percentile statistic per (timestep, feature). Shape ``(H, D)``.
        p98: 98th-percentile statistic per (timestep, feature). Shape ``(H, D)``.
        skip_feature_indices: Feature-dimension indices to leave un-normalised
            (e.g. 6D rotation columns). ``None`` means normalise everything.
        clip_min: Lower clamp bound after normalisation (default -1.5).
        clip_max: Upper clamp bound after normalisation (default 1.5).
    """

    def __init__(
        self,
        p02: Tensor,
        p98: Tensor,
        skip_feature_indices: Sequence[int] | None = None,
        clip_min: float = DEFAULT_CLIP_MIN,
        clip_max: float = DEFAULT_CLIP_MAX,
    ) -> None:
        super().__init__()
        assert p02.shape == p98.shape, f"p02 {p02.shape} != p98 {p98.shape}"
        assert p02.ndim == 2, f"Expected 2-D stats (horizon, action_dim), got {p02.ndim}-D"

        self.register_buffer("p02", p02.clone().float())
        self.register_buffer("p98", p98.clone().float())

        # Boolean mask: True = normalise, False = pass-through
        mask = torch.ones(p02.shape[-1], dtype=torch.bool)
        if skip_feature_indices is not None:
            for idx in skip_feature_indices:
                mask[idx] = False
        self.register_buffer("normalize_mask", mask)

        self.clip_min = clip_min
        self.clip_max = clip_max

    # Declared for typing (populated by register_buffer)
    p02: Tensor
    p98: Tensor
    normalize_mask: Tensor

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Normalise actions.

        Args:
            x: Action tensor. Accepted shapes:
                - ``(B, H, D)``  — batch of action chunks
                - ``(H, D)``     — single action chunk  (batch dim added/removed automatically)

        Returns:
            Normalised tensor of the same shape.
        """
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze = True

        B, H, D = x.shape
        assert H <= self.p02.shape[0], (
            f"Sequence length {H} exceeds stats horizon {self.p02.shape[0]}"
        )
        assert D == self.p02.shape[1], (
            f"Feature dim {D} != stats feature dim {self.p02.shape[1]}"
        )

        # Slice stats to match the actual horizon (allows using a subset)
        lo = self.p02[:H]  # (H, D)
        hi = self.p98[:H]  # (H, D)
        denom = hi - lo  # (H, D)

        out = x.clone()

        # Apply normalisation only to non-rotation features
        mask = self.normalize_mask  # (D,)
        out[..., mask] = (
            2.0 * (x[..., mask] - lo[..., mask]) / (denom[..., mask] + 1e-8) - 1.0
        )
        out[..., mask] = out[..., mask].clamp(self.clip_min, self.clip_max)

        if squeeze:
            out = out.squeeze(0)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Inverse Normalizer
# ──────────────────────────────────────────────────────────────────────────────


class StepwisePercentileUnnormalize(nn.Module):
    """Inverse of :class:`StepwisePercentileNormalize`.

    Applies:

        x = (y + 1) / 2 * (p98 - p02) + p02

    Note: values that were clipped during normalisation cannot be perfectly recovered.

    Args:
        p02: 2nd-percentile statistic per (timestep, feature). Shape ``(H, D)``.
        p98: 98th-percentile statistic per (timestep, feature). Shape ``(H, D)``.
        skip_feature_indices: Feature-dimension indices that were left un-normalised.
    """

    def __init__(
        self,
        p02: Tensor,
        p98: Tensor,
        skip_feature_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        assert p02.shape == p98.shape
        assert p02.ndim == 2

        self.register_buffer("p02", p02.clone().float())
        self.register_buffer("p98", p98.clone().float())

        mask = torch.ones(p02.shape[-1], dtype=torch.bool)
        if skip_feature_indices is not None:
            for idx in skip_feature_indices:
                mask[idx] = False
        self.register_buffer("normalize_mask", mask)

    p02: Tensor
    p98: Tensor
    normalize_mask: Tensor

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Unnormalise actions.

        Args:
            x: Normalised action tensor, shape ``(B, H, D)`` or ``(H, D)``.

        Returns:
            Unnormalised tensor of the same shape.
        """
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze = True

        B, H, D = x.shape
        assert H <= self.p02.shape[0]
        assert D == self.p02.shape[1]

        lo = self.p02[:H]
        hi = self.p98[:H]

        out = x.clone()
        mask = self.normalize_mask
        out[..., mask] = (x[..., mask] + 1.0) / 2.0 * (hi[..., mask] - lo[..., mask]) + lo[..., mask]

        if squeeze:
            out = out.squeeze(0)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Statistics Computation
# ──────────────────────────────────────────────────────────────────────────────


def compute_stepwise_percentile_stats(
    dataset,
    action_key: str = "action",
    horizon: int | None = None,
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
) -> dict[str, Tensor]:
    """Compute per-timestep-index percentile statistics from a LeRobot dataset.

    Iterates over all samples in the dataset and groups action values by their
    timestep index within the action chunk.  Then computes the given percentiles
    independently for each (timestep_index, feature_dim) pair.

    Args:
        dataset: A LeRobot dataset (or any indexable object whose ``__getitem__``
            returns a dict with *action_key* mapping to a Tensor of shape
            ``(H, D)`` or ``(D,)``).
        action_key: Key used to retrieve action tensors from dataset items.
        horizon: Expected action chunk length.  If ``None``, inferred from the
            first sample.
        low_percentile: Lower percentile (default 0.02 = 2nd percentile).
        high_percentile: Upper percentile (default 0.98 = 98th percentile).

    Returns:
        Dictionary with keys ``"p_low"`` and ``"p_high"``, each a ``(H, D)``
        Tensor of the corresponding percentile values.
    """
    # Collect all action chunks
    all_actions: list[Tensor] = []
    for i in range(len(dataset)):
        item = dataset[i]
        action = item[action_key]
        if not isinstance(action, Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)  # (1, D)
        all_actions.append(action)

    # Determine horizon
    if horizon is None:
        horizon = all_actions[0].shape[0]

    action_dim = all_actions[0].shape[-1]

    # Bucket by timestep index
    # Each bucket[t] collects all values at position t across all chunks
    buckets: list[list[Tensor]] = [[] for _ in range(horizon)]
    for action_chunk in all_actions:
        H = action_chunk.shape[0]
        for t in range(min(H, horizon)):
            buckets[t].append(action_chunk[t])

    # Compute percentiles per timestep
    p_low = torch.zeros(horizon, action_dim)
    p_high = torch.zeros(horizon, action_dim)

    for t in range(horizon):
        if len(buckets[t]) == 0:
            continue
        stacked = torch.stack(buckets[t], dim=0)  # (N, D)
        p_low[t] = torch.quantile(stacked.float(), low_percentile, dim=0)
        p_high[t] = torch.quantile(stacked.float(), high_percentile, dim=0)

    return {"p_low": p_low, "p_high": p_high}


def make_stepwise_normalizer_pair(
    dataset=None,
    stats: dict[str, Tensor] | None = None,
    skip_feature_indices: Sequence[int] | None = None,
    action_key: str = "action",
    horizon: int | None = None,
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
    clip_min: float = DEFAULT_CLIP_MIN,
    clip_max: float = DEFAULT_CLIP_MAX,
) -> tuple[StepwisePercentileNormalize, StepwisePercentileUnnormalize]:
    """Convenience factory to create a matched normalize/unnormalize pair.

    Either provide pre-computed *stats* (dict with ``"p_low"`` and ``"p_high"`` keys)
    or a *dataset* from which statistics will be computed on the fly.

    Args:
        dataset: LeRobot dataset (used only when *stats* is ``None``).
        stats: Pre-computed percentile stats as returned by
            :func:`compute_stepwise_percentile_stats`.
        skip_feature_indices: Indices of features to skip (e.g.
            ``UMI_ROTATION_FEATURE_INDICES``).
        action_key: Key in dataset items for the action tensor.
        horizon: Action chunk length (inferred if ``None``).
        low_percentile: Lower percentile for stat computation.
        high_percentile: Upper percentile for stat computation.
        clip_min: Lower clamp bound for the normalizer.
        clip_max: Upper clamp bound for the normalizer.

    Returns:
        ``(normalizer, unnormalizer)`` tuple.

    Raises:
        ValueError: If neither *dataset* nor *stats* is provided.
    """
    if stats is None:
        if dataset is None:
            raise ValueError("Provide either `dataset` or `stats`.")
        stats = compute_stepwise_percentile_stats(
            dataset,
            action_key=action_key,
            horizon=horizon,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
        )

    p02 = stats["p_low"]
    p98 = stats["p_high"]

    normalizer = StepwisePercentileNormalize(
        p02=p02,
        p98=p98,
        skip_feature_indices=skip_feature_indices,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    unnormalizer = StepwisePercentileUnnormalize(
        p02=p02,
        p98=p98,
        skip_feature_indices=skip_feature_indices,
    )
    return normalizer, unnormalizer
