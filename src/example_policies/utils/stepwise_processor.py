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

"""Stepwise percentile normalizer as lerobot ProcessorStep wrappers.

These processor steps wrap :class:`StepwisePercentileNormalize` and
:class:`StepwisePercentileUnnormalize` into lerobot's ``ProcessorStep`` API so
they can be inserted into ``PolicyProcessorPipeline`` for both training and
inference.

During training the preprocessor normalises the *action* targets inside the batch.
During inference the postprocessor unnormalises the policy's predicted actions.

Observation normalization is delegated to the standard ``NormalizerProcessorStep``
that is already in the pipeline — the stepwise steps only touch actions.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry

from .stepwise_normalize import (
    StepwisePercentileNormalize,
    StepwisePercentileUnnormalize,
)

# ──────────────────────────────────────────────────────────────────────────────
# Stats file utilities
# ──────────────────────────────────────────────────────────────────────────────

STEPWISE_STATS_FILENAME = "stepwise_percentile_stats.json"


def save_stepwise_stats(stats: dict[str, Tensor], path: pathlib.Path | str) -> None:
    """Save stepwise percentile stats (p_low, p_high) to a JSON file.

    Args:
        stats: Dict with ``"p_low"`` and ``"p_high"`` Tensors of shape ``(H, D)``.
        path: Directory or file path. If a directory, saves as
            ``stepwise_percentile_stats.json`` inside it.
    """
    path = pathlib.Path(path)
    if path.is_dir():
        path = path / STEPWISE_STATS_FILENAME
    data = {k: v.tolist() for k, v in stats.items()}
    path.write_text(json.dumps(data, indent=2))


def load_stepwise_stats(path: pathlib.Path | str) -> dict[str, Tensor]:
    """Load stepwise percentile stats from a JSON file.

    Args:
        path: File path or directory containing ``stepwise_percentile_stats.json``.

    Returns:
        Dict with ``"p_low"`` and ``"p_high"`` Tensors of shape ``(H, D)``.
    """
    path = pathlib.Path(path)
    if path.is_dir():
        path = path / STEPWISE_STATS_FILENAME
    data = json.loads(path.read_text())
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Forward normalizer processor step
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
@ProcessorStepRegistry.register(name="stepwise_normalizer_processor")
class StepwiseNormalizerProcessorStep(ProcessorStep):
    """Processor step that applies stepwise percentile normalization to actions.

    Wraps :class:`StepwisePercentileNormalize` for use in a
    ``PolicyProcessorPipeline`` (preprocessor during training).

    Only the ``"action"`` key in the transition is modified — observations pass
    through unchanged (they are handled by the regular ``NormalizerProcessorStep``).

    Args:
        p_low: 2nd-percentile stats, shape ``(H, D)``.
        p_high: 98th-percentile stats, shape ``(H, D)``.
        skip_feature_indices: Feature indices to leave un-normalised (e.g. 6D
            rotation features).
        clip_min: Lower clamp bound (default -1.5).
        clip_max: Upper clamp bound (default 1.5).
    """

    p_low: Tensor | None = None
    p_high: Tensor | None = None
    skip_feature_indices: list[int] = field(default_factory=list)
    clip_min: float = -1.5
    clip_max: float = 1.5

    _normalizer: StepwisePercentileNormalize | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        if self.p_low is not None and self.p_high is not None:
            self._build_normalizer()

    def _build_normalizer(self):
        self._normalizer = StepwisePercentileNormalize(
            p02=self.p_low,
            p98=self.p_high,
            skip_feature_indices=self.skip_feature_indices or None,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = dict(transition)

        action = new_transition.get(TransitionKey.ACTION)
        if action is None or self._normalizer is None:
            return new_transition

        if isinstance(action, torch.Tensor):
            # Move normalizer to same device if needed
            if self._normalizer.p02.device != action.device:
                self._normalizer = self._normalizer.to(action.device)
            new_transition[TransitionKey.ACTION] = self._normalizer(action)

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "skip_feature_indices": self.skip_feature_indices,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }

    def state_dict(self) -> dict[str, Tensor]:
        if self._normalizer is None:
            return {}
        return {
            "p02": self._normalizer.p02.cpu(),
            "p98": self._normalizer.p98.cpu(),
            "normalize_mask": self._normalizer.normalize_mask.cpu(),
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        if "p02" in state and "p98" in state:
            self.p_low = state["p02"]
            self.p_high = state["p98"]
            self._build_normalizer()
            # Restore the mask if available
            if "normalize_mask" in state:
                self._normalizer.normalize_mask.copy_(state["normalize_mask"])


# ──────────────────────────────────────────────────────────────────────────────
# Inverse normalizer processor step
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
@ProcessorStepRegistry.register(name="stepwise_unnormalizer_processor")
class StepwiseUnnormalizerProcessorStep(ProcessorStep):
    """Processor step that applies stepwise percentile unnormalization to actions.

    Wraps :class:`StepwisePercentileUnnormalize` for use in a
    ``PolicyProcessorPipeline`` (postprocessor during inference).

    Args:
        p_low: 2nd-percentile stats, shape ``(H, D)``.
        p_high: 98th-percentile stats, shape ``(H, D)``.
        skip_feature_indices: Feature indices that were left un-normalised.
    """

    p_low: Tensor | None = None
    p_high: Tensor | None = None
    skip_feature_indices: list[int] = field(default_factory=list)
    start_step: int = 0

    _unnormalizer: StepwisePercentileUnnormalize | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        if self.p_low is not None and self.p_high is not None:
            self._build_unnormalizer()

    def _build_unnormalizer(self):
        self._unnormalizer = StepwisePercentileUnnormalize(
            p02=self.p_low,
            p98=self.p_high,
            skip_feature_indices=self.skip_feature_indices or None,
            start_step=self.start_step,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = dict(transition)

        action = new_transition.get(TransitionKey.ACTION)
        if action is None or self._unnormalizer is None:
            return new_transition

        if isinstance(action, torch.Tensor):
            if self._unnormalizer.p02.device != action.device:
                self._unnormalizer = self._unnormalizer.to(action.device)
            new_transition[TransitionKey.ACTION] = self._unnormalizer(action)

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "skip_feature_indices": self.skip_feature_indices,
            "start_step": self.start_step,
        }

    def state_dict(self) -> dict[str, Tensor]:
        if self._unnormalizer is None:
            return {}
        return {
            "p02": self._unnormalizer.p02.cpu(),
            "p98": self._unnormalizer.p98.cpu(),
            "normalize_mask": self._unnormalizer.normalize_mask.cpu(),
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        if "p02" in state and "p98" in state:
            self.p_low = state["p02"]
            self.p_high = state["p98"]
            self._build_unnormalizer()
            if "normalize_mask" in state:
                self._unnormalizer.normalize_mask.copy_(state["normalize_mask"])
