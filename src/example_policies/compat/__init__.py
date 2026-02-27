"""Compatibility module for lerobot API changes.

This module provides compatibility shims for APIs that changed between lerobot versions.

Stepwise normalization utilities have moved to :mod:`example_policies.utils`.
Imports here are kept for backward compatibility.
"""

from .normalize import Normalize, Unnormalize

# Re-export stepwise utilities from their new home in utils/
from example_policies.utils.stepwise_normalize import (  # noqa: F401
    StepwisePercentileNormalize,
    StepwisePercentileUnnormalize,
    compute_stepwise_percentile_stats,
    make_stepwise_normalizer_pair,
)
from example_policies.utils.stepwise_processor import (  # noqa: F401
    StepwiseNormalizerProcessorStep,
    StepwiseUnnormalizerProcessorStep,
    load_stepwise_stats,
    save_stepwise_stats,
)

__all__ = [
    "Normalize",
    "Unnormalize",
    "StepwisePercentileNormalize",
    "StepwisePercentileUnnormalize",
    "compute_stepwise_percentile_stats",
    "make_stepwise_normalizer_pair",
    "StepwiseNormalizerProcessorStep",
    "StepwiseUnnormalizerProcessorStep",
    "load_stepwise_stats",
    "save_stepwise_stats",
]
