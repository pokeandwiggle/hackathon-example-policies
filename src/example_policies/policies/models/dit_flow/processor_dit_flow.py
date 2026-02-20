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

"""Processor pipeline for DiTFlow policy (new lerobot API)."""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from example_policies.utils.chunk_relative_processor import AbsTcpToChunkRelativeStep
from example_policies.utils.stepwise_processor import (
    StepwiseNormalizerProcessorStep,
    StepwiseUnnormalizerProcessorStep,
    load_stepwise_stats,
)
from .configuration_dit_flow import DiTFlowConfig

logger = logging.getLogger(__name__)


def make_ditflow_pre_post_processors(
    config: DiTFlowConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for a DiTFlow policy.

    The pre-processing pipeline prepares the input data for the model by:
    1. Renaming features.
    2. (Optional) Converting absolute TCP actions to chunk-relative UMI-delta.
    3. Normalizing the input and output features based on dataset statistics.
    4. (Optional) Applying stepwise percentile normalization to actions.
    5. Adding a batch dimension.
    6. Moving the data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving the data to the CPU.
    2. (Optional) Applying stepwise percentile unnormalization to actions.
    3. Unnormalizing the output features to their original scale.

    When ``config.use_chunk_relative_actions`` is ``True``, absolute TCP actions
    (16-dim) from the dataset are converted to chunk-relative UMI-delta (20-dim)
    at training time, matching TRI's LBM paper (arXiv:2507.05331).

    When ``config.use_stepwise_normalization`` is ``True``, the standard normalizer
    uses ``IDENTITY`` for actions (set automatically in ``DiTFlowConfig.__post_init__``)
    and a :class:`StepwiseNormalizerProcessorStep` / :class:`StepwiseUnnormalizerProcessorStep`
    handles action normalization using per-timestep-index percentile stats.

    Args:
        config: The configuration object for the DiTFlow policy,
            containing feature definitions, normalization mappings, and device information.
        dataset_stats: A dictionary of statistics used for normalization.
            Defaults to None.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    # ── Build chunk-relative step if requested ─────────────────────────────
    chunk_relative_step: AbsTcpToChunkRelativeStep | None = None

    if config.use_chunk_relative_actions:
        chunk_relative_step = AbsTcpToChunkRelativeStep(
            obs_tcp_left_pos_indices=config.obs_tcp_left_pos_indices,
            obs_tcp_left_quat_indices=config.obs_tcp_left_quat_indices,
            obs_tcp_right_pos_indices=config.obs_tcp_right_pos_indices,
            obs_tcp_right_quat_indices=config.obs_tcp_right_quat_indices,
        )
        logger.info(
            "Chunk-relative conversion enabled: abs TCP (16-dim) → UMI delta (20-dim)"
        )

    # ── Build stepwise normalizer/unnormalizer if requested ──────────────
    stepwise_normalizer: StepwiseNormalizerProcessorStep | None = None
    stepwise_unnormalizer: StepwiseUnnormalizerProcessorStep | None = None

    if config.use_stepwise_normalization:
        p_low, p_high = None, None

        # Load precomputed per-timestep percentile stats
        if config.stepwise_stats_path is not None:
            stats_path = pathlib.Path(config.stepwise_stats_path)
            if stats_path.exists():
                stats = load_stepwise_stats(stats_path)
                p_low = stats["p_low"]
                p_high = stats["p_high"]
                logger.info("Loaded stepwise percentile stats from %s", stats_path)
            else:
                logger.warning(
                    "stepwise_stats_path=%s does not exist; "
                    "stepwise normalizer will be created without stats "
                    "(expect them to be loaded from checkpoint).",
                    stats_path,
                )

        stepwise_normalizer = StepwiseNormalizerProcessorStep(
            p_low=p_low,
            p_high=p_high,
            skip_feature_indices=config.stepwise_skip_feature_indices,
            clip_min=config.stepwise_clip_min,
            clip_max=config.stepwise_clip_max,
        )
        stepwise_unnormalizer = StepwiseUnnormalizerProcessorStep(
            p_low=p_low,
            p_high=p_high,
            skip_feature_indices=config.stepwise_skip_feature_indices,
        )

    # ── Preprocessor (training & eval input) ─────────────────────────────
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
    ]
    if chunk_relative_step is not None:
        input_steps.append(chunk_relative_step)
    input_steps.append(
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    )
    if stepwise_normalizer is not None:
        input_steps.append(stepwise_normalizer)
    input_steps += [
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
    ]

    # ── Postprocessor (inference output) ─────────────────────────────────
    output_steps = []
    if stepwise_unnormalizer is not None:
        output_steps.append(stepwise_unnormalizer)
    output_steps += [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
