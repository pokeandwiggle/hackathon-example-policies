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

"""Compute stepwise percentile stats from parquet action data.

Reads absolute TCP actions and observation.state from parquet files,
converts to chunk-relative UMI-delta (matching TRI's LBM paper,
arXiv:2507.05331), then computes per-timestep percentile statistics.
The result is saved as a JSON file that can be loaded by
:func:`~.stepwise_processor.load_stepwise_stats`.
"""

from __future__ import annotations

import pathlib
from collections import defaultdict

import pandas as pd
import torch

from .stepwise_normalize import compute_stepwise_percentile_stats
from .stepwise_processor import STEPWISE_STATS_FILENAME, load_stepwise_stats, save_stepwise_stats


def compute_stepwise_stats_from_parquet(
    data_dir: pathlib.Path | str,
    horizon: int,
    *,
    obs_tcp_left_pos_indices: list[int],
    obs_tcp_left_quat_indices: list[int],
    obs_tcp_right_pos_indices: list[int],
    obs_tcp_right_quat_indices: list[int],
    force: bool = False,
) -> str:
    """Compute stepwise percentile stats from absolute TCP actions.

    Reads absolute TCP actions and observation.state from parquet files,
    converts to chunk-relative UMI-delta (all steps relative to the TCP at
    chunk start), then computes per-timestep percentile statistics.

    This matches TRI's LBM paper (arXiv:2507.05331, §4.4.2) where
    "actions further into the future have a wider spread."

    Args:
        data_dir: Root directory of a LeRobot dataset.
        horizon: Action chunk length.
        obs_tcp_left_pos_indices: Indices into observation.state for left TCP pos.
        obs_tcp_left_quat_indices: Indices for left TCP quaternion.
        obs_tcp_right_pos_indices: Indices for right TCP position.
        obs_tcp_right_quat_indices: Indices for right TCP quaternion.
        force: Recompute even if cached stats exist.

    Returns:
        Absolute path to the saved stats JSON file.
    """
    from .chunk_relative_processor import abs_tcp_to_chunk_relative_umi_delta
    from ..data_ops.utils.rotation_6d import quat_to_6d_torch

    data_dir = pathlib.Path(data_dir)
    stats_path = data_dir / STEPWISE_STATS_FILENAME

    if not force and stats_path.exists():
        existing = load_stepwise_stats(stats_path)
        if existing["p_low"].shape[0] == horizon:
            print(f"Reusing existing stepwise stats: {stats_path} (horizon={horizon})")
            return str(stats_path)
        print(
            f"Existing stats have horizon={existing['p_low'].shape[0]}, "
            f"but training horizon={horizon}. Recomputing..."
        )

    print(f"\nComputing chunk-relative stepwise percentile stats (horizon={horizon})...")

    parquet_dir = data_dir / "data"
    all_actions: list[torch.Tensor] = []
    all_obs_states: list[torch.Tensor] = []
    episode_indices: list[int] = []

    for parquet_file in sorted(parquet_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_file)
        if "action" not in df.columns or "observation.state" not in df.columns:
            continue
        for a, s, ep_idx in zip(
            df["action"].tolist(),
            df["observation.state"].tolist(),
            df["episode_index"].tolist(),
        ):
            all_actions.append(torch.tensor(a, dtype=torch.float32))
            all_obs_states.append(torch.tensor(s, dtype=torch.float32))
            episode_indices.append(ep_idx)

    if not all_actions:
        raise RuntimeError(
            f"No actions found in {parquet_dir}. Cannot compute stepwise stats."
        )

    # Group by episode
    episodes: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    for action, obs_state, ep_idx in zip(all_actions, all_obs_states, episode_indices):
        episodes[ep_idx].append((action, obs_state))

    # Build chunk-relative UMI-delta chunks per episode (sliding window)
    chunks: list[torch.Tensor] = []
    for ep_idx in sorted(episodes.keys()):
        ep_data = episodes[ep_idx]
        for i in range(len(ep_data) - horizon + 1):
            # Reference TCP is the observation.state at the chunk start
            _, ref_obs_state = ep_data[i]

            # Extract reference TCP poses
            ref_pos_l = ref_obs_state[obs_tcp_left_pos_indices]
            ref_quat_l = ref_obs_state[obs_tcp_left_quat_indices]
            ref_pos_r = ref_obs_state[obs_tcp_right_pos_indices]
            ref_quat_r = ref_obs_state[obs_tcp_right_quat_indices]

            ref_rot6d_l = quat_to_6d_torch(ref_quat_l)
            ref_rot6d_r = quat_to_6d_torch(ref_quat_r)

            # Stack H absolute TCP actions
            abs_chunk = torch.stack(
                [ep_data[i + k][0] for k in range(horizon)]
            )  # (H, 16)

            # Convert to chunk-relative UMI delta
            umi_chunk = abs_tcp_to_chunk_relative_umi_delta(
                abs_chunk.unsqueeze(0),  # (1, H, 16)
                ref_pos_l.unsqueeze(0),
                ref_rot6d_l.unsqueeze(0),
                ref_pos_r.unsqueeze(0),
                ref_rot6d_r.unsqueeze(0),
            ).squeeze(0)  # (H, 20)

            chunks.append(umi_chunk)

    if not chunks:
        raise RuntimeError(
            f"Not enough frames to form chunks of horizon={horizon}. "
            f"Dataset has {len(all_actions)} total frames across "
            f"{len(episodes)} episode(s)."
        )

    print(f"  Built {len(chunks)} chunk-relative action chunks from {len(episodes)} episode(s)")

    class _ChunkDataset:
        def __init__(self, items: list[torch.Tensor]):
            self._items = items

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {"action": self._items[idx]}

    stats = compute_stepwise_percentile_stats(
        dataset=_ChunkDataset(chunks),
        action_key="action",
        horizon=horizon,
    )

    save_stepwise_stats(stats, stats_path)
    h, d = stats["p_low"].shape
    print(f"  Saved {stats_path.name}  (horizon={h}, action_dim={d})")
    return str(stats_path)
