import pathlib

import numpy as np
import pandas as pd

from ...utils.constants import ACTION, EPISODE_DIR, META_DIR
from ..config.pipeline_config import PipelineConfig


class PostLerobotPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.processors = self.build_post_pipeline(config)

    def build_post_pipeline(self, config):
        processors = []

        if config.requires_termination_signal():
            processors.append(termination_processor)
        return processors

    def read_parquet(self, filepath: pathlib.Path) -> pd.DataFrame:
        df = pd.read_parquet(filepath)
        return df

    def process_ep(self, filepath: pathlib.Path):
        df = self.read_parquet(filepath)
        for processor in self.processors:
            df = processor(self.config, df)
        return df

    def process_lerobot(self, lerobot_dir: pathlib.Path):
        print("\nProcessing LeRobot dataset post-conversion...")
        ep_dir = lerobot_dir / EPISODE_DIR
        meta_stats_df = pd.read_json(
            lerobot_dir / META_DIR / "episodes_stats.jsonl", lines=True
        )

        for episode_idx in meta_stats_df.index:
            ep_filepath = ep_dir / f"episode_{episode_idx:06d}.parquet"
            print(f"Processing episode file: {ep_filepath.name}...")
            df = self.process_ep(ep_filepath)

            meta_stats_df = self.modify_episode_metadata(meta_stats_df, episode_idx, df)

            df.to_parquet(ep_filepath)
        meta_stats_df.to_json(
            lerobot_dir / META_DIR / "episodes_stats.jsonl",
            orient="records",
            lines=True,
        )
        print("Post-processing complete.\n")

    def modify_episode_metadata(
        self, meta_stats_df: pd.DataFrame, ep_idx: int, new_ep_df: pd.DataFrame
    ):
        episode_stat_dict = meta_stats_df.loc[ep_idx, "stats"]
        episode_stat_df = pd.DataFrame.from_dict(episode_stat_dict, orient="index")

        for ft_row in episode_stat_df.iterrows():
            feature_name = ft_row[0]
            if feature_name in new_ep_df.columns:
                feature_array = np.stack(new_ep_df[feature_name].values)

                # Lerobot requires at least 1D arrays for stats
                episode_stat_df.at[feature_name, "mean"] = np.atleast_1d(
                    np.mean(feature_array, axis=0)
                ).tolist()
                episode_stat_df.at[feature_name, "std"] = np.atleast_1d(
                    np.std(feature_array, axis=0)
                ).tolist()
                episode_stat_df.at[feature_name, "min"] = np.atleast_1d(
                    np.min(feature_array, axis=0)
                ).tolist()
                episode_stat_df.at[feature_name, "max"] = np.atleast_1d(
                    np.max(feature_array, axis=0)
                ).tolist()

        meta_stats_df.at[ep_idx, "stats"] = episode_stat_df.to_dict(orient="index")
        return meta_stats_df


def termination_processor(config: PipelineConfig, df: pd.DataFrame) -> pd.DataFrame:
    last_n_indices = df.index[-int(config.termination_horizon_frames) :]
    for idx in last_n_indices:
        action: np.ndarray = df.loc[idx, ACTION].copy()
        action[-1] = 1.0
        df.at[idx, ACTION] = action

    return df
    