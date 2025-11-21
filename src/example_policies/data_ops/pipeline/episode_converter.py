"""Episode converter for processing MCAP files into dataset format."""

import pathlib
import time

from example_policies.data_ops.config import pipeline_config
from example_policies.data_ops.pipeline.dataset_writer import DatasetWriter
from example_policies.data_ops.pipeline.frame_buffer import FrameBuffer


class EpisodeConverter:
    """Converts MCAP episodes to dataset format."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        config: pipeline_config.PipelineConfig,
        features: dict,
    ):
        self.config = config
        self.frame_buffer = FrameBuffer(config)
        self.dataset_writer = DatasetWriter(output_dir, features, config)

        # Tracking
        self.episode_counter = 0
        self.episode_mapping: dict[int, str] = {}
        self.blacklist: list[int] = []

        # Episode state
        # Seen Frames used to achieve subsampling, e.g., only save every Nth frame
        # Subsample offset allows starting at a different frame within the subsampling window
        self.seen_frames = config.subsample_offset
        self.saved_frames = 0

        # Progress tracking
        self.start_time = time.time()
        self.last_print_time = time.time()
        self.total_frames = 0

    def reset_episode_state(self) -> None:
        """Reset state for new episode."""
        self.seen_frames = self.config.subsample_offset
        self.saved_frames = 0

    def process_message(self, topic: str, schema_name: str, msg_data: bytes) -> None:
        """Process a single MCAP message.

        Args:
            topic: ROS topic name
            schema_name: Schema name for the message
            msg_data: Raw message data
        """
        self.frame_buffer.add_msg(topic, schema_name, msg_data)

        if not self.frame_buffer.is_complete():
            return

        # Check subsampling
        if self.seen_frames % self.config.capture_frequency != 0:
            self.frame_buffer.reset()
            self.seen_frames += 1
            return

        self.seen_frames += 1
        self.total_frames += 1

        # Add frame to dataset
        perform_save = self.dataset_writer.add_frame(self.frame_buffer)
        self.frame_buffer.reset()

        if not perform_save:
            return

        self.saved_frames += 1
        self._print_progress()

    def _print_progress(self) -> None:
        """Print progress at intervals."""
        if self.saved_frames % self.config.capture_frequency == 0:
            now = time.time()
            elapsed_total = now - self.start_time
            elapsed_since_print = now - self.last_print_time
            fps = self.total_frames / elapsed_total if elapsed_total > 0 else 0

            print(
                f"  - Seen / Saved: {self.seen_frames} / {self.saved_frames} "
                f"in {elapsed_since_print:.2f}s | Total Time: {elapsed_total:.2f}s | FPS: {fps:.2f}",
                end="\r",
            )
            self.last_print_time = now

    def finalize_episode(self, episode_idx: int, episode_path: pathlib.Path) -> bool:
        """Finalize current episode and save if valid.

        Args:
            episode_idx: Index of the episode being processed
            episode_path: Path to the episode file

        Returns:
            True if episode was saved, False otherwise
        """
        print()  # New line after progress

        if self.saved_frames == 0:
            return False

        print(f"Saving {episode_path} processed with {self.seen_frames} frames.")

        perform_save = self.dataset_writer.save_episode(episode_idx)
        if not perform_save:
            return False

        # Track episode
        self.episode_mapping[self.episode_counter] = str(episode_path)

        # Check if episode is too short
        if self.saved_frames < self.config.min_episode_frames:
            print(
                f"Episode too short ({self.saved_frames} frames), Adding to Blacklist."
            )
            self.blacklist.append(self.episode_counter)

        self.episode_counter += 1
        return True

    def get_total_time(self) -> float:
        """Get total elapsed time."""
        return time.time() - self.start_time
