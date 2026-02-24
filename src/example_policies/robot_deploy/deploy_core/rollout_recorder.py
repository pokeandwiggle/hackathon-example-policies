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

"""Records policy rollouts into a LeRobot v3.0 dataset.

Usage:
    recorder = RolloutRecorder.from_policy_bundle(output_dir, policy_bundle, fps=10)
    recorder.start_episode()
    for step in steps:
        result = runner.run_step_recorded(policy_bundle)
        recorder.record_step(result)
    recorder.end_episode()
    recorder.close()
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .deployment_structures import PolicyBundle


@dataclass
class StepResult:
    """Data captured from a single inference step."""

    observation: dict | None = None
    action: torch.Tensor | None = None  # raw policy output (before translation)
    termination_signal: float | None = None


class RolloutRecorder:
    """Records policy rollouts (observations + actions) into a LeRobot v3.0 dataset.

    Each call to start_episode() / end_episode() corresponds to one episode in the dataset.
    Only the raw policy action (before action translation) is stored, matching the
    representation the model was trained on.
    """

    def __init__(
        self,
        output_dir: pathlib.Path,
        features: dict[str, Any],
        fps: int,
        task_name: str = "policy_rollout",
        image_keys: list[str] | None = None,
    ):
        self.output_dir = pathlib.Path(output_dir)
        self.fps = fps
        self.task_name = task_name
        self.image_keys = image_keys or []
        self._episode_active = False
        self._frame_count = 0
        self._total_episodes = 0
        self._outcomes: list[str] = []  # per-episode outcome labels

        # Create the LeRobot v3.0 dataset
        # Use threads only (no subprocesses) — multiprocessing image writers
        # deadlock in Jupyter notebooks after KeyboardInterrupt.
        self.dataset = LeRobotDataset.create(
            repo_id="local_only",
            fps=fps,
            root=self.output_dir,
            robot_type="panda_bimanual",
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            features=features,
            vcodec="libsvtav1",
        )
        print(f"RolloutRecorder: dataset created at {self.output_dir}")
        print(f"  Features: {list(features.keys())}")
        print(f"  FPS: {fps}")

    @classmethod
    def from_policy_bundle(
        cls,
        output_dir: pathlib.Path | str,
        policy_bundle: PolicyBundle,
        fps: int = 10,
        task_name: str = "policy_rollout",
    ) -> RolloutRecorder:
        """Create a recorder that matches the feature spec of a loaded policy.

        Extracts observation state names, action names, and image keys from the
        policy's config metadata so the recorded dataset is compatible with the
        model that produced it.
        """
        cfg = policy_bundle.config
        metadata = getattr(cfg, "metadata", None)
        if metadata is None:
            raise ValueError(
                "Policy config has no metadata. Cannot auto-detect features. "
                "Please provide features manually."
            )

        features_meta = metadata["features"]
        features: dict[str, Any] = {}
        image_keys: list[str] = []

        # -- observation.state --
        state_info = features_meta["observation.state"]
        features["observation.state"] = {
            "dtype": "float32",
            "shape": tuple(state_info["shape"]),
            "names": state_info["names"],
        }

        # -- action --
        # When chunk-relative actions are enabled the model outputs 20-dim
        # UMI-delta actions, but the checkpoint metadata still stores the
        # original 16-dim TCP shape.  Override to match the actual output.
        if getattr(cfg, "use_chunk_relative_actions", False):
            from example_policies.utils.action_order import UMI_ACTION_DIM

            umi_action_names = (
                [f"umi_delta_tcp_left_dpos_{i}" for i in "xyz"]
                + [f"umi_delta_tcp_left_rot6d_{i}" for i in range(6)]
                + [f"umi_delta_tcp_right_dpos_{i}" for i in "xyz"]
                + [f"umi_delta_tcp_right_rot6d_{i}" for i in range(6)]
                + ["gripper_left", "gripper_right"]
            )
            features["action"] = {
                "dtype": "float32",
                "shape": (UMI_ACTION_DIM,),
                "names": umi_action_names,
            }
        else:
            action_info = features_meta["action"]
            features["action"] = {
                "dtype": "float32",
                "shape": tuple(action_info["shape"]),
                "names": action_info["names"],
            }

        # -- images (video features) --
        for key, info in features_meta.items():
            if not key.startswith("observation.images."):
                continue
            # Use the shape from cfg.input_features for the actual resolution
            if key in cfg.input_features:
                input_shape = cfg.input_features[key].shape  # (C, H, W)
                h, w = input_shape[1], input_shape[2]
            else:
                # Fallback to metadata shape
                h, w = info["shape"][0], info["shape"][1]

            features[key] = {
                "dtype": "video",
                "shape": [h, w, 3],
                "names": ["height", "width", "channel"],
            }
            image_keys.append(key)

        return cls(
            output_dir=pathlib.Path(output_dir),
            features=features,
            fps=fps,
            task_name=task_name,
            image_keys=image_keys,
        )

    def start_episode(self) -> None:
        """Begin recording a new episode."""
        if self._episode_active:
            print("Warning: start_episode() called while episode already active. Ending previous episode.")
            self.end_episode()
        self._episode_active = True
        self._frame_count = 0
        print(f"Recording episode {self._total_episodes + 1}...")

    def record_step(self, step_result: StepResult) -> None:
        """Record a single inference step (observation + action) as one frame.

        Args:
            step_result: The StepResult returned by InferenceRunner.run_step_recorded().
        """
        if not self._episode_active:
            raise RuntimeError("record_step() called without start_episode()")

        if step_result.observation is None or step_result.action is None:
            return  # Skip empty steps (e.g., no observation from robot)

        frame = self._build_frame(step_result)
        frame["task"] = self.task_name
        self.dataset.add_frame(frame)
        self._frame_count += 1

    def end_episode(self, outcome: str | None = None) -> bool:
        """End the current episode and save it.

        Args:
            outcome: Optional label like ``"success"`` or ``"failure"``.
                If provided, the task string for every frame in this episode
                is updated to ``"{task_name} ({outcome})"`` before saving.

        Returns:
            True if the episode was saved (had frames), False otherwise.
        """
        if not self._episode_active:
            print("Warning: end_episode() called without active episode.")
            return False

        self._episode_active = False

        if self._frame_count == 0:
            print("Episode had no frames, skipping save.")
            return False

        # Relabel task with outcome before saving
        if outcome and self.dataset.episode_buffer is not None:
            task_label = f"{self.task_name} ({outcome})"
            n = len(self.dataset.episode_buffer["task"])
            self.dataset.episode_buffer["task"] = [task_label] * n

        self._total_episodes += 1
        print(f"Encoding episode {self._total_episodes} ({self._frame_count} frames)...")
        self._save_episode_quiet()
        label = f" [{outcome}]" if outcome else ""
        print(f"Saved episode {self._total_episodes} ({self._frame_count} frames){label}")
        if outcome:
            self._outcomes.append(outcome)
        return True

    @property
    def outcomes(self) -> list[str]:
        """List of outcome labels (e.g. 'success', 'failure') for each saved episode."""
        return list(self._outcomes)

    @property
    def success_rate(self) -> float:
        """Fraction of episodes rated as 'success'. Returns 0.0 if no outcomes recorded."""
        if not self._outcomes:
            return 0.0
        return sum(1 for o in self._outcomes if o == "success") / len(self._outcomes)

    def close(self) -> None:
        """Finalize the dataset. Call this when all rollouts are done."""
        if self._episode_active:
            self.end_episode()
        self._run_quiet(self.dataset.finalize)
        print(
            f"RolloutRecorder: dataset finalized at {self.output_dir} "
            f"({self._total_episodes} episodes)"
        )

    # ------------------------------------------------------------------
    # Helpers to suppress native ffmpeg / SVT-AV1 stderr spam
    # ------------------------------------------------------------------

    def _run_quiet(self, fn, *args, **kwargs):
        """Call *fn* while redirecting native stderr to /dev/null.

        SVT-AV1 and ffmpeg write info/warning banners directly to file
        descriptor 2 (stderr), bypassing Python's logging. Temporarily
        pointing fd 2 at /dev/null silences them.
        """
        stderr_fd = 2
        saved_fd = os.dup(stderr_fd)
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, stderr_fd)
            os.close(devnull)
            return fn(*args, **kwargs)
        finally:
            os.dup2(saved_fd, stderr_fd)
            os.close(saved_fd)

    def _save_episode_quiet(self):
        """Save episode with video encoding noise suppressed."""
        self._run_quiet(self.dataset.save_episode, parallel_encoding=False)

    def _build_frame(self, step_result: StepResult) -> dict:
        """Convert a StepResult into a flat dict suitable for LeRobotDataset.add_frame()."""
        obs = step_result.observation
        action = step_result.action

        frame: dict[str, Any] = {}

        # State: squeeze batch dim, move to CPU
        state_tensor = obs["observation.state"]
        if state_tensor.dim() > 1:
            state_tensor = state_tensor.squeeze(0)
        frame["observation.state"] = state_tensor.cpu().numpy().astype(np.float32)

        # Action: squeeze batch dim, move to CPU
        action_tensor = action
        if action_tensor.dim() > 1:
            action_tensor = action_tensor.squeeze(0)
        frame["action"] = action_tensor.cpu().numpy().astype(np.float32)

        # Images: convert from (1, C, H, W) uint8 tensor → (H, W, C) numpy for video encoding
        for key in self.image_keys:
            if key in obs:
                img_tensor = obs[key]
                if img_tensor.dim() == 4:
                    img_tensor = img_tensor.squeeze(0)  # (C, H, W)
                # CHW → HWC
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                frame[key] = img_np

        return frame
