import pathlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from example_policies.data_ops.review.timing_plot import (
    save_timing_plot as _save_timing_plot,
)

import torch

from example_policies.robot_deploy.deploy_core.action_chunk_blender import (
    ActionChunkBlender,
)
from example_policies.robot_deploy.deploy_core.deployment_structures import (
    InferenceConfig,
    PolicyBundle,
)
from example_policies.robot_deploy.deploy_core.rollout_recorder import StepResult
from example_policies.robot_deploy.robot_io.robot_interface import RobotInterface
from example_policies.utils.action_order import ActionMode, GET_TERMINATION_IDX


@dataclass
class TimingStats:
    """Per-rollout timing statistics."""

    step_durations: list[float] = field(default_factory=list)
    inference_durations: list[float] = field(default_factory=list)
    overrun_durations: list[float] = field(default_factory=list)
    step_is_inference: list[bool] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.step_durations)

    @property
    def n_overruns(self) -> int:
        return len(self.overrun_durations)

    def _sub_stats(self, durations: list[float], target_period: float) -> str:
        """Format stats for a subset of step durations."""
        import statistics

        if not durations:
            return "  (no steps)"
        actual_hz = [1.0 / d for d in durations if d > 0]
        mean_hz = statistics.mean(actual_hz)
        min_hz = min(actual_hz)
        std_hz = statistics.stdev(actual_hz) if len(actual_hz) > 1 else 0.0
        mean_ms = statistics.mean(durations) * 1000
        max_ms = max(durations) * 1000
        p95_ms = sorted(durations)[int(len(durations) * 0.95)] * 1000
        n_over = sum(1 for d in durations if d > target_period * 1.01)
        return (
            f"  n={len(durations):>5d}  "
            f"freq: mean={mean_hz:.1f} Hz, min={min_hz:.1f} Hz, std={std_hz:.2f} Hz  │  "
            f"step: mean={mean_ms:.1f} ms, p95={p95_ms:.1f} ms, max={max_ms:.1f} ms  │  "
            f"overruns: {n_over}"
        )

    def summary(self, target_period: float) -> str:
        """Return a human-readable summary of timing deviations."""
        if not self.step_durations:
            return "No steps recorded."

        target_hz = 1.0 / target_period

        lines = [
            f"Timing stats ({self.n_steps} steps, target {target_hz:.1f} Hz / {target_period * 1000:.1f} ms):",
        ]

        # --- Separate stats for inference vs queue-replay steps ---
        if self.step_is_inference:
            inf_durs = [
                d
                for d, is_inf in zip(self.step_durations, self.step_is_inference)
                if is_inf
            ]
            queue_durs = [
                d
                for d, is_inf in zip(self.step_durations, self.step_is_inference)
                if not is_inf
            ]
            lines.append(f"  Inference steps (new chunk):")
            lines.append(f"  {self._sub_stats(inf_durs, target_period)}")
            lines.append(f"  Queue-replay steps:")
            lines.append(f"  {self._sub_stats(queue_durs, target_period)}")

        return "\n".join(lines)

    def save_plot(
        self, target_period: float, output_path: pathlib.Path
    ) -> pathlib.Path:
        """Generate and save a timing analysis plot. Returns the output path."""
        return _save_timing_plot(self, target_period, output_path)


class InferenceRunner:
    """Core inference loop execution."""

    def __init__(
        self,
        robot_interface: RobotInterface,
        config: InferenceConfig,
        verbose: bool = True,
        blender: Optional[ActionChunkBlender] = None,
    ):
        self.robot_interface = robot_interface
        self.config = config
        self.period = 1.0 / config.hz
        self.step = 0
        self.verbose = verbose

        # Optional action-chunk blender for smooth chunk transitions
        self._blender = blender

        # Observation captured at the start of each action chunk.
        # UMI-delta actions are chunk-relative (all steps expressed relative to
        # the TCP at chunk prediction time), so the translator must use this
        # observation rather than the *current* one which has drifted.
        self._chunk_observation: Optional[dict] = None

        # Per-rollout timing instrumentation
        self._timing: TimingStats = TimingStats()

    @property
    def timing_stats(self) -> TimingStats:
        """Access timing stats for the current/last rollout."""
        return self._timing

    def reset(self):
        """Reset runner state for a new rollout / policy switch.

        Sets ``step`` back to 0 so the next ``run_step_recorded()`` call
        will trigger the full reset chain (policy, translator, interface).
        """
        self.step = 0
        self._chunk_observation = None
        self._timing = TimingStats()
        self._step_was_inference = False
        if self._blender is not None:
            self._blender.reset()

    def run_step(self, policy_bundle: PolicyBundle) -> Optional[float]:
        """Execute one inference step. Returns termination signal if present."""
        result = self.run_step_recorded(policy_bundle)
        return result.termination_signal

    def run_step_recorded(self, policy_bundle: PolicyBundle) -> StepResult:
        """Execute one inference step. Returns full StepResult with observation, action, and termination signal."""
        start_time = time.monotonic()

        # Reset policy *and* interface state at the very first step
        if self.step == 0:
            if self.verbose:
                print("\n=== RESETTING POLICY ===")
            policy_bundle.reset()
            self._chunk_observation = None
            # Clear stale last_command and quaternion-continuity state so the
            # first observation of this rollout is not contaminated by the
            # previous one.
            self.robot_interface.reset()

        observation = self.robot_interface.get_observation(policy_bundle.config.device)

        if not observation:
            self._finish_step(start_time)
            return StepResult()

        # Detect whether a new action chunk will be predicted this step.
        # The policy's action queue is empty → select_action will call
        # predict_action_chunk internally, producing a fresh chunk.
        policy = policy_bundle.policy
        _queues = getattr(policy, "_queues", None)
        is_new_chunk = _queues is not None and len(_queues.get("action", [])) == 0

        # Seed the blender with the current robot pose on the first step
        # so it has a reference for blending the first chunk transition.
        if is_new_chunk and self._blender is not None and self._blender._last_sent_action is None:
            self._blender._last_sent_action = self._current_pose_as_action(
                policy_bundle, observation,
            )

        # When blending is enabled, temporarily expand the action queue AND
        # the model's n_action_steps so generate_actions() returns the full
        # horizon (chunk_size) instead of cropping to n_action_steps.
        _saved_n_action_steps = None
        if is_new_chunk and self._blender is not None:
            _queues["action"] = deque(maxlen=self._blender.chunk_size)
            _saved_n_action_steps = policy_bundle.policy.config.n_action_steps
            # generate_actions() slices actions[:, n_obs_steps-1 : n_obs_steps-1 + n_action_steps]
            # so the maximum extractable actions = horizon - (n_obs_steps - 1).
            n_obs_steps = getattr(policy_bundle.policy.config, "n_obs_steps", 1)
            max_extractable = self._blender.chunk_size - (n_obs_steps - 1)
            policy_bundle.policy.config.n_action_steps = max_extractable

        inference_start = time.monotonic()
        with torch.inference_mode():
            action, termination_signal = self._process_action(
                policy_bundle,
                observation,
                is_new_chunk=is_new_chunk,
            )

        # Restore original n_action_steps so the queue drains at the right rate.
        if _saved_n_action_steps is not None:
            policy_bundle.policy.config.n_action_steps = _saved_n_action_steps
        if is_new_chunk:
            self._timing.inference_durations.append(time.monotonic() - inference_start)
        # NOTE: step_is_inference is recorded in _finish_step together with
        # step_durations so both lists stay in sync on KeyboardInterrupt.
        self._step_was_inference = is_new_chunk

        if is_new_chunk and self.verbose:
            print("\n=== RAW MODEL PREDICTION ===")
            policy_bundle.printer.print(self.step, observation, action, raw_action=True)

        # --- Update chunk observation (used for UMI-delta translation) ---
        if is_new_chunk:
            self._chunk_observation = observation

        # === Action translation (with optional blending) ===
        if self._blender is not None:
            action_translated = self._run_blended_step(
                policy_bundle,
                observation,
                action,
                is_new_chunk,
            )
        else:
            action_translated = self._run_normal_translation(
                policy_bundle,
                observation,
                action,
            )

        if self.verbose:
            print("\n=== ABSOLUTE ROBOT COMMANDS ===")
            policy_bundle.printer.print(
                self.step, observation, action_translated, raw_action=False
            )

        self.robot_interface.send_action(
            action_translated,
            policy_bundle.translator.action_mode,
            self.config.controller,
        )

        self._finish_step(start_time, is_new_chunk)
        self.step += 1
        return StepResult(
            observation=observation,
            action=action,
            termination_signal=termination_signal,
        )

    # ------------------------------------------------------------------
    # Translation helpers
    # ------------------------------------------------------------------

    def _run_normal_translation(
        self,
        policy_bundle: PolicyBundle,
        observation: dict,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Standard per-step action translation (no blending)."""
        if (
            policy_bundle.translator.action_mode == ActionMode.UMI_DELTA_TCP
            and self._chunk_observation is not None
        ):
            translator_obs = self._chunk_observation
        else:
            translator_obs = observation

        return policy_bundle.translator.translate(action, translator_obs)

    def _run_blended_step(
        self,
        policy_bundle: PolicyBundle,
        observation: dict,
        action: torch.Tensor,
        is_new_chunk: bool,
    ) -> torch.Tensor:
        """Translation with action-chunk blending.

        On new-chunk steps: translates the full raw chunk to absolute TCP,
        passes it to the blender for overlap / offset-decay blending,
        then pops the first blended action.

        On subsequent steps: pops the next blended action from the stored
        chunk (normal translation is skipped).
        """
        assert self._blender is not None

        if is_new_chunk:
            # --- Extract full unnormalized chunk ---
            queue = policy_bundle.policy._queues["action"]
            remaining = list(queue)  # chunk_size - 1 items (expanded deque)
            full_raw = [action] + remaining  # chunk_size items, each (1, D)

            # Resize queue back to n_action_steps so the queue drains at
            # the correct rate and triggers re-prediction after
            # n_action_steps steps.
            n_remain = min(self._blender.n_action_steps - 1, len(remaining))
            policy_bundle.policy._queues["action"] = deque(
                remaining[:n_remain],
                maxlen=self._blender.n_action_steps,
            )

            # --- Translate full chunk to absolute TCP ---
            translated = self._translate_full_chunk(
                full_raw,
                policy_bundle,
                observation,
            )

            # --- Blend and store ---
            self._blender.on_new_chunk(translated)

        # Pop next blended action (works on both new-chunk and later steps)
        return self._blender.pop_action()

    def _translate_full_chunk(
        self,
        raw_actions: list[torch.Tensor],
        policy_bundle: PolicyBundle,
        observation: dict,
    ) -> list[torch.Tensor]:
        """Translate every action in a chunk to absolute TCP space.

        Returns a list of ``(16,)`` tensors (batch dim squeezed).

        For ``UMI_DELTA_TCP`` and ``TCP`` / ``TELEOP`` the translator is
        stateless per action, so we can safely call it in a loop without
        corrupting accumulated state.
        """
        translator = policy_bundle.translator
        mode = translator.action_mode

        if mode == ActionMode.UMI_DELTA_TCP:
            obs = self._chunk_observation or observation
        else:
            obs = observation

        translated: list[torch.Tensor] = []
        for raw in raw_actions:
            t = translator.translate(raw, obs)
            translated.append(t[0].clone())  # squeeze batch → (16,)

        return translated

    def _current_pose_as_action(
        self,
        policy_bundle: PolicyBundle,
        observation: dict,
    ) -> torch.Tensor:
        """Build a 16-dim absolute TCP action from the current observation state."""
        translator = policy_bundle.translator
        device = observation["observation.state"].device

        # 14-dim TCP pose: [L_pos(3), L_quat(4), R_pos(3), R_quat(4)]
        tcp_pose = translator._init_last_action_from_observation(observation, device)

        action = torch.zeros(16, device=device)
        action[:14] = tcp_pose[0]

        # Gripper widths: look up by feature name in the observation state
        state = observation["observation.state"]
        feature_names = translator._state_feature_names
        for name in ("robotiq_left", "panda_left"):
            if name in feature_names:
                action[14] = state[0, feature_names.index(name)]
                break
        for name in ("robotiq_right", "panda_right"):
            if name in feature_names:
                action[15] = state[0, feature_names.index(name)]
                break

        return action

    def _process_action(
        self,
        policy_bundle: PolicyBundle,
        observation: dict,
        *,
        is_new_chunk: bool = False,
    ) -> tuple[torch.Tensor, Optional[float]]:
        """Process action from policy, extracting termination signal if present."""
        # Apply preprocessor if available (normalization)
        if policy_bundle.preprocessor is not None:
            observation = policy_bundle.preprocessor(observation)

        action = policy_bundle.policy.select_action(observation)

        # Apply postprocessor if available (unnormalization)
        if policy_bundle.postprocessor is not None:
            if is_new_chunk:
                # --- Bug-fix: unnormalize the full chunk at once ---
                # The stepwise percentile unnormalizer uses per-timestep stats
                # (p02[k], p98[k]).  If we unnormalize one action at a time, it
                # always sees H=1 and applies step-0 stats to every action.
                # Instead, we stack the entire chunk, unnormalize with correct
                # per-step stats, then put the individual actions back.
                queue = policy_bundle.policy._queues["action"]
                remaining = list(queue)
                # Stack: each item is (1, D); result is (1, H, D)
                full_chunk = torch.stack([action] + remaining, dim=1)
                full_chunk = policy_bundle.postprocessor(full_chunk)
                action = full_chunk[:, 0]  # (1, D)
                queue.clear()
                for t in range(1, full_chunk.shape[1]):
                    queue.append(full_chunk[:, t])
            else:
                # Action was already unnormalized when the chunk was processed.
                # No postprocessor call needed.
                pass

        termination_signal = None
        if policy_bundle.has_termination:
            term_index = GET_TERMINATION_IDX(policy_bundle.translator.action_mode)
            termination_signal = action[0, term_index].item()
            if self.verbose:
                print(f"Termination signal: {termination_signal:.4f}")

        return action, termination_signal

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def _finish_step(self, start_time: float, is_inference: bool = False):
        """Wait to maintain control frequency and record timing."""
        elapsed = time.monotonic() - start_time
        sleep_time = self.period - elapsed
        if sleep_time < 0:
            if self.verbose:
                print(
                    f"Warning: cannot maintain desired frequency of {self.config.hz} Hz"
                )
            self._timing.overrun_durations.append(-sleep_time)
            # Record actual step duration (no sleep, took longer than period)
            self._timing.step_durations.append(elapsed)
        else:
            time.sleep(sleep_time)
            # Record exact period as step duration (we slept the remainder)
            self._timing.step_durations.append(time.monotonic() - start_time)
        # Always appended together with step_durations so the two lists
        # stay in sync even when KeyboardInterrupt fires during sleep.
        self._timing.step_is_inference.append(is_inference)

    def print_timing_summary(self):
        """Print timing summary for the current rollout."""
        print(f"\n{self._timing.summary(self.period)}")

    def save_timing_plot(self, output_path: pathlib.Path) -> pathlib.Path:
        """Save a timing analysis plot for the current rollout."""
        return self._timing.save_plot(self.period, output_path)
