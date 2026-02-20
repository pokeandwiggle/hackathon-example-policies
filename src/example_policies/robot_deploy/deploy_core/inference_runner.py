import time
from typing import Optional

import torch

from example_policies.robot_deploy.deploy_core.deployment_structures import (
    InferenceConfig,
    PolicyBundle,
)
from example_policies.robot_deploy.deploy_core.rollout_recorder import StepResult
from example_policies.robot_deploy.robot_io.robot_interface import RobotInterface
from example_policies.utils.action_order import ActionMode, GET_TERMINATION_IDX


class InferenceRunner:
    """Core inference loop execution."""

    def __init__(
        self,
        robot_interface: RobotInterface,
        config: InferenceConfig,
        verbose: bool = True,
    ):
        self.robot_interface = robot_interface
        self.config = config
        self.period = 1.0 / config.hz
        self.step = 0
        self.verbose = verbose

        # Observation captured at the start of each action chunk.
        # UMI-delta actions are chunk-relative (all steps expressed relative to
        # the TCP at chunk prediction time), so the translator must use this
        # observation rather than the *current* one which has drifted.
        self._chunk_observation: Optional[dict] = None

    def run_step(self, policy_bundle: PolicyBundle) -> Optional[float]:
        """Execute one inference step. Returns termination signal if present."""
        result = self.run_step_recorded(policy_bundle)
        return result.termination_signal

    def run_step_recorded(self, policy_bundle: PolicyBundle) -> StepResult:
        """Execute one inference step. Returns full StepResult with observation, action, and termination signal."""
        start_time = time.monotonic()

        # Reset policy at the very first step
        if self.step == 0:
            if self.verbose:
                print("\n=== RESETTING POLICY ===")
            policy_bundle.policy.reset()
            self._chunk_observation = None
            
        observation = self.robot_interface.get_observation(policy_bundle.config.device)

        if not observation:
            self._wait_for_period(start_time)
            return StepResult()

        # Detect whether a new action chunk will be predicted this step.
        # The policy's action queue is empty → select_action will call
        # predict_action_chunk internally, producing a fresh chunk.
        policy = policy_bundle.policy
        _queues = getattr(policy, "_queues", None)
        is_new_chunk = (
            _queues is not None and len(_queues.get("action", [])) == 0
        )

        with torch.inference_mode():
            action, termination_signal = self._process_action(
                policy_bundle, observation, is_new_chunk=is_new_chunk,
            )

            if self.verbose:
                print("\n=== RAW MODEL PREDICTION ===")
                policy_bundle.printer.print(self.step, observation, action, raw_action=True)

        # --- Bug-fix: use chunk-start observation for UMI-delta translation ---
        # UMI-delta actions are expressed relative to the TCP at the start of
        # the action chunk.  Using the *current* observation (which drifts as
        # the robot moves) introduces exponentially growing rotation errors for
        # steps > 0 in the chunk.
        if is_new_chunk:
            self._chunk_observation = observation

        if (
            policy_bundle.translator.action_mode == ActionMode.UMI_DELTA_TCP
            and self._chunk_observation is not None
        ):
            translator_obs = self._chunk_observation
        else:
            translator_obs = observation

        action_translated = policy_bundle.translator.translate(
            action, translator_obs
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

        self._wait_for_period(start_time)
        self.step += 1
        return StepResult(
            observation=observation,
            action=action,
            termination_signal=termination_signal,
        )

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

    def _wait_for_period(self, start_time: float):
        """Wait to maintain control frequency."""
        elapsed = time.monotonic() - start_time
        sleep_time = self.period - elapsed
        if sleep_time < 0:
            if self.verbose:
                print(f"Warning: cannot maintain desired frequency of {self.config.hz} Hz")
        else:
            time.sleep(sleep_time)
