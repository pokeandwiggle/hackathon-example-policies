import time
from typing import Optional

import torch

from example_policies.robot_deploy.deploy_core.deployment_structures import (
    InferenceConfig,
    PolicyBundle,
)
from example_policies.robot_deploy.robot_io.robot_interface import RobotInterface
from example_policies.utils.action_order import GET_TERMINATION_IDX


class InferenceRunner:
    """Core inference loop execution."""

    def __init__(
        self,
        robot_interface: RobotInterface,
        config: InferenceConfig,
    ):
        self.robot_interface = robot_interface
        self.config = config
        self.period = 1.0 / config.hz
        self.step = 0

    def run_step(self, policy_bundle: PolicyBundle) -> Optional[float]:
        """Execute one inference step. Returns termination signal if present."""
        start_time = time.time()
        observation = self.robot_interface.get_observation(policy_bundle.config.device)

        if not observation:
            self._wait_for_period(start_time)
            return None

        with torch.inference_mode():
            action, termination_signal = self._process_action(
                policy_bundle, observation
            )

            print("\n=== RAW MODEL PREDICTION ===")
            policy_bundle.printer.print(self.step, observation, action, raw_action=True)

        action_translated = policy_bundle.translator.translate(action, observation)
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
        return termination_signal

    def _process_action(
        self, policy_bundle: PolicyBundle, observation: dict
    ) -> tuple[torch.Tensor, Optional[float]]:
        """Process action from policy, extracting termination signal if present."""
        action = policy_bundle.policy.select_action(observation)

        termination_signal = None
        if policy_bundle.has_termination:
            term_index = GET_TERMINATION_IDX(policy_bundle.translator.action_mode)
            termination_signal = action[0, term_index].item()
            print(f"Termination signal: {termination_signal:.4f}")

        return action, termination_signal

    def _wait_for_period(self, start_time: float):
        """Wait to maintain control frequency."""
        elapsed = time.time() - start_time
        sleep_time = self.period - elapsed
        if sleep_time < 0:
            print(f"Warning: cannot maintain desired frequency of {self.config.hz} Hz")
        else:
            time.sleep(sleep_time)
