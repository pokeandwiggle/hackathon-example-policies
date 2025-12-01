from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lerobot.policies.pretrained import PreTrainedPolicy

from example_policies.robot_deploy.deploy_core.action_translator import ActionTranslator
from example_policies.robot_deploy.utils import print_info


@dataclass
class PolicyBundle:
    """Container for policy and related components."""

    policy: PreTrainedPolicy
    config: object
    translator: ActionTranslator
    printer: print_info.InfoPrinter
    checkpoint_path: Path
    has_termination: bool

    @property
    def name(self) -> str:
        return self.checkpoint_path.name

    def reset(self):
        self.policy.reset()


@dataclass
class InferenceConfig:
    """Configuration for inference execution."""

    hz: float = 10.0
    device: str = "cpu"
    controller: Optional[object] = None
