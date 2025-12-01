from pathlib import Path
from typing import List

from ..utils import print_info
from .action_translator import ActionTranslator
from .deployment_structures import PolicyBundle
from .policy_loader import load_policy


class PolicyManager:
    """Manages loading and lifecycle of policies."""

    @staticmethod
    def _has_termination_signal(cfg) -> bool:
        """Check if a policy supports termination signal output."""
        if not hasattr(cfg, "metadata") or cfg.metadata is None:
            return False
        try:
            action_names = cfg.metadata["features"]["action"]["names"]
            return "termination_signal" in action_names
        except (KeyError, TypeError):
            return False

    @staticmethod
    def load_single(checkpoint_path: Path, device: str) -> PolicyBundle:
        """Load a single policy from checkpoint path."""
        print(f"Loading checkpoint: {checkpoint_path.name}")
        policy, cfg = load_policy(checkpoint_path)
        policy.to(device)

        policy_bundle = PolicyBundle(
            policy=policy,
            config=cfg,
            translator=ActionTranslator(cfg),
            printer=print_info.InfoPrinter(cfg),
            checkpoint_path=checkpoint_path,
            has_termination=PolicyManager._has_termination_signal(cfg),
        )
        print(f"  Termination signal support: {policy_bundle.has_termination}")
        return policy_bundle

    @staticmethod
    def load_multiple(checkpoint_paths: List[Path], device: str) -> List[PolicyBundle]:
        """Load multiple policies from checkpoint paths."""
        policies = []
        for checkpoint_path in checkpoint_paths:
            policy_bundle = PolicyManager.load_single(checkpoint_path, device)
            policies.append(policy_bundle)
        return policies
