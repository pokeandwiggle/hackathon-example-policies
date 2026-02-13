from typing import List

from .deployment_structures import PolicyBundle


class PolicySwitcher:
    """Manages policy switching state and logic (UI-agnostic)."""

    def __init__(self, policies: List[PolicyBundle]):
        self.policies = policies
        self.current_idx = 0
        self.policy_step = 0
        self.global_step = 0

    @property
    def current_policy(self) -> PolicyBundle:
        """Get the currently active policy."""
        return self.policies[self.current_idx]

    def switch_to(self, new_idx: int) -> bool:
        """Switch to a new policy by index. Returns True if policy changed.

        Args:
            new_idx: Index of policy to switch to

        Returns:
            True if policy changed, False if same policy selected
        """
        if not 0 <= new_idx < len(self.policies):
            raise ValueError(f"Invalid policy index: {new_idx}")

        if new_idx != self.current_idx:
            self.current_idx = new_idx
            self.current_policy.reset()
            self.policy_step = 0
            return True

        # Same policy selected - reset step counter
        self.policy_step = 0
        return False

    def increment_steps(self):
        """Increment step counters."""
        self.global_step += 1
        self.policy_step += 1

    def should_prompt(self, prompt_interval: int) -> bool:
        """Check if we should prompt for policy selection based on interval.

        Args:
            prompt_interval: Steps between prompts (0 to disable)

        Returns:
            True if prompt should be shown
        """
        return (
            prompt_interval > 0
            and self.policy_step > 0
            and self.policy_step % prompt_interval == 0
        )
