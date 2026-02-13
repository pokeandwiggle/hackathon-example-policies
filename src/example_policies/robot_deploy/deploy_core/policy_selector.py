from typing import List

from .deployment_structures import PolicyBundle


class PolicySelector:
    """Handles user interface for policy selection."""

    @staticmethod
    def prompt_for_selection(
        policies: List[PolicyBundle],
        current_idx: int,
        global_step: int,
        policy_step: int,
        reason: str = "",
    ) -> int:
        """Display policies and prompt user to select one.

        Args:
            policies: Available policies
            current_idx: Index of currently active policy
            global_step: Total step count
            policy_step: Step count for current policy
            reason: Reason for prompting

        Returns:
            Selected policy index
        """
        PolicySelector._display_header(global_step, policy_step, reason)
        PolicySelector._display_policies(policies, current_idx)
        return PolicySelector._get_user_input(len(policies), current_idx)

    @staticmethod
    def _display_header(global_step: int, policy_step: int, reason: str):
        """Display prompt header."""
        print("\n" + "=" * 70)
        print(
            f"POLICY SELECTION PROMPT (Global Step {global_step}, Policy Step {policy_step})"
        )
        if reason:
            print(f"Reason: {reason}")
        print("=" * 70)

    @staticmethod
    def _display_policies(policies: List[PolicyBundle], current_idx: int):
        """Display available policies."""
        print(f"\nCurrent policy: [{current_idx}] {policies[current_idx].name}")
        print("\nAvailable policies:")

        for i, policy_bundle in enumerate(policies):
            marker = " <-- CURRENT" if i == current_idx else ""
            term_info = (
                " [termination support]" if policy_bundle.has_termination else ""
            )
            print(f"  [{i}] {policy_bundle.name}{term_info}{marker}")

    @staticmethod
    def _get_user_input(num_policies: int, current_idx: int) -> int:
        """Get and validate user selection."""
        while True:
            try:
                user_input = input(
                    f"\nSelect policy [0-{num_policies - 1}] or press Enter to continue: "
                )
                if user_input.strip() == "":
                    return current_idx
                selected_idx = int(user_input)
                if 0 <= selected_idx < num_policies:
                    return selected_idx
                print(
                    f"Invalid selection. Please choose between 0 and {num_policies - 1}"
                )
            except ValueError:
                print("Invalid input. Please enter a number or press Enter.")
            except KeyboardInterrupt:
                raise KeyboardInterrupt("User interrupted policy selection.")
