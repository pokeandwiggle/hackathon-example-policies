"""Policy-specific saliency implementations."""

from . import ditflow

# Registry of policy-specific modules
# Each module should have: supports_policy(policy) and compute_action_with_gradients(policy, batch)
POLICY_MODULES = [
    ditflow,
    # Add more policy modules here (e.g., diffusion, act, etc.)
]


def get_action_with_gradients(policy, batch):
    """
    Get action prediction with gradients for any supported policy.

    Dispatches to the appropriate policy-specific implementation.

    Args:
        policy: The policy model
        batch: Input batch with gradients enabled on observation tensors

    Returns:
        tuple: (action, processed_batch)

    Raises:
        NotImplementedError: If no implementation supports this policy type
    """
    for module in POLICY_MODULES:
        if module.supports_policy(policy):
            return module.compute_action_with_gradients(policy, batch)

    policy_name = (
        policy.config.name if hasattr(policy.config, "name") else type(policy).__name__
    )
    raise NotImplementedError(
        f"Saliency analysis not implemented for policy type: {policy_name}. "
        f"Add a new module in saliency/policies/ with supports_policy() and "
        f"compute_action_with_gradients() functions."
    )


__all__ = ["get_action_with_gradients"]
