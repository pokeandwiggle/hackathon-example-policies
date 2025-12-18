"""Base interface for policy-specific saliency implementations."""


def compute_action_with_gradients(policy, batch):
    """
    Compute action prediction with gradients enabled.

    This should bypass any @torch.no_grad() decorators in the policy's
    inference path and return actions with gradient tracking.

    Args:
        policy: The policy model
        batch: Input batch with gradients enabled on observation tensors

    Returns:
        tuple: (action, processed_batch)
            - action: Predicted action tensor with gradients
            - processed_batch: Batch dict with retained gradients on observations
    """
    raise NotImplementedError("Subclasses must implement compute_action_with_gradients")
