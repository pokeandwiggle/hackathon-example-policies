"""Core saliency computation logic."""

import torch

from .policies import get_action_with_gradients


def compute_saliency(policy, batch, image_keys, state_key="observation.state"):
    """
    Compute saliency maps for all image inputs and state vector.

    This function:
    1. Enables gradients on observation tensors
    2. Gets action prediction with gradients (using policy-specific method)
    3. Backpropagates from action output
    4. Extracts and processes gradients as saliency maps

    Args:
        policy: The policy model
        batch: Input batch dictionary (will not be modified)
        image_keys: List of keys corresponding to image observations
        state_key: Key for state vector observation

    Returns:
        tuple: (image_saliency_maps, state_saliency, action_output)
            - image_saliency_maps: Dict mapping image keys to saliency tensors [H, W]
            - state_saliency: Tensor of state gradients [D] or None
            - action_output: The predicted action [action_dim]
    """
    was_training = policy.training
    policy.train()

    # Create batch copy with gradients enabled on observations
    grad_batch = _create_grad_batch(batch, image_keys, state_key)

    # Get action prediction with gradients
    action, processed_batch = get_action_with_gradients(policy, grad_batch)

    if not action.requires_grad:
        raise RuntimeError("Action output does not require gradients.")

    # Backward pass from action output
    action.sum().backward()

    # Extract gradients
    image_saliency_maps = _extract_image_gradients(processed_batch, image_keys)
    state_saliency = _extract_state_gradients(processed_batch, state_key)

    # Restore original training mode
    policy.train(was_training)

    return image_saliency_maps, state_saliency, action.detach().cpu()


def _create_grad_batch(batch, image_keys, state_key):
    """Create a batch copy with gradients enabled on observation tensors."""
    grad_batch = {}
    for key, value in batch.items():
        if key in image_keys or key == state_key:
            if torch.is_tensor(value):
                grad_batch[key] = value.detach().requires_grad_(True)
            else:
                grad_batch[key] = value
        else:
            if torch.is_tensor(value):
                grad_batch[key] = value.detach()
            else:
                grad_batch[key] = value
    return grad_batch


def _extract_image_gradients(processed_batch, image_keys):
    """
    Extract and process image gradients from the processed batch.

    Handles both stacked and unstacked image representations.
    """
    image_saliency_maps = {}

    # Check if images were stacked into "observation.images"
    if (
        "observation.images" in processed_batch
        and processed_batch["observation.images"].grad is not None
    ):
        # Images were stacked - split gradients back to individual cameras
        stacked_grad = processed_batch["observation.images"].grad
        # Shape: [B, T, N_cameras, C, H, W]

        for idx, key in enumerate(image_keys):
            grad_val = stacked_grad[:, :, idx]  # [B, T, C, H, W]
            grad_val = grad_val.mean(
                dim=1
            )  # Average over temporal dimension -> [B, C, H, W]
            grad_val = grad_val.abs()
            saliency = grad_val.max(dim=1)[0]  # Max over channels -> [B, H, W]
            image_saliency_maps[key] = saliency.detach().cpu()
    else:
        # Images weren't stacked - collect from individual keys
        for key in image_keys:
            if key in processed_batch and processed_batch[key].grad is not None:
                grad_val = processed_batch[key].grad
                if grad_val.ndim == 5:  # [B, T, C, H, W]
                    grad_val = grad_val.mean(dim=1)  # [B, C, H, W]
                grad_val = grad_val.abs()
                saliency = grad_val.max(dim=1)[0]  # [B, H, W]
                image_saliency_maps[key] = saliency.detach().cpu()

    return image_saliency_maps


def _extract_state_gradients(processed_batch, state_key):
    """Extract and process state vector gradients."""
    if state_key not in processed_batch or processed_batch[state_key].grad is None:
        return None

    grad_val = processed_batch[state_key].grad
    if grad_val.ndim == 3:  # [B, T, D]
        grad_val = grad_val.mean(dim=1)  # [B, D]
    return grad_val.abs().detach().cpu()
