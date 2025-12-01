"""DiT Flow policy saliency implementation."""

import torch


def supports_policy(policy) -> bool:
    """Check if policy is a DiT Flow model."""
    return hasattr(policy, "dit_flow")


def compute_action_with_gradients(policy, batch):
    """
    Compute DiT Flow action with gradients.

    Bypasses @torch.no_grad() by manually preprocessing and calling
    the velocity network directly.

    Returns:
        tuple: (action, processed_batch)
    """
    # Normalize inputs (creates new tensors without gradients)
    batch = policy.normalize_inputs(batch)

    # Re-enable gradients on observations
    image_keys = [k for k in batch.keys() if k.startswith("observation.images.")]
    state_key = "observation.state"
    for key in image_keys + [state_key]:
        if key in batch and torch.is_tensor(batch[key]):
            batch[key] = batch[key].detach().requires_grad_(True)

    # Stack images for the model
    if policy.config.image_features:
        batch = dict(batch)
        batch["observation.images"] = torch.stack(
            [batch[key] for key in policy.config.image_features], dim=-4
        )

    # Add temporal dimension and retain gradients
    keys_to_retain = set(image_keys) | {state_key, "observation.images"}
    n_obs_steps = policy.config.n_obs_steps
    for key in batch:
        if key.startswith("observation.") and torch.is_tensor(batch[key]):
            batch[key] = (
                batch[key]
                .unsqueeze(1)
                .repeat(1, n_obs_steps, *([1] * (batch[key].ndim - 1)))
            )
            if key in keys_to_retain:
                batch[key].retain_grad()

    # Encode observations into conditioning vector
    global_cond = policy.dit_flow._prepare_global_conditioning(batch)

    # Compute velocity at a fixed point (instead of running full ODE sampling)
    batch_size = global_cond.shape[0]
    device = global_cond.device
    dtype = global_cond.dtype

    x = torch.zeros(
        batch_size,
        policy.config.horizon,
        policy.config.action_feature.shape[0],
        device=device,
        dtype=dtype,
        requires_grad=False,
    )
    t = torch.ones(batch_size, device=device, dtype=dtype) * 0.5

    # Get velocity prediction (has gradients w.r.t. global_cond)
    velocity = policy.dit_flow.velocity_net.forward(x, t, global_cond)

    # Extract action steps from velocity
    start = n_obs_steps - 1
    end = start + policy.config.n_action_steps
    actions = velocity[:, start:end, :]

    return actions[:, 0], batch
