import math
from functools import partial

import einops
import torch
from torch import nn


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less than input dimensions"
        )
    return x[(...,) + (None,) * dims_to_append]


def make_sample_density(
    sigma_sample_density_type, sigma_max=80, sigma_min=0.001, sigma_data=0.5
):
    if sigma_sample_density_type == "loglogistic":
        loc = math.log(sigma_data)
        scale = 0.5
        min_value = sigma_min
        max_value = sigma_max
        return partial(
            rand_log_logistic,
            loc=loc,
            scale=scale,
            min_value=min_value,
            max_value=max_value,
        )


def rand_log_logistic(
    shape,
    loc=0.0,
    scale=1.0,
    min_value=0.0,
    max_value=float("inf"),
    device="cpu",
    dtype=torch.float32,
):
    """Draws samples from an optionally truncated log-logistic distribution."""
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = (
        torch.rand(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf)
        + min_cdf
    )
    return u.logit().mul(scale).add(loc).exp().to(dtype)


def get_sigmas_exponential(n, sigma_min, sigma_max, device="cpu"):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(
        math.log(sigma_max), math.log(sigma_min), n, device=device
    ).exp()
    return torch.cat([sigmas, sigmas.new_zeros([1])])


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class BESO_TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.sigma_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, sigma):
        sigmas = sigma.log() / 4
        sigmas = einops.rearrange(sigmas, "b -> b 1")
        emb_t = self.sigma_emb(sigmas)
        if len(emb_t.shape) == 2:
            emb_t = einops.rearrange(emb_t, "b d -> b 1 d")
        return emb_t


@torch.no_grad()
def sample_ddim(
    model,
    state,
    action,
    goal,
    sigmas,
    scaler=None,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
):
    """
    DPM-Solver 1( or DDIM sampler"""
    extra_args = {} if extra_args is None else extra_args
    s_in = action.new_ones([action.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    # print("sigmas:", sigmas)

    for i in range(len(sigmas) - 1):
        # predict the next action
        if isinstance(state, tuple):
            denoised = model(
                state[0], state[1], action, goal, sigmas[i] * s_in, **extra_args
            )
        else:
            denoised = model(state, action, goal, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "action": action,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        # print("h",h,"Change h",(-h).expm1())
        action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised
    return action
