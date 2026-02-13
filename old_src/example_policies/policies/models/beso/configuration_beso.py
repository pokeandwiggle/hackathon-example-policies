from lerobot.policies.diffusion.configuration_diffusion import (
    DiffusionConfig,
    PreTrainedConfig,
)


@PreTrainedConfig.register_subclass("beso")
class BesoConfig(DiffusionConfig):
    embed_dim = 512
    n_obs_steps = 1
