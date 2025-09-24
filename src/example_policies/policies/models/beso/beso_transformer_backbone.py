import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import BESO_TimeEmbedding


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# RMSNorm -- Better, simpler alternative to LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale, self.eps = dim**-0.5, eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# SwishGLU -- A Gated Linear Unit (GLU) with the Swish activation; always better than GELU MLP!
class SwishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.act, self.project = nn.SiLU(), nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)


class Attention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
        block_size: int = 100,
        causal: bool = False,
        bias=False,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd
        self.causal = causal

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash and causal:
            print(
                "WARNING: Using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
        # Dynamically compute causal mask instead of using a fixed bias buffer
        self.block_size = block_size
        self.qk_norm = qk_norm
        # init qk norm if enabled
        if self.qk_norm:
            self.q_norm = RMSNorm(n_embd // self.n_head, eps=1e-6)
            self.k_norm = RMSNorm(n_embd // self.n_head, eps=1e-6)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x, context=None, custom_attn_mask=None):
        B, T, C = x.size()

        if context is not None:
            k = (
                self.key(context)
                .view(B, -1, self.n_head, C // self.n_head)
                .transpose(1, 2)
            )
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = (
                self.value(context)
                .view(B, -1, self.n_head, C // self.n_head)
                .transpose(1, 2)
            )
        else:
            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=custom_attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0,
                is_causal=self.causal,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # Optimize custom attention masking
            if custom_attn_mask is not None:
                att = att.masked_fill(custom_attn_mask == 0, float("-inf"))
            elif self.causal:
                # Dynamically compute causal mask based on current sequence length T
                causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(
                    1, 1, T, T
                )
                att = att.masked_fill(causal_mask == 0, float("-inf"))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(
        self,
        n_embd: int,
        bias: bool,
        use_swish: bool = True,
        use_relus: bool = False,
        dropout: float = 0,
    ):
        super().__init__()
        layers = []

        if use_swish:
            layers.append(SwishGLU(n_embd, 4 * n_embd))
        else:
            layers.append(nn.Linear(n_embd, 4 * n_embd, bias=bias))
            if use_relus:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.GELU())

        layers.append(nn.Linear(4 * n_embd, n_embd, bias=bias))
        layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        attn_pdrop: float,
        resid_pdrop: float,
        mlp_pdrop: float,
        block_size: int = 100,
        causal: bool = True,
        use_cross_attention: bool = False,
        bias: bool = False,  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        qk_norm: bool = True,
    ):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd, eps=1e-6)
        self.attn = Attention(
            n_embd, n_heads, attn_pdrop, resid_pdrop, block_size, causal, bias, qk_norm
        )
        self.use_cross_attention = use_cross_attention

        if self.use_cross_attention:
            self.cross_att = Attention(
                n_embd,
                n_heads,
                attn_pdrop,
                resid_pdrop,
                block_size,
                causal,
                bias,
                qk_norm,
            )
            self.ln3 = RMSNorm(n_embd, eps=1e-6)

        self.ln_2 = RMSNorm(n_embd, eps=1e-6)
        self.mlp = MLP(n_embd, bias=bias, dropout=mlp_pdrop)

    def forward(self, x, context=None, custom_attn_mask=None):
        x = x + self.attn(self.ln_1(x), custom_attn_mask=custom_attn_mask)
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(
                self.ln3(x), context, custom_attn_mask=custom_attn_mask
            )
        x = x + self.mlp(self.ln_2(x))
        return x


class AdaLNZero(nn.Module):
    """
    AdaLN-Zero modulation for conditioning.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # Initialize weights and biases to zero
        # nn.init.zeros_(self.modulation[1].weight)
        # nn.init.zeros_(self.modulation[1].bias)

    def forward(self, c):
        return self.modulation(c).chunk(6, dim=-1)


def modulate(x, shift, scale):
    return shift + (x * (scale))


class ConditionedBlock(Block):
    """
    Block with AdaLN-Zero conditioning.
    """

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        attn_pdrop: float,
        resid_pdrop: float,
        mlp_pdrop: float,
        block_size: int = 100,
        causal: bool = True,
        use_cross_attention: bool = False,
        bias: bool = False,  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        qk_norm: bool = True,
    ):
        super().__init__(
            n_embd,
            n_heads,
            attn_pdrop,
            resid_pdrop,
            mlp_pdrop,
            block_size,
            causal,
            use_cross_attention,
            bias,
            qk_norm,
        )

        self.adaLN_zero = AdaLNZero(n_embd)

    def forward(self, x, c, context=None, custom_attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_zero(c)
        )

        # Attention with modulation
        x_attn = self.ln_1(x)
        x_attn = modulate(x_attn, shift_msa, scale_msa)
        x = x + gate_msa * self.attn(x_attn, custom_attn_mask=custom_attn_mask)

        # Cross attention if used
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(
                self.ln3(x), context, custom_attn_mask=custom_attn_mask
            )

        # MLP with modulation
        x_mlp = self.ln_2(x)
        x_mlp = modulate(x_mlp, shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_mlp)

        return x


class Noise_Dec_only(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        device: str,
        goal_conditioned: bool,
        embed_dim: int,
        embed_pdrob: float,
        goal_seq_len: int,
        obs_seq_len: int,
        action_seq_len: int,
        linear_output: bool = False,
        use_ada_conditioning: bool = False,
        diffusion_type: str = "beso",  # ddpm, beso or rf,
        use_pos_emb: bool = True,
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            n_heads=8,
            attn_pdrop=0.3,
            resid_pdrop=0.1,
            n_layers=6,
            block_size=goal_seq_len
            + obs_seq_len
            + action_seq_len
            + 1,  # +1 for time token
            causal=True,
            bias=False,
            use_cross_attention=False,
        )

        self.device = device

        # mainly used for language condition or goal image condition
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0

        # the seq_size is the number of tokens in the input sequence
        self.seq_size = goal_seq_len + obs_seq_len + action_seq_len

        # linear embedding for the state
        self.tok_emb = nn.Linear(state_dim, embed_dim)

        # linear embedding for the goal
        self.goal_emb = nn.Linear(goal_dim, embed_dim)
        # linear embedding for the action

        self.action_emb = nn.Linear(action_dim, embed_dim)

        self.diffusion_type = diffusion_type

        if diffusion_type == "beso":
            self.sigma_emb = BESO_TimeEmbedding(embed_dim)
        else:
            raise ValueError(f"Diffusion type {diffusion_type} is not supported")

        # position embedding
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_size, embed_dim))
        else:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, goal_seq_len + action_seq_len, embed_dim)
            )

        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        self.use_ada_conditioning = use_ada_conditioning

        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100), nn.GELU(), nn.Linear(100, self.action_dim)
            )
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        # logger.info(
        #     "number of parameters: %e", sum(p.numel() for p in self.parameters())
        # )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, states, actions, goals, sigma):
        b, t, dim = states.size()
        _, t_a, _ = actions.size()

        if self.goal_conditioned:
            goal_embed = self.goal_emb(goals)
            goal_embed += self.pos_emb[:, : self.goal_seq_len, :]
            goal_x = self.drop(goal_embed)
        # print("ACTION SHAPE BEFORE ERROR",actions.shape)
        action_embed = self.action_emb(actions)
        action_x = self.drop(action_embed)

        state_embed = self.tok_emb(states)
        if self.use_pos_emb:
            state_embed += self.pos_emb[
                :, self.goal_seq_len + t_a : (self.goal_seq_len + t_a + t), :
            ]
        state_x = self.drop(state_embed)

        emb_t = self.sigma_emb(sigma)

        if self.goal_conditioned:
            input_seq = torch.cat([emb_t, goal_x, state_x, action_x], dim=1)
        else:
            input_seq = torch.cat([emb_t, state_x, action_x], dim=1)

        if self.use_ada_conditioning:
            encoder_output = self.encoder(input_seq, emb_t)
        else:
            encoder_output = self.encoder(input_seq)

        pred_actions = self.action_pred(encoder_output[:, -self.action_seq_len :, :])
        return pred_actions


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        attn_pdrop: float,
        resid_pdrop: float,
        n_layers: int,
        block_size: int = 100,
        causal: bool = True,
        qk_norm: bool = True,
        bias: bool = False,
        mlp_pdrop: float = 0,
        use_cross_attention: bool = False,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    n_heads,
                    attn_pdrop,
                    resid_pdrop,
                    mlp_pdrop,
                    block_size,
                    causal=causal,
                    use_cross_attention=use_cross_attention,
                    bias=bias,
                    qk_norm=qk_norm,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln = RMSNorm(embed_dim, eps=1e-6)

    def forward(self, x, custom_attn_mask=None):
        for layer in self.blocks:
            x = layer(x, custom_attn_mask=custom_attn_mask)
        x = self.ln(x)
        return x
