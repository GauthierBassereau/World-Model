"""Output heads for the spatial-temporal world-model backbone.

The wide head follows the lightweight DDT head used by RAE/RAEv2: the main
transformer remains comparatively narrow, while a small number of wider
decoder blocks turn its patch features into the clean-latent prediction.
"""

from functools import lru_cache
from math import isqrt
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.world_model.components import RMSNorm


def _modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return x * (1.0 + scale) + shift


def _rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent feature pairs, matching the RAEv2 2D-RoPE layout."""
    paired = x.unflatten(-1, (-1, 2))
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


@lru_cache(maxsize=None)
def _spatial_rope_cache(
    num_tokens: int,
    head_dim: int,
    base: float,
    device_str: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_size = isqrt(num_tokens)
    if grid_size * grid_size != num_tokens:
        raise ValueError(
            f"The DH head requires a square patch grid, but received {num_tokens} tokens."
        )
    if head_dim % 4 != 0:
        raise ValueError(
            f"The DH head dimension per attention head must be divisible by 4, got {head_dim}."
        )

    device = torch.device(device_str)
    axis_dim = head_dim // 2
    frequencies = base ** (
        -torch.arange(0, axis_dim, 2, device=device, dtype=torch.float32) / axis_dim
    )
    axis_angles = torch.outer(
        torch.arange(grid_size, device=device, dtype=torch.float32),
        frequencies,
    )
    angles = torch.cat(
        (
            axis_angles[:, None, :].expand(-1, grid_size, -1),
            axis_angles[None, :, :].expand(grid_size, -1, -1),
        ),
        dim=-1,
    ).reshape(num_tokens, axis_dim)
    angles = angles.repeat_interleave(2, dim=-1)
    return angles.cos(), angles.sin()


class DHAttention(nn.Module):
    """QK-normalized spatial attention used by the wide decoder head."""

    def __init__(self, dim: int, num_heads: int, qk_norm_eps: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch, tokens, dim = x.shape
        shape = (batch, tokens, self.num_heads, self.head_dim)
        q = self.q(x).view(shape).transpose(1, 2)
        k = self.k(x).view(shape).transpose(1, 2)
        v = self.v(x).view(shape).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)

        cos, sin = rope
        cos = cos.to(dtype=q.dtype).view(1, 1, tokens, self.head_dim)
        sin = sin.to(dtype=q.dtype).view(1, 1, tokens, self.head_dim)
        q = q * cos + _rotate_pairs(q) * sin
        k = k * cos + _rotate_pairs(k) * sin

        output = F.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).contiguous().view(batch, tokens, dim)
        return self.proj(output)


class DHSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class DHDecoderBlock(nn.Module):
    """A wide spatial block modulated patch-by-patch by narrow backbone tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_multiplier: float,
        qk_norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = DHAttention(dim, num_heads, qk_norm_eps)
        self.mlp = DHSwiGLU(dim, int(2.0 / 3.0 * dim * mlp_multiplier))
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        rope: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaln_modulation(condition).chunk(6, dim=-1)
        x = x + gate_attn * self.attn(
            _modulate(self.norm1(x), shift_attn, scale_attn),
            rope,
        )
        x = x + gate_mlp * self.mlp(
            _modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class DHFinalLayer(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.adaln_modulation(condition).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class DHOutputHead(nn.Module):
    """Lightweight, wide DDT/DH prediction head from RAE and RAEv2.

    Temporal information is already present in backbone_tokens. The head runs
    spatially within each frame, exactly like the decoder stage of DDT, and
    predicts clean RAE latents (x-prediction).
    """

    def __init__(
        self,
        input_dim: int,
        backbone_dim: int,
        hidden_dim: int = 2048,
        depth: int = 2,
        num_heads: int = 16,
        mlp_multiplier: float = 4.0,
        qk_norm_eps: float = 1e-6,
        rope_base: float = 10000.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("DH head depth must be at least 1.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"DH hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}."
            )
        if (hidden_dim // num_heads) % 4 != 0:
            raise ValueError("DH attention head dimension must be divisible by 4 for 2D RoPE.")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.rope_base = rope_base
        self.gradient_checkpointing = gradient_checkpointing
        self.noisy_input_proj = nn.Linear(input_dim, hidden_dim)
        self.condition_proj = (
            nn.Linear(backbone_dim, hidden_dim)
            if backbone_dim != hidden_dim
            else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [
                DHDecoderBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_multiplier=mlp_multiplier,
                    qk_norm_eps=qk_norm_eps,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = DHFinalLayer(hidden_dim, input_dim)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

        # RAEv2 initializes its patch-size-1 noisy-input embedder as a flattened
        # convolution, while the remaining head projections retain PyTorch's
        # standard Linear initialization.
        nn.init.xavier_uniform_(self.noisy_input_proj.weight)
        nn.init.zeros_(self.noisy_input_proj.bias)

        for block in self.blocks:
            nn.init.zeros_(block.adaln_modulation[-1].weight)
            nn.init.zeros_(block.adaln_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaln_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaln_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        backbone_tokens: torch.Tensor,
        timestep_embedding: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, tokens, _ = noisy_latents.shape
        condition = F.silu(
            backbone_tokens + timestep_embedding.unsqueeze(dim=2)
        )
        condition = self.condition_proj(condition).flatten(0, 1)
        x = self.noisy_input_proj(noisy_latents).flatten(0, 1)
        rope = _spatial_rope_cache(
            tokens,
            self.hidden_dim // self.num_heads,
            self.rope_base,
            str(noisy_latents.device),
        )
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(
                    lambda hidden, cond, current_block=block: current_block(
                        hidden,
                        cond,
                        rope,
                    ),
                    x,
                    condition,
                    use_reentrant=False,
                )
            else:
                x = block(x, condition, rope)
        output = self.final_layer(x, condition)
        return output.view(batch, frames, tokens, -1)
