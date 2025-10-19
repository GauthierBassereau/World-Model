"""
DreamerV4-style block-causal transformer used as the world model backbone.

The model treats tokens as a 2D grid with time and space axes. It alternates
space-only attention on every layer with occasional temporal attention layers,
mirroring the architecture described in the DreamerV4 paper.

Expected inputs:
    noisy_latents : [B, T, S, D]     (diffused tokens for each spatial position)
    noise_levels  : [B, T]           (discrete timestep index mapped to [0, 1])
    actions       : [B, T, action_dim] optional
    action_mask   : [B, T] optional  (True when actions are present; False falls back to learned no-action token)

Each timestep is augmented with the following prompts:
    • action token   – projected (or fallback) action embedding (read-only)
    • noise token    – learned embedding of the diffusion level (read-only)
    • register tokens – trainable slots that interact with the latent grid

Spatial attention operates within a timestep; latents communicate with their
frame-specific prompts and registers only. Temporal attention remains causal
and now mixes only corresponding tokens across time, limited to the most recent
`config.temporal_context` frames.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


def _qk_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    denom = torch.rsqrt(torch.clamp(x.pow(2).mean(dim=-1, keepdim=True), min=eps))
    return x * denom


@lru_cache(maxsize=None)
def _rope_cache(length: int, dim: int, base: float, device_str: str) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device_str)
    positions = torch.arange(length, device=device, dtype=torch.float32)
    freqs = base ** (-torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    angles = positions[:, None] * freqs[None, :]
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos, sin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(torch.clamp(x.pow(2).mean(dim=-1, keepdim=True), min=self.eps))
        return self.weight * x * scale


class SwiGLUMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.w1(x)) * self.w2(x)
        return self.proj(gated)


class SpatialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_logit_soft_cap: float,
        qk_norm_eps: float,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads.")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_logit_soft_cap = attn_logit_soft_cap
        self.qk_norm_eps = qk_norm_eps

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, time_steps, tokens, _ = x.shape

        q = self.q_proj(x).view(bsz, time_steps, tokens, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, time_steps, tokens, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, time_steps, tokens, self.num_heads, self.head_dim)

        cos = rope_cos.to(q.dtype).view(1, 1, tokens, 1, self.head_dim)
        sin = rope_sin.to(q.dtype).view(1, 1, tokens, 1, self.head_dim)
        q, k = _apply_rope(q, k, cos, sin)

        q = _qk_norm(q, self.qk_norm_eps)
        k = _qk_norm(k, self.qk_norm_eps)

        q = q.permute(0, 1, 3, 2, 4)  # [B, T, H, S, D]
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.attn_logit_soft_cap > 0:
            cap = self.attn_logit_soft_cap
            attn_logits = cap * torch.tanh(attn_logits / cap)

        if attn_mask is not None:
            if attn_mask.shape != (tokens, tokens):
                raise ValueError("attn_mask must have shape [tokens, tokens].")
            mask = attn_mask.to(dtype=torch.bool, device=attn_logits.device)
            attn_logits = attn_logits.masked_fill(
                ~mask.view(1, 1, 1, tokens, tokens),
                torch.finfo(attn_logits.dtype).min,
            )

        attn_weights = attn_logits.softmax(dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.permute(0, 1, 3, 2, 4).contiguous().view(bsz, time_steps, tokens, -1)
        return self.out_proj(out)


class TemporalAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_logit_soft_cap: float,
        qk_norm_eps: float,
        context_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads.")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_logit_soft_cap = attn_logit_soft_cap
        self.qk_norm_eps = qk_norm_eps
        if context_length is not None and context_length < 1:
            raise ValueError("context_length must be >= 1 when provided.")
        self.context_length = context_length

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        bsz, time_steps, tokens, _ = x.shape

        q = self.q_proj(x).view(bsz, time_steps, tokens, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, time_steps, tokens, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, time_steps, tokens, self.num_heads, self.head_dim)

        cos = rope_cos.to(q.dtype).view(1, time_steps, 1, 1, self.head_dim)
        sin = rope_sin.to(q.dtype).view(1, time_steps, 1, 1, self.head_dim)
        q, k = _apply_rope(q, k, cos, sin)

        q = _qk_norm(q, self.qk_norm_eps)
        k = _qk_norm(k, self.qk_norm_eps)

        q = q.permute(0, 2, 3, 1, 4)  # [B, S, H, T, D]
        k = k.permute(0, 2, 3, 1, 4)
        v = v.permute(0, 2, 3, 1, 4)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.attn_logit_soft_cap > 0:
            cap = self.attn_logit_soft_cap
            attn_logits = cap * torch.tanh(attn_logits / cap)

        mask = torch.ones(
            time_steps,
            time_steps,
            dtype=torch.bool,
            device=x.device,
        )
        mask = torch.tril(mask, diagonal=0)
        if self.context_length is not None:
            max_context = min(self.context_length, time_steps)
            if max_context < time_steps:
                context_band = torch.triu(mask, diagonal=-(max_context - 1))
                mask = mask & context_band
        attn_logits = attn_logits.masked_fill(~mask, torch.finfo(attn_logits.dtype).min)

        attn_weights = attn_logits.softmax(dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.permute(0, 3, 1, 2, 4).contiguous().view(bsz, time_steps, tokens, -1)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_multiplier: float,
        attn_logit_soft_cap: float,
        qk_norm_eps: float,
        use_temporal: bool,
        temporal_context: Optional[int],
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal

        self.spatial_norm = RMSNorm(dim)
        self.spatial_attn = SpatialAttention(
            dim=dim,
            num_heads=num_heads,
            attn_logit_soft_cap=attn_logit_soft_cap,
            qk_norm_eps=qk_norm_eps,
        )

        if use_temporal:
            self.temporal_norm = RMSNorm(dim)
            self.temporal_attn = TemporalAttention(
                dim=dim,
                num_heads=num_heads,
                attn_logit_soft_cap=attn_logit_soft_cap,
                qk_norm_eps=qk_norm_eps,
                context_length=temporal_context,
            )
        else:
            self.temporal_norm = None
            self.temporal_attn = None

        hidden_dim = int(dim * mlp_multiplier)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = SwiGLUMlp(dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        spatial_rope: Tuple[torch.Tensor, torch.Tensor],
        temporal_rope: Tuple[torch.Tensor, torch.Tensor],
        spatial_mask: Optional[torch.Tensor] = None,
        special_token_mask: Optional[torch.Tensor] = None,
        temporal_token_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        spatial_cos, spatial_sin = spatial_rope
        temporal_cos, temporal_sin = temporal_rope

        spatial_out = self.spatial_attn(self.spatial_norm(x), spatial_cos, spatial_sin, attn_mask=spatial_mask)
        if special_token_mask is not None:
            spatial_out = spatial_out.masked_fill(special_token_mask.view(1, 1, -1, 1), 0.0) # Not updating special tokens like action/noise/register
        x = x + spatial_out

        if self.use_temporal and self.temporal_attn is not None and self.temporal_norm is not None:
            temporal_input = self.temporal_norm(x)
            token_updates: Optional[torch.Tensor]
            if temporal_token_indices is not None:
                if temporal_token_indices.numel() == 0:
                    token_updates = None
                else:
                    temporal_subset = temporal_input.index_select(2, temporal_token_indices)
                    temporal_subset_out = self.temporal_attn(temporal_subset, temporal_cos, temporal_sin)
                    token_updates = torch.zeros_like(x)
                    scatter_index = (
                        temporal_token_indices.view(1, 1, -1, 1)
                        .expand(x.shape[0], x.shape[1], -1, x.shape[-1])
                    )
                    token_updates.scatter_(2, scatter_index, temporal_subset_out)
            else:
                token_updates = self.temporal_attn(temporal_input, temporal_cos, temporal_sin)

            if token_updates is not None:
                if special_token_mask is not None:
                    token_updates = token_updates.masked_fill(special_token_mask.view(1, 1, -1, 1), 0.0)
                x = x + token_updates

        mlp_out = self.mlp(self.mlp_norm(x))
        if special_token_mask is not None:
            mlp_out = mlp_out.masked_fill(special_token_mask.view(1, 1, -1, 1), 0.0)
        x = x + mlp_out
        return x


@dataclass
class WorldModelConfig:
    latent_dim: int = 1024
    input_dim: Optional[int] = None
    action_dim: int = 6
    num_registers: int = 4
    depth: int = 24
    num_heads: int = 12
    mlp_multiplier: float = 4.0
    temporal_attention_interval: int = 4
    temporal_context: int = 30
    rope_base: float = 10_000.0
    attn_logit_soft_cap: float = 50.0
    qk_norm_eps: float = 1e-6


class WorldModelBackbone(nn.Module):
    def __init__(self, config: Optional[WorldModelConfig] = None) -> None:
        super().__init__()
        self.config = config or WorldModelConfig()

        if self.config.latent_dim % self.config.num_heads:
            raise ValueError("latent_dim must be divisible by num_heads.")
        input_dim = self.config.input_dim or self.config.latent_dim
        if input_dim > self.config.latent_dim:
            raise ValueError("input_dim cannot exceed latent_dim.")

        self.input_dim = input_dim
        self.input_proj = (
            nn.Linear(self.input_dim, self.config.latent_dim, bias=False)
            if self.input_dim != self.config.latent_dim
            else nn.Identity()
        )

        self.noise_embed = nn.Sequential(
            nn.Linear(1, self.config.latent_dim),
            nn.SiLU(),
            nn.Linear(self.config.latent_dim, self.config.latent_dim),
        )

        self.no_action_embed = nn.Parameter(torch.empty(self.config.latent_dim))
        nn.init.normal_(self.no_action_embed, std=0.02)

        self.action_proj = nn.Linear(self.config.action_dim, self.config.latent_dim)

        self.register_tokens = nn.Parameter(
            torch.empty(self.config.num_registers, self.config.latent_dim)
        )
        nn.init.normal_(self.register_tokens, std=0.02)

        blocks = []
        for layer_idx in range(self.config.depth):
            use_temporal = (layer_idx % self.config.temporal_attention_interval) == 0
            blocks.append(
                TransformerBlock(
                    dim=self.config.latent_dim,
                    num_heads=self.config.num_heads,
                    mlp_multiplier=self.config.mlp_multiplier,
                    attn_logit_soft_cap=self.config.attn_logit_soft_cap,
                    qk_norm_eps=self.config.qk_norm_eps,
                    use_temporal=use_temporal,
                    temporal_context=self.config.temporal_context,
                )
            )
        self.layers = nn.ModuleList(blocks)

        self.final_norm = RMSNorm(self.config.latent_dim)
        self.output_proj = nn.Linear(self.config.latent_dim, self.config.input_dim or self.config.latent_dim)

    def _get_spatial_rope(self, num_tokens: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = _rope_cache(
            num_tokens,
            self.config.latent_dim // self.config.num_heads,
            self.config.rope_base,
            str(device),
        )
        return cos, sin

    def _get_temporal_rope(self, time_steps: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = _rope_cache(
            time_steps,
            self.config.latent_dim // self.config.num_heads,
            self.config.rope_base,
            str(device),
        )
        return cos, sin

    @staticmethod
    def _build_temporal_indices(
        latent_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.arange(latent_tokens, device=device, dtype=torch.long)

    @staticmethod
    def _build_spatial_masks(
        latent_tokens: int,
        num_registers: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens_per_frame = latent_tokens + 2 + num_registers
        spatial_mask = torch.zeros(tokens_per_frame, tokens_per_frame, dtype=torch.bool, device=device)
        spatial_mask[:latent_tokens, :latent_tokens] = True
        spatial_mask[:latent_tokens, latent_tokens:latent_tokens + 2] = True
        if num_registers > 0:
            spatial_mask[:latent_tokens, latent_tokens + 2 :] = True
        spatial_mask[latent_tokens, latent_tokens] = True
        spatial_mask[latent_tokens + 1, latent_tokens + 1] = True
        if num_registers > 0:
            register_slice = slice(latent_tokens + 2, tokens_per_frame)
            spatial_mask[register_slice, :latent_tokens] = True
            spatial_mask[register_slice, latent_tokens:latent_tokens + 2] = True
            spatial_mask[register_slice, register_slice] = True

        special_token_mask = torch.zeros(tokens_per_frame, dtype=torch.bool, device=device)
        special_token_mask[latent_tokens:latent_tokens + 2] = True
        return spatial_mask, special_token_mask

    def forward(
        self,
        noisy_latents: torch.Tensor,
        noise_levels: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        single_frame_mask: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del single_frame_mask  # reserved for future conditioning

        batch, time_steps, latent_tokens, _ = noisy_latents.shape
        device = noisy_latents.device

        x = self.input_proj(noisy_latents)
        action_tokens = self._action_tokens(actions, action_mask, batch, time_steps, x.dtype, device)
        noise_tokens = self.noise_embed(noise_levels.unsqueeze(-1)).unsqueeze(2).to(dtype=x.dtype)

        pieces = [x, action_tokens, noise_tokens]
        if self.config.num_registers:
            registers = (
                self.register_tokens.to(device=device, dtype=x.dtype)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(batch, time_steps, -1, -1)
            )
            pieces.append(registers)

        x = torch.cat(pieces, dim=2)

        spatial_mask, special_token_mask = self._build_spatial_masks(
            latent_tokens,
            self.config.num_registers,
            device=device,
        )

        spatial_rope = self._get_spatial_rope(x.size(2), device)
        temporal_rope = self._get_temporal_rope(time_steps, device)
        temporal_token_indices = self._build_temporal_indices(latent_tokens, device)

        for block in self.layers:
            x = block(
                x,
                spatial_rope,
                temporal_rope,
                spatial_mask=spatial_mask,
                special_token_mask=special_token_mask,
                temporal_token_indices=temporal_token_indices,
            )

        latents = self.final_norm(x[..., :latent_tokens, :])
        pred_velocity = self.output_proj(latents)
        return {"pred_velocity": pred_velocity}

    def _action_tokens(
        self,
        actions: Optional[torch.Tensor],
        action_mask: Optional[torch.Tensor],
        batch: int,
        time_steps: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        base = (
            self.no_action_embed.view(1, 1, 1, -1)
            .to(device=device, dtype=dtype)
            .expand(batch, time_steps, 1, -1)
        )
        if actions is None or self.action_proj is None:
            return base

        projected = self.action_proj(actions).unsqueeze(2).to(dtype=dtype)
        if action_mask is None:
            return base + projected

        mask = action_mask.to(device=device, dtype=torch.bool).unsqueeze(-1).unsqueeze(-1)
        return torch.where(mask, base + projected, base)
