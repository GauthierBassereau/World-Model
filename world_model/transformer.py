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

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.debugger import TransformerDebugVisualizer


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
            spatial_out = spatial_out.masked_fill(special_token_mask.view(1, 1, -1, 1), 0.0)
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
class TransformerDebugConfig:
    enabled: bool = False
    output_dir: str = "debug/transformer"
    dummy_time_steps: int = 6
    num_patch_plots: int = 4
    num_tokens: int = 226

    def validate(self) -> None:
        if self.dummy_time_steps < 2:
            raise ValueError("world_model.debug.dummy_time_steps must be at least 2.")
        if self.num_patch_plots < 1:
            raise ValueError("world_model.debug.num_patch_plots must be at least 1.")
        if self.num_tokens < 1:
            raise ValueError("world_model.debug.num_tokens must be at least 1.")


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
    debug: TransformerDebugConfig = field(default_factory=TransformerDebugConfig)


class WorldModelBackbone(nn.Module):
    def __init__(self, config: Optional[WorldModelConfig] = None) -> None:
        super().__init__()
        self.config = config or WorldModelConfig()

        if self.config.latent_dim % self.config.num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads.")
        if self.config.temporal_context < 1:
            raise ValueError("WorldModelConfig.temporal_context must be >= 1.")
        if self.config.input_dim is not None:
            if self.config.input_dim < 1:
                raise ValueError("WorldModelConfig.input_dim must be >= 1 when provided.")
            if self.config.input_dim > self.config.latent_dim:
                raise ValueError("WorldModelConfig.input_dim cannot exceed latent_dim.")

        self.input_dim = self.config.input_dim or self.config.latent_dim
        if self.input_dim < 1:
            raise ValueError("Resolved input dimension must be >= 1.")
        if self.input_dim < self.config.latent_dim:
            self.input_proj = nn.Linear(self.input_dim, self.config.latent_dim, bias=False)
        else:
            self.input_proj = None

        self.noise_embed = nn.Sequential(
            nn.Linear(1, self.config.latent_dim),
            nn.SiLU(),
            nn.Linear(self.config.latent_dim, self.config.latent_dim),
        )

        self.no_action_embed = nn.Parameter(torch.empty(self.config.latent_dim))
        nn.init.normal_(self.no_action_embed, std=0.02)

        if self.config.action_dim > 0:
            self.action_proj = nn.Linear(self.config.action_dim, self.config.latent_dim)
        else:
            self.register_parameter("action_proj", None)

        self.register_tokens = nn.Parameter(
            torch.empty(self.config.num_registers, self.config.latent_dim)
        )
        if self.config.num_registers > 0:
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
        self.config.debug.validate()

        if self.config.debug.enabled:
            latent_tokens = max(1, int(self.config.debug.num_tokens))
            total_tokens = latent_tokens + 2 + self.config.num_registers
            spatial_mask_cpu, _ = self._build_spatial_masks(
                latent_tokens,
                self.config.num_registers,
                device=torch.device("cpu"),
            )
            temporal_indices = self._build_temporal_indices(
                latent_tokens,
                device=torch.device("cpu"),
            )
            visualizer = TransformerDebugVisualizer(self.config.debug, self.config)
            visualizer.maybe_run(
                latent_tokens=latent_tokens,
                tokens_per_frame=total_tokens,
                temporal_indices=temporal_indices.cpu(),
                temporal_context=self.config.temporal_context,
                spatial_mask=spatial_mask_cpu.cpu(),
            )

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
        if latent_tokens < 1:
            raise ValueError("latent_tokens must be >= 1 for temporal attention.")
        indices = torch.arange(latent_tokens, device=device, dtype=torch.long)
        return indices

    @staticmethod
    def _build_spatial_masks(
        latent_tokens: int,
        num_registers: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if latent_tokens < 1:
            raise ValueError("latent_tokens must be >= 1 for spatial attention.")
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
        action_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        bsz, time_steps, tokens, _ = noisy_latents.shape
        device = noisy_latents.device

        if action_mask is not None and action_mask.shape != (bsz, time_steps):
            raise ValueError("action_mask must have shape [B, T].")
        action_mask_bool: Optional[torch.Tensor]
        if action_mask is not None:
            action_mask_bool = action_mask.to(dtype=torch.bool, device=device)
        else:
            action_mask_bool = None

        x = noisy_latents
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected noisy_latents last dimension {self.input_dim}, got {x.shape[-1]}."
            )
        if self.input_proj is not None:
            x = self.input_proj(x)

        default_action_token = (
            self.no_action_embed.view(1, 1, 1, -1)
            .to(x.dtype)
            .expand(bsz, time_steps, 1, -1)
        )
        if actions is not None:
            if self.action_proj is None:
                raise ValueError("Model configured without action projection but actions tensor was provided.")
            if actions.dim() != 3:
                raise ValueError("actions tensor must be of shape [B, T, action_dim].")
            if actions.shape[1] != time_steps:
                raise ValueError("actions must align with the time dimension of the latent sequence.")
            projected_actions = self.action_proj(actions).unsqueeze(2).to(x.dtype)
            combined_actions = default_action_token + projected_actions
            if action_mask_bool is not None:
                mask = action_mask_bool.unsqueeze(-1).unsqueeze(-1)
                action_tokens = torch.where(mask, combined_actions, default_action_token)
            else:
                action_tokens = combined_actions
        else:
            if action_mask_bool is not None:
                raise ValueError("action_mask provided without actions tensor.")
            action_tokens = default_action_token

        noise_tokens = self.noise_embed(noise_levels.unsqueeze(-1)).unsqueeze(2).to(x.dtype)

        to_concat = [x, action_tokens, noise_tokens]
        num_registers = self.config.num_registers
        if num_registers > 0:
            register_tokens = (
                self.register_tokens.to(x.dtype)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(bsz, time_steps, -1, -1)
            )
            to_concat.append(register_tokens)

        x = torch.cat(to_concat, dim=2)

        latent_tokens_per_frame = tokens
        tokens_per_frame = latent_tokens_per_frame + 2 + num_registers

        spatial_mask, special_token_mask = self._build_spatial_masks(
            latent_tokens_per_frame,
            num_registers,
            device=device,
        )

        spatial_rope = self._get_spatial_rope(tokens_per_frame, device)
        temporal_rope = self._get_temporal_rope(time_steps, device)
        temporal_token_indices = self._build_temporal_indices(latent_tokens_per_frame, device)

        for block in self.layers:
            x = block(
                x,
                spatial_rope,
                temporal_rope,
                spatial_mask=spatial_mask,
                special_token_mask=special_token_mask,
                temporal_token_indices=temporal_token_indices,
            )

        latent_slice = x[..., :latent_tokens_per_frame, :]
        latent_slice = self.final_norm(latent_slice)
        pred_velocity = self.output_proj(latent_slice)
        return {"pred_velocity": pred_velocity}
