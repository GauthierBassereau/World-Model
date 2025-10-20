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

        if attn_mask is not None:
            mask = mask.to(dtype=torch.bool, device=attn_logits.device)
            if mask.shape[-2:] != (time_steps, time_steps):
                raise ValueError("attn_mask last two dims must match sequence length.")

            attn_logits = attn_logits.masked_fill(
                ~mask,
                torch.finfo(attn_logits.dtype).min,
            )

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
            )
        else:
            self.temporal_norm = None
            self.temporal_attn = None

        hidden_dim = int(dim * mlp_multiplier)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = SwiGLUMlp(dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor, # [B, T, S, D]
        spatial_rope: Tuple[torch.Tensor, torch.Tensor], # (cos, sin)
        temporal_rope: Tuple[torch.Tensor, torch.Tensor], # (cos, sin)
        spatial_mask: torch.Tensor, # [S, S]
        temporal_mask: Optional[torch.Tensor], # [T, T]
        update_mask: torch.Tensor, # [S]
    ) -> torch.Tensor:
        spatial_cos, spatial_sin = spatial_rope
        temporal_cos, temporal_sin = temporal_rope

        spatial_out = self.spatial_attn(
            self.spatial_norm(x), 
            spatial_cos, 
            spatial_sin, 
            attn_mask=spatial_mask
        )

        spatial_out = spatial_out * update_mask.view(1, 1, -1, 1)
        x = x + spatial_out

        if self.use_temporal:
            temporal_out = self.temporal_attn(
                self.temporal_norm(x),
                temporal_cos,
                temporal_sin,
                attn_mask=temporal_mask,
            )
            temporal_out = temporal_out * update_mask.view(1, 1, -1, 1)
            x = x + temporal_out

        mlp_out = self.mlp(self.mlp_norm(x))
        mlp_out = mlp_out * update_mask.view(1, 1, -1, 1)
        x = x + mlp_out
        
        return x


@dataclass
class WorldModelConfig:
    latent_dim: int = 1024
    input_dim: int = 768
    action_dim: int = 6
    num_registers: int = 4
    depth: int = 24
    num_heads: int = 12
    mlp_multiplier: float = 4.0
    temporal_attention_interval: int = 4
    temporal_context_length: int = 30
    rope_base: float = 10_000.0
    attn_logit_soft_cap: float = 50.0
    qk_norm_eps: float = 1e-6


class WorldModelBackbone(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config

        self.input_proj = (
            nn.Linear(self.config.input_dim, self.config.latent_dim, bias=False)
            if self.config.input_dim != self.config.latent_dim
            else nn.Identity()
        )

        # For now it is a simple SiLU MLP, like used in other diffusion models, but need to check I think there are more options
        self.noise_embed = nn.Sequential(
            nn.Linear(1, self.config.latent_dim),
            nn.SiLU(),
            nn.Linear(self.config.latent_dim, self.config.latent_dim),
        )

        # The default base action token is a learned embedding, and each action is simply added to it after projection
        self.base_action_embed = nn.Parameter(torch.empty(self.config.latent_dim))
        nn.init.normal_(self.base_action_embed, std=0.02)

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
                )
            )
        self.layers = nn.ModuleList(blocks)

        self.final_norm = RMSNorm(self.config.latent_dim)
        self.output_proj = nn.Linear(self.config.latent_dim, self.config.input_dim)

    @staticmethod
    def _build_spatial_mask(
        latent_tokens: int,
        num_registers: int,
        device: torch.device,
    ) -> torch.Tensor:
        tokens_per_frame = latent_tokens + 2 + num_registers
        mask = torch.zeros(tokens_per_frame, tokens_per_frame, dtype=torch.bool, device=device)
        reg_end = num_registers
        noise_idx = num_registers
        action_idx = num_registers + 1
        latent_start = num_registers + 2
        
        # Latents attend to: themselves, noise, action, registers
        mask[latent_start:, latent_start:] = True  # latents to latents
        mask[latent_start:, noise_idx] = True      # latents to noise
        mask[latent_start:, action_idx] = True     # latents to action
        mask[latent_start:, :reg_end] = True   # latents to registers
        # Noise attends to itself
        mask[noise_idx, noise_idx] = True
        # Action attends to itself
        mask[action_idx, action_idx] = True
        # Registers attend to: latents, noise, action, themselves
        mask[:reg_end, latent_start:] = True   # registers to latents
        mask[:reg_end, noise_idx] = True       # registers to noise
        mask[:reg_end, action_idx] = True      # registers to action
        mask[:reg_end, :reg_end] = True        # registers to registers
        
        return mask

    @staticmethod
    def _build_temporal_mask(
        time_steps: int,
        context_length: Optional[int],
        independant_frame_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.tril(
            torch.ones(time_steps, time_steps, dtype=torch.bool, device=device),
            diagonal=0,
        )
        if context_length is not None and context_length < time_steps:
            context_band = torch.triu(mask, diagonal=-(context_length - 1))
            mask = mask & context_band
        mask = mask.view(1, 1, 1, time_steps, time_steps)
        
        if independant_frame_mask is not None:
            independent = independant_frame_mask.to(device=device, dtype=torch.bool).view(batch_size, 1, 1, 1, 1)
            identity_mask = torch.eye(time_steps, dtype=torch.bool, device=device).view(1, 1, 1, time_steps, time_steps)
            mask = torch.where(independent, identity_mask, mask) # Sequences marked as independent only keep intra-frame temporal attention, which in this case there is actually none, but cleaner like this

        return mask
    
    @staticmethod
    def _build_update_mask(
        latent_tokens: int,
        num_registers: int,
        device: torch.device,
    ) -> torch.Tensor:
        tokens_per_frame = latent_tokens + 2 + num_registers # NOTE here 2 is hardcoded need only one action and one noise level token
        mask = torch.zeros(tokens_per_frame, dtype=torch.bool, device=device)
        
        mask[:num_registers] = True # Registers can be updated
        mask[num_registers + 2:] = True # Latents can be updated
        
        return mask


    def forward(
        self,
        noisy_latents: torch.Tensor, # (batch, frames, tokens, token_dimension)
        noise_levels: torch.Tensor, # (b, f)
        actions: Optional[torch.Tensor] = None, # (b, f, action_dimension)
        # --------- specific arguments for training purposes
        independant_frame_mask: Optional[torch.Tensor] = None, # (b)
        action_mask: Optional[torch.Tensor] = None, # (b)
    ) -> Dict[str, torch.Tensor]:
        # --------- get input info
        bsz, time_steps, latent_tokens, _ = noisy_latents.shape
        device = noisy_latents.device

        # --------- preprocess the inputs and create tokens sequences
        noisy_tokens = self.input_proj(noisy_latents)
        
        actions_proj = self.action_proj(actions).unsqueeze(2)
        actions_embed = self.base_action_embed.view(1, 1, 1, -1).expand(bsz, time_steps, 1, -1)
        action_tokens = torch.where(action_mask.unsqueeze(-1).unsqueeze(-1), actions_embed + actions_proj, actions_embed)
        
        noise_tokens = self.noise_embed(noise_levels.unsqueeze(-1)).unsqueeze(2)
        
        register_tokens = self.register_tokens.unsqueeze(0).unsqueeze(0).expand(bsz, time_steps, -1, -1)

        x = torch.cat((register_tokens, noise_tokens, action_tokens, noisy_tokens), dim=2)

        # --------- create attention masks and positional encodings
        spatial_mask = self._build_spatial_mask(latent_tokens, self.config.num_registers, device)
        temporal_mask = self._build_temporal_mask(time_steps, self.config.temporal_context_length, independant_frame_mask, bsz, device)
        update_mask = self._build_update_mask(latent_tokens, self.config.num_registers, device) # Those special tokens (i.e. registers, actions and noise levels) will not be modified. Not sure this is how it is usually done, need to check
        
        spatial_rope = _rope_cache(
            latent_tokens,
            self.config.latent_dim // self.config.num_heads,
            self.config.rope_base,
            str(device),
        )
        temporal_rope = _rope_cache(
            time_steps,
            self.config.latent_dim // self.config.num_heads,
            self.config.rope_base,
            str(device),
        )

        for block in self.layers:
            x = block(
                x,
                spatial_rope,
                temporal_rope,
                spatial_mask=spatial_mask,
                temporal_mask=temporal_mask,
                update_mask=update_mask,
            )

        latents = self.final_norm(x[..., :latent_tokens, :])
        pred_velocity = self.output_proj(latents)
        
        return {"pred_velocity": pred_velocity}
