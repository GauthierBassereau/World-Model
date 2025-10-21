from functools import lru_cache
from typing import Optional, Tuple

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
            mask = attn_mask.to(dtype=torch.bool, device=attn_logits.device).view(1, 1, tokens, tokens)
            attn_logits = attn_logits.masked_fill(~mask, torch.finfo(attn_logits.dtype).min)

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
        attn_mask: Optional[torch.Tensor] = None, # [B, S, T, D]
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

        q = q.permute(0, 2, 3, 1, 4)
        k = k.permute(0, 2, 3, 1, 4)
        v = v.permute(0, 2, 3, 1, 4)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.attn_logit_soft_cap > 0:
            cap = self.attn_logit_soft_cap
            attn_logits = cap * torch.tanh(attn_logits / cap)

        if attn_mask is not None:
            mask = attn_mask.to(dtype=torch.bool, device=attn_logits.device)
            if mask.shape != (bsz, 1, time_steps, time_steps):
                raise ValueError("attn_mask last two dims must match sequence length.")

            mask = mask.unsqueeze(2)  # [B, 1, 1, T, T]
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
        frozen_prefix_tokens: int = 0,
    ) -> None:
        super().__init__()
        assert frozen_prefix_tokens > 0, "frozen_prefix_tokens variable needs to be > 0"
        
        self.use_temporal = use_temporal
        self.frozen_prefix_tokens = frozen_prefix_tokens

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

    def _build_update_mask(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        tokens = x.shape[2]
        mask = torch.ones(tokens, dtype=x.dtype, device=x.device)
        mask[:self.frozen_prefix_tokens] = 0.0
        return mask.view(1, 1, tokens, 1)

    def forward(
        self,
        x: torch.Tensor, # [B, S, T, D]
        spatial_rope: Tuple[torch.Tensor, torch.Tensor], # (cos, sin)
        temporal_rope: Tuple[torch.Tensor, torch.Tensor], # (cos, sin)
        spatial_mask: torch.Tensor, # [B, 1, S, S]
        temporal_mask: Optional[torch.Tensor], # [B, 1, T, T]
    ) -> torch.Tensor:
        spatial_cos, spatial_sin = spatial_rope
        temporal_cos, temporal_sin = temporal_rope
        update_mask = self._build_update_mask(x)

        spatial_out = self.spatial_attn(
            self.spatial_norm(x), 
            spatial_cos, 
            spatial_sin, 
            attn_mask=spatial_mask
        )
        spatial_out = spatial_out * update_mask

        x = x + spatial_out

        if self.use_temporal:
            temporal_out = self.temporal_attn(
                self.temporal_norm(x),
                temporal_cos,
                temporal_sin,
                attn_mask=temporal_mask,
            )
            temporal_out = temporal_out * update_mask
            
            x = x + temporal_out

        mlp_out = self.mlp(self.mlp_norm(x))
        mlp_out = mlp_out * update_mask
            
        x = x + mlp_out
        
        return x
