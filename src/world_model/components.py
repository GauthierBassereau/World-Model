from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

@lru_cache(maxsize=None)
def _rope_cache(length: int, dim: int, base: float, device_str: str) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device_str)
    positions = torch.arange(length, device=device, dtype=torch.float32)
    freqs = base ** (-torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    angles = positions[:, None] * freqs[None, :]
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos, sin


class SignalEmbedder(nn.Module):
    def __init__(self, model_dim, base_freq_dim=256, scale=1000.0, max_period=10000):
        super().__init__()
        self.model_dim = model_dim
        self.scale = scale
        
        self.mlp = nn.Sequential(
            nn.Linear(base_freq_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        half_dim = base_freq_dim // 2
        emb = math.log(max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer('freqs', emb)

    def forward(self, signal_levels):
        scaled_signal = signal_levels * self.scale  
        args = scaled_signal[:, None].float() * self.freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        embedding = self.mlp(embedding)
        token = embedding.unsqueeze(1)
        return token


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
        self.w12 = nn.Linear(dim, 2*hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm_eps: float = 1e-6,
        attn_logit_softcapping: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_logit_softcapping = attn_logit_softcapping

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, _ = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            cos, sin = rope
            if cos.ndim == 2:
                cos = cos[-L:, :].view(1, 1, L, -1)
                sin = sin[-L:, :].view(1, 1, L, -1)
            q, k = _apply_rope(q, k, cos, sin)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        if self.attn_logit_softcapping is not None:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = torch.tanh(attn / self.attn_logit_softcapping) * self.attn_logit_softcapping
            if mask is not None:
                attn = attn + mask
            x = (attn.softmax(dim=-1) @ v)
        else: # Softcapping not supported with fused attention...
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        x = x.transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(x), (k, v)


class TransformerBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        mlp_multiplier: float, 
        qk_norm_eps: float, 
        attn_logit_softcapping: Optional[float], 
        use_temporal: bool
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal

        self.spatial_norm = RMSNorm(dim)
        self.spatial_attn = Attention(dim, num_heads, qk_norm_eps, attn_logit_softcapping)
        
        self.temporal_norm = RMSNorm(dim) if use_temporal else None
        self.temporal_attn = Attention(dim, num_heads, qk_norm_eps, attn_logit_softcapping) if use_temporal else None

        self.mlp_norm = RMSNorm(dim)
        self.mlp = SwiGLUMlp(dim, int(dim * mlp_multiplier))

    def forward(
        self,
        x: torch.Tensor,
        spatial_rope: Tuple[torch.Tensor, torch.Tensor],
        temporal_rope: Tuple[torch.Tensor, torch.Tensor],
        spatial_mask: torch.Tensor,
        temporal_mask: Optional[torch.Tensor],
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, S, T, D = x.shape
        
        resid = x
        x_spatial = self.spatial_norm(x).view(B * S, T, D)
        x_spatial, _ = self.spatial_attn(x_spatial, spatial_rope, spatial_mask)
        x = resid + x_spatial.view(B, S, T, D)

        new_kv = None
        if self.use_temporal:
            resid = x
            x_temporal = self.temporal_norm(x).permute(0, 2, 1, 3).reshape(B * T, S, D)
            
            t_mask = temporal_mask
            if t_mask is not None and t_mask.dim() == 4: # [B, 1, S, S_k]
                t_mask = t_mask.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(B * T, 1, S, -1)

            x_temporal, new_kv = self.temporal_attn(x_temporal, temporal_rope, t_mask, kv_cache)
            x = resid + x_temporal.view(B, T, S, D).permute(0, 2, 1, 3)
        
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_kv