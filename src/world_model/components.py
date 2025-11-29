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

def _build_attention_bias(
    mask: torch.Tensor,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    allowed = mask.to(device=device, dtype=torch.bool)
    if allowed.dim() == 2:
        allowed = allowed.unsqueeze(0)
    if allowed.dim() != 3:
        raise ValueError("attention mask must have rank 2 or 3.")
    if allowed.shape[0] == 1 and batch_size != 1:
        allowed = allowed.expand(batch_size, -1, -1)
    elif allowed.shape[0] != batch_size:
        raise ValueError("attention mask batch size mismatch.")

    bias = torch.zeros(
        batch_size,
        1,
        allowed.shape[1],
        allowed.shape[2],
        dtype=dtype,
        device=device,
    )
    bias = bias.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)
    return bias

class TimestepEmbedder(nn.Module):
    # https://github.com/facebookresearch/DiT/blob/main/models.py
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


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
        qkv_bias: bool = False,
        qk_norm_eps: Optional[float] = 1e-6,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        fused_attn: bool = True,
        attn_logit_softcapping: Optional[float] = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads.")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        self.attn_logit_softcapping = attn_logit_softcapping

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if qk_norm_eps is None:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        else:
            self.q_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)

        if rope is not None:
            cos, sin = rope
            cos = cos.to(dtype=q.dtype, device=q.device).view(1, seq_len, 1, self.head_dim)
            sin = sin.to(dtype=q.dtype, device=q.device).view(1, seq_len, 1, self.head_dim)
            q, k = _apply_rope(q, k, cos, sin)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.permute(0, 2, 1, 3)  # [B, H, L, D]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        new_kv_cache = (k, v)

        attn_mask_formatted: Optional[torch.Tensor] = None
        if attn_mask is not None:
            attn_mask_formatted = attn_mask.to(dtype=q.dtype, device=q.device)

        use_fused = self.fused_attn and (self.attn_logit_softcapping is None)

        if use_fused:
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask_formatted,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if self.attn_logit_softcapping is not None:
                attn_logits = attn_logits / self.attn_logit_softcapping
                attn_logits = torch.tanh(attn_logits)
                attn_logits = attn_logits * self.attn_logit_softcapping

            if attn_mask_formatted is not None:
                attn_logits = attn_logits + attn_mask_formatted
            
            attn_weights = attn_logits.softmax(dim=-1)
            attn_weights = self.attn_drop(attn_weights)
            attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        return attn_output, new_kv_cache



class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_multiplier: float,
        qk_norm_eps: float,
        use_temporal: bool,
        num_registers: int,
        frozen_token_index: int = 0, # Defaults to Noise token at index 0
        attn_logit_softcapping: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal
        self.num_registers = num_registers
        self.frozen_token_index = frozen_token_index

        # Indices for slicing: [Noise (freeze) | Action (update) | Registers (update) | Latents (update)]

        self.spatial_norm = RMSNorm(dim)
        self.spatial_attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=False,
            qk_norm_eps=qk_norm_eps,
            attn_logit_softcapping=attn_logit_softcapping,
        )

        if use_temporal:
            self.temporal_norm = RMSNorm(dim)
            self.temporal_attn = Attention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=False,
                qk_norm_eps=qk_norm_eps,
                attn_logit_softcapping=attn_logit_softcapping,
            )
        else:
            self.temporal_norm = None
            self.temporal_attn = None

        hidden_dim = int(dim * mlp_multiplier)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = SwiGLUMlp(dim, hidden_dim)

    def _build_update_mask(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.shape[2]
        mask = torch.ones(tokens, dtype=x.dtype, device=x.device)
        mask[self.frozen_token_index] = 0.0
        
        return mask.view(1, 1, tokens, 1)

    def forward(
        self,
        x: torch.Tensor, # [B, S, T, D]
        spatial_rope: Tuple[torch.Tensor, torch.Tensor], 
        temporal_rope: Tuple[torch.Tensor, torch.Tensor],
        spatial_mask: torch.Tensor, 
        temporal_mask: Optional[torch.Tensor],
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        spatial_cos, spatial_sin = spatial_rope
        update_mask = self._build_update_mask(x)
        bsz, time_steps, tokens, dim = x.shape
        
        spatial_input = self.spatial_norm(x).reshape(bsz * time_steps, tokens, dim)
        spatial_bias = _build_attention_bias(
            spatial_mask,
            batch_size=bsz * time_steps,
            dtype=spatial_input.dtype,
            device=spatial_input.device,
        )
        spatial_out, _ = self.spatial_attn(
            spatial_input,
            rope=(spatial_cos, spatial_sin),
            attn_mask=spatial_bias,
        )
        spatial_out = spatial_out.view(bsz, time_steps, tokens, dim) * update_mask
        x = x + spatial_out
        

        active_indices = torch.cat([
            torch.arange(0, self.frozen_token_index, device=x.device),
            torch.arange(self.frozen_token_index + 1, tokens, device=x.device)
        ])

        if self.use_temporal:
            x_active = x.index_select(2, active_indices) # [B, T, Active_Tokens, D]
            active_tokens_count = x_active.shape[2]
            
            temporal_cos, temporal_sin = temporal_rope
            temporal_input = self.temporal_norm(x_active).permute(0, 2, 1, 3).reshape(bsz * active_tokens_count, time_steps, dim)
            
            temporal_bias = None
            if temporal_mask is not None:
                expanded_mask = temporal_mask.view(bsz, 1, time_steps, time_steps)
                expanded_mask = expanded_mask.expand(bsz, active_tokens_count, time_steps, time_steps)
                expanded_mask = expanded_mask.reshape(bsz * active_tokens_count, time_steps, time_steps)
                temporal_bias = _build_attention_bias(
                    expanded_mask,
                    batch_size=bsz * active_tokens_count,
                    dtype=temporal_input.dtype,
                    device=temporal_input.device,
                )
                
            temporal_out, new_kv_cache = self.temporal_attn(
                temporal_input,
                rope=(temporal_cos, temporal_sin),
                attn_mask=temporal_bias,
                kv_cache=kv_cache,
            )
            
            temporal_out = temporal_out.view(bsz, active_tokens_count, time_steps, dim).permute(0, 2, 1, 3)
            
            full_temporal_out = torch.zeros_like(x, dtype=temporal_out.dtype)
            full_temporal_out.index_copy_(2, active_indices, temporal_out)
            
            x = x + full_temporal_out

        x_active = x.index_select(2, active_indices)
        mlp_out = self.mlp(self.mlp_norm(x_active))
        
        full_mlp_out = torch.zeros_like(x, dtype=mlp_out.dtype)
        full_mlp_out.index_copy_(2, active_indices, mlp_out)
            
        x = x + full_mlp_out
        
        return x, new_kv_cache if self.use_temporal else None