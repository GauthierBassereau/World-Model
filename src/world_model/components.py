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
    ) -> torch.Tensor:
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
        return attn_output



class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_multiplier: float,
        qk_norm_eps: float,
        use_temporal: bool,
        num_registers: int,
        frozen_conditioning_tokens: int = 2, # Defaults to Noise + Action
        attn_logit_softcapping: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal
        self.num_registers = num_registers
        self.frozen_conditioning_tokens = frozen_conditioning_tokens

        # Indices for slicing: [Registers (update) | Conditioning (freeze) | Latents (update)]

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
        # [B, T, Tokens, D]
        tokens = x.shape[2]
        mask = torch.ones(tokens, dtype=x.dtype, device=x.device)
        
        # Only freeze the conditioning tokens (Noise/Action)
        start_freeze = self.num_registers
        end_freeze = self.num_registers + self.frozen_conditioning_tokens
        
        mask[start_freeze:end_freeze] = 0.0
        
        return mask.view(1, 1, tokens, 1)

    def forward(
        self,
        x: torch.Tensor, # [B, S, T, D]
        spatial_rope: Tuple[torch.Tensor, torch.Tensor], 
        temporal_rope: Tuple[torch.Tensor, torch.Tensor],
        spatial_mask: torch.Tensor, 
        temporal_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        spatial_cos, spatial_sin = spatial_rope
        update_mask = self._build_update_mask(x)
        bsz, time_steps, tokens, dim = x.shape

        # 1. Spatial Attention (Full Context)
        spatial_input = self.spatial_norm(x).reshape(bsz * time_steps, tokens, dim)
        spatial_bias = _build_attention_bias(
            spatial_mask,
            batch_size=bsz * time_steps,
            dtype=spatial_input.dtype,
            device=spatial_input.device,
        )
        spatial_out = self.spatial_attn(
            spatial_input,
            rope=(spatial_cos, spatial_sin),
            attn_mask=spatial_bias,
        )
        spatial_out = spatial_out.view(bsz, time_steps, tokens, dim) * update_mask
        x = x + spatial_out

        # 2. Temporal Attention (Optimization: Skip Frozen Tokens)
        # Only run temporal attention on tokens that are actually allowed to update (Registers + Latents)
        start_freeze = self.num_registers
        end_freeze = self.num_registers + self.frozen_conditioning_tokens
        
        # Creating a boolean mask of tokens to process
        active_indices = torch.cat([
            torch.arange(0, start_freeze, device=x.device),
            torch.arange(end_freeze, tokens, device=x.device)
        ])

        if self.use_temporal:
            # Extract only active tokens for temporal processing
            x_active = x.index_select(2, active_indices) # [B, T, Active_Tokens, D]
            active_tokens_count = x_active.shape[2]
            
            temporal_cos, temporal_sin = temporal_rope
            temporal_input = self.temporal_norm(x_active).permute(0, 2, 1, 3).reshape(bsz * active_tokens_count, time_steps, dim)
            
            temporal_bias = None
            if temporal_mask is not None:
                # temporal_mask is [B, 1, T, T]
                expanded_mask = temporal_mask.view(bsz, 1, time_steps, time_steps)
                expanded_mask = expanded_mask.expand(bsz, active_tokens_count, time_steps, time_steps)
                expanded_mask = expanded_mask.reshape(bsz * active_tokens_count, time_steps, time_steps)
                temporal_bias = _build_attention_bias(
                    expanded_mask,
                    batch_size=bsz * active_tokens_count,
                    dtype=temporal_input.dtype,
                    device=temporal_input.device,
                )
                
            temporal_out = self.temporal_attn(
                temporal_input,
                rope=(temporal_cos, temporal_sin),
                attn_mask=temporal_bias,
            )
            
            temporal_out = temporal_out.view(bsz, active_tokens_count, time_steps, dim).permute(0, 2, 1, 3)
            
            # Scatter back to full size variable (initialized as zero so frozen stays zero)
            full_temporal_out = torch.zeros_like(x)
            full_temporal_out.index_copy_(2, active_indices, temporal_out)
            
            x = x + full_temporal_out

        x_active = x.index_select(2, active_indices)
        mlp_out = self.mlp(self.mlp_norm(x_active))
        
        full_mlp_out = torch.zeros_like(x)
        full_mlp_out.index_copy_(2, active_indices, mlp_out)
            
        x = x + full_mlp_out
        
        return x


# ==============================================================================
# Updated Backbone & Config
# ==============================================================================

@dataclass
class WorldModelConfig:
    latent_dim: int = 1024
    input_dim: int = 768
    action_dim: int = 8 # Updated to match YAML (7 joint + 1 gripper)
    num_registers: int = 4
    depth: int = 24
    num_heads: int = 16
    mlp_multiplier: float = 4.0
    temporal_attention_interval: int = 4
    temporal_context_length: int = 9 # Updated to match YAML
    rope_base: float = 1000.0
    qk_norm_eps: float = 1e-6
    attn_logit_softcapping: Optional[float] = 50.0 # Updated to match YAML (allows fused attn disable check)


class WorldModelBackbone(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config

        self.input_proj = (
            nn.Linear(self.config.input_dim, self.config.latent_dim, bias=False)
            if self.config.input_dim != self.config.latent_dim
            else nn.Identity()
        )

        self.noise_embed = nn.Sequential(
            nn.Linear(1, self.config.latent_dim),
            nn.SiLU(),
            nn.Linear(self.config.latent_dim, self.config.latent_dim),
        )

        self.base_action_embed = nn.Parameter(torch.empty(self.config.latent_dim))
        self.action_proj = nn.Linear(self.config.action_dim, self.config.latent_dim)

        self.register_tokens = nn.Parameter(
            torch.empty(self.config.num_registers, self.config.latent_dim)
        )

        blocks = []
        for layer_idx in range(self.config.depth):
            use_temporal = (layer_idx % self.config.temporal_attention_interval) == 0
            blocks.append(
                TransformerBlock(
                    dim=self.config.latent_dim,
                    num_heads=self.config.num_heads,
                    mlp_multiplier=self.config.mlp_multiplier,
                    qk_norm_eps=self.config.qk_norm_eps,
                    use_temporal=use_temporal,
                    num_registers=self.config.num_registers,
                    frozen_conditioning_tokens=2, # Noise + Action
                    attn_logit_softcapping=config.attn_logit_softcapping,
                )
            )
        self.layers = nn.ModuleList(blocks)

        self.final_norm = RMSNorm(self.config.latent_dim)
        self.output_proj = nn.Linear(self.config.latent_dim, self.config.input_dim)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        # Fix: Improved Initialization for Deep Transformers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                # Truncated normal is better for deep transformers than Xavier
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        nn.init.normal_(self.base_action_embed, std=0.02)
        nn.init.normal_(self.register_tokens, std=0.02)
        
        nn.init.zeros_(self.noise_embed[-1].weight)
        nn.init.zeros_(self.noise_embed[-1].bias)
        
        with torch.no_grad():
            self.action_proj.weight.mul_(0.1) 
        nn.init.zeros_(self.action_proj.bias)
        
        # Zero-init output projections for identity-at-init behavior
        for block in self.layers:
            nn.init.zeros_(block.spatial_attn.out_proj.weight)
            if block.temporal_attn is not None:
                nn.init.zeros_(block.temporal_attn.out_proj.weight)
            nn.init.zeros_(block.mlp.w3.weight)
        
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

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
        mask[latent_start:, latent_start:] = True
        mask[latent_start:, noise_idx] = True
        mask[latent_start:, action_idx] = True
        mask[latent_start:, :reg_end] = True
        
        # Noise attends to itself
        mask[noise_idx, noise_idx] = True
        
        # Action attends to itself
        mask[action_idx, action_idx] = True
        
        # Registers attend to: latents, noise, action, themselves
        mask[:reg_end, latent_start:] = True
        mask[:reg_end, noise_idx] = True
        mask[:reg_end, action_idx] = True
        mask[:reg_end, :reg_end] = True
        
        return mask

    @staticmethod
    def _build_temporal_mask(
        time_steps: int,
        context_length: Optional[int],
        independent_frames_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        # Fix: Diagonal 0 (Causal) instead of -1 (Strictly Past)
        mask = torch.tril(
            torch.ones(time_steps, time_steps, dtype=torch.bool, device=device),
            diagonal=0, 
        )
        if context_length is not None and context_length < time_steps:
            context_band = torch.triu(mask, diagonal=-(context_length))
            mask = mask & context_band
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, time_steps, time_steps).clone()
        
        if independent_frames_mask is not None:
            independent = independent_frames_mask.to(device=device, dtype=torch.bool).view(batch_size, 1, 1, 1)
            all_false = torch.zeros_like(mask, dtype=torch.bool)
            mask = torch.where(independent, all_false, mask)

        return mask

    def forward(
        self,
        noisy_latents: torch.Tensor, # [Batch, Sequence, Tokens, Dimension]
        noise_levels: torch.Tensor, # [B, S, 1]
        actions: Optional[torch.Tensor] = None, # [B, S, D]
        independent_frames_mask: Optional[torch.Tensor] = None, # [B, 1]
        actions_mask: Optional[torch.Tensor] = None, # [B, 1]
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps, latent_tokens, _ = noisy_latents.shape
        device = noisy_latents.device

        noisy_tokens = self.input_proj(noisy_latents)
        
        actions_embed = self.base_action_embed.view(1, 1, 1, -1).expand(batch_size, time_steps, 1, -1)
        if actions is not None:
            action_tokens = actions_embed + self.action_proj(actions).unsqueeze(2)
        else:
            action_tokens = actions_embed
        if actions_mask is not None:
            mask = actions_mask.to(dtype=torch.bool, device=device)
        return {} # Dummy return to allow import