from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn

from src.world_model.components import (
    TransformerBlock,
    RMSNorm,
    _rope_cache,
    SignalEmbedder,
)

@dataclass
class WorldModelConfig:
    latent_dim: int = 1024
    input_dim: int = 768
    action_dim: int = 8
    num_registers: int = 4
    depth: int = 24
    num_heads: int = 16
    mlp_multiplier: float = 4.0
    temporal_attention_interval: int = 4
    temporal_context_length: int = 10
    rope_base: float = 10000.0
    qk_norm_eps: float = 1e-6
    attn_logit_softcapping: Optional[float] = 50.0

@dataclass
class WorldModelOutput:
    latents: torch.Tensor
    kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]

# There is this thing that if independant frames are specified, then the rope values are still the one as if part of the sequence.
# Honestly not sure if it is something I want to fix or not
class WorldModelBackbone(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.input_dim, config.latent_dim, bias=False) if config.input_dim != config.latent_dim else nn.Identity()
        self.signal_embed = SignalEmbedder(config.latent_dim, base_freq_dim=256, scale=1000.0, max_period=10000)
        self.base_action_embed = nn.Parameter(torch.randn(config.latent_dim) * 0.02)
        self.action_proj = nn.Linear(config.action_dim, config.latent_dim)
        self.register_tokens = nn.Parameter(torch.randn(config.num_registers, config.latent_dim) * 0.02)

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=config.latent_dim,
                num_heads=config.num_heads,
                mlp_multiplier=config.mlp_multiplier,
                qk_norm_eps=config.qk_norm_eps,
                attn_logit_softcapping=config.attn_logit_softcapping,
            )
            for _ in range(config.depth)
        ])

        self.final_norm = RMSNorm(config.latent_dim)
        self.output_proj = nn.Linear(config.latent_dim, config.input_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.apply(_basic_init)

        for layer in self.layers:
            nn.init.constant_(layer.attn.out_proj.weight, 0)
            nn.init.constant_(layer.mlp.w3.weight, 0) 

        nn.init.normal_(self.base_action_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.register_tokens, mean=0.0, std=0.02)
        nn.init.constant_(self.output_proj.weight, 0)

    def _get_spatial_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=torch.bool) # full attention baby

    def _get_temporal_mask(
        self, 
        batch_size: int,
        seq_len: int,
        cache_len: int,
        independent_frames: Optional[torch.Tensor],
        device: torch.device
    ) -> torch.Tensor:
        total_len = cache_len + seq_len
        mask = torch.full((batch_size, 1, seq_len, total_len), float("-inf"), device=device)
        
        # Causal Mask
        if seq_len > 1:
            causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            mask[..., -seq_len:, -seq_len:].masked_fill_(causal, float("-inf"))
            mask[..., :cache_len] = 0.0
            mask[..., -seq_len:, -seq_len:].masked_fill_(~causal, 0.0)
        else:
            mask.fill_(0.0)

        # Temporal Context Window
        if self.config.temporal_context_length is not None:
            queries = torch.arange(cache_len, total_len, device=device).view(1, 1, seq_len, 1)
            keys = torch.arange(total_len, device=device).view(1, 1, 1, total_len)
            dist = queries - keys
            mask.masked_fill_(dist > self.config.temporal_context_length, float("-inf"))

        # Independent Frames masking
        if independent_frames is not None:
            is_indep = independent_frames.bool().unsqueeze(1).unsqueeze(3) # [B, 1, T, 1]
            q_idx = torch.arange(cache_len, total_len, device=device).unsqueeze(1) # [T, 1]
            k_idx = torch.arange(total_len, device=device).unsqueeze(0) # [1, Total]
            not_self = (q_idx != k_idx).unsqueeze(0).unsqueeze(0) # [1, 1, T, Total]
            mask = torch.where(is_indep & not_self, float("-inf"), mask)

        return mask

    def forward(
        self,
        noisy_latents: torch.Tensor, # [B, T, S, D]
        signal_levels: torch.Tensor, # [B, T]
        actions: Optional[torch.Tensor] = None, # [B, T, D]
        independent_frames: Optional[torch.Tensor] = None, # [B, T]
        use_actions: Optional[torch.Tensor] = None, # [B, T]
        kv_cache: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> WorldModelOutput:
        
        B, T, S, D_in = noisy_latents.shape
        device = noisy_latents.device
        D = self.config.latent_dim
        
        x = self.input_proj(noisy_latents)
        sig_emb = self.signal_embed(signal_levels.flatten()).view(B, T, 1, -1)
        reg_emb = self.register_tokens.view(1, 1, self.config.num_registers, -1).expand(B, T, -1, -1)
        act_emb = self.base_action_embed.view(1, 1, 1, -1).expand(B, T, 1, -1)
        
        if actions is not None:
            proj_act = self.action_proj(actions).unsqueeze(2)
            act_emb = act_emb + proj_act
            if use_actions is not None:
                mask = use_actions.bool().view(B, T, 1, 1)
                act_emb = torch.where(mask, act_emb, self.base_action_embed.view(1, 1, 1, -1))

        # [Signal, Action, Registers, Latents]
        x = torch.cat((sig_emb, act_emb, reg_emb, x), dim=2)
        S_total = x.shape[2]
        
        # Get cache length
        temp_cache_len = 0
        if kv_cache is not None:
            for i, c in enumerate(kv_cache):
                is_temporal = (i % self.config.temporal_attention_interval == 0) and (i != 0)
                if is_temporal and c is not None:
                    temp_cache_len = c[0].shape[2] 
                    break

        spatial_mask = self._get_spatial_mask(S_total, device)
        temporal_mask = self._get_temporal_mask(B, T, temp_cache_len, independent_frames, device)

        spatial_rope = _rope_cache(S_total, self.config.latent_dim // self.config.num_heads, self.config.rope_base, str(device))
        temporal_rope = _rope_cache(temp_cache_len + T, self.config.latent_dim // self.config.num_heads, self.config.rope_base, str(device))

        new_kv_cache = []

        for i, block in enumerate(self.layers):
            is_temporal = (i % self.config.temporal_attention_interval == 0) and (i != 0) and (i != self.config.depth - 1) # temporal should probably be neither first or last...
            layer_cache = kv_cache[i] if (kv_cache is not None and is_temporal) else None

            if is_temporal:
                x_in = x.transpose(1, 2).contiguous().view(B * S_total, T, D) # [B, T, S, D] -> [B, S, T, D] -> [B*S, T, D]
                t_mask = temporal_mask.unsqueeze(1).expand(-1, S_total, -1, -1, -1).reshape(B * S_total, 1, T, temp_cache_len + T)
                
                x_out, new_cache = block(x_in, temporal_rope, t_mask, layer_cache)
                x = x_out.view(B, S_total, T, D).transpose(1, 2) # [B*S, T, D] -> [B, S, T, D] -> [B, T, S, D]
                
            else:
                x_in = x.contiguous().view(B * T, S_total, D)
                x_out, new_cache = block(x_in, spatial_rope, spatial_mask, layer_cache)
                x = x_out.view(B, T, S_total, D)

            new_kv_cache.append(new_cache if is_temporal else None)

        latents = x[..., 2 + self.config.num_registers :, :]
        latents = self.final_norm(latents)
        output = self.output_proj(latents)

        return WorldModelOutput(latents=output, kv_cache=new_kv_cache)