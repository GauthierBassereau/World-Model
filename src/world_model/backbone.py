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
                use_temporal=(i % config.temporal_attention_interval == 0)
            )
            for i in range(config.depth)
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

        scaled_std = 0.02 / torch.sqrt(torch.tensor(2 * self.config.depth, dtype=torch.float32))

        for layer in self.layers:
            torch.nn.init.trunc_normal_(layer.spatial_attn.out_proj.weight, std=scaled_std)
            if layer.use_temporal:
                torch.nn.init.trunc_normal_(layer.temporal_attn.out_proj.weight, std=scaled_std)
            torch.nn.init.trunc_normal_(layer.mlp.w3.weight, std=scaled_std)

        nn.init.constant_(self.output_proj.weight, 0)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0)
        
        nn.init.normal_(self.base_action_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.register_tokens, mean=0.0, std=0.02)

    def _get_masks(
        self, 
        batch_size: int, 
        seq_len: int, 
        latent_tokens: int,
        cache_len: int,
        independent_frames: Optional[torch.Tensor], 
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Spatial
        tokens_per_frame = latent_tokens + 2 + self.config.num_registers
        spatial_mask = torch.zeros((1, 1, tokens_per_frame, tokens_per_frame), device=device, dtype=torch.bool)
        
        # Temporal
        total_len = cache_len + seq_len
        temporal_mask = torch.full((batch_size, 1, seq_len, total_len), float("-inf"), device=device)
        
        if seq_len > 1:
            causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            temporal_mask[..., -seq_len:, -seq_len:].masked_fill_(causal, float("-inf"))
            temporal_mask[..., :cache_len] = 0.0
            temporal_mask[..., -seq_len:, -seq_len:].masked_fill_(~causal, 0.0)
        else:
            temporal_mask.fill_(0.0)

        if self.config.temporal_context_length is not None:
            queries = torch.arange(cache_len, total_len, device=device).view(1, 1, seq_len, 1)
            keys = torch.arange(total_len, device=device).view(1, 1, 1, total_len)
            dist = queries - keys
            temporal_mask.masked_fill_(dist > self.config.temporal_context_length, float("-inf"))

        if independent_frames is not None:
            is_indep = independent_frames.bool().unsqueeze(1).unsqueeze(3) # [B, 1, S, 1]
            not_self = torch.arange(total_len, device=device).unsqueeze(0) != torch.arange(cache_len, total_len, device=device).unsqueeze(1)
            not_self = not_self.unsqueeze(0).unsqueeze(0) # [1, 1, S, Total]
            temporal_mask = torch.where(is_indep & not_self, torch.tensor(float("-inf"), device=device), temporal_mask)

        return spatial_mask, temporal_mask

    def forward(
        self,
        noisy_latents: torch.Tensor, # [B, S, T, D]
        signal_levels: torch.Tensor, # [B, S]
        actions: Optional[torch.Tensor] = None, # [B, S, D]
        independent_frames: Optional[torch.Tensor] = None, # [B, S]
        use_actions: Optional[torch.Tensor] = None, # [B, S]
        kv_cache: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> WorldModelOutput:
        
        B, S, T, D = noisy_latents.shape
        device = noisy_latents.device
        
        x = self.input_proj(noisy_latents)
        sig_emb = self.signal_embed(signal_levels.flatten()).view(B, S, 1, -1)
        reg_emb = self.register_tokens.view(1, 1, self.config.num_registers, -1).expand(B, S, -1, -1)
        act_emb = self.base_action_embed.view(1, 1, 1, -1).expand(B, S, 1, -1)
        if actions is not None:
            proj_act = self.action_proj(actions).unsqueeze(2)
            act_emb = act_emb + proj_act
            if use_actions is not None:
                act_emb = torch.where(use_actions.bool().view(B, S, 1, 1), act_emb, self.base_action_embed.view(1, 1, 1, -1))

        # [Signal, Action, Registers, Latents]
        x = torch.cat((sig_emb, act_emb, reg_emb, x), dim=2)
        tokens_per_frame = x.shape[2]

        cache_len = 0
        if kv_cache is not None:
            for c in kv_cache:
                if c is not None: # Check the first non-None cache to find temporal length, some layers might be None depending on temporal_attention_interval...
                    cache_len = c[0].shape[2]
                    break
        
        spatial_mask, temporal_mask = self._get_masks(B, S, T, cache_len, independent_frames, device)

        spatial_rope = _rope_cache(tokens_per_frame, self.config.latent_dim // self.config.num_heads, self.config.rope_base, str(device))
        temporal_rope = _rope_cache(cache_len + S, self.config.latent_dim // self.config.num_heads, self.config.rope_base, str(device))

        new_kv_cache = []
        for i, block in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            
            x, new_cache = block(
                x, 
                spatial_rope, 
                temporal_rope, 
                spatial_mask, 
                temporal_mask, 
                layer_cache
            )
            new_kv_cache.append(new_cache)

        latents = x[..., 2 + self.config.num_registers :, :]
        latents = self.final_norm(latents)
        output = self.output_proj(latents)

        return WorldModelOutput(latents=output, kv_cache=new_kv_cache)