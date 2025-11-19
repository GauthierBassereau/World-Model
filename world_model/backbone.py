from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from world_model.components import (
    TransformerBlock,
    RMSNorm,
    _rope_cache,
)

@dataclass
class WorldModelConfig:
    latent_dim: int = 1024
    input_dim: int = 768
    action_dim: int = 6
    num_registers: int = 4
    depth: int = 24
    num_heads: int = 16
    mlp_multiplier: float = 4.0
    temporal_attention_interval: int = 4
    temporal_context_length: int = 30
    rope_base: float = 1000.0
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

        # For now it is a simple SiLU MLP, like used in other diffusion models, but need to check I think there are more options, see in RAE and Lightning DiT they talk about GaussianFourierEmbedding
        self.noise_embed = nn.Sequential(
            nn.Linear(1, self.config.latent_dim),
            nn.SiLU(),
            nn.Linear(self.config.latent_dim, self.config.latent_dim),
        )

        # The default base action token is a learned embedding, and each action is simply added to it after projection
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
                    frozen_prefix_tokens=self.config.num_registers + 2,
                )
            )
        self.layers = nn.ModuleList(blocks)

        self.final_norm = RMSNorm(self.config.latent_dim)
        self.output_proj = nn.Linear(self.config.latent_dim, self.config.input_dim)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        nn.init.normal_(self.base_action_embed, std=0.02)
        nn.init.normal_(self.register_tokens, std=0.02)
        
        nn.init.zeros_(self.noise_embed[-1].weight)
        nn.init.zeros_(self.noise_embed[-1].bias)
        
        with torch.no_grad():
            self.action_proj.weight.mul_(0.1) # making it small
        nn.init.zeros_(self.action_proj.bias)
        
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
        independent_frames_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.tril(
            torch.ones(time_steps, time_steps, dtype=torch.bool, device=device),
            diagonal=-1, # no same frame attention
        )
        if context_length is not None and context_length < time_steps:
            context_band = torch.triu(mask, diagonal=-(context_length))
            mask = mask & context_band
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, time_steps, time_steps).clone() # need to clone so each batch has its own storage since some are different.
        
        if independent_frames_mask is not None:
            independent = independent_frames_mask.to(device=device, dtype=torch.bool).view(batch_size, 1, 1, 1)
            all_false = torch.zeros_like(mask, dtype=torch.bool)
            mask = torch.where(independent, all_false, mask)

        return mask # [B, 1, T, T] # need for expanding batch size since not the same temporal mask for all batch (independant frame)

    def forward(
        self,
        noisy_latents: torch.Tensor, # [Batch, Sequence, Tokens, Dimension]
        noise_levels: torch.Tensor, # [B, S, 1]
        actions: Optional[torch.Tensor] = None, # [B, S, D]
        # ------------------ specific arguments for training purposes
        independent_frames_mask: Optional[torch.Tensor] = None, # [B, 1]
        actions_mask: Optional[torch.Tensor] = None, # [B, 1]
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps, latent_tokens, _ = noisy_latents.shape
        device = noisy_latents.device

        # ------------------ preprocess the inputs and create tokens sequences
        noisy_tokens = self.input_proj(noisy_latents)
        
        actions_embed = self.base_action_embed.view(1, 1, 1, -1).expand(batch_size, time_steps, 1, -1)
        if actions is not None:
            action_tokens = actions_embed + self.action_proj(actions).unsqueeze(2)
        else:
            action_tokens = actions_embed
        if actions_mask is not None:
            mask = actions_mask.to(dtype=torch.bool, device=device).unsqueeze(-1).unsqueeze(-1)
            action_tokens = torch.where(mask, action_tokens, actions_embed)
        
        noise_tokens = self.noise_embed(noise_levels.unsqueeze(-1)).unsqueeze(2)
        
        register_tokens = self.register_tokens.unsqueeze(0).unsqueeze(0).expand(batch_size, time_steps, -1, -1)

        x = torch.cat((register_tokens, noise_tokens, action_tokens, noisy_tokens), dim=2)
        tokens_per_frame = x.shape[2]

        # ------------------ create attention masks and positional encodings
        spatial_mask = self._build_spatial_mask(
            latent_tokens, 
            self.config.num_registers, 
            device
        )
        spatial_rope = _rope_cache(
            tokens_per_frame,
            self.config.latent_dim // self.config.num_heads,
            self.config.rope_base,
            str(device),
        )
        temporal_mask = self._build_temporal_mask(
            time_steps, 
            self.config.temporal_context_length, 
            independent_frames_mask, 
            batch_size, 
            device
        )
        temporal_rope = _rope_cache(
            time_steps,
            self.config.latent_dim // self.config.num_heads,
            self.config.rope_base,
            str(device),
        )

        # ------------------ finally layers
        for block in self.layers:
            x = block(
                x,
                spatial_rope,
                temporal_rope,
                spatial_mask=spatial_mask,
                temporal_mask=temporal_mask,
            )

        latents = self.final_norm(x[..., self.config.num_registers + 2 :, :])
        pred_clean_latents = self.output_proj(latents)

        return {"pred_clean_latents": pred_clean_latents}


if __name__ == "__main__":
    config = WorldModelConfig()
    model = WorldModelBackbone(config)
    total_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Trainable parameters with default config: {total_parameters:,}")
