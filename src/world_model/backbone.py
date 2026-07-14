from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.world_model.components import (
    TransformerBlock,
    RMSNorm,
    _rope_cache,
    SignalEmbedder,
    GaussianFourierSignalEmbedder,
    LearnedSignalEmbedder,
)
from src.world_model.heads import DHOutputHead

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
    bottleneck_dim: Optional[int] = None
    use_action_token: bool = True  # False during pretraining to exclude action token from sequence
    use_signal_token: bool = True
    num_signal_tokens: int = 1
    num_action_tokens: int = 1
    signal_embedding_type: str = "learned"  # {"fourier", "gaussian_fourier", "learned"}
    signal_embedding_num_bins: int = 1024
    predict_patch_difficulty: bool = False
    patch_difficulty_logvar_min: float = -10.0
    patch_difficulty_logvar_max: float = 10.0
    output_head: str = "linear"  # {"linear", "dh"}
    dh_hidden_dim: int = 2048
    dh_depth: int = 2
    dh_num_heads: int = 16
    dh_mlp_multiplier: float = 4.0
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        self.signal_embedding_type = self.signal_embedding_type.lower()
        if self.signal_embedding_type not in {"fourier", "gaussian_fourier", "learned"}:
            raise ValueError(f"Unknown signal_embedding_type: {self.signal_embedding_type}")
        self.output_head = self.output_head.lower()
        if self.output_head in {"ddt", "wide"}:
            self.output_head = "dh"
        if self.output_head not in {"linear", "dh"}:
            raise ValueError(f"Unknown output_head: {self.output_head}")
        if self.num_signal_tokens < 1:
            raise ValueError("num_signal_tokens must be at least 1.")
        if self.num_action_tokens < 1:
            raise ValueError("num_action_tokens must be at least 1.")
        if self.signal_embedding_num_bins < 2:
            raise ValueError("signal_embedding_num_bins must be >= 2.")
        if self.patch_difficulty_logvar_min >= self.patch_difficulty_logvar_max:
            raise ValueError("patch_difficulty_logvar_min must be smaller than patch_difficulty_logvar_max.")
        if self.output_head == "dh":
            if self.dh_depth < 1:
                raise ValueError("dh_depth must be at least 1.")
            if self.dh_hidden_dim % self.dh_num_heads != 0:
                raise ValueError("dh_hidden_dim must be divisible by dh_num_heads.")
            if (self.dh_hidden_dim // self.dh_num_heads) % 4 != 0:
                raise ValueError("DH attention head dimension must be divisible by 4.")

@dataclass
class WorldModelOutput:
    latents: Optional[torch.Tensor]
    kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
    patch_logvar: Optional[torch.Tensor] = None

# There is this thing that if independant frames are specified, then the rope values are still the one as if part of the sequence.
# Honestly not sure if it is something I want to fix or not
class WorldModelBackbone(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config

        if config.bottleneck_dim is not None:
            self.input_proj = nn.Sequential(
                nn.Linear(config.input_dim, config.bottleneck_dim, bias=False),
                nn.Linear(config.bottleneck_dim, config.latent_dim, bias=False))
        else:
            self.input_proj = nn.Linear(config.input_dim, config.latent_dim, bias=False) if config.input_dim != config.latent_dim else nn.Identity()
        if config.use_signal_token:
            if config.signal_embedding_type == "learned":
                self.signal_embed = LearnedSignalEmbedder(config.latent_dim, num_bins=config.signal_embedding_num_bins)
            elif config.signal_embedding_type == "gaussian_fourier":
                self.signal_embed = GaussianFourierSignalEmbedder(config.latent_dim)
            else:
                self.signal_embed = SignalEmbedder(config.latent_dim, base_freq_dim=256, scale=1000.0, max_period=10000)
            if config.num_signal_tokens > 1:
                self.signal_token_offsets = nn.Parameter(
                    torch.empty(config.num_signal_tokens, config.latent_dim)
                )
        # Only create action parameters when use_action_token is True (for finetuning with actions)
        if config.use_action_token:
            action_token_shape = (
                (config.latent_dim,)
                if config.num_action_tokens == 1
                else (config.num_action_tokens, config.latent_dim)
            )
            self.base_action_embed = nn.Parameter(torch.empty(action_token_shape))
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
        if config.output_head == "linear":
            self.output_proj = nn.Linear(config.latent_dim, config.input_dim)
        else:
            self.dh_head = DHOutputHead(
                input_dim=config.input_dim,
                backbone_dim=config.latent_dim,
                hidden_dim=config.dh_hidden_dim,
                depth=config.dh_depth,
                num_heads=config.dh_num_heads,
                mlp_multiplier=config.dh_mlp_multiplier,
                qk_norm_eps=config.qk_norm_eps,
                rope_base=config.rope_base,
                gradient_checkpointing=config.gradient_checkpointing,
            )
        if config.predict_patch_difficulty:
            self.patch_logvar_proj = nn.Linear(config.latent_dim, 1)
        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        self.apply(_init_weights)

        res_scale = 1.0 / (2.0 * self.config.depth) ** 0.5
        for block in self.layers:
            if hasattr(block.attn, 'out_proj'):
                 block.attn.out_proj.weight.data.mul_(res_scale)
            if hasattr(block.mlp, 'w3'):
                 block.mlp.w3.weight.data.mul_(res_scale)

        if self.config.use_action_token:
            action_std = (
                0.02
                if self.config.num_action_tokens == 1
                else self.config.latent_dim ** -0.5
            )
            nn.init.normal_(self.base_action_embed, mean=0.0, std=action_std)
        if self.config.use_signal_token and self.config.num_signal_tokens > 1:
            nn.init.normal_(
                self.signal_token_offsets,
                mean=0.0,
                std=self.config.latent_dim ** -0.5,
            )
        nn.init.normal_(self.register_tokens, mean=0.0, std=0.02)
        if self.config.output_head == "linear":
            torch.nn.init.zeros_(self.output_proj.weight)
        else:
            # self.apply above also visits the head, so restore the RAEv2 zero
            # initialization for all AdaLN gates and the final prediction layer.
            self.dh_head.initialize_weights()
        if self.config.predict_patch_difficulty:
            torch.nn.init.zeros_(self.patch_logvar_proj.weight)
            torch.nn.init.zeros_(self.patch_logvar_proj.bias)

    def _get_spatial_mask(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        return None

    def _run_transformer_block(
        self,
        block: TransformerBlock,
        x: torch.Tensor,
        rope: Tuple[torch.Tensor, torch.Tensor],
        mask: Optional[torch.Tensor],
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run one backbone block, optionally recomputing it in backward.

        Checkpointing is deliberately restricted to training with gradients.
        Evaluation and cache-based rollout therefore retain their normal path.
        """
        if not (
            self.config.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return block(x, rope, mask, kv_cache)

        # Flatten the nested cache tuple for broad torch.compile/checkpoint
        # compatibility. In normal training kv_cache is None; it is captured
        # here so the same helper also remains correct for future use cases.
        def block_forward(hidden: torch.Tensor):
            output, (key, value) = block(hidden, rope, mask, kv_cache)
            return output, key, value

        output, key, value = checkpoint(
            block_forward,
            x,
            use_reentrant=False,
        )
        return output, (key, value)

    def _embed_signal(
        self,
        signal_levels: torch.Tensor,
    ) -> torch.Tensor:
        """Return one base timestep embedding per frame, shaped [B, T, D]."""
        batch, frames = signal_levels.shape
        if self.config.signal_embedding_type == "fourier":
            embedding = self.signal_embed(signal_levels.flatten())
            return embedding.squeeze(dim=1).view(batch, frames, -1)
        return self.signal_embed(signal_levels)

    def _signal_tokens(
        self,
        timestep_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.num_signal_tokens == 1:
            return timestep_embedding.unsqueeze(dim=2)
        return (
            timestep_embedding.unsqueeze(dim=2)
            + self.signal_token_offsets.view(
                1,
                1,
                self.config.num_signal_tokens,
                -1,
            )
        )

    def _base_action_tokens(
        self,
        batch: int,
        frames: int,
    ) -> torch.Tensor:
        if self.config.num_action_tokens == 1:
            base = self.base_action_embed.view(1, 1, 1, -1)
        else:
            base = self.base_action_embed.view(
                1,
                1,
                self.config.num_action_tokens,
                -1,
            )
        return base.expand(batch, frames, -1, -1)

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
        signal_levels: torch.Tensor, # [B, T], original frame-level signal
        actions: Optional[torch.Tensor] = None, # [B, T, D]
        independent_frames: Optional[torch.Tensor] = None, # [B, T]
        use_actions: Optional[torch.Tensor] = None, # [B, T]
        kv_cache: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        return_prediction: bool = True,
    ) -> WorldModelOutput:
        
        B, T, S, _ = noisy_latents.shape
        device = noisy_latents.device
        D = self.config.latent_dim
        raw_noisy_latents = noisy_latents

        if signal_levels.shape != (B, T):
            raise ValueError(
                f"signal_levels must contain one frame-level value with shape {(B, T)}, "
                f"got {tuple(signal_levels.shape)}. Per-patch signal conditioning is not supported."
            )

        x = self.input_proj(noisy_latents)
        reg_emb = self.register_tokens.view(1, 1, self.config.num_registers, -1).expand(B, T, -1, -1)
        prefix_tokens = []
        timestep_embedding = torch.zeros(
            B,
            T,
            D,
            device=device,
            dtype=x.dtype,
        )
        if self.config.use_signal_token:
            timestep_embedding = self._embed_signal(
                signal_levels
            ).to(dtype=x.dtype)
            prefix_tokens.append(self._signal_tokens(timestep_embedding))
        
        # this if is basically to be able to pretrain without action data, just on raw video, making finetuning easier
        if self.config.use_action_token:
            # [Signal, Action, Registers, Latents]
            base_act_emb = self._base_action_tokens(B, T)
            act_emb = base_act_emb
            
            if actions is not None:
                proj_act = self.action_proj(actions).unsqueeze(2)
                act_emb = act_emb + proj_act
                if use_actions is not None:
                    mask = use_actions.bool().view(B, T, 1, 1)
                    act_emb = torch.where(mask, act_emb, base_act_emb)

            prefix_tokens.append(act_emb)
            prefix_tokens.append(reg_emb)
            x = torch.cat((*prefix_tokens, x), dim=2)
        else:
            # [optional signal, registers, latents] - no action token during pretraining
            prefix_tokens.append(reg_emb)
            x = torch.cat((*prefix_tokens, x), dim=2)
        num_prefix_tokens = sum(token.shape[2] for token in prefix_tokens)
        
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
            is_temporal = (i % self.config.temporal_attention_interval == 0) and (i != 0) and (i != self.config.depth - 1) # temporal should be neither first or last...
            layer_cache = kv_cache[i] if (kv_cache is not None and is_temporal) else None

            if is_temporal:
                x_in = x.transpose(1, 2).contiguous().view(B * S_total, T, D) # [B, T, S, D] -> [B, S, T, D] -> [B*S, T, D]
                t_mask = temporal_mask.unsqueeze(1).expand(-1, S_total, -1, -1, -1).reshape(B * S_total, 1, T, temp_cache_len + T)
                
                x_out, new_cache = self._run_transformer_block(
                    block,
                    x_in,
                    temporal_rope,
                    t_mask,
                    layer_cache,
                )
                x = x_out.view(B, S_total, T, D).transpose(1, 2) # [B*S, T, D] -> [B, S, T, D] -> [B, T, S, D]
                
            else:
                x_in = x.contiguous().view(B * T, S_total, D)
                x_out, new_cache = self._run_transformer_block(
                    block,
                    x_in,
                    spatial_rope,
                    spatial_mask,
                    layer_cache,
                )
                x = x_out.view(B, T, S_total, D)

            new_kv_cache.append(new_cache if is_temporal else None)

        latents = x[..., num_prefix_tokens:, :]
        latents = self.final_norm(latents)
        if not return_prediction:
            return WorldModelOutput(
                latents=None,
                kv_cache=new_kv_cache,
                patch_logvar=None,
            )

        patch_logvar = None
        if self.config.predict_patch_difficulty:
            patch_logvar = self.patch_logvar_proj(latents).squeeze(-1)
            patch_logvar = patch_logvar.clamp(
                min=self.config.patch_difficulty_logvar_min,
                max=self.config.patch_difficulty_logvar_max,
            )
        if self.config.output_head == "linear":
            output = self.output_proj(latents)
        else:
            output = self.dh_head(
                raw_noisy_latents,
                latents,
                timestep_embedding,
            )

        return WorldModelOutput(latents=output, kv_cache=new_kv_cache, patch_logvar=patch_logvar)
