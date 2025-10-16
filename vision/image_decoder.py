"""
ViT-style decoder that reconstructs RGB frames from frozen DINO patch tokens.

The architecture mirrors the stage-1 decoder used in the DiT-RAE pipeline while
remaining intentionally compact. It supports rectangular image resolutions by
tracking the height/width patch grid explicitly instead of assuming square
inputs.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2tuple(value: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(value, Sequence):
        if len(value) != 2:
            raise ValueError(f"Expected a length-2 sequence, received {value!r}.")
        return int(value[0]), int(value[1])
    return int(value), int(value)


@dataclass
class ImageDecoderConfig:
    """Hyperparameters for the ViT decoder."""

    image_size: Tuple[int, int] = (180, 320)  # (height, width)
    patch_size: Tuple[int, int] | int = 16
    latent_grid: Optional[Tuple[int, int]] = None
    resize_mode: Optional[str] = "bilinear"
    latent_dim: int = 768
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-6
    channels: int = 3

    def validate(self) -> None:
        if self.depth <= 0:
            raise ValueError("Decoder depth must be positive.")
        if self.num_heads <= 0:
            raise ValueError("Decoder num_heads must be positive.")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})."
            )
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")
        h, w = self.image_size
        patch_h, patch_w = _to_2tuple(self.patch_size)
        if self.latent_grid is not None:
            gh, gw = self.latent_grid
            if gh <= 0 or gw <= 0:
                raise ValueError("latent_grid entries must be positive integers.")
        elif h % patch_h != 0 or w % patch_w != 0:
            raise ValueError(
                "image_size must be divisible by patch_size along each dimension "
                f"(received image_size={self.image_size}, patch_size={self.patch_size})."
            )
        if self.resize_mode is not None and self.resize_mode not in {"nearest", "bilinear", "bicubic"}:
            raise ValueError("resize_mode must be one of {None, 'nearest', 'bilinear', 'bicubic'}.")


class _FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        qkv_bias: bool,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
            bias=qkv_bias,
        )
        self.ff_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.ff = _FeedForward(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_out
        ff_input = self.ff_norm(x)
        x = x + self.ff(ff_input)
        return x


class VisionDecoder(nn.Module):
    """
    ViT decoder that maps DINO tokens back to RGB pixel space.

    Args:
        config: :class:`ImageDecoderConfig` describing architecture and output resolution.
    """

    def __init__(self, config: ImageDecoderConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        patch_h, patch_w = _to_2tuple(config.patch_size)
        image_h, image_w = config.image_size
        if config.latent_grid is not None:
            self.grid_h, self.grid_w = config.latent_grid
        else:
            self.grid_h = image_h // patch_h
            self.grid_w = image_w // patch_w
        self.num_patches = self.grid_h * self.grid_w
        self.patch_area = patch_h * patch_w * config.channels
        self._decoded_size = (self.grid_h * patch_h, self.grid_w * patch_w)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.embed_dim))
        self.token_dropout = nn.Dropout(config.dropout)

        proj_in = []
        if config.latent_dim != config.embed_dim:
            proj_in.append(nn.Linear(config.latent_dim, config.embed_dim))
        else:
            proj_in.append(nn.Identity())
        proj_in.append(nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps))
        self.input_projection = nn.Sequential(*proj_in)

        blocks = [
            _DecoderBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                qkv_bias=config.qkv_bias,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.depth)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.output_projection = nn.Linear(config.embed_dim, self.patch_area)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode a batch of latent tokens into RGB images.

        Args:
            tokens: Tensor of shape ``[B, num_tokens, latent_dim]`` containing
                spatial tokens produced by a frozen encoder (e.g. DINO).

        Returns:
            Reconstructed images with shape ``[B, channels, H, W]`` in ``[0, 1]`` range.
        """
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [B, N, C], got {tuple(tokens.shape)}.")
        batch, seq_len, _ = tokens.shape
        if seq_len != self.num_patches:
            raise ValueError(
                f"Decoder expects {self.num_patches} tokens, received {seq_len}. "
                "Ensure the encoder resolution matches decoder.image_size."
            )

        x = self.input_projection(tokens)
        x = x + self.pos_embed
        x = self.token_dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        patch_tokens = self.output_projection(x)
        recon = self._unpatchify(patch_tokens)
        target_h, target_w = self.config.image_size
        if recon.shape[-2:] != (target_h, target_w):
            if self.config.resize_mode is None:
                raise ValueError(
                    "Decoder output grid does not match requested image_size and resize_mode=None. "
                    "Specify image_size divisible by patch_size or enable interpolation."
                )
            align_corners = False if self.config.resize_mode in {"bilinear", "bicubic"} else None
            recon = F.interpolate(
                recon,
                size=(target_h, target_w),
                mode=self.config.resize_mode,
                align_corners=align_corners,
            )
        return recon

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch = patches.shape[0]
        patch_h, patch_w = _to_2tuple(self.config.patch_size)
        x = patches.view(batch, self.grid_h, self.grid_w, patch_h, patch_w, self.config.channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(batch, self.config.channels, self.grid_h * patch_h, self.grid_w * patch_w)
        return x
