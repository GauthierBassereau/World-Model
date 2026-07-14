"""Token-compatible RAEv2 autoencoder for the world model."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rae_dino.encoder.dinov3 import DINOv3MultiLayerEncoder
from src.rae_dino.profiles import (
    RAEV2_MODEL_REPO,
    RAEV2_MODEL_REVISION,
    RAEv2Profile,
    get_raev2_profile,
    resolve_raev2_assets,
)


def _load_tensor_mapping(path: Path, *, mmap: bool = False) -> Mapping[str, torch.Tensor]:
    kwargs = {"map_location": "cpu", "weights_only": True}
    if mmap:
        kwargs["mmap"] = True
    try:
        value = torch.load(path, **kwargs)
    except TypeError:
        # Compatibility with older cluster PyTorch releases that predate one
        # of the safe-loading keyword arguments.
        kwargs.pop("mmap", None)
        try:
            value = torch.load(path, **kwargs)
        except TypeError:
            value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a tensor mapping in {path}, found {type(value).__name__}.")
    return value


def _load_decoder(
    profile: RAEv2Profile,
    checkpoint_path: Path,
) -> nn.Module:
    # Keep the heavyweight transformers decoder import out of config parsing
    # and lightweight unit tests.
    from src.rae_dino.decoder.mae_decoder import GeneralDecoder
    from src.rae_dino.decoder.utils import ViTMAEConfig

    config_path = Path(__file__).resolve().parent / "decoder" / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config_dict = json.load(handle)
    if not isinstance(config_dict.get("patch_size"), (int, list, tuple)):
        config_dict["patch_size"] = profile.decoder_patch_size

    config = ViTMAEConfig(**config_dict)
    config.hidden_size = profile.latent_dim
    config.patch_size = profile.decoder_patch_size
    config.image_size = profile.resolution
    decoder = GeneralDecoder(config, num_patches=profile.tokens_per_frame)

    state_dict = _load_tensor_mapping(checkpoint_path, mmap=True)
    incompatible = decoder.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "RAEv2 decoder checkpoint did not exactly match the bundled ViTXL decoder: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}."
        )
    return decoder


def _validated_stats(
    stats: Mapping[str, torch.Tensor],
    profile: RAEv2Profile,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        mean = stats["mean"]
        variance = stats["var"]
    except KeyError as exc:
        raise KeyError("RAEv2 normalization stats must contain both 'mean' and 'var'.") from exc

    expected = (
        profile.latent_dim,
        profile.resolution // profile.patch_size,
        profile.resolution // profile.patch_size,
    )
    if tuple(mean.shape) != expected or tuple(variance.shape) != expected:
        raise ValueError(
            f"RAEv2 stats for {profile.name!r} must have shape {expected}; "
            f"found mean={tuple(mean.shape)}, var={tuple(variance.shape)}."
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
        raise ValueError("RAEv2 normalization stats contain non-finite values.")
    if not torch.all(variance >= 0):
        raise ValueError("RAEv2 normalization variance contains negative values.")
    return mean.float().unsqueeze(0), variance.float().unsqueeze(0)


class RAEv2Autoencoder(nn.Module):
    """RAEv2 internals with the existing world-model token API.

    ``encode`` accepts RGB tensors in either [0, 1] or [0, 255] and returns
    normalized patch tokens of shape ``[B, N, C]``.  ``decode`` accepts those
    same normalized tokens and returns raw RAEv2 RGB predictions at the fixed
    profile resolution.  Unlike the legacy RAE decoder, RAEv2 decoder outputs
    are already in RGB space and must not be de-normalized with ImageNet stats.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_mean: torch.Tensor,
        latent_var: torch.Tensor,
        profile: RAEv2Profile,
        noise_tau: float = 0.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if noise_tau < 0:
            raise ValueError("noise_tau must be non-negative.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.encoder = encoder
        self.decoder = decoder
        self.profile = profile
        self.resolution = profile.resolution
        self.encoder_input_size = profile.resolution
        self.encoder_patch_size = profile.patch_size
        self.latent_dim = profile.latent_dim
        self.base_patches = profile.tokens_per_frame
        self.noise_tau = float(noise_tau)
        self.eps = float(eps)
        self.do_normalization = True

        expected = (
            1,
            self.latent_dim,
            self.resolution // self.encoder_patch_size,
            self.resolution // self.encoder_patch_size,
        )
        if tuple(latent_mean.shape) != expected or tuple(latent_var.shape) != expected:
            raise ValueError(
                f"latent stats must both have shape {expected}; found "
                f"mean={tuple(latent_mean.shape)}, var={tuple(latent_var.shape)}."
            )
        self.register_buffer("latent_mean", latent_mean.float().contiguous())
        self.register_buffer("latent_var", latent_var.float().contiguous())

    @classmethod
    def from_pretrained(
        cls,
        profile_name: str = "dinov3b-k11",
        *,
        repo_id: str = RAEV2_MODEL_REPO,
        revision: str = RAEV2_MODEL_REVISION,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        dinov3_repo_dir: Optional[str] = None,
        noise_tau: float = 0.0,
        eps: float = 1e-5,
    ) -> "RAEv2Autoencoder":
        profile = get_raev2_profile(profile_name)
        assets = resolve_raev2_assets(
            profile,
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        encoder = DINOv3MultiLayerEncoder.from_checkpoint(
            profile.encoder_model_name,
            assets.encoder_checkpoint,
            layer_indices=profile.encoder_layers,
            resolution=profile.resolution,
            patch_size=profile.patch_size,
            latent_dim=profile.latent_dim,
            repo_dir=dinov3_repo_dir,
            local_files_only=local_files_only,
        )
        decoder = _load_decoder(profile, assets.decoder_checkpoint)
        stats = _load_tensor_mapping(assets.normalization_stats)
        mean, variance = _validated_stats(stats, profile)
        return cls(
            encoder=encoder,
            decoder=decoder,
            latent_mean=mean,
            latent_var=variance,
            profile=profile,
            noise_tau=noise_tau,
            eps=eps,
        )

    def _tokens_to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected latent tokens [B, N, C], found shape {tuple(tokens.shape)}.")
        batch, count, channels = tokens.shape
        if count != self.base_patches or channels != self.latent_dim:
            raise ValueError(
                f"Expected latent shape [B, {self.base_patches}, {self.latent_dim}], "
                f"found {tuple(tokens.shape)}."
            )
        side = math.isqrt(count)
        if side * side != count:
            raise ValueError(f"Latent token count {count} is not a square grid.")
        return tokens.transpose(1, 2).reshape(batch, channels, side, side)

    @staticmethod
    def _grid_to_tokens(grid: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = grid.shape
        return grid.reshape(batch, channels, height * width).transpose(1, 2).contiguous()

    def noising(self, tokens: torch.Tensor) -> torch.Tensor:
        sigma = self.noise_tau * torch.rand(
            (tokens.shape[0],) + (1,) * (tokens.ndim - 1),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens + sigma * torch.randn_like(tokens)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected RGB images [B, 3, H, W], found {tuple(images.shape)}.")
        if not torch.is_floating_point(images):
            images = images.float()
        if images.numel() and images.detach().amax().item() <= 1.0:
            images = images * 255.0
        if images.shape[-2:] != (self.resolution, self.resolution):
            images = F.interpolate(
                images,
                size=(self.resolution, self.resolution),
                mode="bicubic",
                align_corners=False,
            )

        tokens = self.encoder(images)
        if self.training and self.noise_tau > 0:
            tokens = self.noising(tokens)
        grid = self._tokens_to_grid(tokens)
        mean = self.latent_mean.to(dtype=grid.dtype)
        variance = self.latent_var.to(dtype=grid.dtype)
        normalized = (grid - mean) / torch.sqrt(variance + self.eps)
        return self._grid_to_tokens(normalized)

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        grid = self._tokens_to_grid(tokens)
        mean = self.latent_mean.to(dtype=grid.dtype)
        variance = self.latent_var.to(dtype=grid.dtype)
        grid = grid * torch.sqrt(variance + self.eps) + mean
        decoder_tokens = self._grid_to_tokens(grid)
        output = self.decoder(decoder_tokens, drop_cls_token=False).logits
        # RAEv2 decoders were trained directly against [0, 1] RGB images.
        return self.decoder.unpatchify(output)

    def forward(self, images: torch.Tensor, return_latent: bool = False):
        latent = self.encode(images)
        reconstruction = self.decode(latent)
        if return_latent:
            return reconstruction, latent
        return reconstruction
