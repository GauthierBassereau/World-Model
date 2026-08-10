"""Configuration and construction for visual autoencoders."""

from dataclasses import dataclass, field
from typing import Optional

import torch.nn as nn

from src.rae_dino.profiles import (
    RAEV2_MODEL_REPO,
    RAEV2_MODEL_REVISION,
    get_raev2_profile,
)


@dataclass
class RAEv2AutoencoderConfig:
    profile: str = "dinov3b-k11"
    repo_id: str = RAEV2_MODEL_REPO
    revision: str = RAEV2_MODEL_REVISION
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    dinov3_repo_dir: Optional[str] = None
    noise_tau: float = 0.0
    eps: float = 1e-5


@dataclass
class LegacyRAEAutoencoderConfig:
    dinov2_path: str = "facebook/dinov2-with-registers-base"
    encoder_input_size: int = 224
    decoder_config_path: str = "src/rae_dino/decoder/config.json"
    decoder_patch_size: int = 16
    pretrained_decoder_path: Optional[str] = (
        "src/rae_dino/decoder/decoder_weights/ViTXL_n08.pt"
    )
    noise_tau: float = 0.0
    normalization_stat_path: Optional[str] = "src/rae_dino/encoder/stat.pt"
    eps: float = 1e-5


@dataclass
class AutoencoderConfig:
    """Nested autoencoder selection suitable for Pyrallis YAML configs."""

    type: str = "raev2"
    raev2: RAEv2AutoencoderConfig = field(default_factory=RAEv2AutoencoderConfig)
    legacy: LegacyRAEAutoencoderConfig = field(default_factory=LegacyRAEAutoencoderConfig)


def build_autoencoder(config: AutoencoderConfig) -> nn.Module:
    autoencoder_type = config.type.strip().lower().replace("-", "_")
    if autoencoder_type in {"raev2", "rae_v2"}:
        from src.rae_dino.raev2 import RAEv2Autoencoder

        return RAEv2Autoencoder.from_pretrained(
            profile_name=config.raev2.profile,
            repo_id=config.raev2.repo_id,
            revision=config.raev2.revision,
            cache_dir=config.raev2.cache_dir,
            local_files_only=config.raev2.local_files_only,
            dinov3_repo_dir=config.raev2.dinov3_repo_dir,
            noise_tau=config.raev2.noise_tau,
            eps=config.raev2.eps,
        )

    if autoencoder_type in {"legacy", "rae", "legacy_rae", "dinov2"}:
        # Import lazily: selecting RAEv2 should not instantiate or download the
        # legacy Transformers DINOv2 model.
        from src.rae_dino.rae import RAE

        return RAE(
            dinov2_path=config.legacy.dinov2_path,
            encoder_input_size=config.legacy.encoder_input_size,
            decoder_config_path=config.legacy.decoder_config_path,
            decoder_patch_size=config.legacy.decoder_patch_size,
            pretrained_decoder_path=config.legacy.pretrained_decoder_path,
            noise_tau=config.legacy.noise_tau,
            normalization_stat_path=config.legacy.normalization_stat_path,
            eps=config.legacy.eps,
        )

    raise ValueError(
        f"Unknown autoencoder type {config.type!r}. Expected 'raev2' or 'legacy'."
    )


def validate_autoencoder_input_dim(
    autoencoder: nn.Module,
    world_model_input_dim: int,
) -> None:
    latent_dim = getattr(autoencoder, "latent_dim", None)
    if latent_dim is None:
        raise ValueError("The configured autoencoder does not expose latent_dim.")
    if int(latent_dim) != int(world_model_input_dim):
        raise ValueError(
            f"Autoencoder latent_dim={latent_dim} does not match "
            f"world_model.input_dim={world_model_input_dim}. Update both when "
            "switching RAEv2 profiles."
        )


def configured_autoencoder_resolution(config: AutoencoderConfig) -> int:
    autoencoder_type = config.type.strip().lower().replace("-", "_")
    if autoencoder_type in {"raev2", "rae_v2"}:
        return get_raev2_profile(config.raev2.profile).resolution
    if autoencoder_type in {"legacy", "rae", "legacy_rae", "dinov2"}:
        return int(config.legacy.encoder_input_size)
    raise ValueError(
        f"Unknown autoencoder type {config.type!r}. Expected 'raev2' or 'legacy'."
    )
