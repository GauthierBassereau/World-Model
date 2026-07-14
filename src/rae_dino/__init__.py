"""Representation autoencoders used by the world model."""

from src.rae_dino.config import (
    AutoencoderConfig,
    LegacyRAEAutoencoderConfig,
    RAEv2AutoencoderConfig,
    build_autoencoder,
    configured_autoencoder_resolution,
    validate_autoencoder_input_dim,
)

__all__ = [
    "AutoencoderConfig",
    "LegacyRAEAutoencoderConfig",
    "RAEv2AutoencoderConfig",
    "build_autoencoder",
    "configured_autoencoder_resolution",
    "validate_autoencoder_input_dim",
]
