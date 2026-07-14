"""Released, mutually compatible RAEv2 asset profiles.

The encoder representation, decoder, and normalization statistics are an
atomic set: mixing files from different profiles silently produces invalid
latents and reconstructions.  Keep the profile table explicit for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


RAEV2_MODEL_REPO = "nyu-visionx/RAEv2-models"
# Pin the released snapshot so a future update of ``main`` cannot change the
# world-model latent space underneath an experiment.
RAEV2_MODEL_REVISION = "9770b7b980fa1875c8e6d65f226c615c0ce908a8"


@dataclass(frozen=True)
class RAEv2Profile:
    name: str
    encoder_model_name: str
    encoder_checkpoint: str
    decoder_checkpoint: str
    normalization_stats: str
    encoder_layers: Tuple[int, ...]
    resolution: int
    patch_size: int
    latent_dim: int
    decoder_patch_size: int = 16

    @property
    def tokens_per_frame(self) -> int:
        return (self.resolution // self.patch_size) ** 2

    @property
    def asset_filenames(self) -> Tuple[str, str, str]:
        return (
            self.encoder_checkpoint,
            self.decoder_checkpoint,
            self.normalization_stats,
        )


# DINOv3-B has twelve transformer blocks indexed 0..11.  RAEv2's released
# maximal multi-layer profile uses blocks 1..11 (K=11), matching its convention
# of excluding the first block for the "full" MLS representation.
RAEV2_PROFILES = {
    "dinov3b-k11": RAEv2Profile(
        name="dinov3b-k11",
        encoder_model_name="dinov3_vitb16",
        encoder_checkpoint=(
            "encoders/dinov3/"
            "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        ),
        decoder_checkpoint="stage1/imagenet/dinov3b-k11/decoder.pt",
        normalization_stats="stage1/imagenet/dinov3b-k11/stats.pt",
        encoder_layers=tuple(range(1, 12)),
        resolution=256,
        patch_size=16,
        latent_dim=768,
    ),
    "dinov3l-k7-general": RAEv2Profile(
        name="dinov3l-k7-general",
        encoder_model_name="dinov3_vitl16",
        encoder_checkpoint=(
            "encoders/dinov3/"
            "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        ),
        decoder_checkpoint="stage1/general/dinov3l-k7/decoder.pt",
        normalization_stats="stage1/general/dinov3l-k7/stats.pt",
        encoder_layers=(11, 13, 15, 17, 19, 21, 23),
        resolution=256,
        patch_size=16,
        latent_dim=1024,
    ),
    "dinov3l-k23-general": RAEv2Profile(
        name="dinov3l-k23-general",
        encoder_model_name="dinov3_vitl16",
        encoder_checkpoint=(
            "encoders/dinov3/"
            "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        ),
        decoder_checkpoint="stage1/general/dinov3l-k23/decoder.pt",
        normalization_stats="stage1/general/dinov3l-k23/stats.pt",
        encoder_layers=tuple(range(1, 24)),
        resolution=256,
        patch_size=16,
        latent_dim=1024,
    ),
}


@dataclass(frozen=True)
class ResolvedRAEv2Assets:
    profile: RAEv2Profile
    encoder_checkpoint: Path
    decoder_checkpoint: Path
    normalization_stats: Path


def get_raev2_profile(name: str) -> RAEv2Profile:
    try:
        return RAEV2_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(RAEV2_PROFILES))
        raise ValueError(f"Unknown RAEv2 profile {name!r}. Available profiles: {available}.") from exc


def resolve_raev2_assets(
    profile: RAEv2Profile,
    *,
    repo_id: str = RAEV2_MODEL_REPO,
    revision: str = RAEV2_MODEL_REVISION,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> ResolvedRAEv2Assets:
    """Download one pinned profile snapshot and return its three local paths."""

    # Import lazily so config parsing and unit tests do not require Hub/network
    # initialization.
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(profile.asset_filenames),
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )
    paths = [snapshot / filename for filename in profile.asset_filenames]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"RAEv2 profile {profile.name!r} is incomplete in {snapshot}: {missing}"
        )
    return ResolvedRAEv2Assets(
        profile=profile,
        encoder_checkpoint=paths[0],
        decoder_checkpoint=paths[1],
        normalization_stats=paths[2],
    )
