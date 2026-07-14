"""DINOv3 encoder used by the released RAEv2 multi-layer checkpoints.

The multi-layer representation follows the official RAEv2 implementation at
https://github.com/nanovisionx/RAEv2 (commit 8a0d238), licensed CC BY-NC 4.0.
In particular, released checkpoints use a layer mean plus a broadcast global
mean from the final selected layer; this is intentionally more specific than
the paper's shorthand description of multi-layer "sum".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
from filelock import FileLock
from torchvision import transforms


DINOV3_HUB_REF = (
    "facebookresearch/dinov3:94a96ac83c2446f15f9bdcfae23cad3c6a9d4988"
)
DINOV3_HUB_COMMIT = DINOV3_HUB_REF.rsplit(":", 1)[1]


def load_dinov3_model(
    model_name: str,
    checkpoint_path: Path,
    *,
    repo_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> nn.Module:
    """Load pinned DINOv3 source code with an explicit local checkpoint."""

    local_repo = repo_dir or os.environ.get("DINOV3_REPO_DIR")
    if local_repo is None and local_files_only:
        cached_repo = (
            Path(torch.hub.get_dir())
            / f"facebookresearch_dinov3_{DINOV3_HUB_COMMIT}"
        )
        if (cached_repo / "hubconf.py").is_file():
            local_repo = str(cached_repo)
        else:
            raise FileNotFoundError(
                "DINOv3 source is not present in the torch.hub cache. "
                "Pre-cache the pinned repository or set autoencoder.raev2."
                "dinov3_repo_dir before using local_files_only=true."
            )
    hub_lock = Path(torch.hub.get_dir()) / ".world_model_dinov3.lock"
    hub_lock.parent.mkdir(parents=True, exist_ok=True)
    # torchrun constructs the autoencoder before this project's trainer
    # initializes torch.distributed. A filesystem lock prevents ranks sharing a
    # cache from racing while torch.hub fetches source code or materializes the
    # checkpoint for the first time.
    with FileLock(str(hub_lock)):
        if local_repo is not None:
            hubconf = Path(local_repo).expanduser() / "hubconf.py"
            if not hubconf.is_file():
                raise FileNotFoundError(f"DINOV3_REPO_DIR has no hubconf.py: {hubconf}")
            return torch.hub.load(
                str(hubconf.parent),
                model_name,
                source="local",
                trust_repo=True,
                skip_validation=True,
                weights=str(checkpoint_path),
            )

        return torch.hub.load(
            DINOV3_HUB_REF,
            model_name,
            source="github",
            trust_repo=True,
            skip_validation=True,
            weights=str(checkpoint_path),
        )


class DINOv3MultiLayerEncoder(nn.Module):
    """Frozen DINOv3 patch encoder with released RAEv2 MLS semantics."""

    def __init__(
        self,
        model: nn.Module,
        *,
        layer_indices: Sequence[int],
        resolution: int = 256,
        patch_size: int = 16,
        latent_dim: Optional[int] = None,
        strip_norm_affine: bool = True,
    ) -> None:
        super().__init__()
        if not layer_indices:
            raise ValueError("layer_indices must contain at least one transformer block.")
        if resolution <= 0 or resolution % patch_size != 0:
            raise ValueError("resolution must be positive and divisible by patch_size.")

        self.model = model
        self.layer_indices = tuple(int(index) for index in layer_indices)
        self.resolution = int(resolution)
        self.patch_size = int(patch_size)
        self.hidden_size = int(latent_dim or getattr(model, "embed_dim"))

        if strip_norm_affine:
            # RAE/RAEv2 intentionally remove learned affine parameters from the
            # final representation normalization.
            self.model.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)

        self.model.requires_grad_(False)
        self.model.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Lambda(lambda image: image / 255.0),
                transforms.Resize((self.resolution, self.resolution), antialias=True),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self.train(False)

    @classmethod
    def from_checkpoint(
        cls,
        model_name: str,
        checkpoint_path: Path,
        *,
        layer_indices: Sequence[int],
        resolution: int = 256,
        patch_size: int = 16,
        latent_dim: Optional[int] = None,
        repo_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> "DINOv3MultiLayerEncoder":
        # Always construct on CPU.  Callers can then move the complete
        # autoencoder directly to CPU, CUDA, or MPS without a transient cuda:0
        # allocation (which is especially important under torchrun).
        model = load_dinov3_model(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            repo_dir=repo_dir,
            local_files_only=local_files_only,
        )
        return cls(
            model,
            layer_indices=layer_indices,
            resolution=resolution,
            patch_size=patch_size,
            latent_dim=latent_dim,
        )

    def train(self, mode: bool = True) -> "DINOv3MultiLayerEncoder":
        # The representation encoder is frozen even if a containing world-model
        # module is switched to training mode.
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images)
        outputs = self.model.get_intermediate_layers(
            images,
            n=list(self.layer_indices),
            reshape=False,
            return_class_token=False,
            norm=True,
        )
        if not isinstance(outputs, (list, tuple)):
            raise TypeError("DINOv3 get_intermediate_layers did not return a sequence.")
        outputs = tuple(outputs)
        if len(outputs) != len(self.layer_indices):
            raise RuntimeError(
                "DINOv3 returned "
                f"{len(outputs)} layers for {len(self.layer_indices)} requested indices."
            )

        patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
        final_mean = outputs[-1].mean(dim=1, keepdim=True)
        return patch_tokens + final_mean
