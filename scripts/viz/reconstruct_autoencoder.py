"""Reconstruct one image with a released RAEv2 profile.

The first run downloads the profile's encoder, decoder, and normalization
statistics into the Hugging Face cache.  Outputs are written under ``archive/``
by default so reconstruction quality remains an explicit experiment artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rae_dino import AutoencoderConfig, build_autoencoder


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("--device must be one of: auto, cpu, cuda, mps.")
    return torch.device(requested)


def load_rgb_image(path: Path, resolution: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = ImageOps.fit(
            image,
            (resolution, resolution),
            method=Image.Resampling.BICUBIC,
        )
        return pil_to_tensor(image).float().div_(255.0).unsqueeze(0)


def _comparison(original: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    separator = torch.ones(
        (original.shape[0], original.shape[1], original.shape[2], 8),
        dtype=original.dtype,
    )
    return torch.cat((original, separator, reconstruction), dim=-1)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def reconstruct(args: argparse.Namespace) -> dict:
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image does not exist: {args.image}")

    config = AutoencoderConfig()
    config.type = "raev2"
    config.raev2.profile = args.profile
    config.raev2.cache_dir = str(args.cache_dir) if args.cache_dir else None
    config.raev2.local_files_only = args.local_files_only
    config.raev2.dinov3_repo_dir = (
        str(args.dinov3_repo_dir) if args.dinov3_repo_dir else None
    )

    device = resolve_device(args.device)
    autoencoder = build_autoencoder(config).eval().to(device)
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    original = load_rgb_image(args.image, autoencoder.resolution)
    with torch.inference_mode():
        latent = autoencoder.encode(original.to(device))
        reconstruction = autoencoder.decode(latent).float().clamp(0.0, 1.0)
    reconstruction_cpu = reconstruction.cpu()

    mse = torch.mean((original - reconstruction_cpu) ** 2).item()
    psnr = -10.0 * math.log10(max(mse, 1e-12))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or f"{args.profile}_{args.image.stem}"
    original_path = args.output_dir / f"{stem}_original.png"
    reconstruction_path = args.output_dir / f"{stem}_reconstruction.png"
    comparison_path = args.output_dir / f"{stem}_comparison.png"
    metadata_path = args.output_dir / f"{stem}.json"

    save_image(original, original_path)
    save_image(reconstruction_cpu, reconstruction_path)
    save_image(_comparison(original, reconstruction_cpu), comparison_path)

    metadata = {
        "profile": args.profile,
        "input": _portable_path(args.image),
        "device": str(device),
        "image_shape": list(original.shape),
        "latent_shape": list(latent.shape),
        "reconstruction_shape": list(reconstruction_cpu.shape),
        "mse": mse,
        "psnr_db": psnr,
        "original": _portable_path(original_path),
        "reconstruction": _portable_path(reconstruction_path),
        "comparison": _portable_path(comparison_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata["metadata"] = _portable_path(metadata_path)
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="RGB input image.")
    parser.add_argument("--profile", default="dinov3b-k11")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("archive/autoencoder_reconstruction"),
    )
    parser.add_argument("--stem", default=None, help="Optional output filename prefix.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dinov3-repo-dir", type=Path, default=None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Fail instead of downloading when the profile is absent from cache.",
    )
    return parser.parse_args(argv)


def main() -> None:
    metadata = reconstruct(parse_args())
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
