#!/usr/bin/env python3
"""CLI utility to encode and decode a single image with the RAE."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _setup_matplotlib_cache() -> None:
    mpl_dir = Path(os.environ.get("MPLCONFIGDIR", Path.cwd() / ".matplotlib"))
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)


_setup_matplotlib_cache()

import matplotlib.pyplot as plt  # noqa: E402

from rae_dino.rae import RAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single image through the RAE encoder/decoder."
    )
    parser.add_argument("image_path", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on (e.g., cpu, cuda, cuda:0).",
    )
    return parser.parse_args()


def load_image(image_path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    array = np.array(image).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def show_images(original: torch.Tensor, reconstructed: torch.Tensor) -> None:
    original_np = original.squeeze(0).permute(1, 2, 0).cpu().numpy()
    reconstructed_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    reconstructed_np = np.clip(reconstructed_np, 0.0, 1.0)

    plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(original_np)
    ax1.set_title("Original")
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(reconstructed_np)
    ax2.set_title("RAE Reconstruction")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    device = torch.device(args.device)

    rae = RAE()
    rae.to(device)
    rae.eval()

    image_tensor = load_image(args.image_path, device)

    with torch.no_grad():
        latent = rae.encode(image_tensor)
        print(f"Encoded latent shape: {tuple(latent.shape)}")
        reconstructed = rae.decode(latent)

    show_images(image_tensor.cpu(), reconstructed.cpu())


if __name__ == "__main__":
    main()
