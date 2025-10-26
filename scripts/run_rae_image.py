#!/usr/bin/env python3
"""CLI utility to encode and decode a single image with the RAE."""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rae_dino.rae import RAE
from world_model.flow_matching import DiffusionConfig, sample_base_noise


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


def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(array, 0.0, 1.0)


def show_results(
    original: torch.Tensor,
    reconstructions: List[Dict[str, Any]],
    *,
    output_path: Path,
) -> None:
    images = [("Original", original)]
    for item in reconstructions:
        signal = item["signal"]
        mse = item["mse"]
        title = f"Signal={signal:.2f}\nMSE={mse:.4f}"
        images.append((title, item["image"]))

    total = len(images)
    cols = min(3, total)
    rows = math.ceil(total / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_2d(axes).reshape(-1)

    for ax, (title, tensor) in zip(axes, images):
        ax.imshow(_to_numpy_image(tensor))
        ax.set_title(title)
        ax.axis("off")

    for leftover_ax in axes[len(images) :]:
        leftover_ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved comparison grid to {output_path}")
    plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    device = torch.device(args.device)

    rae = RAE()
    rae.to(device)
    rae.eval()

    encoder_params = sum(p.numel() for p in rae.encoder.parameters())
    decoder_params = sum(p.numel() for p in rae.decoder.parameters())
    print(f"Number of parameters in encoder: {encoder_params}")
    print(f"Number of parameters in decoder: {decoder_params}")
    

    image_tensor = load_image(args.image_path, device)

    with torch.no_grad():
        latent = rae.encode(image_tensor)
        print(f"Encoded latent shape: {tuple(latent.shape)}")

        latents_seq = latent.unsqueeze(1)  # [B, T, tokens, dim] with T=1
        diffusion_cfg = DiffusionConfig()
        base_noise = sample_base_noise(latents_seq, diffusion_cfg)

        signal_levels = [1.0, 0.98, 0.95, 0.90, 0.75, 0.50, 0.30, 0.10]
        reconstructions = []

        for level in signal_levels:
            tau = torch.full(
                (latent.shape[0], 1),
                fill_value=level,
                device=latents_seq.device,
                dtype=latents_seq.dtype,
            )
            tau_factor = tau.unsqueeze(-1).unsqueeze(-1)
            noisy_latents = (1.0 - tau_factor) * base_noise + tau_factor * latents_seq
            mse = torch.mean((noisy_latents - latents_seq) ** 2).item()

            decoded = rae.decode(noisy_latents.squeeze(1))
            reconstructions.append(
                {
                    "signal": level,
                    "mse": mse,
                    "image": decoded.cpu(),
                }
            )
            print(f"Signal level {level:.2f}: MSE(noisy, clean) = {mse:.6f}")

    archive_dir = REPO_ROOT / "archive"
    output_path = archive_dir / f"{args.image_path.stem}_rae_denoise.png"
    show_results(
        image_tensor.cpu(),
        reconstructions,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
