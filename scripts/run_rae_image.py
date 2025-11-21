"""CLI utility to encode and decode a single image with the RAE."""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

ENCODER_INPUT_SIZE = 224
COMPARISON_IMAGE_SIZE = 256
CPU_DEVICE = torch.device("cpu")

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

from src.rae_dino.rae import RAE
from src.world_model.diffusion import DiffusionConfig, sample_base_noise


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


def _pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.array(image).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def load_image(image_path: Path, device: torch.device) -> Tuple[Image.Image, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    return image, _pil_to_tensor(image, device)


def crop_and_resize_to_square(image: Image.Image, size: int = 224) -> Image.Image:
    width, height = image.size
    if width == height:
        cropped = image
    else:
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        cropped = image.crop((left, top, right, bottom))
    return cropped.resize((size, size), Image.BICUBIC)


def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(array, 0.0, 1.0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (_to_numpy_image(tensor) * 255).astype("uint8")
    return Image.fromarray(array)


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    image = tensor_to_pil(tensor)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    print(f"Saved image to {path}")


def compute_difference_tensor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(a - b)
    return torch.clamp(diff, 0.0, 1.0)


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


    pil_image, original_tensor = load_image(args.image_path, device)
    resized_image = crop_and_resize_to_square(pil_image, size=ENCODER_INPUT_SIZE)
    resized_tensor = _pil_to_tensor(resized_image, device)

    comparison_image = crop_and_resize_to_square(pil_image, size=COMPARISON_IMAGE_SIZE)
    comparison_tensor_cpu = _pil_to_tensor(comparison_image, CPU_DEVICE)

    archive_tests_dir = REPO_ROOT / "archive" / "rae_tests"
    noisy_dir = archive_tests_dir / "noisy"
    archive_tests_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    image_stem = args.image_path.stem

    original_image_path = archive_tests_dir / f"{image_stem}_original_image.png"
    pil_image.save(original_image_path)
    print(f"Saved image to {original_image_path}")

    original_image_256_path = archive_tests_dir / f"{image_stem}_original_image_{COMPARISON_IMAGE_SIZE}.png"
    comparison_image.save(original_image_256_path)
    print(f"Saved image to {original_image_256_path}")

    with torch.no_grad():
        original_latent = rae.encode(original_tensor)
        print(f"Original latent shape: {tuple(original_latent.shape)}")
        original_reconstruction = rae.decode(original_latent).cpu()

        resized_latent = rae.encode(resized_tensor)
        print(f"224x224 latent shape: {tuple(resized_latent.shape)}")
        resized_reconstruction = rae.decode(resized_latent).cpu()

        original_output_path = (
            archive_tests_dir / f"{image_stem}_original_reconstruction.png"
        )
        resized_output_path = (
            archive_tests_dir / f"{image_stem}_224_reconstruction.png"
        )
        save_tensor_image(original_reconstruction, original_output_path)
        save_tensor_image(resized_reconstruction, resized_output_path)

        original_recon_256_image = crop_and_resize_to_square(
            tensor_to_pil(original_reconstruction),
            size=COMPARISON_IMAGE_SIZE,
        )
        original_recon_256_path = (
            archive_tests_dir
            / f"{image_stem}_original_reconstruction_{COMPARISON_IMAGE_SIZE}.png"
        )
        original_recon_256_image.save(original_recon_256_path)
        print(f"Saved image to {original_recon_256_path}")
        original_recon_256_tensor_cpu = _pil_to_tensor(
            original_recon_256_image,
            CPU_DEVICE,
        )

        latents_seq = resized_latent.unsqueeze(1)  # [B, T, tokens, dim] with T=1
        diffusion_cfg = DiffusionConfig()
        base_noise = sample_base_noise(latents_seq, diffusion_cfg)

        signal_levels = [1.0, 0.98, 0.95, 0.90, 0.75, 0.50, 0.30, 0.10]
        reconstructions = []

        for level in signal_levels:
            tau = torch.full(
                (resized_latent.shape[0], 1),
                fill_value=level,
                device=latents_seq.device,
                dtype=latents_seq.dtype,
            )
            tau_factor = tau.unsqueeze(-1).unsqueeze(-1)
            noisy_latents = (1.0 - tau_factor) * base_noise + tau_factor * latents_seq
            mse = torch.mean((noisy_latents - latents_seq) ** 2).item()

            decoded = rae.decode(noisy_latents.squeeze(1)).cpu()
            reconstructions.append(
                {
                    "signal": level,
                    "mse": mse,
                    "image": decoded,
                }
            )
            print(f"Signal level {level:.2f}: MSE(noisy, clean) = {mse:.6f}")

            noise_filename = (
                f"{image_stem}_224_signal_{int(level * 100):03d}.png"
            )
            save_tensor_image(decoded, noisy_dir / noise_filename)

        diff_224_vs_original_path = (
            archive_tests_dir
            / f"{image_stem}_diff_224_reconstruction_vs_original_{COMPARISON_IMAGE_SIZE}.png"
        )
        diff_original_vs_original_path = (
            archive_tests_dir
            / f"{image_stem}_diff_original_reconstruction_vs_original_{COMPARISON_IMAGE_SIZE}.png"
        )
        diff_recon_pair_path = (
            archive_tests_dir
            / f"{image_stem}_diff_original_{COMPARISON_IMAGE_SIZE}_vs_224_reconstruction.png"
        )

        diff_224_vs_original = compute_difference_tensor(
            resized_reconstruction,
            comparison_tensor_cpu,
        )
        save_tensor_image(diff_224_vs_original, diff_224_vs_original_path)

        diff_original_vs_original = compute_difference_tensor(
            original_recon_256_tensor_cpu,
            comparison_tensor_cpu,
        )
        save_tensor_image(diff_original_vs_original, diff_original_vs_original_path)

        diff_recon_pair = compute_difference_tensor(
            original_recon_256_tensor_cpu,
            resized_reconstruction,
        )
        save_tensor_image(diff_recon_pair, diff_recon_pair_path)

    noise_grid_path = noisy_dir / f"{image_stem}_224_noise_grid.png"
    show_results(
        resized_tensor.cpu(),
        reconstructions,
        output_path=noise_grid_path,
    )


if __name__ == "__main__":
    main()
