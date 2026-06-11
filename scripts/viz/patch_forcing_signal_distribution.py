import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.diffusion.patch_forcing import sample_ltg_patch_signal_levels
from src.diffusion.signal_scheduler import SignalScheduler, SignalSchedulerConfig


def parse_base_steps(value: str) -> list[float]:
    steps = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not steps:
        raise argparse.ArgumentTypeError("base steps must contain at least one value.")
    for step in steps:
        if not 0.0 <= step <= 1.0:
            raise argparse.ArgumentTypeError("base steps must be in [0, 1].")
    return steps


def make_fake_image(grid_size: int, patch_pixels: int) -> np.ndarray:
    size = grid_size * patch_pixels
    y, x = np.mgrid[0:size, 0:size]
    xn = x / max(size - 1, 1)
    yn = y / max(size - 1, 1)

    image = np.stack(
        [
            0.18 + 0.52 * xn,
            0.20 + 0.55 * (1.0 - yn),
            0.35 + 0.35 * np.sin(2.0 * np.pi * (xn + yn)) ** 2,
        ],
        axis=-1,
    )

    circle = (xn - 0.34) ** 2 + (yn - 0.36) ** 2 < 0.13**2
    image[circle] = np.array([0.94, 0.35, 0.22])

    rect = (0.58 < xn) & (xn < 0.86) & (0.55 < yn) & (yn < 0.78)
    image[rect] = np.array([0.18, 0.50, 0.88])

    stripe = np.abs(yn - (0.18 + 0.52 * xn)) < 0.025
    image[stripe] = np.array([0.95, 0.85, 0.30])

    return np.clip(image, 0.0, 1.0)


def draw_grid(ax: plt.Axes, grid_size: int, patch_pixels: int, color: str = "white", alpha: float = 0.42) -> None:
    size = grid_size * patch_pixels
    for i in range(grid_size + 1):
        pos = i * patch_pixels - 0.5
        ax.axhline(pos, color=color, linewidth=0.45, alpha=alpha)
        ax.axvline(pos, color=color, linewidth=0.45, alpha=alpha)
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def upsample_patch_grid(patch_grid: np.ndarray, patch_pixels: int) -> np.ndarray:
    return np.repeat(np.repeat(patch_grid, patch_pixels, axis=0), patch_pixels, axis=1)


def sample_selected_patch_grids(
    scheduler: SignalScheduler,
    raw_base_steps: Sequence[float],
    grid_size: int,
    ltg_std: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_base = torch.tensor(raw_base_steps, dtype=torch.float32).view(-1, 1)
    base_signal = scheduler.signal_from_base(raw_base).view(-1)
    patch_signal = sample_ltg_patch_signal_levels(
        base_signal_levels=base_signal.view(-1, 1),
        num_tokens=grid_size * grid_size,
        ltg_std=ltg_std,
        generator=generator,
    ).view(len(raw_base_steps), grid_size, grid_size)
    return base_signal, patch_signal


def plot_fake_image(image: np.ndarray, grid_size: int, patch_pixels: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.imshow(image)
    draw_grid(ax, grid_size, patch_pixels)
    ax.set_title("Synthetic image with DINO patch grid")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_patch_maps(
    image: np.ndarray,
    patch_signal: torch.Tensor,
    raw_base_steps: Sequence[float],
    base_signal: torch.Tensor,
    grid_size: int,
    patch_pixels: int,
    output_path: Path,
) -> None:
    count = patch_signal.shape[0]
    cols = min(4, count)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.35 * rows), squeeze=False)

    last_overlay = None
    for idx, ax in enumerate(axes.flat):
        if idx >= count:
            ax.axis("off")
            continue

        patch_grid = patch_signal[idx].cpu().numpy()
        signal_image = upsample_patch_grid(patch_grid, patch_pixels)
        ax.imshow(image)
        last_overlay = ax.imshow(signal_image, cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.76)
        draw_grid(ax, grid_size, patch_pixels, color="black", alpha=0.26)

        title = (
            f"u={raw_base_steps[idx]:.2f}  base={base_signal[idx].item():.3f}\n"
            f"mean={patch_grid.mean():.3f}  range=[{patch_grid.min():.3f}, {patch_grid.max():.3f}]"
        )
        ax.set_title(title, fontsize=8.5)

    if last_overlay is not None:
        fig.colorbar(last_overlay, ax=axes.ravel().tolist(), shrink=0.82, label="Patch signal level")
    fig.suptitle("LTG patch signal maps under resolution-shifted base signal levels", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_summary(
    scheduler: SignalScheduler,
    raw_base_steps: Sequence[float],
    base_signal: torch.Tensor,
    grid_size: int,
    ltg_std: float,
    num_distribution_samples: int,
    num_repeats_per_step: int,
    generator: torch.Generator,
    output_path: Path,
) -> None:
    num_tokens = grid_size * grid_size
    raw_distribution = torch.rand(num_distribution_samples, 1, generator=generator)
    base_distribution = scheduler.signal_from_base(raw_distribution).view(-1)
    patch_distribution = sample_ltg_patch_signal_levels(
        base_signal_levels=base_distribution.view(-1, 1),
        num_tokens=num_tokens,
        ltg_std=ltg_std,
        generator=generator,
    ).flatten()

    repeated_base = base_signal.repeat_interleave(num_repeats_per_step).view(-1, 1)
    repeated_patch = sample_ltg_patch_signal_levels(
        base_signal_levels=repeated_base,
        num_tokens=num_tokens,
        ltg_std=ltg_std,
        generator=generator,
    ).view(len(raw_base_steps), num_repeats_per_step * num_tokens)
    quantiles = torch.quantile(repeated_patch, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0]), dim=1)
    means = repeated_patch.mean(dim=1)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

    ax = axes[0]
    bins = np.linspace(0.0, 1.0, 60)
    ax.hist(base_distribution.cpu().numpy(), bins=bins, density=True, alpha=0.55, label="Base signal")
    ax.hist(patch_distribution.cpu().numpy(), bins=bins, density=True, alpha=0.55, label="Patch signal")
    ax.set_xlabel("Signal level")
    ax.set_ylabel("Density")
    ax.set_title("Resolution-shifted base distribution and LTG patch distribution")
    ax.grid(True, alpha=0.24)
    ax.legend()

    ax = axes[1]
    x = base_signal.cpu().numpy()
    order = np.argsort(x)
    x = x[order]
    q = quantiles[:, order].cpu().numpy()
    mean = means[order].cpu().numpy()
    ax.fill_between(x, q[0], q[4], color="#8fb3ff", alpha=0.20, label="min-max")
    ax.fill_between(x, q[1], q[3], color="#3f7fe0", alpha=0.28, label="10-90%")
    ax.plot(x, q[2], color="#102a56", linewidth=2.0, label="median")
    ax.plot(x, mean, color="#d04a02", linewidth=1.8, linestyle="--", label="mean")
    ax.plot(x, x, color="#202020", linewidth=1.0, linestyle=":", label="base upper bound")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Base signal level")
    ax.set_ylabel("Patch signal level")
    ax.set_title("Patch signal statistics conditioned on base signal")
    ax.grid(True, alpha=0.24)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize patch-forcing signal distributions.")
    parser.add_argument("--output_dir", type=str, default="archive/patch_forcing_signal_distribution")
    parser.add_argument("--grid_size", type=int, default=16)
    parser.add_argument("--patch_pixels", type=int, default=16)
    parser.add_argument("--ltg_std", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--base_steps",
        type=parse_base_steps,
        default=parse_base_steps("0.02,0.05,0.10,0.20,0.35,0.55,0.75,0.95"),
        help="Comma-separated raw scheduler coordinates in [0, 1].",
    )
    parser.add_argument("--num_distribution_samples", type=int, default=4096)
    parser.add_argument("--num_repeats_per_step", type=int, default=256)
    parser.add_argument("--resolution_shift_base_dimension", type=int, default=4096)
    parser.add_argument("--resolution_shift_effective_latent_dimension", type=int, default=196608)
    args = parser.parse_args()

    if args.grid_size < 1:
        raise ValueError("--grid_size must be positive.")
    if args.patch_pixels < 1:
        raise ValueError("--patch_pixels must be positive.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(args.seed)
    scheduler = SignalScheduler(
        SignalSchedulerConfig(
            mode="resolution_shift",
            resolution_shift_base_dimension=args.resolution_shift_base_dimension,
            resolution_shift_effective_latent_dimension=args.resolution_shift_effective_latent_dimension,
        )
    )

    image = make_fake_image(args.grid_size, args.patch_pixels)
    base_signal, patch_signal = sample_selected_patch_grids(
        scheduler=scheduler,
        raw_base_steps=args.base_steps,
        grid_size=args.grid_size,
        ltg_std=args.ltg_std,
        generator=generator,
    )

    fake_image_path = output_dir / "fake_image_grid.png"
    patch_maps_path = output_dir / "patch_signal_maps.png"
    summary_path = output_dir / "patch_signal_summary.png"

    plot_fake_image(image, args.grid_size, args.patch_pixels, fake_image_path)
    plot_patch_maps(
        image=image,
        patch_signal=patch_signal,
        raw_base_steps=args.base_steps,
        base_signal=base_signal,
        grid_size=args.grid_size,
        patch_pixels=args.patch_pixels,
        output_path=patch_maps_path,
    )
    plot_distribution_summary(
        scheduler=scheduler,
        raw_base_steps=args.base_steps,
        base_signal=base_signal,
        grid_size=args.grid_size,
        ltg_std=args.ltg_std,
        num_distribution_samples=args.num_distribution_samples,
        num_repeats_per_step=args.num_repeats_per_step,
        generator=generator,
        output_path=summary_path,
    )

    print(f"Saved fake image grid to: {fake_image_path}")
    print(f"Saved patch signal maps to: {patch_maps_path}")
    print(f"Saved patch signal summary to: {summary_path}")


if __name__ == "__main__":
    main()
