import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.training.world_trainer import load_training_config
from src.world_model.diffusion import NoiseScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample flow-matching noise levels and plot their distribution.")
    parser.add_argument("--config", type=Path, required=True, help="Training YAML config.")
    parser.add_argument(
        "--samples",
        type=int,
        default=10_000,
        help="Total tau samples to collect (default: 200k).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size per scheduler call (default: 512).",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=4,
        help="Sequence length per scheduler call (default: 4).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of histogram bins over [0, 1] (default: 80).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("archive/fm_noise_schedule.png"),
        help="Output path for the plot (default: archive/fm_noise_schedule.png).",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234).")
    return parser.parse_args()


def sample_noise(cfg, total: int, batch: int, seq_len: int) -> np.ndarray:
    scheduler = NoiseScheduler(cfg.diffusion)
    latents = torch.zeros(batch, seq_len, 1, 1)
    tau_buf = torch.empty(total, dtype=torch.float32)
    filled = 0
    while filled < total:
        tau = scheduler.sample(latents)
        tau = tau.to(dtype=torch.float32).view(-1)
        take = min(tau.numel(), total - filled)
        tau_buf[filled : filled + take] = tau[:take]
        filled += take
    return tau_buf.numpy()


def plot_noise(
    tau: np.ndarray,
    bins: int,
    output: Path,
) -> None:
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(tau, bins=edges)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    width = edges[1] - edges[0]
    ax.bar(centers, counts, width=width * 0.9, color="tab:blue", alpha=0.8)
    ax.set_xlabel("tau")
    ax.set_ylabel("Samples")
    ax.set_title("Flow-Matching Noise Level Distribution")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = load_training_config(args.config)
    tau = sample_noise(cfg, args.samples, args.batch_size, args.sequence_length)
    print(
        f"Collected {tau.size} samples "
        f"(tau mean {tau.mean():.4f} ± {tau.std():.4f})."
    )
    plot_noise(tau, args.bins, args.output)
    print(f"Saved histogram to {args.output.resolve()}")


if __name__ == "__main__":
    main()
