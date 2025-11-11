import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from training.world_trainer import load_training_config
from world_model.flow_matching import DimensionShiftedUniformScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample flow-matching noise levels and plot their importance weights."
    )
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
        default=Path("archive/fm_noise_weights.png"),
        help="Output path for the plot (default: archive/fm_noise_weights.png).",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234).")
    return parser.parse_args()


def sample_noise(cfg, total: int, batch: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    scheduler = DimensionShiftedUniformScheduler(cfg.diffusion)
    latents = torch.zeros(batch, seq_len, 1, 1)
    tau_buf = torch.empty(total, dtype=torch.float32)
    weight_buf = torch.empty(total, dtype=torch.float32)
    filled = 0
    while filled < total:
        tau, weights = scheduler.sample(latents)
        tau = tau.to(dtype=torch.float32).view(-1)
        weights = weights.to(dtype=torch.float32).view(-1)
        take = min(tau.numel(), total - filled)
        tau_buf[filled : filled + take] = tau[:take]
        weight_buf[filled : filled + take] = weights[:take]
        filled += take
    return tau_buf.numpy(), weight_buf.numpy()


def plot_noise(
    tau: np.ndarray,
    weights: np.ndarray,
    bins: int,
    output: Path,
) -> None:
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(tau, bins=edges)
    bin_indices = np.clip(np.digitize(tau, edges, right=True) - 1, 0, bins - 1)
    weight_sums = np.zeros(bins, dtype=np.float64)
    np.add.at(weight_sums, bin_indices, weights)
    weight_means = weight_sums / np.maximum(counts, 1e-8)

    fig, (ax_counts, ax_weights) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    width = edges[1] - edges[0]
    ax_counts.bar(centers, counts, width=width * 0.9, color="tab:blue", alpha=0.8)
    ax_counts.set_ylabel("Samples")
    ax_counts.set_title("Flow-Matching Noise Level Distribution")
    ax_counts.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    ax_weights.bar(centers, weight_means, width=width * 0.9, color="tab:orange", alpha=0.85)
    ax_weights.set_xlabel("tau")
    ax_weights.set_ylabel("Mean weight")
    ax_weights.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = load_training_config(args.config)
    tau, weights = sample_noise(cfg, args.samples, args.batch_size, args.sequence_length)
    print(
        f"Collected {tau.size} samples "
        f"(tau mean {tau.mean():.4f} ± {tau.std():.4f}, weight mean {weights.mean():.4f})."
    )
    plot_noise(tau, weights, args.bins, args.output)
    print(f"Saved histogram to {args.output.resolve()}")


if __name__ == "__main__":
    main()
