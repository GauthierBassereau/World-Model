import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from training.dataset import DataloaderConfig, DatasetConfig, build_world_model_dataloader
from training.world_trainer import load_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample action sequences from a dataset configuration and plot per-dimension histograms."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML config file containing a training dataset configuration.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Total number of action vectors to collect before generating histograms (default: 1000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("archive/action_distribution.png"),
        help="Path where the histogram figure will be saved (default: action_distribution.png).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override dataloader batch size used for sampling.",
    )
    parser.add_argument(
        "--preserve-dropout",
        action="store_true",
        help="Use the drop_action_probability and independant_frames_probability from the config "
        "(by default they are forced to zero for clean statistics).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for dataloader sampling (default: 1234).",
    )
    return parser.parse_args()


def prepare_dataset_and_loader(
    config_path: Path,
    *,
    batch_size_override: int | None,
    preserve_dropout: bool,
) -> Tuple[DatasetConfig, DataloaderConfig, torch.utils.data.DataLoader]:
    training_cfg = load_training_config(config_path)

    dataset_cfg = training_cfg.dataset
    if not preserve_dropout:
        dataset_cfg = replace(
            dataset_cfg,
            drop_action_probability=0.0,
            independant_frames_probability=0.0,
        )

    dataloader_cfg = training_cfg.dataloader
    shuffle = True
    if batch_size_override is not None:
        dataloader_cfg = replace(dataloader_cfg, batch_size=max(1, batch_size_override))
    elif dataloader_cfg.batch_size <= 0:
        dataloader_cfg = replace(dataloader_cfg, batch_size=1)
    if not dataloader_cfg.shuffle:
        dataloader_cfg = replace(dataloader_cfg, shuffle=shuffle)

    loader = build_world_model_dataloader(
        dataset_cfg=dataset_cfg,
        dataloader_cfg=dataloader_cfg,
        device=torch.device("cpu"),
    )
    return dataset_cfg, dataloader_cfg, loader


def collect_action_samples(
    loader: torch.utils.data.DataLoader,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    action_vectors = []

    iterator = iter(loader)
    with torch.no_grad():
        while len(action_vectors) < num_samples:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            actions = batch.sequence_actions  # [B, T, D]
            mask = batch.actions_mask.to(dtype=torch.bool)

            valid = actions[mask]
            if valid.numel() == 0:
                continue

            for vector in valid:
                action_vectors.append(vector.cpu().numpy())
                if len(action_vectors) >= num_samples:
                    break

    stacked = np.stack(action_vectors, axis=0)
    return stacked


def plot_histograms(data: np.ndarray, output_path: Path) -> None:
    dims = data.shape[1]
    cols = min(4, dims)
    rows = math.ceil(dims / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows), squeeze=False)
    for idx in range(rows * cols):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        if idx >= dims:
            ax.axis("off")
            continue
        ax.hist(data[:, idx], bins=64, color="tab:blue", alpha=0.75)
        ax.set_title(f"Dimension {idx}", fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.suptitle("Action Value Distribution", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    _, dataloader_cfg, loader = prepare_dataset_and_loader(
        args.config,
        batch_size_override=args.batch_size,
        preserve_dropout=args.preserve_dropout,
    )
    print(
        f"Sampling up to {args.num_samples} action vectors "
        f"using batch_size={dataloader_cfg.batch_size} from dataset..."
    )
    samples = collect_action_samples(loader, args.num_samples, args.seed)
    print(f"Collected {samples.shape[0]} action vectors of dimension {samples.shape[1]}.")

    means = samples.mean(axis=0)
    stds = samples.std(axis=0)
    print("Per-dimension summary statistics:")
    for idx, (mean, std) in enumerate(zip(means, stds)):
        print(f"  dim {idx:02d} | mean={mean:+.4f} std={std:.4f}")

    plot_histograms(samples, args.output)
    print(f"Histogram saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
