import argparse
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import pyrallis
from training.dataset import DataloaderConfig, DatasetConfig, build_world_model_dataloader
from training.world_trainer import WorldModelTrainingConfig


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
        "--stats-output",
        type=Path,
        default=Path("configs/action_distribution_stats.pt"),
        help=(
            "Path where per-dimension action statistics (mean/std/min/max) will be saved "
            "as a PyTorch file (default: action_distribution_stats.pt)."
        ),
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
        help="Use the drop_action_probability and independent_frames_probability from the config "
        "(by default they are forced to zero for clean statistics).",
    )
    parser.add_argument(
        "--preserve-normalization",
        action="store_true",
        help="Use the action normalization settings from the config "
        "(by default they are disabled to inspect raw action values).",
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
    preserve_normalization: bool,
) -> Tuple[DatasetConfig, DataloaderConfig, torch.utils.data.DataLoader]:
    training_cfg = pyrallis.parse(config_class=WorldModelTrainingConfig, config_path=str(config_path), args=[])

    dataset_cfg = training_cfg.train_data.train_dataset
    if not preserve_dropout:
        dataset_cfg = replace(
            dataset_cfg,
            drop_action_probability=0.0,
            independent_frames_probability=0.0,
        )
    if not preserve_normalization and dataset_cfg.action_normalization is not None:
        dataset_cfg = replace(
            dataset_cfg,
            action_normalization=None,
            action_normalization_params=None,
        )

    dataloader_cfg = training_cfg.train_data.train_dataloader
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
) -> np.ndarray:
    action_vectors = []
    last_report_count = -1

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
                current_count = len(action_vectors)
                if current_count != last_report_count:
                    print(
                        f"Collecting action vectors: {current_count}/{num_samples}",
                        end="\r",
                        flush=True,
                    )
                    last_report_count = current_count
                if len(action_vectors) >= num_samples:
                    break

    stacked = np.stack(action_vectors, axis=0)
    print()  # ensure the next print starts on a new line
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _, dataloader_cfg, loader = prepare_dataset_and_loader(
        args.config,
        batch_size_override=args.batch_size,
        preserve_dropout=args.preserve_dropout,
        preserve_normalization=args.preserve_normalization,
    )
    try:
        dataset_length = len(loader.dataset)
        print(f"Dataset length: {dataset_length}")
    except TypeError:
        print("Dataset length: unknown (dataset does not define __len__)")
    print(
        f"Sampling up to {args.num_samples} action vectors "
        f"using batch_size={dataloader_cfg.batch_size} from dataset..."
    )
    samples = collect_action_samples(loader, args.num_samples)
    print(f"Collected {samples.shape[0]} action vectors of dimension {samples.shape[1]}.")

    means = samples.mean(axis=0)
    stds = samples.std(axis=0)
    minimums = samples.min(axis=0)
    maximums = samples.max(axis=0)

    print("Per-dimension summary statistics:")
    for idx, (mean, std, min_val, max_val) in enumerate(zip(means, stds, minimums, maximums)):
        print(
            f"  dim {idx:02d} | mean={mean:+.4f} std={std:.4f} "
            f"min={min_val:+.4f} max={max_val:+.4f}"
        )

    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": torch.tensor(means, dtype=torch.float32),
            "std": torch.tensor(stds, dtype=torch.float32),
            "min": torch.tensor(minimums, dtype=torch.float32),
            "max": torch.tensor(maximums, dtype=torch.float32),
        },
        args.stats_output,
    )
    print(f"Statistics saved to {args.stats_output.resolve()}")

    plot_histograms(samples, args.output)
    print(f"Histogram saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
