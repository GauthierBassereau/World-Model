import logging
import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import torch

from src.dataset.world_dataset import WorldDataset, WorldDatasetConfig


@dataclass
class ActionDistributionConfig:
    output_dir: str = "archive/action_distribution"
    num_samples: int = 1000
    dataset: WorldDatasetConfig = field(default_factory=WorldDatasetConfig)


def main():
    parser = argparse.ArgumentParser(description="Action Distribution Visualization")
    parser.add_argument("--config_path", type=str, default="configs/viz_action_distribution.yaml")
    args = parser.parse_args()

    config = pyrallis.parse(config_class=ActionDistributionConfig, config_path=args.config_path)
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/action_distribution.log"),
        ],
    )
    logger = logging.getLogger()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = WorldDataset(config.dataset, logger=logger)
    logger.info(f"Dataset size: {len(dataset)}")

    logger.info(f"\nCollecting {config.num_samples} samples...")
    all_actions = []
    
    indices = np.random.choice(len(dataset), size=config.num_samples, replace=False)
    
    for idx_num, idx in enumerate(indices):
        batch = dataset[idx]
        
        actions = batch.sequence_actions
        mask = batch.use_actions
        
        valid_actions = actions[mask]
        
        if valid_actions.numel() > 0:
            all_actions.append(valid_actions)
        
        logger.info(f"  Collected {idx_num + 1}/{config.num_samples} samples")
    
    all_actions = torch.cat(all_actions, dim=0).cpu().numpy()
    logger.info(f"\nTotal action frames collected: {all_actions.shape[0]}")
    logger.info(f"Action dimensions: {all_actions.shape[1]}")
    
    action_dim = all_actions.shape[1]
    
    dimension_names = [
        "Delta X (m)",
        "Delta Y (m)", 
        "Delta Z (m)",
        "Delta Rot X (rad)",
        "Delta Rot Y (rad)",
        "Delta Rot Z (rad)",
        "Gripper",
    ]
    
    if len(dimension_names) != action_dim:
        dimension_names = [f"Action Dim {i}" for i in range(action_dim)]
    
    logger.info("\n" + "=" * 80)
    logger.info("ACTION STATISTICS")
    logger.info("=" * 80)
    logger.info(f"{'Dimension':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'>3std':<8} {'>5std':<8} {'MaxZ':<8}")
    logger.info("-" * 95)
    
    for i in range(action_dim):
        dim_data = all_actions[:, i]
        mean = np.mean(dim_data)
        std = np.std(dim_data)
        min_val = np.min(dim_data)
        max_val = np.max(dim_data)
        
        z_scores = np.abs((dim_data - mean) / (std + 1e-8))
        outliers_3std = np.sum(z_scores > 3)
        outliers_5std = np.sum(z_scores > 5)
        max_z = np.max(z_scores)
        
        logger.info(f"{dimension_names[i]:<20} {mean:<10.6f} {std:<10.6f} {min_val:<10.6f} {max_val:<10.6f} {outliers_3std:<8} {outliers_5std:<8} {max_z:<8.2f}")
    
    logger.info("=" * 80)
    
    logger.info("\nGenerating distribution plots...")
    
    for i in range(action_dim):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        dim_data = all_actions[:, i]
        
        n_bins = 50
        counts, bins, patches = ax.hist(dim_data, bins=n_bins, alpha=0.7, color='steelblue', edgecolor='black')
        
        mean = np.mean(dim_data)
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
        
        std = np.std(dim_data)
        ax.axvline(mean - std, color='orange', linestyle=':', linewidth=1.5, label=f'±1 Std: {std:.4f}')
        ax.axvline(mean + std, color='orange', linestyle=':', linewidth=1.5)
        
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distribution of {dimension_names[i]}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        z_scores = np.abs((dim_data - mean) / (std + 1e-8))
        outliers_3std = np.sum(z_scores > 3)
        outliers_5std = np.sum(z_scores > 5)
        max_z = np.max(z_scores)

        stats_text = f"Statistics:\n"
        stats_text += f"Mean: {mean:.6f}\n"
        stats_text += f"Std: {std:.6f}\n"
        stats_text += f"Min: {np.min(dim_data):.6f}\n"
        stats_text += f"Max: {np.max(dim_data):.6f}\n"
        stats_text += f"Median: {np.median(dim_data):.6f}\n"
        stats_text += f"Outliers >3std: {outliers_3std}\n"
        stats_text += f"Outliers >5std: {outliers_5std}\n"
        stats_text += f"Max Z-Score: {max_z:.2f}"
        
        ax.text(0.98, 0.98, stats_text, 
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        plot_path = output_dir / f"action_dim_{i}_{dimension_names[i].replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        logger.info(f"  Saved plot for dimension {i}: {plot_path}")
    
    logger.info("\nGenerating combined plot...")
    n_cols = 3
    n_rows = (action_dim + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten() if action_dim > 1 else [axes]
    
    for i in range(action_dim):
        ax = axes[i]
        dim_data = all_actions[:, i]
        
        n_bins = 50
        ax.hist(dim_data, bins=n_bins, alpha=0.7, color='steelblue', edgecolor='black')
        
        mean = np.mean(dim_data)
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
        
        std = np.std(dim_data)
        ax.axvline(mean - std, color='orange', linestyle=':', linewidth=1.5)
        ax.axvline(mean + std, color='orange', linestyle=':', linewidth=1.5)
        
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(dimension_names[i], fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for i in range(action_dim, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Action Distribution Across All Dimensions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    combined_plot_path = output_dir / "action_distribution_combined.png"
    plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\nSaved combined plot to: {combined_plot_path}")
    logger.info(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
