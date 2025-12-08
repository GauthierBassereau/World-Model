"""
Noise Analysis Visualization Script

This script analyzes how noise at different signal levels affects encoded images.
It measures L1 and L2 losses between noisy and clean encoded representations,
generates plots, and displays decoded images at each noise level.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import torch

from src.dataset.world_dataset import WorldDataset, WorldDatasetConfig
from src.rae_dino.rae import RAE


@dataclass
class NoiseAnalysisConfig:
    output_dir: str = "archive/noise_analysis"
    signal_levels: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9])
    # Loss weighting parameters (matching training config)
    loss_weighting_intercept: float = 0.1
    loss_weighting_slope: float = 0.9
    dataset: WorldDatasetConfig = field(default_factory=WorldDatasetConfig)


class DummyLogger:
    """Minimal logger for dataset initialization."""
    def info(self, msg: str) -> None:
        print(f"[INFO] {msg}")
    
    def warning(self, msg: str) -> None:
        print(f"[WARNING] {msg}")


def compute_losses(noisy_latents: torch.Tensor, clean_latents: torch.Tensor) -> tuple[float, float]:
    """Compute L1 and L2 (RMSE) losses between noisy and clean latents."""
    l1_loss = torch.mean(torch.abs(noisy_latents - clean_latents)).item()
    l2_loss = torch.sqrt(torch.mean((noisy_latents - clean_latents) ** 2)).item()
    return l1_loss, l2_loss


def compute_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute MSE loss (mean squared error)."""
    return torch.mean((pred - target) ** 2).item()


def add_noise(latents: torch.Tensor, signal_level: float, noise: torch.Tensor) -> torch.Tensor:
    """Add noise to latents following the diffusion model convention.
    
    signal_level = 1.0 -> clean signal (no noise)
    signal_level = 0.0 -> pure noise (no signal)
    """
    return (1.0 - signal_level) * noise + signal_level * latents


def main():
    parser = argparse.ArgumentParser(description="Noise Analysis Visualization")
    parser.add_argument("--config_path", type=str, default="configs/viz_noise_analysis.yaml")
    args = parser.parse_args()

    config = pyrallis.parse(config_class=NoiseAnalysisConfig, config_path=args.config_path)
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize dataset
    logger = DummyLogger()
    dataset = WorldDataset(config.dataset, logger=logger)
    print(f"Dataset size: {len(dataset)}")

    # Initialize autoencoder
    autoencoder = RAE()
    autoencoder.to(device)
    autoencoder.eval()

    # Get first frame from dataset
    batch = dataset[0]
    # sequence_frames shape: [T, C, H, W] where T is sequence_length
    print(f"Raw sequence_frames shape: {batch.sequence_frames.shape}")
    
    if batch.sequence_frames.dim() == 4:
        # Shape is [T, C, H, W], take first frame
        frame = batch.sequence_frames[0]  # [C, H, W]
    else:
        # Shape is [C, H, W] directly
        frame = batch.sequence_frames
    
    frame = frame.unsqueeze(0).to(device)  # Add batch dimension [1, C, H, W]
    frame = frame.float() / 255.0

    print(f"Frame shape: {frame.shape}")

    # Encode the clean frame
    with torch.no_grad():
        clean_latents = autoencoder.encode(frame)  # [1, tokens, dim]
    
    print(f"Latent shape: {clean_latents.shape}")

    # Generate base noise (same noise for all signal levels for fair comparison)
    base_noise = torch.randn_like(clean_latents)

    # Sort signal levels for consistent plotting
    signal_levels = sorted(config.signal_levels)
    
    # Storage for results
    l1_losses = []
    l2_losses = []
    velocity_losses = []
    reweighted_losses = []
    decoded_images = []

    print("\nAnalyzing noise levels...")
    for signal_level in signal_levels:
        with torch.no_grad():
            # Add noise
            noisy_latents = add_noise(clean_latents, signal_level, base_noise)
            
            # ===== Position Loss (L1, L2) =====
            l1, l2 = compute_losses(noisy_latents, clean_latents)
            l1_losses.append(l1)
            l2_losses.append(l2)
            
            # ===== Velocity Loss =====
            # Velocity loss = position_mse / (1 - signal_level)²
            position_mse = torch.mean((noisy_latents - clean_latents) ** 2).item()
            denom = max((1.0 - signal_level) ** 2, 0.05 ** 2)  # Clamp to avoid division by zero
            velocity_loss = position_mse / denom
            velocity_losses.append(velocity_loss)
            
            # ===== Reweighted Loss =====
            # weight = intercept + slope * signal_level
            weight = config.loss_weighting_intercept + config.loss_weighting_slope * signal_level
            position_mse = torch.mean((noisy_latents - clean_latents) ** 2).item()
            reweighted_loss = position_mse * weight
            reweighted_losses.append(reweighted_loss)
            
            # Decode for visualization
            decoded = autoencoder.decode(noisy_latents)
            decoded = decoded.clamp(0, 1)
            decoded_images.append(decoded.cpu())
        
        print(f"  Signal level {signal_level:.2f}: L1={l1:.4f}, L2={l2:.4f}, Velocity={velocity_loss:.4f}, Reweighted={reweighted_loss:.4f}")

    # Also decode clean image for reference
    with torch.no_grad():
        clean_decoded = autoencoder.decode(clean_latents)
        clean_decoded = clean_decoded.clamp(0, 1).cpu()

    # ==================== Plot 1: Loss vs Signal Level ====================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(signal_levels, l1_losses, 'o-', label='L1 Loss', linewidth=2, markersize=8)
    ax.plot(signal_levels, l2_losses, 's-', label='L2 Loss (RMSE)', linewidth=2, markersize=8)
    ax.plot(signal_levels, velocity_losses, '^-', label='Velocity Loss (MSE)', linewidth=2, markersize=8)
    ax.plot(signal_levels, reweighted_losses, 'd-', label=f'Reweighted Loss (w={config.loss_weighting_intercept}+{config.loss_weighting_slope}*s)', linewidth=2, markersize=8)
    
    ax.set_xlabel('Signal Level', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Latent Space Losses vs Signal Level', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_xticks(signal_levels)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    loss_plot_path = output_dir / "loss_vs_signal_level.png"
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f"\nSaved loss plot to: {loss_plot_path}")

    # ==================== Plot 2: Decoded Images Grid ====================
    n_images = len(signal_levels) + 1  # +1 for clean image
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).flatten() if n_images > 1 else [axes]
    
    # Plot clean image first
    clean_img = clean_decoded[0].permute(1, 2, 0).numpy()
    axes[0].imshow(clean_img)
    axes[0].set_title('Clean (signal=1.0)', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # Plot noisy images
    for idx, (signal_level, decoded, l1, l2) in enumerate(zip(signal_levels, decoded_images, l1_losses, l2_losses)):
        ax = axes[idx + 1]
        img = decoded[0].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f'Signal: {signal_level:.2f}\nL1: {l1:.3f}, L2: {l2:.3f}', fontsize=10)
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Decoded Images at Different Signal Levels', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    images_plot_path = output_dir / "decoded_images_grid.png"
    plt.savefig(images_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved decoded images to: {images_plot_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Signal Level':<15} {'L1 Loss':<12} {'L2 (RMSE)':<12} {'Velocity':<12} {'Reweighted':<12}")
    print("-" * 80)
    for signal_level, l1, l2, vel, rew in zip(signal_levels, l1_losses, l2_losses, velocity_losses, reweighted_losses):
        print(f"{signal_level:<15.2f} {l1:<12.4f} {l2:<12.4f} {vel:<12.4f} {rew:<12.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
