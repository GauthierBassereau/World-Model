import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from src.dataset.lerobot_dataset import LeRobotDataset, LeRobotDatasetConfig
from src.training.logger import WorldModelLogger, LoggingConfig
from src.rae_dino.rae import RAE
from src.dataset.collator import StackCollator

def main():
    output_dir = Path("archive/latent_distr")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    logging_config = LoggingConfig(output_dir=str(output_dir), run_name="latent_distr_viz")
    logger = WorldModelLogger(logging_config, is_main_process=True)

    dataset_config = LeRobotDatasetConfig(
        repo_id="aractingi/droid_1.0.1",
        sequence_length=15,
        fps=3.0,
    )
    
    print("Initializing dataset...")
    dataset = LeRobotDataset(dataset_config, logger)
    print(f"Dataset size: {len(dataset)}")

    autoencoder = RAE()
    autoencoder.to(device)
    autoencoder.eval()
    
    collator = StackCollator(sequence_length_distribution={1: 1.0})
    
    batch_size = 32
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=collator,
        drop_last=True
    )
    
    latents_list = []
    max_batches = 200
    
    print(f"Processing {max_batches} batches of size {batch_size}...")
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=max_batches):
            if i >= max_batches:
                break
            
            frames = batch.sequence_frames.to(device)
            frames = frames.squeeze(1)
            frames = frames.float() / 255.0
            
            latents = autoencoder.encode(frames)
            
            latents_list.append(latents.cpu().numpy())
            
    latents_all = np.concatenate(latents_list, axis=0)
    latents_flat = latents_all.flatten()
    
    print(f"Collected {latents_flat.size} latent values.")
    print(f"Mean: {np.mean(latents_flat):.4f}, Std: {np.std(latents_flat):.4f}")
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(latents_flat, bins=100, density=True, alpha=0.6, color='b', label='Latents')
    
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, 0, 1)
    plt.plot(x, p, 'k', linewidth=2, label='Standard Normal (0, 1)')
    
    mu, std = norm.fit(latents_flat)
    p_fit = norm.pdf(x, mu, std)
    plt.plot(x, p_fit, 'r--', linewidth=2, label=f'Fitted Normal ({mu:.2f}, {std:.2f})')
    
    plt.title("Latent Distribution vs Normal Distribution")
    plt.xlabel("Latent Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = output_dir / "latent_distribution.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
