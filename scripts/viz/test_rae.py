"""Quick sanity check: encode→decode with the configured RAE and measure PSNR/SSIM."""
import torch
import math
from torchvision.io import read_image
from torchvision.transforms.functional import resize, center_crop
from src.rae_dino import AutoencoderConfig, build_autoencoder
from pathlib import Path

@torch.no_grad()
def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    rae = build_autoencoder(AutoencoderConfig()).to(device).eval()

    # Grab a few frames from the lerobot cache or provide your own paths
    # Option 1: from HF cache (adjust if needed)
    import random
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("Gaugou/ur5_WM_eval")
    
    n_samples = min(16, len(ds))
    indices = sorted(random.sample(range(len(ds)), n_samples))
    psnrs, ssims = [], []

    for i in indices:
        item = ds[i]
        img = item["observation.images.front"]  # [C, H, W] float [0,1] or uint8
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        resolution = getattr(rae, "resolution", getattr(rae, "encoder_input_size", 256))
        img = resize(img, resolution, antialias=True)
        img = center_crop(img, resolution)
        img = img.unsqueeze(0).to(device)

        z = rae.encode(img)
        rec = rae.decode(z).clamp(0, 1)
        rec = torch.nn.functional.interpolate(rec, size=img.shape[-2:], mode='bicubic', align_corners=False).clamp(0, 1)

        # PSNR
        mse = (img - rec).pow(2).mean().item()
        psnr = -10 * math.log10(mse + 1e-10)
        psnrs.append(psnr)

        # SSIM (simplified per-image)
        ssim = _ssim(img.squeeze(0), rec.squeeze(0))
        ssims.append(ssim)

        print(f"  sample {i:3d}  PSNR={psnr:.2f} dB  SSIM={ssim:.4f}")

    print(f"\n{'='*50}")
    print(f"  Mean PSNR: {sum(psnrs)/len(psnrs):.2f} dB")
    print(f"  Mean SSIM: {sum(ssims)/len(ssims):.4f}")
    print(f"{'='*50}")
    print(f"  RAE paper reference: ~24-26 dB PSNR on ImageNet 256x256")
    print(f"  If your numbers are far below that, something is off.")

def _ssim(img1, img2, window_size=11):
    """Simple SSIM between two [C,H,W] tensors in [0,1]."""
    C1, C2 = 0.01**2, 0.03**2
    # Use average pooling as a simple windowed mean
    pad = window_size // 2
    k = torch.ones(1, 1, window_size, window_size, device=img1.device) / (window_size**2)
    
    vals = []
    for c in range(img1.shape[0]):
        x = img1[c:c+1].unsqueeze(0)  # [1,1,H,W]
        y = img2[c:c+1].unsqueeze(0)
        mu_x = torch.nn.functional.conv2d(x, k, padding=pad)
        mu_y = torch.nn.functional.conv2d(y, k, padding=pad)
        sigma_x2 = torch.nn.functional.conv2d(x*x, k, padding=pad) - mu_x**2
        sigma_y2 = torch.nn.functional.conv2d(y*y, k, padding=pad) - mu_y**2
        sigma_xy = torch.nn.functional.conv2d(x*y, k, padding=pad) - mu_x*mu_y
        ssim_map = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(sigma_x2 + sigma_y2 + C2))
        vals.append(ssim_map.mean().item())
    return sum(vals) / len(vals)

if __name__ == "__main__":
    main()
