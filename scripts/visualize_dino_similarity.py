"""
Interactive visualization tool for exploring DINO patch similarities.

Usage:
    python scripts/visualize_dino_similarity.py path/to/image.jpg
"""

import argparse
from typing import Optional
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from vision.dino_v3 import DinoV3Embedder, DinoVisionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize cosine similarity between DINO patch tokens.")
    parser.add_argument("image_path", type=Path, help="Path to the RGB image to visualize.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override (e.g. cuda, cuda:0, cpu). Defaults to CPU if unavailable.",
    )
    parser.add_argument(
        "--crop-square",
        action="store_true",
        help="Center crop the image to a square before encoding.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Optional square size to center-crop (and resize) the image before encoding.",
    )
    return parser.parse_args()


def load_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    min_side = min(width, height)
    left = (width - min_side) // 2
    top = (height - min_side) // 2
    right = left + min_side
    bottom = top + min_side
    return image.crop((left, top, right, bottom))


def crop_to_size(image: Image.Image, size: int) -> Image.Image:
    if size <= 0:
        raise ValueError("input_size must be positive.")
    return ImageOps.fit(image, (size, size), method=Image.BILINEAR, centering=(0.5, 0.5))


def build_embedder(args: argparse.Namespace) -> DinoV3Embedder:
    vision_cfg = DinoVisionConfig()
    embedder = DinoV3Embedder(vision_cfg)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    embedder.to(device)
    return embedder


def compute_tokens(
    embedder: DinoV3Embedder,
    image: Image.Image,
    crop_square: bool,
    target_size: Optional[int],
) -> tuple[torch.Tensor, tuple[int, int], Image.Image]:
    if crop_square:
        image = center_crop_square(image)
    if target_size is not None:
        image = crop_to_size(image, target_size)
    image_array = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        tokens = embedder(image_array)

    patch_size = embedder.model.config.patch_size
    if isinstance(patch_size, (tuple, list)):
        patch_h = int(patch_size[0])
        patch_w = int(patch_size[1] if len(patch_size) > 1 else patch_size[0])
    else:
        patch_h = patch_w = int(patch_size)

    img_h, img_w = image_array.shape[-2], image_array.shape[-1]
    if img_h % patch_h != 0 or img_w % patch_w != 0:
        raise ValueError(
            f"Input dimensions ({img_h}, {img_w}) must be divisible by the patch size "
            f"({patch_h}, {patch_w})."
        )

    grid_h = img_h // patch_h
    grid_w = img_w // patch_w
    tokens = tokens.squeeze(0).to(torch.float32)

    patch_tokens = tokens

    expected = grid_h * grid_w
    if patch_tokens.shape[0] > expected:
        patch_tokens = patch_tokens[-expected:]
    elif patch_tokens.shape[0] < expected:
        raise ValueError(
            f"Expected {expected} patch tokens for grid {grid_h}x{grid_w}, got {patch_tokens.shape[0]}."
        )

    return patch_tokens, (grid_h, grid_w), image


def main() -> None:
    args = parse_args()
    image = load_image(args.image_path)
    embedder = build_embedder(args)
    tokens, (patch_h, patch_w), resized_image = compute_tokens(
        embedder,
        image,
        crop_square=args.crop_square,
        target_size=args.input_size,
    )

    normalized_tokens = F.normalize(tokens, dim=-1)

    width, height = resized_image.size

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(resized_image)
    ax.set_title("Click a patch to view cosine similarities")
    ax.axis("off")

    heatmap = ax.imshow(
        np.zeros((patch_h, patch_w)),
        extent=(0, width, height, 0),
        cmap="magma",
        alpha=0.6,
        vmin=-1.0,
        vmax=1.0,
        origin="upper",
    )
    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")

    state = {
        "tokens": normalized_tokens,
        "patch_h": patch_h,
        "patch_w": patch_w,
        "width": width,
        "height": height,
        "heatmap": heatmap,
        "fig": fig,
    }

    def onclick(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        col = int(np.clip(event.xdata / state["width"] * state["patch_w"], 0, state["patch_w"] - 1))
        row = int(np.clip(event.ydata / state["height"] * state["patch_h"], 0, state["patch_h"] - 1))
        index = row * state["patch_w"] + col

        selected = state["tokens"][index]
        similarities = torch.matmul(state["tokens"], selected.unsqueeze(-1)).squeeze(-1)
        sim_grid = similarities.view(state["patch_h"], state["patch_w"]).cpu().numpy()

        state["heatmap"].set_data(sim_grid)
        state["heatmap"].set_clim(vmin=sim_grid.min(), vmax=sim_grid.max())
        ax.set_title(f"Patch ({row}, {col}) similarity")
        state["fig"].canvas.draw_idle()
        print(f"Clicked patch ({row}, {col}) -> similarity range [{sim_grid.min():.4f}, {sim_grid.max():.4f}]")

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()


if __name__ == "__main__":
    main()
