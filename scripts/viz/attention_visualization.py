"""
Attention Visualization for World Model
========================================
Produces publication-quality figures:
  1. Spatial self-attention patterns across selected layers
  2. Temporal self-attention patterns across all temporal layers
  3. (Bonus) Spatial attention from selected query patches onto the 2D patch grid

Usage:
    python scripts/viz/attention_visualization.py \
        --checkpoint checkpoints/pretraining/world_model_step_009000.pt \
        --config configs/pretraining.yaml \
        --output_dir assets/attention_viz \
        --num_frames 10 \
        --device cuda
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn as nn
import yaml

from src.world_model.backbone import WorldModelBackbone, WorldModelConfig
from src.world_model.components import Attention, _apply_rope, _rope_cache


# ──────────────────────────────────────────────────────────────────────────────
# Style constants
# ──────────────────────────────────────────────────────────────────────────────

FONT_FAMILY = "serif"
TITLE_SIZE = 16
SUBTITLE_SIZE = 13
LABEL_SIZE = 11
TICK_SIZE = 9
CBAR_SIZE = 9

SPATIAL_CMAP = LinearSegmentedColormap.from_list(
    "spatial_attn",
    ["#0d0d1a", "#1b1b4d", "#3a2d8e", "#6b3fa0", "#a94dbd",
     "#e066a0", "#f5a673", "#fef0d5"],
    N=256,
)
TEMPORAL_CMAP = LinearSegmentedColormap.from_list(
    "temporal_attn",
    ["#0a0a23", "#0f2b5b", "#1a5276", "#1e8449", "#52be80",
     "#abebc6", "#f9e79f", "#fdebd0"],
    N=256,
)


def set_plot_style():
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": LABEL_SIZE,
        "axes.titlesize": SUBTITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Attention capture via monkey-patching
# ──────────────────────────────────────────────────────────────────────────────

_captured_attentions: Dict[int, torch.Tensor] = {}


def _make_capturing_forward(layer_idx: int):
    """Return a patched forward that stores attention weights after softmax."""

    def capturing_forward(self, x, rope=None, mask=None, kv_cache=None):
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            cos, sin = rope
            if cos.ndim == 2:
                cos = cos[-L:, :].view(1, 1, L, -1)
                sin = sin[-L:, :].view(1, 1, L, -1)
            q, k = _apply_rope(q, k, cos, sin)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # Always use the manual path so we can capture weights
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.attn_logit_softcapping is not None:
            attn = (
                torch.tanh(attn / self.attn_logit_softcapping)
                * self.attn_logit_softcapping
            )
        if mask is not None:
            attn = attn + mask
        attn_weights = attn.softmax(dim=-1)

        _captured_attentions[layer_idx] = attn_weights.detach().cpu().float()

        x = attn_weights @ v
        x = x.transpose(1, 2).contiguous().reshape(B, L, -1)
        return self.out_proj(x), (k, v)

    return capturing_forward


def patch_model_for_capture(model: WorldModelBackbone) -> Dict:
    originals = {}
    for i, block in enumerate(model.layers):
        attn = block.attn
        originals[i] = attn.forward
        attn.forward = _make_capturing_forward(i).__get__(attn, Attention)
    return originals


def restore_model(model: WorldModelBackbone, originals: Dict):
    for i, block in enumerate(model.layers):
        block.attn.forward = originals[i]


# ──────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> WorldModelConfig:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    wm = raw["world_model"]
    return WorldModelConfig(
        **{k: v for k, v in wm.items() if k in WorldModelConfig.__dataclass_fields__}
    )


def load_model(
    config: WorldModelConfig, checkpoint_path: str, device: torch.device
) -> WorldModelBackbone:
    model = WorldModelBackbone(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt

    # Strip DDP 'module.' prefix if present
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)

    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Loaded model ({n_params:,} params) on {device}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Forward pass
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_forward_and_capture(
    model: WorldModelBackbone,
    num_frames: int,
    num_patches: int,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    global _captured_attentions
    _captured_attentions = {}

    B, T, S, D = 1, num_frames, num_patches, model.config.input_dim
    noisy_latents = torch.randn(B, T, S, D, device=device)
    signal_levels = torch.full((B, T), 0.5, device=device)

    originals = patch_model_for_capture(model)
    try:
        model(noisy_latents=noisy_latents, signal_levels=signal_levels)
    finally:
        restore_model(model, originals)

    return dict(_captured_attentions)


# ──────────────────────────────────────────────────────────────────────────────
# Layer classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_layers(config: WorldModelConfig) -> Tuple[List[int], List[int]]:
    spatial, temporal = [], []
    for i in range(config.depth):
        is_temporal = (
            (i % config.temporal_attention_interval == 0)
            and (i != 0)
            and (i != config.depth - 1)
        )
        (temporal if is_temporal else spatial).append(i)
    return spatial, temporal


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — Spatial Attention
# ──────────────────────────────────────────────────────────────────────────────

def plot_spatial_attention(
    captured: Dict[int, torch.Tensor],
    spatial_layers: List[int],
    config: WorldModelConfig,
    num_patches: int,
    output_path: Path,
):
    # Pick ≤6 evenly-spaced spatial layers
    if len(spatial_layers) > 6:
        idx = np.linspace(0, len(spatial_layers) - 1, 6, dtype=int)
        selected = [spatial_layers[i] for i in idx]
    else:
        selected = spatial_layers

    n = len(selected)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    num_prefix = config.num_registers
    if config.use_signal_token:
        num_prefix += 1
    if config.use_action_token:
        num_prefix += 1
    S_total = num_prefix + num_patches

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.6 * ncols, 5.0 * nrows),
        constrained_layout=True,
    )
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    fig.suptitle(
        "Spatial Self-Attention",
        fontsize=TITLE_SIZE, fontweight="bold", y=1.02,
    )

    for k, layer_i in enumerate(selected):
        r, c = divmod(k, ncols)
        ax = axes[r, c]

        # [B*T, H, S_total, S_total] → mean → [S_total, S_total]
        attn_avg = captured[layer_i].mean(dim=(0, 1)).numpy()

        im = ax.imshow(
            attn_avg, cmap=SPATIAL_CMAP, aspect="equal",
            interpolation="nearest", vmin=0,
        )
        ax.set_title(f"Layer {layer_i}", fontweight="semibold")

        # Dashed lines separating prefix from patches
        ax.axhline(num_prefix - 0.5, color="w", lw=0.8, ls="--", alpha=0.7)
        ax.axvline(num_prefix - 0.5, color="w", lw=0.8, ls="--", alpha=0.7)

        ax.set_xlabel("Key token")
        ax.set_ylabel("Query token")

        ticks = [num_prefix // 2, num_prefix + num_patches // 2]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["prefix", "patches"])
        ax.set_yticks(ticks)
        ax.set_yticklabels(["prefix", "patches"])

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
        cbar.ax.tick_params(labelsize=CBAR_SIZE)

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].set_visible(False)

    prefix_parts = []
    if config.use_signal_token:
        prefix_parts.append("σ = signal")
    if config.use_action_token:
        prefix_parts.append("a = action")
    prefix_parts.append(f"r = register (×{config.num_registers})")
    prefix_desc = ", ".join(prefix_parts)
    fig.text(
        0.5, -0.01,
        f"Dashed lines separate prefix tokens from patch tokens.  "
        f"Prefix: {prefix_desc}",
        ha="center", fontsize=TICK_SIZE, fontstyle="italic", color="#555",
    )

    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Saved spatial attention → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Temporal Attention
# ──────────────────────────────────────────────────────────────────────────────

def plot_temporal_attention(
    captured: Dict[int, torch.Tensor],
    temporal_layers: List[int],
    config: WorldModelConfig,
    num_frames: int,
    output_path: Path,
):
    n = len(temporal_layers)
    if n == 0:
        print("⚠ No temporal layers — skipping temporal figure.")
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.0 * ncols, 4.6 * nrows),
        constrained_layout=True,
    )
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    fig.suptitle(
        "Temporal Self-Attention (Causal)",
        fontsize=TITLE_SIZE, fontweight="bold", y=1.02,
    )

    for k, layer_i in enumerate(temporal_layers):
        r, c = divmod(k, ncols)
        ax = axes[r, c]

        # [B*S_total, H, T, T] → mean → [T, T]
        attn_avg = captured[layer_i].mean(dim=(0, 1)).numpy()

        im = ax.imshow(
            attn_avg, cmap=TEMPORAL_CMAP, aspect="equal",
            interpolation="nearest", vmin=0,
        )
        ax.set_title(f"Layer {layer_i}", fontweight="semibold")
        ax.set_xlabel("Key (frame index)")
        ax.set_ylabel("Query (frame index)")

        T = attn_avg.shape[0]
        if T <= 20:
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
        else:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
        cbar.ax.tick_params(labelsize=CBAR_SIZE)

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].set_visible(False)

    fig.text(
        0.5, -0.01,
        f"Causal mask with temporal context window = "
        f"{config.temporal_context_length}.  "
        f"Averaged over all heads & spatial positions.",
        ha="center", fontsize=TICK_SIZE, fontstyle="italic", color="#555",
    )

    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Saved temporal attention → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Bonus — Spatial attention projected onto the 2-D patch grid
# ──────────────────────────────────────────────────────────────────────────────

def plot_spatial_attention_grid(
    captured: Dict[int, torch.Tensor],
    spatial_layers: List[int],
    config: WorldModelConfig,
    num_patches: int,
    output_path: Path,
    query_positions: Optional[List[int]] = None,
):
    grid_side = int(np.sqrt(num_patches))
    assert grid_side ** 2 == num_patches, "num_patches must be a perfect square"

    num_prefix = config.num_registers
    if config.use_signal_token:
        num_prefix += 1
    if config.use_action_token:
        num_prefix += 1

    # Use a mid-depth spatial layer
    layer_i = spatial_layers[len(spatial_layers) // 2]
    attn_avg = captured[layer_i].mean(dim=(0, 1)).numpy()  # [S_total, S_total]

    if query_positions is None:
        mid = grid_side // 2
        query_positions = [
            0,
            grid_side - 1,
            mid * grid_side + mid,
            (grid_side - 1) * grid_side,
            num_patches - 1,
        ]

    n = len(query_positions)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), constrained_layout=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"Spatial Attention from Query Patches  —  Layer {layer_i}",
        fontsize=TITLE_SIZE, fontweight="bold", y=1.06,
    )

    for ax, qp in zip(axes, query_positions):
        q_idx = num_prefix + qp
        attn_row = attn_avg[q_idx, num_prefix:]  # [num_patches]
        attn_grid = attn_row.reshape(grid_side, grid_side)

        im = ax.imshow(attn_grid, cmap=SPATIAL_CMAP, interpolation="bilinear", vmin=0)

        qr, qc = divmod(qp, grid_side)
        ax.plot(qc, qr, marker="x", color="white", markersize=10, markeredgewidth=2.5)

        ax.set_title(f"Query ({qr},{qc})", fontweight="semibold")
        ax.set_xticks([])
        ax.set_yticks([])

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
        cbar.ax.tick_params(labelsize=CBAR_SIZE)

    fig.text(
        0.5, -0.02,
        f"Attention projected onto the {grid_side}×{grid_side} patch grid.  "
        f"✕ marks the query.",
        ha="center", fontsize=TICK_SIZE, fontstyle="italic", color="#555",
    )

    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Saved spatial attention grid → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize spatial & temporal self-attention in the World Model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/pretraining.yaml",
        help="YAML config file",
    )
    parser.add_argument(
        "--output_dir", type=str, default="assets/attention_viz",
        help="Directory for saved figures",
    )
    parser.add_argument(
        "--num_frames", type=int, default=10,
        help="Number of time steps in the dummy sequence",
    )
    parser.add_argument(
        "--num_patches", type=int, default=256,
        help="Number of spatial patches (e.g. 16×16 = 256 for DINOv2-base @ 224px)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda / cpu)",
    )
    args = parser.parse_args()

    set_plot_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    # ── Load ─────────────────────────────────────────────────────────────
    config = load_config(args.config)
    model = load_model(config, args.checkpoint, device)

    spatial_layers, temporal_layers = classify_layers(config)
    print(f"  Spatial layers  ({len(spatial_layers)}): {spatial_layers}")
    print(f"  Temporal layers ({len(temporal_layers)}): {temporal_layers}")

    # ── Forward pass ─────────────────────────────────────────────────────
    print(f"\nRunning forward pass  (T={args.num_frames}, S={args.num_patches}) …")
    captured = run_forward_and_capture(
        model, args.num_frames, args.num_patches, device,
    )
    print(f"  Captured attention from {len(captured)} layers.\n")

    # ── Figures ──────────────────────────────────────────────────────────
    plot_spatial_attention(
        captured, spatial_layers, config,
        args.num_patches,
        output_dir / "spatial_attention.png",
    )

    plot_temporal_attention(
        captured, temporal_layers, config,
        args.num_frames,
        output_dir / "temporal_attention.png",
    )

    plot_spatial_attention_grid(
        captured, spatial_layers, config,
        args.num_patches,
        output_dir / "spatial_attention_grid.png",
    )

    print(f"\n{'='*60}")
    print(f"  All figures saved to {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
