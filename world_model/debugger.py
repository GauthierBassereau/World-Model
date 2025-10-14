"""
Utilities to visualise transformer masks for debugging.

The visualiser renders the temporal masking strategy used by the world model
transformer so researchers can confirm that only matching patches interact
across time.
"""

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import torch


class TransformerDebugVisualizer:
    def __init__(self, debug_cfg: Any, model_cfg: Any) -> None:
        self.cfg = debug_cfg
        self.model_cfg = model_cfg
        self.enabled = getattr(debug_cfg, "enabled", False)
        self._has_run = False
        if not self.enabled:
            self.output_dir = None
            return

        self.output_dir = Path(getattr(debug_cfg, "output_dir", "debug/transformer"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def maybe_run(
        self,
        *,
        latent_tokens: int,
        tokens_per_frame: int,
        temporal_indices: torch.Tensor,
        temporal_context: int,
        spatial_mask: torch.Tensor,
    ) -> None:
        if not self.enabled or self._has_run:
            return
        self._has_run = True

        temporal_indices = temporal_indices.to(dtype=torch.long, device="cpu")
        time_steps = max(2, int(getattr(self.cfg, "dummy_time_steps", 6)))
        spatial_mask = spatial_mask.to(dtype=torch.float32, device="cpu")

        temporal_mask = torch.tril(torch.ones(time_steps, time_steps, dtype=torch.float32))
        if temporal_context > 0 and temporal_context < time_steps:
            context_band = torch.triu(
                torch.ones(time_steps, time_steps, dtype=torch.float32),
                diagonal=-(temporal_context - 1),
            )
            temporal_mask = temporal_mask * context_band

        participation = torch.zeros(tokens_per_frame, dtype=torch.float32)
        participation[temporal_indices] = 1.0

        category_ids = torch.zeros(tokens_per_frame, dtype=torch.long)
        if latent_tokens > 0:
            category_ids[:latent_tokens] = 0
        if tokens_per_frame > latent_tokens:
            category_ids[latent_tokens] = 1  # action
        if tokens_per_frame > latent_tokens + 1:
            category_ids[latent_tokens + 1] = 2  # noise
        if tokens_per_frame > latent_tokens + 2:
            category_ids[latent_tokens + 2 :] = 3  # registers

        palette = ["#1E88E5", "#F4511E", "#8E24AA", "#43A047"]
        category_names = ["latent", "action", "noise", "register"]
        category_counts = {}
        for idx, name in enumerate(category_names):
            if (category_ids == idx).any():
                category_counts[name] = int((category_ids == idx).sum().item())

        self._plot_temporal_mask(temporal_mask)
        self._plot_spatial_mask(spatial_mask, category_ids, palette, category_names)
        self._plot_token_participation(participation, category_ids, palette)
        self._plot_patch_masks(temporal_mask, latent_tokens)
        self._write_summary(
            temporal_mask=temporal_mask,
            spatial_mask=spatial_mask,
            participation=participation,
            temporal_indices=temporal_indices,
            temporal_context=temporal_context,
            time_steps=time_steps,
            tokens_per_frame=tokens_per_frame,
            latent_tokens=latent_tokens,
            category_counts=category_counts,
        )

    def _plot_temporal_mask(self, mask: torch.Tensor) -> None:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(mask.numpy(), origin="lower", cmap="Blues")
        ax.set_xlabel("Key time index")
        ax.set_ylabel("Query time index")
        ax.set_title("Temporal causal mask")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(self.output_dir / "temporal_mask.png", dpi=200)
        plt.close(fig)

    def _plot_spatial_mask(
        self,
        spatial_mask: torch.Tensor,
        category_ids: torch.Tensor,
        palette: list[str],
        category_names: list[str],
    ) -> None:
        tokens_per_frame = spatial_mask.size(0)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(spatial_mask.numpy(), origin="lower", cmap="Greens")
        ax.set_xlabel("Key token index (same frame)")
        ax.set_ylabel("Query token index (same frame)")
        ax.set_title("Spatial attention mask (single frame)")

        cm = ListedColormap(palette[: int(category_ids.max().item()) + 1])
        top_ax = ax.inset_axes([0.0, 1.02, 1.0, 0.06], transform=ax.transAxes)
        top_ax.imshow(
            category_ids.unsqueeze(0).numpy(),
            aspect="auto",
            cmap=cm,
            vmin=0,
            vmax=int(category_ids.max().item()),
        )
        top_ax.set_axis_off()

        left_ax = ax.inset_axes([-0.08, 0.0, 0.06, 1.0], transform=ax.transAxes)
        left_ax.imshow(
            category_ids.unsqueeze(1).numpy(),
            aspect="auto",
            cmap=cm,
            vmin=0,
            vmax=int(category_ids.max().item()),
        )
        left_ax.set_axis_off()

        handles = []
        for idx, name in enumerate(category_names):
            if (category_ids == idx).any():
                handles.append(mpatches.Patch(color=palette[idx], label=name.capitalize()))
        if handles:
            ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.35, 1.02))

        tick_positions = []
        tick_labels = []
        previous_id = None
        for token_idx in range(tokens_per_frame):
            cat_id = int(category_ids[token_idx].item())
            if cat_id != previous_id:
                tick_positions.append(token_idx)
                tick_labels.append(category_names[cat_id])
                previous_id = cat_id
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(self.output_dir / "spatial_mask.png", dpi=200)
        plt.close(fig)

    def _plot_token_participation(
        self,
        participation: torch.Tensor,
        category_ids: torch.Tensor,
        palette: list[str],
    ) -> None:
        tokens_per_frame = participation.numel()
        indices = list(range(tokens_per_frame))
        colors = []
        for idx in range(tokens_per_frame):
            base_color = palette[int(category_ids[idx].item())]
            colors.append(base_color if participation[idx] > 0 else "#CFD8DC")

        fig, ax = plt.subplots(figsize=(max(6, tokens_per_frame * 0.25), 3.5))
        ax.bar(indices, participation.numpy(), color=colors, edgecolor="#37474F")
        ax.set_ylim(0.0, 1.2)
        ax.set_xlabel("Token index within frame")
        ax.set_ylabel("Participates in temporal attention")
        ax.set_title("Temporal token participation")

        fig.tight_layout()
        fig.savefig(self.output_dir / "token_participation.png", dpi=200)
        plt.close(fig)

    def _plot_patch_masks(self, temporal_mask: torch.Tensor, latent_tokens: int) -> None:
        if latent_tokens <= 0:
            return
        num_plots = min(latent_tokens, int(getattr(self.cfg, "num_patch_plots", 4)))
        if num_plots <= 0:
            return
        rows = math.ceil(num_plots / 2)
        cols = min(2, num_plots)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if hasattr(axes, "flat"):
            axes_iter = list(axes.flat)
        else:
            axes_iter = [axes]

        grid_side = int(math.sqrt(latent_tokens))
        is_square = grid_side * grid_side == latent_tokens

        for plot_idx in range(len(axes_iter)):
            ax = axes_iter[plot_idx]
            if plot_idx >= num_plots:
                ax.axis("off")
                continue
            ax.imshow(temporal_mask.numpy(), origin="lower", cmap="viridis")
            if is_square:
                row_idx = plot_idx // grid_side
                col_idx = plot_idx % grid_side
                ax.set_title(f"Patch ({row_idx}, {col_idx})")
            else:
                ax.set_title(f"Patch index {plot_idx}")
            ax.set_xlabel("Key time index")
            ax.set_ylabel("Query time index")

        fig.tight_layout()
        fig.savefig(self.output_dir / "patch_temporal_masks.png", dpi=200)
        plt.close(fig)

    def _write_summary(
        self,
        *,
        temporal_mask: torch.Tensor,
        spatial_mask: torch.Tensor,
        participation: torch.Tensor,
        temporal_indices: torch.Tensor,
        temporal_context: int,
        time_steps: int,
        tokens_per_frame: int,
        latent_tokens: int,
        category_counts: dict[str, int],
    ) -> None:
        payload = {
            "temporal_context": int(temporal_context),
            "dummy_time_steps": int(time_steps),
            "tokens_per_frame": int(tokens_per_frame),
            "latent_tokens": int(latent_tokens),
            "num_registers": int(getattr(self.model_cfg, "num_registers", 0)),
            "active_token_indices": [int(idx) for idx in temporal_indices.tolist()],
            "temporal_mask_density": float(temporal_mask.mean().item()),
            "spatial_mask_density": float(spatial_mask.mean().item()),
            "active_token_fraction": float(participation.mean().item()),
            "token_category_counts": category_counts,
        }
        target = self.output_dir / "transformer_debug_summary.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
