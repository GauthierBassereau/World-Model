"""
Offline diffusion visualisation utilities.

The helpers here mirror the previous debugging hooks but are designed to be
invoked explicitly from standalone scripts so they no longer couple to the
training loop.
"""

import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from training.diffusion.config import DiffusionConfig, DiffusionDebugConfig


class DiffusionVisualizer:
    """
    Generate plots and lightweight summaries for a batch of diffusion tensors.
    """

    def __init__(
        self,
        debug_cfg: DiffusionDebugConfig,
        schedule_cfg: DiffusionConfig,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = debug_cfg
        self.schedule_cfg = schedule_cfg
        self.enabled = debug_cfg.enabled
        self._has_run = False
        if not self.enabled:
            self.output_dir = None
            return

        base_dir = Path(output_dir) if output_dir is not None else Path(debug_cfg.output_dir)
        self.output_dir = base_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        *,
        latents: torch.Tensor,
        signal_levels: torch.Tensor,
        base_noise: torch.Tensor,
        noisy_latents: torch.Tensor,
        target_velocity: torch.Tensor,
    ) -> None:
        if not self.enabled or self._has_run:
            return
        self._has_run = True
        assert self.output_dir is not None

        summary = self._build_summary(
            latents=latents,
            signal_levels=signal_levels,
            base_noise=base_noise,
            noisy_latents=noisy_latents,
            target_velocity=target_velocity,
        )
        self._write_summary(summary)
        self._plot_signal_histogram(signal_levels)
        self._plot_signal_heatmap(signal_levels)
        self._plot_schedule_curve(latents)
        self._plot_component_histograms(
            tensors=(base_noise, noisy_latents, target_velocity),
            labels=("base noise", "noisy latents", "target velocity"),
        )

    def _write_summary(self, payload: Dict[str, object]) -> None:
        target = self.output_dir / "diffusion_summary.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _tensor_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        flat = tensor.detach().flatten()
        count = int(flat.numel())
        if count == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        flat = flat.to(dtype=torch.float32)
        mean = float(flat.mean().item())
        std = float(flat.std(unbiased=False).item())
        min_val = float(flat.min().item())
        max_val = float(flat.max().item())
        return {
            "count": count,
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
        }

    def _build_summary(
        self,
        *,
        latents: torch.Tensor,
        signal_levels: torch.Tensor,
        base_noise: torch.Tensor,
        noisy_latents: torch.Tensor,
        target_velocity: torch.Tensor,
    ) -> Dict[str, object]:
        batch, steps, tokens, dim = latents.shape
        effective_dim = tokens * dim
        alpha = math.sqrt(effective_dim / float(self.schedule_cfg.base_dimension))

        return {
            "batch_shape": {
                "batch": batch,
                "steps": steps,
                "tokens": tokens,
                "dim": dim,
            },
            "schedule": {
                "min_signal": float(self.schedule_cfg.min_signal),
                "max_signal": float(self.schedule_cfg.max_signal),
                "base_dimension": int(self.schedule_cfg.base_dimension),
                "effective_dimension": int(effective_dim),
                "dimension_shift_alpha": float(alpha),
            },
            "components": {
                "signal_levels": self._tensor_stats(signal_levels),
                "base_noise": self._tensor_stats(base_noise),
                "noisy_latents": self._tensor_stats(noisy_latents),
                "target_velocity": self._tensor_stats(target_velocity),
            },
            "config": {
                "num_hist_bins": int(self.cfg.num_hist_bins),
                "schedule_plot_points": int(self.cfg.schedule_plot_points),
                "max_time_series": int(self.cfg.max_time_series),
            },
        }

    def _plot_signal_histogram(self, signal_levels: torch.Tensor) -> None:
        values = signal_levels.detach().flatten().to(dtype=torch.float32)
        if values.numel() == 0:
            return

        values = values.cpu()
        min_signal = float(self.schedule_cfg.min_signal)
        max_signal = float(self.schedule_cfg.max_signal)
        hist = torch.histc(values, bins=self.cfg.num_hist_bins, min=min_signal, max=max_signal)
        edges = torch.linspace(min_signal, max_signal, steps=self.cfg.num_hist_bins + 1)

        fig, ax = plt.subplots(figsize=(6, 4))
        width = edges[1] - edges[0]
        ax.bar(
            edges[:-1].numpy(),
            hist.numpy(),
            width=float(width),
            align="edge",
            color="#3D5AFE",
            edgecolor="#1A237E",
            alpha=0.85,
        )
        ax.set_xlabel("Signal level τ")
        ax.set_ylabel("Count")
        ax.set_title("Signal distribution")
        ax.set_xlim(min_signal, max_signal)
        fig.tight_layout()
        fig.savefig(self.output_dir / "signal_histogram.png", dpi=200)
        plt.close(fig)

    def _plot_signal_heatmap(self, signal_levels: torch.Tensor) -> None:
        matrix = signal_levels.detach().to(dtype=torch.float32).cpu()
        if matrix.ndim != 2:
            matrix = matrix.view(matrix.shape[0], -1)
        matrix = matrix[: self.cfg.max_time_series]
        if matrix.numel() == 0:
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(
            matrix.numpy(),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="magma",
            vmin=float(self.schedule_cfg.min_signal),
            vmax=float(self.schedule_cfg.max_signal),
        )
        ax.set_xlabel("Sequence step")
        ax.set_ylabel("Batch index")
        ax.set_title("Signal levels per sequence")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="τ")
        fig.tight_layout()
        fig.savefig(self.output_dir / "signal_heatmap.png", dpi=200)
        plt.close(fig)

    def _plot_schedule_curve(self, latents: torch.Tensor) -> None:
        batch, steps, tokens, dim = latents.shape
        effective_dim = tokens * dim
        alpha = math.sqrt(effective_dim / float(self.schedule_cfg.base_dimension))

        base = torch.linspace(0.0, 1.0, steps=self.cfg.schedule_plot_points, dtype=torch.float32)
        shifted = base.clone()
        if not math.isclose(alpha, 1.0):
            shifted = (alpha * shifted) / (1.0 + (alpha - 1.0) * shifted)
        mapped = shifted * (self.schedule_cfg.max_signal - self.schedule_cfg.min_signal) + self.schedule_cfg.min_signal

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(base.numpy(), mapped.numpy(), color="#FF7043", linewidth=2.2)
        ax.set_xlabel("Base uniform sample u")
        ax.set_ylabel("Signal level τ")
        ax.set_title("Dimension-shifted schedule")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        fig.savefig(self.output_dir / "schedule_curve.png", dpi=200)
        plt.close(fig)

    def _plot_component_histograms(
        self,
        *,
        tensors: Sequence[torch.Tensor],
        labels: Sequence[str],
    ) -> None:
        fig, axes = plt.subplots(1, len(tensors), figsize=(5 * len(tensors), 4))
        if len(tensors) == 1:
            axes = [axes]

        for ax, tensor, label in zip(axes, tensors, labels):
            values = tensor.detach().to(dtype=torch.float32).cpu().flatten()
            if values.numel() == 0:
                ax.set_visible(False)
                continue
            min_val = float(values.min().item())
            max_val = float(values.max().item())
            if math.isclose(min_val, max_val):
                max_val = min_val + 1e-6
            hist = torch.histc(values, bins=self.cfg.num_hist_bins, min=min_val, max=max_val)
            edges = torch.linspace(min_val, max_val, steps=self.cfg.num_hist_bins + 1)
            width = edges[1] - edges[0]
            ax.bar(
                edges[:-1].numpy(),
                hist.numpy(),
                width=float(width),
                align="edge",
                color="#26A69A",
                edgecolor="#004D40",
                alpha=0.8,
            )
            ax.set_title(f"{label.title()} distribution")
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(self.output_dir / "component_histograms.png", dpi=200)
        plt.close(fig)
