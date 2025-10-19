"""
Structured logging utilities for the world model trainer.

The ``WorldModelLogger`` centralises local logging, WandB integration, and
auxiliary visualisations so that the trainer can remain focused on the
optimization loop.
"""

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from training.world_trainer import LoggingConfig


def _create_local_logger() -> logging.Logger:
    logger = logging.getLogger("world_model_trainer")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("\x1b[90m%(asctime)s | %(levelname)s | %(message)s\x1b[0m")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class WorldModelLogger:
    def __init__(
        self,
        logging_cfg: "LoggingConfig",
        noise_log_limit: int = 100,
    ) -> None:
        self.cfg = logging_cfg
        self.local = _create_local_logger()
        self.sample_interval = logging_cfg.sample_interval
        self.noise_log_limit = noise_log_limit
        self.noise_log_count = 0
        self.current_step = 0
        self.current_micro_step = 0
        self.wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None
        self._wandb = None

    # ------------------------------------------------------------------ helpers
    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.local.info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.local.warning(message, *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.local.debug(message, *args, **kwargs)

    # ------------------------------------------------------------------ lifecycle
    def init_wandb(self, config: Dict[str, Any]) -> Optional["wandb.sdk.wandb_run.Run"]:
        if self.wandb_run is not None or self._wandb is not None:
            return self.wandb_run
        try:
            import wandb  # type: ignore[attr-defined]
        except ImportError:
            self.info("Weights & Biases not available; skipping logging.")
            self._wandb = None
            self.wandb_run = None
            return None
        self._wandb = wandb
        self.wandb_run = wandb.init(
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.cfg.run_name,
            config=config,
        )
        return self.wandb_run

    def close(self) -> None:
        if self.wandb_run:
            self.wandb_run.finish()
        self.wandb_run = None
        self._wandb = None

    # ------------------------------------------------------------------ state
    def start_step(self, step: int) -> None:
        self.current_step = step
        self.current_micro_step = 0

    def start_micro_step(self, micro_step: int) -> None:
        self.current_micro_step = micro_step

    # ------------------------------------------------------------------ logging hooks
    def log_dataloader(self, dataloader: DataLoader) -> None:
        try:
            steps = len(dataloader)
        except TypeError:
            steps = None

        if steps is None:
            self.info("Dataloader length unavailable (object does not implement __len__).")
        else:
            self.info("Dataloader contains %d steps.", steps)

        if self.wandb_run is not None:
            self.wandb_run.log({"debug/dataloader_length": steps if steps is not None else "unknown"})

    def log_training_metrics(self, metrics: Dict[str, float]) -> None:
        if self.cfg.log_interval and self.current_step % self.cfg.log_interval == 0:
            message = " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
            self.info("step=%d %s", self.current_step, message)
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {f"train/{key}": value for key, value in metrics.items()},
                    step=self.current_step,
                )

    def log_distr_noise(self, tau: torch.Tensor, bins: int = 50) -> None:
        if self.noise_log_count >= self.noise_log_limit:
            return

        values = tau.detach().to(dtype=torch.float32).flatten()
        if values.numel() == 0:
            return

        min_val = float(values.min().item())
        max_val = float(values.max().item())
        mean_val = float(values.mean().item())
        std_val = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
        hist = torch.histc(values, bins=bins, min=min_val, max=max_val) if values.numel() > 1 else values

        self.info(
            "step=%d micro_step=%d | tau mean=%.4f std=%.4f min=%.4f max=%.4f bins=%d",
            self.current_step,
            self.current_micro_step,
            mean_val,
            std_val,
            min_val,
            max_val,
            bins if values.numel() > 1 else 1,
        )
        self.debug(
            "Tau histogram counts: %s",
            hist.tolist() if values.numel() > 1 else [float(values.item())],
        )

        if self.wandb_run is not None and self._wandb is not None:
            self.wandb_run.log(
                {
                    "debug/tau_hist": self._wandb.Histogram(values.cpu().numpy()),
                    "debug/tau_stats": {
                        "mean": mean_val,
                        "std": std_val,
                        "min": min_val,
                        "max": max_val,
                        "step": self.current_step,
                        "micro_step": self.current_micro_step,
                    },
                }
            )

        self.noise_log_count += 1

    def log_sample_sequence(self, frames: torch.Tensor, tau: torch.Tensor) -> None:
        if self.sample_interval is None or self.sample_interval <= 0:
            return
        if self.current_step % self.sample_interval != 0 or self.current_micro_step != 0:
            return

        frames = frames.detach().to(dtype=torch.float32)
        if frames.ndim != 4:
            self.warning(
                "log_sample_sequence expected frames with shape [T, C, H, W], got %s",
                tuple(frames.shape),
            )
            return

        if frames.max() > 1.5:
            frames = frames / 255.0
        frames = frames.clamp(0.0, 1.0)

        self.info(
            "Logging sample sequence at step=%d micro_step=%d (frames=%d).",
            self.current_step,
            self.current_micro_step,
            frames.shape[0],
        )

        if self.wandb_run is None or self._wandb is None:
            return

        video_frames = (frames.mul(255).to(torch.uint8)).permute(0, 2, 3, 1).cpu().numpy()
        video = self._wandb.Video(video_frames, fps=4, format="mp4")

        table = self._wandb.Table(columns=["frame", "tau"])
        tau_values = tau.detach().to(dtype=torch.float32).view(-1).cpu().tolist()
        for idx, value in enumerate(tau_values):
            table.add_data(idx, value)

        self.wandb_run.log(
            {
                "sample/frames": video,
                "sample/tau": table,
            },
            step=self.current_step,
        )
