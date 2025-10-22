import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.dataset import (
    DataloaderConfig,
    DatasetConfig,
    WorldModelBatch,
    build_world_model_dataloader,
)

from .config import RolloutConfig
from .evaluator import WorldModelEvaluator


class DatasetEvaluator:
    """
    Drive batched evaluation of the world model against a dataset.

    The runner reuses the training dataloader utilities to fetch sequences,
    rolls out the model with Dreamer diffusion forcing, computes simple metrics,
    and optionally stores side-by-side videos under ``archive/evaluation_runs``.
    """

    def __init__(
        self,
        evaluator: WorldModelEvaluator,
        rollout_cfg: RolloutConfig,
        dataset_cfg: Optional[DatasetConfig] = None,
        dataloader_cfg: Optional[DataloaderConfig] = None,
        output_root: Optional[Path] = None,
    ) -> None:
        self.evaluator = evaluator
        self.rollout_cfg = rollout_cfg
        self.dataset_cfg = dataset_cfg or evaluator.training_cfg.dataset
        base_loader_cfg = dataloader_cfg or evaluator.training_cfg.dataloader
        self.dataloader_cfg = replace(base_loader_cfg)
        self.dataloader_cfg.shuffle = False
        self.output_root = Path(output_root or "archive/evaluation_runs")
        self.output_root.mkdir(parents=True, exist_ok=True)

    def build_dataloader(self) -> DataLoader:
        return build_world_model_dataloader(
            dataset_cfg=self.dataset_cfg,
            dataloader_cfg=self.dataloader_cfg,
            device=self.evaluator.device,
        )

    def run(
        self,
        *,
        max_batches: Optional[int] = None,
        save_videos: bool = True,
        video_fps: int = 15,
    ) -> Dict[str, float]:
        dataloader = self.build_dataloader()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = self.output_root / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        latent_sq_sum = 0.0
        latent_elem = 0
        pixel_sq_sum = 0.0
        pixel_elem = 0
        processed_sequences = 0

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            metrics = self._process_batch(
                batch=batch,
                batch_idx=batch_idx,
                run_dir=run_dir,
                save_videos=save_videos,
                video_fps=video_fps,
            )

            latent_sq_sum += metrics["latent_sq"]
            latent_elem += metrics["latent_elem"]
            pixel_sq_sum += metrics["pixel_sq"]
            pixel_elem += metrics["pixel_elem"]
            processed_sequences += metrics["sequences"]

        results = {
            "sequences": processed_sequences,
            "latent_mse": latent_sq_sum / latent_elem if latent_elem else 0.0,
            "pixel_mse": pixel_sq_sum / pixel_elem if pixel_elem else 0.0,
        }

        summary_path = run_dir / "metrics.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

        return results

    def _process_batch(
        self,
        batch: WorldModelBatch,
        batch_idx: int,
        run_dir: Path,
        save_videos: bool,
        video_fps: int,
    ) -> Dict[str, float]:
        frames = batch.sequence_frames
        actions = batch.sequence_actions
        actions_mask = batch.actions_mask
        independent_mask = batch.independant_frames_mask

        if frames.ndim != 5:
            raise ValueError("Expected frames tensor with shape [B, T, C, H, W].")

        batch_size, total_steps = frames.shape[0], frames.shape[1]
        context_steps = self.rollout_cfg.context
        horizon = self.rollout_cfg.horizon
        if context_steps + horizon > total_steps:
            raise ValueError(
                f"Batch provides {total_steps} steps but rollout requires "
                f"{context_steps + horizon}."
            )

        context_frames = frames[:, :context_steps, ...]
        target_frames = frames[:, context_steps : context_steps + horizon, ...]

        context_actions = None
        future_actions = None
        context_action_mask = None
        future_action_mask = None

        if actions is not None:
            if actions.shape[:2] < (batch_size, total_steps):
                raise ValueError("Actions tensor shape mismatch.")
            context_actions = actions[:, :context_steps, :].clone()
            future_actions = actions[:, context_steps : context_steps + horizon, :].clone()

        if actions_mask is not None:
            context_action_mask = actions_mask[:, :context_steps].clone()
            future_action_mask = actions_mask[:, context_steps : context_steps + horizon].clone()

        rollout = self.evaluator.rollout(
            context_frames=context_frames,
            context_actions=context_actions,
            context_actions_mask=context_action_mask,
            rollout_actions=future_actions,
            rollout_action_mask=future_action_mask,
            rollout_steps=horizon,
            decode=True,
            independant_frames_mask=independent_mask,
        )

        generated_latents = rollout["generated_latents"]
        target_latents = self.evaluator.encode_frames(target_frames)

        latent_diff = generated_latents - target_latents
        latent_sq = torch.sum(latent_diff.pow(2)).item()
        latent_elem = latent_diff.numel()

        generated_frames = rollout["generated_frames"]
        target_frames_cpu = target_frames.cpu()

        pixel_diff = generated_frames - target_frames_cpu
        pixel_sq = torch.sum(pixel_diff.pow(2)).item()
        pixel_elem = pixel_diff.numel()

        if save_videos:
            self._export_videos(
                context_frames=context_frames.cpu(),
                generated_frames=generated_frames,
                target_frames=target_frames_cpu,
                run_dir=run_dir,
                batch_idx=batch_idx,
                fps=video_fps,
            )

        return {
            "latent_sq": latent_sq,
            "latent_elem": float(latent_elem),
            "pixel_sq": pixel_sq,
            "pixel_elem": float(pixel_elem),
            "sequences": float(batch_size),
        }

    def _export_videos(
        self,
        *,
        context_frames: torch.Tensor,
        generated_frames: torch.Tensor,
        target_frames: torch.Tensor,
        run_dir: Path,
        batch_idx: int,
        fps: int,
    ) -> None:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError(
                "imageio is required to export evaluation videos. Install it or "
                "disable save_videos."
            ) from exc

        batch_size = context_frames.shape[0]
        context_steps = context_frames.shape[1]

        for idx in range(batch_size):
            clip_path = run_dir / f"sample_{batch_idx:04d}_{idx:02d}.mp4"
            with imageio.get_writer(clip_path, mode="I", fps=fps, codec="libx264") as writer:
                context = context_frames[idx]
                predicted = generated_frames[idx]
                target = target_frames[idx]

                for step in range(context_steps + predicted.shape[0]):
                    if step < context_steps:
                        pred_frame = context[step]
                        tgt_frame = context[step]
                        label_color = (0, 0.7, 0)
                    else:
                        pred_frame = predicted[step - context_steps]
                        tgt_frame = target[step - context_steps]
                        label_color = (0.7, 0, 0)

                    panel = self._stack_frames(pred_frame, tgt_frame, label_color)
                    writer.append_data(panel)

    @staticmethod
    def _stack_frames(
        predicted: torch.Tensor,
        target: torch.Tensor,
        label_color: Tuple[float, float, float],
    ) -> np.ndarray:
        if predicted.ndim != 3 or target.ndim != 3:
            raise ValueError("Expected frame tensors with shape [C, H, W].")
        predicted = predicted.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)

        top = predicted
        bottom = target
        stacked = torch.cat((top, bottom), dim=1)
        frame = stacked.permute(1, 2, 0).numpy()
        overlay = np.ones_like(frame) * np.array(label_color, dtype=np.float32)
        frame = 0.85 * frame + 0.15 * overlay
        return (frame * 255.0).astype(np.uint8)
