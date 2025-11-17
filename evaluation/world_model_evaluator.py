from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from training.dataset import (
    DataloaderConfig,
    DatasetConfig,
    WorldModelBatch,
    build_world_model_dataloader,
)
from world_model.flow_matching import (
    DiffusionConfig,
    EulerSolver,
    EulerSolverConfig,
    sample_base_noise,
)


@dataclass
class EvaluationConfig:
    max_batches: Optional[int] = 2
    num_video_samples: int = 1
    batch_size: Optional[int] = None
    rollout_start_frame: int = 6
    rollout_signal_level: float = 0.9
    clean_signal_level: float = 1.0


@dataclass
class EvaluationSummary:
    metrics: Dict[str, float]
    videos: Dict[str, torch.Tensor]


class SequenceErrorStats:
    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.total_sum = 0.0
        self.total_sq_sum = 0.0
        self.total_count = 0.0
        self.step_sum = torch.zeros(max_steps, dtype=torch.float64)
        self.step_sq_sum = torch.zeros(max_steps, dtype=torch.float64)
        self.step_count = torch.zeros(max_steps, dtype=torch.float64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target must share the same shape for MAE stats.")
        if prediction.ndim != 4:
            raise ValueError("Expected tensors of shape [B, T, tokens, dim] for MAE stats.")
        diff = torch.abs(prediction - target)
        per_step_sum = diff.sum(dim=(0, 2, 3)).detach().double().cpu()
        per_step_sq_sum = diff.pow(2).sum(dim=(0, 2, 3)).detach().double().cpu()
        elements_per_step = diff.shape[0] * diff.shape[2] * diff.shape[3]
        length = min(per_step_sum.shape[0], self.max_steps)
        if length <= 0:
            return
        self.step_sum[:length] += per_step_sum[:length]
        self.step_sq_sum[:length] += per_step_sq_sum[:length]
        self.step_count[:length] += elements_per_step
        self.total_sum += float(per_step_sum[:length].sum().item())
        self.total_sq_sum += float(per_step_sq_sum[:length].sum().item())
        self.total_count += float(elements_per_step * length)

    def as_dict(self, prefix: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.total_count > 0:
            mean = self.total_sum / self.total_count
            mean_sq = self.total_sq_sum / self.total_count
            var = max(mean_sq - mean * mean, 0.0)
            metrics[f"{prefix}/mean"] = mean
            metrics[f"{prefix}/var"] = var
        valid = self.step_count > 0
        means = torch.zeros_like(self.step_sum)
        vars = torch.zeros_like(self.step_sq_sum)
        means[valid] = self.step_sum[valid] / self.step_count[valid]
        vars[valid] = self.step_sq_sum[valid] / self.step_count[valid]
        vars = torch.clamp(vars - means.pow(2), min=0.0)
        for idx in range(self.max_steps):
            if self.step_count[idx] > 0:
                metrics[f"{prefix}/mean_t+{idx}"] = float(means[idx].item())
                metrics[f"{prefix}/var_t+{idx}"] = float(vars[idx].item())
        return metrics


class WorldModelEvaluator:
    def __init__(
        self,
        config: EvaluationConfig,
        dataset_cfg: DatasetConfig,
        dataloader_cfg: DataloaderConfig,
        diffusion_cfg: DiffusionConfig,
        autoencoder: nn.Module,
        device: torch.device,
        *,
        seed: int = 0,
        solver_cfg: Optional[EulerSolverConfig] = None,
    ) -> None:
        self.cfg = config
        if self.cfg.max_batches is not None and self.cfg.max_batches < 1:
            self.cfg.max_batches = 1
        if self.cfg.rollout_start_frame < 1:
            raise ValueError("evaluation.rollout_start_frame must be >= 1.")
        if not (0.0 < self.cfg.rollout_signal_level <= 1.0):
            raise ValueError("evaluation.rollout_signal_level must be in (0, 1].")
        if not (0.0 < self.cfg.clean_signal_level <= 1.0):
            raise ValueError("evaluation.clean_signal_level must be in (0, 1].")
        if self.cfg.clean_signal_level < self.cfg.rollout_signal_level:
            raise ValueError("clean_signal_level should be >= rollout_signal_level.")
        self.device = device
        self.autoencoder = autoencoder
        self.autoencoder.eval()
        self.flow_cfg = diffusion_cfg
        self.solver = EulerSolver(solver_cfg or EulerSolverConfig())
        if dataset_cfg.sequence_length_eval is not None:
            self.max_sequence_length = int(dataset_cfg.sequence_length_eval)
        else:
            self.max_sequence_length = max(
                int(key) for key in dataset_cfg.sequence_length_distribution.keys()
            )
        if self.max_sequence_length < 2:
            raise ValueError("Evaluation sequence length must be >= 2.")
        base_context = min(self.cfg.rollout_start_frame, self.max_sequence_length - 1)
        self.context_limit = max(1, base_context)
        self.max_future_steps = max(1, self.max_sequence_length - self.context_limit)
        loader_cfg = dataloader_cfg
        if loader_cfg.shuffle:
            loader_cfg = replace(loader_cfg, shuffle=False)
        if self.cfg.batch_size is not None and self.cfg.batch_size > 0:
            loader_cfg = replace(loader_cfg, batch_size=self.cfg.batch_size)
        try:
            self.dataloader = build_world_model_dataloader(
                dataset_cfg=dataset_cfg,
                dataloader_cfg=loader_cfg,
                grad_accum_steps=1,
                device=device,
                rank=None,
                world_size=1,
                seed=seed,
                split="eval",
            )
            print(f"Length of evaluation dataloader: {len(self.dataloader)}")
        except ValueError:
            self.dataloader = None
        self.scenarios: List[Tuple[str, bool]] = [("actions", True), ("no_actions", False)]

    def evaluate(self, model: nn.Module) -> Optional[EvaluationSummary]:
        if self.dataloader is None:
            return None

        stats: Dict[str, SequenceErrorStats] = {
            scenario: SequenceErrorStats(self.max_future_steps) for scenario, _ in self.scenarios
        }

        video_samples: List[Dict[str, torch.Tensor]] = []
        was_training = model.training
        model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                batch_stats, sample_payloads = self._evaluate_batch(model, batch)
                for scenario, value in batch_stats.items():
                    stats[scenario].update(value[0], value[1])
                if sample_payloads and self.cfg.num_video_samples > 0:
                    remaining = self.cfg.num_video_samples - len(video_samples)
                    if remaining > 0:
                        video_samples.extend(sample_payloads[:remaining])
                if self.cfg.max_batches is not None and (batch_idx + 1) >= self.cfg.max_batches:
                    break

        if was_training:
            model.train()

        metrics: Dict[str, float] = {}
        for scenario, _ in self.scenarios:
            metrics.update(stats[scenario].as_dict(f"eval/{scenario}"))

        if self.cfg.num_video_samples > 0:
            video_samples = video_samples[: self.cfg.num_video_samples]
        else:
            video_samples = []
        videos: Dict[str, torch.Tensor] = {}
        for sample_idx, payload in enumerate(video_samples):
            for key, frames in payload.items():
                videos[f"{key}_{sample_idx}"] = frames

        return EvaluationSummary(metrics=metrics, videos=videos)

    def _evaluate_batch(
        self,
        model: nn.Module,
        batch: WorldModelBatch,
    ) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        frames = batch.sequence_frames.to(self.device, non_blocking=True)
        actions = batch.sequence_actions.to(self.device, non_blocking=True)
        independant_mask = None
        actions_mask = None

        latents = self._encode_frames(frames)
        batch_size, seq_len = latents.shape[0], latents.shape[1]
        context_len = min(self.cfg.rollout_start_frame, max(seq_len - 1, 1))
        future_len = max(seq_len - context_len, 0)

        observed_latents = latents[:, :context_len, ...].detach()
        target_future = latents[:, context_len:, ...].detach() if future_len > 0 else None

        batch_metrics: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        scenario_sequences: Dict[str, torch.Tensor] = {}

        for scenario, use_actions in self.scenarios:
            if future_len <= 0:
                scenario_sequences[scenario] = observed_latents
                empty = torch.zeros_like(observed_latents[:, :0])
                batch_metrics[scenario] = (empty, empty)
                continue

            generated_frames: List[torch.Tensor] = []
            predictions: List[torch.Tensor] = []

            for step in range(future_len):
                past_generated = torch.cat(generated_frames, dim=1) if generated_frames else None
                noise_frame = sample_base_noise(
                    latents[:, context_len + step : context_len + step + 1, ...],
                    self.flow_cfg,
                )

                seq_latents_parts = [observed_latents]
                signal_parts = [
                    torch.full(
                        (batch_size, context_len),
                        self.cfg.clean_signal_level,
                        dtype=latents.dtype,
                        device=self.device,
                    )
                ]
                if past_generated is not None:
                    seq_latents_parts.append(past_generated)
                    signal_parts.append(
                        torch.full(
                            (batch_size, past_generated.shape[1]),
                            self.cfg.rollout_signal_level,
                            dtype=latents.dtype,
                            device=self.device,
                        )
                    )
                seq_latents_parts.append(noise_frame)
                signal_parts.append(
                    torch.zeros(
                        (batch_size, 1),
                        dtype=latents.dtype,
                        device=self.device,
                    )
                )

                seq_latents_tensor = torch.cat(seq_latents_parts, dim=1)
                signal_tensor = torch.cat(signal_parts, dim=1)

                length = seq_latents_tensor.shape[1]
                model_kwargs = self._build_model_kwargs(
                    actions,
                    actions_mask,
                    independant_mask,
                    length,
                    use_actions,
                )

                with torch.no_grad():
                    rollout = self.solver.sample(
                        model,
                        seq_latents_tensor,
                        initial_signal=signal_tensor,
                        **model_kwargs,
                    )

                if past_generated is not None:
                    updated = rollout[:, context_len : context_len + past_generated.shape[1], ...].detach()
                    for idx in range(len(generated_frames)):
                        generated_frames[idx] = updated[:, idx : idx + 1, ...]

                pred_frame = rollout[:, -1:, ...].detach()
                generated_frames.append(pred_frame)
                predictions.append(pred_frame)

            predicted_stack = torch.cat(predictions, dim=1)
            batch_metrics[scenario] = (predicted_stack, target_future)

            full_sequence = torch.cat([observed_latents, torch.cat(generated_frames, dim=1)], dim=1)
            scenario_sequences[scenario] = full_sequence

        video_samples = self._prepare_video_samples(frames, scenario_sequences)
        return batch_metrics, video_samples

    def _build_model_kwargs(
        self,
        actions: Optional[torch.Tensor],
        actions_mask: Optional[torch.Tensor],
        independant_mask: Optional[torch.Tensor],
        length: int,
        use_actions: bool,
    ) -> Dict[str, torch.Tensor]:
        kwargs: Dict[str, torch.Tensor] = {}
        if independant_mask is not None:
            kwargs["independant_frames_mask"] = independant_mask[:, :length]
        if use_actions and actions is not None:
            kwargs["actions"] = actions[:, :length]
            if actions_mask is not None:
                kwargs["actions_mask"] = actions_mask[:, :length]
        return kwargs

    def _prepare_video_samples(
        self,
        frames: torch.Tensor,
        scenario_predictions: Dict[str, torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        if self.cfg.num_video_samples <= 0:
            return []
        available = min(self.cfg.num_video_samples, frames.shape[0])
        samples: List[Dict[str, torch.Tensor]] = []
        for idx in range(available):
            entry: Dict[str, torch.Tensor] = {
                "eval/ground_truth": frames[idx].detach().cpu(),
            }
            for scenario, latents in scenario_predictions.items():
                decoded = self._decode_latents(latents[idx : idx + 1]).squeeze(0).cpu()
                entry[f"eval/{scenario}"] = decoded
            samples.append(entry)
        return samples

    @torch.no_grad()
    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = frames.shape
        flat = frames.view(batch * steps, channels, height, width)
        latents = self.autoencoder.encode(flat)
        tokens, dim = latents.shape[1], latents.shape[2]
        return latents.view(batch, steps, tokens, dim)

    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch, steps, tokens, dim = latents.shape
        flat = latents.view(batch * steps, tokens, dim)
        frames = self.autoencoder.decode(flat)
        channels, height, width = frames.shape[1:]
        return frames.view(batch, steps, channels, height, width)
