
import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Subset

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
    video_sample_indices: Optional[List[int]] = None
    batch_size: Optional[int] = None
    rollout_start_frame: int = 6
    rollout_signal_level: float = 0.9
    clean_signal_level: float = 1.0
    precision: str = "bf16"


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

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target must share the same shape for MAE stats.")
        if prediction.ndim != 4:
            raise ValueError("Expected tensors of shape [B, T, tokens, dim] for MAE stats.")

        abs_diff = torch.abs(prediction - target)
        squared_abs_diff = abs_diff.pow(2)

        if valid_mask.shape[0] != prediction.shape[0] or valid_mask.shape[1] != prediction.shape[1]:
            raise ValueError("valid_mask must match the batch and time dimensions of the tensors.")
        mask = valid_mask.to(device=prediction.device, dtype=prediction.dtype).unsqueeze(-1).unsqueeze(-1)
        expanded_mask = mask.expand_as(abs_diff)
        abs_diff = abs_diff * expanded_mask
        squared_abs_diff = squared_abs_diff * expanded_mask
        per_step_count = expanded_mask.detach().double().cpu().sum(dim=(0, 2, 3))

        per_step_sum = abs_diff.sum(dim=(0, 2, 3)).detach().double().cpu()
        per_step_sq_sum = squared_abs_diff.sum(dim=(0, 2, 3)).detach().double().cpu()
        length = min(per_step_sum.shape[0], self.max_steps)
        if length <= 0:
            return
        self.step_sum[:length] += per_step_sum[:length]
        self.step_sq_sum[:length] += per_step_sq_sum[:length]
        self.step_count[:length] += per_step_count[:length]
        self.total_sum += float(per_step_sum[:length].sum().item())
        self.total_sq_sum += float(per_step_sq_sum[:length].sum().item())
        self.total_count += float(per_step_count[:length].sum().item())

    def as_dict(self, scenario: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        rollout_prefix = f"evaluation_rollouts_metrics/{scenario}"
        if self.total_count > 0:
            mean = self.total_sum / self.total_count
            mean_sq = self.total_sq_sum / self.total_count
            var = max(mean_sq - mean * mean, 0.0)
            l2 = math.sqrt(max(mean_sq, 0.0))
            metrics[f"evaluation/l1_loss/{scenario}"] = mean
            metrics[f"evaluation/l2_loss/{scenario}"] = l2
            metrics[f"evaluation/var/{scenario}"] = var
        valid = self.step_count > 0
        means = torch.zeros_like(self.step_sum)
        vars = torch.zeros_like(self.step_sq_sum)
        means[valid] = self.step_sum[valid] / self.step_count[valid]
        vars[valid] = self.step_sq_sum[valid] / self.step_count[valid]
        vars = torch.clamp(vars - means.pow(2), min=0.0)
        for idx in range(self.max_steps):
            if self.step_count[idx] > 0:
                metrics[f"{rollout_prefix}/mean_t+{idx}"] = float(means[idx].item())
                metrics[f"{rollout_prefix}/var_t+{idx}"] = float(vars[idx].item())
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
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        is_main_process: bool = True,
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
        self.rank = rank
        self.world_size = world_size or 1
        self.is_main_process = is_main_process
        self.autoencoder = autoencoder
        self.autoencoder.eval()
        self.flow_cfg = diffusion_cfg
        self.solver = EulerSolver(solver_cfg or EulerSolverConfig())
        self.max_sequence_length = max(
            int(key) for key in dataset_cfg.sequence_length_distribution.keys()
        )
        if self.max_sequence_length < 2:
            raise ValueError("Evaluation sequence length must be >= 2.")
        base_context = min(self.cfg.rollout_start_frame, self.max_sequence_length - 1)
        self.context_limit = max(1, base_context)
        self.max_future_steps = max(1, self.max_sequence_length - self.context_limit)
        try:
            self.dataloader = build_world_model_dataloader(
                dataset_cfg=dataset_cfg,
                dataloader_cfg=dataloader_cfg,
                grad_accum_steps=1,
                device=device,
                rank=self.rank if self.world_size > 1 else None,
                world_size=self.world_size if self.world_size > 1 else None,
                seed=seed,
            )
            print(
                f"Length of evaluation dataloader (rank {self.rank if self.rank is not None else 0}): {len(self.dataloader)}"
            )
            if self.cfg.max_batches is not None:
                self.cfg.max_batches = min(self.cfg.max_batches, len(self.dataloader))
                print(f"Limiting evaluation to {self.cfg.max_batches} batches based on dataloader length.")
            self._sampled_dataset_indices = self._materialize_sampler_indices(self.dataloader)
            self._sampled_episode_ids = self._dataset_indices_to_episode_ids(
                getattr(self.dataloader, "dataset", None), self._sampled_dataset_indices
            )
        except ValueError:
            self.dataloader = None
            self._sampled_dataset_indices = None
            self._sampled_episode_ids = None
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

        sample_offset = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                print(f"[rank {self.rank if self.rank is not None else 0}] Evaluating batch {batch_idx + 1}/{self.cfg.max_batches if self.cfg.max_batches is not None else len(self.dataloader)}...")
                batch_size = batch.sequence_frames.shape[0]
                dataset_indices = None
                dataset_episode_ids = None
                if self._sampled_dataset_indices is not None:
                    end = sample_offset + batch_size
                    dataset_indices = self._sampled_dataset_indices[sample_offset:end]
                    if self._sampled_episode_ids is not None:
                        dataset_episode_ids = self._sampled_episode_ids[sample_offset:end]
                    sample_offset = end
                batch_stats, sample_payloads = self._evaluate_batch(
                    model,
                    batch,
                    dataset_indices,
                    dataset_episode_ids,
                )
                for scenario, value in batch_stats.items():
                    prediction, target, valid_mask = value
                    stats[scenario].update(prediction, target, valid_mask)
                
                if sample_payloads:
                    video_samples.extend(sample_payloads)
                print(f"[rank {self.rank if self.rank is not None else 0}] Completed evaluation for batch {batch_idx + 1}.")

                if self.cfg.max_batches is not None and (batch_idx + 1) >= self.cfg.max_batches:
                    break

        if was_training:
            model.train()

        self._reduce_stats(stats)
        video_samples = self._gather_video_samples(video_samples)

        metrics: Dict[str, float] = {}
        for scenario, _ in self.scenarios:
            metrics.update(stats[scenario].as_dict(scenario))

        videos: Dict[str, torch.Tensor] = {}
        for payload in video_samples:
            # payload now includes "index" to identify the sample
            idx = payload.pop("index")
            for key, frames in payload.items():
                videos[f"evaluation/video_samples/sample_{idx}/{key}"] = frames

        return EvaluationSummary(metrics=metrics, videos=videos)

    def _evaluate_batch(
        self,
        model: nn.Module,
        batch: WorldModelBatch,
        dataset_indices: Optional[List[int]],
        dataset_episode_ids: Optional[List[int]],
    ) -> Tuple[
        Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        List[Dict[str, torch.Tensor]],
    ]:
        dtype = torch.bfloat16 if self.cfg.precision == "bf16" else torch.float32
        is_autocast = self.cfg.precision == "bf16"
        
        with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=is_autocast):
            frames = batch.sequence_frames.to(self.device, non_blocking=True)
            actions = batch.sequence_actions.to(self.device, non_blocking=True)
            independent_mask = None
            actions_mask = None
            frames_valid_mask = batch.frames_valid_mask.to(self.device, non_blocking=True)

            latents = self._encode_frames(frames)
            batch_size, seq_len = latents.shape[0], latents.shape[1]
            context_len = min(self.cfg.rollout_start_frame, max(seq_len - 1, 1))
            future_len = max(seq_len - context_len, 0)

            observed_latents = latents[:, :context_len, ...].detach()
            target_future = latents[:, context_len:, ...].detach() if future_len > 0 else None

            batch_metrics: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            scenario_sequences: Dict[str, torch.Tensor] = {}
            future_mask = frames_valid_mask[:, context_len:]

            for scenario, use_actions in self.scenarios:
                if future_len <= 0:
                    scenario_sequences[scenario] = observed_latents
                    empty = torch.zeros_like(observed_latents[:, :0])
                    empty_mask = future_mask
                    batch_metrics[scenario] = (empty, empty, empty_mask)
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
                        independent_mask,
                        length,
                        use_actions,
                    )

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
                batch_metrics[scenario] = (predicted_stack, target_future, future_mask)

                full_sequence = torch.cat([observed_latents, torch.cat(generated_frames, dim=1)], dim=1)
                scenario_sequences[scenario] = full_sequence

            video_samples = self._prepare_video_samples(
                frames, scenario_sequences, dataset_indices, dataset_episode_ids
            )
            return batch_metrics, video_samples

    def _reduce_stats(self, stats: Dict[str, SequenceErrorStats]) -> None:
        if self.world_size <= 1 or not dist.is_initialized():
            return
        for scenario_stats in stats.values():
            totals = torch.tensor(
                [
                    scenario_stats.total_sum,
                    scenario_stats.total_sq_sum,
                    scenario_stats.total_count,
                ],
                device=self.device,
                dtype=torch.float64,
            )
            step_sum = scenario_stats.step_sum.to(self.device)
            step_sq_sum = scenario_stats.step_sq_sum.to(self.device)
            step_count = scenario_stats.step_count.to(self.device)

            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            dist.all_reduce(step_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(step_sq_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(step_count, op=dist.ReduceOp.SUM)

            scenario_stats.total_sum = float(totals[0].item())
            scenario_stats.total_sq_sum = float(totals[1].item())
            scenario_stats.total_count = float(totals[2].item())
            scenario_stats.step_sum = step_sum.cpu()
            scenario_stats.step_sq_sum = step_sq_sum.cpu()
            scenario_stats.step_count = step_count.cpu()

    def _gather_video_samples(self, local_samples: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        if self.world_size <= 1 or not dist.is_initialized():
            return local_samples
        gathered: List[Optional[List[Dict[str, torch.Tensor]]]] = [None for _ in range(self.world_size)] if self.is_main_process else None
        dist.gather_object(local_samples, object_gather_list=gathered, dst=0)
        if not self.is_main_process:
            return []
        all_samples: List[Dict[str, torch.Tensor]] = []
        for entry in gathered or []:
            if entry:
                all_samples.extend(entry)
        return all_samples

    def _build_model_kwargs(
        self,
        actions: Optional[torch.Tensor],
        actions_mask: Optional[torch.Tensor],
        independent_mask: Optional[torch.Tensor],
        length: int,
        use_actions: bool,
    ) -> Dict[str, torch.Tensor]:
        kwargs: Dict[str, torch.Tensor] = {}
        if independent_mask is not None:
            kwargs["independent_frames_mask"] = independent_mask[:, :length]
        if use_actions and actions is not None:
            kwargs["actions"] = actions[:, :length]
            if actions_mask is not None:
                kwargs["actions_mask"] = actions_mask[:, :length]
        return kwargs

    def _prepare_video_samples(
        self,
        frames: torch.Tensor,
        scenario_predictions: Dict[str, torch.Tensor],
        dataset_indices: Optional[List[int]],
        dataset_episode_ids: Optional[List[int]],
    ) -> List[Dict[str, Union[torch.Tensor, int]]]:
        target_indices = self.cfg.video_sample_indices
        if not target_indices:
            return []
        
        samples: List[Dict[str, Union[torch.Tensor, int]]] = []
        batch_size = frames.shape[0]
        
        for batch_idx in range(batch_size):
            if dataset_indices is None or batch_idx >= len(dataset_indices):
                continue
            current_idx = dataset_indices[batch_idx]
            episode_id = None
            if dataset_episode_ids is not None and batch_idx < len(dataset_episode_ids):
                episode_id = dataset_episode_ids[batch_idx]
            match_key = episode_id if episode_id is not None else current_idx
            if match_key not in target_indices:
                continue
                
            entry: Dict[str, Union[torch.Tensor, int]] = {
                "ground_truth": frames[batch_idx].detach().cpu(),
                "index": match_key,
            }
            for scenario, latents in scenario_predictions.items():
                decoded = self._decode_latents(latents[batch_idx : batch_idx + 1]).squeeze(0).cpu()
                entry[scenario] = decoded
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

    def _dataset_indices_to_episode_ids(
        self,
        dataset: Optional[torch.utils.data.Dataset],
        indices: Optional[List[int]],
    ) -> Optional[List[int]]:
        if dataset is None or indices is None:
            return None

        # Primary path: datasets used for eval midpoints carry an episode_ids list aligned to dataset length.
        if hasattr(dataset, "episode_ids"):
            ep_ids = getattr(dataset, "episode_ids")
            try:
                return [int(ep_ids[i]) for i in indices if i < len(ep_ids)]
            except Exception:
                pass

        # Fallback: use hf_dataset episode_index column if present.
        if hasattr(dataset, "hf_dataset"):
            try:
                ep_col = dataset.hf_dataset["episode_index"]
                result: List[int] = []
                for i in indices:
                    if i < len(ep_col):
                        val = ep_col[i]
                        result.append(int(val.item()) if hasattr(val, "item") else int(val))
                return result or None
            except Exception:
                return None

        return None

    def _materialize_sampler_indices(self, dataloader: torch.utils.data.DataLoader) -> Optional[List[int]]:
        sampler = getattr(dataloader, "sampler", None)
        if sampler is None:
            return None
        try:
            indices = list(iter(sampler))
        except Exception:
            return None
        return indices
