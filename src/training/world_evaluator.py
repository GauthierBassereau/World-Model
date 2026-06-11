from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

import pyrallis
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.training.logger import WorldModelLogger

from src.dataset.collator import StackCollator
from src.dataset.world_dataset import WorldDatasetConfig, WorldBatch
from src.dataset.loader import DataloaderConfig
from src.dataset.lerobot_dataset import LeRobotDataset, LeRobotDatasetConfig
from src.diffusion.signal_scheduler import SignalSchedulerConfig
from src.diffusion.common import calculate_velocity_1_to_2
from src.diffusion.euler_solver import EulerSolver, EulerSolverConfig
from src.world_model.rollout import collect_rollout_latents
from src.world_model.utils import expand_signal_levels_for_model, global_signal_levels_from_signal


@dataclass
class EvaluationConfig:
    max_batches: Optional[int] = None
    video_sample_indices: Optional[Union[str, List[int]]] = None
    batch_size: Optional[int] = None
    rollout_start_frame: int = 3
    rollout_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    rollout_signal_level: float = 0.9
    rollout_add_noise: bool = True
    teacher_forced_batch_size: int = 64
    precision: str = "bf16"
    denoising_metrics_indices: Optional[List[int]] = None
    save_denoising_frames: bool = True


@dataclass
class EvaluationSummary:
    metrics: Dict[str, float]
    videos: Dict[str, torch.Tensor]
    plots: Dict[str, Any]

class WorldModelEvaluator:
    def __init__(
        self,
        config: EvaluationConfig,
        dataset_cfg: WorldDatasetConfig,
        dataloader_cfg: DataloaderConfig,
        signal_scheduler_cfg: SignalSchedulerConfig,
        euler_solver_cfg: EulerSolverConfig,
        autoencoder: nn.Module,
        logger: WorldModelLogger,
        device: torch.device,
        seed: int,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        is_main_process: bool = True,
    ) -> None:
        self.config = config
        self.logger = logger
        if self.config.rollout_start_frame < 1:
            raise ValueError("evaluation.rollout_start_frame must be >= 1.")
        if self.config.teacher_forced_batch_size < 1:
            raise ValueError("evaluation.teacher_forced_batch_size must be >= 1.")
        self.rollout_horizons = sorted({int(h) for h in self.config.rollout_horizons})
        if not self.rollout_horizons or self.rollout_horizons[0] < 1:
            raise ValueError("evaluation.rollout_horizons must contain positive integers.")
        if not (0.0 < self.config.rollout_signal_level <= 1.0):
            raise ValueError("evaluation.rollout_signal_level must be in (0, 1], diffusion forcing has 0.9 by default")
        self.device = device
        self.rank = rank
        self.world_size = world_size or 1
        self.is_main_process = is_main_process
        self.autoencoder = autoencoder
        self.autoencoder.eval()
        self.euler_solver = EulerSolver(euler_solver_cfg, signal_scheduler_cfg)
        self.max_sequence_length = max(int(key) for key in dataset_cfg.sequence_length_distribution.keys())
        if self.max_sequence_length < 2:
            raise ValueError("Evaluation sequence length must be >= 2.")
        self.dataset_name, dataset = self._build_eval_dataset(dataset_cfg)
        self.dataloader = self._build_eval_dataloader(dataset_cfg, dataloader_cfg, dataset, seed)
        if self.is_main_process:
            self.logger.info(
                "Evaluation dataset %s has %d sampleable indices.",
                self.dataset_name,
                len(dataset),
            )
            self.logger.info(f"Evaluation dataloader has length: {len(self.dataloader)}")
            if self.config.max_batches is not None:
                self.config.max_batches = min(self.config.max_batches, len(self.dataloader))
                self.logger.info(f"Limiting evaluation to {self.config.max_batches} batches based on dataloader length.")

    def _build_eval_dataset(
        self,
        dataset_cfg: WorldDatasetConfig,
    ) -> Tuple[str, LeRobotDataset]:
        if len(dataset_cfg.datasets) != 1:
            raise ValueError(
                "Evaluation expects exactly one dataset. "
                "WorldDataset-based resampling was removed from evaluation."
            )

        dataset_name, ds_conf_dict = next(iter(dataset_cfg.datasets.items()))
        ds_type = ds_conf_dict.get("type")
        if ds_type != "lerobot":
            raise ValueError(f"Unsupported evaluation dataset type: {ds_type}")

        ds_conf_dict = ds_conf_dict.copy()
        del ds_conf_dict["type"]
        ds_conf_dict["sequence_length"] = self.max_sequence_length
        ds_conf_dict["fps"] = dataset_cfg.fps
        ds_conf_dict["action_dim"] = dataset_cfg.action_dim

        ds_cfg = pyrallis.decode(LeRobotDatasetConfig, ds_conf_dict)
        return dataset_name, LeRobotDataset(ds_cfg, logger=self.logger)

    def _build_eval_dataloader(
        self,
        dataset_cfg: WorldDatasetConfig,
        dataloader_cfg: DataloaderConfig,
        dataset: LeRobotDataset,
        seed: int,
    ) -> DataLoader:
        global_batch_size = self.config.batch_size or dataloader_cfg.batch_size
        if global_batch_size <= 0:
            raise ValueError("Evaluation batch size must be > 0.")

        distributed = self.world_size > 1
        if distributed and global_batch_size % self.world_size != 0:
            raise ValueError("Evaluation batch size must be divisible by world_size.")

        local_batch_size = global_batch_size // self.world_size if distributed else global_batch_size
        sampler: Optional[DistributedSampler] = None
        if distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank or 0,
                shuffle=dataloader_cfg.shuffle,
                drop_last=False,
                seed=seed,
            )

        collate = StackCollator(sequence_length_distribution=dataset_cfg.sequence_length_distribution)
        return DataLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=dataloader_cfg.shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=dataloader_cfg.num_workers,
            pin_memory=dataloader_cfg.pin_memory,
            collate_fn=collate,
            persistent_workers=True if dataloader_cfg.num_workers > 0 else False,
        )

    @torch.no_grad()
    def evaluate(self, model: nn.Module) -> Optional[EvaluationSummary]:
        metric_names = ["evaluation/teacher_forced/final_mse"] + [
            f"evaluation/rollout/mse_horizon_{horizon}"
            for horizon in self.rollout_horizons
        ]
        metric_sums: Dict[str, float] = {key: 0.0 for key in metric_names}
        metric_counts: Dict[str, float] = {key: 0.0 for key in metric_names}
        solver_velocity_sums = torch.zeros(self.euler_solver.config.number_steps, dtype=torch.float64)
        solver_velocity_count = 0.0
        video_samples: List[Dict[str, torch.Tensor]] = []

        was_training = model.training
        model.eval()

        for batch_idx, batch in enumerate(self.dataloader):
            self.logger.info(f"Evaluating batch {batch_idx + 1}/{self.config.max_batches if self.config.max_batches is not None else len(self.dataloader)}...")
            
            batch_metrics, batch_solver_velocity_sums, batch_solver_velocity_count, batch_video_samples = self._evaluate_batch(model, batch)
            self._accumulate_metric_stats(metric_sums, metric_counts, batch_metrics)
            solver_velocity_sums += batch_solver_velocity_sums
            solver_velocity_count += batch_solver_velocity_count
            video_samples.extend(batch_video_samples)

            self.logger.info(f"[rank {self.rank if self.rank is not None else 0}] Completed evaluation for batch {batch_idx + 1}.")

            if self.config.max_batches is not None and (batch_idx + 1) >= self.config.max_batches:
                break

        if was_training:
            model.train()

        solver_velocity_sums, solver_velocity_count = self._reduce_metric_totals(
            metric_sums,
            metric_counts,
            solver_velocity_sums,
            solver_velocity_count,
        )
        if self.config.video_sample_indices is not None:
            video_samples = self._gather_video_samples(video_samples)

        metrics: Dict[str, float] = {}
        for key, value_sum in metric_sums.items():
            count = metric_counts.get(key, 0.0)
            if count > 0.0:
                metrics[key] = value_sum / count

        plots: Dict[str, Any] = {}

        if solver_velocity_count > 0.0:
            solver_velocity_values = (solver_velocity_sums / solver_velocity_count).tolist()
            metrics["evaluation/teacher_forced/solver_velocity_mse"] = float(
                solver_velocity_sums.sum().item() / (solver_velocity_count * solver_velocity_sums.numel())
            )
            plots["evaluation/plots/teacher_forced_solver_velocity_mse"] = {
                "xs": list(range(len(solver_velocity_values))),
                "ys": [solver_velocity_values],
                "keys": ["velocity_mse"],
                "title": "Teacher-forced solver velocity MSE",
                "xname": "solver_step",
            }

        rollout_metric_keys = [
            key
            for key in metrics
            if key.startswith("evaluation/rollout/mse_horizon_")
        ]
        if rollout_metric_keys:
            horizons = sorted(int(key.rsplit("_", 1)[-1]) for key in rollout_metric_keys)
            plots["evaluation/plots/rollout_horizon_mse"] = {
                "xs": horizons,
                "ys": [[metrics[f"evaluation/rollout/mse_horizon_{h}"] for h in horizons]],
                "keys": ["mse"],
                "title": "Rollout MSE by horizon",
                "xname": "horizon",
            }

        videos: Dict[str, torch.Tensor] = {}
        for payload in video_samples:
            sample_index = int(payload.pop("index").item())
            for key, frames in payload.items():
                videos[f"evaluation/video_samples/sample_{sample_index}/{key}"] = frames

        summary = EvaluationSummary(metrics=metrics, videos=videos, plots=plots)
        self.logger.log_evaluation(summary)
        return summary

    def _evaluate_batch(
        self,
        model: nn.Module,
        batch: WorldBatch,
    ) -> Tuple[
        Dict[str, Tuple[float, float]],
        torch.Tensor,
        float,
        List[Dict[str, torch.Tensor]],
    ]:
        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float32
        is_autocast = self.config.precision == "bf16"
        
        with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=is_autocast):
            frames = batch.sequence_frames.to(self.device, non_blocking=True)
            actions = batch.sequence_actions.to(self.device, non_blocking=True)
            frames_valid_mask = ~batch.padded_frames.to(self.device, non_blocking=True)
            use_actions = batch.use_actions.to(self.device, non_blocking=True).bool() & frames_valid_mask

            latents = self._encode_frames(frames)

            batch_metrics, solver_velocity_sums, solver_velocity_count = self._evaluate_teacher_forced(
                model,
                latents,
                actions,
                use_actions,
                frames_valid_mask,
            )
            rollout_metrics, video_samples = self._evaluate_rollout(
                model,
                latents,
                actions,
                use_actions,
                frames_valid_mask,
                batch,
            )
            batch_metrics.update(rollout_metrics)

            return batch_metrics, solver_velocity_sums, solver_velocity_count, video_samples

    def _evaluate_teacher_forced(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        actions: torch.Tensor,
        use_actions: torch.Tensor,
        frames_valid_mask: torch.Tensor,
    ) -> Tuple[Dict[str, Tuple[float, float]], torch.Tensor, float]:
        valid_transitions = frames_valid_mask[:, :-1] & frames_valid_mask[:, 1:]
        if not valid_transitions.any():
            return {}, torch.zeros(self.euler_solver.config.number_steps, dtype=torch.float64), 0.0

        context_latents = latents[:, :-1][valid_transitions].unsqueeze(1).detach()
        target_latents = latents[:, 1:][valid_transitions].unsqueeze(1).detach()
        context_actions = actions[:, :-1][valid_transitions].unsqueeze(1)
        target_actions = actions[:, 1:][valid_transitions].unsqueeze(1)
        context_use_actions = use_actions[:, :-1][valid_transitions].unsqueeze(1)
        target_use_actions = use_actions[:, 1:][valid_transitions].unsqueeze(1)

        final_mse_sum = 0.0
        generated_count = 0.0
        solver_velocity_sums = torch.zeros(self.euler_solver.config.number_steps, dtype=torch.float64)
        chunk_size = self.config.teacher_forced_batch_size

        for start in range(0, target_latents.shape[0], chunk_size):
            end = min(start + chunk_size, target_latents.shape[0])
            predicted, chunk_solver_sums = self._sample_teacher_forced_chunk(
                model,
                context_latents[start:end],
                target_latents[start:end],
                context_actions[start:end],
                target_actions[start:end],
                context_use_actions[start:end],
                target_use_actions[start:end],
            )
            target = target_latents[start:end]
            per_frame_mse = (predicted.float() - target.float()).pow(2).mean(dim=(1, 2, 3))
            final_mse_sum += per_frame_mse.sum().item()
            generated_count += float(per_frame_mse.numel())
            solver_velocity_sums += chunk_solver_sums.cpu()

        return (
            {"evaluation/teacher_forced/final_mse": (final_mse_sum, generated_count)},
            solver_velocity_sums,
            generated_count,
        )

    def _sample_teacher_forced_chunk(
        self,
        model: nn.Module,
        context_latents: torch.Tensor,
        target_latents: torch.Tensor,
        context_actions: torch.Tensor,
        target_actions: torch.Tensor,
        context_use_actions: torch.Tensor,
        target_use_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = target_latents.shape[0]
        device = target_latents.device
        signal_dtype = target_latents.dtype

        context_signal = torch.ones((batch_size, 1), device=device, dtype=signal_dtype)
        context_signal = expand_signal_levels_for_model(model, context_signal, context_latents.shape[2])
        context_output = model(
            noisy_latents=context_latents,
            signal_levels=context_signal,
            global_signal_levels=global_signal_levels_from_signal(context_signal),
            actions=context_actions,
            independent_frames=None,
            use_actions=context_use_actions,
            kv_cache=None,
        )
        kv_cache = context_output.kv_cache

        if self.euler_solver.config.timestep_schedule == "linear":
            times = torch.linspace(0.0, 1.0, self.euler_solver.config.number_steps + 1, device=device)
        else:
            times = self.euler_solver.signal_scheduler.get_timesteps(self.euler_solver.config.number_steps).to(device=device)

        x = torch.randn_like(target_latents)
        solver_velocity_sums = torch.zeros(self.euler_solver.config.number_steps, device=device, dtype=torch.float64)

        for step_idx, t_curr in enumerate(times[:-1]):
            t_next = times[step_idx + 1]
            dt = t_next - t_curr
            signal_levels = torch.full((batch_size, 1), float(t_curr.item()), device=device, dtype=signal_dtype)
            signal_levels = expand_signal_levels_for_model(model, signal_levels, target_latents.shape[2])

            output = model(
                noisy_latents=x,
                signal_levels=signal_levels,
                global_signal_levels=global_signal_levels_from_signal(signal_levels),
                actions=target_actions,
                independent_frames=None,
                use_actions=target_use_actions,
                kv_cache=kv_cache,
            )
            target_signal = torch.ones_like(signal_levels)
            predicted_velocity = calculate_velocity_1_to_2(
                latent_1=x,
                latent_2=output.latents,
                signal_levels_1=signal_levels,
                signal_levels_2=target_signal,
                min_denom=self.euler_solver.config.min_denom,
            )
            target_velocity = calculate_velocity_1_to_2(
                latent_1=x,
                latent_2=target_latents,
                signal_levels_1=signal_levels,
                signal_levels_2=target_signal,
                min_denom=self.euler_solver.config.min_denom,
            )
            per_frame_velocity_mse = (
                predicted_velocity.float() - target_velocity.float()
            ).pow(2).mean(dim=(1, 2, 3))
            solver_velocity_sums[step_idx] = per_frame_velocity_mse.sum().to(dtype=torch.float64)
            x = x + predicted_velocity * dt.to(dtype=x.dtype)

        return x, solver_velocity_sums

    def _evaluate_rollout(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        actions: torch.Tensor,
        use_actions: torch.Tensor,
        frames_valid_mask: torch.Tensor,
        batch: WorldBatch,
    ) -> Tuple[Dict[str, Tuple[float, float]], List[Dict[str, torch.Tensor]]]:
        seq_len = latents.shape[1]
        context_len = min(self.config.rollout_start_frame, max(seq_len - 1, 1))
        future_len = seq_len - context_len
        if future_len <= 0:
            return {}, []

        context_valid = frames_valid_mask[:, :context_len].all(dim=1)
        if not context_valid.any():
            return {}, []

        predicted_stack, full_sequence, _ = collect_rollout_latents(
            model,
            self.euler_solver,
            latents,
            context_len,
            future_len,
            self.config.rollout_signal_level,
            rollout_add_noise=self.config.rollout_add_noise,
            actions=actions,
            use_actions=use_actions,
            independent_frames=None,
            target_latents=None,
            denoising_metrics_indices=None,
        )

        target_future = latents[:, context_len:context_len + future_len].detach()
        per_frame_mse = (predicted_stack.float() - target_future.float()).pow(2).mean(dim=(2, 3))

        metrics: Dict[str, Tuple[float, float]] = {}
        future_valid = frames_valid_mask[:, context_len:context_len + future_len]
        for horizon in self.rollout_horizons:
            if horizon > future_len:
                continue
            horizon_valid = context_valid & future_valid[:, :horizon].all(dim=1)
            if not horizon_valid.any():
                continue
            horizon_errors = per_frame_mse[horizon_valid, :horizon]
            metrics[f"evaluation/rollout/mse_horizon_{horizon}"] = (
                horizon_errors.sum().item(),
                float(horizon_errors.numel()),
            )

        video_samples = self._prepare_rollout_video_samples(latents, full_sequence, batch)
        return metrics, video_samples

    def _prepare_rollout_video_samples(
        self,
        latents: torch.Tensor,
        rollout_latents: torch.Tensor,
        batch: WorldBatch,
    ) -> List[Dict[str, torch.Tensor]]:
        if self.config.video_sample_indices is None:
            return []

        select_all = self.config.video_sample_indices == "all"
        samples: List[Dict[str, torch.Tensor]] = []
        dataset_indices = batch.dataset_indices.detach().cpu().tolist()
        episode_ids = batch.episode_ids.detach().cpu().tolist()

        for batch_idx in range(latents.shape[0]):
            dataset_index = dataset_indices[batch_idx]
            episode_id = episode_ids[batch_idx]
            match_key = episode_id if episode_id != -1 else dataset_index
            if not select_all and match_key not in self.config.video_sample_indices:
                continue

            samples.append(
                {
                    "index": torch.tensor(match_key, dtype=torch.long),
                    "ground_truth": self._decode_latents(latents[batch_idx : batch_idx + 1]).squeeze(0).cpu(),
                    "rollout": self._decode_latents(rollout_latents[batch_idx : batch_idx + 1]).squeeze(0).cpu(),
                }
            )

        return samples

    def _gather_video_samples(self, local_samples: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        if self.world_size <= 1 or not dist.is_initialized():
            return local_samples

        run_name = self.logger.cfg.run_name or "run"
        safe_run_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in run_name)
        temp_dir = Path(".") / "temp_gather_videos" / f"{safe_run_name}_step_{self.logger.current_step}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        rank = self.rank or 0
        torch.save(local_samples, temp_dir / f"samples_rank_{rank}.pt")
        dist.barrier()

        gathered_samples: List[Dict[str, torch.Tensor]] = []
        if self.is_main_process:
            for source_rank in range(self.world_size):
                rank_file = temp_dir / f"samples_rank_{source_rank}.pt"
                try:
                    gathered_samples.extend(torch.load(rank_file, map_location="cpu"))
                except Exception as exc:
                    self.logger.warning("Failed to load eval video samples from rank %d: %s", source_rank, exc)

            import shutil

            try:
                shutil.rmtree(temp_dir)
            except Exception as exc:
                self.logger.warning("Failed to clean up eval video gather dir %s: %s", temp_dir, exc)

        dist.barrier()
        return gathered_samples

    @staticmethod
    def _accumulate_metric_stats(
        metric_sums: Dict[str, float],
        metric_counts: Dict[str, float],
        batch_metrics: Dict[str, Tuple[float, float]],
    ) -> None:
        for key, (value_sum, count) in batch_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value_sum)
            metric_counts[key] = metric_counts.get(key, 0.0) + float(count)

    def _reduce_metric_totals(
        self,
        metric_sums: Dict[str, float],
        metric_counts: Dict[str, float],
        solver_velocity_sums: torch.Tensor,
        solver_velocity_count: float,
    ) -> Tuple[torch.Tensor, float]:
        if self.world_size <= 1 or not dist.is_initialized():
            return solver_velocity_sums, solver_velocity_count

        for key in sorted(set(metric_sums.keys()) | set(metric_counts.keys())):
            totals = torch.tensor(
                [metric_sums.get(key, 0.0), metric_counts.get(key, 0.0)],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            metric_sums[key] = totals[0].item()
            metric_counts[key] = totals[1].item()

        reduced_solver_sums = solver_velocity_sums.to(device=self.device, dtype=torch.float64)
        reduced_solver_count = torch.tensor(solver_velocity_count, device=self.device, dtype=torch.float64)
        dist.all_reduce(reduced_solver_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(reduced_solver_count, op=dist.ReduceOp.SUM)
        return reduced_solver_sums.cpu(), reduced_solver_count.item()

    @torch.no_grad()
    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.float() / 255.0
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
