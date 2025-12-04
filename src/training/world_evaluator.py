import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.distributed as dist
import torch.nn as nn

from src.training.logger import WorldModelLogger

from src.dataset.world_dataset import WorldDatasetConfig, WorldBatch, WorldDataset
from src.dataset.loader import build_world_dataloader, DataloaderConfig
from src.diffusion.signal_scheduler import SignalSchedulerConfig
from src.diffusion.euler_solver import EulerSolver, EulerSolverConfig
from src.world_model.rollout import rollout_latents


@dataclass
class EvaluationConfig:
    max_batches: Optional[int] = None
    video_sample_indices: Optional[List[int]] = None
    batch_size: Optional[int] = None
    rollout_start_frame: int = 3
    rollout_signal_level: float = 0.9
    precision: str = "bf16"
    denoising_metrics_indices: Optional[List[int]] = None


@dataclass
class EvaluationSummary:
    metrics: Dict[str, float]
    videos: Dict[str, torch.Tensor]
    plots: Dict[str, Any]

# TODO dirty code... works for now
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
        self.max_future_steps = max(1, self.max_sequence_length - self.config.rollout_start_frame)
        dataset = WorldDataset(dataset_cfg, logger=self.logger, seed=seed)
        if self.is_main_process:
            self.logger.info(dataset)
        self.dataloader = build_world_dataloader(
            dataset=dataset,
            dataloader_cfg=dataloader_cfg,
            grad_accum_steps=1,
            rank=self.rank if self.world_size > 1 else None,
            world_size=self.world_size if self.world_size > 1 else None,
            seed=seed,
        )
        if self.is_main_process:
            self.logger.info(f"Evaluation dataloader has length: {len(self.dataloader)}")
            if self.config.max_batches is not None:
                self.config.max_batches = min(self.config.max_batches, len(self.dataloader))
                self.logger.info(f"Limiting evaluation to {self.config.max_batches} batches based on dataloader length.")
        self.scenarios: List[Tuple[str, bool]] = [("actions", True), ("no_actions", False)]
        self.ground_truth_logged = False

    @torch.no_grad()
    def evaluate(self, model: nn.Module) -> Optional[EvaluationSummary]:
        metrics_accumulator: Dict[str, List[float]] = {}
        denoising_metrics_accumulator: Dict[str, Dict[int, Dict[int, List[float]]]] = {} 
        # scenario -> rollout_step -> diffusion_step -> list of values

        video_samples: List[Dict[str, torch.Tensor]] = []
        was_training = model.training
        model.eval()

        sample_offset = 0
        for batch_idx, batch in enumerate(self.dataloader):
            self.logger.info(f"Evaluating batch {batch_idx + 1}/{self.config.max_batches if self.config.max_batches is not None else len(self.dataloader)}...")
            
            batch_metrics, sample_payloads, batch_denoising = self._evaluate_batch(model, batch)
            
            for key, val in batch_metrics.items():
                if key not in metrics_accumulator:
                    metrics_accumulator[key] = []
                metrics_accumulator[key].append(val)
            
            for scenario, d_metrics in batch_denoising.items():
                if scenario not in denoising_metrics_accumulator:
                    denoising_metrics_accumulator[scenario] = {}
                for r_step, steps_data in d_metrics.items():
                    if r_step not in denoising_metrics_accumulator[scenario]:
                        denoising_metrics_accumulator[scenario][r_step] = {}
                    for d_step, val in steps_data.items():
                        if d_step not in denoising_metrics_accumulator[scenario][r_step]:
                            denoising_metrics_accumulator[scenario][r_step][d_step] = []
                        denoising_metrics_accumulator[scenario][r_step][d_step].append(val)
            
            if sample_payloads:
                video_samples.extend(sample_payloads)
            self.logger.info(f"[rank {self.rank if self.rank is not None else 0}] Completed evaluation for batch {batch_idx + 1}.")

            if self.config.max_batches is not None and (batch_idx + 1) >= self.config.max_batches:
                break

        if was_training:
            model.train()

        self._reduce_metrics(metrics_accumulator)
        
        video_samples = self._gather_video_samples(video_samples)

        metrics: Dict[str, float] = {}
        for key, values in metrics_accumulator.items():
            if values:
                metrics[key] = sum(values) / len(values)
        
        plots: Dict[str, Any] = {}
        # Process denoising metrics for plotting
        # We want one plot per scenario per metric type (l1, l2)
        # Each plot has multiple lines (one per rollout step)

        videos: Dict[str, torch.Tensor] = {}
        for payload in video_samples:
            # payload now includes "index" to identify the sample
            idx = payload.pop("index")
            for key, frames in payload.items():
                videos[f"evaluation/video_samples/sample_{idx}/{key}"] = frames

        # Actually constructing the plots dictionary here
        for scenario, r_steps_data in denoising_metrics_accumulator.items():
             xs = []
             l1_ys = []
             l2_ys = []
             keys = []
             
             # Assuming all rollout steps have same diffusion steps
             first_r_step = next(iter(r_steps_data))
             xs = sorted(r_steps_data[first_r_step].keys())
             
             for r_step in sorted(r_steps_data.keys()):
                 keys.append(f"rollout_step_{r_step}")
                 l1_series = []
                 l2_series = []
                 for d_step in xs:
                     # vals is list of {"l1": ..., "l2": ...}
                     vals_list = r_steps_data[r_step][d_step]
                     # Average
                     avg_l1 = sum(v["l1"] for v in vals_list) / len(vals_list)
                     avg_l2 = sum(v["l2"] for v in vals_list) / len(vals_list)
                     l1_series.append(avg_l1)
                     l2_series.append(avg_l2)
                 l1_ys.append(l1_series)
                 l2_ys.append(l2_series)
             
             plots[f"denoising_l1/{scenario}"] = {"xs": xs, "ys": l1_ys, "keys": keys, "title": f"Denoising L1 ({scenario})"}
             plots[f"denoising_l2/{scenario}"] = {"xs": xs, "ys": l2_ys, "keys": keys, "title": f"Denoising L2 ({scenario})"}

        for scenario, _ in self.scenarios:
             step_keys = [k for k in metrics.keys() if k.startswith(f"evaluation_rollouts_metrics/{scenario}/l1_step_")]
             if step_keys:
                 steps = sorted([int(k.split("_")[-1]) for k in step_keys])
                 l1_vals = [metrics[f"evaluation_rollouts_metrics/{scenario}/l1_step_{t}"] for t in steps]
                 l2_vals = [metrics[f"evaluation_rollouts_metrics/{scenario}/l2_step_{t}"] for t in steps]
                 
                 plots[f"rollout_l1/{scenario}"] = {
                     "xs": steps,
                     "ys": [l1_vals],
                     "keys": ["l1_error"],
                     "title": f"Rollout L1 Error ({scenario})",
                     "xname": "rollout_step"
                 }
                 plots[f"rollout_l2/{scenario}"] = {
                     "xs": steps,
                     "ys": [l2_vals],
                     "keys": ["l2_error"],
                     "title": f"Rollout L2 Error ({scenario})",
                     "xname": "rollout_step"
                 }

        self.logger.log_evaluation(EvaluationSummary(metrics=metrics, videos=videos, plots=plots))
        self.ground_truth_logged = True

    def _evaluate_batch(
        self,
        model: nn.Module,
        batch: WorldBatch,
    ) -> Tuple[
        Dict[str, float],
        List[Dict[str, torch.Tensor]],
        Dict[str, Dict[int, Dict[int, Dict[str, float]]]], # scenario -> rollout_step -> diffusion_step -> metrics
    ]:
        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float32
        is_autocast = self.config.precision == "bf16"
        
        with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=is_autocast):
            frames = batch.sequence_frames.to(self.device, non_blocking=True)
            actions = batch.sequence_actions.to(self.device, non_blocking=True)
            independent_frames = None
            use_actions = None
            frames_valid_mask = ~batch.padded_frames.to(self.device, non_blocking=True)

            latents = self._encode_frames(frames)
            seq_len = latents.shape[1]
            context_len = min(self.config.rollout_start_frame, max(seq_len - 1, 1))
            future_len = max(seq_len - context_len, 0)

            observed_latents = latents[:, :context_len, ...].detach()
            target_future = latents[:, context_len:, ...].detach() if future_len > 0 else None

            batch_metrics: Dict[str, float] = {}
            batch_denoising: Dict[str, Dict[int, Dict[int, Dict[str, float]]]] = {}
            scenario_sequences: Dict[str, torch.Tensor] = {}
            future_mask = frames_valid_mask[:, context_len:]

            for scenario, use_actions_flag in self.scenarios:
                if future_len <= 0:
                    scenario_sequences[scenario] = observed_latents
                    # Log 0 metrics for empty future
                    batch_metrics[f"evaluation/l1_loss/{scenario}"] = 0.0
                    batch_metrics[f"evaluation/l2_loss/{scenario}"] = 0.0
                    continue

                use_actions_tensor = None
                current_actions = None
                if use_actions_flag:
                    current_actions = actions
                    use_actions_tensor = torch.ones(
                        (latents.shape[0], latents.shape[1]), 
                        dtype=torch.bool, 
                        device=self.device
                    )

                predicted_stack, full_sequence, denoising_metrics = rollout_latents(
                    model,
                    self.euler_solver,
                    latents,
                    context_len,
                    future_len,
                    self.config.rollout_signal_level,
                    actions=current_actions,
                    use_actions=use_actions_tensor,
                    independent_frames=independent_frames,
                    target_latents=target_future,
                    denoising_metrics_indices=self.config.denoising_metrics_indices,
                )
                
                
                # Restore rollout metrics
                # predicted_stack: [B, T, tokens, dim]
                # target_future: [B, T, tokens, dim]
                if target_future is not None:
                    diff = predicted_stack - target_future
                    l1_loss = diff.abs().mean().item()
                    l2_loss = diff.pow(2).mean().sqrt().item()
                    batch_metrics[f"evaluation/l1_loss/{scenario}"] = l1_loss
                    batch_metrics[f"evaluation/l2_loss/{scenario}"] = l2_loss
                    
                    # Per step metrics
                    # diff is [B, T, tokens, dim]
                    per_step_l1 = diff.abs().mean(dim=(0, 2, 3)).cpu() # [T]
                    per_step_l2 = diff.pow(2).mean(dim=(0, 2, 3)).sqrt().cpu() # [T]
                    
                    for t in range(per_step_l1.shape[0]):
                        batch_metrics[f"evaluation_rollouts_metrics/{scenario}/l1_step_{t}"] = per_step_l1[t].item()
                        batch_metrics[f"evaluation_rollouts_metrics/{scenario}/l2_step_{t}"] = per_step_l2[t].item()

                # Process denoising metrics
                # denoising_metrics keys are "step_{t}/l1_step_{i}"
                if denoising_metrics:
                    batch_denoising[scenario] = {}
                    for k, v in denoising_metrics.items():
                        # k format: step_{t}/l1_step_{i}
                        parts = k.split('/')
                        rollout_step = int(parts[0].split('_')[1])
                        metric_part = parts[1] # l1_step_{i}
                        metric_type = metric_part.split('_')[0] # l1 or l2
                        diffusion_step = float(metric_part.split('_')[2])
                        
                        if rollout_step not in batch_denoising[scenario]:
                            batch_denoising[scenario][rollout_step] = {}
                        if diffusion_step not in batch_denoising[scenario][rollout_step]:
                            batch_denoising[scenario][rollout_step][diffusion_step] = {}
                        
                        batch_denoising[scenario][rollout_step][diffusion_step][metric_type] = v
                    
                scenario_sequences[scenario] = full_sequence
            
            video_samples = self._prepare_video_samples(latents, scenario_sequences, batch)
            return batch_metrics, video_samples, batch_denoising

    def _reduce_metrics(self, metrics: Dict[str, List[float]]) -> None:
        if self.world_size <= 1 or not dist.is_initialized():
            return
        
        # Calculate local means first
        local_means = {}
        for k, v in metrics.items():
            if v:
                local_means[k] = sum(v) / len(v)
            else:
                local_means[k] = 0.0
                
        # Gather all keys
        all_keys = sorted(list(local_means.keys()))
        
        # We need to ensure all processes have the same keys for all_reduce
        # This is a bit tricky if keys are dynamic. 
        # Assuming keys are consistent across ranks for now as they depend on config.
        
        for key in all_keys:
            val = torch.tensor(local_means.get(key, 0.0), device=self.device)
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            # Average over world size
            metrics[key] = [val.item() / self.world_size]

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

    def _prepare_video_samples(
        self,
        latents: torch.Tensor,
        scenario_predictions: Dict[str, torch.Tensor],
        batch: WorldBatch,
    ) -> List[Dict[str, Union[torch.Tensor, int]]]:
        target_indices = self.config.video_sample_indices
        if not target_indices:
            return []
        
        samples: List[Dict[str, Union[torch.Tensor, int]]] = []
        batch_size = latents.shape[0]
        
        # Extract metadata from batch
        dataset_indices = batch.dataset_indices.detach().cpu().tolist()
        dataset_episode_ids = None
        if hasattr(batch, "episode_ids"):
             dataset_episode_ids = batch.episode_ids.detach().cpu().tolist()

        for batch_idx in range(batch_size):
            current_idx = dataset_indices[batch_idx]
            episode_id = None
            if dataset_episode_ids is not None:
                episode_id = dataset_episode_ids[batch_idx]
            
            match_key = episode_id if (episode_id is not None and episode_id != -1) else current_idx
            if match_key not in target_indices:
                continue
                
            entry: Dict[str, Union[torch.Tensor, int]] = {
                "index": match_key,
            }
            if not self.ground_truth_logged:
                entry["ground_truth"] = self._decode_latents(latents[batch_idx : batch_idx + 1]).squeeze(0).cpu()
            for scenario, scenario_latents in scenario_predictions.items():
                decoded = self._decode_latents(scenario_latents[batch_idx : batch_idx + 1]).squeeze(0).cpu()
                entry[scenario] = decoded
            samples.append(entry)
        return samples

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


