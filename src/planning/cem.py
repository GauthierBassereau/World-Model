from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.diffusion.euler_solver import EulerSolver
from src.planning.action_tokens import ActionTokenBuilder
from src.planning.config import CEMConfig, PlanningConfig, VisualizationConfig
from src.planning.data import PlanningSample
from src.planning.visualization import (
    log_wandb_image,
    log_wandb_metrics,
    log_wandb_video,
    save_image,
    save_trajectory_plot,
    save_video,
)
from src.training.logger import WorldModelLogger
from src.world_model.rollout import collect_rollout_latents


@dataclass
class CEMIterationRecord:
    iteration: int
    best_cost: float
    best_latent_cost: float
    best_action_cost: float
    mean_elite_cost: float
    std_mean: float


@dataclass
class CEMResult:
    best_deltas: torch.Tensor
    best_cost: float
    best_latent_cost: float
    best_action_cost: float
    best_sequence: torch.Tensor
    iteration_records: List[CEMIterationRecord] = field(default_factory=list)


@dataclass
class PlanningResult:
    mode: str
    deltas: torch.Tensor
    planned_positions: Optional[torch.Tensor]
    final_sequence: torch.Tensor
    output_dir: Path


class CEMWorldModelPlanner:
    def __init__(
        self,
        *,
        planning_cfg: PlanningConfig,
        cem_cfg: CEMConfig,
        visualization_cfg: VisualizationConfig,
        model: nn.Module,
        autoencoder: nn.Module,
        solver: EulerSolver,
        action_builder: ActionTokenBuilder,
        logger: WorldModelLogger,
        device: torch.device,
    ) -> None:
        self.planning_cfg = planning_cfg
        self.cem_cfg = cem_cfg
        self.visualization_cfg = visualization_cfg
        self.model = model
        self.autoencoder = autoencoder
        self.solver = solver
        self.action_builder = action_builder
        self.logger = logger
        self.device = device
        self.delta_dim = action_builder.delta_dim
        precision = planning_cfg.precision.lower()
        if precision not in {"bf16", "fp32", "float32"}:
            raise ValueError("planning.precision must be one of: bf16, fp32, float32.")
        self.autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.use_autocast = precision == "bf16"
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(planning_cfg.seed))
        self._wandb_step = 0

        mode = planning_cfg.mode.lower()
        if mode not in {"classic_cem", "mpc_cem"}:
            raise ValueError("planning.mode must be either 'classic_cem' or 'mpc_cem'.")
        if cem_cfg.population_size < 2:
            raise ValueError("cem.population_size must be >= 2.")
        if not (0.0 < cem_cfg.elite_fraction <= 1.0):
            raise ValueError("cem.elite_fraction must be in (0, 1].")

    @torch.no_grad()
    def run(self, sample: PlanningSample) -> PlanningResult:
        output_dir = self._make_output_dir(sample)
        self._save_static_artifacts(sample, output_dir)

        frames = sample.frames.to(self.device, non_blocking=True)
        latents = self._encode_frames(frames)
        context_latents = latents[:, : sample.context_frames]
        goal_latent = latents[:, sample.goal_local_frame]
        context_actions = sample.context_actions.to(self.device, non_blocking=True).float()
        context_use_actions = sample.context_use_actions.to(self.device, non_blocking=True).bool()
        initial_state = None
        if sample.initial_ee_state is not None:
            initial_state = sample.initial_ee_state.to(self.device, non_blocking=True).float()

        if self.planning_cfg.mode.lower() == "classic_cem":
            return self._run_classic(
                sample=sample,
                output_dir=output_dir,
                context_latents=context_latents,
                context_actions=context_actions,
                context_use_actions=context_use_actions,
                initial_state=initial_state,
                goal_latent=goal_latent,
            )
        return self._run_mpc(
            sample=sample,
            output_dir=output_dir,
            context_latents=context_latents,
            context_actions=context_actions,
            context_use_actions=context_use_actions,
            initial_state=initial_state,
            goal_latent=goal_latent,
        )

    def _run_classic(
        self,
        *,
        sample: PlanningSample,
        output_dir: Path,
        context_latents: torch.Tensor,
        context_actions: torch.Tensor,
        context_use_actions: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        goal_latent: torch.Tensor,
    ) -> PlanningResult:
        result = self._optimize_once(
            name="classic",
            output_dir=output_dir,
            context_latents=context_latents,
            context_actions=context_actions,
            context_use_actions=context_use_actions,
            initial_state=initial_state,
            goal_latent=goal_latent,
            horizon=sample.horizon,
            reference_positions=sample.reference_ee_positions,
        )

        planned_positions = self._planned_positions(initial_state, result.best_deltas)
        self._save_final_artifacts(
            output_dir=output_dir,
            prefix="classic",
            final_sequence=result.best_sequence,
            planned_positions=planned_positions,
            reference_positions=sample.reference_ee_positions,
        )
        return PlanningResult(
            mode="classic_cem",
            deltas=result.best_deltas.detach().cpu(),
            planned_positions=planned_positions,
            final_sequence=result.best_sequence.detach().cpu(),
            output_dir=output_dir,
        )

    def _run_mpc(
        self,
        *,
        sample: PlanningSample,
        output_dir: Path,
        context_latents: torch.Tensor,
        context_actions: torch.Tensor,
        context_use_actions: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        goal_latent: torch.Tensor,
    ) -> PlanningResult:
        if initial_state is None and self.action_builder.has_state_token:
            raise ValueError("MPC planning with state-token actions needs initial_ee_state.")

        max_steps = self.planning_cfg.mpc.steps if self.planning_cfg.mpc.steps is not None else sample.target_step
        execute_steps = max(1, int(self.planning_cfg.mpc.execute_steps))
        executed = 0
        current_state = initial_state
        rolling_latents = context_latents
        rolling_actions = context_actions
        rolling_use_actions = context_use_actions
        initial_context = context_latents
        predicted_chunks: List[torch.Tensor] = []
        executed_delta_chunks: List[torch.Tensor] = []

        while executed < max_steps:
            remaining_to_goal = max(1, sample.target_step - executed)
            if self.planning_cfg.mpc.shrink_horizon_to_goal:
                horizon = min(sample.horizon, remaining_to_goal)
            else:
                horizon = sample.horizon

            result = self._optimize_once(
                name=f"mpc_step_{executed:03d}",
                output_dir=output_dir,
                context_latents=rolling_latents,
                context_actions=rolling_actions,
                context_use_actions=rolling_use_actions,
                initial_state=current_state,
                goal_latent=goal_latent,
                horizon=horizon,
                reference_positions=sample.reference_ee_positions,
            )

            take = min(execute_steps, result.best_deltas.shape[0], max_steps - executed)
            chosen_deltas = result.best_deltas[:take].contiguous()
            chosen_tokens = self.action_builder.build_future_tokens(chosen_deltas.unsqueeze(0), current_state)
            predicted_stack, _ = self._rollout_tokens(
                context_latents=rolling_latents,
                context_actions=rolling_actions,
                context_use_actions=rolling_use_actions,
                future_tokens=chosen_tokens,
            )

            predicted_chunks.append(predicted_stack)
            executed_delta_chunks.append(chosen_deltas)
            if self.action_builder.has_state_token:
                states = self.action_builder.integrate_states(current_state, chosen_deltas.unsqueeze(0))
                assert states is not None
                current_state = states[0, -1].detach()

            rolling_latents = torch.cat([rolling_latents, predicted_stack], dim=1)
            rolling_actions = torch.cat([rolling_actions, chosen_tokens], dim=1)
            chosen_use_actions = torch.ones((1, take), device=self.device, dtype=torch.bool)
            rolling_use_actions = torch.cat([rolling_use_actions, chosen_use_actions], dim=1)
            rolling_latents, rolling_actions, rolling_use_actions = self._truncate_context(
                rolling_latents,
                rolling_actions,
                rolling_use_actions,
            )
            executed += take
            self.logger.info("MPC executed %d/%d planned steps.", executed, max_steps)

        executed_deltas = torch.cat(executed_delta_chunks, dim=0) if executed_delta_chunks else torch.empty(0, self.delta_dim)
        future_latents = torch.cat(predicted_chunks, dim=1) if predicted_chunks else torch.empty_like(context_latents[:, :0])
        final_sequence = torch.cat([initial_context, future_latents], dim=1)
        planned_positions = self._planned_positions(initial_state, executed_deltas)
        if self.action_builder.has_state_token and sample.initial_ee_state is not None:
            planned_positions = self._planned_positions(sample.initial_ee_state.to(self.device).float(), executed_deltas)

        self._save_final_artifacts(
            output_dir=output_dir,
            prefix="mpc",
            final_sequence=final_sequence,
            planned_positions=planned_positions,
            reference_positions=sample.reference_ee_positions,
        )
        return PlanningResult(
            mode="mpc_cem",
            deltas=executed_deltas.detach().cpu(),
            planned_positions=planned_positions,
            final_sequence=final_sequence.detach().cpu(),
            output_dir=output_dir,
        )

    def _optimize_once(
        self,
        *,
        name: str,
        output_dir: Path,
        context_latents: torch.Tensor,
        context_actions: torch.Tensor,
        context_use_actions: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        goal_latent: torch.Tensor,
        horizon: int,
        reference_positions: Optional[torch.Tensor],
    ) -> CEMResult:
        mean = torch.zeros(horizon, self.delta_dim, device=self.device)
        std = self._config_vector(self.cem_cfg.initial_delta_std, "initial_delta_std")
        min_std = self._config_vector(self.cem_cfg.min_delta_std, "min_delta_std")
        max_abs = self._config_vector(self.cem_cfg.max_delta_abs, "max_delta_abs")
        elite_count = max(1, int(round(self.cem_cfg.population_size * self.cem_cfg.elite_fraction)))

        best_deltas: Optional[torch.Tensor] = None
        best_sequence: Optional[torch.Tensor] = None
        best_cost = float("inf")
        best_latent_cost = float("inf")
        best_action_cost = float("inf")
        records: List[CEMIterationRecord] = []

        for iteration in range(self.cem_cfg.iterations):
            samples = self._sample_population(mean, std, max_abs)
            costs, latent_costs, action_costs, iteration_best_sequence = self._evaluate_population(
                deltas=samples,
                context_latents=context_latents,
                context_actions=context_actions,
                context_use_actions=context_use_actions,
                initial_state=initial_state,
                goal_latent=goal_latent,
                return_best_sequence=True,
            )

            elite = torch.topk(costs, k=elite_count, largest=False)
            elite_samples = samples[elite.indices]
            elite_mean = elite_samples.mean(dim=0)
            elite_std = elite_samples.std(dim=0, unbiased=False).clamp_min(min_std)
            mean = self.cem_cfg.mean_momentum * mean + (1.0 - self.cem_cfg.mean_momentum) * elite_mean
            std = self.cem_cfg.std_momentum * std + (1.0 - self.cem_cfg.std_momentum) * elite_std
            std = std.clamp_min(min_std)

            iteration_best_idx = int(torch.argmin(costs).item())
            iteration_best_cost = float(costs[iteration_best_idx].item())
            iteration_latent_cost = float(latent_costs[iteration_best_idx].item())
            iteration_action_cost = float(action_costs[iteration_best_idx].item())
            if iteration_best_cost < best_cost:
                best_cost = iteration_best_cost
                best_latent_cost = iteration_latent_cost
                best_action_cost = iteration_action_cost
                best_deltas = samples[iteration_best_idx].detach().clone()
                best_sequence = iteration_best_sequence.detach().clone()

            record = CEMIterationRecord(
                iteration=iteration,
                best_cost=iteration_best_cost,
                best_latent_cost=iteration_latent_cost,
                best_action_cost=iteration_action_cost,
                mean_elite_cost=float(elite.values.mean().item()),
                std_mean=float(std.mean().item()),
            )
            records.append(record)
            self._log_iteration_metrics(name, record)
            self.logger.info(
                "%s iteration=%d best=%.6f latent=%.6f action=%.6f elite_mean=%.6f",
                name,
                iteration,
                record.best_cost,
                record.best_latent_cost,
                record.best_action_cost,
                record.mean_elite_cost,
            )

            if iteration in self.visualization_cfg.log_iterations:
                planned_positions = self._planned_positions(initial_state, samples[iteration_best_idx])
                self._save_iteration_artifacts(
                    output_dir=output_dir,
                    name=name,
                    iteration=iteration,
                    sequence=iteration_best_sequence,
                    planned_positions=planned_positions,
                    reference_positions=reference_positions,
                )

        mean_costs, mean_latent_costs, mean_action_costs, mean_sequence = self._evaluate_population(
            deltas=mean.unsqueeze(0),
            context_latents=context_latents,
            context_actions=context_actions,
            context_use_actions=context_use_actions,
            initial_state=initial_state,
            goal_latent=goal_latent,
            return_best_sequence=True,
        )
        final_mean_cost = float(mean_costs[0].item())
        if final_mean_cost < best_cost:
            best_cost = final_mean_cost
            best_latent_cost = float(mean_latent_costs[0].item())
            best_action_cost = float(mean_action_costs[0].item())
            best_deltas = mean.detach().clone()
            best_sequence = mean_sequence.detach().clone()

        if best_deltas is None or best_sequence is None:
            raise RuntimeError("CEM did not produce a valid candidate.")

        return CEMResult(
            best_deltas=best_deltas,
            best_cost=best_cost,
            best_latent_cost=best_latent_cost,
            best_action_cost=best_action_cost,
            best_sequence=best_sequence,
            iteration_records=records,
        )

    def _evaluate_population(
        self,
        *,
        deltas: torch.Tensor,
        context_latents: torch.Tensor,
        context_actions: torch.Tensor,
        context_use_actions: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        goal_latent: torch.Tensor,
        return_best_sequence: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        deltas = deltas.to(device=self.device, dtype=torch.float32)
        population = deltas.shape[0]
        costs: List[torch.Tensor] = []
        latent_costs: List[torch.Tensor] = []
        action_costs: List[torch.Tensor] = []
        best_cost = float("inf")
        best_sequence: Optional[torch.Tensor] = None
        batch_size = max(1, int(self.cem_cfg.rollout_batch_size))

        for start in range(0, population, batch_size):
            end = min(start + batch_size, population)
            chunk_deltas = deltas[start:end]
            future_tokens = self.action_builder.build_future_tokens(chunk_deltas, initial_state)
            predicted_stack, full_sequence = self._rollout_tokens(
                context_latents=context_latents,
                context_actions=context_actions,
                context_use_actions=context_use_actions,
                future_tokens=future_tokens,
            )
            terminal = predicted_stack[:, -1].float()
            goal = goal_latent.to(device=terminal.device, dtype=terminal.dtype).expand_as(terminal)
            chunk_latent_cost = (terminal - goal).pow(2).mean(dim=(1, 2))
            chunk_action_cost = chunk_deltas.pow(2).mean(dim=(1, 2))
            chunk_cost = chunk_latent_cost + self.cem_cfg.action_l2_weight * chunk_action_cost

            costs.append(chunk_cost)
            latent_costs.append(chunk_latent_cost)
            action_costs.append(chunk_action_cost)

            if return_best_sequence:
                local_idx = int(torch.argmin(chunk_cost).item())
                local_cost = float(chunk_cost[local_idx].item())
                if local_cost < best_cost:
                    best_cost = local_cost
                    best_sequence = full_sequence[local_idx : local_idx + 1].detach()

        all_costs = torch.cat(costs, dim=0)
        all_latent_costs = torch.cat(latent_costs, dim=0)
        all_action_costs = torch.cat(action_costs, dim=0)
        if best_sequence is None:
            best_sequence = torch.empty(0, device=self.device)
        return all_costs, all_latent_costs, all_action_costs, best_sequence

    def _rollout_tokens(
        self,
        *,
        context_latents: torch.Tensor,
        context_actions: torch.Tensor,
        context_use_actions: torch.Tensor,
        future_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = future_tokens.shape[0]
        repeated_latents = context_latents.expand(batch_size, -1, -1, -1).contiguous()
        repeated_actions = context_actions.expand(batch_size, -1, -1).contiguous()
        repeated_use_actions = context_use_actions.expand(batch_size, -1).contiguous()
        actions = torch.cat([repeated_actions, future_tokens], dim=1)
        future_use_actions = torch.ones(
            batch_size,
            future_tokens.shape[1],
            device=future_tokens.device,
            dtype=torch.bool,
        )
        use_actions = torch.cat([repeated_use_actions, future_use_actions], dim=1)

        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_autocast):
            predicted_stack, full_sequence, _ = collect_rollout_latents(
                self.model,
                self.solver,
                repeated_latents,
                context_len=context_latents.shape[1],
                future_len=future_tokens.shape[1],
                rollout_signal_level=self.planning_cfg.rollout_signal_level,
                actions=actions,
                use_actions=use_actions,
                independent_frames=None,
                target_latents=None,
                denoising_metrics_indices=None,
            )
        return predicted_stack.float(), full_sequence.float()

    def _sample_population(self, mean: torch.Tensor, std: torch.Tensor, max_abs: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(
            self.cem_cfg.population_size,
            mean.shape[0],
            mean.shape[1],
            device=self.device,
            generator=self.generator,
        )
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        samples = torch.clamp(samples, min=-max_abs.view(1, 1, -1), max=max_abs.view(1, 1, -1))
        if self.cem_cfg.include_zero_sequence:
            samples[0].zero_()
        if samples.shape[0] > 1:
            samples[1] = torch.clamp(mean, min=-max_abs.view(1, -1), max=max_abs.view(1, -1))
        return samples

    def _planned_positions(
        self,
        initial_state: Optional[torch.Tensor],
        deltas: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.action_builder.has_state_token:
            return None
        states = self.action_builder.integrate_states(initial_state, deltas.unsqueeze(0))
        if states is None:
            return None
        return states[0, :, :3].detach().cpu()

    def _config_vector(self, values: List[float], name: str) -> torch.Tensor:
        tensor = torch.tensor(values, device=self.device, dtype=torch.float32)
        if tensor.numel() == 1:
            tensor = tensor.expand(self.delta_dim)
        if tensor.numel() != self.delta_dim:
            raise ValueError(f"cem.{name} must have length 1 or {self.delta_dim}, got {tensor.numel()}.")
        return tensor.view(1, self.delta_dim).expand(-1, -1).reshape(self.delta_dim)

    def _truncate_context(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        use_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_context = self.planning_cfg.max_context_frames
        if max_context is None or latents.shape[1] <= max_context:
            return latents, actions, use_actions
        truncated_use_actions = use_actions[:, -max_context:].clone()
        truncated_use_actions[:, 0] = False
        return latents[:, -max_context:], actions[:, -max_context:], truncated_use_actions

    @torch.no_grad()
    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.float() / 255.0
        batch, steps, channels, height, width = frames.shape
        flat = frames.view(batch * steps, channels, height, width)
        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_autocast):
            latents = self.autoencoder.encode(flat)
        tokens, dim = latents.shape[1], latents.shape[2]
        return latents.view(batch, steps, tokens, dim).float()

    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch, steps, tokens, dim = latents.shape
        flat = latents.view(batch * steps, tokens, dim)
        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_autocast):
            frames = self.autoencoder.decode(flat)
        channels, height, width = frames.shape[1:]
        return frames.view(batch, steps, channels, height, width).float()

    def _make_output_dir(self, sample: PlanningSample) -> Path:
        start_cs = int(round(self.planning_cfg.start_time_s * 100.0))
        goal_cs = int(round(self.planning_cfg.goal_time_s * 100.0))
        output_dir = (
            Path(self.planning_cfg.output_dir)
            / f"episode_{sample.episode_index:04d}"
            / f"{self.planning_cfg.mode}_start_{start_cs:05d}_goal_{goal_cs:05d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_static_artifacts(self, sample: PlanningSample, output_dir: Path) -> None:
        if self.visualization_cfg.save_real_video:
            path = output_dir / "real_video_start_to_goal.mp4"
            saved = save_video(sample.real_video_frames, path, self.visualization_cfg.fps, self.logger)
            if saved is not None:
                log_wandb_video(
                    self.logger,
                    "planning/reference/real_video_start_to_goal",
                    sample.real_video_frames,
                    self.visualization_cfg.fps,
                    step=self._next_wandb_step(),
                )
        if self.visualization_cfg.save_goal_image:
            path = output_dir / "goal_image.png"
            save_image(sample.goal_image, path)
            log_wandb_image(self.logger, "planning/reference/goal_image", path, step=self._next_wandb_step())

    def _save_iteration_artifacts(
        self,
        *,
        output_dir: Path,
        name: str,
        iteration: int,
        sequence: torch.Tensor,
        planned_positions: Optional[torch.Tensor],
        reference_positions: Optional[torch.Tensor],
    ) -> None:
        iteration_dir = output_dir / name / f"iteration_{iteration:03d}"
        if self.visualization_cfg.decode_best_samples:
            decoded = self._decode_latents(sequence).squeeze(0).detach().cpu()
            video_path = iteration_dir / "best_decoded_rollout.mp4"
            saved = save_video(decoded, video_path, self.visualization_cfg.fps, self.logger)
            if saved is not None:
                log_wandb_video(
                    self.logger,
                    f"planning/{name}/iteration_{iteration:03d}/best_decoded_rollout",
                    decoded,
                    self.visualization_cfg.fps,
                    step=self._next_wandb_step(),
                )
        if self.visualization_cfg.save_trajectory_plots and planned_positions is not None:
            plot_path = save_trajectory_plot(
                planned_positions=planned_positions,
                reference_positions=reference_positions,
                path=iteration_dir / "ee_trajectory.png",
                title=f"{name} iteration {iteration}",
            )
            log_wandb_image(
                self.logger,
                f"planning/{name}/iteration_{iteration:03d}/ee_trajectory",
                plot_path,
                step=self._next_wandb_step(),
            )

    def _save_final_artifacts(
        self,
        *,
        output_dir: Path,
        prefix: str,
        final_sequence: torch.Tensor,
        planned_positions: Optional[torch.Tensor],
        reference_positions: Optional[torch.Tensor],
    ) -> None:
        final_dir = output_dir / prefix / "final"
        if self.visualization_cfg.save_planned_video:
            decoded = self._decode_latents(final_sequence).squeeze(0).detach().cpu()
            video_path = final_dir / "planned_decoded_rollout.mp4"
            saved = save_video(decoded, video_path, self.visualization_cfg.fps, self.logger)
            if saved is not None:
                log_wandb_video(
                    self.logger,
                    f"planning/{prefix}/final/planned_decoded_rollout",
                    decoded,
                    self.visualization_cfg.fps,
                    step=self._next_wandb_step(),
                )
        if self.visualization_cfg.save_trajectory_plots and planned_positions is not None:
            plot_path = save_trajectory_plot(
                planned_positions=planned_positions,
                reference_positions=reference_positions,
                path=final_dir / "ee_trajectory.png",
                title=f"{prefix} final trajectory",
            )
            log_wandb_image(
                self.logger,
                f"planning/{prefix}/final/ee_trajectory",
                plot_path,
                step=self._next_wandb_step(),
            )

    def _log_iteration_metrics(self, name: str, record: CEMIterationRecord) -> None:
        payload: Dict[str, float] = {
            f"planning/{name}/best_cost": record.best_cost,
            f"planning/{name}/best_latent_cost": record.best_latent_cost,
            f"planning/{name}/best_action_cost": record.best_action_cost,
            f"planning/{name}/mean_elite_cost": record.mean_elite_cost,
            f"planning/{name}/std_mean": record.std_mean,
            f"planning/{name}/iteration": float(record.iteration),
        }
        log_wandb_metrics(self.logger, payload, step=self._next_wandb_step())

    def _next_wandb_step(self) -> int:
        step = self._wandb_step
        self._wandb_step += 1
        return step
