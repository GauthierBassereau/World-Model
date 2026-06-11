from dataclasses import dataclass
from typing import Any
import torch
from src.diffusion.common import calculate_velocity_1_to_2
from src.diffusion.signal_scheduler import SignalScheduler, SignalSchedulerConfig
from src.world_model.utils import (
    expand_signal_levels_for_model,
    global_signal_levels_from_signal,
    model_uses_patch_signal_conditioning,
    unwrap_world_model,
)

@dataclass
class EulerSolverConfig:
    number_steps: int = 50
    min_denom: float = 0.05
    timestep_schedule: str = "signal" # "signal" or "linear"
    sampler: str = "euler" # "euler" or "adaptive"
    adaptive_percentile: float = 0.7
    adaptive_inner_steps: int = 4

    def __post_init__(self):
        self.timestep_schedule = self.timestep_schedule.lower()
        if self.timestep_schedule not in ["linear", "signal"]:
            raise ValueError(f"Unknown timestep_schedule: {self.timestep_schedule}")
        self.sampler = self.sampler.lower()
        if self.sampler not in ["euler", "adaptive"]:
            raise ValueError(f"Unknown sampler: {self.sampler}")
        if not (0.0 < self.adaptive_percentile < 1.0):
            raise ValueError("adaptive_percentile must be in (0, 1).")
        if self.adaptive_inner_steps < 1:
            raise ValueError("adaptive_inner_steps must be >= 1.")

class EulerSolver:
    def __init__(self, config: EulerSolverConfig, signal_scheduler_cfg: SignalSchedulerConfig):
        self.config = config
        self.signal_scheduler = SignalScheduler(signal_scheduler_cfg)

    def sample(
        self,
        model,
        latents: torch.Tensor,
        kv_cache: list,
        actions: torch.Tensor = None,
        use_actions: torch.Tensor = None,
        target_clean_latent: torch.Tensor = None,
        independent_frames: torch.Tensor = None,
        denoising_indices: list[int] = None,
    ) -> tuple[torch.Tensor, dict[int, dict[str, Any]]]:
        batch_size, _, tokens, _ = latents.shape
        device = latents.device

        if self.config.timestep_schedule == "linear":
            times = torch.linspace(0.0, 1.0, self.config.number_steps + 1, device=device, dtype=latents.dtype)
        elif self.config.timestep_schedule == "signal":
            times = self.signal_scheduler.get_timesteps(self.config.number_steps).to(device=device, dtype=latents.dtype)

        x = latents
        # Step -> {latents: Tensor, l1: float, l2: float}
        denoising_data = {}

        if self.config.sampler == "adaptive":
            self._validate_adaptive_model(model)
            return self._sample_adaptive(
                model=model,
                x=x,
                times=times,
                kv_cache=kv_cache,
                actions=actions,
                use_actions=use_actions,
                target_clean_latent=target_clean_latent,
                independent_frames=independent_frames,
                denoising_indices=denoising_indices,
                denoising_data=denoising_data,
            )

        for i, t_curr in enumerate(times[:-1]):
            t_next = times[i+1]
            dt = t_next - t_curr
            t_input = torch.full((batch_size, 1), t_curr, device=device, dtype=latents.dtype)
            t_input = expand_signal_levels_for_model(model, t_input, tokens)
            
            output = model(
                x, 
                t_input,
                global_signal_levels=global_signal_levels_from_signal(t_input),
                kv_cache=kv_cache,
                actions=actions,
                use_actions=use_actions,
                independent_frames=independent_frames,
            )
            
            target_signal = torch.ones_like(t_input)
            velocity = calculate_velocity_1_to_2(
                latent_1=x, 
                latent_2=output.latents, 
                signal_levels_1=t_input, 
                signal_levels_2=target_signal,
                min_denom=self.config.min_denom
            )
            
            x = x + velocity * dt.to(dtype=x.dtype)
            
            # Capture data if this step is requested
            if denoising_indices is not None and i in denoising_indices:
                denoising_data[i] = self._build_denoising_step_data(x, target_clean_latent)
            
        return x, denoising_data

    def _sample_adaptive(
        self,
        model,
        x: torch.Tensor,
        times: torch.Tensor,
        kv_cache: list,
        actions: torch.Tensor = None,
        use_actions: torch.Tensor = None,
        target_clean_latent: torch.Tensor = None,
        independent_frames: torch.Tensor = None,
        denoising_indices: list[int] = None,
        denoising_data: dict[int, dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, dict[int, dict[str, Any]]]:
        batch_size, _, tokens, _ = x.shape
        device = x.device
        dtype = x.dtype
        denoising_data = denoising_data or {}
        inner_steps = self.config.adaptive_inner_steps
        condition_patches = model_uses_patch_signal_conditioning(model)

        for i, t_curr in enumerate(times[:-1]):
            t_next = times[i + 1]
            dt = (t_next - t_curr).to(dtype=dtype)
            t_patch = torch.full((batch_size, 1, tokens), t_curr, device=device, dtype=dtype)
            target_signal = torch.ones_like(t_patch)
            global_signal_levels = t_patch.amax(dim=-1)
            model_signal_levels = t_patch if condition_patches else global_signal_levels

            output = model(
                x,
                model_signal_levels,
                global_signal_levels=global_signal_levels,
                kv_cache=kv_cache,
                actions=actions,
                use_actions=use_actions,
                independent_frames=independent_frames,
            )
            if output.patch_logvar is None:
                raise RuntimeError("Adaptive sampler requires model outputs with patch_logvar.")

            velocity = calculate_velocity_1_to_2(
                latent_1=x,
                latent_2=output.latents,
                signal_levels_1=t_patch,
                signal_levels_2=target_signal,
                min_denom=self.config.min_denom,
            )
            easy_mask = self._easy_patch_mask(output.patch_logvar)
            full_step = torch.full_like(t_patch, dt)
            inner_step = full_step / float(inner_steps)
            step = torch.where(easy_mask, full_step, inner_step)
            x = x + velocity * step.unsqueeze(-1)
            t_patch = t_patch + step

            for _ in range(1, inner_steps):
                remaining = (t_next.to(dtype=dtype) - t_patch).clamp_min(0.0)
                step = torch.minimum(inner_step, remaining)
                global_signal_levels = t_patch.amax(dim=-1)
                model_signal_levels = t_patch if condition_patches else global_signal_levels
                output = model(
                    x,
                    model_signal_levels,
                    global_signal_levels=global_signal_levels,
                    kv_cache=kv_cache,
                    actions=actions,
                    use_actions=use_actions,
                    independent_frames=independent_frames,
                )
                velocity = calculate_velocity_1_to_2(
                    latent_1=x,
                    latent_2=output.latents,
                    signal_levels_1=t_patch,
                    signal_levels_2=target_signal,
                    min_denom=self.config.min_denom,
                )
                x = x + velocity * step.unsqueeze(-1)
                t_patch = t_patch + step

            if denoising_indices is not None and i in denoising_indices:
                denoising_data[i] = self._build_denoising_step_data(x, target_clean_latent)

        return x, denoising_data

    def _easy_patch_mask(self, patch_logvar: torch.Tensor) -> torch.Tensor:
        difficulty = patch_logvar.float()
        threshold = torch.quantile(
            difficulty,
            self.config.adaptive_percentile,
            dim=-1,
            keepdim=True,
        )
        return difficulty <= threshold.to(device=patch_logvar.device, dtype=difficulty.dtype)

    def _build_denoising_step_data(
        self,
        x: torch.Tensor,
        target_clean_latent: torch.Tensor = None,
    ) -> dict[str, Any]:
        step_data = {}
        # x is [B, 1, tokens, dim], we want [1, tokens, dim] -> [tokens, dim]
        step_data["latents"] = x[0, 0].detach().cpu()

        if target_clean_latent is not None:
            # target_clean_latent is [B, 1, tokens, dim]
            diff = x[0, 0] - target_clean_latent[0, 0]
            step_data["l1"] = torch.abs(diff).mean().item()
            step_data["l2"] = torch.sqrt((diff ** 2).mean()).item()

        return step_data

    @staticmethod
    def _validate_adaptive_model(model) -> None:
        unwrapped = unwrap_world_model(model)
        config = getattr(unwrapped, "config", None)
        if config is None:
            return
        if not getattr(config, "predict_patch_difficulty", False):
            raise ValueError("Adaptive sampler requires world_model.predict_patch_difficulty=true.")
