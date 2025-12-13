from dataclasses import dataclass
from typing import Any
import torch
from src.diffusion.common import calculate_velocity_1_to_2
from src.dataset.common import WorldBatch
from src.diffusion.signal_scheduler import SignalScheduler, SignalSchedulerConfig

@dataclass
class EulerSolverConfig:
    number_steps: int = 50
    min_denom: float = 0.05
    timestep_schedule: str = "signal" # "signal" or "linear"

    def __post_init__(self):
        self.timestep_schedule = self.timestep_schedule.lower()
        if self.timestep_schedule not in ["linear", "signal"]:
            raise ValueError(f"Unknown timestep_schedule: {self.timestep_schedule}")

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
        batch_size, _, tokens, dim = latents.shape
        device = latents.device
        
        if self.config.timestep_schedule == "linear":
            times = torch.linspace(0.0, 1.0, self.config.number_steps + 1)
        elif self.config.timestep_schedule == "signal":
            times = self.signal_scheduler.get_timesteps(self.config.number_steps)

        x = latents
        # Step -> {latents: Tensor, l1: float, l2: float}
        denoising_data = {}
        
        for i, t_curr in enumerate(times[:-1]):
            t_next = times[i+1]
            dt = t_next - t_curr
            t_input = torch.full((batch_size, 1), t_curr, device=device)
            
            output = model(
                x, 
                t_input,
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
            
            x = x + velocity * dt
            
            # Capture data if this step is requested
            if denoising_indices is not None and i in denoising_indices:
                step_data = {}
                # Capture the first batch element's latent
                # x is [B, 1, tokens, dim], we want [1, tokens, dim] -> [tokens, dim]
                step_data["latents"] = x[0, 0].detach().cpu()
                
                if target_clean_latent is not None:
                    # target_clean_latent is [B, 1, tokens, dim]
                    diff = x[0, 0] - target_clean_latent[0, 0]
                    l1 = torch.abs(diff).mean().item()
                    l2 = torch.sqrt((diff ** 2).mean()).item()
                    step_data["l1"] = l1
                    step_data["l2"] = l2
                
                denoising_data[i] = step_data
            
        return x, denoising_data