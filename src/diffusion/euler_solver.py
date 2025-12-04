from dataclasses import dataclass
import torch
from src.diffusion.common import calculate_velocity_1_to_2
from src.dataset.common import WorldBatch
from src.diffusion.signal_scheduler import SignalScheduler, SignalSchedulerConfig

@dataclass
class EulerSolverConfig:
    number_steps: int = 50
    min_denom: float = 0.05

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
    ) -> tuple[torch.Tensor, dict[str, float]]:
        batch_size, _, tokens, dim = latents.shape
        device = latents.device
        
        times = self.signal_scheduler.get_timesteps(self.config.number_steps)

        x = latents
        metrics = {}
        
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
            
            # for logging per timestep metrics
            if target_clean_latent is not None:
                diff = output.latents - target_clean_latent
                l1 = torch.abs(diff).mean().item()
                l2 = torch.sqrt((diff ** 2).mean()).item()
                metrics[f"l1_step_{t_curr.item():.4f}"] = l1
                metrics[f"l2_step_{t_curr.item():.4f}"] = l2
            
        return x, metrics