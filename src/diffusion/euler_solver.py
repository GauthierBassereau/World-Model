from dataclasses import dataclass
import torch
from src.diffusion.common import calculate_velocity_1_to_2
from src.dataset.common import WorldBatch

@dataclass
class EulerSolverConfig:
    number_steps: int = 50
    min_denom: float = 0.05

class EulerSolver:
    def __init__(self, config: EulerSolverConfig):
        self.config = config

    def sample(
        self,
        model,
        latents: torch.Tensor,
        kv_cache: list,
        actions: torch.Tensor = None,
        actions_mask: torch.Tensor = None,
        independent_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, _, tokens, dim = latents.shape
        device = latents.device
        
        step_indices = torch.arange(self.config.number_steps, dtype=torch.float32, device=device)
        t_steps = step_indices / self.config.number_steps
        dt = 1.0 / self.config.number_steps

        x = latents
        
        for i, t_val in enumerate(t_steps):
            t = torch.full((batch_size, 1), t_val, device=device)
            
            pred_clean, _ = model.forward_inference(
                x, 
                t, 
                kv_cache=kv_cache,
                actions=actions,
                independent_frames_mask=independent_mask,
                actions_mask=actions_mask
            )
            
            target_signal = torch.ones_like(t)
            velocity = calculate_velocity_1_to_2(
                latent_1=x, 
                latent_2=pred_clean, 
                signal_levels_1=t, 
                signal_levels_2=target_signal, 
                min_denom=self.config.min_denom
            )
            
            x = x + velocity * dt
            
        return x