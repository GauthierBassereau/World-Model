import math
import torch
from dataclasses import dataclass

@dataclass
class SignalSchedulerConfig:
    mode: str = "resolution_shift"  # {"resolution_shift", "linear_shift", "uniform"}
    min_value: float = 0.0
    max_value: float = 1.0
    # resolution_shift parameters
    resolution_shift_base_dimension: int = 4_096
    resolution_shift_effective_latent_dimension: int = 196_608 # 768*16*16
    # linear_shift parameters
    linear_shift_slope: float = -0.9
    linear_shift_intercept: float = 0.1

class SignalScheduler:
    def __init__(self, config: SignalSchedulerConfig) -> None:
        self.config = config
        self._mode = config.mode.lower()
        
        if self._mode == "resolution_shift":
            self.alpha = math.sqrt(float(config.resolution_shift_base_dimension) / float(config.resolution_shift_effective_latent_dimension)) # this is inverse of the noise shift from RAE paper
        elif self._mode == "linear_shift":
            self.slope = torch.as_tensor(config.linear_shift_slope, dtype=torch.float32)
            self.intercept = torch.as_tensor(config.linear_shift_intercept, dtype=torch.float32)

    def sample(self, latents: torch.Tensor) -> torch.Tensor:
        batch, steps, tokens, dim = latents.shape

        base = torch.rand((batch, steps), dtype=torch.float32)
        signal_level = self._sample_from_mode(base)
        signal_level = signal_level * (self.config.max_value - self.config.min_value) + self.config.min_value
        return signal_level.to(device=latents.device, dtype=latents.dtype)

    def get_timesteps(self, num_steps: int) -> torch.Tensor:
        linear_t = torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float32)
        return self._sample_from_mode(linear_t)

    def _sample_from_mode(self, base: torch.Tensor) -> torch.Tensor:
        mode = self._mode
        if mode == "uniform":
            return base
        if mode == "linear_shift":
            return self.slope * base + self.intercept
        if mode == "resolution_shift":
            return self._sample_signal(base)
        raise RuntimeError(f"Unsupported noise sampling mode: {mode}")

    def _sample_signal(self, base: torch.Tensor) -> torch.Tensor:
        if math.isclose(self.alpha, 1.0):
            return base
        alpha = torch.as_tensor(self.alpha, dtype=base.dtype, device=base.device)
        denom = 1.0 + (alpha - 1.0) * base
        denom = torch.clamp_min(denom, 1e-8)
        return (alpha * base) / denom