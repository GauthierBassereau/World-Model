from dataclasses import dataclass

import torch


@dataclass
class DiffusionConfig:
    min_signal: float = 0.0
    max_signal: float = 1.0
    base_dimension: int = 4_096
    noise_mean: float = 0.0
    noise_std: float = 1.0

    def validate(self) -> None:
        if not 0.0 <= self.min_signal < self.max_signal <= 1.0:
            raise ValueError("Expected 0.0 <= min_signal < max_signal <= 1.0 for flow matching.")
        if self.base_dimension <= 0:
            raise ValueError("diffusion.base_dimension must be strictly positive.")
        std_tensor = torch.as_tensor(self.noise_std, dtype=torch.float32)
        if torch.any(std_tensor <= 0):
            raise ValueError("diffusion.noise_std must be strictly positive.")
