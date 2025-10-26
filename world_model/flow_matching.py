from dataclasses import dataclass

import math
from typing import Optional, Union

import torch

# ------------------------------------------------------------------ Scheduler and Noise sampler
@dataclass
class DiffusionConfig:
    min_signal: float = 0.0
    max_signal: float = 1.0
    base_dimension: int = 4_096
    effective_latent_dimension: int = 196_608 # 768*16*16
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


class DimensionShiftedUniformScheduler:
    """
    Uniform signal sampler with the dimension-dependent shift introduced in DiT-RAE.

    The scheduler draws baseline values from U(0, 1) and shifts them according to the effective
    latent dimensionality before mapping to [min_signal, max_signal].
    """
    def __init__(self, config: DiffusionConfig) -> None:
        self.config = config
        self._min_signal = torch.tensor(config.min_signal, dtype=torch.float32)
        self._max_signal = torch.tensor(config.max_signal, dtype=torch.float32)
        self.alpha = math.sqrt(float(config.effective_latent_dimension) / float(self.config.base_dimension))

    def sample(self, latents: torch.Tensor) -> torch.Tensor:
        batch, steps, tokens, dim = latents.shape
        device = latents.device
        dtype = latents.dtype

        base = torch.rand((batch, steps), dtype=torch.float32)

        if not math.isclose(self.alpha, 1.0):
            base = (self.alpha * base) / (1.0 + (self.alpha - 1.0) * base)

        min_signal = self._min_signal
        max_signal = self._max_signal
        shifted = base * (max_signal - min_signal) + min_signal
        return shifted.to(device=device, dtype=dtype)


def sample_base_noise(latents: torch.Tensor, config: DiffusionConfig) -> torch.Tensor:
    """Sample the base noise prior used for flow matching."""
    mean = torch.as_tensor(
        config.noise_mean,
        device=latents.device,
        dtype=latents.dtype,
    )
    std = torch.as_tensor(
        config.noise_std,
        device=latents.device,
        dtype=latents.dtype,
    )
    noise = torch.randn_like(latents)
    return noise * std + mean


# ------------------------------------------------------------------ ODE Solver
@dataclass
class EulerSolverConfig:
    step_size: float = 0.02 # 50 steps
    min_signal: float = 0.0
    max_signal: float = 1.0

    def validate(self) -> None:
        if self.step_size <= 0.0 or not math.isfinite(self.step_size):
            raise ValueError("EulerSolver.step_size must be a positive finite value.")
        if not 0.0 <= self.min_signal < self.max_signal <= 1.0:
            raise ValueError("Expected 0.0 <= min_signal < max_signal <= 1.0 for EulerSolver.")


class EulerSolver:
    def __init__(self, config: EulerSolverConfig) -> None:
        self.config = config
        self.config.validate()

    def _prepare_initial_signal(
        self,
        initial_signal: Optional[Union[float, torch.Tensor]],
        batch: int,
        time_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cfg = self.config
        if initial_signal is None:
            return torch.full(
                (batch, time_steps),
                cfg.min_signal,
                device=device,
                dtype=dtype,
            )

        signal_tensor = torch.as_tensor(initial_signal, device=device, dtype=dtype)
        if signal_tensor.ndim == 0:
            return signal_tensor.expand(batch, time_steps)
        if signal_tensor.ndim == 1:
            if signal_tensor.shape[0] != batch:
                raise ValueError(
                    "Initial signal 1D tensor must have length equal to batch size or be scalar."
                )
            return signal_tensor.unsqueeze(1).expand(batch, time_steps)
        if signal_tensor.ndim == 2:
            if signal_tensor.shape == (batch, time_steps):
                return signal_tensor
            if signal_tensor.shape == (batch, 1):
                return signal_tensor.expand(batch, time_steps)
            if signal_tensor.shape == (1, time_steps):
                return signal_tensor.expand(batch, time_steps)
            raise ValueError(
                "Initial signal 2D tensor must match (batch, time_steps), (batch, 1) or (1, time_steps)."
            )
        if signal_tensor.ndim == 3 and signal_tensor.shape[2] == 1:
            squeezed = signal_tensor.squeeze(-1)
            if squeezed.shape == (batch, time_steps):
                return squeezed
        raise ValueError(
            "Initial signal must be None, scalar, or have shape compatible with (batch, time_steps)."
        )

    def sample(
        self,
        model,
        latents: torch.Tensor,
        initial_signal: Optional[Union[float, torch.Tensor]] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        if latents.ndim != 4:
            raise ValueError("Expected latents to have shape [B, T, tokens, dim].")

        cfg = self.config
        total_range = cfg.max_signal - cfg.min_signal
        if total_range <= 0.0:
            raise ValueError("EulerSolver received zero integration range; check signal bounds.")

        batch, time_steps = latents.shape[0], latents.shape[1]
        device = latents.device
        dtype = latents.dtype

        current_signal = self._prepare_initial_signal(
            initial_signal, batch, time_steps, device, dtype
        )
        if torch.any(current_signal < cfg.min_signal) or torch.any(current_signal > cfg.max_signal):
            raise ValueError("Initial signal must lie within [min_signal, max_signal].")
        target_signal = torch.full(
            (batch, time_steps),
            cfg.max_signal,
            device=device,
            dtype=dtype,
        )

        num_steps = max(1, math.ceil(total_range / cfg.step_size))
        current_latents = latents.clone()

        for _ in range(num_steps):
            if not torch.any(current_signal < target_signal).item():
                break
            noise_levels = current_signal

            model_inputs = dict(model_kwargs)
            model_inputs["noise_levels"] = noise_levels
            outputs = model(current_latents, **model_inputs)
            if not isinstance(outputs, dict):
                raise TypeError("Model is expected to return a dict with a 'pred_velocity' entry.")
            velocity = outputs.get("pred_velocity")
            if velocity is None:
                raise KeyError("Model output is missing required 'pred_velocity'.")
            if velocity.shape != current_latents.shape:
                raise ValueError(
                    "Model returned 'pred_velocity' with shape "
                    f"{tuple(velocity.shape)}, expected {tuple(current_latents.shape)}."
                )
            if velocity.dtype != dtype:
                velocity = velocity.to(dtype)

            delta = (target_signal - current_signal).clamp(min=0.0, max=cfg.step_size)
            delta_factor = delta.unsqueeze(-1).unsqueeze(-1)
            current_latents = current_latents + delta_factor * velocity
            current_signal = current_signal + delta

        return current_latents
