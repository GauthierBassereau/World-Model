from dataclasses import dataclass

import math
from typing import Optional, Union

import torch

# ------------------------------------------------------------------ Scheduler and Noise sampler
@dataclass
class DiffusionConfig:
    base_dimension: int = 4_096
    effective_latent_dimension: int = 196_608 # 768*16*16
    noise_mean: float = 0.0
    noise_std: float = 1.0
    signal_weighting: str = "dimension"  # {"dimension", "linear", "none"}
    linear_weight_slope: float = 0.9
    linear_weight_intercept: float = 0.1

    def validate(self) -> None:
        if self.base_dimension <= 0:
            raise ValueError("diffusion.base_dimension must be strictly positive.")
        std_tensor = torch.as_tensor(self.noise_std, dtype=torch.float32)
        if torch.any(std_tensor <= 0):
            raise ValueError("diffusion.noise_std must be strictly positive.")
        mode = self.signal_weighting.lower()
        if mode not in {"dimension", "linear", "none"}:
            raise ValueError(
                "diffusion.signal_weighting must be one of {'dimension', 'linear', 'none'}."
            )
        if mode == "linear":
            intercept = float(self.linear_weight_intercept)
            slope = float(self.linear_weight_slope)
            if intercept < 0.0:
                raise ValueError("Linear weighting intercept must be non-negative.")
            if intercept + slope < 0.0:
                raise ValueError("Linear weighting must remain non-negative over [0, 1].")


class DimensionShiftedUniformScheduler:
    """
    Uniform signal sampler with optional importance weights matching the DiT-RAE schedule.

    The scheduler draws baseline values from U(0, 1), keeps them in that range, and returns
    per-sample weights so downstream losses can be reweighted without resampling.
    """
    def __init__(self, config: DiffusionConfig) -> None:
        self.config = config
        self.alpha = math.sqrt(float(config.effective_latent_dimension) / float(self.config.base_dimension))
        self._weighting_mode = config.signal_weighting.lower()
        self._linear_slope = float(config.linear_weight_slope)
        self._linear_intercept = float(config.linear_weight_intercept)

    def sample(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, tokens, dim = latents.shape
        device = latents.device
        dtype = latents.dtype

        base = torch.rand((batch, steps), dtype=torch.float32, device=device)

        weights = self._compute_weights(base)
        weights = weights.to(device=device, dtype=dtype)
        return base.to(device=device, dtype=dtype), weights

    def _compute_weights(self, base: torch.Tensor) -> torch.Tensor:
        mode = self._weighting_mode
        if mode == "none":
            return torch.ones_like(base)
        if mode == "linear":
            weights = self._linear_intercept + self._linear_slope * base
            return torch.clamp(weights, min=0.0)
        if mode == "dimension":
            if math.isclose(self.alpha, 1.0):
                return torch.ones_like(base)
            denom = self.alpha - (self.alpha - 1.0) * base
            denom = torch.clamp(denom, min=1e-8)
            weights = self.alpha / (denom * denom)
            return weights
        raise RuntimeError(f"Unsupported weighting mode: {mode}")


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
                raise TypeError("Model is expected to return a dict with a 'pred_clean_latents' entry.")
            pred_clean = outputs.get("pred_clean_latents")
            if pred_clean is None:
                raise KeyError("Model output is missing required 'pred_clean_latents'.")
            if pred_clean.shape != current_latents.shape:
                raise ValueError(
                    "Model returned 'pred_clean_latents' with shape "
                    f"{tuple(pred_clean.shape)}, expected {tuple(current_latents.shape)}."
                )
            if pred_clean.dtype != dtype:
                pred_clean = pred_clean.to(dtype)

            denom = 1.0 - noise_levels
            valid = denom > 1e-5
            denom = torch.where(valid, denom, torch.ones_like(denom))
            denom = denom.unsqueeze(-1).unsqueeze(-1)
            velocity = (pred_clean - current_latents) / denom
            velocity = torch.where(
                valid.unsqueeze(-1).unsqueeze(-1),
                velocity,
                torch.zeros_like(velocity),
            )

            delta = (target_signal - current_signal).clamp(min=0.0, max=cfg.step_size)
            delta_factor = delta.unsqueeze(-1).unsqueeze(-1)
            current_latents = current_latents + delta_factor * velocity
            current_signal = current_signal + delta

        return current_latents
