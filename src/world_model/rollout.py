from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.world_model.diffusion import DiffusionConfig, EulerSolver, sample_base_noise


def _build_model_kwargs(
    actions: Optional[torch.Tensor],
    actions_mask: Optional[torch.Tensor],
    independent_mask: Optional[torch.Tensor],
    length: int,
    use_actions: bool,
) -> Dict[str, torch.Tensor]:
    kwargs: Dict[str, torch.Tensor] = {}
    if independent_mask is not None:
        kwargs["independent_frames_mask"] = independent_mask[:, :length]
    if use_actions and actions is not None:
        kwargs["actions"] = actions[:, :length]
        if actions_mask is not None:
            kwargs["actions_mask"] = actions_mask[:, :length]
    return kwargs


def rollout_latents(
    model: nn.Module,
    solver: EulerSolver,
    latents: torch.Tensor,
    *,
    flow_cfg: DiffusionConfig,
    context_len: int,
    future_len: int,
    rollout_signal_level: float,
    use_actions: bool,
    actions: Optional[torch.Tensor] = None,
    actions_mask: Optional[torch.Tensor] = None,
    independent_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run autoregressive rollouts on latent sequences.

    Returns (predicted_future, full_sequence_with_predictions).
    """
    if latents.ndim != 4:
        raise ValueError("Expected latents to have shape [B, T, tokens, dim].")
    batch_size, seq_len = latents.shape[0], latents.shape[1]
    if context_len < 1 or context_len > seq_len:
        raise ValueError("context_len must be in [1, sequence_length].")
    if future_len < 0 or context_len + future_len > seq_len:
        raise ValueError("future_len must satisfy context_len + future_len <= sequence_length.")

    observed_latents = latents[:, :context_len, ...].detach()
    if future_len == 0:
        empty = torch.zeros_like(observed_latents[:, :0])
        return empty, observed_latents

    generated_frames = []
    predictions = []

    for step in range(future_len):
        past_generated = torch.cat(generated_frames, dim=1) if generated_frames else None
        noise_frame = sample_base_noise(
            latents[:, context_len + step : context_len + step + 1, ...],
            flow_cfg,
        )

        # Noise the context and past generated frames
        # We use the same noise logic for both: signal * clean + (1 - signal) * noise
        
        # 1. Context
        context_noise = sample_base_noise(observed_latents, flow_cfg)
        noised_context = rollout_signal_level * observed_latents + (1 - rollout_signal_level) * context_noise
        
        seq_latents_parts = [noised_context]
        signal_parts = [
            torch.full(
                (batch_size, context_len),
                rollout_signal_level,
                dtype=latents.dtype,
                device=latents.device,
            )
        ]

        # 2. Past generated
        if past_generated is not None:
            past_gen_noise = sample_base_noise(past_generated, flow_cfg)
            noised_past_gen = rollout_signal_level * past_generated + (1 - rollout_signal_level) * past_gen_noise
            
            seq_latents_parts.append(noised_past_gen)
            signal_parts.append(
                torch.full(
                    (batch_size, past_generated.shape[1]),
                    rollout_signal_level,
                    dtype=latents.dtype,
                    device=latents.device,
                )
            )

        seq_latents_parts.append(noise_frame)
        signal_parts.append(
            torch.zeros(
                (batch_size, 1),
                dtype=latents.dtype,
                device=latents.device,
            )
        )

        seq_latents_tensor = torch.cat(seq_latents_parts, dim=1)
        signal_tensor = torch.cat(signal_parts, dim=1)

        length = seq_latents_tensor.shape[1]
        model_kwargs = _build_model_kwargs(
            actions,
            actions_mask,
            independent_mask,
            length,
            use_actions,
        )

        rollout = solver.sample(
            model,
            seq_latents_tensor,
            initial_signal=signal_tensor,
            **model_kwargs,
        )

        if past_generated is not None:
            updated = rollout[:, context_len : context_len + past_generated.shape[1], ...].detach()
            for idx in range(len(generated_frames)):
                generated_frames[idx] = updated[:, idx : idx + 1, ...]

        pred_frame = rollout[:, -1:, ...].detach()
        generated_frames.append(pred_frame)
        predictions.append(pred_frame)

    predicted_stack = torch.cat(predictions, dim=1)
    full_sequence = torch.cat([observed_latents, torch.cat(generated_frames, dim=1)], dim=1)
    return predicted_stack, full_sequence
