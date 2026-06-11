from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

from src.diffusion.euler_solver import EulerSolver
from src.world_model.utils import expand_signal_levels_for_model, global_signal_levels_from_signal


def _apply_rollout_signal(
    clean_latents: torch.Tensor,
    signal_level: float,
    add_noise: bool,
) -> torch.Tensor:
    if not add_noise:
        return signal_level * clean_latents

    noise = torch.randn_like(clean_latents)
    return signal_level * clean_latents + (1.0 - signal_level) * noise


def rollout_latents(
    model: nn.Module,
    solver: EulerSolver,
    latents: torch.Tensor,
    context_len: int,
    future_len: int,
    rollout_signal_level: float,
    actions: Optional[torch.Tensor] = None,
    use_actions: Optional[torch.Tensor] = None,
    independent_frames: Optional[torch.Tensor] = None,
    target_latents: Optional[torch.Tensor] = None,
    denoising_metrics_indices: Optional[List[int]] = None,
    rollout_add_noise: bool = True,
):
    batch_size, _, tokens, dim = latents.shape
    device = latents.device
    
    kv_cache = None

    if context_len > 0:
        context_frames = latents[:, :context_len]
        context_signal = torch.ones((batch_size, context_len), device=device, dtype=latents.dtype)
        context_signal = expand_signal_levels_for_model(model, context_signal, tokens)
        ctx_actions = actions[:, :context_len] if actions is not None else None
        ctx_use_actions = use_actions[:, :context_len] if use_actions is not None else None
        ctx_indep = independent_frames[:, :context_len] if independent_frames is not None else None
        
        with torch.no_grad():
            output = model(
                noisy_latents=context_frames,
                signal_levels=context_signal,
                global_signal_levels=global_signal_levels_from_signal(context_signal),
                actions=ctx_actions,
                independent_frames=ctx_indep,
                use_actions=ctx_use_actions,
                kv_cache=None
            )
        kv_cache = output.kv_cache

    for t in range(future_len):
        x = torch.randn(batch_size, 1, tokens, dim, device=device)
        
        current_actions = actions[:, context_len + t : context_len + t + 1] if actions is not None else None
        current_use_action = use_actions[:, context_len + t : context_len + t + 1] if use_actions is not None else None
        
        current_denoising_indices = None
        target_clean_latent = None
        
        if t == 0:
            if denoising_metrics_indices is not None:
                current_denoising_indices = denoising_metrics_indices
                if target_latents is not None:
                    target_clean_latent = target_latents[:, t:t+1]

        clean_frame, step_denoising_data = solver.sample(
            model,
            x,
            kv_cache=kv_cache,
            actions=current_actions,
            use_actions=current_use_action,
            target_clean_latent=target_clean_latent,
            independent_frames=None,
            denoising_indices=current_denoising_indices,
        )
        
        yield t, clean_frame, step_denoising_data
        
        if t < future_len - 1:
            noisy_next_input = _apply_rollout_signal(
                clean_frame,
                rollout_signal_level,
                rollout_add_noise,
            )
            signal = torch.full((batch_size, 1), rollout_signal_level, device=device, dtype=clean_frame.dtype)
            signal = expand_signal_levels_for_model(model, signal, tokens)
            
            with torch.no_grad():
                output = model(
                    noisy_latents=noisy_next_input,
                    signal_levels=signal,
                    global_signal_levels=global_signal_levels_from_signal(signal),
                    actions=current_actions,
                    independent_frames=None,
                    use_actions=current_use_action,
                    kv_cache=kv_cache
                )
            kv_cache = output.kv_cache


def collect_rollout_latents(
    model: nn.Module,
    solver: EulerSolver,
    latents: torch.Tensor,
    context_len: int,
    future_len: int,
    rollout_signal_level: float,
    actions: Optional[torch.Tensor] = None,
    use_actions: Optional[torch.Tensor] = None,
    independent_frames: Optional[torch.Tensor] = None,
    target_latents: Optional[torch.Tensor] = None,
    denoising_metrics_indices: Optional[List[int]] = None,
    rollout_add_noise: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, Any]]]:
    predicted_frames = []
    denoising_data_all = {}
    
    for t, clean_frame, step_denoising_data in rollout_latents(
        model,
        solver,
        latents,
        context_len,
        future_len,
        rollout_signal_level,
        actions,
        use_actions,
        independent_frames,
        target_latents,
        denoising_metrics_indices,
        rollout_add_noise=rollout_add_noise,
    ):
        predicted_frames.append(clean_frame)
        if step_denoising_data:
            denoising_data_all = step_denoising_data

    predicted_stack = torch.cat(predicted_frames, dim=1)
    full_sequence = torch.cat([latents[:, :context_len], predicted_stack], dim=1)
    
    return predicted_stack, full_sequence, denoising_data_all
