from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.diffusion.euler_solver import EulerSolver


def _apply_noise(clean_latents: torch.Tensor, signal_level: float) -> torch.Tensor:
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
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    
    batch_size, _, tokens, dim = latents.shape
    device = latents.device
    
    kv_cache = None
    

    # Context Phase (Batched Prefill)
    if context_len > 0:
        context_frames = latents[:, :context_len]
        noisy_context = _apply_noise(context_frames, rollout_signal_level)
        context_signal = torch.full((batch_size, context_len), rollout_signal_level, device=device)
        ctx_actions = actions[:, :context_len] if actions is not None else None
        ctx_use_actions = use_actions[:, :context_len] if use_actions is not None else None
        ctx_indep = independent_frames[:, :context_len] if independent_frames is not None else None
        
        with torch.no_grad():
            output = model(
                noisy_latents=noisy_context,
                signal_levels=context_signal,
                actions=ctx_actions,
                independent_frames=ctx_indep,
                use_actions=ctx_use_actions,
                kv_cache=None
            )
        kv_cache = output.kv_cache

    predicted_frames = []
    all_metrics = {}
    
    # Future Phase (Autoregressive Loop)
    for t in range(future_len):
        x = torch.randn(batch_size, 1, tokens, dim, device=device)
        
        current_actions = actions[:, context_len + t : context_len + t + 1] if actions is not None else None
        current_use_action = use_actions[:, context_len + t : context_len + t + 1] if use_actions is not None else None
        
        target_clean_latent = None
        if target_latents is not None and denoising_metrics_indices is not None:
             if t in denoising_metrics_indices:
                 target_clean_latent = target_latents[:, t:t+1]

        clean_frame, metrics = solver.sample(
            model,
            x,
            kv_cache=kv_cache,
            actions=current_actions,
            use_actions=current_use_action,
            target_clean_latent=target_clean_latent,
            independent_frames=None,
        )
        
        if metrics:
            for k, v in metrics.items():
                all_metrics[f"step_{t}/{k}"] = v
        
        predicted_frames.append(clean_frame)
        
        # Update KV cache for next step
        if t < future_len - 1:
            noisy_next_input = _apply_noise(clean_frame, rollout_signal_level)
            signal = torch.full((batch_size, 1), rollout_signal_level, device=device)
            
            with torch.no_grad():
                output = model(
                    noisy_latents=noisy_next_input,
                    signal_levels=signal,
                    actions=current_actions,
                    independent_frames=None,
                    use_actions=current_use_action,
                    kv_cache=kv_cache
                )
            kv_cache = output.kv_cache

    predicted_stack = torch.cat(predicted_frames, dim=1)
    full_sequence = torch.cat([latents[:, :context_len], predicted_stack], dim=1)
    
    return predicted_stack, full_sequence, all_metrics