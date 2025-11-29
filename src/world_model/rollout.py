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
    actions_mask: Optional[torch.Tensor] = None,
    independent_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    batch_size, _, tokens, dim = latents.shape
    device = latents.device
    
    kv_cache = None
    
    # 1. Process context frames
    for t in range(context_len):
        current_frame = latents[:, t:t+1]
        
        # Add noise to reach rollout_signal_level
        noisy_frame = _apply_noise(current_frame, rollout_signal_level)
        signal = torch.full((batch_size, 1), rollout_signal_level, device=device)
        
        current_actions = actions[:, t:t+1] if actions is not None else None
        
        # Update KV cache
        _, kv_cache = model.forward_inference(
            noisy_frame,
            signal,
            kv_cache=kv_cache,
            actions=current_actions,
            independent_frames_mask=independent_mask,
            actions_mask=actions_mask
        )

    # 2. Generate future frames
    predicted_frames = []
    
    for t in range(future_len):
        # Start from pure noise
        x = torch.randn(batch_size, 1, tokens, dim, device=device)
        
        current_actions = actions[:, context_len + t : context_len + t + 1] if actions is not None else None
        
        # Denoise using solver
        clean_frame = solver.sample(
            model,
            x,
            kv_cache=kv_cache,
            actions=current_actions,
            actions_mask=actions_mask,
            independent_mask=independent_mask
        )
        
        predicted_frames.append(clean_frame)
        
        # If not the last frame, update KV cache for the next step
        if t < future_len - 1:
            noisy_next_input = _apply_noise(clean_frame, rollout_signal_level)
            signal = torch.full((batch_size, 1), rollout_signal_level, device=device)
            
            _, kv_cache = model.forward_inference(
                noisy_next_input,
                signal,
                kv_cache=kv_cache,
                actions=current_actions,
                independent_frames_mask=independent_mask,
                actions_mask=actions_mask
            )

    predicted_stack = torch.cat(predicted_frames, dim=1)
    full_sequence = torch.cat([latents[:, :context_len], predicted_stack], dim=1)
    
    return predicted_stack, full_sequence