from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from rae_dino.rae import RAE
from training.world_trainer import WorldModelTrainingConfig, load_training_config
from world_model.backbone import WorldModelBackbone
from world_model.flow_matching import (
    DiffusionConfig,
    EulerSolver,
    EulerSolverConfig,
    sample_base_noise,
)

from .config import EvaluatorConfig


@dataclass
class _HistoryState:
    latents: torch.Tensor
    actions: Optional[torch.Tensor]
    actions_mask: Optional[torch.Tensor]


class WorldModelEvaluator:
    """
    High-level helper that loads a trained world model checkpoint and runs Dreamer-style
    diffusion rollouts with optional action conditioning.
    """

    def __init__(
        self,
        training_cfg: WorldModelTrainingConfig,
        model: nn.Module,
        autoencoder: RAE,
        device: Optional[torch.device] = None,
        solver_cfg: Optional[EulerSolverConfig] = None,
    ) -> None:
        self.training_cfg = training_cfg
        self.model = model.eval()
        self.autoencoder = autoencoder.eval()

        self.device = torch.device(device or training_cfg.trainer.device or "cpu")
        self.model.to(self.device)
        self.autoencoder.to(self.device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        for param in self.autoencoder.parameters():
            param.requires_grad_(False)

        self.diffusion_cfg: DiffusionConfig = training_cfg.diffusion
        self.solver_cfg = solver_cfg or EulerSolverConfig()
        self.solver = EulerSolver(self.solver_cfg)

    # --------------------------------------------------------------------- factory
    @classmethod
    def from_config(cls, cfg: EvaluatorConfig) -> "WorldModelEvaluator":
        config_path = Path(cfg.config_path)
        checkpoint_path = Path(cfg.checkpoint_path)
        training_cfg = load_training_config(config_path)

        model = WorldModelBackbone(training_cfg.world_model)
        autoencoder = cls._build_autoencoder(training_cfg)

        device = torch.device(cfg.device) if cfg.device is not None else None
        evaluator = cls(
            training_cfg=training_cfg,
            model=model,
            autoencoder=autoencoder,
            device=device,
            solver_cfg=cfg.solver,
        )
        evaluator._load_checkpoint(checkpoint_path, use_ema=cfg.use_ema_weights)
        return evaluator

    @staticmethod
    def _build_autoencoder(cfg: WorldModelTrainingConfig) -> RAE:
        vision = cfg.vision
        autoencoder = RAE()
        autoencoder.eval()
        return autoencoder

    def _load_checkpoint(self, checkpoint_path: Path, use_ema: bool = True) -> None:
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        if use_ema and "ema_model" in ckpt:
            state_dict = ckpt["ema_model"]
        else:
            state_dict = ckpt["model"]
        self.model.load_state_dict(state_dict)

    # ---------------------------------------------------------------- encode/decode
    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError("Frames must have shape [B, T, C, H, W].")
        batch, steps, channels, height, width = frames.shape
        device = self.device
        frames = frames.to(device=device, dtype=torch.float32)
        flat = frames.view(batch * steps, channels, height, width)
        latents = self.autoencoder.encode(flat)
        if latents.ndim != 3:
            raise RuntimeError(f"Autoencoder returned latents with shape {tuple(latents.shape)}")
        tokens = latents.shape[1]
        dim = latents.shape[2]
        return latents.view(batch, steps, tokens, dim)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 4:
            raise ValueError("Latents must have shape [B, T, tokens, dim].")
        batch, steps, tokens, dim = latents.shape
        device = self.device
        latents = latents.to(device=device, dtype=torch.float32)
        flat = latents.view(batch * steps, tokens, dim)
        frames = self.autoencoder.decode(flat)
        if frames.ndim != 4:
            raise RuntimeError(f"Decoder returned frames with shape {tuple(frames.shape)}")
        channels = frames.shape[1]
        height = frames.shape[2]
        width = frames.shape[3]
        return frames.view(batch, steps, channels, height, width)

    # ------------------------------------------------------------------- rollout
    @torch.no_grad()
    def rollout(
        self,
        *,
        context_frames: Optional[torch.Tensor] = None,
        context_latents: Optional[torch.Tensor] = None,
        context_actions: Optional[torch.Tensor] = None,
        context_actions_mask: Optional[torch.Tensor] = None,
        rollout_actions: Optional[torch.Tensor] = None,
        rollout_action_mask: Optional[torch.Tensor] = None,
        rollout_steps: int,
        decode: bool = True,
        independant_frames_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if context_latents is None:
            if context_frames is None:
                raise ValueError("Provide either context_frames or context_latents.")
            context_latents = self.encode_frames(context_frames)
        else:
            context_latents = context_latents.to(self.device)

        if context_latents.ndim != 4:
            raise ValueError("context_latents must have shape [B, T_ctx, tokens, dim].")

        batch, context_steps, tokens, dim = context_latents.shape
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be strictly positive.")

        action_dim = self.training_cfg.world_model.action_dim
        dtype = context_latents.dtype
        device = self.device

        context_history = _HistoryState(
            latents=context_latents,
            actions=None,
            actions_mask=None,
        )
        if context_actions is not None:
            if context_actions.shape != (batch, context_steps, action_dim):
                raise ValueError(
                    "context_actions must match shape [B, context_steps, action_dim]."
                )
            context_history.actions = context_actions.to(device=device, dtype=dtype).clone()
            if context_actions_mask is not None:
                if context_actions_mask.shape != (batch, context_steps):
                    raise ValueError(
                        "context_actions_mask must match shape [B, context_steps]."
                    )
                context_history.actions_mask = context_actions_mask.to(
                    device=device, dtype=torch.bool
                ).clone()
            else:
                context_history.actions_mask = torch.ones(
                    batch, context_steps, dtype=torch.bool, device=device
                )
        elif context_actions_mask is not None:
            raise ValueError("context_actions_mask provided but context_actions is None.")

        if independant_frames_mask is not None:
            if independant_frames_mask.shape != (batch, 1):
                raise ValueError("independant_frames_mask must have shape [B, 1].")
            independent_mask = independant_frames_mask.to(device=device, dtype=torch.bool)
        else:
            independent_mask = torch.zeros(batch, 1, dtype=torch.bool, device=device)

        generated_latents = []
        generated_actions: Optional[torch.Tensor] = None
        if rollout_actions is not None:
            if rollout_actions.shape != (batch, rollout_steps, action_dim):
                raise ValueError(
                    "rollout_actions must match shape [B, rollout_steps, action_dim]."
                )
            generated_actions = rollout_actions.to(device=device, dtype=dtype).clone()

        generated_action_mask: Optional[torch.Tensor] = None
        if rollout_action_mask is not None:
            if rollout_action_mask.shape != (batch, rollout_steps):
                raise ValueError(
                    "rollout_action_mask must match shape [B, rollout_steps]."
                )
            generated_action_mask = rollout_action_mask.to(
                device=device, dtype=torch.bool
            ).clone()
            if generated_actions is None:
                generated_actions = torch.zeros(
                    batch,
                    rollout_steps,
                    action_dim,
                    device=device,
                    dtype=dtype,
                )

        for step in range(rollout_steps):
            current_action = None
            if generated_actions is not None:
                current_action = generated_actions[:, step, :]

            current_mask = None
            if generated_action_mask is not None:
                current_mask = generated_action_mask[:, step].view(batch, 1)
            elif current_action is not None:
                current_mask = torch.ones(batch, 1, dtype=torch.bool, device=device)

            if current_mask is not None and current_action is None:
                current_action = torch.zeros(batch, action_dim, device=device, dtype=dtype)

            if context_history.actions is None and (
                current_action is not None or current_mask is not None
            ):
                history_len = context_history.latents.shape[1]
                context_history.actions = torch.zeros(
                    batch,
                    history_len,
                    action_dim,
                    device=device,
                    dtype=dtype,
                )
                context_history.actions_mask = torch.zeros(
                    batch,
                    history_len,
                    dtype=torch.bool,
                    device=device,
                )

            next_latent = self._sample_next_latent(
                history=context_history,
                current_action=current_action,
                current_action_mask=current_mask,
                action_dim=action_dim,
                independent_mask=independent_mask,
            )

            context_history.latents = torch.cat(
                (context_history.latents, next_latent), dim=1
            )

            if context_history.actions is not None:
                if current_action is None:
                    action_to_cat = torch.zeros(
                        batch,
                        1,
                        action_dim,
                        device=device,
                        dtype=context_history.actions.dtype,
                    )
                else:
                    action_to_cat = current_action.view(batch, 1, action_dim)

                if current_mask is None:
                    mask_to_cat = torch.ones(
                        batch,
                        1,
                        dtype=torch.bool,
                        device=device,
                    ) if current_action is not None else torch.zeros(
                        batch,
                        1,
                        dtype=torch.bool,
                        device=device,
                    )
                else:
                    mask_to_cat = current_mask

                context_history.actions = torch.cat(
                    (context_history.actions, action_to_cat),
                    dim=1,
                )
                context_history.actions_mask = torch.cat(
                    (context_history.actions_mask, mask_to_cat),
                    dim=1,
                )

            generated_latents.append(next_latent)

        generated_latents_tensor = torch.cat(generated_latents, dim=1)
        all_latents = context_history.latents

        output: Dict[str, torch.Tensor] = {
            "context_latents": context_latents,
            "generated_latents": generated_latents_tensor,
            "all_latents": all_latents,
        }

        if decode:
            decoded_generated = self.decode_latents(generated_latents_tensor).cpu()
            output["generated_frames"] = decoded_generated
            if context_frames is not None:
                all_frames = torch.cat(
                    (context_frames.cpu(), decoded_generated), dim=1
                )
            else:
                all_frames = self.decode_latents(all_latents).cpu()
            output["all_frames"] = all_frames

        return output

    def _sample_next_latent(
        self,
        *,
        history: _HistoryState,
        current_action: Optional[torch.Tensor],
        current_action_mask: Optional[torch.Tensor],
        action_dim: int,
        independent_mask: torch.Tensor,
    ) -> torch.Tensor:
        device = self.device
        dtype = history.latents.dtype
        batch = history.latents.shape[0]
        current_len = history.latents.shape[1]
        tokens = history.latents.shape[2]
        dim = history.latents.shape[3]

        base_latents = torch.zeros(
            batch,
            1,
            tokens,
            dim,
            device=device,
            dtype=dtype,
        )
        noisy_latents = sample_base_noise(base_latents, self.diffusion_cfg)

        context_noise = torch.full(
            (batch, current_len),
            fill_value=self.solver_cfg.max_signal,
            device=device,
            dtype=dtype,
        )

        action_mask_tail: Optional[torch.Tensor] = None
        if current_action is not None:
            current_action = current_action.to(device=device, dtype=dtype)
        if current_action_mask is not None:
            action_mask_tail = current_action_mask.to(device=device, dtype=torch.bool)

        history_actions = history.actions
        history_action_mask = history.actions_mask

        def conditioned_forward(
            latents: torch.Tensor,
            noise_levels: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            full_latents = torch.cat((history.latents, latents), dim=1)
            full_noise = torch.cat((context_noise, noise_levels), dim=1)

            full_actions: Optional[torch.Tensor] = None
            full_mask: Optional[torch.Tensor] = None
            action_tail: Optional[torch.Tensor] = None
            mask_tail: Optional[torch.Tensor] = action_mask_tail

            if current_action is not None:
                action_tail = current_action.view(batch, 1, action_dim)
            elif mask_tail is not None:
                action_tail = torch.zeros(
                    batch,
                    1,
                    action_dim,
                    device=device,
                    dtype=dtype,
                )

            if history_actions is not None:
                actions = history_actions
                masks = history_action_mask
                if action_tail is None:
                    action_tail = torch.zeros(
                        batch,
                        1,
                        action_dim,
                        device=device,
                        dtype=actions.dtype,
                    )
                if mask_tail is None:
                    mask_tail = torch.zeros(
                        batch,
                        1,
                        dtype=torch.bool,
                        device=device,
                    )
                full_actions = torch.cat((actions, action_tail), dim=1)
                full_mask = torch.cat((masks, mask_tail), dim=1)
            elif action_tail is not None:
                full_actions = action_tail
                if mask_tail is None:
                    mask_tail = torch.zeros(
                        batch,
                        1,
                        dtype=torch.bool,
                        device=device,
                    )
                full_mask = mask_tail

            outputs = self.model(
                full_latents,
                noise_levels=full_noise,
                actions=full_actions,
                actions_mask=full_mask,
                independant_frames_mask=independent_mask,
            )
            pred_clean = outputs.get("pred_clean_latents")
            if pred_clean is None:
                raise RuntimeError("Model output missing 'pred_clean_latents'.")
            tail_length = latents.shape[1]
            return {"pred_clean_latents": pred_clean[:, -tail_length:, :, :]}

        sampled = self.solver.sample(
            conditioned_forward,
            noisy_latents,
            initial_signal=self.solver_cfg.min_signal,
        )
        return sampled
