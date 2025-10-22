from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import RolloutConfig
from .evaluator import WorldModelEvaluator


@dataclass
class ActionBinding:
    key: str
    description: str
    vector: torch.Tensor


class InteractiveWorldModelUI:
    """
    Lightweight CLI to explore the world model by stepping through generations.

    Users type key combinations (e.g. ``wd``) to queue actions, hit ``enter`` or
    ``step`` to sample the next frame, toggle hold to reuse the last action, and
    ``save`` to export a video plus rollout tensor under ``archive/interactive_sessions``.
    """

    def __init__(
        self,
        evaluator: WorldModelEvaluator,
        rollout_cfg: Optional[RolloutConfig] = None,
    ) -> None:
        self.evaluator = evaluator
        self.device = evaluator.device
        self.rollout_cfg = rollout_cfg or RolloutConfig(context=0, horizon=1, decode=True)

        self.action_dim = evaluator.training_cfg.world_model.action_dim
        self.translation_step = 0.02
        self.rotation_step = 0.05

        self._bindings: Dict[str, ActionBinding] = {}
        self._build_default_bindings()

        self.context_frames: Optional[torch.Tensor] = None
        self.history_latents: Optional[torch.Tensor] = None
        self.history_actions: Optional[torch.Tensor] = None
        self.history_action_mask: Optional[torch.Tensor] = None

        self.generated_latents: List[torch.Tensor] = []
        self.generated_frames: List[torch.Tensor] = []

        self.pending_action = torch.zeros(self.action_dim, device=self.device)
        self.apply_action = False
        self.hold_action = False
        self.step_count = 0

    # ------------------------------------------------------------------ setup
    def load_context(
        self,
        frames: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        actions_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if frames.ndim == 4:
            frames = frames.unsqueeze(0)
        if frames.ndim != 5:
            raise ValueError("Context frames must have shape [T, C, H, W] or [B, T, C, H, W].")
        if frames.shape[0] != 1:
            raise ValueError("Interactive UI currently supports a single sequence (batch size = 1).")

        self.context_frames = frames.to(self.device, dtype=torch.float32)
        self.history_latents = self.evaluator.encode_frames(self.context_frames)

        if actions is not None:
            if actions.ndim == 2:
                actions = actions.unsqueeze(0)
            if actions.shape != (1, frames.shape[1], self.action_dim):
                raise ValueError(
                    "Actions must have shape [T, action_dim] or [1, T, action_dim] matching the context."
                )
            self.history_actions = actions.to(self.device, dtype=torch.float32).clone()
            if actions_mask is not None:
                if actions_mask.ndim == 1:
                    actions_mask = actions_mask.unsqueeze(0)
                if actions_mask.shape != (1, frames.shape[1]):
                    raise ValueError("Action mask must match the context length.")
                self.history_action_mask = actions_mask.to(self.device, dtype=torch.bool).clone()
            else:
                self.history_action_mask = torch.ones(
                    1, frames.shape[1], dtype=torch.bool, device=self.device
                )
        else:
            self.history_actions = None
            self.history_action_mask = None

        self.generated_latents.clear()
        self.generated_frames.clear()
        self.pending_action.zero_()
        self.apply_action = False
        self.hold_action = False
        self.step_count = 0

    def _build_default_bindings(self) -> None:
        self._bindings.clear()

        def register(key: str, description: str, index: int, magnitude: float) -> None:
            if index >= self.action_dim:
                return
            vector = torch.zeros(self.action_dim)
            vector[index] = magnitude
            self._bindings[key] = ActionBinding(key=key, description=description, vector=vector)

        register("d", "+X translation", 0, self.translation_step)
        register("a", "-X translation", 0, -self.translation_step)
        register("w", "+Y translation", 1, self.translation_step)
        register("s", "-Y translation", 1, -self.translation_step)
        register("r", "+Z translation", 2, self.translation_step)
        register("f", "-Z translation", 2, -self.translation_step)
        register("j", "+Yaw rotation", 5, self.rotation_step)
        register("l", "-Yaw rotation", 5, -self.rotation_step)
        register("i", "+Pitch rotation", 4, self.rotation_step)
        register("k", "-Pitch rotation", 4, -self.rotation_step)
        register("u", "+Roll rotation", 3, self.rotation_step)
        register("o", "-Roll rotation", 3, -self.rotation_step)

    # ------------------------------------------------------------------- loop
    def run(self) -> None:
        if self.history_latents is None or self.context_frames is None:
            raise RuntimeError("Call load_context(...) before starting the interactive session.")

        self._print_header()
        while True:
            try:
                raw = input("action> ").strip()
            except EOFError:
                print("\nExiting interactive session.")
                break

            if not raw:
                self._advance_one_step()
                continue

            cmd = raw.lower()
            if cmd in {"quit", "exit"}:
                break
            if cmd in {"step", "run"}:
                self._advance_one_step()
                continue
            if cmd in {"clear", "reset"}:
                self.pending_action.zero_()
                self.apply_action = False
                print("Cleared pending action.")
                continue
            if cmd in {"hold", "toggle"}:
                self.hold_action = not self.hold_action
                state = "enabled" if self.hold_action else "disabled"
                print(f"Hold mode {state}.")
                continue
            if cmd.startswith("save"):
                name = cmd[4:].strip("_- ")
                self._save_session(name or None)
                continue
            if cmd == "status":
                self._print_status()
                continue
            if cmd.startswith("scale "):
                self._adjust_scale(cmd)
                continue

            vector, used, unknown = self._parse_binding_string(cmd)
            if unknown:
                print(f"Ignored unknown keys: {', '.join(sorted(unknown))}")
            if used:
                self.pending_action = vector.to(self.device)
                self.apply_action = bool(torch.any(vector != 0.0))
                print(
                    f"Queued action from keys [{', '.join(used)}]: "
                    f"{self._format_vector(self.pending_action)}"
                )
            else:
                print("No valid bindings found; type 'status' for help.")

    # ------------------------------------------------------------------ helpers
    def _advance_one_step(self) -> None:
        if self.history_latents is None:
            raise RuntimeError("Context not loaded.")

        rollout_actions: Optional[torch.Tensor] = None
        rollout_mask: Optional[torch.Tensor] = None

        if self.apply_action or self.history_actions is not None:
            action_tensor = self.pending_action.view(1, 1, -1)
            rollout_actions = action_tensor
            mask_value = self.apply_action
            rollout_mask = torch.tensor([[mask_value]], dtype=torch.bool, device=self.device)

            if self.history_actions is None:
                history_len = self.history_latents.shape[1]
                self.history_actions = torch.zeros(
                    1, history_len, self.action_dim, device=self.device
                )
                self.history_action_mask = torch.zeros(
                    1, history_len, dtype=torch.bool, device=self.device
                )
        else:
            action_tensor = None

        history_actions = self.history_actions
        history_mask = self.history_action_mask

        result = self.evaluator.rollout(
            context_latents=self.history_latents,
            context_actions=history_actions,
            context_actions_mask=history_mask,
            rollout_actions=rollout_actions,
            rollout_action_mask=rollout_mask,
            rollout_steps=1,
            decode=True,
        )

        new_latent = result["generated_latents"]  # [1, 1, tokens, dim]
        new_frame = result["generated_frames"][0, 0]  # [C, H, W]

        # Update histories for next step
        self.history_latents = result["all_latents"]
        if self.history_actions is not None:
            if rollout_actions is None:
                appended_action = torch.zeros(
                    1, 1, self.action_dim, device=self.device, dtype=self.history_actions.dtype
                )
            else:
                appended_action = rollout_actions.to(self.device)
            if rollout_mask is None:
                appended_mask = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
            else:
                appended_mask = rollout_mask
            self.history_actions = torch.cat((self.history_actions, appended_action), dim=1)
            self.history_action_mask = torch.cat((self.history_action_mask, appended_mask), dim=1)

        self.generated_latents.append(new_latent[0, 0].detach().cpu())
        self.generated_frames.append(new_frame.detach().cpu())
        self.step_count += 1

        print(f"Generated step {self.step_count} | action applied: {self.apply_action}")

        if not self.hold_action:
            self.pending_action.zero_()
            self.apply_action = False

    def _save_session(self, name: Optional[str]) -> None:
        if self.context_frames is None or self.history_latents is None:
            print("No context loaded; nothing to save.")
            return

        session_dir = Path("archive/interactive_sessions")
        session_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        label = name or f"session_{timestamp}"
        target_dir = session_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)

        all_generated = torch.stack(self.generated_frames) if self.generated_frames else None

        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            print(
                "imageio not available; skipping video export. "
                "Install imageio to enable this feature."
            )
            imageio = None  # type: ignore

        if imageio is not None:
            video_frames = []
            context = self.context_frames[0].cpu()
            for frame in context:
                video_frames.append(self._to_numpy_frame(frame))
            if all_generated is not None:
                for frame in all_generated:
                    video_frames.append(self._to_numpy_frame(frame))
            video_path = target_dir / "rollout.mp4"
            with imageio.get_writer(video_path, mode="I", fps=12, codec="libx264") as writer:
                for frame in video_frames:
                    writer.append_data(frame)
            print(f"Saved video to {video_path}")

        payload = {
            "context_frames": self.context_frames.cpu(),
            "generated_frames": all_generated.cpu() if all_generated is not None else None,
            "all_latents": self.history_latents.cpu(),
            "actions": self.history_actions.cpu() if self.history_actions is not None else None,
            "actions_mask": self.history_action_mask.cpu()
            if self.history_action_mask is not None
            else None,
            "step_count": self.step_count,
            "translation_step": self.translation_step,
            "rotation_step": self.rotation_step,
        }
        torch.save(payload, target_dir / "rollout.pt")
        print(f"Saved tensors to {target_dir / 'rollout.pt'}")

    def _adjust_scale(self, command: str) -> None:
        tokens = command.split()
        if len(tokens) != 3 or tokens[1] not in {"translation", "rotation"}:
            print("Usage: scale translation 0.02 | scale rotation 0.05")
            return
        try:
            value = float(tokens[2])
        except ValueError:
            print("Scale value must be numeric.")
            return
        if tokens[1] == "translation":
            self.translation_step = value
        else:
            self.rotation_step = value
        self._build_default_bindings()
        print(
            f"Updated scales: translation={self.translation_step:.4f}, "
            f"rotation={self.rotation_step:.4f}"
        )

    def _parse_binding_string(self, text: str) -> Tuple[torch.Tensor, List[str], List[str]]:
        vector = torch.zeros(self.action_dim)
        used = []
        unknown = []
        for char in text:
            if char in self._bindings:
                vector += self._bindings[char].vector
                used.append(char)
            elif char.isalpha():
                unknown.append(char)
        return vector, used, unknown

    def _print_status(self) -> None:
        print("--- Session status ---")
        print(f"Steps generated: {self.step_count}")
        print(f"Hold action: {'on' if self.hold_action else 'off'}")
        print(f"Pending action: {self._format_vector(self.pending_action)}")
        print(
            f"Action scales -> translation: {self.translation_step:.4f}, "
            f"rotation: {self.rotation_step:.4f}"
        )
        keys = ", ".join(f"{b.key}:{b.description}" for b in self._bindings.values())
        print(f"Bindings: {keys}")

    def _print_header(self) -> None:
        print("Interactive world model session")
        print("Type key combos like 'wd' to queue actions (WASD = xy, R/F = z, IJKLUO = rotations).")
        print("Commands: step, hold, clear, status, save [name], scale translation|rotation value, quit")
        print("Press Enter with no input to advance one step.")

    @staticmethod
    def _format_vector(vector: torch.Tensor) -> str:
        values = vector.cpu().tolist()
        formatted = ", ".join(f"{v:+.4f}" for v in values)
        return f"[{formatted}]"

    @staticmethod
    def _to_numpy_frame(frame: torch.Tensor) -> np.ndarray:
        frame = frame.clamp(0.0, 1.0)
        array = frame.permute(1, 2, 0).numpy()
        return (array * 255.0).astype("uint8")
