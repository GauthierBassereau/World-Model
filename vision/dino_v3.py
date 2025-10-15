"""
Wrapper around the DINOv3 vision transformer that converts RGB frames into patch
tokens ready for the world model. The encoder is loaded directly from the
Hugging Face hub using the transformers library.

Input tensors:
    images: float Tensor of shape [B, 3, H, W] in [0, 1] or [0, 255]

Outputs:
    tokens: float Tensor of shape [B, num_tokens, embed_dim] where num_tokens
    equals the spatial grid implied by the encoder patch size. The last
    dimension matches the transformer hidden size (768 for ViT-B).

Config keys used with this module:
    vision.repo_id         # Hugging Face model id, defaults to facebook/dinov3-vitb16-pretrain-lvd1689m
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


@dataclass
class DinoVisionConfig:
    """Hyperparameters for the frozen DINO encoder."""

    repo_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    output_dtype: Union[torch.dtype, str] = torch.float32


class DinoV3Embedder(nn.Module):
    """
    Frozen wrapper that projects batched RGB images to DINO patch tokens.

    The encoder keeps gradients disabled by default so it can be reused safely
    across dataloaders. The class exposes only the forward method needed by the
    training pipeline in this repository.
    """

    def __init__(self, config: Optional[DinoVisionConfig] = None) -> None:
        super().__init__()
        self.config = config or DinoVisionConfig()

        self.processor = AutoImageProcessor.from_pretrained(self.config.repo_id)
        self.model = AutoModel.from_pretrained(self.config.repo_id, dtype=torch.float32)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        if isinstance(self.config.output_dtype, str):
            self.output_dtype = getattr(torch, self.config.output_dtype)
        else:
            self.output_dtype = self.config.output_dtype

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Return DINO patch tokens for a batch of RGB frames.

        Args:
            images: Tensor in [B, 3, H, W]; uint8 tensors are scaled down to
                [0, 1] before normalization.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images of shape [B, 3, H, W], got {tuple(images.shape)}")

        processed = self._prepare_inputs(images)
        pixel_values = processed.to(self.model.device)

        patch_size = self.model.config.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_h = int(patch_size[0])
            patch_w = int(patch_size[1] if len(patch_size) > 1 else patch_size[0])
        else:
            patch_h = patch_w = int(patch_size)

        outputs = self.model(pixel_values=pixel_values)

        tokens = outputs.last_hidden_state  # [B, tokens, dim] (CLS + registers + patches)
        spatial_tokens = tokens[:, 1:, :]

        grid_h = pixel_values.shape[-2] // patch_h
        grid_w = pixel_values.shape[-1] // patch_w
        expected = grid_h * grid_w

        if spatial_tokens.shape[1] > expected:
            spatial_tokens = spatial_tokens[:, -expected:, :]
        elif spatial_tokens.shape[1] < expected:
            raise ValueError(
                f"Received {spatial_tokens.shape[1]} spatial tokens but expected {expected} "
                f"for grid {grid_h}x{grid_w}. "
                "Make sure input dimensions are multiples of the patch size."
            )

        return spatial_tokens.to(self.output_dtype)

    def _prepare_inputs(self, images: torch.Tensor) -> torch.Tensor:
        frames = images.detach().cpu()
        if torch.is_floating_point(frames):
            if frames.max() <= 1.0:
                frames = frames * 255.0
            frames = frames.clamp(0.0, 255.0).round().to(torch.uint8)
        else:
            frames = frames.clamp(0, 255).to(torch.uint8)

        frames = frames.permute(0, 2, 3, 1)  # [B, H, W, C]
        image_list: List = [frame.numpy() for frame in frames]
        inputs = self.processor(
            images=image_list,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        return inputs["pixel_values"]


if __name__ == "__main__":
    embedder = DinoV3Embedder()
    dummy_frames = torch.rand(1, 3, 180, 320)
    with torch.no_grad():
        tokens = embedder(dummy_frames)
    print(f"tokens shape: {tuple(tokens.shape)}")
    print(f"latents mean: {tokens.mean().item():.6f}")
    print(f"latents std: {tokens.std(unbiased=False).item():.6f}")
