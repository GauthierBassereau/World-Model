"""This entire rae_dino module is modified from the RAE repository"""

import torch
import torch.nn as nn
from src.rae_dino.decoder.mae_decoder import GeneralDecoder
from src.rae_dino.encoder.dinov2 import Dinov2withNorm
from transformers import AutoConfig, AutoImageProcessor
from typing import Optional
from math import sqrt


class RAE(nn.Module):
    def __init__(
        self,
        # ---- encoder configs ----
        dinov2_path: str = "facebook/dinov2-with-registers-base",
        encoder_input_size: int = 224,  # always 224 because decoder is expecting this.
        # ---- decoder configs ----
        decoder_config_path: str = "src/rae_dino/decoder/config.json",
        decoder_patch_size: int = 16,
        pretrained_decoder_path: str = "src/rae_dino/decoder/decoder_weights/ViTXL_n08.pt",
        # ---- noising and normalization -----
        noise_tau: float = 0.0,  # For inference set to 0.0
        normalization_stat_path: Optional[str] = "src/rae_dino/encoder/stat.pt",
        eps: float = 1e-5,
    ):
        super().__init__()
        proc = AutoImageProcessor.from_pretrained(dinov2_path, use_fast=False)
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
        self.encoder = Dinov2withNorm(dinov2_path)
        # see if the encoder has patch size attribute            
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert self.encoder_input_size % self.encoder_patch_size == 0, f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2 # number of patches of the latent
        
        # decoder
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim # set the hidden size of the decoder to be the same as the encoder's output
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches)) 
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)
        # load pretrained decoder weights
        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location='cpu')
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                print(f"Missing keys when loading pretrained decoder: {keys.missing_keys}")
        self.noise_tau = noise_tau
        self.eps = eps
        self.latent_mean: Optional[torch.Tensor] = None
        self.latent_var: Optional[torch.Tensor] = None
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location='cpu')
            mean = stats.get("mean", None)
            var = stats.get("var", None)
            self.latent_mean = self._reshape_stat_tensor(mean, "mean") if mean is not None else None
            self.latent_var = self._reshape_stat_tensor(var, "var") if var is not None else None
            self.do_normalization = True
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            self.do_normalization = False
        self._input_hw: Optional[tuple[int, int]] = None

    def _reshape_stat_tensor(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        tensor = tensor.to(torch.float32)
        if tensor.ndim == 3:
            channels, height, width = tensor.shape
            if channels != self.latent_dim:
                raise ValueError(f"{name} stats expected {self.latent_dim} channels but found {channels}.")
            tensor = tensor.view(self.latent_dim, height * width).transpose(0, 1)
        elif tensor.ndim == 2:
            rows, cols = tensor.shape
            if rows == self.latent_dim and cols == self.base_patches:
                tensor = tensor.transpose(0, 1)
            elif rows == self.base_patches and cols == self.latent_dim:
                tensor = tensor.contiguous()
            else:
                raise ValueError(
                    f"{name} stats shape {tuple(tensor.shape)} incompatible with "
                    f"({self.base_patches}, {self.latent_dim})."
                )
        elif tensor.ndim == 1:
            if tensor.shape[0] != self.latent_dim:
                raise ValueError(f"{name} stats length {tensor.shape[0]} must equal latent dim {self.latent_dim}.")
            tensor = tensor.unsqueeze(0).expand(self.base_patches, -1)
        else:
            raise ValueError(f"{name} stats with ndim={tensor.ndim} unsupported.")
        return tensor.unsqueeze(0).contiguous()
            
    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        noise = noise_sigma * torch.randn_like(x)
        return x + noise

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # normalize input
        _, _, h, w = x.shape
        self._input_hw = (h, w)
        if (h, w) != (self.encoder_input_size, self.encoder_input_size):
            x = nn.functional.interpolate(x, size=(self.encoder_input_size, self.encoder_input_size), mode='bicubic', align_corners=False)
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        z = self.encoder(x)
        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
        output = self.decoder(z, drop_cls_token=False).logits
        x_rec = self.decoder.unpatchify(output)
        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        if self._input_hw is not None and self._input_hw != (self.encoder_input_size, self.encoder_input_size):
            x_rec = nn.functional.interpolate(
                x_rec, size=self._input_hw, mode="bicubic", align_corners=False
            )
        return x_rec
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
