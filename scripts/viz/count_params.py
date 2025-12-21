import sys
import os
import torch

from src.world_model.backbone import WorldModelBackbone, WorldModelConfig

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def main():
    config = WorldModelConfig(
        latent_dim=1536,
        input_dim=768,
        bottleneck_dim=256,
        action_dim=7,
        num_registers=4,
        depth=28,
        num_heads=1,
        mlp_multiplier=2.666667,
        temporal_attention_interval=4,
        temporal_context_length=15,
        rope_base=10000.0,
        qk_norm_eps=1e-6,
        attn_logit_softcapping=50.0,
    )

    print("Initializing WorldModelBackbone with the following config:")
    print(config)

    model = WorldModelBackbone(config)
    
    params = count_parameters(model)
    print(f"Number of parameters: {params:,}")
    print(f"Number of parameters (millions): {params/1e6:.2f}M")

if __name__ == "__main__":
    main()
