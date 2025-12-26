# World Model | Master Thesis (in progress)

Build a latent world model that combines Dreamer V4 training with DINO representation encoder for robotic planning, goal-conditioned MPC and offline/model-based RL.
Learning general world dynamics and robot actions from large scale video datasets, both non-robot data and robot data.

## Research Foundations

- Image encoder -> Dinov2 RAE autoencoder, see paper [https://arxiv.org/abs/2510.11690]
- World model's architecture -> Dreamerv4, Spatial Temporal Transformer, see paper [https://arxiv.org/abs/2509.24527]
- World model's training objective -> Diffusion forcing (not yet shortcut forcing like Dreamer V4), see paper [https://arxiv.org/abs/2407.01392]
- Model Predictive Control on dino features -> DINO-WM, see paper [https://arxiv.org/abs/2411.04983]

Other related papers:
- JIT: [https://arxiv.org/abs/2511.13720]
- Self-Forcing: [https://arxiv.org/abs/2506.08009]
- Diffusion Forcing: [https://arxiv.org/abs/2407.01392]
- Shortcut Diffusion: [https://arxiv.org/abs/2410.12557]

## Integrated Architecture

- Freeze the Dinov2 encoder and use MAE decoder from RAE paper (only for visualization purposes).
- Use the Dreamer V4 block-causal transformer, alternating spatial-only and temporal-only attention layers, QKNorm, attention logit soft capping, pre-layer RMSNorm, and SwiGLU MLPs, and KV caching.
- Token sequence per step: `[timestep | actions | registers | dino latents]`

## Data

-  Learning diverse world dynamics -> EPIC-KITCHENS, BridgeV2, Droid
<!-- Potential additions: Ego4D, Something-Something V2, EgoDex, agibot_alpha, OpenVid1M, HowTo100M -->
-  Learning robot interactions + action conditioning with world -> SOAR

Links to HF datasets:
- [https://huggingface.co/datasets/Gaugou/BridgeV2]
- [https://huggingface.co/datasets/Gaugou/Soar]
- [https://huggingface.co/datasets/Gaugou/epic_kitchens_100]
- [https://huggingface.co/datasets/lerobot/droid_1.0.1]

---

Todos:
  - Implement alternating sequence lengths training, keeping always same amount of tokens per step but different batch size or/and accumulation steps
  - Separate WorldBatch and WorldBatchMetadata -> cleaner, better for logs
  - Shortcut forcing
  - Dinov3 (training decoder using RAE codebase ?)
  - Add a DH head to backbone following RAE paper -> RAE uses v-space prediction, and it is probably why DH head is so effective as explained in the JIT paper. This could be a good experiment to confirm the findings in JIT.