# World Model | Master Thesis (in progress)

## 1. Motivation and Scope
- Build a task-agnostic latent world model that combines Dreamer V4 shortcut forcing with DINO-style latent planning to support goal-conditioned MPC.
- Target robot: single 6-DoF arm commanded with delta end-effector poses.
- Data regime: train exclusively on LeRobot-format corpora composed of 95% real-world raw manipulation videos without actions and 5% action-labeled demonstrations from a forthcoming in-house robot dataset. The unlabeled portion is currently the DROID dataset converted to LeRobot; the labeled portion will be supplied by the future custom dataset.

## 2. Research Foundations

- Image encoder -> DinoV3
- World model's architecure -> Dreamer 4
- World model's trianing -> DiT-RAE mixed with Shortcute Forcing from Dreamer 4
- MPC on dino features -> DINO-WM

## 3. Integrated Architecture

### 3.1 Observation Pipeline
- Freeze DINOv3-B (ViT-B/16) and integrate it directly inside the LeRobot dataset loader so frames arrive as patch tokens.
- Return the 14×14 patch grid tokens (dim 768); append Dreamer-style register tokens for global aggregation.
- Normalize inputs with DINO preprocessing and apply consistent temporal augmentations (color jitter, random crop) during data loading.

### 3.2 Dynamics Transformer
- Use the Dreamer V4 block-causal transformer without policy heads, alternating spatial-only and temporal-only attention layers.
- Temporal layers share KV caches (GQA) for efficiency; employ QKNorm, attention logit soft capping, pre-layer RMSNorm, and SwiGLU MLPs.
- Sequence per step: `[register tokens | latent tokens | shortcut tokens | action tokens]`, with learned embeddings standing in for unknown actions on raw clips.
- Optimize the shortcut forcing objective (Equation 7 in Dreamer 4) with x-prediction, flow matching, bootstrap terms, and four shortcut steps (`K = 4`).

### 3.3 Decoder for Visualization
- From DiT RAE paper, ViT decoder from MAE.
- Train the decoder on frozen latents with L2 + LPIPS + GAN

## 4. Data Strategy

- **Unlabeled 95%**: DROID dataset converted to LeRobot format; only RGB video streams are consumed, and action columns are ignored. For now only using DROID, but here is a short list of potential very interesting datasets:
  - airoa-moma (https://huggingface.co/datasets/airoa-org/airoa-moma)
  - agibot_alpha (https://huggingface.co/datasets/cadene/agibot_alpha_v30)
- **Labeled 5%**: forthcoming in-house LeRobot dataset capturing the target robot with action deltas.