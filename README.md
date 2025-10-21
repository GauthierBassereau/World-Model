# World Model | Master Thesis (in progress)

## 1. Motivation and Scope
- Build a task-agnostic latent world model that combines Dreamer V4 shortcut forcing with DINO-style latent planning to support goal-conditioned MPC.
- Target robot: single 6-DoF arm commanded with delta end-effector poses.
- Data regime: train exclusively on LeRobot-format corpora composed of 95% real-world raw manipulation videos without actions and 5% action-labeled demonstrations from a forthcoming in-house robot dataset. The unlabeled portion is currently the DROID dataset converted to LeRobot; the labeled portion will be supplied by the future custom dataset.

## 2. Research Foundations

- Image encoder -> Dinov2 RAE autoencoder
- World model's architecure -> Dreamer 4
- World model's training -> Shortcut Forcing from Dreamer 4
- MPC on dino features -> DINO-WM

## 3. Integrated Architecture

- Freeze the Dinov2 encoder and use MAE decoder from RAE paper.
- Use the Dreamer V4 block-causal transformer without policy heads, alternating spatial-only and temporal-only attention layers.
- Temporal layers share KV caches (GQA) for efficiency; employ QKNorm, attention logit soft capping, pre-layer RMSNorm, and SwiGLU MLPs.
- Sequence per step: `[register tokens | latent tokens | shortcut tokens | action tokens]`, with learned embeddings standing in for unknown actions on raw clips.

## 4. Data Strategy

- **Unlabeled 95%**: DROID dataset converted to LeRobot format; only RGB video streams are consumed, and action columns are ignored. For now only using DROID, but here is a short list of potential very interesting datasets:
  - airoa-moma (https://huggingface.co/datasets/airoa-org/airoa-moma)
  - agibot_alpha (https://huggingface.co/datasets/cadene/agibot_alpha_v30)
- **Labeled 5%**: forthcoming in-house LeRobot dataset capturing the target robot with action deltas.