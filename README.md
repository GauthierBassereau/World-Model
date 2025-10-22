# World Model | Master Thesis (in progress)

## 1. Motivation and Scope
- Build a task-agnostic latent world model that combines Dreamer V4 shortcut forcing with DINO-style latent planning to support goal-conditioned MPC and offline RL.
- Target robot: single 6-DoF arm commanded with delta end-effector poses.
- Data regime: train exclusively on LeRobot-format corpora composed of 95% real-world raw manipulation videos without actions and 5% action-labeled demonstrations from a forthcoming in-house robot dataset. The unlabeled portion is currently the DROID dataset converted to LeRobot; the labeled portion will be supplied by a future custom dataset, for now it is a subset of droid.

## 2. Research Foundations

- Image encoder -> Dinov2 RAE autoencoder
- World model's architecure -> Dreamer 4
- World model's training -> Shortcut Forcing from Dreamer 4
- MPC on dino features -> DINO-WM

## 3. Integrated Architecture

- Freeze the Dinov2 encoder and use MAE decoder from RAE paper (for visualization purposes).
- Use the Dreamer V4 block-causal transformer without policy heads, alternating spatial-only and temporal-only attention layers.
- Temporal layers share KV caches (GQA) for efficiency; employ QKNorm, attention logit soft capping, pre-layer RMSNorm, and SwiGLU MLPs.
- Sequence per step: `[register tokens | latent tokens | shortcut tokens | action tokens]`, with learned embeddings standing in for unknown actions on raw clips.

---

Potential very interesting datasets:
  - airoa-moma (https://huggingface.co/datasets/airoa-org/airoa-moma)
  - agibot_alpha (https://huggingface.co/datasets/cadene/agibot_alpha_v30)

Todos:
  - Change from v prediction to x prediction following dreamerv4 paper, need to understand the maths behind too because I am really not sure about the why.
  - Create a proper evaluation module.
  - Add a DH head to backbone following RAE paper.
  - Need to rewrite the attentions with fused kernels
  - Need to rewrite the attention module and simply permute the input so it becomes either spatial or temporal I am really dumb for not doing that before