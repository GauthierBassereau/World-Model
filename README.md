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

## 4. Data

1/ Learning diverse world dynamics, interactions between objects -> HowTo100M, Ego4D, Something-Something V2, EPIC-KITCHENS
2/ Learning robot interactions + action conditioning with world -> DROID, SOAR, agibot_alpha

Note: most of this dataset have language instruction for each task -> I wonder how fast the WM can learn to condition its generation on text instead of action.

---

Todos:
  - Add gripper values to actions, and use actions.cartesian instead of observations. Make a script to visualize the distribution of the actions deltas, if need normalization etc...
  - Review evaluation code, for now has been vibe-coded by codex, add an eval pipeline to the trainer too
  - Make something to visualize the attention masks
  - Add one of the non-robot dataset (HowTo100M would be great)
  - Add a DH head to backbone following RAE paper
  - Shortcut forcing