# World Model | Master Thesis (in progress)

## 1. Motivation and Scope
- Build a task-agnostic latent world model that combines Dreamer V4 shortcut forcing with DINO-style latent planning to support goal-conditioned MPC.
- Target robot: single 6-DoF arm commanded with delta end-effector poses.
- Data regime: train exclusively on LeRobot-format corpora composed of 95% real-world raw manipulation videos without actions and 5% action-labeled demonstrations from a forthcoming in-house robot dataset. The unlabeled portion is currently the DROID dataset converted to LeRobot; the labeled portion will be supplied by the future custom dataset.

## 2. Research Foundations

Image encoder -> DinoV3
World model's architecure -> Dreamer 4
World model's trianing -> DiT-RAE mixed with Shortcute Forcing from Dreamer 4
MPC on dino features -> DINO-WM

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
- Optional lightweight transposed-convolution decoder for qualitative inspection of rollouts.
- Train the decoder on frozen latents with L2 + LPIPS but keep gradients from touching the dynamics model to preserve latent prediction quality.

## 4. Data Strategy

- **Unlabeled 95%**: DROID dataset converted to LeRobot format; only RGB video streams are consumed, and action columns are ignored. For now only using DROID, but here is a short list of potential very interesting datasets:
  - airoa-moma (https://huggingface.co/datasets/airoa-org/airoa-moma)
  - agibot_alpha (https://huggingface.co/datasets/cadene/agibot_alpha_v30)
- **Labeled 5%**: forthcoming in-house LeRobot dataset capturing the target robot with action deltas; collection pipeline still to be built.
- All sources follow the LeRobot directory schema (episodes, parquet indices, media folders) to simplify ingestion and batching.
- Reference implementations for loading these datasets live in `lerobot_tests/dataset_download.py` (offline DataLoader) and `lerobot_tests/test_streaming_dataset.py` (streaming iterator with delta timestamps).

## 5. Training Workflow
- Build a LeRobot streaming dataset that yields mixed minibatches (e.g., 19 unlabeled sequences for every labeled sequence to match the 95/5 ratio).
- Dry-run data ingestion using the helpers in `lerobot_tests/` to confirm camera keys, FPS, and delta timestamp stacking before launching large jobs.
- Tokenize frames on the fly via the frozen DINO encoder, cache tokens when possible, and stitch sequences to target context lengths (64–160 frames).
- Apply Dreamer V4 shortcut forcing loss with curriculum on `tau` and rollout horizon; monitor latent MSE separately for unlabeled and labeled subsets.
- Incorporate gradient accumulation to handle long contexts with small per-device batch sizes (1–2 sequences per GPU).

## 6. Evaluation Protocol

### 6.1 World Model Quality
- Measure latent-space distance `||z_pred - z_target||^2` on held-out DROID clips (unlabeled) and robot validation sequences (labeled) for horizons of 1, 5, 10, and 20 steps.
- Track shortcut loss components, action-conditioned rollout deviation, and robustness of null-action predictions on raw clips.
- Decode rollouts with the optional decoder for qualitative checks, acknowledging pixel fidelity is not the primary objective.
- Run ablations on shortcut forcing, labeled fraction (5% vs. synthetic increases), and context length scheduling.

### 6.2 MPC Performance
- Evaluate CEM MPC latency, success rate on goal-image reaching tasks, and smoothness of executed action deltas using the labeled robot dataset.
- Compare against a baseline that omits to use diffusion for learning dynamics from unlabeled data and quantify benefits of the approach.
- Stress-test domain shifts (lighting, backgrounds) since the encoder remains frozen, should be trivial for dino.


## 10. References
See `archive/important_papers`.
