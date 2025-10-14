# Dreamer4-DINO World Model Project

## 1. Motivation and Scope
- Build a task-agnostic latent world model that combines Dreamer V4 shortcut forcing with DINO-style latent planning to support goal-conditioned MPC.
- Target robot: single 6-DoF arm commanded with delta end-effector poses (XYZ position deltas, optional wrist rotation, and gripper control) at 10–20 Hz once deployed.
- Data regime: train exclusively on LeRobot-format corpora composed of 95% real-world raw manipulation videos without actions and 5% action-labeled demonstrations from a forthcoming in-house robot dataset. The unlabeled portion is currently the DROID dataset converted to LeRobot; the labeled portion will be supplied by the future custom dataset.

## 2. Research Foundations

### Dreamer V4 (Training Agents Inside of Scalable World Models)
- Adopt Phase 1 (tokenizer + dynamics pretraining) of the Dreamer V4 recipe, emphasizing shortcut forcing and causal transformers without policy/reward heads.
- World model backbone: block-causal transformer with alternating spatial and temporal attention, register tokens, RoPE, grouped-query attention, and shortcut forcing (flow matching + bootstrap) to reach stable predictions in four denoising steps.
- Tokenizer: causal masked autoencoder with tanh bottleneck; patch dropout encourages spatial consistency and supports fast inference on a single GPU.
- Action conditioning is optional for unlabeled data—Dreamer-style learned null tokens allow the model to mix clips with and without action annotations.

### DINO-WM (World Models on Pre-trained Visual Features Enable Zero-shot Planning)
- Observation encoder is frozen DINO patch embeddings (we use DINOv3-B). Transition model predicts next latent frame using a causal ViT; decoder is only for visualization.
- Training objective is latent-space MSE between predicted and encoded latents with teacher forcing; no pixel reconstruction is required for core training.
- Planning relies on Cross-Entropy Method MPC minimizing latent distance to a goal latent (`||z_T - z_goal||^2`), enabling zero-shot generalization across tasks when latent space is well aligned.

## 3. Integrated Architecture

### 3.1 Observation Pipeline
- Freeze DINOv3-B (ViT-B/16) and integrate it directly inside the LeRobot dataset loader so frames arrive as patch tokens.
- Return the 14×14 patch grid tokens (dim 768) and optional CLS token for diagnostics; append Dreamer-style register tokens for global aggregation.
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

### 4.1 Dataset Mix
- **Unlabeled 95%**: DROID dataset converted to LeRobot format; only RGB video streams are consumed, and action columns are ignored. For now only using DROID, but here is a short list of potential very interesting datasets:
  - airoa-moma (https://huggingface.co/datasets/airoa-org/airoa-moma)
  - agibot_alpha (https://huggingface.co/datasets/cadene/agibot_alpha_v30)
- **Labeled 5%**: forthcoming in-house LeRobot dataset capturing the target robot with action deltas; collection pipeline still to be built.
- All sources follow the LeRobot directory schema (episodes, parquet indices, media folders) to simplify ingestion and batching.
- Reference implementations for loading these datasets live in `lerobot_tests/dataset_download.py` (offline DataLoader) and `lerobot_tests/test_streaming_dataset.py` (streaming iterator with delta timestamps).

### 4.2 Handling Unlabeled Clips
- Strip or mask action fields within the DROID parquet files, replacing them with the learned null-action embedding during training.
- Maintain metadata about camera viewpoints and scene IDs so curriculum schedules can interleave similar contexts.
- Periodically validate frame integrity and timestamps to guarantee consistent sequence lengths before batching.

### 4.3 Handling Action-Labeled Robot Clips
- Record new demonstrations with synchronized sensor streams and delta end-effector actions, exporting directly through `lerobot` tooling.
- Use the same image resolution and normalization as DROID to minimize domain shift; add robot-specific labels (task tags, workspace ID) for filtering.
- Hold out a validation split for the action-conditioned rollout metrics and MPC evaluation.

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
- Compare against a baseline that omits shortcut forcing (pure teacher-forced ViT dynamics) to quantify benefits.
- Stress-test domain shifts (lighting, backgrounds) since the encoder remains frozen.

## 7. Repository Layout
```
├── configs/
│   ├── world_model.yaml    # Dreamer4-DINO hyperparameters
│   ├── decoder.yaml        # reconstruction head options
│   └── mpc.yaml            # planning defaults
├── lerobot_tests/
│   ├── dataset_download.py       # offline DataLoader example with delta timestamps
│   ├── lerobot_dataset.py        # upstream LeRobot dataset implementation snapshot
│   └── test_streaming_dataset.py
├── world_model/
│   ├── transformer.py
│   ├── shortcut_losses.py
│   └── rollout.py
├── training/
│   ├── world_model_trainer.py
│   ├── decoder_trainer.py
│   └── datasets.py
├── planning/
│   ├── cem_mpc.py
│   └── goal_latent.py
├── vision/
│   ├── dino_v3.py         # frozen encoder wrapper
│   └── image_decoder.py   # optional visualization decoder
└── scripts/
    ├── preprocess_lerobot.py
    ├── evaluate_world_model.py
    ├── run_cem_mpc.py
    └── visualize_rollouts.py
```

## 8. Config Highlights
- `model_dim`: 768, `num_layers`: 24, `num_heads`: 16, `num_registers`: 8 (tune to match hardware).
- `shortcut_steps K`: 4, `tau_min`: 0.02, `tau_max`: 1.0, `ramp_weight`: `(0.9 * tau + 0.1)`.
- `tau_distribution` / `tau_power` and `noise_mean` / `noise_std`: bias sampling toward cleaner latents and align the base noise with dataset statistics when training flow matching.
- `context_frames`: `{64, 160}`, with per-device batch size 1–2 and gradient accumulation.
- Action encoder hidden dim: 128; learned null-action embedding shared across unlabeled clips.
- Optimizer: AdamW (`lr = 1e-4`, `betas = (0.9, 0.95)`, weight decay `0.1`) with gradient clipping at 1.0 and bfloat16 mixed precision; flip on `ema.decay` to maintain a smoothed copy of the world model.

## 9. Deployment Considerations
- Real-time inference targets ~20 FPS on a single A100/H100 by caching KV states and reusing shortcut forcing steps.
- Use Torch Dynamo (`torch.compile`) or TensorRT for additional speed-ups; preallocate CUDA graphs for inference + MPC loops.
- Enforce safety guards (joint limits, velocity clamps) before sending commands; optionally extend with control barrier functions.
- When new LeRobot episodes arrive, perform low-learning-rate fine-tuning with EMA tracking while monitoring for catastrophic forgetting.

## 10. References
- Hafner et al., 2025. *Training Agents Inside of Scalable World Models (Dreamer 4).*
- Zhou et al., 2024. *DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning.*
