# Diffusion World Model for Robotics

**Master's thesis, University of Tartu, 2026**

This repository began as a diffusion-style world model for robot manipulation. The thesis baseline encodes video with frozen DINOv2 features, predicts future semantic patches with a 400M parameter spatial-temporal transformer, and uses those imagined latent rollouts for offline goal-conditioned planning. The codebase now also contains a post-thesis extension based on RAEv2, patch-level noising, richer conditioning tokens, and an optional wide diffusion head.

I built the full stack around the idea: UR5 data recording through LeRobot and RTDE, dataset converters for mixed video training, DINOv2/RAE encoding and decoding, distributed world-model training, held-out rollout evaluation, and a Cross-Entropy Method planner over UR5 end-effector actions.

<p align="center">
  <img src="assets/readme/method_pipeline.png" alt="DINOv2 latent diffusion world model training pipeline" width="100%">
</p>

[Thesis PDF](thesis/thesis.pdf) | [Thesis source](thesis/thesis.tex)

## Current Extension: RAEv2 and Patch-Level Noising

The quantitative world-model and planning results later in this README describe the original thesis baseline. The active UR5 training config now explores a separate extension inspired by [RAEv2](https://arxiv.org/abs/2605.18324), the original [RAE](https://arxiv.org/abs/2510.11690), [Patch Forcing](https://arxiv.org/abs/2604.19141), and the scalable world-model architecture in [Dreamer 4](https://arxiv.org/abs/2509.24527).

### RAEv2 autoencoder

The default `dinov3b-k11` profile is an atomic set of a frozen DINOv3-B encoder, its matching pretrained ViT-XL decoder, and positional latent normalization statistics. At 256 x 256 resolution, the public interface is:

```text
RGB [B, 3, 256, 256]  ->  latent tokens [B, 256, 768]  ->  RGB [B, 3, 256, 256]
```

The assets come from `nyu-visionx/RAEv2-models` at pinned revision `9770b7b980fa1875c8e6d65f226c615c0ce908a8`. The first construction downloads only the three files belonging to the selected profile into the Hugging Face cache; later runs reuse them. `cache_dir` can relocate the cache, and `local_files_only: true` makes cluster jobs fail rather than access the network. Keeping the three files behind one profile prevents accidental encoder/decoder/statistics mismatches.

| Profile | Encoder aggregation | Latent contract | Intended use |
|---|---|---|---|
| `dinov3b-k11` | DINOv3-B, blocks 1-11 | `256 x 768` | Default, lower memory |
| `dinov3l-k7-general` | DINOv3-L, 7-layer MLS | `256 x 1024` | General-data reconstruction/generation tradeoff |
| `dinov3l-k23-general` | DINOv3-L, 23-layer MLS | `256 x 1024` | Highest-reconstruction profile |

The large profiles also require `world_model.input_dim: 1024`. To reproduce an encode/decode check on CPU, CUDA, or Apple MPS:

```bash
python scripts/viz/reconstruct_autoencoder.py \
  --image /path/to/ur5_frame.png \
  --profile dinov3b-k11 \
  --device auto \
  --stem ur5_dinov3b_k11
```

The checked-in UR5 run achieved **28.96 dB PSNR** on Apple MPS; its [metrics](archive/autoencoder_reconstruction/ur5_dinov3b_k11.json) and source images live under `archive/autoencoder_reconstruction/`.

<p align="center">
  <img src="archive/autoencoder_reconstruction/ur5_dinov3b_k11_comparison.png" alt="UR5 input and DINOv3-B K11 RAEv2 reconstruction" width="82%">
</p>

### Conditioning, noising, and prediction head

The current per-frame token layout is:

```text
[time_1..time_4 | action_1..action_4 | register_1..register_4 | 256 DINOv3 patch tokens]
```

The four time tokens share one Gaussian-Fourier embedding plus four learned offsets. During LTG patch noising they encode the original frame-level upper bound sampled before per-patch corruption. Individual patches may be made noisier, but their noise levels are deliberately not provided to the model. This differs intentionally from Patch Forcing, which conditions each patch on its own timestep: here the model must recognize from the latent content that one generated patch is less reliable than its neighbors. The actual per-patch signal levels are still used to construct the noisy input and compute the loss.

Actions use the same pattern: the action vector is projected once and added to four learned token offsets. When actions are masked or unavailable, the four learned base tokens remain, preserving a fixed layout across robot and passive-video data.

The DreamerV4-style backbone is otherwise unchanged: spatial attention operates within frames, causal temporal attention operates across matching token positions, and rollout still uses the temporal KV cache. The model continues to predict the clean latent `x`; training converts that prediction to velocity using each patch's real signal level and applies the existing velocity loss.

`world_model.output_head` selects the prediction head:

- `linear` preserves the thesis-style normalized linear projection and supports legacy ablations.
- `dh` uses the RAE/RAEv2 wide head: two spatial transformer blocks of width 2048 and 16 heads, conditioned patch-by-patch by the temporally informed backbone. `world_model.gradient_checkpointing` applies one memory/speed choice consistently to every backbone and DH transformer block during training.

During rollout, context priming and insertion of generated feedback call the backbone in cache-only mode. They update temporal keys and values but skip the prediction projection or expensive DH head; denoising steps still execute the selected head normally. This is an inference optimization and does not change rollout semantics.

### Checkpoint configuration

New training checkpoints save the complete experiment config alongside model and optimizer state. Evaluation and planning default to `use_checkpoint_config: true`, restoring the saved world-model architecture and autoencoder profile before construction. This keeps the latent width, token counts, and `linear`/`dh` head consistent with the weights. Set it to `false` only for a deliberate override; legacy checkpoints without a saved config continue to use the YAML values. Exact resume is strict, while non-resume checkpoint loading can initialize newly added modules for finetuning and reports missing or unexpected keys.

When EMA is enabled with a delayed `start_step`, it is initialized from the online weights at that step. Periodic evaluation uses the online model before EMA starts and the EMA model afterward. Active EMA weights are saved in the checkpoint, and standalone evaluation and playground scripts prefer them automatically.

To continue a stopped run, set `trainer.resume: true`, point `trainer.load_checkpoint` at the checkpoint, and keep the original seed. Current checkpoints restore the optimizer, global step, per-rank RNG state, sampler epoch, position within that epoch, EMA state, and W&B run ID. Changing the seed changes the permutation underneath the saved cursor and can both repeat old samples and skip unseen ones. Keep the world size, global batch size, and gradient accumulation unchanged for the closest continuation. Legacy checkpoints have no saved RNG or W&B state, but their data cursor is inferred from `epoch` and `epoch_step`.

## Core Question

The final thesis focuses on a recorded UR5 target domain and a concrete evaluation question:

> Can a world model trained in a frozen semantic visual representation use both passive video and action-labeled robot data to produce stable action-conditioned rollouts, and can those rollouts be useful for offline visual planning?

The target robot dataset is intentionally small: around two hours of UR5 interaction data with 10 tabletop objects, varied camera viewpoints, and no fixed imitation task. The rest of the training signal comes from mixed video: BridgeData V2, EPIC-KITCHENS, and DROID. After temporal resampling and selection, the training corpus is roughly 1,000 hours at 5 Hz.

## Thesis Baseline System

| Component | Thesis setup |
|---|---|
| Visual state | Frozen DINOv2-base patch features |
| Per-frame latent | `16 x 16` patch grid, 768 features per patch, 196,608 scalars total |
| Decoder | RAE decoder, used only for visualization |
| World model | 24-block block-causal transformer, width 1024, 16 heads, about 400M parameters |
| Attention | Spatial layers within each frame, causal temporal layers across frames |
| Context | 24 frames at 5 Hz, so 4.8 seconds of temporal memory |
| Action token | 7D relative local UR5 end-effector delta: translation, rotation vector, gripper |
| Training objective | Diffusion Forcing / flow matching velocity loss in DINO feature space |
| Rollout | Euler denoising with KV caching, generated latents fed back at signal level 0.8 |
| Planning | CEM over action sequences, scored by terminal DINO feature distance plus action penalty |

The thesis token layout per frame was:

```text
[signal | action | register_1..register_4 | 256 DINO patch tokens]
```

Action-free video uses the same layout. When no action is available, the action position stays as a learned base token. This made it possible to train one model from both action-labeled UR5 data and passive video without changing the input interface halfway through training.

## Training Data

| Dataset | Weight in final mixture | Actions | Role |
|---|---:|---|---|
| UR5 recordings | 25% | yes | Target robot embodiment and action-conditioned transitions |
| BridgeData V2 | 15% | no | Robot manipulation video from a different setup |
| EPIC-KITCHENS | 30% | no | Human-object interaction and contact-rich egocentric video |
| DROID | 30% | no | Diverse in-the-wild robot manipulation video |

Training used distributed data parallelism, bfloat16 autocast, AdamW, gradient clipping, and four NVIDIA H200 GPUs on the University of Tartu UT Rocket cluster.

## Results

All quantitative results below are held-out UR5 evaluation results. Metrics are mean squared error in normalized DINO latent space, averaged over patch tokens and feature dimensions. Lower is better.

### Mixed Video Delayed Overfitting

The UR5-only model reached its best checkpoint after only 2,000 optimizer steps and then overfit. Adding external robot and human-interaction video let training continue much longer and improved rollout quality.

<p align="center">
  <img src="assets/readme/mixed_data_results.png" alt="Mixed data rollout results" width="82%">
</p>

| Training mixture | Best step | Teacher forcing | Rollout H=5 | Rollout H=10 | Rollout H=20 |
|---|---:|---:|---:|---:|---:|
| UR5 only | 2,000 | 0.166 | 0.224 | 0.257 | 0.312 |
| 50% UR5, 50% mixed | 5,500 | 0.153 | 0.210 | 0.238 | 0.280 |
| 25% UR5, 75% mixed | 15,500 | **0.141** | 0.186 | 0.210 | 0.242 |
| 10% UR5, 90% mixed | 35,500 | 0.148 | **0.178** | **0.199** | **0.236** |

The 10% UR5 mixture gave the lowest rollout errors, but it took more than twice as many steps as the 25% UR5 setting and slightly worsened one-step prediction. For this model size and compute budget, 25% UR5 was the better practical tradeoff.

### High-Dimensional DINO Latents Needed Their Own Noise Schedule

A DINO frame is much larger than a compact VAE latent. The final model used a resolution-shifted signal schedule that samples more heavily from low signal levels, where the input is closer to noise.

<p align="center">
  <img src="assets/readme/noise_schedule_distributions.png" alt="Signal-level sampling schedules" width="72%">
</p>

| Signal schedule | Best step | Teacher forcing | Rollout H=5 | Rollout H=10 | Rollout H=20 |
|---|---:|---:|---:|---:|---:|
| Uniform | 15,500 | 0.142 | 0.213 | 0.245 | 0.283 |
| Logit-normal | 16,000 | **0.141** | 0.205 | 0.232 | 0.263 |
| Resolution-shifted | 15,500 | **0.141** | **0.186** | **0.210** | **0.242** |

The one-step teacher-forced numbers look almost identical for logit-normal and resolution-shifted training. The difference appears once predictions are fed back autoregressively: the resolution-shifted schedule is clearly better at longer horizons.

### Clean Feedback Was Not Best

During rollout, generated latents are inserted back into the temporal cache. Feeding them back as perfectly clean states was worse than adding a controlled amount of corruption. The best long-horizon result came from signal level `0.8`.

<p align="center">
  <img src="assets/readme/rollout_feedback_signal.png" alt="Rollout feedback signal sweep" width="72%">
</p>

This was a useful finding because it goes against the obvious default. In this high-dimensional DINO space, partially corrupted generated history kept the rollout distribution closer to what the model saw during Diffusion Forcing training.

### Action-Conditioned Rollouts Stayed Coherent

For qualitative evaluation, the model receives 10 clean context frames, then only the future UR5 action sequence. The rollout is decoded with the RAE decoder for inspection; training and metrics stay entirely in DINO feature space.

<p align="center">
  <img src="assets/readme/rollout_comparison_part1.png" alt="UR5 action-conditioned rollout comparison part 1" width="100%">
</p>

<p align="center">
  <img src="assets/readme/rollout_comparison_part2.png" alt="UR5 action-conditioned rollout comparison part 2" width="100%">
</p>

The useful part: the background does not drift away and the robot motion follows the provided actions over the shown horizon. The weak part is also visible: object identity becomes unstable during contact. The model learned robot kinematics more reliably than contact-rich object dynamics.

## Offline Planning

The planner encodes the current image and a goal image with DINOv2, samples candidate UR5 action sequences, rolls them out through the world model, and scores the terminal predicted state by DINO feature distance to the goal. The decoder is only used afterward to see what the selected latent plan looks like.

Before using DINO distance as a cost, I compared it with pixel MSE on a held-out UR5 sequence with brightness and contrast perturbations. Pixel distance was much more sensitive to photometric changes. DINO distance tracked progress toward the goal more smoothly.

<p align="center">
  <img src="assets/readme/dino_vs_pixel_distance.png" alt="DINO feature distance compared with pixel distance" width="82%">
</p>

The CEM run shown in the thesis used 30 iterations, population 64, elite fraction 0.1, a 10-step action horizon, rollout signal level 0.8, 20 Euler denoising steps, and action penalty weight 0.2.

<table>
  <tr>
    <td align="center"><strong>CEM latent cost</strong></td>
    <td align="center"><strong>Terminal translation error</strong></td>
  </tr>
  <tr>
    <td><img src="assets/readme/cem_cost_convergence.png" alt="CEM cost convergence"></td>
    <td><img src="assets/readme/cem_translation_error.png" alt="CEM translation error"></td>
  </tr>
</table>

The cost decreases, the sampling distribution contracts, and the final end-effector translation error reaches the centimeter to sub-centimeter range, with combined terminal translation error around 1 cm.

<p align="center">
  <img src="assets/readme/cem_trajectory_3d.png" alt="CEM 3D trajectory evolution" width="58%">
</p>

The decoded rollout below is the important sanity check: the optimized action sequence produces a coherent imagined motion toward the visual goal, not just a lower number in latent space.

<p align="center">
  <img src="assets/readme/cem_planning_rollout.png" alt="Decoded CEM planning rollout" width="100%">
</p>

This is still offline planning. One illustrative CEM run took roughly two minutes on one H200 GPU, so the current diffusion planner is too slow for real-time control without distillation, fewer denoising steps, a learned action proposal, or another acceleration method.

## What the Thesis Baseline Shows

- Frozen DINOv2 features are a practical state space for visual robot world models.
- Mixed passive video and robot video reduced overfitting on a small UR5 dataset.
- Long-horizon rollout quality depended strongly on the signal schedule and feedback corruption level.
- DINO-space goal costs were meaningful enough for an offline CEM planner to find a visually coherent action sequence.
- The main failure mode is still object interaction: contact, occlusion, and small manipulated objects are harder than robot arm motion.

## Repository Map

| Path | Purpose |
|---|---|
| `src/world_model/` | Block-causal spatial-temporal backbone, linear/DH heads, and autoregressive latent rollout |
| `src/diffusion/` | Signal schedules, LTG patch forcing, and Euler solver |
| `src/rae_dino/` | Pinned RAEv2 profiles and DINOv3 loader, plus the legacy DINOv2 RAE wrapper |
| `src/dataset/` | LeRobot-style dataset loading, mixing, padding, and action preprocessing |
| `src/training/` | Trainer, evaluator, logging, and distributed training utilities |
| `src/planning/` | CEM planner, action token builder, planning data loading, visualizations |
| `scripts/data/` | Dataset conversion utilities |
| `scripts/viz/reconstruct_autoencoder.py` | Reproducible RAEv2 encode/decode check on CPU, CUDA, or MPS |
| `scripts/train_world_model.py` | Main world-model training entrypoint |
| `scripts/evaluate_checkpoint.py` | Held-out rollout evaluation |
| `scripts/plan_with_world_model.py` | Goal-conditioned CEM planning |
| `archive/autoencoder_reconstruction/` | Checked-in RAEv2 reconstruction images and metrics |
| `tests/` | Unit and smoke tests for profiles, conditioning, heads, noising, rollout, and configs |
| `thesis/` | Thesis source, PDF, and original figures |

## Limitations

The planning results are offline diagnostics on held-out UR5 episodes, not closed-loop robot execution. The model is also still weak on contact-rich object interactions, and the planner is far too slow for direct real-time control. Those are the next engineering problems: faster sampling, better object-interaction modeling, and closed-loop validation on the robot.
