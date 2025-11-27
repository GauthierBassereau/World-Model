# World Model | Master Thesis (in progress)

## 1. Motivation and Scope
- Build a latent world model that combines Dreamer V4 shortcut forcing with DINO-style latent planning to support goal-conditioned MPC and offline RL.
- Data regime: train a world model on non-robot data and robot data. Kinetics 700 for learning general world dynamcis and DROID (using LeRobotDataset) for learning robot dynamics and actions' impact on world.

## 2. Research Foundations

- Image encoder -> Dinov2 RAE autoencoder
- World model's architecure -> Dreamer 4, Spatial Temporal Transformer
- World model's training objective -> Shortcut Forcing from Dreamer 4
- Model Predictive Control on dino features -> DINO-WM

## 3. Integrated Architecture

- Freeze the Dinov2 encoder and use MAE decoder from RAE paper (for visualization purposes).
- Use the Dreamer V4 block-causal transformer, alternating spatial-only and temporal-only attention layers, QKNorm, attention logit soft capping, pre-layer RMSNorm, and SwiGLU MLPs.
- Sequence per step: `[register tokens | shortcut tokens | action tokens | dino tokens]`

## 4. Data

-  Learning diverse world dynamics, interactions between objects -> Kinetics 700 (~700Gb) ---(?)--> OpenVid1M, HowTo100M, Ego4D, Something-Something V2, EPIC-KITCHENS, EgoDex
-  Learning robot interactions + action conditioning with world -> DROID (~400Gb) ---(?)--> SOAR, agibot_alpha

Note: most of these datasets have language instructions for each video -> I wonder how fast the WM can learn to condition its generation on text instead of action ?

---

Todos:
  - For inference need to add noise to past at 0.1, not just the noise level token has it !!!
  - Create a script to visualize the distribution of latents compared to normal distribution
  - Implement the option to have x-loss with reweighting just like Dreamerv4
  - Feature to get only valid sequences in droid, no padded frames, more efficient training...
  - Right now if fps is not correct in kinetics it raises an error and so worlddataset try another random item, but this also skips the other datasets times... need to have the try and except in each dataset I think to prevent this. Also the worlddataset should probably have a reference dataset so the epoch is defined by its length and not the biggest dataset (which is probably the open images v7)
  - Weigth_decay with no warmup ?
  - EMA impact ?
  - Look for better handling of ODE solver and sampling on edges 0 and 1, look at RAE, for Euler but also maybe Heun
  - Shortcut forcing
  - (Maybe to try someday) Add a DH head to backbone following RAE paper -> I am really not sure this is useful because I think RAE paper uses v-space prediction, and so that's why DH head is so effective as explained in the JIT paper