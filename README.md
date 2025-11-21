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

-  Learning diverse world dynamics, interactions between objects -> Kinetics 700 (~700Gb) ---(?)--> OpenVid1M, HowTo100M, Ego4D, Something-Something V2, EPIC-KITCHENS
-  Learning robot interactions + action conditioning with world -> DROID (~400Gb) ---(?)--> SOAR, agibot_alpha

Note: most of these datasets have language instructions for each video -> I wonder how fast the WM can learn to condition its generation on text instead of action ?

---

Todos:
  - Add a log of time/step
  - Add a script that takes a run and create a graph with Y axis the L1 loss on eval and X axis the rollout horizon (t+1, t+2 etc...) to see accumulation and error. And plot the curve for different eval timesteps to see the evolution.
  - [In progress] split droid dataset ? somehow without burning all hpc credits
  - Fix eval dataset for eval, converting index of entire dataset to the subset one ?
  - Take the rollout function out of the eval, I think it should be something appart that can be called for other things, I am thinking for the futur interactive world model. And then I think I can merge the rest of the eval code direclty into trainer. Need to be careful to not make this too hard to read, has to be super clean.
  - Also the whole z predicted trnasformed to velocity, I think this transformation should be a function, maybe put it in flow matching, I can reuse it later, and even in Euler solver.
  - Implement the opetion to have x-loss with reweigthing just like Dreamerv4
  - Shortcut forcing
  - (Maybe to try someday) Add a DH head to backbone following RAE paper -> I am really not sure this is useful because I think RAE paper uses v-space prediction, and so that's why DH head is so effective as explained in the JIT paper