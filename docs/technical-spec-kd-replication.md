# Technical Specification: Faithful K&D Replication
### `evo-reward` — Phase 0 and Phase 1a

*All numbers drawn directly from Kanagawa & Doya (2025), arXiv:2507.09992, Appendix A and Tables 2–4. Where the paper is ambiguous, the decision is noted explicitly. The goal of this document is that a coding agent can implement the full simulation without reading the paper.*

---

## 1. World and Physics

| Parameter | Value | Source |
|-----------|-------|--------|
| World size | 960 × 960 units | Appendix A |
| Boundary | Square, impassable walls | Section 3 |
| Physics | 2D rigid-body | Section 3 |
| Timestep | 1 simulation step | (implicit) |
| Total steps per run | 10.24 million (1024 × 10⁴) | Section 4 |

Agents move by applying force to two rear points on their body — `f = (f_left, f_right)^T`. This differential drive produces both translation and rotation. The physics engine resolves collisions and wall bounces each step.

---

## 2. Agent Bodies

| Parameter | Prey | Predator | Source |
|-----------|------|----------|--------|
| Body shape | Circle | Circle | Section 3 |
| Radius | 10 units | 14 units | Appendix A |
| Motor outputs | 2 scalars (left/right force) | 2 scalars | Section 3 |
| Max motor output norm `F` | 114 (derived: `0 < ‖f‖ < 114`) | 114 | Appendix A note |

---

## 3. Sensors

Each agent has two sensor arrays:

### 3.1 Proximity Sensors (vision-like)

| Parameter | Value | Source |
|-----------|-------|--------|
| Count | 32 sensors | Section 3 |
| Field of view | 120-degree forward arc | Section 3 |
| Max range | 200 units | emevo code (`sensor_length = 200.0`); paper Appendix A says 120 — see D8 |
| Channels per sensor | 4 (prey, predator, food, wall) | emevo code; winner-take-all per sensor — see D7 |
| Output encoding | Inverse distance, scaled to [0, 1] | Section 3 |
| Object types detected | Food, other agents (prey/predator), walls | Section 3 |
| Reading formula | `s^k = 1` on contact, `s^k = 0` when no object in range | Section 3 |

Each sensor has **4 per-type channels** (prey, predator, food, wall). Within each sensor, only the channel for the closest detected object type gets a positive reading; others are set to -1.0 (winner-take-all). This produces a sensor array of shape (32, 4) = 128 values. The 32 sensors are distributed evenly across the 120° arc.

> **Code vs. paper:** The paper Appendix A states max range = 120 units, but the emevo source (`config/env/20251122-predator-square.toml:12`) uses `sensor_length = 200.0`. We follow the code. See deviation D8 in `emevo-diff.md`.

For the reward function, the relevant aggregation is `max_k s^{i,k}_{pred}` (maximum proximity sensor reading for predators) and `max_k s^{i,k}_{prey}` (maximum for conspecifics). This is the scalar fed into the reward.

### 3.2 Tactile / Collision Sensors

| Parameter | Value | Source |
|-----------|-------|--------|
| Count | 18 sensors | Section 3 |
| Spacing | 20-degree intervals around body | Section 3 |
| Function | Detect local contact; approximate collision location | Section 3 |

### 3.3 Full Observation Vector (Policy Input)

The policy MLP receives, at every step:
- Proximity sensor readings: 32 sensors × 4 channels = **128 values** (each in [-1, 1])
- Tactile sensor readings: 4 types × 18 bins = **72 values** (binary contact per type per bin)
- Agent's own velocity: **2 values** (vx, vy) — 2D, not scalar speed
- Agent's own heading angle: 1 value
- Agent's own angular velocity: 1 value
- Agent's own energy level: 1 value

**Total obs dimension: 205 scalars** (128 + 72 + 2 + 1 + 1 + 1). Verified against emevo source (`circle_foraging.py:740-748`). See deviation D7 in `emevo-diff.md`.

---

## 4. Eating and Predation

### Prey eating food

- Prey eat food by making contact within a **120-degree forward range** (same arc as proximity sensors)
- Contact distance: within the mouth range (120 units forward)
- Energy gain per food item: `e_food = 1.0`

### Predator hunting prey

- Predators hunt by initiating contact within a **mouth range** (the "medium" default is a 60-degree arc, range 40–80 units — see Figure 3)
- Default mouth range for baseline experiment: **medium (60°)**
- Energy gain per prey caught: determined by digestion rate `η` — predators gain approximately 6–10 energy units per catch (see energy update equation, Section 3)
- Predators **cannot** eat food items

---

## 5. Food Dynamics

| Parameter | Value | Source |
|-----------|-------|--------|
| Max food items `n_max` | 600 | emevo code (`n_max_foods = 600`); paper Appendix A says 100 — see D5 |
| Food growth rate `g` | 0.5 items/step (linear) | emevo code (`food_num_fn = ["linear", 40, 0.5, 600]`) |
| Regeneration rule | `n_{t+1} = min(n_t + g - n_t^{eaten}, n_max)` | Section 3 |
| Spawn location | Random position in world | Section 3 |
| Food regeneration trigger | When integer part of `n_t` exceeds current food count | Section 3 |
| Default Δn (food rate param) | 0.5 (used in main experiments) | Section 4 |

Food is regenerated at a random location each time. The food count `n_t` is an internal real-valued counter; physical food items are spawned when `floor(n_t)` exceeds the current item count.

---

## 6. Energy Dynamics

### 6.1 Energy Update (per step)

**Prey:**
```
Δe^i_j = n^i_j * e_food - c_a * ‖a^i_j‖ - c_b
```

**Predator:**
```
Δe^i_j = Σ_{k ∈ Prey eaten by i} η * e^k - d_a * ‖a^i_j‖ - d_b
```

Where:
- `n^i_j` = number of food items eaten at step j
- `‖a^i_j‖` = Euclidean norm of the motor output vector
- `η` = predator digestion rate (energy fraction extracted from prey)

### 6.2 Energy Consumption Parameters

| Parameter | Value | Applies to | Source |
|-----------|-------|------------|--------|
| `c_b` (prey basal metabolism) | 1.0 × 10⁻⁴ | Prey | emevo code (`basic_energy_consumption`) |
| `c_a` (prey motor cost) | 2.5 × 10⁻⁶ | Prey | emevo code (`force_energy_consumption`) |
| `d_b` (predator basal metabolism) | 4 × 10⁻³ | Predator | Table 2 (matches code) |
| `d_a` (predator motor cost) | 5 × 10⁻⁵ | Predator | Table 2 (matches code) |

> **Code vs. paper (prey only):** Paper Table 2 lists `c_b = 2.5×10⁻³` and `c_a = 1×10⁻⁴`, but the emevo source (`config/env/20251122-predator-square.toml:24-25`) uses `basic_energy_consumption = 1e-4` and `force_energy_consumption = 2.5e-6`. We follow the code. Predator values match between paper and code.

**Note from paper:** "Because `0 < ‖f‖ < 114`, `c_a ‖a‖_j` is twice as large as `c_b` when the motor output is maximum. Also, predators consume about 10 times as much energy as prey." This sanity check applies to the paper's Table 2 values; the code values produce different ratios but are what the experiments actually used.

---

## 7. Birth and Death

### 7.1 Hazard Function (Death Probability per Step)

```
h(t, e) = κ_h * (1 - 1 / (1 + α_e * exp(-β_h * e))) * α_t * exp(β_t * t)
```

Where `t` is the agent's age in steps and `e` is current energy.

| Parameter | Prey | Predator | Source |
|-----------|------|----------|--------|
| `κ_h` | 0.01 | 0.01 | Table 3 |
| `α_e` | 0.02 | 0.02 | Table 3 |
| `β_h` | 0.2 | 0.2 | Table 3 |
| `α_t` | 4 × 10⁻⁷ | 2 × 10⁻⁷ | Table 3 |
| `β_t` | 2 × 10⁻⁶ | 4 × 10⁻⁶ | Table 3 |
| `ζ` | 10 | 100 | Table 3 |

**Additional death rule:** Agents also die deterministically if `e < 0`.

**Interpretation from paper:** "The survival probability significantly decreases when `e < 20` while the effect of aging is milder." Maximum expected lifetime is approximately 1 × 10⁶ steps.

### 7.2 Birth Function (Reproduction Probability per Step)

```
b(e) = κ_b / (1 + exp(ζ - β_b * e))
```

Wait — the paper's Equation 3 is: `b(e) = κ_b / (1 + exp(ζ - β_b * e))`... let me reconcile: the paper defines `b(e) = κ_b / (1 + exp(-β_b * e))` with a shift parameter `ζ`. The exact form from the paper (Eq. 3) is:

```
b(e) = κ_b / (1 + exp(ζ - β_b * e))
```

Where `ζ` shifts the inflection point (energy threshold for reproduction).

| Parameter | Value | Source |
|-----------|-------|--------|
| `κ_b` | 1 × 10⁻³ | Table 3 |
| `β_b` | 0.1 | Table 3 |
| `ζ` (prey) | 10 | Table 3 |
| `ζ` (predator) | 100 | Table 3 |

**Interpretation:** "Prey require between 20 and 30 energy units for reproduction, while predators require a much higher range of 240 to 260 units." (Section 3) This is the key constraint that keeps predator populations small.

### 7.3 Reproduction Mechanics

- Reproduction is **asexual**
- Offspring inherits parent's reward weights with mutation (see Section 9)
- Offspring spawned at a random location sampled from a Gaussian centered on the parent
- Parent loses fraction `energy_share_ratio = 0.4` of its energy; child receives the same amount. (This is **not** the predator digestion rate η=0.6 — it is a separate parameter. Verified: `config/env/20251122-predator-square.toml:26` `energy_share_ratio = 0.4`.)
- Offspring gets a **fresh, randomly initialized policy network** — no policy inheritance

### 7.4 Population Caps

| Species | Cap |
|---------|-----|
| Prey | 450 |
| Predator | 50 |

Source: Section 3. These caps prevent runaway population growth.

### 7.5 Initial Population

| Species | Initial count |
|---------|--------------|
| Prey | 150 |
| Predator | 10 |

Source: Section 4.

---

## 8. Reward Function

The reward for agent `i` at step `j` is:

```
r^i_j = w^i_eat * n^i_j
      + 0.01 * w^i_act * (1/F) * ‖f^i_j‖
      + 0.1  * w^i_prey * max_k s^{i,k}_prey
      + 0.1  * w^i_pred * max_k s^{i,k}_pred
```

Where:
- `n^i_j` = food items / prey eaten at step j (usually 0, occasionally 1)
- `‖f^i_j‖` = Euclidean norm of motor output (normalized by `F = 114`)
- `max_k s^{i,k}_prey` = maximum proximity sensor reading for conspecifics
- `max_k s^{i,k}_pred` = maximum proximity sensor reading for predators/prey (opposing species)
- `w^i_eat, w^i_act, w^i_prey, w^i_pred` = the **reward genome** — four evolved scalar weights

**Scaling note:** The 0.01 and 0.1 coefficients are fixed — they scale the smaller, continuous signals (motor, sensor) to be comparable in magnitude to the sparse eating signal. These are NOT part of the genome; they are fixed architecture.

**For prey:** `max_k s^{i,k}_pred` reads predator proximity sensors; `max_k s^{i,k}_prey` reads other prey proximity sensors.

**For predators:** The same equation applies with `w_prey` tracking prey proximity and `w_pred` tracking other predator proximity. Predator `n^i_j` counts prey caught (not food).

---

## 9. Reward Genome: Initialization and Mutation

### 9.1 Initialization

Reward weights `(w_eat, w_act, w_prey, w_pred)` for each agent in the initial population are sampled from:
```
w ~ N(0, 0.1)
```
Each weight independently.

### 9.2 Mutation

At reproduction, the child's reward weights are:
```
w'_child = clip(w_parent + δ, -100, 100)
```

Where `δ` is sampled from a **Student's t-distribution with 2 degrees of freedom** and **scale 0.4**:
```
δ ~ t(df=2, scale=0.4)
```

This is heavy-tailed (Cauchy-like), enabling occasional large jumps in reward space. The clip to [-100, 100] prevents numerical instability but is rarely active in practice.

**Critical note:** The paper says "Student's t-distribution with 2 degrees of freedom and a scale of 0.4." This is NOT a Gaussian. Use `scipy.stats.t(df=2, scale=0.4)` or equivalent. Earlier project documents say "Cauchy" — the t(df=2) is close but not identical to Cauchy (t with df=1). Use t(df=2) as specified in the paper.

---

## 10. Policy Network

| Parameter | Value | Source |
|-----------|-------|--------|
| Architecture | 3-layer MLP | Section 3 |
| Hidden units | 64 per layer | Table 4 |
| Input | Full observation vector (sensors + proprioception + energy) | Section 3 / Figure 4 |
| Output | 2 motor forces, as Gaussian distribution (mean + state-independent variance) | Section 3 |
| Action sampling | Sample from Gaussian policy | Section 3 |
| Policy per agent | **Independent** — each agent has its own network | K&D faithful |

Each newborn starts with a **randomly initialized** policy network. There is no policy inheritance.

---

## 11. PPO Hyperparameters

All from Table 4 of the paper.

| Parameter | Value |
|-----------|-------|
| Discount factor γ | 0.999 |
| Rollout steps N | 1024 |
| Minibatch size | 256 |
| Optimization epochs per update | 10 |
| PPO clipping parameter ε | 0.2 |
| Entropy coefficient | 0.001 |
| GAE parameter λ | 0.95 |
| Adam learning rate | 3 × 10⁻⁴ |
| Adam ε | 1 × 10⁻⁷ |

**PPO update schedule:** Policy updates happen **occasionally**, not every step. From Algorithm 1 (Figure 5 in the paper): "Once in N steps, update agent's policy via RL." Each agent accumulates N=1024 steps of experience, then performs 10 epochs of PPO updates on that rollout. In K&D's independent-policy setup, each agent runs its own PPO update when it has accumulated N steps — this is O(agents) updates per N steps.

**Value function clipping:** Not specified in the paper. Use standard PPO with value function clipping (`clip_vloss=True`) as a safe default.

**Normalization:** Not specified. Use observation normalization via running mean/std (standard practice). Confirm against emevo source.

---

## 12. Simulation Loop (Algorithm 1)

Per step `j`:

1. Each agent `i` observes the environment → constructs `o^i`
2. Each agent samples motor action `f^i` from its policy `π(·|o^i)`
3. Step physics: apply motor forces, resolve collisions, update positions
4. Update each agent's energy level `e^i` (Eq. 1)
5. Process birth and death:
   - Agent dies if `e^i < 0` OR with probability `h(t^i, e^i)`
   - Agent reproduces with probability `b(e^i)` (if alive); offspring inherits mutated reward genome + fresh policy
6. Compute reward `r^i_j` for each agent (Eq. from Section 8 above)
7. Once every N=1024 steps (per agent): update agent's policy via PPO on the accumulated rollout
8. Regenerate food: update `n_t` and spawn items as needed

**Key subtlety:** Steps 5 and 6 interact — an agent that dies in step 5 does not receive a reward in step 6. Newborns start accumulating their first rollout from step 0 of their life.

---

## 13. Metrics to Log (Every Generation)

The paper's notion of "generation" is approximate (no discrete generations in continuous birth-death). Log metrics every **N_log steps** (e.g., every 10,000 steps = ~10 reward updates per agent) and every time the population crosses a generation boundary (tracked via total births).

### Required for replication validation:

- **Reward weight trajectories** — mean of each weight across all living agents, per timestep. This is K&D Figure 7.
- **Reward weight KDE** — kernel density estimate of the population distribution of each weight at the end of the run (last 5000 agents born). This is K&D Figure 8 / 12.
- **Population size** — count of living prey and predators per timestep. This is K&D Figure 6.
- **Mean energy** — mean energy of living agents per species.
- **Capture rate** — prey caught per step by predator population.
- **Food consumption rate** — food items eaten per step by prey population.

### Checkpoint format:

Save every 25,000 steps (approximately). Each checkpoint: `{step}_{seed}.pkl` containing:
- All agent reward genomes (weights)
- All agent ages and energies
- Population sizes
- Config used

---

## 14. Success Criteria for Phase 1a

A run is considered a successful replication if, by step 10 million:

1. **Fear emerges:** Mean `w_pred` for prey is **negative** across at least 3 of 5 seeds.
2. **Social affiliation emerges:** Mean `w_prey` for prey is **positive** across at least 3 of 5 seeds.
3. **Food reward is positive:** Mean `w_eat` for both species is **positive** across all seeds (this is the easiest to replicate).
4. **Population oscillations visible:** Prey and predator population time series show Lotka-Volterra-like coupled oscillations with period ~1 million steps.
5. **No extinction:** Neither species goes extinct in any seed.

**Quantitative reference from paper:** Prey evolved ~473–501 generations, predators ~47–59 generations over 10.24 million steps under default conditions. Generation count is a useful sanity check on lifecycle dynamics.

---

## 15. Codebase Structure for Phase 0

Implement in this exact order. Each module has a single clear responsibility. Do not mix concerns.

```
src/
├── environment.py     # World state, physics step, food dynamics, collision detection
├── agents.py          # Observation construction from world state; reward computation
├── policy.py          # MLP policy network; action sampling; independent-policy management
├── reward.py          # Reward genome (linear weights); reward computation given genome + obs
├── evolution.py       # t(df=2) mutation; reward weight clipping; newborn initialization
├── lifecycle.py       # Birth probability b(e); hazard h(t,e); birth/death events each step
├── ppo.py             # PPO update (rollout buffer, GAE, clipped surrogate, value loss)
└── metrics.py         # All logging; checkpoint save/load; KDE computation

configs/
└── baseline_faithful.yaml   # All parameters from this spec, K&D defaults

scripts/
└── run_experiment.py        # Entry point: load config, init world, run loop, save results
```

### `baseline_faithful.yaml` — complete parameter listing

```yaml
# Identity
experiment_name: baseline_faithful
policy_mode: independent
lifecycle_mode: continuous
reward_type: linear
social_obs: position_only
policy_type: mlp
coevolution_mode: concurrent

# World
world_size: 960
total_steps: 10_240_000

# Population
prey_initial: 150
predator_initial: 10
prey_cap: 450
predator_cap: 50

# Agent bodies
prey_radius: 10
predator_radius: 14
max_motor_norm: 114.0  # F

# Sensors
n_proximity_sensors: 32
proximity_fov_deg: 120
proximity_max_range: 120
n_tactile_sensors: 18
tactile_spacing_deg: 20

# Food
food_max: 100
food_growth_rate: 0.02  # g
food_regen_rate: 0.5    # Δn

# Energy — prey
prey_e_food: 1.0
prey_c_b: 2.5e-3
prey_c_a: 1.0e-4

# Energy — predator
predator_d_b: 4.0e-3
predator_d_a: 5.0e-5
predator_eta: 0.6        # digestion rate; confirm exact value from emevo source

# Predator mouth
predator_mouth_deg: 60   # medium (default)
predator_mouth_range_min: 40
predator_mouth_range_max: 80

# Hazard function h(t, e)
kappa_h: 0.01
alpha_e: 0.02
beta_h: 0.2
alpha_t_prey: 4.0e-7
alpha_t_pred: 2.0e-7
beta_t_prey: 2.0e-6
beta_t_pred: 4.0e-6
zeta_prey: 10.0    # NOTE: this parameter name is overloaded; check paper's exact formulation
zeta_pred: 100.0

# Birth function b(e)
kappa_b: 1.0e-3
beta_b: 0.1
zeta_b_prey: 10.0   # inflection point (energy for 50% birth prob)
zeta_b_pred: 100.0

# Reward genome
reward_weights_init_std: 0.1
mutation_df: 2          # t-distribution degrees of freedom
mutation_scale: 0.4
weight_clip: 100.0

# Policy network
policy_hidden_size: 64
policy_n_layers: 3

# PPO
gamma: 0.999
rollout_steps: 1024
minibatch_size: 256
ppo_epochs: 10
clip_epsilon: 0.2
entropy_coef: 0.001
gae_lambda: 0.95
lr: 3.0e-4
adam_eps: 1.0e-7

# Logging
checkpoint_interval_steps: 25_000
log_interval_steps: 10_000

# Seeds (run separately)
seed: 0  # override per run; K&D used 5 seeds
```

---

## 16. Open Questions / Items to Verify Against emevo Source

The following are not fully specified in the paper and should be verified against `github.com/oist/emevo` before finalizing the implementation:

1. **Exact predator digestion rate η** — paper says "predators gain 6–10 energy per capture" but η is not explicitly tabled.
2. **Offspring spawn distribution** — "random location sampled from a Gaussian centered on the parent." What is the standard deviation of this Gaussian?
3. **Energy at birth for offspring** — not specified in the paper. Likely the parent's energy × η (parent pays η of its energy; offspring starts with that amount). Verify.
4. **Observation vector ordering** — the exact concatenation order of sensors/proprioception matters for weight initialization and PPO stability. Match emevo exactly.
5. **Value function normalization** — whether K&D normalize advantages or rewards before PPO updates.
6. **Food regeneration rate parameter interpretation** — Δn = 0.5 means one food item every two steps on average, but the exact formula using the internal counter `n_t` should be verified.
7. **Zeta parameter in hazard function** — Table 3 lists `ζ = 10 (prey) / 100 (predator)` but this parameter doesn't appear in the hazard function `h(t, e)` as written (it appears in `b(e)`). Verify whether ζ is shared between h and b or separate.

---

## 17. Extensions — Placeholder Section (Not for Phase 0–1a)

The following are out of scope for the faithful replication phase. They are noted here so the codebase can be structured to support them without requiring rewrites:

- **Shared policy** (`policy_mode: shared`): One policy network per species, conditioned on reward genome. Implement as config flag; do not build yet.
- **MLP reward genome** (`reward_type: mlp`): Replace 4-scalar genome with small MLP. Implement as config flag; do not build yet.
- **Social observation** (`social_obs: position_heading_velocity`): Expand obs vector. Design obs layout to allow this extension by concatenation.
- **Temporal reward context** (`reward_context_window: k`): Rolling buffer of past k observations fed to reward function. Design reward.py interface to accept a buffer rather than a single obs.
- **LSTM policy** (`policy_type: lstm`): Policy network with recurrent hidden state. ppo.py must handle rollout buffering for recurrent networks.

The config flags for all of these should exist in `baseline_faithful.yaml` but point to the baseline/faithful variants.
