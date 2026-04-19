# Development Roadmap
### `evo-reward` — Full Project Lifecycle

*This document covers all phases of development, from K&D replication through extension experiments, plus an ongoing track of improvements and enhancements that sit outside the main experimental timeline. It is the master planning document for coding agents and contributors.*

---

## The Inviolable Rule

**Phase 0 and Phase 1a (K&D replication) must be complete and validated before any other work begins.** No extension code, no visualization tooling, no performance optimization, no new features — until fear and social affiliation reliably emerge from the baseline. Everything else in this document is conditional on that.

---

## Phase Structure Overview

```
Phase 0   Infrastructure + smoke test                     [~1 week]
Phase 1a  Faithful K&D replication — the proof of concept [~1-2 weeks, 1-2 seeds]
Phase 1b  Architectural comparison (shared vs. independent) [~1 week]
Phase 2   One-axis ablations                              [~2 weeks]
Phase 3   Key combinations                                [~2 weeks]
Phase 4   Depth investment + report                       [~2 weeks]
Phase 5   Relaxing simplifications (stretch/post-semester)
─────────────────────────────────────────────────────────
Track E   Engineering improvements (ongoing, never blocks science)
```

---

## Phase 0: Infrastructure

**Goal:** A working simulation loop that can be validated mechanically — before any evolutionary signal is expected.

**Build order (strict — each step depends on the previous):**

1. `environment.py` — 2D world, rigid-body physics, food spawning, proximity/tactile sensors, wall collisions
2. `lifecycle.py` — energy update equations, hazard `h(t,e)`, birth `b(e)`, death and reproduction events
3. `agents.py` — observation vector construction from world state; reward computation from genome
4. `reward.py` — linear reward genome (4 weights); reward scalar from genome + obs stimuli
5. `evolution.py` — Student's t(df=2, scale=0.4) mutation; weight clipping; newborn initialization
6. `policy.py` — independent per-agent MLP (2 hidden layers, 64 units, tanh); sigmoid action mapping to [-20, 80]; Gaussian action head
7. `ppo.py` — rollout buffer; GAE; clipped surrogate loss; value loss; entropy bonus; Adam optimizer
8. `metrics.py` — reward weight logging; population size; checkpoint save/load
9. `scripts/run_experiment.py` — single-command entrypoint from config YAML

**Phase 0 validation gate** (all must pass before Phase 1a):

See `tests/test_phase0.py` — automated checks listed in the Testing section below.

---

## Phase 1a: Faithful K&D Replication

**Config:** `configs/baseline_faithful.yaml` — see `docs/technical-spec-kd-replication.md` for all exact parameter values.

**Run:** `python scripts/run_experiment.py --config configs/baseline_faithful.yaml --seed 0`

**Duration:** ~10–12 hours per seed on A100. Run 1–2 seeds first; if qualitative result emerges, add seeds 2–5.

### Success criteria (quantitative, checked by `scripts/validate_replication.py`)

| Criterion | Threshold | K&D reference |
|-----------|-----------|---------------|
| Mean prey `w_pred` < 0 at step 10M | At least 3 of 5 seeds | Figure 7 |
| Mean prey `w_prey` > 0 at step 10M | At least 3 of 5 seeds | Figure 7 |
| Mean prey `w_eat` > 0 at step 10M | All seeds | Figure 7 |
| Mean predator `w_prey` > 0 at step 10M | At least 3 of 5 seeds | Figure 7 |
| Prey population oscillates (not flat, not extinct) | All seeds | Figure 6 |
| Predator population oscillates | All seeds | Figure 6 |
| Oscillation period ~1M steps | At least 2 of 5 seeds | Figure 6 |
| Prey generations in run: ~473–501 | Within ±20% | Section 4 |
| Predator generations in run: ~47–59 | Within ±20% | Section 4 |
| No extinction (either species) | All seeds | Section 4 |

**Deliverable:** Plots matching K&D Figures 6, 7, and 8 (population dynamics, reward weight trajectories, reward weight KDE scatter). These are generated automatically by `analysis/dashboards.py`.

**Scope note — predator lethality sweep omitted:** K&D run three mouth-size conditions (small/medium/large, Figure 3) and report that larger mouths → stronger evolved fear. Phase 1a uses the **medium** condition only. The small/large conditions are treated as an optional post-Phase-1a follow-up, not a blocker for the gate above. Do not claim reproduction of K&D's environmental-sensitivity finding from medium-only data.

**If replication fails:** Stop. Do not proceed to Phase 1b. Debug using the K&D open-source code (`github.com/oist/emevo`) as ground truth — run their code on the same seed and compare step-by-step. The most likely failure modes, in order:
1. PPO not converging (agents not learning to eat) — check learning rate, rollout length
2. Evolution not running (weights not drifting) — check mutation scale, birth/death rates
3. Population going extinct — check energy parameters against `technical-spec-kd-replication.md`
4. Fear not emerging despite agents learning — check reward scaling coefficients (0.01, 0.1 in reward equation)

---

## Phase 1b: Architectural Comparison

**Config:** `configs/baseline_simplified.yaml` — shared policy + continuous birth-death, otherwise identical to Phase 1a.

**Goal:** Test whether the core result (fear + social affiliation) is robust to shared policy. This is both a validation step and a scientific contribution (H1–H7).

**Runs:** 1–2 seeds. Compare qualitatively to Phase 1a.

**Decision rule:**
- If fear + social affiliation emerge → shared policy becomes default for Phase 2+. Document quantitative differences (convergence speed, diversity, etc.) as results.
- If the core result breaks → fall back to independent policies for all extensions. Write up the failure as a finding.

---

## Phase 2: One-Axis Ablations

Run each of the four extension axes independently against the simplified baseline. 1 seed per condition for breadth scan; add seeds where signal appears.

| Condition | What changes | Key question |
|-----------|-------------|--------------|
| `axis1_mlp_reward.yaml` | Linear → MLP genome | Does evolution use nonlinearity? |
| `axis2_social_obs.yaml` | Position → pos+heading+vel | Does `w_prey` shift when behavior is observable? |
| `axis3_temporal_reward.yaml` | Instantaneous → context window k=10 | Does prediction-error-like structure emerge? |
| `axis4_lstm_policy.yaml` | FF MLP → LSTM policy | Does memory improve survival? Does evolution respond? |

---

## Phase 3: Key Combinations

Core 2×2 (temporal × social), LSTM interactions, and optionally iterative best response. 1 seed per new cell, then depth on what's interesting.

---

## Phase 4: Depth + Report

5 seeds on the 3–5 most interesting conditions. Full factorial breadth (1 seed each) on remaining cells. Report written alongside this phase.

---

## Phase 5: Relaxing Simplifications (Stretch)

Independent policies on key extensions; larger populations; generational batching comparison; IBR as full axis. Post-semester or if time permits.

---

## Testing

Tests live in `tests/`. They are separate from the main experiment scripts and should run in minutes, not hours. They use small synthetic configurations (tiny world, few agents, few steps) to check that components behave correctly.

### `tests/test_phase0.py` — Phase 0 validation gate

These must all pass before any experiment is run.

**Environment tests:**
- `test_world_bounds`: Agents placed at edges do not escape the 960×960 world
- `test_food_regeneration`: Food count approaches `n_max=100` from 0 within expected steps given `g=0.02`
- `test_food_regeneration_rate`: At `Δn=0.5`, food regenerates at ~1 item per 2 steps on average
- `test_sensor_contact`: Agent placed adjacent to food item reads proximity sensor = 1.0 for that sensor
- `test_sensor_range`: Agent at distance > 120 from all objects reads all proximity sensors = 0.0
- `test_sensor_fov`: Object directly behind agent (outside 120° arc) reads sensor = 0.0
- `test_collision_wall`: Agent moving into wall bounces back (position stays in bounds after 100 steps)
- `test_collision_agents`: Two overlapping agents separate (rigid body resolution)

**Energy and lifecycle tests:**
- `test_energy_decrease_idle`: Agent with no motor output and no food intake loses `c_b` energy per step
- `test_energy_decrease_motor`: Agent at max motor output loses approximately `c_b + c_a * F` per step
- `test_energy_gain_food`: Prey eating one food item gains exactly `e_food = 1.0` energy
- `test_energy_predation`: Predator catching prey gains `η * prey_energy` (verify η value from emevo source)
- `test_death_starvation`: Agent with e < 0 is removed from population on next step
- `test_death_hazard`: At very high age and low energy, `h(t, e)` approaches `κ_h` (numerically verify formula)
- `test_birth_probability`: Agent with e = 30 (prey) has `b(e) ≈ κ_b / (1 + exp(ζ - β_b * 30))` (verify formula)
- `test_no_birth_low_energy`: Agent with e = 5 has `b(e)` ≈ 0 (well below birth threshold)
- `test_offspring_fresh_policy`: Newborn agent's policy weights are independent of parent's policy weights
- `test_offspring_genome`: Newborn's reward weights are parent's weights + t(df=2) noise (statistical test over 1000 births)
- `test_mutation_distribution`: 10,000 mutation samples from t(df=2, scale=0.4); verify df=2 (not Gaussian, not Cauchy)

**Reward function tests:**
- `test_reward_food_term`: Agent eating 1 food item with `w_eat = 1.0` gets reward contribution = 1.0
- `test_reward_motor_scaling`: Motor reward is scaled by `0.01 / F`; verify numerically
- `test_reward_sensor_scaling`: Sensor reward is scaled by `0.1`; verify
- `test_reward_zero_genome`: Agent with all-zero genome gets reward = 0 at every step regardless of state
- `test_reward_negative_w_pred`: Agent with `w_pred = -10` near a predator gets negative reward contribution
- `test_reward_sign_fear`: Negative `w_pred` × positive sensor reading = negative reward (fear drives avoidance)

**PPO tests:**
- `test_ppo_loss_decreases`: PPO loss decreases over 5 update epochs on a fixed rollout
- `test_ppo_action_entropy`: Entropy coefficient > 0 maintains non-collapsed action distribution
- `test_ppo_value_estimate`: Value network output is non-constant after 100 update steps on non-trivial rollout
- `test_rollout_buffer_gae`: GAE returns for a constant reward sequence match closed-form solution
- `test_ppo_gradient_flow`: All policy network parameters receive non-zero gradients after one update

**Evolution + metrics tests:**
- `test_checkpoint_roundtrip`: Save and load a checkpoint; all agent genomes, ages, energies match
- `test_metrics_population_count`: Logged population count matches actual number of agents in world
- `test_metrics_weight_mean`: Logged mean reward weight matches mean of actual genome population

### `tests/test_phase1a.py` — Short replication smoke test

Run a **tiny** version of the faithful config (20 prey, 5 predators, 50,000 steps) and check statistical trends — not the final result, but direction of travel. Takes ~5 minutes on CPU.

- `test_w_eat_positive_trend`: `w_eat` mean is higher at step 50k than at step 0 (food reward should reliably increase)
- `test_population_stable`: Neither species goes extinct in 50k steps with reasonable parameters
- `test_reward_weights_drift`: Standard deviation of `w_pred` across population is > 0.05 at step 50k (evolution is running)
- `test_no_nan_reward`: No NaN values appear in reward signals or weight trajectories

### `tests/test_components.py` — Unit tests for each module

Fast deterministic unit tests; should run in < 30 seconds total.

- Reward genome: correct output shapes; clipping behavior; mutation scale
- Hazard function: matches formula values at known (t, e) pairs
- Birth function: matches formula values at known e values; monotone increasing in e
- Observation vector: correct dimension; correct ordering; no NaN
- Sensor aggregation: `max_k s^{i,k}_pred` returns maximum, not sum or mean
- Config loading: all required keys present; no unknown keys; type checks

---

## Track E: Engineering Improvements

These are independent of the experimental phases and should never block science work. Each improvement is a self-contained task that can be picked up when compute is idle or between phase transitions.

Priority order: items marked **[HIGH]** should be done before Phase 2 since they affect all subsequent runs. **[MEDIUM]** items are valuable but not urgent. **[LOW]** items are nice-to-have.

### E1. Performance and Speed **[HIGH]**

The faithful K&D mode runs ~10–12 hours per seed on A100. The shared-policy simplified mode is estimated at 2–4 hours. Any speedup here multiplies across every future experiment.

**E1a. JAX JIT compilation of the simulation step**
The inner simulation loop (physics step → energy update → sensor computation → reward) should be compiled via `jax.jit`. The main complication is that birth/death events change the number of agents, which is dynamic — this requires either a fixed-capacity array with a "dead" mask, or padding to max population. The fixed-capacity masked array approach is more JIT-friendly. Speedup estimate: 3–10× on the inner loop.

**E1b. Vectorized agent operations via `jax.vmap`**
Replace Python loops over agents with `jax.vmap` for observation construction, reward computation, and policy forward passes. Prerequisite: E1a (needs fixed-size arrays).

**E1c. Profile before optimizing**
Use JAX's profiler (`jax.profiler.trace`) to identify actual bottlenecks before writing any optimization code. Common surprises: data transfer between CPU/GPU, Python-side birth/death bookkeeping, PPO minibatch construction.

**E1d. Parallel seed execution**
Run multiple seeds in parallel rather than sequentially. Either: (a) launch separate processes per seed (simple, requires multiple GPUs or sequential CPU runs), or (b) vectorize seeds via an outer `vmap` over the full simulation (complex but potentially very fast). Start with (a).

### E2. Visualization and Monitoring **[HIGH — needed before Phase 1a results are meaningful]**

**E2a. Reward weight trajectory plot** (K&D Figure 7 style)
`analysis/dashboards.py` — reads `metrics.npz`, plots mean ± std of each reward weight over time, one color per seed. This is the primary result figure. Must match K&D's style closely for direct comparison.

**E2b. Reward weight KDE / scatter plot** (K&D Figure 8 style)
Kernel density estimate of population reward weight distribution at a given timepoint (default: last 5000 agents born). Overlay multiple seeds. Also: scatter plot colored by agent lifetime and dot size by number of offspring.

**E2c. Population dynamics plot** (K&D Figure 6 style)
Prey and predator population counts over time. Two subplots (full timeline + zoomed window). Should visually reproduce the Lotka-Volterra-like oscillations.

**E2d. Live monitoring dashboard (TensorBoard / Weights & Biases)**
Log key metrics every N steps during training: population sizes, mean reward weights, mean energy, capture rate. Viewable without waiting for the run to finish. This is critical for catching divergence or extinction early in a 10-hour run.
Implementation: add W&B or TensorBoard logging to `metrics.py` as a config-toggleable option (`logging.wandb: true`).

**E2e. Episode replay / visualization**
Save agent trajectories (positions, actions, energies) during a representative window (e.g., 10,000 steps at generation 100, 200, 300). Render as a 2D animation:
- Blue circles = prey, red circles = predators, green dots = food
- Line trails showing recent movement
- Agent size proportional to energy (optional)
- Color intensity proportional to fear-reward weight magnitude (optional)

Implementation options, in order of effort:
1. **Offline matplotlib animation** — save trajectory data during run; render post-hoc as MP4. Simple, no real-time constraint.
2. **Pygame live viewer** — attach to a running experiment via shared memory or socket; view current state in real time. More complex but enables watching evolution unfold.
3. **Web-based viewer** — render to HTML canvas via a lightweight server. Accessible from anywhere, shareable.

Start with option 1 (offline matplotlib). Option 2 is the most useful for debugging and demo purposes and should be prioritized after Phase 1a is working.

**E2f. Reward function heatmap** (for MLP genome, Phase 2+)
Plot MLP reward output as a 2D heatmap over pairs of stimulus variables (e.g., predator_proximity × conspecific_proximity, holding others constant). Reveals whether the evolved reward function is nonlinear and what its structure is. Only relevant for Axis 1 / Axis 3 conditions.

### E3. Analysis and Metrics **[MEDIUM]**

**E3a. Phylogenetic tree reconstruction**
Track parent-child relationships during the run (parent ID stored with each newborn). Post-hoc, reconstruct the evolutionary tree. Visualize as a pruned tree of lineages that survived to the end. This directly reproduces K&D's Figure 9 and reveals whether fear evolves once and spreads, or independently multiple times.

**E3b. Reward weight branching analysis**
At the end of a run, cluster the reward weight population (KMeans or DBSCAN in (w_pred, w_prey) space). Count clusters and their sizes. Track cluster membership over time to see if branches appear, merge, or go extinct. Reproduces K&D's Figure 8 analysis.

**E3c. Behavioral phenotype clustering**
For each agent sampled at a given generation, run a fixed behavioral probe (standardized scenarios: predator at fixed distance, food at fixed location, isolated vs. grouped) and record the action taken. Cluster agents by behavioral phenotype rather than genome. Compare behavioral cluster count to genome cluster count — do genome polymorphisms map to behavioral polymorphisms?

**E3d. Cross-generation dominance tournament**
Take policy snapshots from generations G1, G2, G3 (e.g., every 100 generations). Run each generation's prey against each generation's predators in a held-out arena. Report survival rates. This reveals evolutionary progress — does generation G3 prey consistently survive better against all predator generations, or only against its contemporary? Reproduces the "evolutionary arms race vs. cycling" distinction.

**E3e. Automatic `validate_replication.py` script**
Script that loads a completed run's `metrics.npz` and checks all Phase 1a success criteria (see Phase 1a section above). Returns PASS / FAIL with specific failure messages. Should be run automatically at the end of every Phase 1a seed.

### E4. Codebase Ergonomics **[MEDIUM]**

**E4a. Config validation**
On startup, validate the config YAML against a schema: required keys present, types correct, values in valid ranges (e.g., `population_size > 0`, `mutation_scale > 0`). Fail fast with a clear error message rather than crashing 5 hours into a run with a KeyError.

**E4b. Experiment registry**
A simple `experiments.yaml` that tracks which conditions have been run, how many seeds, and where results are stored. Prevents accidentally re-running expensive conditions and makes it easy to check which cells of the experimental matrix are complete.

**E4c. Reproducibility verification**
Given a saved config + seed, re-running should produce identical results. Add a `--verify-reproducibility` flag to `run_experiment.py` that runs the first 1000 steps twice and asserts identical outputs. JAX's random key handling makes this achievable but requires careful key management throughout the codebase.

**E4d. `emevo` source diff document**
A short document (`docs/emevo-diff.md`) listing every place our implementation deviates from K&D's open-source code, with justification. This is essential for debugging — when results diverge, this document is the first place to look.

### E5. Infrastructure for Extensions **[LOW — implement only when extensions begin]**

These are placeholders that should be designed as hooks in Phase 0 but not implemented until the relevant phase.

**E5a. Shared policy architecture**
Single policy network per species, conditioned on reward genome. Add `reward_genome_embedding_dim` to config; concatenate embedding to obs before first hidden layer. Behind `policy_mode: shared` config flag.

**E5b. MLP reward genome**
Replace 4-scalar reward weight vector with a small MLP (~2 hidden layers, 8 units each, tanh activations, scalar output). The genome is the flattened weight vector of this MLP. The mutation operator applies t(df=2) noise to the flattened vector. Behind `reward_type: mlp` config flag.

**E5c. Observation expansion for social obs**
Add conspecific heading (1 value per visible neighbor) and velocity (2 values per visible neighbor) to the observation vector. The existing sensor architecture detects proximity; heading/velocity requires tracking the velocity state of nearby agents. Behind `social_obs: position_heading_velocity` config flag.

**E5d. Temporal reward context window**
Rolling buffer of the last `k` observation vectors. The reward function (MLP) takes the flattened buffer as input. Buffer is reset at birth (zeros). Behind `reward_context_window: k` config flag. Requires `reward_type: mlp`.

**E5e. LSTM policy**
Replace feedforward MLP policy with LSTM. Hidden state `h_t` is carried between steps within an episode but reset at birth. PPO rollout must store and replay hidden states. Behind `policy_type: lstm` config flag. Requires careful rollout buffer modifications in `ppo.py`.

---

## Capacity Utilization Metrics (Phase 2+)

For each extension axis, the primary scientific question is whether evolution *uses* the added capacity. These metrics provide quantitative answers. Implement in `analysis/capacity_util.py` when the relevant phase begins.

**Axis 1 — Reward nonlinearity utilization:**
Fit a linear function `ŷ = Wx + b` to the MLP reward output across 10,000 sampled states. Compute `residual = ‖MLP(x) - ŷ‖² / ‖MLP(x)‖²`. High residual = evolution found nonlinear reward useful.

**Axis 2 — Social observation utilization:**
Mutual information `I(conspecific_heading_velocity ; agent_action)` estimated via MINE or binned histograms across a held-out rollout. Zero MI = agent ignores the social channel.

**Axis 3 — Temporal reward utilization:**
Autocorrelation of `r(t)` across the context window. Also: sensitivity ratio `‖∂r/∂obs(t)‖ / ‖∂r/∂obs(t-k)‖` — does reward weight recent observations more than old ones?

**Axis 4 — LSTM memory utilization:**
Hidden state entropy `H(h_t)` across a trajectory. Ablation: zero `h_t` mid-episode; measure performance drop.

---

## Document Index (What Lives Where)

```
docs/
├── technical-spec-kd-replication.md  # All K&D parameters; exact reward equations; success criteria
├── experimental-plan.md              # Phase descriptions; hypotheses H1-H7; compute budget
├── full-extension-design-doc.md      # Scientific rationale for each extension axis
├── background.md                     # Conceptual intro for new contributors
├── development-roadmap.md            # THIS FILE — phase structure; tests; Track E improvements
├── emevo-diff.md                     # [TO CREATE] Deviations from K&D's open-source code
└── interfaces.md                     # [TO CREATE] Module contracts; data structures; obs layout

tests/
├── test_phase0.py                    # Phase 0 validation gate (automated)
├── test_phase1a.py                   # Short replication smoke test
└── test_components.py                # Unit tests for each module

scripts/
├── run_experiment.py                 # Single run from config
├── run_sweep.py                      # Batch launcher
├── validate_replication.py           # [TO CREATE] Phase 1a success criteria checker
└── analyze_results.py                # Post-hoc analysis from checkpoints

configs/
├── baseline_faithful.yaml            # K&D faithful (Phase 1a)
├── baseline_simplified.yaml          # Shared policy + continuous (Phase 1b+)
├── axis1_mlp_reward.yaml
├── axis2_social_obs.yaml
├── axis3_temporal_reward.yaml
└── axis4_lstm_policy.yaml
```

---

## The Gate Sequence

Every gate is a hard stop. A coding agent should not proceed past any gate without explicit confirmation that it passed.

```
Gate 0: test_phase0.py passes (all unit tests green)
         ↓
Gate 1a: validate_replication.py passes on ≥1 seed
         (fear emerges, social affiliation emerges, populations oscillate)
         ↓
Gate 1b: Shared policy preserves core result (or fallback decision made)
         ↓
Gate 2: At least 1 axis shows nonzero capacity utilization in 1-seed sweep
        (if all flat: longer runs, larger pop, then re-evaluate)
         ↓
Gate 3: 3-5 interesting conditions identified for depth investment
         ↓
Gate 4: Report submitted
```
