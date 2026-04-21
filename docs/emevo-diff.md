# emevo Deviation Log
### `evo-reward` — What we change from K&D's open-source code and why

*This is a living document. Every intentional or discovered deviation from `github.com/oist/emevo` is recorded here with justification. When Phase 1a fails to reproduce K&D's results, this document is the first place to look. If a deviation is listed here, it was a known choice. If a bug is found, add it here so the same mistake is never made twice.*

*Reference: `github.com/oist/emevo`, 2025 predator-prey paper branch (confirm branch name — the 2024 single-species experiments are on the `alife2024` branch; the 2025 paper branch may be `main` or a named branch).*

---

## How to Use This Document

```
### [D#] Short title
Component:   affected module(s)
emevo:       what K&D's code does
Ours:        what we do instead
Reason:      why
Risk:        what breaks if this choice is wrong
Resolution:  how to verify or revert
```

---

## Open Items — Must Resolve Before Writing Affected Module

These are not yet deviations; they are unknowns. Resolve each against the emevo source before the affected module is written. Check out the 2025 branch of `github.com/oist/emevo` and search for each. Update status to ✅ when resolved.

| # | Question | Affects | Status |
|---|----------|---------|--------|
| 1 | Proximity sensors: single channel per sensor (closest object of any type) vs separate per-type channels? | `obs_dim`, `agents.py`, reward stimulus extraction | ✅ **Separate per-type channels.** 4 channels per sensor: [prey, predator, food, wall]. Winner-take-all: only the channel for the closest object type gets a positive value; others = -1.0. Sensor shape: (32, 4) → flattened 128. Source: `circle_foraging.py:742` `sensor=BoxSpace(shape=(n_agent_sensors, self._n_obj))`, `circle_foraging_with_predator.py:84-111` `_observe_closest()`, line 736 `self._n_obj = N_OBJECTS + _n_additional_objs = 3 + 1 = 4`. |
| 2 | Velocity in obs: scalar `‖v‖` or 2D `(vx, vy)`? | `obs_dim`, all downstream obs consumers | ✅ **2D velocity (vx, vy)**, not scalar speed. Source: `circle_foraging.py:744` `velocity=BoxSpace(low=-MAX_VELOCITY, high=MAX_VELOCITY, shape=(2,))`, line 953 `velocity=stated.circle.v.xy`. |
| 3 | Action clipping: `[-20, 80]` per component (2024 paper) — same in 2025? | `policy.py`, `environment.py` | ✅ **Same: [-20, 80]**. Source: `config/env/20251122-predator-square.toml:20-21` `max_force=80.0, min_force=-20.0`, `circle_foraging.py:739` `act_space = BoxSpace(low=min_force, high=max_force, shape=(2,))`. Note: emevo maps network output through `sigmoid_scale` to [-20, 80] (`cf_predator.py:122`), then clips in step (`circle_foraging_with_predator.py:401`). |
| 4 | Value network: shared trunk with policy head + value head, or fully separate network? | `policy.py` architecture | ✅ **Shared trunk** with 2 hidden layers (64 units each, tanh activation), then separate value head (→1) and policy mean head (→2). Plus learned log_std parameter (shape (2,), state-independent). Source: `rl/ppo_normal.py:28-65` `NormalPPONet` class. Note: only 2 hidden layers in the torso, not 3 — "3-layer" in the paper likely counts input→hidden1→hidden2→output. |
| 5 | `energy_share_ratio` η: exact value; does parent lose `η·e` while offspring starts with `η·e`, or are the fractions different? | `lifecycle.py` | ✅ **η = 0.4**. Parent loses η×energy, child receives η×energy (same fraction). Applies to both species uniformly. Source: `config/env/20251122-predator-square.toml:26` `energy_share_ratio = 0.4`, `env.py:32-48` `Status.activate()`. Note: `predator_digestive_rate = 0.6` is a separate parameter for how much prey energy a predator absorbs when eating — not the birth energy share. |
| 6 | `spawn_spread`: std dev of offspring spawn position Gaussian | `evolution.py` | ✅ **100.0 world units**. Source: `config/env/20251122-predator-square.toml:10` `neighbor_stddev = 100.0`, `circle_foraging_with_predator.py:290-291` `LocGaussian(agent_loc, ones * neighbor_stddev)`. |
| 7 | Observation normalization: running mean/std before policy input? | `agents.py`, `policy.py` | ✅ **No normalization.** Raw observations are fed directly to the network. No running mean/std, no batch norm, no layer norm. Confirmed by: `cf_predator.py:118-119` `obs_t_array = obs_t.as_array(); net_out = ppo.vmap_apply(network, obs_t_array)` — no transform between obs and network. `ppo_normal.py:59-65` `NormalPPONet.__call__` — no normalization layers in the network. |
| 8 | Initial population energy: what energy do initial-population agents start with? | `environment.py` | ✅ **100.0 for both species.** Source: `config/env/20251122-predator-square.toml:22,37` `init_energy = 100.0, predator_init_energy = 100.0`, `circle_foraging_with_predator.py:621-625` reset function. |
| 9 | Food capacity: Appendix A says `n_max=100`; 2025 paper body text says 600. Which did the actual experiments use? | `environment.py`, food dynamics | ✅ **600.** The body text is correct. Source: `config/env/20251122-predator-square.toml:3-4` `n_max_foods = 600, food_num_fn = ["linear", 40, 0.5, 600]`. Food dynamics: starts at 40 items, grows by 0.5 per step (linear), caps at 600. Max regen per step: `n_max_food_regen = 10`. |

**Reference branch:** `gecco2026` of `github.com/oist/emevo`. The 960×960 predator-prey config is `config/env/20251122-predator-square.toml`. Prey birth/death: `config/bd/20240916-sel-a4e7-d15.toml`. Predator birth/death: `config/bd/20241229-predator-d100.toml`. Mutation: `config/gops/20250805-mutation-t2-clip100.toml` (clip ±100, matching paper).

**Confirmed obs_dim = 205**: 128 (32 sensors × 4 channels) + 72 (4 types × 18 tactile bins) + 2 (velocity) + 1 (angle) + 1 (angular velocity) + 1 (energy).

---

## Confirmed Deviations

### [D1] Mutation: Student's t(df=2, scale=0.4) clipped to ±100 — not Cauchy

**Component:** `evolution.py` / `mutate_genome()`

**emevo (2024 paper / alife2024 branch):** Cauchy distribution (= t with df=1), scale=0.02, clipped to [-10, 10].

**emevo (2025 paper):** Student's t with df=2, scale=0.4, clipped to [-100, 100]. The paper states explicitly: "Student's t-distribution with 2 degrees of freedom and a scale of 0.4."

**Ours:** Student's t(df=2, scale=0.4), clipped to [-100, 100]. Matches 2025 paper.

**Reason:** We replicate the 2025 paper. The two papers intentionally use different mutation operators. Earlier project documents described mutation as "Cauchy" — that was imprecise. t(df=2) is heavier-tailed than Gaussian but lighter than Cauchy (df=1). They are not interchangeable.

**Risk:** If the 2025 emevo source actually still uses the 2024 Cauchy operator (paper/code mismatch), our results will differ. Priority: verify against emevo source.

**Implementation:** `scipy.stats.t(df=2, scale=0.4).rvs(size=4)` for tests. For JAX: approximate via ratio-of-normals method or pass pre-sampled noise. Do NOT use `jax.random.normal()`.

---

### [D2] Population sizes: faithful config matches K&D; simplified configs use ~80

**Component:** `configs/`

**emevo:** prey_initial=150, predator_initial=10, prey_cap=450, predator_cap=50.

**baseline_faithful.yaml:** Matches emevo exactly.

**baseline_simplified.yaml and extension configs:** prey ~80, predators scaled proportionally.

**Reason:** Smaller populations reduce compute cost for the extension experiments. The faithful replication uses K&D's exact setup.

**Risk:** Smaller populations increase genetic drift, potentially weakening evolutionary signal in extension experiments. If Phase 2 results are flat or ambiguous, scaling population up is the first diagnostic.

---

### [D3] Physics engine: phyjax2d as external dependency

**Component:** `environment.py`, `requirements.txt`

**emevo:** Uses `phyjax2d`, authored by Kanagawa (`github.com/kngwyu/phyjax2d`). May be bundled or installed as a package depending on the branch.

**Ours:** Import phyjax2d as an external pip dependency. Do not vendor its source into our repo.

**Reason:** Cleaner dependency management; inherits upstream fixes automatically.

**Action required:** Pin the exact phyjax2d version used by emevo's 2025 branch in `requirements.txt`. Breaking API changes in phyjax2d will silently corrupt physics behavior.

---

### [D4] Independent policy mode: staggered per-agent PPO, not synchronized

**Component:** `ppo.py`, main loop

**emevo:** Each agent runs its own PPO update asynchronously when its personal rollout buffer fills (N=1024 steps). Updates are staggered since agents are born at different times.

**Ours (Phase 0–1a, independent mode):** Same — per-agent asynchronous PPO. No synchronization.

**Ours (Phase 1b+, shared mode):** Single PPO update per species at each rollout cycle. Deliberate scientific departure — see experimental plan H1–H7.

**Risk:** Any synchronization of the independent-mode updates would diverge from K&D's dynamics. The asynchronous per-agent update is a feature, not an implementation detail.

---

### ~~[D5] Food capacity ambiguity: using n_max=100 from parameter tables~~

**RESOLVED:** The body text (600) is correct, not Appendix A (100). Verified against `config/env/20251122-predator-square.toml` on gecco2026 branch: `n_max_foods = 600, food_num_fn = ["linear", 40, 0.5, 600]`.

**Component:** `environment.py`, `configs/baseline_faithful.yaml`

**emevo:** `food_max = 600`, linear growth starting from 40 items, growth rate 0.5/step, cap at 600. Max regeneration per step: 10 items (`n_max_food_regen = 10`).

**Ours:** Updated to `food_max = 600` to match emevo.

**Previous incorrect assumption:** We assumed 100 based on Appendix A parameter tables. The tables appear to contain a typo or refer to a different experimental condition.

---

### [D6] Codebase structure: functional modules vs emevo's class hierarchy

**Component:** All modules

**emevo:** Uses Python classes with state (an `Environment` object, agent objects, etc.).

**Ours:** Functional style — pure functions and dataclasses, no mutable class state. All state lives in `WorldState` which is passed explicitly.

**Reason:** JAX-friendly design; easier to JIT compile; clearer data flow; easier to test. The functional style matches how JAX is intended to be used.

**Risk:** None for correctness — the functional interface is equivalent. Potential confusion when reading emevo source and comparing to our code because the call patterns look different even when the math is identical.

---

### [D7] Observation vector: 205 dimensions, not 54

**Component:** `agents.py`, `policy.py`, all obs consumers

**emevo:** Observations are structured as a NamedTuple with per-type sensor channels:
- Proximity sensors: (32 sensors × 4 types) = 128 values. Each sensor raycasts separately for prey, predator, food, wall. Winner-take-all: only the closest type gets a positive value per sensor; others = -1.0.
- Tactile/collision: (4 types × 18 bins) = 72 values. Binary contact per type per bin.
- Velocity: 2D (vx, vy) = 2 values.
- Angle: 1, Angular velocity: 1, Energy: 1.
- **Total: 205.**

Source: `circle_foraging.py:740-748` obs_space definition, `circle_foraging_with_predator.py:84-111,466-473` observation construction.

**Ours (previous assumption):** 54 dimensions (32 single-channel proximity + 18 single-channel tactile + 1 speed + 1 angle + 1 ang_vel + 1 energy). This was completely wrong.

**Ours (corrected):** 205 dimensions, matching emevo exactly.

**Risk:** This changes the entire observation pipeline, policy network input size, and stimulus extraction logic. The reward stimulus extraction must select the correct sensor channels (index 0 = prey, index 1 = predator).

**Discovered:** Phase 0, during open-item resolution against emevo gecco2026 branch.

---

### [D8] Sensor range: 200 units, not 120

**Component:** `environment.py`, `configs/baseline_faithful.yaml`

**emevo:** `sensor_length = 200.0` for both prey and predators. Source: `config/env/20251122-predator-square.toml:12,36`.

**Paper (Appendix A):** States "max range 120" for proximity sensors.

**Ours:** Updated to 200.0 to match emevo source code. The paper Appendix A value appears to be outdated or refers to a different condition.

**Risk:** Sensor range affects how far agents can "see". 200 vs 120 substantially changes the information available to policies and the selection pressure on reward weights.

**Discovered:** Phase 0, during open-item resolution.

---

### [D9] Network architecture: 2 hidden layers in shared trunk, not 3

**Component:** `policy.py`

**emevo:** `NormalPPONet` has a shared torso with 2 hidden layers (each 64 units, tanh activation), followed by separate value head (→1) and policy mean head (→2), plus learned log_std (2,). Source: `rl/ppo_normal.py:28-65`.

Architecture: input → Linear(input, 64) → tanh → Linear(64, 64) → tanh → {Linear(64, 1) value, Linear(64, 2) policy mean}

**Paper:** "3-layer MLP" (Table 4). This likely means 3 linear layers total (2 hidden + 1 output), not 3 hidden layers.

**Ours:** Updated to 2 hidden layers to match emevo. `policy_n_layers` reinterpreted as total linear layers in the torso (2), not total hidden layers (which was our previous reading).

**Risk:** An extra hidden layer would add capacity and change learning dynamics.

**Discovered:** Phase 0, during open-item resolution.

---

### [D10] Action mapping: sigmoid scaling, not hard clipping

**Component:** `policy.py`, `environment.py`

**emevo:** Network outputs raw unbounded values. These are mapped through `sigmoid_scale(x) = (high - low) * sigmoid(x) + low` to produce actions in [-20, 80]. The environment step then clips (which is a no-op since sigmoid already maps to range). Source: `cf_predator.py:122` `env.act_space.sigmoid_scale(actions)`, `circle_foraging_with_predator.py:401` `jax.vmap(self.act_space.clip)(action)`.

**Our assumption:** Hard clipping of network output to [-20, 80].

**Ours:** Use sigmoid scaling to match emevo. This is a smooth, differentiable mapping vs hard clipping, which matters for gradient-based optimization.

**Discovered:** Phase 0, during open-item resolution.

---

## Template for New Entries

When a deviation is discovered during implementation, add an entry here immediately. Never delete entries — mark reverted deviations as ~~struck through~~ with a resolution note.

```markdown
### [D#] Short title

**Component:** module(s) affected

**emevo:** what the emevo source does (include file path if known, e.g. `src/emevo/env.py:L234`)

**Ours:** what we do

**Reason:** why

**Risk:** what breaks if this is wrong

**Discovered:** Phase X, approximate date

**Resolution:** how to verify correctness or revert if needed
```

---

### [D11] Food not in phyjax2d physics

**Component:** `environment.py`

**emevo:** Food items are static circles in the phyjax2d Space. Agents physically collide with food (bounce off). Eating is detected via tactile sensor bins (mouth_range indices).

**Ours:** Food is managed in Python only — positions stored in `world.food_positions`, not added to phyjax2d. Agents do not physically collide with food. Eating is detected by distance check in `check_eating()`.

**Reason:** Adding 600 static circles to phyjax2d increases physics step time from ~8ms to ~43ms per step. Since food-agent physics collisions are not critical to the evolutionary dynamics (what matters is eating detection and energy), we skip food in phyjax2d.

**Risk:** Agents passing through food positions instead of bouncing off could affect movement patterns. Low risk for replication since the key dynamics (eating, predation, reproduction) are unaffected.

**Discovered:** Phase 0, Session 5.

**Resolution:** If Phase 1a fails to reproduce K&D results, add food as static circles to phyjax2d and accept the performance cost.

---

### [D12] Sensor bin placement

**Component:** `environment.py:compute_proximity_sensors`

**emevo:** Uses phyjax2d raycast functions for sensor computation. Exact bin placement may differ.

**Ours:** 32 sensor bins evenly dividing the 120° FOV. Bin centers at `heading - fov/2 + (i+0.5) * bin_width`. Small epsilon (1e-9) added to bin half-width for floating-point boundary handling.

**Reason:** Pure-Python sensor implementation for portability and testability. Bin placement ensures no gaps at FOV boundaries.

**Risk:** Slight differences in sensor readings compared to emevo's raycast. Low risk — the sensor patterns are qualitatively the same.

**Discovered:** Phase 0, Session 5.

---

### [D13] PPO minibatch update is JIT-compiled

**Component:** `src/ppo.py`

**emevo:** emevo's PPO implementation details vary; emevo uses JAX throughout so similar JIT patterns likely apply.

**Ours:** The per-minibatch gradient step (`jax.value_and_grad` + `optax.adam.update`) is wrapped in a module-level `@jax.jit` function, cached by hyperparameter tuple. Without this, JAX re-traces the backward pass on every call (10 epochs × 4 minibatches = 40 re-traces per PPO update; with 45+ simultaneous agents at step 1024, this was >1,800 re-traces ≈ 30+ minutes before the first log line).

**Reason:** Pure performance fix. The math is identical — same loss function, same optimizer, same gradients. Only the execution path changes (traced once vs. traced every call).

**Risk:** None. JIT is semantically transparent for pure JAX computations. The compiled function produces identical outputs to the eager version.

**Discovered:** Phase 0, Session 6 (smoke test took >27 minutes to reach step 10k; profiled to PPO re-tracing).

**Resolution:** Verified: benchmark shows first call 0.73s (compile), subsequent calls 0.07s; 43/43 tests pass.

---

### [D14] Batched policy inference uses power-of-2 padding

**Component:** `scripts/run_experiment.py:_get_batched_sampler`

**emevo:** emevo manages population with fixed-capacity JAX arrays (slots), so vmap is always called on the same shape regardless of current population.

**Ours:** Initial implementation cached the JIT-compiled vmap function keyed by exact agent count. Each birth/death changed n_agents, triggering a new JAX compilation (~5-10s). With population growing from 45→151 over 20k steps, this caused ~106 recompilations.

Fix: cache key is now the next power of 2 (64, 128, 256…). Inputs are zero-padded to that size; outputs are sliced to actual n_agents. Maximum ~log2(max_pop) compilations across the entire run.

**Reason:** Performance fix. Math is identical — padded rows produce garbage outputs that are immediately discarded.

**Risk:** None. Only the JIT boundary changes, not the computation.

**Discovered:** Phase 0, Session 6.

**Resolution:** Verified via tests (43/43 pass) and smoke-test step-rate improvement.

---

### [D15] Social observation: heading and speed of conspecifics (Axis 2 extension)

**Component:** `agents.py`, `observations.py`

**emevo:** No social observation. Agents perceive conspecifics only through proximity sensor readings (distance/direction). No information about what a conspecific is doing (heading, speed).

**Ours:** When `social_obs = "position_heading_velocity"`, append 10 dims to the observation vector: heading and speed of the 5 closest conspecifics (same species) within proximity range (200 units). obs_dim = 215.

**Reason:** Axis 2 experimental extension. Tests whether richer social information changes evolved reward functions or enables coordinated behavior. The baseline (`social_obs = "position_only"`) remains identical to K&D.

**Risk:** None for baseline replication. Extension configs only affect Axis 2 experiments.

**Discovered:** Phase 2, Axis 2 implementation (Session 9).

---

### D16: Temporal reward context window (Axis 3)

**emevo:** Reward function is instantaneous: `r(t) = w . stimuli(t)`.

**Ours:** When `reward_type = "temporal"`, the reward function operates over a rolling window of the last k stimulus vectors: `r(t) = MLP_θ(stimuli(t-k:t))`. Architecture: input(k*4) → Dense(16, tanh) → Dense(16, tanh) → Dense(1). With k=10: 945 parameters per genome. The obs_buffer (rolling window) is per-agent state, initialized to zeros at birth and shifted each step.

**Reason:** Axis 3 experimental extension. Allows the reward to encode temporal patterns (e.g., fear of an approaching predator whose sensor signal is rising) rather than just instantaneous stimuli. Doya (2002) predicts this should converge toward prediction-error-like reward structures.

**Risk:** None for baseline replication. Extension config only.

**Discovered:** Phase 2, Axis 3 implementation (Session 10).

---

### [D18] Predator feeding mechanics — bug, FIXED 2026-04-21

**emevo (gecco2026):**
1. Catch is gated by `predator_eat_timer <= 0` — each predator has its own
   countdown that resets to `predator_eat_interval = 10` on a catch and
   decrements per step otherwise. Effectively one "eat event" per 10 steps.
2. Catch requires physical contact (distance ≤ sum of radii) in one of
   the tactile bins listed in `predator_mouth_range = [0, 1, 17]` (3
   frontal 20°-wide bins, total 60° arc). Predators cannot catch prey
   they aren't touching. Source:
   [`circle_foraging_with_predator.py` gecco2026 branch](https://github.com/oist/emevo/blob/gecco2026/src/emevo/environments/circle_foraging_with_predator.py),
   `step()` + `_collect_tactile()` + `predator_eat_timer` update at the
   bottom of the step function.

**Ours — BEFORE fix (buggy):**
1. No eat-interval cooldown. Predators could catch every step.
2. `predator_mouth_range_min/max = [40, 80]` interpreted as a radial
   distance range — predators caught prey 40–80 units away (2–6 body
   lengths, not touching).
3. Extra config-value mismatches uncovered while reviewing emevo TOMLs:
   `beta_b = 0.1` (should be 0.4), `zeta_b_prey = 10` (should be 15),
   `beta_t_prey = 2e-6` (should be 4e-6).

**Ours — AFTER fix:** matches emevo. `SimState` gains a new field
`predator_eat_timer: (max_agents,) int32`, initialized to 0 (ready).
`check_eating_jax` now uses contact-based distance + tactile-bin check
+ cooldown gating, and returns the updated timer. Config keys
`predator_mouth_deg`, `predator_mouth_range_min`, `predator_mouth_range_max`
are replaced by `predator_mouth_tactile_bins: [0, 1, 17]` and
`predator_eat_interval: 10`.

**Why it matters (the real-world impact):**
The bug was discovered after running Phase 1a seed 0 for 1.67M steps on
GCP. Predator reward weights were **identical to 3 decimal places for
the final 1M steps** — zero evolution. Root cause: predators were
catching ~10× faster than K&D intended (continuous catches + permissive
geometry), saturated mean energy at ~999 (cap 1000), and the K&D hazard
function `h(t,e) = κ_h · (1 − 1/(1+α_e·exp(−β_h·e))) · α_t·exp(β_t·t)`
vanishes (via `exp(−β_h·1000)` = rounding-error zero) at that energy.
No deaths → no slot turnover → mutated offspring could never be
introduced → reward genome frozen at initialization. PPO still learned
normally per-agent, but without selection on the reward weights, the
population-level reward-weight means stayed at the N(0, σ²_init)
distribution.

**Risk of regression:** test_predator_eating.py pins the corrected
semantics (contact + mouth-bin + cooldown + independent per-predator
timers). Also, old SimState checkpoints from the buggy run lack the
`predator_eat_timer` field and cannot be loaded by the new code —
this is intentional; those runs produced no valid training data.

**Discovered:** Phase 1a seed 0 attempt, 2026-04-21. Re-diagnosed
with the help of an independent agent that confirmed "reward_weights
at step 100K vs step 800K: max |Δ| = 0.000000" for predators.

**Coupled fix:** the D18 fix alone was not sufficient — see D19 below.
Phase 1a was restarted with D18+D19 bundled together on 2026-04-21.

---

### [D19] Slot↔body mismatch + post-step contact miss — bug, FIXED 2026-04-21

**emevo (gecco2026):** per-species slot ranges aren't exposed explicitly
because emevo uses separate physics bodies per agent; phyjax2d bakes
body radius into the slot index at builder time. emevo's step also
captures per-substep contacts directly via
`(state, solver), contacts = jax.lax.scan(... contact.penetration >= 0.0)`
and uses `space.get_contact_mat("circle", "circle", contacts)` to
detect eating (gecco2026 `circle_foraging.py::nstep`).

**Ours — BEFORE fix (buggy, post-D18):**
1. **Slot↔body mismatch.** `init_simstate` placed initial predators
   contiguous after initial prey (slots `n_prey..n_prey+n_pred-1`).
   But `_build_physics` had already bound prey-radius (10) bodies to
   slots `[0, prey_cap)` and predator-radius (14) bodies to
   `[prey_cap, max_agents)`. Result: our "predators" lived in prey-sized
   physics bodies — wrong mass, wrong inertia, and a smaller collision
   surface. Births reused a species-agnostic "lowest free slot" search,
   so offspring hit the wrong body too.
2. **Post-step distance check missed mid-step contacts.** `check_eating_jax`
   computed `dist ≤ sum_radii` at the *end* of each sim step, after
   phyjax2d's velocity solver had already separated colliders across
   5 physics substeps. Many real collisions happened inside a substep
   and showed distance > threshold by step-end.

Net effect: predators observed almost no catches, predator energy
drained, births were nearly impossible — so D18's reward-weight
freeze persisted even after D18 shipped. Phase 1a seed 0 ran for 70K
steps with **1 birth total**.

**Ours — AFTER fix:**
1. **Species-reserved slots.** `init_simstate` places prey at
   `[0, prey_cap)` and predators at `[prey_cap, max_agents)`. The
   static `species` and `radii` arrays are derived from slot index,
   guaranteeing agreement with phyjax2d's per-slot body radii.
   `process_births_and_deaths_jax` picks the first inactive slot
   within the parent's species range.
2. **Emevo-style contact plumbing.** `physics_step` emits
   `contact.penetration >= 0.0` per substep, max-reduces across
   substeps, and `sim_step_core` calls
   `space.get_contact_mat("circle", "circle", contacts)` to build an
   (A, A) bool matrix passed to `check_eating_jax`, which uses it in
   place of the distance check.

**Risk of regression:** `tests/test_simstate_invariants.py` pins the
slot↔body invariant directly (`state.radii[i] == physics.radius[i]`
for every slot). `tests/test_birth_invariants.py` pins the
species-correct offspring slot. `tests/test_sim_dynamics.py` includes
slow-marked integration tests that fail immediately if catches or
births stall. `tests/test_params_match_emevo.py` pins the hardcoded
constants (hazard, birth, reward coefficients, physics iter count)
that would otherwise drift silently.

**Discovered:** Phase 1a post-D18 restart, 2026-04-21. Checkpoint
inspection showed 1 birth / 70K steps and predator weights unchanged
across 60K steps. Root cause surfaced by an audit agent that compared
our init layout to phyjax2d's `_build_physics` radius assignment.

---

### D17: LSTM policy (Axis 4)

**emevo:** Policy is a feedforward MLP (2 hidden layers, 64 units, tanh).

**Ours:** When `policy_type = "lstm"`, the policy uses an LSTM cell: obs(205) → LSTM(64) → Dense(64, tanh) → policy/value heads. ~73,477 parameters. The LSTM hidden state (c, h) persists across timesteps within one agent's lifetime and is reset to zeros at every birth. It is NOT inherited — it is lifetime state, not genome. PPO training uses truncated BPTT with 128-step chunks (8 chunks per 1024-step rollout).

**Reason:** Axis 4 experimental extension. The 2×2 of (temporal reward × LSTM policy) is a key experimental comparison: FF+instant (K&D baseline), FF+temporal (richer signal), LSTM+instant (memory actor), LSTM+temporal (memory + anticipation).

**Risk:** None for baseline replication. Extension config only.

**Discovered:** Phase 2, Axis 4 implementation (Session 10).

---

## Differences That Are NOT Deviations

These look like differences but are not, because they don't affect the science:

- **Different file/module names:** Our `lifecycle.py` does what emevo's birth/death logic does, just organized differently.
- **Different logging format:** We use `.npz` + pickle; emevo may use different formats. Contents are equivalent.
- **Different launch scripts:** Ours target Columbia's cluster or local GPU; emevo's target OIST's cluster.
- **No visualization in core code:** We build visualization in `analysis/` as separate post-hoc scripts. emevo may have integrated rendering. This doesn't affect simulation correctness.
- **Python version / JAX version:** We pin to a specific JAX version. Any JAX version that supports the APIs we use is equivalent for correctness purposes; performance may differ.
