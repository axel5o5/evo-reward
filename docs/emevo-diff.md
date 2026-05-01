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

### [D8] Sensor range: 200 units, not 120 — **REVERTED** (see D27 below)

**Component:** `environment.py`, `configs/baseline_faithful.yaml`

**emevo:** `sensor_length = 200.0` for both prey and predators. Source: `config/env/20251122-predator-square.toml:12,36`.

**Paper (Appendix A):** States "max range 120" for proximity sensors.

**Original decision (Phase 0):** Updated to 200.0 to match emevo source code, assuming the paper Appendix A value was outdated.

**Post-hoc correction:** This was inconsistent with the D22 principle (follow paper text when paper and endpoint-code disagree). Reverted to 120 under D27 after phase1a-v5's extinction analysis implicated over-strong predator-fear signal per step as the driver — a 3× stronger fear reward with range=200 vs 120 made prey evolve fear too quickly for predators to sustain their population across L-V oscillation troughs.

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

---

# Convergence Fixes — Bugs Where Our Implementation Diverged from emevo / K&D

Entries below are **NOT intentional deviations**. Each one is a bug where our
code encoded different semantics than the paper / emevo describes, and the
fix brings us back into convergence with the reference. The `D` numeric
prefix is **historical** (continues the sequence from the intentional-
deviations section above) rather than semantic. The document was originally
called "deviation log" because the first 17 entries were all deliberate
deviations; once Phase 1a surfaced real bugs, we kept extending the same
numeric sequence instead of branching a new ID namespace — the commits and
code comments already reference `D18`, `D19`, etc.

Rule of thumb when reading:
  * Looking for reasons our code *differs* from emevo on purpose?  → above.
  * Looking for bugs we found and fixed (or lessons learned)?      → below.

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

### [D20] Caught prey never deactivated — bug, FIXED 2026-04-21

**emevo (gecco2026):** when a predator catches a prey, the prey is
removed from the world in the same step (its agent record is
deactivated; the physics body is flagged inactive).

**Ours — BEFORE fix (post-D18, post-D19):** `check_eating_jax`
correctly produced `pred_catch_slots` and `update_energies_jax`
correctly credited the predator with `+eta · E_prey`, but nothing
in the pipeline ever set `is_active = False` for the caught prey.
The prey kept existing at its pre-catch energy, the 10-step
`predator_eat_interval` cooldown reset, and 10 steps later the
predator caught the *same* prey again. Net effect: predators ate
for free indefinitely. Saturation at `energy_capacity ≈ 1000` with
no deaths, no population turnover, and no selection pressure on
either species.

**Ours — AFTER fix:** `check_eating_jax` also returns a
`prey_caught_mask: (max_agents,) bool`, and `sim_step_core`
applies a same-step deactivation right after eating-check — clears
`is_active`, zeroes velocity, flips the phyjax2d `circle.is_active`
bit. The energy transfer math is unchanged (`update_energies_jax`
still reads the pre-deactivation energies to compute `eta · E`).

**Why the D18/D19 fixes alone weren't enough:** D18 fixed *how*
catches are detected (contact + mouth + cooldown). D19 fixed *where*
the physics thinks each agent is (correct slot-to-body radius).
D20 fixes *what happens after* a catch. All three had to be right
to see real predator-prey dynamics.

**Risk of regression:** `tests/test_predator_eating.py`:
`test_predator_energy_jumps_after_catch` runs one full `sim_step_core`
over a hand-crafted pred/prey pair and asserts the predator's
energy gained ≈ `eta · prey_energy`. Also
`tests/test_sim_dynamics.py::test_caught_prey_actually_die` and
`test_predator_energy_not_saturated_at_5k` (both `@slow`) check
the macro signal over 5K in-process steps.

**Self-bug (same commit sequence):** the first version of D20
also zeroed caught prey energies during the deactivation step —
right before `update_energies_jax` read them. That made
`pred_gain = eta · 0 = 0`, so predators starved despite catching.
Fixed by leaving energies untouched (inactive slots are filtered
by `is_active` masks everywhere downstream; leaving the stale
energy is harmless and preserves the transfer). Caught on the
relaunched Phase 1a run at step 10K — predator count had dropped
from 50 to 10 in 10 minutes.

**Audit Finding 2 (bundled with D20):** `spawn_offspring_jax` was
writing ~20 SimState fields for a newborn slot but missed
`predator_eat_timer`. A slot that went dormant with timer=9 would
give the newborn a 9-step head start before its first catch.
Moderate-severity bias against successful predator lineages
(suppresses early-life catch success). Fix: reset to 0 alongside
the other per-slot resets. Test:
`tests/test_jax_evolution.py::test_child_predator_eat_timer_reset`.

**Discovered via:** `scripts/verify_replay.py`, a post-hoc replay
integrity check written earlier in the session. Running it against
the in-flight post-D19 replay at step 10001 reported 0 catches, 0
deaths, predator energy min=977/mean=991, and `births=0` —
inconsistent with D18+D19 alone. This led straight to the
"what happens to caught prey?" question and the missing
deactivation.

---

### [D22] Paper-text vs endpoint-code contradiction — config fix, 2026-04-21

**Context.** After D18/D19/D20 landed, a Phase 1a run still crashed: prey
dropped 444 → 44 and predators went extinct at step 90K (K&D Table 1
reports stable populations prey≈349, pred≈23). A deep cross-check of
our config against the paper (arXiv:2507.09992v2) + the emevo gecco2026
branch surfaced three real discrepancies:

| Param | Paper text (Tables 2/3, App. A, Fig 19) | emevo endpoint code (`cf_predator.py` defaults) | Our config (before D22) |
|---|---|---|---|
| world | **square 960×960** (App A) | 1200×600 rectangular (`20241212-predator.toml`) | 960×960 ✓ paper |
| β_t_prey | **2e-6** (Table 3) | 4e-6 (`20240916-sel-a4e7-d15.toml`) | 4e-6 ← endpoint, not paper |
| ζ_b_prey | **10** (Table 3; consistent w/ Fig 19 saturation at e≈25-30) | 15 | 15 ← endpoint, not paper |
| food contact | `prey_r + food_r = 14` (physics contact_mat) | same (physics contact_mat) | `prey_r = 10` only ← neither! |

**Why the contradiction exists:** the emevo README says
`experiments/cf_predator.py` is "THE endpoint for GECCO 2026," and its
hardcoded defaults point to the rectangular world + β=4e-6 + ζ=15
TOMLs. But the paper text explicitly asserts the opposite on all three
params. The paper was likely either run with env-level overrides that
matched Table 3 + Appendix A, OR the TOMLs drifted post-experiment and
no one updated the endpoint defaults. Without access to the author's
actual run command, we can't tell.

**Our call:** match paper text. Rationale:
  1. Appendix A is unambiguous on world geometry ("square 960×960").
  2. Fig 19 (birth function plot) shows prey saturation at e≈25-30 —
     only consistent with ζ=10, not ζ=15.
  3. Paper text "30 energy units are required to increase birth
     probability for prey" — matches ζ=10 saturation, not ζ=15.
  4. Table 1 reports stable populations (349 / 23) — our current
     config produces extinction. Endpoint defaults also presumably
     produce Table 1 dynamics, so at least one of the two works —
     paper text is the less-plausible source of error.

**Fix:**
  * `beta_t_prey`: 4e-6 → **2e-6**
  * `zeta_b_prey`: 15 → **10**
  * Add `food_radius: 4.0` to config
  * `check_eating_jax`: contact threshold `dist ≤ prey_radius` → `dist
    ≤ prey_radius + food_radius = 14` (matches phyjax2d's circle-circle
    contact formula emevo uses via physics engine). Food contact area
    jumps from (10/14)² = 51% of emevo's to 100%.

**Fallback plan.** `configs/archive/baseline_endpoint.yaml` preserves the
endpoint-code parameterization (rectangular 1200×600 world, 9 tactile
bins, 160° FOV, β=4e-6, ζ=15). Not yet runnable — needs code support
for rectangular world and variable tactile bin count. Documented as
TODO at the top of that file. If `baseline_faithful.yaml` still
doesn't produce K&D Table 1 dynamics, the code refactor + endpoint
run is the next hypothesis.

**Risk of regression:** `tests/test_params_match_emevo.py` pins all
three new values (β_t_prey=2e-6, ζ_b_prey=10, food_radius=4). The
food_radius test also greps `src/jax_food.py` to ensure the contact
threshold actually uses the new variable.

**Discovered:** phase1a-v2 crash analysis. Triple-checked against
paper PDF + five candidate emevo bd TOMLs + endpoint experiment script
before commit.

---

### [D31] Proximity sensor heading frame off by 90° — FIXED 2026-04-24

**Bug.** We fixed D26 for tactile bins, but proximity sensors still used a
different heading frame. In both `src/observations.py` and
`src/environment.py`, heading `0` treated world **+x** as forward for
proximity, while phyjax2d/emevo uses world **+y** as forward (same
convention tactile already follows via `-π/2`).

**Impact.** A target directly ahead of an agent (world +y at heading 0)
was outside the center of the proximity FOV, while a target to the right
(world +x) was treated as "in front." This distorts prey fear/chase reward
stimuli and policy learning geometry, and can make ablations around D29/D30
look inconclusive because the underlying stimulus frame is wrong.

**Fix.** Align proximity to the same +y-forward frame as emevo:

- `src/observations.py`
  - `_single_proximity_agents`: `rel_angle = angle_to - obs_angle - π/2`
  - `_single_proximity_food`: same `-π/2` shift
  - `_compute_wall_distances`: ray direction uses `angles + π/2 + offset`
- `src/environment.py::compute_proximity_sensors`
  - sensor centers now use `heading + π/2` as forward before FOV offsets

**Risk of regression.** Added convention-pinning tests:

- `tests/test_paper_anchored.py::test_proximity_forward_convention_reference_path`
- `tests/test_paper_anchored.py::test_proximity_forward_convention_vectorized_path`

and updated `tests/test_phase0.py` contact/range placements to use +y
forward semantics.

---

### [D30] Reward computed from post-step obs, not pre-step — FIXED 2026-04-23

**Bug.** Our `sim_step_core` sampled an action from `all_obs` at the start
of the step, then computed reward later using that same `all_obs`. emevo's
`cf_predator.py` computes reward from `obs_t1.sensor` — the observation the
agent will see *next* step, after physics and contact processing. The
credit signal is paired with the consequence of the action, not with the
stimulus that motivated it.

**Impact.** Temporal shift of the proximity-reward gradient by one step.
Fear/chase learning pairs action at `t` with `sensor(t)` instead of
`sensor(t+1)`. For a predator-prey evasion loop this means the prey's
"I should have moved away" signal is miscomputed as "I moved because I saw
predator here" rather than "I still see predator here after moving."

**Fix.** `jax_sim.py` now rebuilds obs-state from post-physics,
post-catch `sim_state` and calls `obs_fn` a second time for the reward
computation. Pre-step `all_obs` is still used for policy sampling and the
rollout buffer (correct — the action was conditioned on that obs).
Extra cost: one obs_fn call per step (~few ms at 500 agents).

**Verification notes (2026-04-24):** confirmed directly in `emevo_src/`
for both `origin/gecco2026` and the older `origin/predator` branch:
`exec_rollout` always computes reward from `obs_t1 = timestep.obs` after
`env.step(...)`, not from `obs_t` / `obs_t_array`.

---

### [D29] Proximity sensor reward: mean, not max — FIXED 2026-04-23

**Bug.** Our reward code hardcoded `jnp.max(...)` over the 32 proximity
bins for the `s_prey`/`s_pred` stimulus. emevo's `cf_predator.py` defaults
to `sensor_agg_type="mean"`. A single sharp detection in any bin gave a
far stronger fear/chase gradient than emevo produces.

**Impact.** Coupled with D27's already-large pre-fix fear signal: prey
evolve fear too fast, over-evade, starve predators. Post-D27 this effect
is damped but still wrong direction.

**Fix.** `sensor_agg_type` config key, default `"mean"` (paper-faithful).
Set in `baseline_faithful.yaml`. Captured at build-time in `jax_sim.py`
so the JIT trace sees a concrete fn.

**Verification notes (2026-04-24):** in `emevo_src/experiments/cf_predator.py`
the older `origin/predator` branch has mean-only aggregation; later
`origin/gecco2026` adds a max option but keeps the default as
`sensor_agg_type="mean"`.

---

### [D28] Predator energy credit: shared, not deduplicated — FIXED 2026-04-23

**TL;DR: likely root cause of predator knife-edge extinction.**

**Bug.** `check_eating_jax` took `argmin(dist)` over predators and gave
catch credit to exactly one predator per prey. emevo's `cf_predator.py`
does not dedup — every predator whose tactile+contact+cooldown gates
fire on a given prey receives `eta · prey_energy`. If three predators
swarm a prey, emevo credits three transfers; we credited one.

**Impact.** Upper tail of the predator energy distribution is compressed.
With `zeta_b_pred = 100` (saturates breeding at E ≈ 250), only predators
spiking above that threshold reproduce. Our diagnosis of seed 0 and
seed 1 runs showed predator `max(E)` crossing 250 only in the first
~30K steps, then dropping irrecoverably once prey populations dipped.
With shared credit, swarm catches (which are common at high pred
density) give 2-3× the energy, keeping the upper tail above breeding
threshold long enough for offspring to establish.

**Fix.** Removed the `nearest_pred` argmin + `add_catch` scan in
`jax_food.py::check_eating_jax`. New return shape:
`pred_caught_energy[i] = Σⱼ valid_catch[i,j] · energies[j]`;
`pred_n_catches[i] = Σⱼ valid_catch[i,j]`;
`prey_caught_mask[j] = any(valid_catch[:,j])` — each prey still dies
exactly once but can feed multiple predators.
`update_energies_jax` now takes `pred_caught_energy` directly and
applies `η` itself.

---

### [D27] Sensor range 200 → 120 (paper Appendix A) — FIXED 2026-04-22

**Context.** `phase1a-v5` ran the full D18-D26 fix stack, produced two
complete Lotka-Volterra cycles (the first healthy oscillation in this
project), and then extincted at step 410K. Trace analysis
(`scripts/trace_agent.py`) showed:

- Cycle 2 prey `w_pred` evolved to **−1.14** — 3× stronger fear than at
  cycle 1's trough (−0.36).
- The last surviving predator got 0 catches in its final 510 steps
  despite mean action magnitude 41.9 (actively hunting) — starved out.
- Evolved-fear prey were too good at evasion by cycle 2 for a solo
  predator to re-bootstrap the population.

Root-cause hypothesis: our prey receive a stronger per-step fear reward
signal than K&D's prey, causing fear to evolve faster than the paper's
L-V equilibrium assumes.

**The mismatch.** Proximity sensor signal is `s = 1 − dist / max_range`.
At a given distance from a predator:

| dist | our s (range=200) | paper s (range=120) |
|------|-------------------|---------------------|
|  50  |  0.75             |  0.58               |
| 100  |  0.50             |  0.17               |
| 120  |  0.40             |  0.00 (out of range)|
| 150  |  0.25             |  0.00 (out of range)|

At moderate distances our prey sees the predator ≈3× more strongly,
and at long distances (>120) our prey *still* senses the predator
where K&D's prey would be blind. Reward is
`0.1 · w_pred · max_s_pred`, so the per-step fear signal scales
directly with this factor.

**Paper vs emevo.** Paper Appendix A explicitly says "proximity
sensors with a maximum length of **120 units**." Both emevo env TOMLs
(`20241212-predator.toml` and `20251122-predator-square.toml`) set
`sensor_length = 200`. This is a paper-vs-endpoint-code disagreement
of exactly the D22 class — and D22 established the principle of
following paper text in those cases.

**Historical note.** During Phase 0 we documented this as D8 and
chose to match emevo's code (200) on the theory that Appendix A
was outdated. That was before the D22 principle existed. We are now
consistent: D27 reverts D8 and our config follows the paper.

**Fix.** One line in `configs/baseline_faithful.yaml`:
```
proximity_max_range:    120.0   (was 200.0)
```

**Risk of regression.** `tests/test_paper_anchored.py::
test_proximity_sensor_range_matches_paper_appendix_a` pins
`proximity_max_range == 120.0` against paper Appendix A.

**What this does NOT fix.** Predator-side: predators also see prey
at 120 instead of 200 units. That makes hunting harder per step,
partially offsetting the fear-slowdown. Net effect on dynamics is
empirical — will find out in `phase1a-v7`.

---

### [D26] Tactile bin indexing off by 90° — FIXED 2026-04-22

**TL;DR: the single most likely root cause of predator extinction.**

**Bug.** phyjax2d/emevo's convention is that an agent's heading=0 means
its forward direction is **world +y** (not +x). `get_relative_angle`
in phyjax2d subtracts an extra `π/2` when classifying an angle into
the agent's local frame. Our code omitted that offset.

Verified with a minimal simulation: predator at (500, 500), heading=0,
prey at (500, 520) — directly in front per phyjax2d convention.

| | Our pre-D26 code | emevo (phyjax2d) |
|---|---|---|
| `rel_angle` | π/2 (90°) | 0 |
| bin | 5 (center 100°) | 0 |

With `predator_mouth_tactile_bins = [0, 1, 17]` (supposed to be the
60° front arc), the mouth actually pointed **90° to the predator's
right side** — predators could not catch prey directly ahead.

**Tactile observations** had an even worse version of the same bug:
`_single_tactile` didn't take `obs_angle` at all, so bin classification
was in world frame — every agent's "bin 0" was "world +x" regardless
of which way it was facing. Policies can't learn a stable tactile →
action mapping from world-frame input.

**Why this explains the extinction pattern.** Predator population
consistently peaks around the right value (~23, close to K&D SS) then
collapses — consistent with predators being able to catch prey _only_
during chance side-swipes, never by learned pursuit. Once prey
population drops below a critical density, predators stop making
those lucky catches and starve.

**Fix.** Two files, two call sites:

  * `src/jax_food.py::check_eating_jax` — replace the old
    `angle_rel = angles_to_agents - angles - bin_centers` +
    `argmin(|angle_rel|)` approach with emevo's boundary-based
    assignment:

      ```python
      rel = (angle_to_agent - heading - π/2) % 2π
      bin = floor(rel / (2π / n_bins))
      ```

  * `src/observations.py::_single_tactile` — add `obs_angle` to the
    signature and use the same formula. Extracted a helper
    `_bin_in_agent_frame(angle_world, obs_angle, n_bins)` used for
    agents, food, and walls alike. The caller in `_build_obs_fn`
    now passes each agent's heading.

Also updated the pre-JAX Python reference
(`src/environment.py::compute_tactile_sensors`) to use the same
convention so the `tests/test_vectorized_obs.py` comparison tests
stay green.

**FP subtlety.** Emevo uses `(... + 3·TWO_PI - π/2) % TWO_PI`, but
testing revealed the `+ 3·TWO_PI` introduces floating-point rounding
that can snap "directly in front" (rel=0) to `~TWO_PI`, landing in
bin n-1 instead of bin 0. JAX's `%` already returns non-negative for
positive divisors, so we drop the extra offset.

**Tests added.** `tests/test_tactile_bin_indexing.py` (6 new tests):

  * `test_prey_in_front_is_bin_0_heading_0` — catch detection
    correctly places a prey at world +y into bin 0 when heading=0.
  * `test_prey_to_right_heading_0_is_not_in_mouth` — prey at world
    +x is bin 13 (not in mouth [0, 1, 17]) → not caught. Pre-D26
    this prey would have been caught — the smoking-gun bug.
  * `test_prey_in_front_rotates_with_heading` — placing prey
    directly in front of the predator (forward direction derived
    from heading) must be caught at 8 different headings 0°-315°.
    Pre-D26 at heading=90° the "front" direction would have been
    classified into a side-mouth bin → not caught.
  * `test_tactile_food_in_front_lights_bin_0` — tactile observation
    pipeline agrees with the catch pipeline's convention.
  * `test_tactile_to_right_of_heading_0_is_bin_13` — side contact
    classified correctly.
  * `test_tactile_rotates_with_heading` — same world-fixed contact
    moves between bins as observer rotates. Pre-D26 the bin was
    fixed in world frame.

Existing `tests/test_predator_eating.py` geometry updated: previous
tests placed prey at world +x calling it "east" / "in front" (which
matched the pre-D26 broken convention). New placements put prey at
world +y relative to heading=0, which is correctly "in front."

Full fast suite: **203 passed** (was 197; +6 new bin tests).

**Risk of regression.** All six tests in
`test_tactile_bin_indexing.py` would fail if the π/2 offset is ever
removed or if `obs_angle` stops being threaded into `_single_tactile`.

---

### [D25] Rectangular world support — infrastructure, 2026-04-22

**Motivation.** Our simulation hard-coded a square world via
`world_size` as a scalar. Emevo's endpoint predator TOML
(`20241212-predator.toml`) uses a rectangular 1200×600 world; the
paper-text uses 960×960 square. We've been running on paper-text
(square), hitting predator extinction, and want the option to test
endpoint-code params without rewriting every call site.

**Added:** `src/environment.py::world_bounds(config) -> tuple[float, float]`.
Three config shapes supported (checked in order):

  1. explicit rectangle: `world_size_x` + `world_size_y`
  2. tuple in scalar field: `world_size: [x, y]`
  3. legacy scalar (square): `world_size: N` → returns `(N, N)`

Threaded `(world_x, world_y)` through every site that previously used
the `world_size` scalar in the JAX path:

  * `src/environment.py::_build_physics` — `make_square_segments` now
    called with `(0, world_x, 0, world_y)`. Emevo does the same
    (the "square" in the function name is misleading; it produces
    rectangles when xmin≠xmax or ymin≠ymax).
  * `src/jax_state.py::init_simstate` — initial positions + food
    positions use `minval=[0,0], maxval=[x,y]`.
  * `src/jax_food.py::regenerate_food_jax` — food respawns use rect bounds.
  * `src/jax_evolution.py::spawn_offspring_jax` — child positions clamped
    to `[margin, x-margin] × [margin, y-margin]`.
  * `src/observations.py::_compute_wall_distances` + `_single_tactile` —
    wall raycast and wall tactile contact now use separate x/y bounds.

**`configs/archive/baseline_endpoint.yaml`** now runs end-to-end through
`sim_step_core` (previously documented as "not yet runnable"). That
header note is still correct that **tactile mouth-range semantics**
(`"front-wide"` / `"narrow"`) for prey food eating are not yet
decoded from strings — our current prey food eating uses an FOV
check, which is functionally similar for the default arcs but not
bin-identical. Low-severity for a first run.

**Regression suite.** `tests/test_rectangular_world.py`:
  * `world_bounds` resolution for all three config shapes
  * endpoint config bounds, initial positions, food positions all
    fall inside `[0, 1200] × [0, 600]` AND hit the long axis (so we
    haven't accidentally clamped to the smaller dimension)
  * `sim_step_core` runs 5 steps without shape/dtype errors on the
    endpoint config
  * Paper-faithful config still resolves to `(960, 960)` and
    produces the same initial population count (additive-only change)

Full fast suite: 197 passed (was 187; +10 new tests).

**Not a deviation from emevo.** Emevo supports both square (via
xlim=ylim) and rectangle via xlim/ylim. We now match that API shape.

---

### [D24] Energy cost must use act_ratio-scaled action norm — FIXED 2026-04-22

**Context.** Physics/observations audit (post-D23) found that emevo
computes `force_norm = sqrt(f1_raw² + f2_raw²)` AFTER multiplying the
raw action by `act_ratio`, and uses that scaled force_norm in the
energy-cost formula `d_a · force_norm + d_b`. Our code was using the
raw unscaled action norm for energy cost, via
`action_norms = linalg.norm(all_actions)` in `update_energies_jax`.

**Impact direction correction.** The audit report claimed this would
make predators starve — incorrect. `act_ratio = (pred_r/prey_r)² ≈
1.96` for predators, so using the raw norm *undercharges* predator
energy cost by ~49% (not overcharges). Fixing this means predators
pay **more** energy, not less. **Not the extinction blocker.**

Why fix anyway: the reward formula (`motor_norms / F_max`) correctly
uses raw actions to match emevo's `normalize_action(action)` → the
reward + energy-cost pair should be semantically consistent with
emevo regardless of whether it tilts dynamics. This also reduces the
search space for remaining unknowns.

**Fix.** In `src/jax_lifecycle.py::update_energies_jax`:
```python
scaled_actions = all_actions * sim_state.act_ratio
action_norms = jnp.linalg.norm(scaled_actions, axis=1)
```

Reward computation in `src/jax_sim.py` remains on raw actions
(correct — matches emevo `SensorActFoodExtractor.normalize_action`
which calls `act_space.sigmoid_scale(action)` with no act_ratio).

**Risk of regression.** `tests/test_predator_eating.py::TestEnergyCostScaling`
feeds matched raw actions to a prey and a predator and asserts
predator energy drain > 10× prey drain, which can only hold if
act_ratio is applied.

---

### [D23] `rollout_dones` never flagged True on agent death — FIXED 2026-04-22

**Context.** During a PPO-pipeline audit (post-D22), we found that
`sim_step_core` hard-codes
`rollout_dones.at[agent_idx, safe_ptrs].set(False)` on every step. When
an agent dies (D20 catch or hazard/starvation), its last rollout slot
stays marked `done=False`. GAE in `_compute_gae_jax` uses
`(1 - dones)` to mask bootstrap across terminal states — with
`dones` always False, the trajectory is always treated as continuing.

**Severity.** Cosmetic in the *current* architecture. PPO fires only
on `is_active & (ptr >= rollout_steps)`, and dead agents are
`is_active=False`, so their rollouts are never consumed; the stale
`done=False` is never read by GAE in practice. Rebirth via
`spawn_offspring_jax` zeros the rollout anyway. **Not the extinction
blocker.** But the flag should be correct for semantic soundness and
to future-proof any change that relaxes the PPO gate.

**Fix.** Two-part:

  * `sim_step_core`: step-6 `rollout_dones` write takes
    `prey_caught_mask` instead of `False`, so D20 catches set the
    terminal flag at the prey's last-written slot.
  * `process_births_and_deaths_jax`: accept `rollout_ptrs_for_done`
    (= the safe_ptrs used in step 6) and OR-merge `dead_mask` into
    `rollout_dones[slot, rollout_ptrs_for_done[slot]]` via
    `at[].max()`. Hazard/starvation deaths now flag terminal at the
    same rollout slot that holds the dying agent's last transition.

**Risk of regression.** `tests/test_predator_eating.py::TestDoneFlagOnDeath`
places a predator + prey touching in the mouth arc, runs one full
`sim_step_core`, and asserts `rollout_dones[prey_slot, old_ptr] == True`.

**Caveat.** Predator extinction at step 80K persists after this fix in
isolation. The remaining hypothesis space: (a) the endpoint-code
parameterization (rectangular world + 9 tactile bins) is load-bearing
for stable dynamics despite what paper Appendix A says, or (b) another
silent bug we haven't audited. Next session's work.

---

### [D21] Real-time run visibility — instrumentation, 2026-04-21

**Motivation:** D18, D19, and D20 all looked fine in the Step log
because the log surfaced populations and reward-weight means but
no event deltas — by the time a silent dynamics bug became
visible (via replay analysis after a flush), hours of GCP time
had been burned.

**Added:** three int32 scalars on `SimState` (`cum_catches`,
`cum_deaths`, `cum_feedings`) ticked inside `sim_step_core` and
`process_births_and_deaths_jax`; the runner diffs them between
log intervals. `progress.json` gets two new blocks:

- `events_last_interval`: `{catches, deaths, births, feedings, interval_steps}`
- `energy_stats`: `{prey: {min, mean, max}, pred: {min, mean, max}}`

The Step text log now shows `Δ catch=N death=N birth=N feed=N`
and per-species energy bands. Inline `⚠` warnings fire on:

- No catches across 2 consecutive log intervals
- No births across 5 consecutive log intervals
- `min(pred_energy) > energy_capacity * 0.95` (saturation)

Storage: ~12 bytes added to each checkpoint (three int32); ~300
bytes added to each progress.json write.

**Not a deviation from emevo** — this is instrumentation we added;
emevo makes no claim about what's in a training run's log.

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
