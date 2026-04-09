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

## Differences That Are NOT Deviations

These look like differences but are not, because they don't affect the science:

- **Different file/module names:** Our `lifecycle.py` does what emevo's birth/death logic does, just organized differently.
- **Different logging format:** We use `.npz` + pickle; emevo may use different formats. Contents are equivalent.
- **Different launch scripts:** Ours target Columbia's cluster or local GPU; emevo's target OIST's cluster.
- **No visualization in core code:** We build visualization in `analysis/` as separate post-hoc scripts. emevo may have integrated rendering. This doesn't affect simulation correctness.
- **Python version / JAX version:** We pin to a specific JAX version. Any JAX version that supports the APIs we use is equivalent for correctness purposes; performance may differ.
