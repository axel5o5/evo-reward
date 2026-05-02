# AGENTS.md
### Instructions for coding agents working on `evo-reward`

Read this file completely before touching any code.

---

## What This Project Is

A replication and extension of Kanagawa & Doya (2025) — an evolutionary simulation where RL agents (prey and predators) evolve their reward functions via birth-death dynamics. The inner loop is PPO; the outer loop is natural selection on a 4-weight reward genome. The headline result to replicate: prey evolve fear (negative reward for predators) and social affiliation (positive reward for conspecifics) with no hand-design.

**Your current job:** Replicate that result exactly. Nothing else.

---

## The One Rule

**Do not write extension code until `tests/test_phase1a.py` passes on at least one seed.**

Extension code means: shared policy, MLP reward genome, LSTM policy, social observation channels, temporal context window, generational batching. These have config flags and stub functions that raise `NotImplementedError`. Leave them that way.

---

## Start Here — In This Exact Order

### 1. ✅ DONE — Resolve open questions against the emevo source

All 9 open questions in `docs/emevo-diff.md` have been resolved against the emevo `gecco2026` branch. Reference config: `config/env/20251122-predator-square.toml`. Key corrected values are in `configs/baseline_faithful.yaml` and `docs/interfaces.md`. See `docs/emevo-diff.md` for full citations.

**Do not re-do this step.** If you need to verify a parameter, check `docs/emevo-diff.md` first — it has file:line citations for every answer.

### 2. Install dependencies

```bash
# JAX — install GPU variant if on GPU: https://jax.readthedocs.io/en/latest/installation.html
pip install jax jaxlib

# K&D's JAX 2D physics engine — this is not optional, it IS the physics layer
pip install phyjax2d
# If pip install fails: pip install git+https://github.com/kngwyu/phyjax2d

# Neural networks + optimizers
pip install flax optax

# Everything else
pip install scipy numpy matplotlib pyyaml pytest
```

Record exact versions in `requirements.txt` to match what emevo's 2025 branch uses.

### 3. Read these documents fully before implementing anything

**Start with `docs/CURRENT_STATE.md`** — the most recent snapshot of what's
running, what's queued, and the v10 framework (three-tier L1/L2/L3 ladder,
mouth widening, age-keyed LR schedule, death-age ring buffer). Then:

1. `docs/findings.md` (especially §15 strategic reset and §15.20 v10 ladder)
2. `docs/technical-spec-kd-replication.md` — every numerical parameter, the reward equation, success criteria
3. `docs/interfaces.md` — every module's function signatures and data contracts
4. `docs/emevo-diff.md` — what we change from emevo and why
5. `docs/development-roadmap.md` — build order, test specs, gate sequence
6. `configs/baseline_faithful.yaml` — the complete config for Phase 1a

### 4. Build modules in this order

Hard dependency chain — do not reorder:

```
1. ✅ environment.py     world, physics via phyjax2d, sensors, food
2. ✅ lifecycle.py       energy updates, hazard h(t,e), birth b(e), events
3. ✅ agents.py          observation vector construction, stimulus extraction
4. ✅ reward.py          linear reward genome, reward computation
5. ✅ evolution.py       mutation t(df=2), spawn_offspring
6. ✅ policy.py          MLP policy (2 hidden, sigmoid action), value head
7. ✅ ppo.py             GAE, PPO update, rollout buffer management
8. ✅ metrics.py         logging, checkpointing, save/load
9. ✅ scripts/run_experiment.py   ties it all together
```

After each module: run `pytest tests/test_components.py -k <module_name>` before starting the next.

### 5. Pass the Phase 0 gate

```bash
pytest tests/test_phase0.py
```

All tests must pass. Runtime: under 5 minutes on CPU. This validates plumbing before any evolutionary dynamics are involved.

### 6. Run Phase 1a

```bash
python scripts/run_experiment.py --config configs/baseline_faithful.yaml --seed 0
# ~10-12 hours on A100
```

At completion:

```bash
python scripts/validate_replication.py --results results/baseline_faithful/seed_0/
```

If PASS: run seed 1. If FAIL: see the debugging section below. Do not proceed to Phase 1b until at least one seed passes.

---

## Key Numbers (do not substitute)

Values below are from the **emevo gecco2026 source code** (ground truth), which in some cases differs from the paper tables. Where they differ, the code wins — see `docs/emevo-diff.md` for details.

| Parameter | Value | Source |
|-----------|-------|--------|
| World size | 960×960, square | emevo `20251122-predator-square.toml` |
| Prey radius / Predator radius | 10 / 14 | Appendix A |
| **obs_dim** | **205** | emevo source (128 sensor + 72 tactile + 5 scalars) |
| Proximity sensors | 32 sensors, 120° FOV, **range 200**, **4 channels/sensor** | emevo source (paper says range 120 — code uses 200) |
| Sensor channels (per sensor) | [prey, predator, food, wall] — winner-take-all | emevo `circle_foraging_with_predator.py:84-111` |
| Tactile sensors | 18 bins × **4 type channels** = 72 values | emevo source |
| Velocity in obs | **2D (vx, vy)**, not scalar speed | emevo `circle_foraging.py:744` |
| Prey c_b / c_a | **1.0e-4 / 2.5e-6** | emevo source (paper Table 2 lists 2.5e-6 / 1.0e-4 — labels swapped in paper; magnitudes match) |
| Predator d_b / d_a | 4.0e-3 / 5.0e-5 | emevo source (matches paper Table 2) |
| Predator digestive rate (η) | 0.6 | emevo source |
| Initial energy (both species) | **100.0** | emevo source |
| Energy capacity (max) | 1000.0 | emevo source |
| energy_share_ratio | **0.4** (parent loses 40%, child gets 40%) | emevo source |
| spawn_spread (neighbor_stddev) | **100.0** world units | emevo source |
| Food capacity | **600** (body text correct, not Appendix A's 100) | emevo source |
| Food initial / growth rate | 40 / 0.5 per step | emevo source |
| Action range | [-20, 80], **sigmoid mapping** (not hard clip) | emevo source |
| Policy trunk | **2 hidden layers** (64 units, tanh), shared | emevo `ppo_normal.py:28-65` |
| Mutation | Student's t(df=2, scale=0.4), clip ±100 | 2025 paper + emevo `20250805-mutation-t2-clip100.toml` |
| Reward weight init | N(0, 0.1) | Section 4 |
| PPO N / lr / clip / epochs / GAE-λ / γ | 1024 / 3e-4 / 0.2 / 10 / 0.95 / 0.999 | Table 4 |
| Entropy coef | 0.001 | Table 4 |
| Obs normalization | **None** — raw obs to network | emevo source |

**Critical notes:**
- The 2024 paper used Cauchy (df=1, scale=0.02, clip=±10). The 2025 paper uses t(df=2, scale=0.4, clip=±100). We replicate the 2025 paper. Do not use 2024 values.
- Prey energy costs: paper Table 2 has the `c_a` / `c_b` labels swapped (magnitudes match code). Use the code values; the paper's own sanity check at main.tex:443 only holds with code labels.
- The "3-layer MLP" in the paper means 2 hidden layers + 1 output, not 3 hidden layers.

---

## Reward Equation

```
r = w_eat * n_eaten
  + 0.01 * w_act * (‖f_scaled‖ / F_max)
  + 0.1  * w_prey * agg_k(s_prey^k)
  + 0.1  * w_pred * agg_k(s_pred^k)
```

- The 0.01 and 0.1 coefficients are **fixed architecture** — not part of the genome
- `f_scaled = sigmoid_scale(raw_action)` — the post-sigmoid motor output, NOT the raw network output
- `F_max = sqrt(act_high^2 + act_high^2) = sqrt(80^2 + 80^2) ≈ 113.14` — the max norm of the scaled action space
- `s_prey^k` = proximity sensor k, **channel 0** (prey channel). `s_pred^k` = channel 1 (predator channel). Values clipped ≥ 0 before aggregation.
- `agg_k` = emevo default is **mean** over 32 sensors (`sensor_agg_type="mean"`). The paper describes "most prominent" which suggests **max**. Use the config parameter `sensor_agg_type` to control this.
- Genome order: `[w_eat, w_act, w_prey, w_pred]` — canonical everywhere, never permuted

---

## Observation Vector Layout

**obs_dim = 205.** Confirmed against emevo gecco2026 branch. See `docs/interfaces.md` for the full indexed layout.

```
Index 0–127:    proximity sensors    (32, 4)  32 sensors × 4 channels
                                              Channels: [prey, predator, food, wall]
                                              Winner-take-all per sensor: only closest
                                              type is positive; others = -1.0.
                                              Flattened row-major.
Index 128–199:  tactile collision    (4, 18)  4 type channels × 18 bins
                                              Channels: [conspecific, other_species, food, wall]
                                              Binary contact. Flattened row-major.
Index 200–201:  velocity             (2,)     2D (vx, vy), range [-10, 10]
Index 202:      angle                (1,)     heading in radians
Index 203:      angular velocity     (1,)     radians/step
Index 204:      energy               (1,)     raw value, capped at 1000.0
────────────────────────────────────────────────────────────────────────
obs_dim = 205   (128 + 72 + 2 + 1 + 1 + 1)
```

Always use `config["obs_dim"]`, never the literal number.

**Important for reward extraction:** `s_prey^k` comes from sensor k channel 0 (prey). `s_pred^k` from channel 1 (predator). These are specific channels, not a mixed single-channel signal.

---

## Simulation Loop Order (per step)

```
1. Get observations for all agents
2. Sample actions from each agent's policy → write obs/action/logprob/value to rollout
3. Step physics (phyjax2d)
4. Check eating events
5. Compute rewards → write reward to rollout
6. Update energies
7. Process births and deaths
8. Regenerate food
9. PPO update for agents whose rollout buffer is full
10. Log metrics / save checkpoint (on interval)
```

Order matters. Reward computation (step 5) happens before death processing (step 7).

---

## Phase 1a Success Criteria

Checked automatically by `scripts/validate_replication.py`:

| Criterion | Threshold |
|-----------|-----------|
| Mean prey `w_pred` < 0 at step 10M | ≥ 3 of 5 seeds |
| Mean prey `w_prey` > 0 at step 10M | ≥ 3 of 5 seeds |
| Mean prey `w_eat` > 0 at step 10M | All seeds |
| Population oscillates (not flat, not extinct) | All seeds |
| No extinction (either species) | All seeds |

---

## If Phase 1a Fails

Stop. Debug in this order:

1. **Agents not eating food** (PPO not working): Run 10 prey, no predators, 10k steps. Do they learn to approach food? If not: check lr=3e-4, rollout=1024, 2-hidden-layer MLP with tanh, sigmoid action mapping.

2. **Reward weights not drifting** (evolution not running): Print a histogram of 10,000 mutation samples. Should be heavy-tailed, not Gaussian. If it looks Gaussian, you used `jax.random.normal` instead of t(df=2).

3. **Extinction**: Check energy parameters against `configs/baseline_faithful.yaml`. Most likely: c_b or d_b too high, or birth threshold too high.

4. **Fear not emerging** (w_pred stays ≥ 0): Check that `s_pred^k` is high when predator is *close* (inverse distance = 1 at contact). Check the 0.1 scaling coefficient is present. Check capture rate > 0 (predators are actually killing prey).

5. **Wrong distributions**: Compare reward weight KDE to K&D Figure 12. Different mutation distributions (t vs Gaussian vs Cauchy) produce visually distinct KDE shapes.

The emevo source is the ground truth. If you cannot find the bug, run emevo on the same seed and compare outputs step-by-step.

---

## Code Conventions

- **No hardcoded numerics.** Every constant comes from `config[...]`.
- **Functional style.** Pure functions, no mutable module-level state. `WorldState` is the single source of truth passed explicitly everywhere.
- **Genome ordering is canonical.** `[w_eat, w_act, w_prey, w_pred]` always, everywhere, in that order.
- **`obs_dim` from config always.** Extension axes change this value.
- **Extension stubs raise `NotImplementedError`.** Do not delete or implement them.
- **Log every deviation from emevo in `docs/emevo-diff.md` immediately.**
- **Run tests after each module.** Do not stack up untested modules.

---

## Repository Map

```
AGENTS.md                          ← you are here
README.md                          ← project overview

docs/
  technical-spec-kd-replication.md ← ALL K&D parameters (start here for numbers)
  interfaces.md                    ← module contracts and data structures
  emevo-diff.md                    ← deviations from emevo; all 9 items resolved (D1-D10)
  development-roadmap.md           ← phases, full test specs, engineering backlog
  experimental-plan.md             ← scientific phases, hypotheses H1-H7
  full-extension-design-doc.md     ← extension axis rationale (read after baseline works)
  background.md                    ← conceptual intro

configs/
  baseline_faithful.yaml           ← Phase 1a — K&D faithful replication
  baseline_simplified.yaml         ← Phase 1b stub
  axis*.yaml                       ← Phase 2+ stubs

src/
  environment.py   lifecycle.py   agents.py    reward.py
  evolution.py     policy.py      ppo.py       metrics.py

tests/
  test_components.py   ← unit tests per module
  test_phase0.py       ← integration gate before Phase 1a
  test_phase1a.py      ← short smoke test

scripts/
  run_experiment.py          ← main entrypoint
  validate_replication.py    ← Phase 1a PASS/FAIL checker
  analyze_results.py         ← post-hoc analysis

analysis/
  dashboards.py    comparison.py    capacity_util.py

papers/
  kanagawa-doya-2025-v2.pdf  ← the paper being replicated (v2, Feb 2026)
```

---

## Current Status

Session 1 complete: emevo audited, obs_dim=205, all open questions resolved.
Session 2 complete: environment.py + lifecycle.py, 8/8 lifecycle tests green.
Session 3 complete: agents.py, reward.py, evolution.py, ppo.py (GAE), 14/14 tests green.
Session 4 complete: policy.py, ppo.py (full update), metrics.py, 31/31 tests green (4 skipped).
Session 5 complete:
  ✅ Part A: phyjax2d integrated — differential drive matching emevo (dt=0.1, damping=0.8/0.6, 5 sub-steps)
  ✅ Part B: scripts/run_experiment.py — full simulation loop, batched policy via vmap
  ✅ Part C: test_phase0.py — 11/11 tests pass, 43/43 total tests green
Session 6 complete:
  ✅ Perf fix D13: PPO minibatch step JIT-compiled — eliminates re-tracing (was 30+ min at step 1024)
  ✅ Perf fix D14: Batched policy uses power-of-2 padding — eliminates recompile on pop change
  ✅ NumPy rollout buffer + stacked params cache (additional ~2x speedup)
  ✅ 43/43 tests green after all fixes
  ✅ Part D: 25k smoke test PASSED — 9621s (2.6 steps/s avg)
     Step 25k: prey=106, pred=50, food=600, std(w_pred)=0.760 >> 0.05, no crash/NaN
Note: D11 food not in phyjax2d, D12 sensor bin fix, D13 PPO JIT, D14 padded batch — all in emevo-diff.md.
Session 7 complete: Fully-JAX simulation rewrite
  ✅ src/jax_state.py — SimState pytree (flax.struct, 161 leaves, 500 agent slots)
  ✅ src/jax_evolution.py — Pure-JAX Student's t mutation (no scipy)
  ✅ src/jax_food.py — JIT eating detection + food regen (pairwise broadcast + lax.scan)
  ✅ src/jax_lifecycle.py — Vectorized energy/hazard/birth (all jnp, lax.scan for births)
  ✅ src/jax_ppo.py — Per-agent conditional PPO (vmap + lax.cond + lax.fori_loop)
  ✅ src/jax_sim.py — Unified sim_step_core (steps 1-9 JIT-compiled)
  ✅ scripts/run_experiment_jax.py — New runner with PPO integration
  ✅ Vectorized observations, check_eating, update_energies in original code
  ✅ 52/52 tests green, 3 skipped
  Verified: 100 steps no NaN, population dynamics active, food regen, energy decay
Session 8 (partial):
  ✅ analysis/dashboards.py — 3 plotting functions implemented and tested on 25k smoke metrics
     plot_reward_trajectories: K&D Fig 7 style (mean±std, 4 subplots per weight)
     plot_population_dynamics: K&D Fig 6 style (prey/predator pop over time)
     plot_reward_kde: K&D Fig 8/12 style (w_prey vs w_pred scatter, checkpoint or fallback)
  ⏳ Phase 1a: needs GPU — ~46 days on CPU Mac, ~10-12h on A100
Session 9 complete: Axis 2 — Social behavioral observation
  ✅ src/agents.py — get_observation() extended with social_obs branch (heading + speed of N closest conspecifics)
  ✅ src/observations.py — vectorized JAX path extended with _single_social_obs via vmap
  ✅ configs/archive/axis2_social_obs.yaml — social_obs: position_heading_velocity, obs_dim: 215, n_social_neighbors: 5
  ✅ analysis/capacity_util.py — compute_social_obs_utilization() binned MI estimator
  ✅ src/metrics.py — optional trajectory saving (save_trajectories flag, off by default)
  ✅ tests: 5 social obs tests pass, 51/51 total green (3 skipped), 9/9 vectorized obs green
  ✅ D15 documented in emevo-diff.md, interfaces.md updated with finalized social obs layout
Session 10 complete: Axis 3 — Temporal reward context window + Axis 4 — LSTM policy
  ✅ src/reward.py — TemporalRewardMLP class, init_temporal_genome, compute_temporal_reward (945 params for k=10, h=16)
  ✅ src/evolution.py — mutate_temporal_genome (flatten/t(df=2)/clip/unflatten, same pattern as MLP)
  ✅ src/policy.py — LSTMPolicyNetwork class, init_lstm_policy, sample_action_lstm, policy_forward_lstm (73,477 params)
  ✅ src/jax_ppo.py — build_ppo_update_fn_lstm with truncated BPTT (128-step chunks, 8 per rollout)
  ✅ src/jax_state.py — SimState: +obs_buffer (max_agents, k, 4), +lstm_hidden (max_agents, 2, 64), +rollout_init_hidden
  ✅ src/jax_evolution.py — spawn_offspring_jax: obs_buffer/lstm_hidden/rollout_init_hidden reset to zeros at birth
  ✅ configs/archive/axis3_temporal_reward.yaml — reward_type: temporal, reward_context_window: 10
  ✅ configs/archive/axis4_lstm_policy.yaml — policy_type: lstm, lstm_hidden_size: 64, lstm_chunk_length: 128
  ✅ analysis/capacity_util.py — compute_temporal_utilization (autocorrelation + sensitivity ratio)
  ✅ analysis/capacity_util.py — compute_lstm_utilization (hidden entropy + ablation delta)
  ✅ tests: 6 temporal + 7 LSTM tests pass, 70/70 total green (3 skipped), no regressions
  ✅ D16, D17 documented in emevo-diff.md, interfaces.md updated with temporal/LSTM config keys
Next task: Run Phase 1a on GPU, then validate_replication.py, then launch seeds 1-4.

---

## Gate Sequence (never skip)

```
1. ✅ Resolve emevo-diff.md open items 1–9 against emevo source
2. pytest tests/test_components.py       → all green
3. pytest tests/test_phase0.py           → all green
4. ✅ 25k smoke test                     → PASS (prey=106, pred=50, std(w_pred)=0.760)
5. run_experiment.py seed 0              → ~10h run
5. validate_replication.py               → PASS
6. run seeds 1–4
7. Only then: Phase 1b (shared policy comparison)
```
