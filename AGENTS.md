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

### 1. Resolve open questions against the emevo source

Before writing a single line of simulation code, clone emevo and answer the 9 open questions in `docs/emevo-diff.md`. These are not guessable — getting them wrong causes Phase 1a to fail for non-obvious reasons.

```bash
git clone https://github.com/oist/emevo
# The 2025 paper branch is likely main (the 2024 results are on alife2024 branch — confirm)
```

For each open item in `docs/emevo-diff.md`: find the answer in emevo source, mark ✅, update `configs/baseline_faithful.yaml` with the correct value. Resolve items 1 and 2 first (they determine `obs_dim`), then update `docs/interfaces.md`.

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

### 3. Read these five documents fully before implementing anything

1. `docs/technical-spec-kd-replication.md` — every numerical parameter, the reward equation, success criteria
2. `docs/interfaces.md` — every module's function signatures and data contracts
3. `docs/emevo-diff.md` — what we change from emevo and why
4. `docs/development-roadmap.md` — build order, test specs, gate sequence
5. `configs/baseline_faithful.yaml` — the complete config for Phase 1a

### 4. Build modules in this order

Hard dependency chain — do not reorder:

```
1. environment.py     world, physics via phyjax2d, sensors, food
2. lifecycle.py       energy updates, hazard h(t,e), birth b(e), events
3. agents.py          observation vector construction, stimulus extraction
4. reward.py          linear reward genome, reward computation
5. evolution.py       mutation t(df=2), spawn_offspring
6. policy.py          MLP policy, action sampling, value head
7. ppo.py             GAE, PPO update, rollout buffer management
8. metrics.py         logging, checkpointing, save/load
9. scripts/run_experiment.py   ties it all together
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

| Parameter | Value | Source |
|-----------|-------|--------|
| World size | 960×960 | Appendix A |
| Prey radius / Predator radius | 10 / 14 | Appendix A |
| Proximity sensors | 32, 120° FOV, max range 120 | Appendix A |
| Tactile sensors | 18, 20° spacing | Section 3 |
| Prey c_b / c_a | 2.5e-3 / 1.0e-4 | Table 2 |
| Predator d_b / d_a | 4.0e-3 / 5.0e-5 | Table 2 |
| Mutation | Student's t(df=2, scale=0.4), clip ±100 | 2025 paper |
| Reward weight init | N(0, 0.1) | Section 4 |
| PPO N / lr / clip / epochs / GAE-λ / γ | 1024 / 3e-4 / 0.2 / 10 / 0.95 / 0.999 | Table 4 |
| Policy hidden size | 64 | Table 4 |
| Entropy coef | 0.001 | Table 4 |

**Critical mutation note:** The 2024 paper used Cauchy (df=1, scale=0.02, clip=±10). The 2025 paper uses t(df=2, scale=0.4, clip=±100). We replicate the 2025 paper. Do not use 2024 values.

---

## Reward Equation

```
r = w_eat * n_eaten
  + 0.01 * w_act * (‖f‖ / 114.0)
  + 0.1  * w_prey * max_k(s_prey^k)
  + 0.1  * w_pred * max_k(s_pred^k)
```

- The 0.01 and 0.1 coefficients are **fixed architecture** — not part of the genome
- `F = 114.0` is the max motor output norm, also fixed
- Genome order: `[w_eat, w_act, w_prey, w_pred]` — canonical everywhere, never permuted

---

## Observation Vector Layout

```
Index 0–31:   proximity sensors    (32,)  inverse distance [0,1]
Index 32–49:  tactile sensors      (18,)  binary contact
Index 50:     angle                (1,)   radians [-π, π]
Index 51:     speed                (1,)   scalar ‖v‖  ← VERIFY: 2D velocity?
Index 52:     angular velocity     (1,)   radians/step
Index 53:     energy               (1,)   raw value
─────────────────────────────────────────────────────────────
obs_dim = 54  (or 55 if velocity is 2D — resolve from emevo source first)
```

Always use `config["obs_dim"]`, never the literal number.

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

1. **Agents not eating food** (PPO not working): Run 10 prey, no predators, 10k steps. Do they learn to approach food? If not: check lr=3e-4, rollout=1024, 3-layer MLP with tanh.

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
  emevo-diff.md                    ← deviations from emevo; 9 open questions
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

## Gate Sequence (never skip)

```
1. Resolve emevo-diff.md open items 1–9 against emevo source
2. pytest tests/test_components.py       → all green
3. pytest tests/test_phase0.py           → all green
4. run_experiment.py seed 0              → ~10h run
5. validate_replication.py               → PASS
6. run seeds 1–4
7. Only then: Phase 1b (shared policy comparison)
```
