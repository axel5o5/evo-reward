# TODO: Plumb MLP and temporal reward through the JAX runner

**Status:** Not started. Identified 2026-04-27 while building the
`RewardLandscape` heatmap spike — discovered `reward_type` is read by no
runtime code.

## Why this matters

Phase 2 axis experiments are starting (see [experimental-plan.md §Phase 2](../experimental-plan.md)).
Axis 1 (MLP reward) and Axis 3 (temporal reward) both depend on a heritable
reward genome that is *not* the 4-weight linear vector — Axis 1 uses a 121-
parameter MLP (4 → 8 → 8 → 1, tanh), Axis 3 uses a ~945-parameter MLP over a
rolling 10-step stimulus window.

The genome init / mutate / forward functions exist in [src/reward.py](../../src/reward.py)
and [src/evolution.py](../../src/evolution.py), but they are never called by
the runner. Smoke runs of `axis1_mlp_reward` and `axis3_temporal_reward`
([experiments-log.md rows axis1_smoke / axis3_smoke](../experiments-log.md))
completed cleanly — but were silently computing the linear formula throughout.
The `reward_type: mlp` and `reward_type: temporal` flags in the configs are
no-ops.

Until this is fixed, every Axis 1 / Axis 3 result is a baseline run with a
mislabelled config, and no MLP/temporal genome viz on the dashboard can be
backed by real data.

## What is currently broken

Verified via grep across `src/` and `scripts/` on 2026-04-27:

- [src/reward.py](../../src/reward.py) defines `RewardMLP`, `init_mlp_genome`,
  `compute_mlp_reward`, `TemporalRewardMLP`, `init_temporal_genome`,
  `compute_temporal_reward`.
- [src/evolution.py](../../src/evolution.py) defines `mutate_mlp_genome`,
  `mutate_temporal_genome`.
- **None of these symbols are imported** by [src/jax_sim.py](../../src/jax_sim.py),
  [src/jax_state.py](../../src/jax_state.py), [src/jax_evolution.py](../../src/jax_evolution.py),
  [src/jax_lifecycle.py](../../src/jax_lifecycle.py),
  [scripts/run_experiment_jax.py](../../scripts/run_experiment_jax.py), or
  [scripts/run_experiment.py](../../scripts/run_experiment.py).
  Only `tests/test_components.py` and the docs reference them.
- [jax_sim.py:263](../../src/jax_sim.py#L263) computes reward unconditionally
  as the linear formula `jnp.sum(sim_state.reward_weights * stimuli * coefs, axis=1)`.
  No branch on `reward_type`.
- [jax_state.py:38](../../src/jax_state.py#L38) declares `reward_weights` as a
  fixed `(max_agents, 4)` field. There is no SimState slot for an MLP PyTree
  genome or a temporal genome.
- [jax_state.py:69](../../src/jax_state.py#L69) does declare `obs_buffer:
  (max_agents, k, 4)` for Axis 3, but nothing writes to or reads from it.

## What needs to happen

Suggested order — each step independently testable.

### 1. Add genome state to SimState

In [src/jax_state.py](../../src/jax_state.py):

- Add a `reward_genome` field that holds either:
  - a `(max_agents, 4)` linear vector (current behavior, when
    `reward_type == "linear"`), OR
  - a stacked Flax PyTree of MLP/temporal params (each leaf shape
    `(max_agents, ...)`), like `policy_params` already does for policies.

The cleanest factoring is to keep the existing `reward_weights` field for the
linear genome (so existing code paths are untouched) and add a separate
`reward_mlp_params: Any` field that's `None` for linear runs and a stacked
PyTree otherwise. JAX is fine with `None` leaves through `jax.tree_util.register_pytree_node`-marked
SimStates as long as the JIT trace doesn't change shape mid-run.

Initialize in `init_simstate` based on `config["reward_type"]`:

```python
reward_type = config.get("reward_type", "linear")
if reward_type == "linear":
    reward_mlp_params = None
elif reward_type == "mlp":
    keys = jax.random.split(rng_key, max_agents)
    reward_mlp_params = jax.vmap(
        lambda k: init_mlp_genome(k, config)
    )(keys)
elif reward_type == "temporal":
    # similar with init_temporal_genome
    ...
```

Pattern reference: the existing `policy_params` field is already a
vmapped-per-agent PyTree (see how it's initialized for the existing per-agent
policies under `policy_mode: independent`).

### 2. Dispatch reward computation in `jax_sim.py`

In [src/jax_sim.py](../../src/jax_sim.py) around line 261-263, replace the
hard-coded linear formula with a dispatch:

```python
reward_type = config.get("reward_type", "linear")
if reward_type == "linear":
    coefs = jnp.array([1.0, 0.01, 0.1, 0.1])
    stimuli = jnp.stack([n_eaten_reward, motor_norms, s_prey, s_pred], axis=1)
    all_rewards = jnp.sum(sim_state.reward_weights * stimuli * coefs, axis=1)
elif reward_type == "mlp":
    stimuli = jnp.stack([n_eaten_reward, motor_norms, s_prey, s_pred], axis=1)
    all_rewards = jax.vmap(compute_mlp_reward)(sim_state.reward_mlp_params, stimuli)
elif reward_type == "temporal":
    # roll obs_buffer forward by one step, then evaluate MLP over the window
    new_buffer = roll_obs_buffer(sim_state.obs_buffer, stimuli)
    all_rewards = jax.vmap(compute_temporal_reward)(sim_state.reward_mlp_params, new_buffer)
    sim_state = sim_state.replace(obs_buffer=new_buffer)
```

The branch is selected at `build_sim_step` time (closed over from config), not
inside the JIT — no `lax.cond` needed because each run uses one reward type.

The `reward_obs_timing == "post_step"` flag at
[jax_sim.py:67-72](../../src/jax_sim.py#L67-L72) must be honored across all
three branches. The proximity-stimulus `s_prey` / `s_pred` aggregation
([jax_sim.py:248-252](../../src/jax_sim.py#L248-L252)) stays the same —
only the *reduction* of (stimuli, genome) → reward changes.

Note the multiplicative coefficients `[1.0, 0.01, 0.1, 0.1]` are scaling
factors applied to the *stimuli before* dotting with the linear genome (they
keep different stimulus magnitudes commensurate). The MLP eats RAW stimuli
in [src/reward.py::compute_mlp_reward](../../src/reward.py) — confirm whether
to pass raw or pre-scaled stimuli to the MLP. The reward.py docstring says
raw, which matches the synthetic fixtures shipped in
[scripts/bake_mlp_reward_fixture.py](../../scripts/bake_mlp_reward_fixture.py)
and [dashboard/site/public/fixtures/mlp_reward_examples.json](../../dashboard/site/public/fixtures/mlp_reward_examples.json).
If you change this, regenerate the fixture so the heatmap viz stays calibrated.

### 3. Wire genomes through birth in `jax_evolution.py`

The existing spawn path in [src/jax_evolution.py](../../src/jax_evolution.py)
mutates only `reward_weights` (the linear vector). Add a parallel path that
mutates `reward_mlp_params` for the parent slot via
[mutate_mlp_genome](../../src/evolution.py) / `mutate_temporal_genome`.

`mutate_mlp_genome` uses `scipy.stats.t.rvs` for Student's-t mutation noise,
which is host-side numpy — it cannot be called inside JIT. Two options:

1. Stay host-side: pull the parent genome out of JAX, mutate on host,
   `jnp.array` the child back. This forces a sync per birth — fine if births
   are rare (tens to low hundreds per 1000 sim steps) but bad on a hot path.
2. Replace the scipy call with JAX-native Student's-t sampling
   (`jax.random.t` does not exist; use the ratio method or `jax.random.normal`
   divided by `jnp.sqrt(jax.random.chisquare / df)`). Then the entire spawn
   stays in JIT. Recommended.

### 4. Update the recorder to capture genomes

[scripts/replay_recorder.py](../../scripts/replay_recorder.py) currently saves
`reward_weights` as `(L, N, 4)`. For Axis 1/3 runs, it would need to save the
flattened MLP genome — which is constant per agent within their lifetime
under continuous birth-death, so per-frame recording is wasteful.

Better: bump the recorder to **v3** with a new section structure:

```python
# Static linear weights — kept for linear runs (current behavior)
reward_weights: int8  (L, N, 4)        # only when reward_type == "linear"

# v3 additions for MLP / temporal runs
reward_genomes_byid: float32 (n_unique_agents, genome_dim)   # one row per agent_id seen in window
reward_genomes_idmap: int32  (n_unique_agents,)              # agent_id for each row
genome_arch:    str  in meta.json                            # "linear", "mlp", "temporal"
genome_shape:   list  in meta.json                           # [hidden_size] for mlp, [k, hidden] for temporal
```

The dashboard's [replayLoader.ts](../../dashboard/site/src/lib/replayLoader.ts)
gains a `genomesById: Map<number, Float32Array>` field, falling back to
`null` for v2 replays.

Genome size budget: Axis 1 = 121 floats × ~500 unique agents × float32 = 240 KB
per replay. Axis 3 = 945 floats × ~500 = 1.9 MB. Both acceptable.

### 5. Remove the `?lab=mlp` gate on RewardLandscape

Once real genomes ship in v3 replays:

- [dashboard/site/src/components/RewardLandscape.tsx](../../dashboard/site/src/components/RewardLandscape.tsx)
  switches its data source from `fetchMlpFixtures()` to
  `data.genomesById[agentId]`.
- Surface it inside [AgentInspector.tsx](../../dashboard/site/src/components/AgentInspector.tsx)
  for the pinned agent on Axis 1 / Axis 3 replays — that's the natural home.
- The synthetic-fixture path in
  [scripts/bake_mlp_reward_fixture.py](../../scripts/bake_mlp_reward_fixture.py)
  becomes a debug-only utility; the public/fixtures/ JSON can stay as a unit
  test for the TS forward pass.
- Drop the `?lab=mlp` URL gate from [Replay.tsx](../../dashboard/site/src/pages/Replay.tsx).

## Validation gates

After step 2 (dispatch in jax_sim) and step 3 (genome mutation):

1. **Smoke**: re-run `axis1_smoke` and `axis3_smoke` (20K steps).
   Confirm population numbers still land in the same ballpark as
   [experiments-log.md axis1_smoke / axis3_smoke](../experiments-log.md)
   (axis1: prey~340 pred~24; axis3: prey~240 pred~30). If the populations
   crash, the MLP isn't producing a sensible reward signal — likely a
   stimulus-scaling mismatch (see step 2 note on coefs).

2. **Genome drift**: log mean and std of the flattened MLP genome across
   alive agents at each progress checkpoint. A working setup shows the std
   monotonically increasing for the first ~50K steps as Student's-t mutations
   accumulate.

3. **Real-MLP replay**: re-record an axis1 replay with v3 recorder; load
   it in the dashboard with `?lab=mlp`-removed RewardLandscape; click an
   agent and confirm the reward landscape renders. The `linear_evolved` and
   `threshold_fear` fixtures are good visual references for what "looks
   right."

4. **Re-run the smoke sweep formally** so the experiments log can be
   amended: re-tag `axis1_smoke` and `axis3_smoke` as pre-fix
   linear-config-mislabel, run new smokes post-fix, and update the log.

## Out of scope for this task

- The 2×2 architecture comparison (independent vs shared policy, continuous
  vs generational) from [experimental-plan.md §Phase 1b](../experimental-plan.md)
  — that is its own track.
- Phase 1a substrate work (mouth_smol et al.) — that's
  [findings.md §10-§11](../findings.md) territory.

## Correction (2026-04-27): Axis 4 has the same bug

The earlier draft of this doc placed Axis 4 (LSTM policy) out of scope,
claiming it was "already wired through `lstm_hidden` in SimState." That
was wrong. Confirmed via grep:

- `policy_type` has zero references anywhere in `scripts/` or `src/`
- `build_ppo_update_fn_lstm` (defined in [jax_ppo.py:181](../../src/jax_ppo.py#L181))
  is never imported by the runner — `scripts/run_experiment_jax.py:41`
  imports only `build_ppo_update_fn` (the MLP path), and `jax_sim.py:78`
  calls only that
- `lstm_hidden` field exists in SimState but `grep "lstm_hidden" src/jax_sim.py`
  returns zero matches — the field is allocated but never read

So Axis 4 has the same shape of bug as Axis 1 and Axis 3: configs flag
the variant, but the dispatch code is missing. Plumbing scope below
should include `policy_type` dispatch in the runner alongside
`reward_type` / temporal-buffer wiring.

## Files this touches (estimated)

- `src/jax_state.py` — add `reward_mlp_params` field, init dispatch
- `src/jax_sim.py` — branch reward computation on `reward_type`
- `src/jax_evolution.py` — mutate genome on birth; need JAX-native t-dist
  sampler to keep the spawn path in JIT
- `src/reward.py` — possibly inline mutation helpers if the JAX-native
  Student's-t lives there
- `scripts/replay_recorder.py` — v3 schema for genomes-by-id
- `dashboard/site/src/lib/replayLoader.ts` — v3 decode path
- `dashboard/site/src/components/RewardLandscape.tsx` — accept real genomes
- `dashboard/site/src/components/AgentInspector.tsx` — surface
  RewardLandscape for the pinned agent on MLP/temporal runs
- `dashboard/site/src/pages/Replay.tsx` — drop the `?lab=mlp` gate

Estimated effort: 1–2 days for steps 1–3; another half-day for steps 4–5
once real recordings are flowing.
