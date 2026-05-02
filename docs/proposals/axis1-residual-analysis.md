# Proposal: Axis-1 Residual MLP Characterization

**Status:** designed, not yet implemented.
**Drafted:** 2026-05-02 during v8 axis1_residual run @ step ~460K.
**Owner:** open — pick up in any future session.
**Estimated effort:** ~150 LOC for the analysis script + ~10 LOC for the
in-loop L1 logging prerequisite. One focused session.

---

## tl;dr for a future agent

The axis-1 reward genome is `r(s) = r_linear(s) + r_mlp_residual(s)` where
the MLP starts at zero-init and evolves under `residual_mutation_scale=0.03`,
`residual_weight_clip=±5.0`. Every agent carries 25 residual params (a
4→4→1 MLP).

**Question 1 (binary, ~answered): is the MLP being utilized?**
Yes. By step 440K of the v8 run, predator residual L1 went from 0 → mean 2.81
(min 1.78, no surviving flat-zero predators). Prey 0 → mean 3.58. See
"Existing evidence" below for the table.

**Question 2 (the interesting one, this proposal): does the MLP capture
genuine *nonlinear* structure, or is it just adding gain to the same
direction the linear weights already point?** That's what this script
answers.

---

## The two hypotheses

Both are scientifically valid outcomes for axis 1; the goal is to distinguish
them empirically rather than assume.

**H1 — "Reinforcement."** The MLP residual is approximately a linear
function of the stimuli: `r_mlp(s) ≈ b₀ + b·s`, with `b` aligned to the
linear weights (`coefs ⊙ w_linear`). In this case the MLP is just a
gain knob — it makes the existing linear gradient stronger. Result is
publishable as "K&D's linear reward is the right structural form;
extra capacity collapses to extra gain."

**H2 — "Genuine nonlinearity."** `r_mlp(s)` has structure that linear
cannot represent: saturation, thresholds, sign flips at extreme inputs,
or — most interestingly — *interactions between stimuli* (e.g., "action
cost is steep when there's no prey but mild when prey is in sight").
Result is publishable as "MLP capacity finds nonlinear reward structure
that improves fitness."

There's a third mixed case (the most realistic): **partial nonlinearity** —
some agents/lineages are mostly linear, others have meaningful
interaction terms. That's also publishable and probably the most
interesting outcome for a small-MLP regime like this.

---

## Existing evidence (Q1, already collected)

Pulled from `gs://evo-reward-ckpts/results/axis1_residual/seed_0/2026-05-01T2203Z/checkpoints/`.

| step | active preds | pred mean L1 | pred min L1 | prey mean L1 | prey max single param |
|------|--------------|--------------|-------------|--------------|-----------------------|
| init | 9            | 0.000        | 0.000       | 0.000        | 0.000 (zero-init)     |
| 280K | 12           | 1.99         | 0.00        | 2.90         | **5.00 (hit ±5 clip)** |
| 440K | 8            | 2.81         | **1.78**    | 3.58         | 2.33                  |

Key observations:
- Order-of-magnitude bigger than mutation noise alone (`σ=0.03 × √generations`
  predicts ~0.5 by 440K, we see ~2.8). Selection is pushing residuals up.
- No surviving flat-zero predator at 440K → lineage propagation, not just
  random walk.
- One prey lineage hit the clip at 280K then disappeared by 440K. Worth
  watching for clip-hit rates as a signal of pathological evolution.

What this evidence **does not tell us**: whether the residual encodes
something the linear can't, or just makes the linear gradient steeper.
That's Q2.

---

## The three diagnostics

Listed in order of how cleanly they separate H1 from H2.

### 1. R² of best linear fit to `r_mlp`

For each agent a, sample N stimulus vectors `S = {s₁..s_N}` from a
realistic input distribution (see "Stimulus distribution" below). Compute
`y_a = [r_mlp_a(sᵢ)]` for all i. Fit `y_a ≈ b₀ + b·s` by ordinary least
squares. Compute R² = 1 − SSR/SST.

**Interpretation:**
- R² ≥ 0.95 → MLP is essentially linear. Reinforcement (H1).
- 0.7 ≤ R² < 0.95 → MLP has weak nonlinearity. Mixed case.
- R² < 0.7 → MLP has substantial nonlinear structure. H2.

### 2. Cosine alignment between best-linear-fit `b` and `coefs ⊙ w_linear`

Even when R² is high, *what direction* the residual points matters.

- cos(b, coefs ⊙ w_lin) ≈ +1 → reinforcement: MLP makes the same
  gradient stronger.
- cos ≈ 0 → MLP points in a different stimulus combination than the
  linear part (e.g., linear says "more prey better" but MLP encodes
  "more action better" — a different objective dimension).
- cos < 0 → counter-correction: MLP partially cancels the linear part.
  Surprising outcome; would suggest the linear weights are over-shooting
  and the MLP is reining them in.

### 3. Pairwise interaction terms (the punchline for H2)

Linear cannot encode *interactions*: by definition `r_lin(s)` has zero
cross-partials `∂²r_lin / ∂sᵢ∂sⱼ` for i≠j. Any nonzero
`∂²r_mlp / ∂sᵢ∂sⱼ` is structure linear cannot reach.

For 4 stimulus dimensions there are C(4,2)=6 interaction pairs:
(eat,act), (eat,prey), (eat,pred), (act,prey), (act,pred), (prey,pred).

Estimate each by central finite differences over a sampled grid:
```
∂²r_mlp/∂sᵢ∂sⱼ ≈ [r(s+h_i+h_j) − r(s+h_i−h_j) − r(s−h_i+h_j) + r(s−h_i−h_j)] / (4 h²)
```
Compute the magnitude per pair. The agent's "interaction signature" is
a 6-vector. Compare to its own diagonal terms (∂²r/∂sᵢ²) and to the
linear part (which is identically zero on all 2nd derivatives).

**Interpretation:**
- All 6 cross-partials small → nonlinearity is per-axis only (saturation,
  thresholds). Still H2 but a weaker form.
- 1+ cross-partials large → genuine interactions encoded. Strongest H2.
- Per-axis stories worth checking: large `∂²r/∂s_eat²` < 0 means eat
  reward saturates (concave); > 0 means eat reward accelerates.

### Summary statistic

For each agent we get **(R², cosine, max |interaction|)**. Plot the
population in (R², max |interaction|) space — where do the lineages cluster?
A diagonal cloud (high R² ↔ low interaction) just confirms the linear-fit
diagnostic. A vertical spread at a given R² would say "agents with similar
'how linear' scores still differ in *which* nonlinearity they encode."

---

## Stimulus distribution

Two valid choices, do both:

**Sampled-from-replays:** load the most recent N replay files
(gs://evo-reward-replays-public/.../step_0XXXXXXX/), extract the actual
stimulus vectors observed by surviving agents. This gives the *realized*
distribution — what the agents actually experienced. **Recommended primary
distribution.**

**Uniform-grid:** 4D regular grid over `[-1, +1]^4` after the same
post-step normalization the runner applies. Coarse grid — say 8 points per
axis = 4096 evals per agent. Cheap (25 params × 4096 = 100K MAC per agent).
Gives a "what would the agent reward in counterfactual stimuli" answer,
useful for spotting threshold/sign-flip behaviors that don't occur in the
realized distribution.

Run both and compare; if conclusions agree, story is robust.

---

## Implementation sketch

### File: `scripts/analyze_residual.py` (new, ~150 LOC)

```python
"""analyze_residual.py
Per-agent characterization of the MLP residual reward genome.

Usage:
    python3 scripts/analyze_residual.py \
        --checkpoint gs://evo-reward-ckpts/results/axis1_residual/seed_0/2026-05-01T2203Z/checkpoints/step_00440000.npz \
        --config configs/axis1_residual.yaml \
        --out-dir analysis/axis1_residual_step440k/

Produces:
    analysis/.../per_agent.csv        # one row per active agent
    analysis/.../summary.json         # population aggregates
    analysis/.../heatmap_pred_top3.png  # 2D r_mlp slices for top-3 nonlinear preds
"""
```

Function decomposition:

```python
def load_agent_genomes(ckpt_path, config) -> tuple[mlp_apply_fn, params_per_agent, species, is_active, w_linear]:
    """Load checkpoint, return per-agent MLP params + linear weights + masks."""

def stimulus_distribution(config, replay_glob=None) -> np.ndarray:
    """Either sample from replay frames or build uniform grid. Shape (N, 4)."""

def per_agent_diagnostics(mlp_apply_fn, params_per_agent, w_linear, coefs, S) -> pd.DataFrame:
    """For each active agent, return row with: r2_linear_fit, cos_align,
    max_abs_interaction, max_abs_per_axis_curvature, plus 6 individual
    interaction terms."""

def population_summary(df, species_mask) -> dict:
    """Aggregate by species. Mean ± std for each diagnostic."""

def top_k_heatmaps(df, k, mlp_apply_fn, params_per_agent, out_dir):
    """For the k agents with highest max_abs_interaction, plot 2D slices
    of r_mlp(prey, pred) with eat/act fixed at distribution mean."""
```

The MLP apply function reuses `src/reward.py`'s residual model. Pull the
flax module, vmap over agents.

### Prerequisite: residual L1 in `_log_progress`

Edit `scripts/run_experiment_jax.py` `_log_progress` (currently lines
285-380). After the existing `pd_pred_m, pd_pred_s = ...` reads:

```python
# Residual MLP utilization summary. Walks reward_mlp_params, computes
# per-agent L1 norm (sum |w| over all 25 params), then mean±std per species.
import jax.tree_util as jtu
leaves = jtu.tree_leaves(state.reward_mlp_params)
# Each leaf shape (n_agents, ...). Reshape per-agent and sum across leaves.
per_agent_l1 = sum(jnp.abs(leaf).reshape(leaf.shape[0], -1).sum(axis=1) for leaf in leaves)
prey_resid_l1 = jnp.where(prey_mask, per_agent_l1, 0.0)
pred_resid_l1 = jnp.where(pred_mask, per_agent_l1, 0.0)
n_prey_active = jnp.maximum(jnp.sum(prey_mask), 1)
n_pred_active = jnp.maximum(jnp.sum(pred_mask), 1)
prey_resid_mean = float(jnp.sum(prey_resid_l1) / n_prey_active)
pred_resid_mean = float(jnp.sum(pred_resid_l1) / n_pred_active)
# (also compute std the same way; omitted here for brevity)
```

Then in the print line, append `| residL1 prey={prey_resid_mean:.2f} pred={pred_resid_mean:.2f}`.

Also persist into `jax_metrics`: add fields `prey_residual_l1_mean`,
`pred_residual_l1_mean` to `JaxMetrics` and the `record()` function in
`src/jax_metrics.py`. This way the dashboard can plot residual L1
trajectory without reloading checkpoints.

**Cost:** sub-microsecond per log call; ~15 LOC total across runner +
metrics module.

**Not needed if** the analysis script gets implemented and run frequently
enough to provide the same signal — but the in-loop logging is much
cheaper per-data-point than downloading 127 MB checkpoints.

---

## Outputs and interpretation guide

### per_agent.csv columns

| col                       | meaning                                                |
|---------------------------|--------------------------------------------------------|
| agent_id                  | global agent id                                        |
| slot                      | physics slot 0..499                                    |
| species                   | 0=prey, 1=pred                                         |
| age_steps                 | how long this agent has lived                          |
| linear_w_eat,_act,_prey,_pred | the agent's linear weights                          |
| residual_l1               | sum |w| over the 25 residual params                    |
| residual_l_inf            | max |w| over the 25 residual params (clip-hit detector)|
| r2_linear_fit             | R² of best linear fit to r_mlp                          |
| cos_align                 | cosine sim with `coefs ⊙ w_linear`                     |
| max_abs_interaction       | max |∂²r_mlp/∂sᵢ∂sⱼ| over the 6 cross-partials         |
| interaction_eat_act       | individual cross-partial                                |
| interaction_eat_prey      | ...                                                    |
| ... (6 cross-partials)    | ...                                                    |
| max_abs_curvature         | max |∂²r_mlp/∂sᵢ²| over diagonal                       |

### summary.json fields

```json
{
  "step": 440000,
  "n_pred_active": 8,
  "n_prey_active": 283,
  "pred": {
    "r2_linear_fit": {"mean": 0.83, "std": 0.07, "min": 0.71, "max": 0.94},
    "cos_align":     {"mean": 0.62, "std": 0.21, "min": 0.18, "max": 0.91},
    "max_abs_interaction": {"mean": 0.34, "std": 0.18, "min": 0.08, "max": 0.71},
    "fraction_with_R2_below_0.7": 0.0
  },
  "prey": { ... },
  "interpretation_hints": [
    "pred R² mean 0.83 → most preds are mostly-linear",
    "pred cos_align mean 0.62 → MLP partially aligned with linear, not full reinforcement",
    "max interaction term moderate (0.34) → some 2nd-order structure; worth heatmaps"
  ]
}
```

(The interpretation_hints are static heuristics computed from the
aggregates — see the script for the rules.)

### Heatmaps

For the top-3 most-nonlinear surviving predators (highest
`max_abs_interaction`), plot `r_mlp(prey, pred)` as a 2D heatmap with
`eat` and `act` fixed at distribution-mean values. Side-by-side with
`r_linear(prey, pred)` for the same agent. The visual "is the MLP
heatmap a tilted plane like the linear one, or does it have curved
contours / saddle points?" tells the story instantly.

---

## Data sources

**Checkpoints (this run):**
`gs://evo-reward-ckpts/results/axis1_residual/seed_0/2026-05-01T2203Z/checkpoints/`
- 20K cadence to 200K, 50K to 1M, 100K thereafter (v9 taper applies to
  future runs; this run is on flat 20K).
- Each ckpt is ~127 MB.

**Replays (for stimulus distribution):**
`gs://evo-reward-replays-public/axis1_residual/seed_0/2026-05-01T2203Z/`
- Each replay = 109 MB, 10K consecutive frames.
- A single replay yields ~10K × ~30 active agents = ~300K stimulus vectors.
- Frame format documented in `scripts/replay_recorder.py:40-58`.

**Cross-checkpoint comparison:** ideally run the script at step 100K,
300K, 500K, 1M (and however far the run goes). Watch the diagnostics
evolve. If `max_abs_interaction` is *growing* steadily → genuine
discovery of nonlinear structure over evolutionary time, even stronger
H2 evidence than a single-snapshot result.

---

## Validation / sanity checks

Before trusting any result, verify:

1. **Init checkpoint check.** Run the script on `step_00000000.npz` (or
   construct from `init_simstate`). All R² should be undefined (zero
   variance) or 1.0 (perfectly linear: zero is linear). All interactions
   should be exactly 0. Deviations indicate a bug in the script.
2. **Linear-only baseline.** Construct synthetic agents whose MLP params
   are zero. Verify R²=1.0, cos_align=undefined (b=0), max_abs_interaction=0.
3. **Linear-with-known-bias.** Construct synthetic agents whose MLP is
   *exactly* `r_mlp(s) = 2 · (coefs ⊙ w_linear) · s`. Verify R²=1.0,
   cos_align=+1.0, max_abs_interaction=0. This catches sign/normalization
   bugs in the linear-fit cosine computation.
4. **Known-nonlinear agent.** Inject `r_mlp(s) = s_eat · s_prey`. Verify
   R² < 0.5, max_abs_interaction[eat,prey] >> 0, others ≈ 0. This catches
   bugs in the cross-partial estimator.

These four sanity checks should be the first 4 unit tests for the
script.

---

## Open extensions (don't block the core analysis)

These are fruitful follow-up questions once the core script lands.

### Lineage tracking of residual structure

Use `parent_ids` + the residual params to reconstruct ancestry. Do
sibling lineages converge on similar residual structures (= heritable
discovery), or do they diverge (= each lineage exploring its own niche)?
Pairs naturally with `scripts/lineage_analysis.py`.

### Behavioral validation

The residual is presumed to shape behavior — but does it actually? Two
agents with similar linear weights but very different residuals: do they
*move* differently? This needs a behavioral readout from replays
(approach speeds, separation distances, kin-vs-prey reaction times).
Probably 200 LOC, separate effort, but the value is high — connecting
the genome to phenotype is the actual scientific contribution.

### Cross-axis comparison

When axis-2 (social_obs) and axis-3 (temporal) configs run on the same
stable substrate, run the same diagnostic. Does the *kind* of
nonlinearity differ across axes? E.g., does temporal-context
encoding push more interaction-term energy than the baseline linear-extra?

### Mutation-scale sensitivity

The current `residual_mutation_scale=0.03` may be too small or too large
for the "right" regime. A focused sweep (0.01, 0.03, 0.1, 0.3) over a
shorter run (500K steps) on the same scaffold could establish whether
residual L1 saturates differently. Bigger picture: at what scale does
the MLP start to encode interactions vs just gain?

### Capacity sweep

`residual_hidden_size=4` is a guess. At hidden=2 the MLP has ~13 params
and can encode at most 2 axes of nonlinearity. At hidden=8, 49 params
and richer interaction space. Comparing residual L1 vs hidden_size at
matched run length tells us about capacity-vs-evolution tradeoffs —
the kind of result that goes nicely in a discussion section.

---

## Reading list for picking this up cold

In order, ~30 minutes total:

1. [docs/CURRENT_STATE.md](../CURRENT_STATE.md) — what's running, what's
   the state of axis-1.
2. [docs/findings.md §15](../findings.md) — full v8 narrative, why
   axis-1 was designed this way.
3. [configs/axis1_residual.yaml](../../configs/axis1_residual.yaml) lines
   1-46 — the design rationale comment.
4. [src/reward.py](../../src/reward.py) `init_residual_genome`,
   `apply_residual` (or whatever the apply function is called) — what
   the MLP looks like and how it's plumbed.
5. [src/jax_evolution.py](../../src/jax_evolution.py) line 80
   `mutate_residual_genome_jax` — mutation mechanics.
6. [src/jax_state.py:211](../../src/jax_state.py) — where `reward_mlp_params`
   gets initialized.
7. [scripts/run_experiment_jax.py:285](../../scripts/run_experiment_jax.py)
   `_log_progress` — where to plug residual L1 logging.
8. The two-checkpoint inspection above (this doc, "Existing evidence")
   is reproducible end-to-end with ~30 lines of code; redo it as a warm-up.

When the script lands, append a new section to `findings.md` with
results and link back here.

---

## Why this matters (research-level framing)

The published K&D linear reward is a strong baseline because it's
simple and provably stable. Axis 1's contribution is to ask: **can a
small amount of additional capacity, evolved under the same rules,
discover reward structure that linear can't?** That question becomes
testable only with the diagnostics above. Without them, axis 1
produces a number ("residual L1 grew to 2.8") that's hard to argue
from. With them, axis 1 produces a *characterization* — "the residual
is mostly linear-aligned but encodes a small but nonzero (eat × prey)
interaction term that grows over evolutionary time" — which is the
kind of claim that survives review.
