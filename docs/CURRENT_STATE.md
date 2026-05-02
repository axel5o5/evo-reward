# Current state — read this first if you're a new Claude agent

**Last updated:** 2026-05-02 evening (v8 at step ~2.40M / 23.4%, max predator age climbing 214K → 320K; three follow-up analyses on cohort survival flipped one v10 design call — keeping age-keyed LR aggressive at L2. L2 spec now locked. Plan: wait for v8 ≈ 3M, then launch L1. See findings §15.20 + §15.21).

## Where we are in the project

This repo replicates Kanagawa & Doya 2025 (arXiv:2507.09992v2) predator-prey
coevolution and runs three axis experiments that vary one design dimension
each from the paper baseline. The replication itself is done (it works — see
`docs/findings.md` §1-10). What's hard is finding a parameter regime where
both species survive long enough for evolution to do something interesting.

We are currently in **Phase 2** — running the axis experiments. The strategic
reset on 2026-04-29 (findings.md §15) narrowed scope to two axes (residual
reward genome, bin-aligned heading observation) and added stability scaffolds
to prevent extinction. After three baseline-validation iterations the
scaffolds are now strong enough that we're skipping further baseline runs
and going directly to axis-1.

## What's running / queued

- **Axis-1 v8 (residual reward genome) on L3:** running on GCP since
  2026-05-01 22:03 UTC, run_tag `2026-05-01T2203Z`. **At step 2.40M / 10.24M
  (23.4% complete).** Predator macros: pop=9, prey=344, max predator age
  319,897 (same individual went 214K → 320K over the last 140K steps).
  Predator weights still strongest K&D signal seen this project:
  `pred_w_pred = −2.97, pred_w_prey = +4.37`. Prey w_pred drifting more
  negative (mean −1.11 → −1.32 over 140K), tail-driven — slow population-
  wide fear formation. 84 catches/10K steps over the last window, healthy.
  Cohort survival analysis (findings §15.21 Analysis B): 64 predator
  deaths/M-step, 0% survival of newborn cohort over 140K — high turnover
  is the modal case despite long-tail survivors. Config:
  [configs/axis1_residual.yaml](../configs/axis1_residual.yaml).
- **Axis-1 v10-L1 (queued, hold for v8 ≈ 3M):** v10 mechanism additions on
  the cheapest iteration tier (~7.8h/1M CPU, ~2.5× faster than L3). Plan
  is to launch once v8 reaches ~3M to validate the cohort-survival pattern
  with one more checkpoint diff. Config:
  [configs/axis1_residual_mini.yaml](../configs/axis1_residual_mini.yaml).
- **Axis-1 v10-L2 (queued):** v10 mechanisms on the middle tier
  (~13.3h/1M CPU, ~1.5× faster than L3). Config:
  [configs/axis1_residual_fast.yaml](../configs/axis1_residual_fast.yaml).
- **Axis-2 (bin-aligned heading obs):** still queued. Config:
  [configs/axis2_aligned_smol.yaml](../configs/axis2_aligned_smol.yaml)
  (filename is stale; `experiment_name: axis2_aligned`, med-large scale).
- **Axis-3 / axis-4:** deferred (see findings.md §15.4). Configs moved
  into `configs/archive/` on 2026-05-01.

GCP infra: `evo-reward-gpu` VM (single L4), runs in tmux sessions.

> **Note on v8 vs v9.** The live v8 process loaded its config at startup,
> so it's running with the *pre-v9* values: `proximity_max_range: 200` and
> absolute `food_growth_rate: 0.5`. The v9 config tweaks (paper-faithful
> 120 + scale-relative food density) live in the same config files now and
> will take effect on the *next* launch. v8 will continue to completion (or
> until we kill it) on its frozen config.

## v9 config tweaks (2026-05-01) — paper-faithful proximity + per-area food

Two scale-correctness fixes landed today (commit `c7b81e1`). Neither
affect the running v8 process; they apply on the next launch.

1. **`proximity_max_range: 200 → 120`** in the active axis configs. The
   paper (Appendix A) specifies 120; emevo's TOML has 200, which we'd
   carried forward by accident (D27). At our world=880 the old 200
   covered 22.7% of world width, vs paper-spec 12.5% — predators were
   sensing far farther than intended. `baseline_faithful.yaml` was
   already at 120; v9 brings the axis configs in line. See findings §15.17.
2. **Scale-relative food growth rate.** New config key
   `food_growth_rate_at_960sq` (the rate at the paper's 960² world);
   resolver in [src/config_utils.py](../src/config_utils.py) scales it by
   `(world_size / 960)²` so per-unit-area food density stays paper-faithful
   on non-960 worlds. At world=880 → 0.5 × (880/960)² ≈ 0.420.
   `baseline_faithful.yaml` keeps the absolute key (identical at world=960).
3. **Configs reorg.** Top-level `configs/` now contains only the actively-
   running configs + paper reference + runtime/. Superseded and deferred
   configs moved into `configs/archive/`. See [configs/README.md](../configs/README.md)
   and [configs/archive/README.md](../configs/archive/README.md) for
   per-file status.

## ⭐ v10 framework (2026-05-02) — three-tier ladder + mouth/age-LR/death-age

The v8 run revealed three observations that motivated a config revision (findings §15.19):

1. Predator mouth `[0]` is below K&D's smallest mouth `[0, 1, 17]` — predators visibly missing prey passing close-but-off-center.
2. Median predator lives ~6,103 steps = ~6 PPO updates. Within-lifetime training mismatch.
3. We don't capture death-age distributions; live ages are inspection-paradox-biased.

These are addressed in v10 (findings §15.19, §15.20):

- **`predator_mouth_tactile_bins: [0, 1, 17]`** — paper-faithful "small" mouth (60° front arc).
- **Per-agent age-keyed LR schedule** (`lr_schedule_initial=1e-3` → `lr_schedule_final=3e-4` over 30K steps; new fields in `src/jax_ppo.py`). Boosts young agents' learning, decays linearly to base LR. Disabled when `lr_schedule_enable: false`.
- **Death-age ring buffer** (per-species 256-entry; new SimState fields populated in `process_births_and_deaths_jax`; logged in `progress.json` as `death_age_stats`).

### Three-tier ladder (L1 / L2 / L3)

Verified by three diagnostic analyses against the v8 step-2.26M checkpoint (findings §15.20). The headline finding: **population (max_agents²) is the dominant compute lever**; PPO is only 1-3% of CPU wall, so PPO cuts barely move total wall-clock. Hidden-size cuts also costly per SVD finding (top-32 SVs only capture 77.6% of the policy variance; eff. rank @95% ≈ 53 of 64).

| Tier | Config | max_agents | h/1M (CPU) | speedup | use case |
|------|--------|------------|------------|---------|----------|
| **L3** | `axis1_residual.yaml` (live v8) | 415 | 19.8h | 1.00× | paper-comparable; intentional fixes only |
| **L2** | `axis1_residual_fast.yaml` | 335 | 13.3h | **1.49×** | overnight directional answers, hidden=48 |
| **L1** | `axis1_residual_mini.yaml` | 220 | 7.8h | **2.54×** | cheap iteration on v10 mechanisms |

All three carry the v10 changes (mouth, age-LR, death-age). L2 keeps `n_physics_iter=4` (one substep cut), `policy_hidden_size=48` (SVD-defensible compromise). L1 takes `n_physics_iter=3`, `hidden_size=32`, smaller world, and stronger anti-extinction scaffolds (`ddb_max_boost=100`, `α=0.3`) to compensate for the small pop.

### v10 scaffold-tuning principle for sub-paper scales

When `max_agents` shrinks below paper geometry, do NOT scale DDB/DDM thresholds proportionally down. The cost of an LV crash at small pop is *higher*, so the safety net should engage *more* aggressively — not in proportion to cap. Use `ddb_max_boost` as the lever instead (only fires at low pop, safe at healthy pop). At L1 we also reduce `α` from 0.5 to 0.3 (more uniform breeding share) since per-agent variance is harsher when there are only ~10 active predators.

| Knob | L3 | L2 | L1 |
|------|----|----|----|
| `ddb_pred_threshold` | 10 | 10 | 8 |
| `ddb_prey_threshold` | 100 | 100 | 60 |
| `ddm_pred_threshold` | 10 | 10 | 8 |
| `ddb_max_boost` | 50 | **75** | **100** |
| `ddb_boost_distribution_alpha` | 0.5 | 0.5 | **0.3** |

### Implementation map (v10)

| File | Change |
|---|---|
| `configs/axis1_residual_fast.yaml` | NEW — v10-L2 config |
| `configs/axis1_residual_mini.yaml` | NEW — v10-L1 config |
| `src/jax_state.py` | Added `death_age_ring_prey/pred` and indices to `SimState` |
| `src/jax_lifecycle.py` | `_write_death_ages_jax` helper + integration in `process_births_and_deaths_jax` |
| `src/jax_ppo.py` | `_lr_scale_for_age` + per-agent update scaling in MLP and LSTM PPO paths |
| `src/jax_sim.py` | `n_physics_iter` plumbed via config (was hardcoded constant) |
| `scripts/run_experiment_jax.py` | `state.ages` passed to PPO; death-age stats in `progress.json`; one-line death-age summary in tail logs |
| `scripts/bench_l2_vs_l3.py` | NEW — wall-clock A/B benchmark for any of L1/L2/L3 |
| `tests/test_jax_ppo_update.py` | Updated PPO call signature (`state.ages` arg) |

### Deferred features (post-launch)

- **Rate-based α** (alternative to energy-share). Would distribute DDB boost on per-agent catch-rate (predators) or feed-rate (prey). Real fitness signal but requires new SimState fields and checkpoint format bump. Considered, scoped, and deferred — only revisit if v10 results show energy-share α is the bottleneck.

## ⭐ The current scaffold framing (2026-05-01) — DDB+DDM, energy-weighted DDB

The K&D paper has a known stability problem: with eta=0.5 (paper baseline),
predators tend to over-pressure prey, then crash. To get long enough
co-evolution for ablations to be meaningful, we use **two scaffolds operating
in different dimensions**:

- **DDM (Density-Dependent Metabolism, uniform)**: keeps low-energy predators *alive* during LV crashes
- **DDB (Density-Dependent Breeding, energy-weighted)**: keeps low-energy predators *from reproducing*

Together: bad hunters survive but don't propagate their genes. Population
stable, reproduction concentrated on top fitness. Both are explicitly
documented as scaffolds (not biological claims) and disclosed in findings.

**Core idea:** at low predator population, scaffolds activate to prevent
extinction. But the breeding-rate scaffold is allocated by within-species
energy share — so the *total* species-level breeding pressure scales with
population (rescue function preserved) while the *individual* selection
pressure stays sharp (high-energy hunters carry recovery, not random
survivors). This is the result of four iterations of tuning; earlier
designs uniformly rescued all individuals which diluted selection. See
[findings.md §15.11-§15.14](findings.md) for the full journey, especially
**§15.14** for the current design.

If you're tuning scaffolds for a future run, this is the framing:

### DDB — Density-Dependent Breeding (the only scaffold currently active)

When predator population is low, breeding becomes easier in two ways:

1. **Threshold drop.** `zeta_b_eff = zeta_b * factor`, where `factor` follows
   a squared-saturation curve `f(N) = max(floor, N²/(N²+T²))`. With current
   knobs (pred T=10, prey T=100, floor=0): N=4 → 0.14, N=10 → 0.50,
   N=15 → 0.69, N=24 (peak with cap=40) → 0.85, N=30+ → ~0.90+ (off).
2. **Rate boost with continuous distribution `α ∈ [0, 1]`** (`ddb_boost_distribution_alpha`).
   Total species-level breeding budget = `N / max(factor, 1/max_boost)`.
   The budget is redistributed among individuals by `share_i ∝ energy_i^k`
   where `k = α / max(1−α, ε)`. **Total budget preserved at all α**;
   the parameter only changes *how concentrated* the budget is.

   - **α = 0.0**: uniform (every agent gets the same boost). K&D-faithful.
   - **α = 0.5** (current default for axis runs): linear — share ∝ energy. Top agent at energies [800, 100, 100] gets 80% of budget.
   - **α = 1.0**: winner-take-all. Only the top-energy agent breeds.

   Top-agent share at energies [800, 100, 100] for various α:

   | α | top share |
   |---|---|
   | 0.0 | 33% |
   | 0.3 | 67% |
   | 0.5 | 80% (v8) |
   | 0.7 | 92% |
   | 0.9 | 99% |

### DDM — Density-Dependent Metabolism (RESTORED, 2026-05-01 — see §15.15)

`d_b_eff = d_b * factor(N)` — predator passive decay scaled by the same
squared-saturation curve as DDB. At low predator pop, the decay rate
drops, giving low-energy individuals more time. Action cost (`alpha_e *
action_norm`) stays unscaled, so they still pay for moving — DDM only
extends the floor of survival, doesn't make them immortal.

**Why DDM is needed even with energy-weighted DDB:** in deep LV crashes,
ALL predators lose energy simultaneously. Without DDM, full-cost decay
finishes them off before any individual can recover, regardless of how
the breeding boost is allocated. v7 (DDM dropped) went extinct at step
88K to validate this lesson empirically.

### Why this design

Earlier weaker scaffolds (DDB+DDM, floor=0.3, no rate boost) ran for 1.35M
steps then extincted via "trophic-collapse-via-herd" — strong prey herding
evolved without fear. Strong scaffolds (DDB+DDM, floor=0, max_boost=50,
T=4) prevented extinction but lost diversity (peak 18 → 2 ancestral
survivors with near-init weights — see findings §15.12). Med-large scale
+ T=10 (§15.13) preserved diversity but blunted selection. Energy-weighted
boost (§15.14) addressed selection alignment, but dropping DDM (§15.14b /
v7) caused extinction at step 88K because energy-weighted breeding
needs survivors to redistribute among. **v8 keeps both: DDM for population
survival, energy-weighted DDB for reproduction concentrated on top fitness
(§15.15).**

See [docs/findings.md §15.11-§15.15](findings.md) for the full calibration
journey.

## The two axis hypotheses

Each axis varies a single design dimension from the K&D paper baseline.

### Axis 1 — residual reward genome

**Hypothesis:** "K&D's linear reward genome is sufficient — adding nonlinear
capacity does not help."

**Design:** reward = K&D linear + zero-init MLP residual
(input(4) → Dense(4, tanh) → Dense(1), 25 params). At t=0 the residual outputs
zero so the reward is identical to K&D linear. Mutations grow the residual
gradually iff it improves fitness.

Two interpretable outcomes:
- Residual stays ≈0 across the run → "linear is sufficient."
- Residual weights grow → "evolution found nonlinear structure that helps."

Earlier full-MLP-replacement attempts (axis1_v1, v2) extincted before
evolution could find anything useful. The residual design avoids the
bootstrap failure: t=0 reward is proven-stable K&D linear.

**Config:** [configs/axis1_residual.yaml](../configs/axis1_residual.yaml).
**Code:** `src/reward.py::ResidualRewardMLP`, `compute_residual_reward`,
`init_residual_genome`; `src/jax_evolution.py::mutate_residual_genome_jax`.

**Open analysis (designed, not yet implemented):** Q1 (binary "is the
residual being utilized?") is mostly answered — predator residual L1
went 0 → ~2.8 by step 440K of the v8 run. Q2 ("does the residual encode
genuine nonlinear structure, or just reinforce the linear gradient?") is
the more interesting question and needs an offline analysis script. Full
design and rationale in
[docs/proposals/axis1-residual-analysis.md](proposals/axis1-residual-analysis.md).
Pick this up in a future session — ~150 LOC, runs against any checkpoint
in `gs://evo-reward-ckpts/`.

### Axis 2 — bin-aligned heading observation

**Hypothesis:** "Observing other agents' headings (not just positions)
shifts evolution toward fear-of-predator vs. herd-with-prey."

**Design:** per proximity bin, each species channel reports
`[distance, sin(rel_heading), cos(rel_heading)]` of the closest agent of
that species in that bin. obs_dim = 333 (vs 205 baseline).

Earlier social-slot designs (n_kin/n_other) had a binding problem —
policy had to multiplicatively bind "predator in bin X" to "heading in
slot Y". Bin-aligned encoding co-locates kinematics with position,
eliminating binding burden. sin/cos avoids the "no agent" / "agent facing
forward" aliasing bug (both → 0).

**Config:** [configs/axis2_aligned_smol.yaml](../configs/axis2_aligned_smol.yaml)
(filename is historical; `experiment_name: axis2_aligned`, middle-scale).
**Code:** `src/observations.py` — `_single_proximity_agents_with_heading`,
`_per_channel_encoding`. Toggle via `proximity_encoding: "distance_and_heading"`.

## Document layout

- [docs/findings.md](findings.md) — running log of empirical findings.
  §15 is the strategic reset; §15.6-15.11 are the validation iterations
  that led to current scaffold tuning. **Read §15 if you need full context.**
- [docs/scope-reset-2026-04-29.md](scope-reset-2026-04-29.md) — standalone
  handoff doc for partners/teammates with less code context.
- [docs/technical-spec-kd-replication.md](technical-spec-kd-replication.md)
  — replication of the paper itself, all the K&D params and design choices.
- [docs/emevo-diff.md](emevo-diff.md) — running list of bugs and design
  divergences from the upstream emevo reference implementation. D19 (slot
  fix) and D28 (shared-credit catch energy) are the most recent.
- [docs/experiments-log.md](experiments-log.md) — chronological log of runs.
- [docs/proposals/](proposals/) — designed-but-not-yet-implemented analyses
  and follow-ups, structured for a future agent to pick up cold.

## Common workflows

### Smoke test locally
```sh
python3 scripts/run_experiment_jax.py \
    --config configs/axis1_residual.yaml \
    --runtime configs/runtime/mac.yaml \
    --max-steps 5000
```

### Sync changes to GCP VM
```sh
./scripts/sync_to_gpu.sh   # check the script — it's idempotent
```

### Launch a run on GCP
```sh
gcloud compute ssh evo-reward-gpu --command "tmux new -d -s axis1 \
    'cd ~/evo-reward && python3 scripts/run_experiment_jax.py \
     --config configs/axis1_residual.yaml --runtime configs/runtime/gcp.yaml'"
```

### Tail logs
```sh
gcloud compute ssh evo-reward-gpu --command "tmux capture-pane -t axis1 -p | tail -80"
```

## Things to know

- **Don't run a fresh baseline before going to axes.** We've validated the
  scaffolds three times (smol, med, med+DDM). Time pressure is real; axis
  runs themselves serve as additional validation.
- **Mac uses Python 3.13 with system JAX 0.9.2 for smoke tests.** GCP uses
  the venv. See [docs/local-setup.md](local-setup.md).
- **Don't push directly to `main` from the GCP VM** — work happens on the
  Mac, syncs to GCP read-only-ish.
- **The user prefers commits direct to main** for routine edits; skip
  feature branches and PRs.
- **Replays go to `evo-reward-replays-public` GCS bucket.** All axis configs
  have `replay_bucket` set; if you add a new config it needs that line.

## How to think about scaffolds going forward

If a future run still shows extinction or selection issues, the toolbox now is:

| Knob | What it does | When to change it |
|---|---|---|
| `ddb_pred_threshold` (currently 10) | Where the scaffold curve hits 50% (factor=0.5 at N=T) | Lower if extinction happens at deep bottlenecks; raise if selection is too blunted at healthy peak |
| `ddb_floor` (currently 0.0) | Minimum factor — pins the curve at extreme low N | Raise toward 0.1-0.3 if you want baseline selection-pressure relief always present |
| `ddb_max_boost` (currently 50) | Cap on the inverse factor in the rate formula | Lower (e.g., 20) if breeding is too aggressive at deep bottlenecks |
| `ddb_boost_distribution_alpha` (currently 0.5) | How concentrated the rate boost is on high-energy agents | Lower (0.3) if extinction risk during LV crashes; raise (0.7) if selection seems too weak. **Don't go to 0.95+ unless deep bottlenecks are impossible** — see v7 lesson |
| `stability_mechanism` (currently `"ddb"`) | Which scaffolds are active | Add `"ddb_ddm"` if a *true* extinction emergency occurs that DDB-rate-boost alone can't rescue |

**Diagnosis playbook:**

- **Extinction at low N (pred ≤3):** scaffolds aren't strong enough. Check `factor` at the death-point N — if >0.4, lower the threshold or floor.
- **Mean reward weights drift to zero with high variance:** selection is too weak. With energy_weighted boost this shouldn't happen at healthy populations, but if it does, check that population is actually healthy (factor near 1) — if you're stuck in the N=10-15 range, scaffolds are constantly engaged.
- **Population locks at low N never recovers:** ddb_max_boost too low, OR all individuals at zero energy (look at energy quartiles). If everyone's starving, the rate boost can't help — population probably doomed.
- **Bad hunters survive forever:** check that DDM is actually disabled (`stability_mechanism: "ddb"` not `"ddb_ddm"`). DDM is what keeps them alive past starvation.

**The current design philosophy:** scaffold the *population* to prevent extinction (DDM), but allocate the *breeding budget* to high-fitness individuals (DDB with α > 0). The two scaffolds work in different dimensions: DDM keeps low-energy agents alive; α-controlled DDB ensures they don't reproduce. Bad hunters survive temporarily, fail to propagate, eventually starve out. Population stable, selection aligned.

**Don't push α too high.** v7 (effectively α=0.5 + DDM dropped) collapsed at step 88K because energy-weighted breeding has no fitness gradient to concentrate on when ALL predators are simultaneously low-energy. Even with DDM intact, α near 1.0 may have similar fragility — keep α ≤ 0.7 unless you've validated the regime.

## Run history snapshot

| Run | Config | Outcome |
|---|---|---|
| baseline_smol_ddb | small + DDB floor=0.3 | Extinct ~80K |
| baseline_med_ddb | medium + DDB floor=0.3 | Lone-survivor starvation ~100K |
| baseline_med_ddb_ddm | medium + DDB+DDM floor=0.3 | 1.35M then trophic-collapse-via-herd |
| axis1_residual_T4_run1 | medium + DDB+DDM strong floor=0 max_boost=50 | Diversity collapse by 120K (→ §15.12) |
| axis1_residual v3 (uniform boost) | med-large T=10 + DDB+DDM uniform | 230K, healthy LV but selection diluted (→ §15.14) |
| axis1_residual v7 (no DDM) | med-large T=10 + DDB only + α=0.5 | **Extinct at 88K** — DDM not optional (→ §15.15) |
| **axis1_residual v8 (running)** | **med-large T=10 + DDB+DDM + α=0.5** | **At step 2.35M (22.9%): prey=260, pred=12. Predator weights showing strong K&D-aligned signal: w_pred=−2.97±0.96, w_prey=+4.37±0.76. 72 catches/10K steps healthy. Mid-LV-cycle dip from previous 372/14 — scaffolds expected to engage on rebound.** |
| axis1_residual v9 (config-only diff) | v8 settings + paper-faithful proximity (200→120) + scale-relative food_growth_rate (per-area density) | §15.17; takes effect on next launch |
| **axis1_residual_mini v10-L1 (queued)** | **600² / cap 200/20 / hidden=32 / n_phys_iter=3 / ddb_max_boost=100 / α=0.3 / mouth=[0,1,17] / age-LR / death-age ring** | **2.54× faster than L3 on CPU. Cheap iteration tier. §15.20.** |
| **axis1_residual_fast v10-L2 (queued)** | **750² / cap 300/35 / hidden=48 / n_phys_iter=4 / ddb_max_boost=75 / α=0.5 / mouth=[0,1,17] / age-LR / death-age ring** | **1.49× faster than L3 on CPU. Middle tier. §15.20.** |
