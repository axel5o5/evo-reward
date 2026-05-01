# Current state — read this first if you're a new Claude agent

**Last updated:** 2026-05-01 (axis-1 v8 launch — DDB+DDM with α=0.5 boost distribution; α is now a continuous tuning knob in [0,1], see findings §15.16).

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

- **Axis-1 (residual reward genome):** to be launched — configs at
  [configs/axis1_residual.yaml](../configs/axis1_residual.yaml).
- **Axis-2 (bin-aligned heading obs):** queued — configs at
  [configs/axis2_aligned_smol.yaml](../configs/axis2_aligned_smol.yaml)
  (filename is stale; experiment_name is `axis2_aligned`, middle-scale).
- **Axis-3 / axis-4:** deferred (see findings.md §15.4).

GCP infra: `evo-reward-gpu` VM (single L4), runs in tmux sessions.

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
| **axis1_residual v8 (running)** | **med-large T=10 + DDB+DDM + α=0.5** | **TBD** |
