# Current state — read this first if you're a new Claude agent

**Last updated:** 2026-05-03 (v8 went extinct around step 3.7-3.8M; DDB/DDM retuned per §15.22; configs reorganized into per-axis × per-tier layout; axis-2 obs encoding redesigned to approach-angle + speed §15.24. Ready to launch `axis1/tiny.yaml` on GCP. See findings §15.22-§15.24).

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

- **Axis-1 v8 (extincted):** ran on GCP from 2026-05-01 22:03 UTC, run_tag
  `2026-05-01T2203Z`. **Predator pop went 12 → 6 → 7 → 0 between step
  3.50M and 3.80M.** System has been frozen since with prey at cap and
  no predators. Findings §15.22 has the full post-mortem — DDB+DDM
  scaffolds with T=10 don't engage strongly enough at the death-spiral
  operating point (pop=7 with median energy 33). The v8 process should
  be killed to free the GPU before relaunching anything.
- **Axis-1 (next launch, ready):** [configs/axis1/tiny.yaml](../configs/axis1/tiny.yaml).
  Carries §15.22 retune (T_pred=12, T_prey=120 at this tier — graduated
  smaller for the smaller pop), §15.23 config-key rename
  (`density_breeding_threshold_*`, etc.), and the v10 mechanism additions
  (mouth widening, age-keyed LR, death-age ring). Tier `tiny` is for
  cheap iteration (~3-5h/1M GPU). If `tiny` validates the retune,
  promote to `small`/`med`/`full`.
- **Axis-2 (ready, holds for axis-1 validation):**
  [configs/axis2/tiny.yaml](../configs/axis2/tiny.yaml). New approach-angle +
  speed obs encoding from §15.24 (replaces the indirect "distance_and_heading"
  variant). Same scaffolds as axis-1 but `reward_type: linear`.
- **Axis-1 + Axis-2 combined (new):**
  [configs/axis12/tiny.yaml](../configs/axis12/tiny.yaml). Both axes' mechanisms
  stacked. New in §15.24.
- **Axis-3 / axis-4:** deferred (see findings.md §15.4). Configs in
  `configs/archive/`.

Each axis has all four tiers available: `tiny.yaml`, `small.yaml`,
`med.yaml`, `full.yaml`. Tier names map to former L1/L2/L3/L4. See
[`configs/README.md`](../configs/README.md) for the layout, and per-axis
READMEs for what each axis does mechanistically.

GCP infra: `evo-reward-gpu` VM (single L4), runs in tmux sessions.

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

## ⭐ Current run framework (post-§15.22-§15.24, 2026-05-03)

The v8 extinction (§15.22) plus the axis-2 obs redesign (§15.24) collapsed several earlier framings. What remains is one coherent setup we're going forward with.

### v10 mechanism additions (carried into all current configs)

- **`predator_mouth_tactile_bins: [0, 1, 17]`** — paper-faithful "small" mouth (60° front arc), wider than v8's single-bin `[0]`.
- **Per-agent age-keyed LR schedule** (`lr_schedule_initial=1e-3` → `lr_schedule_final=3e-4` over 30K steps). Young agents learn faster, mature agents stabilize.
- **Death-age ring buffer** (per-species 256-entry; new SimState fields populated in `process_births_and_deaths_jax`; logged in `progress.json` as `death_age_stats`).

### §15.22 retune (post-extinction fix)

v8's death spiral happened at predator pop 7 with median energy 33 — the breeding sigmoid's cliff edge at that pop was at E=82 (T=10, factor=0.33), so the agents couldn't breed regardless of how strong the rate boost was. **The fix: bump T values so scaffolds engage at the death-spiral operating point, not only after pop ≤ 3.**

Tier-graduated DDB/DDM thresholds:

| Knob | tiny | small | med | full |
|------|------|-------|-----|------|
| `density_breeding_threshold_pred` | 12 | 17 | 20 | 22 |
| `density_breeding_threshold_prey` | 120 | 170 | 200 | 220 |
| `density_metabolism_threshold_pred` | 12 | 17 | 20 | 22 |
| `density_metabolism_threshold_prey` (NEW) | 120 | 170 | 200 | 220 |
| `density_factor_floor` | 0 | 0 | 0 | 0 |
| `breeding_share_alpha` | **0.3** | 0.5 | 0.5 | 0.5 |

Three structural changes from §15.22-§15.24 alongside the T bumps:
- **DDM extended to prey symmetrically** (`density_metabolism_threshold_prey` is new). Was predator-only.
- **`ddb_max_boost` cap removed.** Replaced with the natural floor `factor ≥ kappa_b` — caps `P_birth ≤ 1` without arbitrarily clipping the recovery curve.
- **§15.23 config-key rename** — `ddb_*`/`ddm_*` → `density_*`. Old names still read as fallback for archived configs.

### Configs are organized by axis × tier

Tier names progress in size: **tiny / small / med / full**.

| Tier | World | Caps (prey/pred) | Hidden | Wall-clock for 1M | Use case |
|---|---|---|---|---|---|
| `tiny` | 600² | 200 / 20 | 32–48 | ~3–5 h GPU | sanity-check + iteration; below paper's selection floor |
| `small` | 750² | 300 / 35 | 48–64 | ~6–8 h GPU | overnight runs |
| `med` | 880² | 375 / 40 | 64 | ~12–18 h GPU | production tier; would cite in a paper run |
| `full` | 960² | 450 / 50 | 64 | ~24–36 h GPU | paper-faithful K&D scale |

Three axes:

| Axis | Folder | Mechanism |
|---|---|---|
| Axis 1 | [`configs/axis1/`](../configs/axis1/) | Residual reward MLP (`reward_type: linear_plus_mlp_residual`) |
| Axis 2 | [`configs/axis2/`](../configs/axis2/) | Approach-angle + speed obs (`proximity_encoding: distance_approach_speed`) |
| Axis 12 | [`configs/axis12/`](../configs/axis12/) | Both axes' mechanisms stacked |

Each axis directory contains `tiny.yaml`, `small.yaml`, `med.yaml`, `full.yaml`, and a `README.md` explaining the mechanism. Filenames inside the folder are just the tier; `experiment_name` inside each file encodes axis + mechanism + tier explicitly so GCS run paths stay unambiguous.

### Implementation map (cumulative §15.22-§15.24)

| File | Change |
|---|---|
| `src/jax_lifecycle.py` | DDB cap removed (§15.22); DDM extended to prey (§15.22); new config-key names with fallback (§15.23) |
| `src/jax_sim.py` | `prey_count` passed through for symmetric DDM (§15.22) |
| `src/observations.py` | New `"distance_approach_speed"` encoding: approach-angle math + speed channel (§15.24) |
| `configs/axis1/{tiny,small,med,full}.yaml` | All four tiers — residual reward MLP (§15.24 reorg) |
| `configs/axis2/{tiny,small,med,full}.yaml` | All four tiers — new obs encoding (§15.24) |
| `configs/axis12/{tiny,small,med,full}.yaml` | NEW — combined axes (§15.24) |
| `tests/test_ddb_ddm.py` | Updated for cap removal + new key fallback test (§15.22, §15.23) |

### Deferred features (post-launch)

- **Rate-based α** (alternative to energy-share). Would distribute DDB boost on per-agent catch-rate (predators) or feed-rate (prey). Real fitness signal but requires new SimState fields and checkpoint format bump. Defer until results show energy-share α is the bottleneck.
- **Approach-angle vs body-orientation ablation** (§15.24). Compare `distance_approach_speed` against the legacy `distance_and_heading` to isolate "did the approach-angle math actually help, or just the speed channel?" Optional follow-up.

## The scaffold framing (2026-05-01, retuned in §15.22-§15.24) — DDB+DDM

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

### DDB — Density-Dependent Breeding

When species population is low, breeding becomes easier in two ways:

1. **Threshold drop.** `zeta_b_eff = zeta_b * factor`, where `factor` follows
   a squared-saturation curve `f(N) = max(floor, N²/(N²+T²))`. With the
   §15.22 retune at `tier = med` (T_pred=20, T_prey=200, floor=0):
   N_pred=7 → 0.11, N_pred=15 → 0.36, N_pred=20 → 0.50, N_pred=40 (cap)
   → 0.80. The §15.22 lesson is that the engagement window must reach
   the *operating-point* pop (e.g., pop=7 during a crash) — earlier
   T=10 had the cliff edge at E=82 at pop=7, which was unreachable
   given typical post-crash energies of E≈33.
2. **Rate boost with continuous distribution `breeding_share_alpha ∈ [0, 1]`.**
   Boost = `1 / max(factor, kappa_b)` (the kappa_b floor mathematically
   guarantees `P_birth ≤ 1` — replaces the §15.22-removed `ddb_max_boost`
   cap). The budget is redistributed by `share_i ∝ energy_i^k` where
   `k = α / max(1−α, ε)`. **Total budget preserved at all α**; only the
   *concentration* changes.

   - **α = 0.0**: uniform (every agent gets the same boost). K&D-faithful.
   - **α = 0.3** (default for `tiny` tier): more uniform — small-pop variance protection.
   - **α = 0.5** (default for `small`/`med`/`full`): linear — share ∝ energy. Top agent at energies [800, 100, 100] gets 80% of budget.
   - **α = 1.0**: winner-take-all. Only the top-energy agent breeds.

   Top-agent share at energies [800, 100, 100] for various α:

   | α | top share |
   |---|---|
   | 0.0 | 33% |
   | 0.3 | 67% |
   | 0.5 | 80% |
   | 0.7 | 92% |
   | 0.9 | 99% |

### DDM — Density-Dependent Metabolism (now symmetric across species, §15.22)

`d_b_eff = d_b * factor(N_pred)` for predators and `c_b_eff = c_b * factor(N_prey)`
for prey, both using the same squared-saturation curve. At low own-species
pop, passive decay drops, giving low-energy individuals more time to recover.
Action cost (`alpha_e * action_norm`) stays unscaled. DDM extends the
survival floor; it doesn't make agents immortal.

**Why DDM is needed even with energy-weighted DDB:** in deep LV crashes,
all agents of a species lose energy simultaneously. Without DDM, full-cost
decay finishes them off before any individual can recover, regardless of
how the breeding boost is allocated. v7 (DDM dropped) went extinct at step
88K to validate this lesson empirically.

**Pre-§15.22, DDM applied only to predators.** The current code applies
it symmetrically — same factor function, separate threshold per species.

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

**Configs:** [`configs/axis1/`](../configs/axis1/) — all four tiers (`tiny.yaml`, `small.yaml`, `med.yaml`, `full.yaml`).
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

**Configs:** [`configs/axis2/`](../configs/axis2/) — all four tiers. Default `proximity_encoding` is `"distance_approach_speed"` (the §15.24 redesign — see findings).
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
    --config configs/axis1/tiny.yaml \
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
     --config configs/axis1/tiny.yaml --runtime configs/runtime/gcp_l4.yaml'"
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
| `density_breeding_threshold_pred` (tier-graduated 12/17/20/22) | Where the scaffold curve hits 50% (factor=0.5 at N=T) | **Raise** if extinction happens at moderate pop (the §15.22 lesson — death spiral fired at pop=7 with T=10 because cliff edge stayed at E=82). Lower if scaffolds are over-engaged at healthy peak. |
| `density_breeding_threshold_prey` (tier-graduated 120/170/200/220) | Same as above, prey side | Should track `pred` knob × ~10 to match the equilibrium pop ratio |
| `density_metabolism_threshold_pred/prey` | DDM threshold per species | Same logic; usually mirrors the breeding threshold |
| `density_factor_floor` (currently 0.0) | Minimum factor at extinction | Stay at 0.0 — any positive floor *weakens* the scaffold |
| `breeding_share_alpha` (0.3 at tiny, 0.5 elsewhere) | How concentrated the rate boost is on high-energy agents | Lower (0.3) at tiny pop or if extinction risk during LV crashes; raise (0.7) if selection seems too weak. **Don't go to 0.95+ unless deep bottlenecks are impossible** — see v7 lesson (§15.15) |
| `stability_mechanism` (currently `"ddb_ddm"`) | Which scaffolds are active | Both are needed — see §15.15 |

**Diagnosis playbook:**

- **Extinction at moderate N (pred 5-12):** This is what killed v8 (§15.22). Check `factor` at the death-point — if >0.3, **raise T** (engage the scaffold earlier). Don't expect rate-boost magnitude to save you; the limiter is the breeding sigmoid's cliff edge `zeta_eff/β`.
- **Extinction at deep low N (pred ≤2):** scaffolds aren't strong enough at extremes. With `factor → 0`, rate boost goes to its natural ceiling `1/kappa_b ≈ 1000×`. If still failing, energies have collapsed below survival regardless — DDM may need a higher T too.
- **Mean reward weights drift to zero with high variance:** selection is too weak. With energy-weighted boost this shouldn't happen at healthy populations, but if it does, check that population is actually healthy (factor near 1) — if you're stuck in the engagement zone (N near T), scaffolds are constantly engaged.
- **Population locks at low N never recovers:** check predator energies — if median < `zeta_b_pred / β_b ≈ 250`, the breeding sigmoid is saturated regardless of factor. The lever is T, not max-boost.
- **Bad hunters survive forever:** unlikely with the current symmetric DDM but possible if DDM threshold is set too high; lower if needed.

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
| axis1_residual v8 | med-large T=10 + DDB+DDM + α=0.5 | **Extincted at step ~3.7M.** DDB+DDM with T=10 didn't engage at the operating-point pop (7) — breeding sigmoid cliff at E=82 was unreachable given typical post-crash energies. (§15.22 post-mortem) |
| axis1_residual v9 (config-only diff) | v8 settings + paper-faithful proximity (200→120) + scale-relative food_growth_rate | §15.17; never launched on its own (rolled into v10/post-§15.22 configs) |
| **axis1/tiny (next launch)** | 600² / cap 200/20 / hidden=32 / `density_breeding_threshold_pred=12` / α=0.3 / mouth=[0,1,17] / age-LR / death-age ring / new approach-angle obs encoding (axis2/axis12 only) | Carries §15.22 retune + §15.24 obs redesign. Cheap iteration tier (~3-5h GPU per 1M). |
