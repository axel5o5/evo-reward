# Current state — read this first if you're a new Claude agent

**Last updated:** 2026-05-01 (axis-1 v2 launch — med-large scale + T=10 scaffolds after diversity-loss diagnosis from v1).

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

## The two scaffolds in play

The K&D paper has a known stability problem: with eta=0.5 (paper baseline),
predators tend to over-pressure prey, then crash. To get long enough
co-evolution for ablations to be meaningful, we add two non-paper scaffolds.
Both are explicitly documented as scaffolds, not biological claims, and we
disclose them in findings.

### DDB — Density-Dependent Breeding

When a species' population is low, breeding becomes easier in two ways:

1. **Threshold drop.** `zeta_b_eff = zeta_b * factor`, where `factor` follows
   a squared-saturation curve `f(N) = max(floor, N²/(N²+T²))`. With current
   knobs (pred T=10, prey T=100, floor=0): N=4 → factor 0.14, N=10 → 0.50,
   N=15 → 0.69, N=24 (peak) → 0.85, N=30+ → ~0.90+ (essentially off).
2. **Rate boost (added 2026-05-01).** `kappa_b_eff = kappa_b / max(factor, 1/50)`.
   Lone survivor's breeding rate scales up to 17× normal at N=1 (capped at
   50× for pop near zero), fading to 1× at healthy pop. This is the more
   biologically defensible of the two scaffolds — low-density populations
   really do have higher per-capita reproductive output in many ecosystems.

### DDM — Density-Dependent Metabolism

Predator passive decay rate scales by the same factor: `d_b_eff = d_b * factor`.
At N=1 a lone predator decays at ~6% normal rate, effectively immortal from
passive decay alone — but action cost (`alpha_e * action_norm`) is unscaled,
so they still pay for moving. Selection pressure on motor learning intact.

### Why both, and why now strong

Earlier weaker versions (floor=0.3, no rate boost) extended runs from
~80K steps (no scaffolds) to ~1.35M steps (DDB+DDM weak), but predators
still went extinct via a "trophic-collapse-via-herd" pattern: prey evolved
strong herd-seeking (`prey_w_prey=+7.69`) without evolving fear
(`prey_w_pred≈0`), so once predators dropped below ~3 they couldn't keep up
with the herd's density-dependent escape and collapsed.

The strengthened scaffolds (floor=0.0, rate boost up to 50×, tighter
thresholds) make extinction at N≤3 effectively impossible: a lone survivor
with any positive energy breeds in ~20-60 steps, restoring genetic diversity
before they can drift into geometric "can't find prey" failure. At healthy
populations the scaffolds are nearly inactive — selection pressure on
hunting and herding weights remains intact in the regime that matters.

See [docs/findings.md](findings.md) §15.11 for the calibration math.

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
