# Scope reset — what's changed and why (2026-04-29)

> **TL;DR.** We've narrowed scope to two axes (1 and 2), redesigned both around their actual failure modes from earlier runs, added a single "stability scaffold" mechanism to prevent extinctions, and shrunk the world geometry to iterate faster. Three experiments queue up over the next ~3 days. Axis-3 and axis-4 are deferred indefinitely.
>
> This document is the handoff for anyone joining the project mid-stream. For deeper rationale see `docs/findings.md` §11–§15.

---

## What changed in one paragraph

After several long runs ending in extinction (axis-1 v1/v2 at ~300-380K, axis-2 v1 at ~1.10M from trophic collapse), we diagnosed two distinct failure modes and redesigned both axes to sidestep them. We also added a single config-gated stability mechanism (Density-Dependent Breeding) that scales the breeding-energy threshold down when species count is low — a soft "rescue" effect at deep bottlenecks. The K&D paper runs to 10M without extinction; we've never reached that. DDB is a pragmatic scaffold to enable long-horizon experiments. We'll test removing it later if stability emerges naturally. We're also running on a smaller world geometry (600 vs 960) with smaller populations to iterate ~1.5-2× faster without sacrificing run length (2M steps preserved).

---

## The four-axis plan, before vs after

| Axis | What it tests | Status |
|---|---|---|
| **Axis 1** — reward genome capacity | Linear (K&D) → MLP-augmented reward | **Redesigned** (residual). In flight after baseline validation. |
| **Axis 2** — observation richness | Position-only → kinematic information about other agents | **Redesigned** (bin-aligned). In flight after axis 1. |
| Axis 3 — temporal reward window | Instant reward → reward over rolling k-step window | **Deferred** |
| Axis 4 — policy memory | Feedforward MLP policy → LSTM policy | **Deferred** |

The combo (axis 1 + axis 2) runs after both individual results land if results warrant it.

---

## Why we deferred axis 3 and axis 4

- **Compute and time.** Each long run is ~$10–20 and ~12-24h. Running all 4 axes plus combinations at multiple seeds isn't feasible in our window.
- **Same bootstrap problem.** Axis 3's 945-parameter temporal MLP faces the same "evolution can't bootstrap a randomly-initialized large reward network before agents starve" problem as the original axis 1. Axis 1's redesign (residual reward) solves this for axis 1 specifically; extending it to axis 3 is its own design exercise.
- **Axis 4 isn't fully wired.** The LSTM PPO path was added but never exercised end-to-end. Would need integration time.
- **Cleaner story.** Two axes with definitive results > four axes with inconclusive results.

---

## Axis 1 redesign — residual reward genome

### The problem we're sidestepping

Original axis 1 replaced the K&D linear reward function with a randomly-initialized 121-parameter MLP. At t=0, this MLP produces essentially random rewards — predators have no useful behavioral gradient, they don't catch prey, they starve. By the time evolution finds something useful, predator population has already collapsed.

We tried three mutation rates (0.01, 0.08, 0.03 was queued but cut). All extincted at ~300-380K, well before the linear baseline reaches stability (which it does around step 700K+).

### The new design

Instead of replacing the linear reward, we *augment* it:

```
reward(stimuli) = sum(coefs · linear_weights · stimuli)  +  MLP_residual(stimuli)
                  └─────── K&D linear baseline ───────┘    └ zero-init perturbation ┘
```

- **Linear part:** 4 weights, K&D init (N(0, 0.1)) and mutation. Same as the proven-stable linear baseline.
- **Residual MLP:** input(4) → Dense(4, tanh) → Dense(1, linear) = **25 parameters, zero-initialized**. Mutates with Student's t, scale 0.03, clip 5.0.

**At t=0 the residual MLP outputs zero**, so the system starts as exact K&D-faithful linear reward. The residual grows via mutation only if it improves fitness.

### What axis 1 will tell us

- **If the residual stays near zero:** linear reward is sufficient on this substrate; nonlinear reward structure isn't needed. (Negative result for axis 1, but a clean one.)
- **If the residual weights grow:** evolution found nonlinear interactions that improve fitness on top of the linear baseline. (Positive result — quantifiable nonlinear structure.)

Either is publishable.

---

## Axis 2 redesign — bin-aligned heading encoding

### The problem we're sidestepping

Original axis 2 added a "social slot" to the observation vector: the heading + speed of the 5 nearest *conspecifics* (same species). Two issues surfaced:

1. **Kin-only.** Neither species saw the other species' kinematics. Predators couldn't anticipate prey trajectories; prey couldn't anticipate predator charges. Whatever effect we observed was indirect (via flocking → spatial concentration → predator advantage).

2. **Trophic collapse.** Seed 0 hit 1M with strong herd-seeking weights (`prey_w_prey = +4.51`) and looked stable. Extension to 2M revealed predator pop surging past normal cycle peaks (28 vs typical 17), driving prey from 450 down to 103, then predators starved. New failure mode: "predators too strong with no brake."

3. **Binding burden.** The social slot tells the policy "the nearest predator is heading this way," but the proximity sensors return distances per angular bin, unlabeled. The policy MLP has to *learn the binding* between social slot and proximity bin — a multiplicative interaction that's slow to learn within a 1M-step training budget.

### The new design

Instead of a separate slot, extend the proximity-sensor channels themselves to include heading. Per angular bin, each species channel reports `[distance, sin(rel_heading), cos(rel_heading)]` of the closest agent of that species in that bin (relative to the observer's own heading).

| Per-bin layout (8 channels) |
|---|
| `[prey_dist, prey_sin_rel, prey_cos_rel, pred_dist, pred_sin_rel, pred_cos_rel, food_dist, wall_dist]` |

obs_dim becomes **333** (vs 205 baseline). Three design choices, each addressing a real concern:

- **Bin-aligned, not separate slot:** kinematics arrive *attached to the same bin* as position info. No binding needed.
- **sin/cos encoding:** a single-scalar relative heading aliases "no agent in bin" with "agent facing same direction as me" (both = 0). sin/cos puts no-agent at (0, 0) — geometrically impossible for any real angle, so magnitude becomes an unambiguous presence signal.
- **Heading-only, no speed:** energy cost is quadratic in action norm so agents cruise at moderate speeds — speed encodes nuance not category. Heading carries the categorical signal ("predator facing me" vs "facing away").

Behind config flag `proximity_encoding: "distance_only" | "distance_and_heading"`. K&D-faithful default unchanged.

### What axis 2 will tell us

- **If trophic collapse is prevented:** kinematics-co-located-with-position lets prey preempt predator surges. Cross-species perception is the lever.
- **If trophic collapse still happens:** the failure isn't about perceptual access. Something deeper (e.g., reward genome insufficiency) is needed.

---

## DDB — the stability scaffold

### Why we added it

Every long run we've done extincts at a deep bottleneck. The pattern:
1. Population drops to N=1-3 (predators) or low-equivalent (prey).
2. Survivors don't breed back fast enough.
3. Population dies out.

K&D's runs go to 10M without this. There's an undiagnosed difference between our setup and theirs — but rather than continue chasing the missing fix, we add a soft scaffold so axis experiments can run to completion. We'll test removing it later if stability emerges naturally.

### The mechanism

Squared-saturation factor on the breeding-energy threshold:

```
f(N) = max(floor, N² / (N² + threshold²))
effective_zeta_breed = zeta_breed × f(N_species)
```

| | threshold | floor | At N=threshold | At 2·threshold | At 4·threshold |
|---|---|---|---|---|---|
| Predator | 5 | 0.3 | f=0.5 | f=0.80 | f=0.94 |
| Prey | 30 | 0.3 | f=0.5 | f=0.80 | f=0.94 |

Lower zeta means breeding requires less energy → easier to breed at low population → faster bottleneck recovery.

**At healthy populations** (typical predator N=15-25 in cycles, prey N=150-450), f≈0.9-0.99 — basically off. **At deep bottleneck** (N=1-3 predator, N<30 prey), f drops toward the floor — strong rescue effect. Smooth in between, no kink.

### Why this is biologically OK

Real Allee effects often work the opposite way (rare animals struggle to find mates), but DDB is empirically grounded in fisheries and endangered-species literature. We're modeling the upside ("rare individuals face less intraspecific competition") without the downside (mate-finding difficulty), which is a simplification.

We're being transparent about this: it's a pragmatic deviation from K&D, not a claim about real biology. Default OFF; turn ON for our axis runs; document with vs without later.

---

## Small-scale baseline — middle-ground after first attempt failed

**First attempt (small-scale 600²) failed.** `baseline_smol_ddb` extincted predators at step ~80K from trophic over-pressure: density was 2.5× original, catches/step were ~5× normal in the first 20K steps, prey crashed faster than they could breed back, predators starved before DDB could rescue them. (DDB lowers the breeding threshold but can't put energy in the bank.)

**Second attempt** (`baseline_med_ddb`) walks back to a less aggressive middle ground:

| Parameter | Original | Failed small | **Middle** (this attempt) |
|---|---|---|---|
| `world_size` | 960 | 600 | **800** (70% area, 1.4× density) |
| `prey_cap` / `predator_cap` | 450 / 50 | 200 / 25 | **300 / 30** (cap ratio 10:1, prey-favored) |
| `prey_initial` / `predator_initial` | 150 / 10 | 75 / 5 | **100 / 7** |
| `food_max` | 600 | 300 | **450** |
| `ddb_pred_threshold` | n/a | 5 | **8** ← bumped (fires earlier in decline) |
| `ddb_prey_threshold` | n/a | 30 | **30** (unchanged) |

Estimated ~50 sps (~2× original speed, 2M runs ≈ 11h). If middle-ground also collapses, fallback is original 960² geometry + DDB (slower but proven) or adding DDM (density-dependent metabolism) on top of DDB.

**Run length kept at 2M.** A user-explicit preference: 1 long run over 4 short ones — for scientific signal at the long-horizon timescale where K&D's results live.

---

## What's running now and what's queued

| # | Run | Purpose | Status |
|---|---|---|---|
| 0a | `baseline_smol_ddb` (linear, 2M) | First validation attempt (600² world) | **Failed** — predators extincted at step ~80K from trophic over-pressure. See findings §15.6 |
| 0b | `baseline_med_ddb` (linear, 2M, middle-ground 800²) | Second validation attempt with less aggressive scale-down + DDB threshold 8 | **Launching** |
| 1 | `axis1_residual_med` (residual reward, 2M) | Does evolution grow useful nonlinear reward structure? | Queued behind 0b validation |
| 2 | `axis2_aligned_med` (bin-aligned obs, 2M) | Does cross-species kinematic perception prevent trophic collapse? | Queued |
| 3 | `axis1+2_combo` (residual + bin-aligned, 2M) | Do the two axes compose, conflict, or amplify? | Queued, gated on results of 1 and 2 |

Each run is ~11h on a single L4 GPU at ~$10. Axis configs (`axis1_residual_med`, `axis2_aligned_med`) will inherit the middle-ground geometry from `baseline_med_ddb` once it validates. Total ~2-3 days for all four runs once baseline lands.

---

## Where to find what

| Resource | Location |
|---|---|
| Full chronological run-by-run log | `docs/experiments-log.md` |
| Cross-cutting findings + design decisions | `docs/findings.md` (§15 is the strategic-reset synthesis) |
| Code-level deviations from emevo | `docs/emevo-diff.md` |
| Configs for each experiment | `configs/baseline_smol_ddb.yaml`, `configs/axis1_residual.yaml`, `configs/axis2_aligned_smol.yaml` |
| Live VM | `evo-reward-gpu` in `us-west1-a` (project `evo-reward`) — tmux sessions per run, logs in `~/<runtag>.log` |
| Replays | `gs://evo-reward-replays-public/<exp>/seed_<N>/<run_tag>/` |
| Checkpoints | `gs://evo-reward-ckpts/results/<exp>/seed_<N>/<run_tag>/checkpoints/` |

---

## What we'll learn even from "negative" outcomes

A common failure mode in this kind of work is treating extinction as a no-result. We've reframed:

- **Axis 1: residual stays near zero** → "linear reward is sufficient on this substrate; the nonlinear capacity isn't useful here." Concrete claim about the limits of reward-genome richness.
- **Axis 1: residual grows** → "evolution found nonlinear reward structure that helps." Quantifiable result with measurable magnitude and structure.
- **Axis 2: trophic collapse despite bin-aligned obs** → "the issue isn't perceptual access; something deeper is needed (richer reward, different population dynamics, etc.)." Localizes the next investigation.
- **Axis 2: stable** → "cross-species kinematic perception is the lever for stability." Strong positive claim.

Either way: we get an interpretable result in 12-24h instead of running 4 days and finding "extinct again, unclear why."
