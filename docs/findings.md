# Findings — What We've Learned About the Kanagawa-Doya Replication

Cross-cutting insights from Phase 1a parameter exploration. Pairs with:
- [`experiments-log.md`](experiments-log.md) — chronological run-by-run table
- [`params-playbook.md`](params-playbook.md) — parameter tuning reference
- [`emevo-diff.md`](emevo-diff.md) — code-level deviations from upstream

Last updated: 2026-04-27.

---

## 1. Our code reproduces emevo's dynamics on seed 0

After 30+ runs across D18-D31 deviation fixes, we built the emevo reproduction harness ([`scripts/emevo_repro/`](../scripts/emevo_repro/)) and ran emevo's own `cf_predator.py` on the gecco2026 branch (commit `f87a880`) at matched seed 0.

**Result:** emevo extincts on seed 0 too. Both with the rectangular config we initially used and the correct square config (paper figures match the square variant — see §2).

| Run | Step | Pred peak | Extinction step |
|---|---|---|---|
| Our d31d (rect, eta=0.6) | 80K | 38 @ 20K | ~80K |
| Emevo rect (eta=0.6) | 102K | 45 @ 30K | ~102K |
| Emevo square (eta=0.6) | 112K | 42 @ 40K | ~112K |

Same overshoot-then-crash shape across all three. Conclusion: our code is not materially broken at the dynamics level. We match emevo's seed-0 behavior within noise.

## 2. Paper uses 960×960 square geometry, not 1200×600 rectangle

Emevo's `gecco2026` branch contains both `config/env/20251001-predator-default.toml` (1200×600 rect) and `config/env/20251122-predator-square.toml` (960×960 square). The rectangular config is **explicitly deprecated** — commit `fd09012` ("predator 1200 (unused now)") tags it as no longer in use. Paper figures match the square geometry.

The square has 28% more area at the same agent/food counts → lower density → fewer forced encounters. Empirically this delays the cycle-1 crash by ~10K-15K steps but does not prevent extinction on seed 0.

## 3. Paper's pred avg = 23 is a survivor-biased cross-seed average

Paper Table 1 reports default-condition `pred_avg = 23`, suggesting all 5 default-condition seeds survived. Paper also openly reports **5/13 survival for n=0.6 food-regen** and **1/6 survival for the large-mouth variant** — extinction is a known feature in their own data. They drop extinct runs from the average.

Taking emevo's own seed-0 extinction together with paper's openly partial seed survival, the most likely interpretation: **paper-default parameters are a knife-edge regime where ~50%+ of seeds extinct, and the published number is the average over the survivors**.

This means:
- Our seed-0 extinction at eta=0.6 isn't necessarily a code bug
- Multi-seed runs are needed to make any quantitative comparison
- "Reproducing the paper" should mean reproducing the qualitative dynamics + cycle structure, not matching pred=23 on a single seed

## 4. The fear-vs-extinction tension

Sweeping `predator_eta` (digestive rate) reveals a fundamental tradeoff:

| eta | Behavior | Cycles | Extinct? | Fear evolution |
|---|---|---|---|---|
| 0.45 | Stable but fragile | 1 (seen to 150K) | No (pred=5) | None (-0.07) |
| 0.50 | Sustained 3 cycles | 3 (seen to 1M) | **Yes ~670K** | None (drift to +1.5) |
| 0.55 | 4 cycles, runaway peak | 4 (seen to 1M) | **Yes ~720K** | **Strong: -5 by cycle 3, -16 post-ext** |
| 0.60 (paper) | Immediate overshoot | 0-1 | **Yes ~80K** | Brief partial (-0.5) before death |

**Pattern:** fear evolves when overshoots create strong predation pressure. The same overshoots that drive fear also drive eventual extinction. Lower eta = stable but no selection for fear. Higher eta = fear evolves but population eventually collapses.

This may be a fundamental dynamics property, OR something stabilizing in paper's exact parameters that we still haven't matched. Open candidates:
- Per-catch geometry (we did D28 shared credit; possibly subtle differences remain)
- PPO update frequency / batch composition
- Reward extractor normalization

## 5. The runaway mechanism

In runs with multiple complete cycles (0.50 to 1M, 0.55 to 1M), each cycle's predator peak grows:
- 0.50: peaks 24 → 25 → 18 (then extinct at trough 4)
- 0.55: peaks 21 → 30 → 38 → 50 (cap, then crash, then extinct)

The driver is co-evolution. Across cycles:
- `pred_w_prey` (chase drive) evolves from -0.4 toward +1.0
- `pred_w_pred` (social/coordinate) evolves from -0.1 toward +0.4
- More skilled predators extract more energy per encounter → more cross E>240 breeding threshold → bigger next-cycle peak

When prey-side fear (`prey_w_pred`) doesn't co-evolve fast enough — which happens until cycle 3 in our 0.55 run — the runaway proceeds unchecked. Fear is the natural counter-force, but it requires sustained predation pressure to evolve, and that same pressure causes the cycles to grow until the system breaks.

## 6. Stable parameter regimes that don't show fear

Both eta=0.45 and the early portion of eta=0.50 (steps 0-500K) maintain populations *without* evolving fear. Prey weights drift on other axes — `prey_w_eat` to +3.67, `prey_w_prey` (flocking) to +4.13 — but `prey_w_pred` stays near zero with high variance.

Hypothesis: when predator pressure is below a threshold, fearful and fearless prey have similar survival rates. Selection on the fear axis is weak. Other strong gradients (food drive, anti-crowding) absorb the mutation budget. Fear stays at the prior.

Implication: getting *both* stable populations and meaningful fear evolution may require a parameter regime we haven't found yet. Or the timescale needed is much longer than 1M steps. Or the paper's reported fear comes from a phase of pre-extinction selection followed by survival on a different seed where the lineage retained the fear weights.

## 7. eta=0.50 is the cleanest baseline for axis experiments — superseded by §10

[Original recommendation from 2026-04-26 — preserved for context. Superseded by `mouth_smol` finding in §10 below.]

For Phase 2 axis comparisons we need a baseline that survives long enough to express the variant's effect. Choices were:
- 0.45: too fragile for cross-cycle comparisons
- 0.50: survives ~670K — long enough for 2 cycles
- 0.55: also dies, at slightly larger overshoots
- 0.60 (paper): dies in cycle 1

Use eta=0.50 for axis runs accepting it will still extinct beyond ~670K, targeting 500K runs.

**This recommendation is now stale** — see §10. `mouth_smol` survives 1M with fear evolution, making it the better axis substrate.

## 8. All four axis configs are mechanically working

Smoke-tested at 20K steps each (2026-04-26):
- `axis1_mlp_reward` — MLP reward genome
- `axis2_social_obs` — neighbor heading+speed in obs vector
- `axis3_temporal_reward` — k-step temporal context for reward
- `axis4_lstm_policy` — LSTM policy network

All compile clean, run at ~30 sps, populations behave normally. Slight per-axis differences at step 20K (axis3 had pred=30 vs ~24 elsewhere — possibly the temporal reward giving predators an advantage at chaining catches, but could be noise within one seed).

## 9. axis2 social_obs alone doesn't fix extinction

Real run at eta=0.50 + `social_obs: position_heading_velocity` (axis2_real_500k, seed 0). Predator extincted at step ~100K — same overshoot-then-crash shape as the linear baseline. Killed at 310K with prey saturated to cap=450 in steady-state predator-extinct mode.

Conclusion: switching prey-side observation to include neighbor heading/velocity does not, by itself, change the extinction dynamics on this substrate. The bottleneck is the substrate, not the observation. Confirms that *all* axis comparisons need a stable substrate first.

## 10. ⭐ `predator_mouth_range = [0]` solves the substrate problem

**The breakthrough run.** `sweep_mouth_smol_1M`, seed 0 (2026-04-27): eta=0.50 + mouth=`[0]` (single 20° tactile bin instead of 60° three-bin arc). Other parameters identical to baseline.

| Run | End step | Outcome | Fear (`prey_w_pred`) |
|---|---|---|---|
| eta=0.50 baseline | ~670K | extinct | drift to +1.5 (no fear) |
| eta=0.55 | ~720K | extinct | -16 sustained (post-extinction) |
| **mouth_smol** | **1M** | **alive** | **-1.97 ± 9.75 sustained** |

Final state at step 1M: prey=371, pred=16, both species alive and oscillating in stable LV cycles (prey 345-450, pred 16-27 across the last 100K steps). Fear evolved to a sustained moderate value, not the runaway -16 we saw post-extinction in 0.55.

**Mechanism (proposed):** Smaller mouth arc → fewer catches per encounter (peak catch counts ~10% lower throughout). Lower per-cycle predator energy windfall → fewer predators cross the E>240 breeding threshold → smaller next-cycle peak. The runaway-breeding loop is broken without zeroing out predation pressure. Predation stays moderate and *sustained*, which is exactly the regime where fear can co-evolve gradually rather than in a desperate post-cap crash like 0.55.

This matches the paper's explicit small-mouth condition (paper Sec. 3.3: small/medium/large mouth variants). Paper reports 1/6 large-mouth survival vs 5/5 medium-default; we now have 1/1 small-mouth survival on the eta=0.50 substrate (n=1 — multi-seed needed).

**Implications:**
- All four axis comparisons should run on `sweep_mouth_smol` substrate, not eta=0.50 plain.
- The hard part was building enough infrastructure to recognize this — the parameter itself is one line of YAML.
- Open: does this hold across seeds? Does the oscillation stay stable past 1M (5M run)?

## 11. Axis 1 (MLP reward) extincts at step 380K on the substrate where linear survives

**Run:** `axis1_mlp_reward` seed 0, 1M steps, mouth_smol substrate (2026-04-28). The genome architecture changes from 4-weight linear to 121-param MLP (input(4)→hidden(8)→hidden(8)→1). All other parameters match `sweep_mouth_smol_1M`, which survives 1M with fear -1.97.

**Trajectory:**

| Step | Prey | Pred | Note |
|---|---|---|---|
| 140K | 364 | 1 | First bottleneck — one survivor with E=276 |
| 250K | 450 | 24 | Survivor's clones rebound |
| 300K | 281 | 30 | Predators actively hunting (LV downturn) |
| 360K | 250 | 5 | Second bottleneck |
| **380K** | 343 | **0** | **Extinction** |
| 1M | 450 | 0 | Prey-only ecosystem rest of run |

**Hypothesis — mutation kick is too weak.** With `mlp_mutation_scale = 0.01` on 121 params, expected per-generation L2 perturbation is √121 · 0.01 = **0.11**. Linear genome's equivalent is √4 · 0.4 = **0.80** — **7× larger**. At population bottlenecks (1–3 predators), the only path back to genetic diversity is mutation noise creating offspring divergence. With MLP's kick 7× weaker, sibling offspring are essentially identical to the founder, selection has zero variance to climb, and any environmental shift kills the lineage uniformly. That's exactly what happened at 380K.

Linear baseline survives the same bottleneck because its kick produces actually-different siblings within a few generations.

**Counter-argument we accept:** the original 0.01 was conservative on purpose — 121-param MLPs encode more complex stimulus→reward maps, and per-param noise can flip an inflection point catastrophically. So you can't *just* match linear's kick without risking that every offspring's reward function is structurally different from parent's. The right answer balances diversity-recovery against structural stability.

**Test #1 result (mut=0.08, 2026-04-28):** the bigger kick **didn't rescue the run — it made it worse, in a different way.**

| Step | v1 (mut=0.01) | v2 (mut=0.08) |
|---|---|---|
| 120K | pred=7 | pred=8 (steady) |
| 140K | pred=**1** (1st bottleneck) | pred=8 (no bottleneck) |
| 250K | pred=24 (recovered) | pred=9 (steady, lower peak) |
| 270K | — | pred=3 (collapsing) |
| 290K | — | pred=1 |
| **300K** | pred=30 | **pred=0 (extinct)** |
| 380K | pred=0 (extinct) | extinct |

v2 *did* maintain real genetic diversity — `pred_w eat = -0.19±0.06` until 260K (vs v1's perpetual ±0.00 post-bottleneck). Per-capita catch rate matched v1. But the predator population peaked at ~15 (vs v1's 30) and **crashed at the first downturn instead of recovering**.

**Revised reading:** higher mutation produces more diverse predators, but each predator's reward function differs noticeably from its parent. Many offspring inherit *bad* reward functions (e.g., reward going away from prey) and starve. Net effect: smaller stable population → less buffer against the LV trough → first downturn = extinction.

The original counter-argument *was* the dominant effect, not the diversity-recovery argument we bet on. **Both 0.01 and 0.08 are wrong, in opposite directions:**
- **0.01:** too clonal → diversity-recovery starvation post-bottleneck
- **0.08:** too noisy → many non-viable offspring, smaller stable pop, dies pre-bottleneck

**Possible next moves (not yet tested):**
1. Try **0.03** — split the difference. If still extincts, the failure isn't a single-knob mutation problem.
2. Try **smaller MLP** (hidden=4 → ~49 params instead of 121) — fewer DOF means each per-param kick rotates the function less.
3. Inspect v2 replays in the AgentInspector to see what kinds of reward landscapes the surviving predators evolved — may reveal *why* viability is so low.
4. Accept that MLP genome is hard to bootstrap on this substrate and move to other axes (axis 2, axis 4) using the proven linear genome.

For now, paused. The mut=0.03 test is queued as future work.

**Same concern still applies to axis 3 (temporal).** `temporal_mutation_scale = 0.005` on 945 params → aggregate kick 0.15. If we revisit axis 3, expect similar instability.

## 12. Axis 2 (richer social obs) — herd-not-fear at 1M, then trophic collapse at 1.1M

**Run:** `axis2_mouth_smol_1M` seed 0 (2026-04-29). Identical to `sweep_mouth_smol_1M` except `social_obs: "position_heading_velocity"` (prey/pred see neighbor heading and velocity, not just position). Linear genome — no mutation-tuning concern.

**At step 1M, both species alive (prey=450, pred=10) and weights looked fundamentally different from the linear baseline:**

| Weight | Linear baseline (pos only) | Axis 2 (pos+heading+velocity) |
|---|---|---|
| `prey_w_prey` (herd) | ≈ 0 | **+4.51 ± 5.23** |
| `prey_w_pred` (fear) | **−1.97 ± 9.75** | +0.17 ± 3.85 |
| `pred_w_prey` (chase) | +1.5-ish | +1.84 ± 0.72 |

We initially read this as "richer obs lets prey substitute herd-seeking for fear" — a different but-equally-valid stable solution. **That reading was wrong, or at least premature.**

**Extension to 2M revealed trophic collapse:**

| Step | Prey | Pred | Δcatch |
|---|---|---|---|
| 1.00M | 450 | 10 | 63 |
| 1.04M | 319 | **28** (peak) | **203** (peak) |
| 1.06M | 147 | 19 | 91 |
| 1.08M | 103 | 5 | 21 |
| **1.10M** | 122 | **0** | 4 |

Predator pop surged to 28, drove prey from 450 down to 103, then starved when prey couldn't recover fast enough. **A new failure mode** — prior extinctions in this codebase have all been "predators too weak"; this is "predators *too strong* with no brake."

**Diagnosis — the local-optimum trap.** Prey have two avenues to handle predators:
1. **Reactive (policy-level):** "I see a predator, I dodge." Handled by the policy network reading the obs vector.
2. **Strategic (reward-level):** "Predator-dense regions are bad even when no immediate danger." Encoded in `prey_w_pred` — fear.

In the linear baseline (position only), reactive dodging is impoverished — prey don't know predator heading or velocity. The only avenue available is the strategic one → fear evolves. In axis 2, richer obs makes reactive dodging much better, so evolution finds that local solution first. The herd weight reinforces it (group dodging > solo). Fear never builds up enough gradient because reactive avoidance is "good enough" *at low predator density*. When the natural LV cycle produces a predator surge that overwhelms reactive avoidance, prey have no strategic backstop — they crash, predators overshoot, both die.

**This isn't necessarily about "fear is required."** It's about whether prey adaptations actually *limit* predator success. In the linear baseline, fear → prey avoid → predator catch rate drops naturally → predator pop self-limits. In axis 2, the herd weight (+4.51) actually *concentrates* prey spatially, which arguably *helps* predators find them. So prey "adapted" without applying real evolutionary pressure on predators. Predators kept getting better unchecked → over-specialized → collapse.

**Important caveat — what the v1 social_obs actually does.** Reading the implementation ([src/observations.py:279-320](src/observations.py#L279-L320)): the upgrade adds heading + speed of the **5 nearest *conspecifics* (same-species)**, not other-species. So in axis 2 v1, neither species sees the other's kinematics directly — both species get info only about their own kind. Predator over-evolution therefore is **not a direct perceptual advantage**; it's mediated through prey behavior change.

The likely chain: prey gets kin kinematics → evolves flocking (`prey_w_prey = +4.51`) → flocking concentrates prey spatially → predators (still seeing prey via baseline proximity sensors) find dense clusters easier → predators *also* upgraded see other-predators → evolve anti-herd (`pred_w_pred = −0.67`) → spread out → cover more ground. Net effect: clustered prey + spread predators = predators always have prey nearby.

**This reframes what axis 2 actually tests.** v1 tested "kin-only social obs" — does seeing flockmates' motion change behavior? Answer: yes, strongly (huge herd weight, trophic collapse). But it didn't test the more interesting questions: what does *cross-species* perception do, and what's the role of perceptual asymmetry?

## 13. Axis 2 redesign — bin-aligned heading channels (the converged design)

**The first redesign attempt (n_kin / n_other social slots) is implemented but not the actual experiment we want to run.** Reasoning through the design surfaced three problems with social-slot encoding:

1. **Binding problem.** The social slot tells the policy "the nearest predator is heading this way at this speed," but the proximity sensors return distances per angular bin, unlabeled. The policy has to *learn the binding* — "the kinematic info pertains to whichever proximity bin has the smallest predator-distance." That's a multiplicative interaction the 64×2 policy MLP can encode in principle, but it's real work to learn within a 1M-step budget. Adding observations that require the policy to learn complex bindings is exactly the wrong move when our problem is "evolution is too slow already."
2. **Speed is probably not worth a dim.** Energy cost is quadratic in action norm, so agents cruise at moderate speeds most of the time. Speed encodes nuance, but heading carries the categorical signal ("predator facing me" vs "facing away").
3. **Heading was absolute world-frame.** The policy has to subtract its own heading at runtime to get the relative angle. Doable but adds noise.

**The bin-aligned design solves all three.** Instead of a separate "social slot," we extend the proximity-sensor channels themselves to include heading. Per angular bin, each species channel gets `[distance, sin(rel_heading), cos(rel_heading)]` instead of just `[distance]`. Kinematics arrive *attached to the same bin* as position info — no binding needed.

**Encoding decisions:**

- **Bin-aligned**, not separate slot. Eliminates the binding problem.
- **Heading-only**, no speed. Saves dims; speed is a refinement we can add later if heading-only proves insufficient.
- **Relative to observer** (`neighbor_heading − observer_heading`). Saves the policy from learning a transformation.
- **sin/cos encoding** of the relative heading. Fixes an aliasing issue: a single-scalar `rel_heading=0` would collide with "no agent in this bin" because zero is a meaningful angle (facing same direction as observer). With sin/cos, `(0, 0)` is geometrically impossible for any real angle, so the policy can read magnitude `√(sin² + cos²)` as an unambiguous presence signal.

**Per-bin layout (8 channels):**
```
[prey_dist, prey_sin_rel, prey_cos_rel,
 pred_dist, pred_sin_rel, pred_cos_rel,
 food_dist, wall_dist]
```
32 bins × 8 channels = 256 dims for the proximity block.
**obs_dim = 256 + 72 + 5 = 333** (vs baseline 205, +63%).

For bins with no agent of a given species: `(sin, cos) = (0, 0)`. The policy can learn to ignore heading info when distance is large (or magnitude is zero) — both signals reinforce the "no agent" interpretation, so the gating is easy.

**Behind a config flag** (`proximity_encoding: "distance_only" | "distance_and_heading"`) so the K&D-faithful baseline still reproduces unchanged.

**The experiment ladder (after axis-1 v3 finishes):**

| Run | Encoding | obs_dim | What it tests |
|---|---|---|---|
| `axis2-aligned` (single seed, 2M) | bin-aligned, heading-only, sin/cos, relative | 333 | Does bin-aligned cross-species perception prevent trophic collapse? |
| (if interesting) seed 1, seed 2 | same | 333 | Reproducibility |
| (if collapse anyway) bin-aligned + speed | extended | 365 | Does speed buy us anything? |
| (if collapse anyway) bin-aligned + initial fear bias | same | 333 | Does perception + biased reward genome stabilize? |

**The n_kin / n_other implementation is preserved** for backward compatibility with axis-2 v1 and as a fallback if we later want a kin-only experiment. It's no longer the planned next experiment.

## 14. Open questions worth follow-up (superseded by §15)

These were the open questions before the 2026-04-29 strategic reset. Some are subsumed; some are deferred. See §15 for the current plan.

1. ~~`axis2-aligned` on mouth_smol~~ — **subsumed by §15.** Will run as `axis2_aligned_smol` on the new baseline (small-scale + DDB).
2. ~~Axis 1 retry (mut=0.03 or smaller MLP)~~ — **abandoned.** Both axis-1 attempts at random-init full-MLP failed (v1 too clonal at mut=0.01, v2 too noisy at mut=0.08, v3 was queued at mut=0.03 but killed before completion since results would just confirm "intermediate is also bad"). Replaced by the residual reward design in §15.
3. **Multi-seed at mouth_smol linear** — still open as a separate validation question. Seed 1 reached step 730K with fear -3.88 before we paused; seed 2/3 untouched. Lower priority than the axis runs now.
4. **mouth_smol past 1M** — answered for axis-2 v1: extincted at 1.10M from trophic collapse. Open for the linear baseline alone.
5. **`zeta_b_pred = 150`** — superseded by DDB (§15), which dynamically adjusts the breeding gate. The static-zeta question is no longer the right framing.
6. **Initial fear bias** — still open as a strong intervention. Lower priority unless DDB doesn't deliver stability.
7. **Audit emevo's catch geometry once more** — still open, lower priority. The fact that K&D goes to 10M and we don't is suggestive of an undiagnosed difference.

## 15. ⭐ Strategic reset (2026-04-29) — narrowed scope, redesigned axes, stability scaffold

The plan from §13 (n_kin/n_other social slots) and §14's axis-1 mut-tuning ladder are both abandoned. After multiple long runs ending in extinction, we reset scope to two axes with redesigned mechanics + a stability scaffold, plus a scaled-down baseline to iterate faster.

### 15.1 The three changes

**(A) Axis-1 → residual reward genome.** Instead of replacing the K&D linear reward with a randomly-initialized 121-param MLP (which kept extincting before evolution found anything useful), the new design *augments* the linear baseline with a small zero-init MLP perturbation:

```
r(stimuli) = sum(coefs · linear_weights · stimuli)  +  MLP_residual(stimuli)
```

- Linear part: 4 weights, K&D init (N(0, 0.1)) and mutation (Student's t, scale 0.4) — proven-stable.
- Residual MLP: input(4) → Dense(4, tanh) → Dense(1) = **25 params**, **zero-initialized**, mutation scale 0.03, weight clip 5.0.

**At t=0 the residual MLP outputs zero**, so the system starts as exact K&D-faithful linear reward. Evolution adds nonlinear structure if and only if it improves fitness. No bootstrap failure mode.

Two interpretations of the result are both informative:
- Residual stays near zero → "linear is sufficient; MLP capacity isn't needed here."
- Residual weights grow → "evolution found nonlinear structure that helps."

Implementation: `reward_type: "linear_plus_mlp_residual"`, new `ResidualRewardMLP` class in [src/reward.py](src/reward.py), dispatch in [src/jax_sim.py](src/jax_sim.py), mutation in [src/jax_evolution.py](src/jax_evolution.py).

**(B) Axis-2 → bin-aligned heading encoding** (already implemented in §13 — preserved).

**(C) DDB stability scaffold.** When species count is below a threshold, the breeding-energy gate `zeta` scales down via squared saturation:

```
f(N) = max(floor, N² / (N² + threshold²))
effective_zeta = zeta * f(N)
```

Predator threshold 5, prey threshold 30, floor 0.3. At N=threshold the gate is half its normal value (much easier to breed); at N=2·threshold it's 80% (mostly off); at healthy N it's negligible. Models a softer version of "competitive release" — sparse populations face less intraspecific competition for resources, breed more easily.

Behind config flag `stability_mechanism: "none" | "ddb"`, default `"none"` to preserve K&D-faithful behavior. New axis configs all set `"ddb"`.

**Pragmatic, not ideologically pure.** Real-world Allee effects often work the opposite way (rare animals struggle to find mates), but DDB is empirically grounded in fisheries / endangered-species recovery work. We're using it as a scaffold to enable long-run experiments and will test removing it later if stability emerges naturally. Default ON for axis runs going forward.

### 15.2 Small-scale baseline

To iterate faster without sacrificing the long-horizon scientific signal, the new baseline (`baseline_smol_ddb`) shrinks the world geometry:

| Param | Original | Small-scale |
|---|---|---|
| `world_size` | 960 | **600** (~40% area) |
| `prey_cap / predator_cap` | 450 / 50 | **200 / 25** |
| `prey_initial / predator_initial` | 150 / 10 | **75 / 5** |
| `food_max` | 600 | **300** |

Per-agent food ratio preserved. Estimated 1.5-2× sps speedup. Run length kept at 2M (user explicit preference: prefer 1 long run over 4 short ones for scientific signal).

### 15.3 Forward experiment ladder

| # | Config | What it tests | Cost (estimated) |
|---|---|---|---|
| **0** | `baseline_smol_ddb` (linear, 2M) | Validation: does the new baseline reproduce LV oscillation + fear evolution? | ~14h, ~$12 |
| 1 | `axis1_residual` (linear+MLP residual, 2M) | Does evolution grow useful nonlinear reward structure on top of K&D linear? | ~14h, ~$12 |
| 2 | `axis2_aligned_smol` (bin-aligned obs, 2M) | Does kinematics-co-located-with-position let prey preempt predator surges? | ~14h, ~$12 |
| 3 | `axis1+2_combo` (residual + bin-aligned, 2M) | Do the two axes compose, conflict, or amplify each other? | ~14h, ~$12 |

Each individual run produces an interpretable result regardless of outcome. Total ~3 days, ~$50.

### 15.4 What we deferred

**Axis-3 (temporal reward genome over k-step window):** the 945-param MLP has the same bootstrap problem as original axis-1, and the residual approach would need to be redesigned for temporal context. Deferred to a future session.

**Axis-4 (LSTM policy network):** PPO with truncated BPTT was wired in commit 5e69965 but never exercised end-to-end. Lower-priority extension. Deferred.

**Combined experiments beyond axis-1+axis-2:** the original 2×2 plan (linear/temporal × position-only/social) is shelved. If the axis-1 and axis-2 results justify it, combinations can be revisited.

### 15.5 What we kept from prior decisions

- mouth_smol substrate (predator_mouth_tactile_bins: [0]) — §10's breakthrough finding still anchors stability.
- bin-aligned heading encoding — §13's design is correct and is being preserved as the axis-2 mechanism.
- 20K checkpoint cadence + replay-uploads-every-flush + 100-replay milestone retention — operationally fine.

### 15.6 First validation attempt — small-scale failed at 2026-04-30

**`baseline_smol_ddb` 2M completed but predators went extinct at step ~80K.** The run finished prey-only for the remaining 1.92M steps. Diagnosis below; second attempt with a less aggressive scale-down is queued.

Trajectory:

| Step | Prey | Pred | Δcatch | Mean E (pred) |
|---|---|---|---|---|
| 10K | 200 | 13 | 110 | 118 |
| 20K | 173 | 14 | 119 | 124 |
| 30K | 114 | 12 | 97 | 128 |
| 50K | 74 | 9 | 43 | 64 |
| 60K | 77 | 4 (DDB fires) | 19 | 35 |
| **80K** | 96 | **0** | 2 | — |

**Why DDB couldn't save them — the key lesson:** at step 60K when pred=4 (below threshold 5), DDB lowered `effective_zeta_b_pred` from 100 to 39. But predator *energy* was only 35 — to actually breed, predators needed energy ≥ effective_zeta/beta_b ≈ 97. DDB lowers the breeding-energy bar; it doesn't put energy in the bank. If predators are starving, DDB can't make them breed.

**Why predators were starving — the upstream cause:** at world_size=600 with caps 200/25, predator-prey density was ~2.5× the K&D-faithful baseline (which used world=960, caps 450/50). Catches/step were ~5× normal in the first 20K steps. Prey crashed from 200 → 74 between step 20K and 50K — faster than they could breed back even with food at cap. Predators then ran out of prey to eat, energy decayed below the breeding threshold, and DDB couldn't catch them in time.

This is a different failure mode than the prior "predators too weak" extinctions — it's the same trophic over-pressure pattern axis-2 v1 hit at 1.1M, but earlier and from geometry rather than from richer obs.

### 15.7 Second attempt — middle-ground geometry + DDB threshold bump

`baseline_med_ddb` (commit TBD) walks back the small-scale geometry to a less aggressive middle ground, and bumps the predator DDB threshold so the rescue fires earlier in the decline.

| Param | Original | Failed small | **Middle (this attempt)** |
|---|---|---|---|
| `world_size` | 960 | 600 | **800** (70% area, 1.4× density vs 2.5× failed) |
| `prey_cap` | 450 | 200 | **300** |
| `predator_cap` | 50 | 25 | **30** (cap ratio 10:1, slightly prey-favored vs original 9:1) |
| `prey_initial` | 150 | 75 | **100** |
| `predator_initial` | 10 | 5 | **7** |
| `food_max` | 600 | 300 | **450** |
| `ddb_pred_threshold` | n/a | 5 | **8** ← bumped to fire earlier in decline |
| `ddb_prey_threshold` | n/a | 30 | **30** (unchanged) |

**Why threshold = 8 (not higher):**
- At pred=8, DDB factor f=0.5 (half-rescue). Above pred=16, f≥0.86 (mostly off).
- Higher threshold (e.g. 10) would fire during normal mid-cycle dynamics (typical pred 15-20), shifting DDB from "scaffold for emergencies" to "permanent tilt."
- The smoking gun in the failed run was at step 50K: pred=9, mean E=64. Threshold=8 would have fired at this point, giving predators a breeding window before their energy crashed.

**One honest caveat:** if the middle-ground geometry doesn't sufficiently reduce predator over-pressure, DDB still can't fix the underlying starvation. In that case we'd add **DDM (density-dependent metabolism)** — slow predator energy decay when rare — as a complementary mechanism. Holding off until needed; one change at a time.

### 15.8 Second attempt — partial rescue, partial collapse

`baseline_med_ddb` ran from 22:26 UTC. Through the first 100K steps:

| Step | Prey | Pred | Δcatch | Pred mean E |
|---|---|---|---|---|
| 30K | 300 | 16 | 97 | 93 |
| 60K | 271 | **6** (DDB fires) | 32 | 68 |
| 70K | 198 | **10 (DDB rescued!)** | 99 | 54 |
| 80K | 148 | 8 | 74 | 41 |
| 90K | 148 | 2 | 33 | 35 |
| 100K | 163 | **1** | 9 | 36 |

**Genuine partial success:** DDB rescued the population once (pred 6→10 at step 70K) — first time we've seen a real population recovery from a bottleneck. Prey held at 150-300 throughout (vs failed run's crash to 74). Geometry+threshold tweaks worked for trophic balance.

**But the lone-survivor problem returned:** at pred=1 with energy=36, even DDB's lowered breeding gate of effective_zeta=30 (need energy ≥ 75) couldn't fire. The lone predator was *starving*, not just bottlenecked. DDB lowers the breeding bar but can't put energy in the bank.

### 15.9 Adding DDM (Density-Dependent Metabolism) — third attempt

Mirrors DDB's shape but applied to predator energy decay rate `d_b`:

```
effective_d_b = d_b * f(N_pred)
where f(N) = max(floor, N² / (N² + threshold²))
```

When N=1 (deep bottleneck), `f=0.3` (floor) → predator metabolism is 30% of normal → lone survivor's energy budget is **3× longer** while waiting for catches. When N=20 (healthy), `f≈0.86` → essentially off, dynamics unchanged.

**The intuition:** DDB lowers the bar for breeding; DDM keeps the lone survivor alive long enough to catch prey and reach the bar. Together they bracket the bottleneck-survival problem from both sides.

**Biological grounding:** real solitary hunters often have lower per-capita metabolic costs than group hunters — less competition for prey patches, less stress, no territorial defense. Empirically observed across ecology. Like DDB, this is a soft analog to a real-world effect, used here as scaffold not as biological claim.

**Config:** `stability_mechanism: "ddb_ddm"` (combined). Same threshold/floor as DDB by default (`ddm_pred_threshold: 8.0`, `ddm_floor: 0.3`). Implementation in [src/jax_lifecycle.py](src/jax_lifecycle.py) — `update_energies_jax` now scales predator `d_b` when `pred_count` is below threshold.

**Predator only.** Prey didn't have a starvation problem in the failed runs — their crashes came from over-predation, which the geometry change addressed. Adding prey-DDM would risk runaway prey growth.

### 15.10 Status snapshot

- Code committed: DDB + residual reward + small-scale (`dd66d92`), middle-ground (`8582397`), DDM (`c56441e`).
- Tests pass (231/231). End-to-end smoke test of `baseline_med_ddb_ddm` runs cleanly.
- First validation `baseline_smol_ddb` 2M: predators extinct at ~80K (trophic over-pressure).
- Second validation `baseline_med_ddb` ran to ~step 100K with pred=1, partial DDB rescue but lone-survivor starvation. Killed to launch v3.
- Third validation `baseline_med_ddb_ddm`: completed 2M steps; predators extinct at ~step 1.35M after multiple LV cycles. Strong herd-seeking evolved (`prey_w_prey=+7.69`) but no fear (`prey_w_pred≈0`) — same trophic-collapse-via-herd pattern as axis-2 v1, just delayed ~12× by the scaffolds.

### 15.11 Strengthened scaffolds — strong DDB+DDM with rate boost

**Diagnosis from the 1.35M extinction.** The DDB+DDM scaffolds did their job (extending co-evolution from 80K → 1.35M) but ultimately couldn't prevent the canonical trophic-collapse failure. Two structural weaknesses:

1. **Floor=0.3 is too generous in the healthy regime.** At pred N=8 the curve is at 0.5 (50% cost), still half-scaffolded. Selection pressure on hunting weights was weakened across the entire low-population regime, not just at the bottleneck.
2. **DDB lowered the breeding bar but didn't accelerate breeding rate.** Even with `zeta_b_eff` near zero at N=1, `kappa_b=1e-3` per step means a lone survivor only attempts a birth every ~1000 steps. To recover from N=1 → N=4 takes thousands of steps × 4 births in best case, much longer if the survivor is a bad hunter — long enough to drift into geometric "can't find prey" failure.

**Fix.** Three knob changes, all applied together:

| Knob | Old | New | Effect |
|---|---|---|---|
| `ddb_pred_threshold` | 8.0 | **4.0** | curve fades faster: N=4 → 0.50, N=8 → 0.80 |
| `ddb_prey_threshold` | 30.0 | **40.0** | scaled to prey cap (10× pred cap) |
| `ddb_floor` / `ddm_floor` | 0.3 | **0.0** | extreme low-N pays near-zero cost |
| `ddb_max_boost` (NEW) | n/a | **50.0** | `kappa_b → kappa_b / max(factor, 1/50)` at low pop |

**Mechanics of the rate boost.** `_batch_birth_prob_jax` now scales `kappa_b` *upward* at low pop using the inverse of the squared-saturation factor, capped at `max_boost`:

```
prey_boost = 1.0 / max(prey_factor, 1.0 / max_boost)
kappa_b_eff_prey = kappa_b * prey_boost
# (and symmetric for predators)
```

Concretely with pred T=4, max_boost=50:

| N_pred | factor | breed bar | breed rate | expected steps/birth |
|---|---|---|---|---|
| 1 | 0.06 | 5.9 | 17× | ~60 |
| 2 | 0.20 | 20 | 5× | ~200 |
| 4 | 0.50 | 50 | 2× | ~500 |
| 8 | 0.80 | 80 | 1.25× | ~800 |
| 15+ | ≈1.0 | ≈100 | 1× | ~1000 |

A lone predator with any positive energy now breeds within ~60 steps. From N=1 → N=4 takes ~600 steps total instead of ~5000. The geometric "lone bad hunter wanders unable to find prey" failure mode is squeezed out — by the time they could starve they've produced a handful of offspring with mutated reward weights, restoring genetic diversity.

**Biological framing.** The rate boost is the more defensible of the two scaffolds: low-density populations in real ecosystems often *do* have higher per-capita reproductive output (less intraspecific competition, more resources per individual). The threshold drop is a pragmatic supplement — biologically less realistic but needed to ensure the boost has a positive-energy individual to act on. Together they're a deliberate scaffold designed to enable the simulation to run on a long enough timescale for evolution to act, not a claim about real-world ecology.

**What this preserves.** At healthy populations (factor > 0.95), both scaffolds are nearly inactive — boost ≈ 1.0×, threshold ≈ normal. Selection pressure on hunting and herding weights is essentially unmodified in the regime where evolution actually has work to do.

**What's now in the axis configs.** Both `configs/axis1_residual.yaml` and `configs/axis2_aligned_smol.yaml` (`experiment_name: axis2_aligned`) now use middle-scale geometry + strong DDB+DDM with rate boost. Skipping a fresh baseline run with these stronger scaffolds in the interest of time — axis runs themselves serve as the validation. Baseline-with-strong-scaffolds can be filled in later if axis results suggest the scaffolds aren't doing their job.

### 15.12 Diversity loss before bottleneck — threshold bump (4→8)

**The problem the first axis-1 launch surfaced.** With the strong scaffolds tuned to engage at N≤4 (`ddb_pred_threshold=4`, `ddm_pred_threshold=4`, floor=0, max_boost=50), the run survived a population crash from peak pred=18 (step 40K) down to pred=2 (step 120K) without going extinct. But inspecting per-genome weights at step 120K revealed the rescue cost too much:

```
step 80K  (pred=8):  std on `prey` weight = 0.59  range [-1.44, +0.64]
step 120K (pred=2):  std on `prey` weight = 0.02  range [+0.13, +0.17]
```

The **two surviving predators at step 120K are the ancestral lineage** — ages 120K and 114K, alive since step 0 / step 6K respectively. Their weights are essentially their initial random draw. One of them has `eat=+0.00` — a predator that gets zero reward signal from catching prey, surviving by physical luck.

The phenotypically-interesting predators died:
- `pred=+1.27, prey=-1.44` (extreme cooperative, prey-avoidant) — gone
- `pred=+0.71` (pro-cooperative hunter) — gone
- `act=+0.80` (high-motor explorer) — gone

Selection wasn't picking the best hunters; it was just **culling the high-variance lineages at random** during the LV crash. The strong scaffolds successfully prevented extinction at N=2, but by the time they engaged the diversity battle was already lost.

**Why scaffolds were too late.** With T=4, the squared-saturation curve is essentially off across the danger zone:

| N (pred) | T=4 factor | T=8 factor |
|---|---|---|
| 8 | 0.80 | 0.50 |
| 10 | 0.86 | 0.61 |
| 12 | 0.90 | 0.69 |
| 15 | 0.93 | 0.78 |
| 20 | 0.96 | 0.86 |

At N=10-15, T=4 means predators pay 86-93% of normal cost — selection is weakly tied to fitness because random death dominates. At N=20+ both T=4 and T=8 are essentially off, so the healthy regime is unchanged.

**Fix.** Bump thresholds 4→8 (predator) and 40→80 (prey, scaled proportionally to caps). Same scaffold *shape*, just engages earlier:
- N=10-15: factor 0.61-0.78 → predators pay 61-78% of normal, breeding rate 1.3-1.6× normal — survivors live measurably longer, weak hunters can be culled by selection rather than disappearing at random.
- N=20+: factor ≥0.86 → essentially unscaffolded, selection on hunting/herding weights intact.

**Validation:** killed the first axis-1 run at step ~120K (preserved checkpoints + replays for diversity-loss writeup) and re-launched with T=8 scaffolds. If selection grips earlier this time, surviving genomes at step 100K+ should retain the wider weight distributions seen at step 80K of the first run.

**Caveat on scale.** Diversity preservation also depends on absolute population size — at predator_cap=30, even peak pred=18 means only ~18 distinct genomes in play simultaneously. Paper baseline (cap=50, peak ~25-30) has ~1.4-1.7× more genomes carrying the population's evolutionary state, which is structurally better for surviving an LV crash without losing rare variants. Middle-scale geometry trades some diversity for ~1.5× faster iteration. The scaffolds in this section partially compensate by preventing deep bottlenecks; they don't fully replace the larger-population effect. If/when an axis result motivates a final paper-scale run, the 960²+cap-50 setup should naturally preserve diversity better even with weaker scaffolds.

### 15.13 Med-large scale + T=10 — final pre-axis tune

After the T=4 → T=8 bump above, decided to also bump scale (med-large: world=880, prey_cap=375, pred_cap=40, food_max=525) to reach a regime closer to paper baseline (cap=50) without paying the full iteration tax. Pred_cap=40 carries ~33% more genome diversity through LV crashes than cap=30, structurally — independent of scaffold timing.

Threshold also bumped one more notch: **T=10 (pred), T=100 (prey)**. With the larger cap, peak predator pop is ~24-28 (vs ~18 at cap=30), so the scaffold curve needs to shift right to keep the healthy regime mostly unscaffolded:

| N (pred) | T=8 | **T=10** | T=12 | comment |
|---|---|---|---|---|
| 8 | 0.50 | 0.39 | 0.31 | pre-bottleneck |
| **10** | 0.61 | **0.50** | 0.41 | LV-crash inflection |
| 12 | 0.69 | 0.59 | 0.50 | mid-crash |
| 15 | 0.78 | 0.69 | 0.61 | early-crash |
| 20 | 0.86 | 0.80 | 0.74 | recovery |
| 24 | 0.90 | **0.85** | 0.80 | ~peak |
| 30 | 0.94 | 0.90 | 0.86 | healthy |

T=10 hits factor=0.5 right at the N=10 LV-crash inflection (where the first run lost most of its diversity), factor=0.69 at N=15 (early crash), and factor=0.85 at N=24 (peak — selection mostly intact). T=12 was rejected as engaging too aggressively at peak (0.80 factor — paying 20% less than normal even when healthy, which would visibly blunt selection in a regime where we want it intact).

**Iteration cost.** ~32 sps at cap=30 → ~26 sps at cap=40 (estimate based on agent-count scaling). 10M steps: 87h → 107h, +20h / +1 day. Acceptable for the diversity gain.

**This is the configuration we're running for axis-1 and axis-2:**
- world=880, prey_cap=375, pred_cap=40
- T=10 (pred), T=100 (prey), floor=0, max_boost=50
- middle-ground between paper baseline (would naturally preserve diversity but slow iteration) and the original small-scale run (fast but hostile to evolution)

### 15.14 Energy-weighted boost + dropping DDM — selection-aligned scaffolds

**The problem with population-only scaffolds.** Even with the T=10 + med-large tune (§15.13) running cleanly at step 200K (pred oscillating 7-12, healthy LV cycles), inspection of the reward-genome means revealed a deeper issue. The means were drifting toward zero with high variance:

| | step 60K | step 200K |
|---|---|---|
| pred_w_eat  | +0.17 ± 0.48 | -0.00 ± 0.64 |
| pred_w_prey | +0.34 ± 0.91 | -0.00 ± 0.86 |

Under healthy selection we'd expect `pred_w_prey` to evolve clearly positive (rewarded for being near prey, encouraging hunting). Instead the population was random-walking around zero — characteristic of weak selection. Three mechanisms were diluting selection:

1. **DDM (decay scaling) keeps bad hunters alive longer.** A predator that catches little would normally starve in ~250 steps; with DDM factor=0.5 they live ~500 steps, doubling their chances to randomly bump into prey. **This is a direct selection-killer.**
2. **DDB threshold drop lets bad hunters reproduce.** A predator that can only reach energy 50 normally couldn't breed; with zeta_eff = 50 they can. Low-fitness genomes propagate.
3. **DDB rate boost is uniform across the species.** Whether a predator has 800 energy or 50, they get the same 2-3× breeding rate boost. The scaffold is *individual-blind* — it rescues the population without distinguishing between high- and low-fitness members.

**The fix: redistribute the rate boost by within-species energy share, drop DDM, keep threshold drop.**

The redistribution preserves the *total* breeding pressure on the species (so the population still recovers from low N) but allocates it proportionally to relative energy:

```
agent_boost_i = species_uniform_boost * (energy_i / Σenergies_in_species) * N_species
```

Properties:
- **At healthy pop** (factor ≈ 1, all energies similar): `agent_boost ≈ 1` for everyone — no scaffold engaged, identical to K&D.
- **At low pop with uniform energies**: `agent_boost = boost_uniform` per agent — same as old uniform scaffold.
- **At low pop with skewed energies**: high-energy agents get most of the boost, low-energy agents get little. Selection-aligned.

Example at pred=8, factor=0.39 (boost_uniform=2.6×, total budget=20.5):

| Energy | Energy share | Old boost | New boost (energy_weighted) |
|---|---|---|---|
| 800 | 47% | 2.6× | **9.6×** |
| 300 | 18% | 2.6× | 3.6× |
| 200 | 12% | 2.6× | 2.4× |
| 100 | 6%  | 2.6× | 1.2× |
| 50  | 3%  | 2.6× | 0.6× (below normal) |

The top hunter breeds 9.6× normal rate; the worst breeds at 0.6× normal — *less* than they would without scaffolds. **Bad hunters are actively penalized when good hunters exist.**

**Why drop DDM.** Once breeding rate is energy-weighted, the only thing DDM does is keep dying-bad-hunters alive longer. They still don't get to breed (their `e_share` is tiny), so they just consume world resources without contributing to evolution. Dropping DDM lets natural starvation prune them. Selection on hunting ability fully restored.

**What's preserved:**
- Population rescue: when pred=2 and both are alive, the breeding rate boost still applies — they'll repopulate quickly.
- Threshold drop: at low pop, even a "current-best" predator with energy 60 (relatively top-of-pack) can breed.
- LV cycling: nothing about the dynamics changes at healthy populations.

**What's improved:**
- Selection differential at the *individual* level is preserved through the entire scaffold envelope.
- `pred_w_prey` and `prey_w_pred` should drift cleanly toward fitness-aligned values (not random-walk around zero).
- Variance reduction over time will be *meaningful* (selection-driven) rather than the v1 kind (random-survivor lottery).

**Implementation** ([src/jax_lifecycle.py](../src/jax_lifecycle.py)):
- New config knob `ddb_boost_distribution: "uniform" | "energy_weighted"` (default uniform — backward compat).
- `_batch_birth_prob_jax` now takes `is_active` to mask inactive slots from the energy denominator.
- 3 new tests in [tests/test_ddb_ddm.py](../tests/test_ddb_ddm.py): redistribution at non-uniform energies, total-budget conservation at uniform energies, zero breeding for zero-energy agents.

**Live config (axis-1 v4):**
- `stability_mechanism: "ddb"` (DDM removed)
- `ddb_pred_threshold: 10.0`, `ddb_prey_threshold: 100.0`
- `ddb_floor: 0.0`, `ddb_max_boost: 50.0`
- `ddb_boost_distribution: "energy_weighted"` (NEW)
- Med-large scale: world=880, prey_cap=375, pred_cap=40

### 15.15 The v7 extinction — DDM is not optional

**v7 launched (med-large + T=10 + DDB only + energy-weighted boost) and predators went extinct at step 88K.** Trajectory:

| step | pred | catches/10K | note |
|---|---|---|---|
| 40K | 22 | 167 | peak |
| 50K | 19 | 119 | crash phase begins |
| 60K | 12 | 90 | descending fast |
| 70K | 6 | 36 | bottleneck |
| 80K | 4 | 34 | edge |
| 90K | 0 | 13 | extinct |

Compare to v3 (uniform boost + DDM) at matched steps: pred=8 at step 80K and step 90K. v3 held steady through the same regime where v7 collapsed.

**The diagnosis.** I dropped DDM and added energy-weighted DDB at the same time, treating them as a coupled change. But they protect against different failure modes:

- **DDM (decay scaling)** keeps the *population* alive during LV crashes. When prey crash, ALL predators lose energy simultaneously. Without DDM, full-cost decay finishes off the predator population before any individual can recover.
- **DDB rate boost (uniform)** kept low-energy predators reproducing. The selection-killer.
- **DDB rate boost (energy-weighted)** redistributes breeding to top performers. But this requires *some* predators to be high-energy — when everyone is starving together (mid-LV crash), there's no fitness gradient to concentrate on.

The earlier v3 ran successfully because:
- DDM kept the population from crashing to extinction (everyone survives the crash, weakened)
- Uniform DDB kept everyone reproducing (some weakly, but enough to recover)

I dropped DDM thinking energy-weighted DDB would handle population recovery on its own. **It can't, because the prerequisite (fitness gradient) doesn't hold in deep bottlenecks.**

**The fix (v8): keep DDM, keep energy-weighted DDB.** Two scaffolds operating in *different dimensions*:

- **DDM (uniform)** keeps low-energy predators *alive* (population persists).
- **DDB (energy-weighted)** keeps low-energy predators *from reproducing* (their `e_share` is small → small breeding boost).

Bad hunters survive but don't propagate. Eventually they starve out (DDM only delays, doesn't prevent — action cost is unscaled). Population stable; reproduction concentrated on top fitness.

The earlier worry that "DDM keeps bad hunters reproducing" was true *only with uniform DDB boost*. With energy-weighted boost, the two scaffolds decouple: DDM preserves the population while energy-weighted DDB preserves selection. They aren't in tension; they're complementary.

**v8 config diff from v7** (only the stability section changes):
```
stability_mechanism: "ddb_ddm"   # was "ddb"
ddm_pred_threshold: 10.0         # NEW (matches DDB threshold)
ddm_floor: 0.0
# everything else identical: T=10, max_boost=50, energy_weighted, etc.
```

**Lesson for future scaffold tuning:** scaffolds operate in different dimensions (survival vs reproduction). Don't drop one because you "don't need it anymore" — first verify which dimension it covers, and what would happen in the regimes where its absence matters. The v7 extinction showed up exactly in the deep-bottleneck regime where DDM's survival function was load-bearing.

**Archived:** `~/evo-reward/results/axis1_residual_v7_no_DDM_extinct` — 88K-step run with full extinction trajectory. Useful as a "publishable failure mode" showing that DDM is necessary even when DDB is energy-weighted.

### 15.16 Continuous α-tuning for DDB boost distribution

After v7 (drop DDM, full energy-weighted breeding) extincted, restoring DDM was clearly necessary (§15.15). But the binary choice — *uniform* vs *full energy-weighted* — is unsatisfying. The right boost distribution probably lives somewhere on a continuum, and may even need to be different for different runs (or different phases of the same run).

**Mechanism.** Added a continuous parameter `ddb_boost_distribution_alpha ∈ [0, 1]`:

- `α = 0.0` → uniform: every active agent gets the same boost (legacy behavior, K&D-faithful)
- `α = 0.5` → linear: per-agent share ∝ energy (the "energy_weighted" of §15.14)
- `α = 1.0` → winner-take-all: only the highest-energy individual breeds

Internally implemented as a power-law on energy: per-agent share ∝ `energy_i^k` where `k = α / max(1−α, 1e-3)`. So:

- α=0 → k=0 → all powers = 1 → uniform shares
- α=0.5 → k=1 → linear shares
- α=0.95 → k=19 → top agent gets ~95%+ of budget
- α→1 → k→∞ → top agent gets 100%

Computed via stable masked log-softmax: `share_i = softmax(k · log(energy_i))`. JAX-jit-friendly, no overflow at high α.

**Total breeding budget preserved at the species level** for all α: Σ shares = 1, so Σ (N · boost_uniform · share_i) = N · boost_uniform. Changing α reallocates the budget but doesn't change its size. Population-rescue function intact at any α.

**Backward compat.** The old string knob `ddb_boost_distribution: "uniform" | "energy_weighted"` still works and maps to α=0 / α=0.5 respectively. Set `ddb_boost_distribution_alpha` to override.

**Why this matters.** v7's failure (§15.15) suggests that very-high α might fail in the same way for the same reason: when ALL predators are simultaneously low-energy, "concentrate breeding on top" has nothing to concentrate on. So α should probably never go too close to 1 in regimes where bottlenecks happen. But α=0.5 (where v8 sits) might also be over-correcting — maybe α=0.3 or α=0.4 is the actual sweet spot. Easy to sweep.

**Default unchanged: α=0.0 (uniform)** for any config that doesn't explicitly set it. Backward-compat preservation.

**Live config (axis-1 v8):** `ddb_boost_distribution_alpha: 0.5` (= legacy "energy_weighted"). DDB+DDM both active, T=10, max_boost=50.

**Future-tuning guidance:** if v8 still shows "weak selection" symptoms (means random-walking around zero), bump α up to 0.7. If v8 shows extinction risk during LV crashes (similar to v7), bump α down to 0.3 or 0.2. The key heuristic: α controls *how steep* the energy-fitness gradient is in the breeding scaffold. Steeper = more selection-aligned but more fragile to "everyone-low-energy" regimes; flatter = more stable but more drift.

| α | Top-agent share at energies [800, 100, 100] | Comment |
|---|---|---|
| 0.0 | 33% | uniform (1/N) |
| 0.3 | 67% | mild bias |
| 0.5 | 80% | linear (current v8) |
| 0.7 | 92% | strong bias |
| 0.9 | 99% | near-winner |
| 1.0 | 100% | winner-take-all |

A natural follow-up if v8 doesn't immediately show what we want: schedule α as a function of step count (linearly increase from 0.3 → 0.6 over the first 1M steps, simulating a "gradual selection sharpening" as evolution gets going).

### 15.17 v9 — paper-faithful proximity range + per-area food density

While v8 was running on GCP we noticed two scale-correctness slips in the active configs that hadn't tracked our world-size shrink (paper 960² → med-large 880²). Neither is severe enough to abort v8, but we applied them so the *next* launch is on a more defensible config (commit `c7b81e1`).

**1. `proximity_max_range: 200 → 120`.** The paper (Appendix A) specifies 120. emevo's TOML has 200, which we'd silently carried forward in every axis and baseline-stepping-stone config (D27 documented this in `baseline_faithful.yaml` only). At world=880, a 200-unit sensor range covers 22.7% of world width — vs the paper-spec 12.5%. Predators were sensing far farther than they're supposed to, which probably made hunting too easy in the small-world regime. Side benefit: per-step distance checks are cheaper since fewer agents fall inside range (area ratio (120/200)² = 0.36).

**2. Scale-relative food growth.** Every config had `food_growth_rate: 0.5` regardless of world size, so per-unit-area food density was inflated on smaller worlds (at world=880, 17% denser than paper). The fix is structural: a new config key `food_growth_rate_at_960sq` is interpreted as "the rate at the paper's 960² world" and scaled by `(world_size / 960)²` at config-load time. Resolver in `src/config_utils.py`. `baseline_faithful.yaml` keeps the legacy absolute key, which is identical at world=960. `axis1_residual.yaml` and `axis2_aligned_smol.yaml` (world=880) now resolve to 0.5 × (880/960)² ≈ 0.420 effective food/step.

**Why these only apply on next launch.** v8 loaded its config at startup (22:03 UTC); the c7b81e1 commit landed ~90 min later. The running process is unaffected. Editing the config file mid-run is safe (it's not re-read), but means git history diverges from the running process — the v8 run_tag pins the live config snapshot to commit `a05bc0a`.

**Side effect of doing this work: configs/ is now reorganized.** Active runs at top level (`axis1_residual.yaml`, `axis2_aligned_smol.yaml`, `baseline_faithful.yaml`); everything else moved into `configs/archive/` with a per-file README of outcomes. See `configs/README.md` and `configs/archive/README.md`.

**What this means for v9.** No new science; just config hygiene. If v8 finishes cleanly, v9 (or whatever we relaunch with these configs) becomes a slightly cleaner re-run that's more directly comparable to `baseline_faithful.yaml`. If v8 dies, v9 is what we'd restart on.

### 15.18 v8 — first evidence of MLP residual utilization (axis-1 Q1 partially answered)

Mid-run inspection of v8 checkpoints at step 280K and 440K confirms the residual MLP is being utilized — answering the binary half of axis 1.

**Per-agent residual L1 (sum of |w| over all 25 residual params):**

| step | active preds | pred mean L1 | pred min L1 | prey mean L1 | prey max single param |
|------|--------------|--------------|-------------|--------------|-----------------------|
| init | 9            | 0.000        | 0.000       | 0.000        | 0.000 (zero-init)     |
| 280K | 12           | 1.99         | 0.00        | 2.90         | **5.00 (hit ±5 clip)** |
| 440K | 8            | 2.81         | **1.78**    | 3.58         | 2.33                  |

**Why this isn't just mutation noise.** At `residual_mutation_scale=0.03` and ~10 generations of inheritance through 440K steps, an unconstrained random walk would predict L1 ~ 0.5. We see ~2.8 on predators — selection is pushing residuals up, not just drifting. By 440K every surviving predator has L1 ≥ 1.78; the "flat-zero residual" lineage is gone from the predator population.

**Clip-hit episode at 280K.** One prey lineage saturated a single residual param at the ±5 clip. By 440K that lineage is gone (max param drops to 2.33). Worth tracking the clip-hit rate over time as a pathology signal.

**What this evidence does NOT answer:** whether the residual encodes structure linear cannot, or just makes the linear gradient steeper. That's Q2 of axis-1 — the more interesting question. Designed in [docs/proposals/axis1-residual-analysis.md](proposals/axis1-residual-analysis.md), to be implemented in a future session. The proposal also covers an optional in-loop residual-L1 logging tweak (~10 LOC, sub-microsecond cost) that would expose this trajectory in real time without needing to download 127 MB checkpoints — useful for failure detection on long runs but not load-bearing for the science.

### 15.19 v10 — todo block (within-lifetime training mismatch + mouth narrowness)

Three observations from the v8 run motivate a v10 config revision.

**1. Predator mouth is narrower than even K&D's smallest config.** Active configs run `predator_mouth_tactile_bins: [0]` (a single 20°-wide bin = 20° catch arc). emevo's "small" config and our test-suite reference value is `[0, 1, 17]` (3 bins = 60° arc, the front-center triplet). The K&D paper sweeps 3 mouth sizes (small/medium/large); we are below the smallest. Replay observation suggests predators are visibly missing prey that pass close to the mouth but slightly off-center — consistent with a sub-paper arc.

**Action:** v10 sets `predator_mouth_tactile_bins: [0, 1, 17]` (paper-faithful "small"). Single-line change in `axis1_residual.yaml` and `axis2_aligned_smol.yaml`. Couples with all three other levers (more catches → more energy → longer pred lifespan → more PPO updates → policies fit genomes better).

**2. Within-lifetime PPO training is starved on predators.** Live-age snapshot at step 2.06M:

| | Prey (n=312) | Predator (n=8) |
|---|---|---|
| median age | 27,972 | **6,103** |
| median age in PPO updates (rollout=1024) | 27 | **6** |
| 95th-pct age | 112,657 | 101,159 |

Median predator dies after ~6 PPO updates — barely enough for a 4-layer 64-hidden MLP to fit the reward genome it's been handed. (Inspection-paradox bias means *actual* mean lifespan-at-death is even lower; live samples skew toward survivors.) Selection thus operates on the rare 95th-pct survivor with ~100 updates, not on the median predator. The mouth-widening above will help indirectly, but a direct lever is also worth a look.

**Action:** add a learning-rate schedule. Current `lr: 3.0e-4` is constant. Two options to evaluate:
- (a) Higher constant LR (`1e-3`) — fits short lives faster, may destabilize long-lived agents.
- (b) **Per-agent LR decay** keyed off agent age — start high (`1e-3`), decay to `3e-4` over ~30K steps. Lets young agents learn fast, lets mature agents stabilize. Implementation: optax schedule indexed by `state.ages[i]`. ~20 LOC in `src/jax_ppo.py`. The per-agent angle is critical because pop-level wall-step decay would punish newborns when the run is mature.

Default to (b) for v10 — has the right inductive bias (newborns need fast learning, apex survivors need stable policies) and matches biological intuition (juvenile plasticity → adult specialization).

**3. Death-age logging.** We don't currently capture lifespan distributions; we track per-slot `ages` but never snapshot at the T→F transition of `is_active`. Live-age snapshots are biased by the inspection paradox.

Post-hoc reconstruction from existing checkpoints is **not** reliable: slots are reused across births, and `agent_ids` for dead slots get overwritten at next birth into that slot. So even with consecutive checkpoints, an agent that dies and gets replaced between two saves is undetectable. Death-age must be captured online.

**Action:** in `src/jax_lifecycle.py::process_births_and_deaths_jax`, before the `is_active` mask is updated, accumulate `ages[died_mask]` into a per-species ring buffer (size ~256 most-recent deaths). In progress lines, log percentiles (5/50/95) per species. ~30 LOC. No checkpoint-format change required (ring buffer can live in `SimState` or as a separate metrics struct).

**Combined v10 launch checklist** (one config + small code patches):
- [ ] `predator_mouth_tactile_bins: [0, 1, 17]` in two configs.
- [ ] Per-agent age-keyed LR schedule (default 1e-3 → 3e-4 over 30K steps).
- [ ] Death-age ring buffer + progress-line percentile logging.
- [ ] Re-validate `tests/test_predator_eating.py` and `tests/test_jax_ppo_update.py` still pass.
- [ ] Document v10 vs v8 deltas as §15.20 once launched.

**Two-track strategy for v10.** We split the v10 work into a paper-faithful track and a rapid-iteration track:

- **v10** (paper-faithful): the three changes above (mouth, age-LR, death-age logging) and nothing else. This is the run we'd cite in publications. Comparable directly to v8 because it preserves PPO geometry and policy capacity.
- **v10-fast** (branched, for cheap iteration): v10 + a **PPO/policy speed combo** that trades paper faithfulness for wall-clock. Used to iterate on design choices that don't need paper-absolute numbers — e.g., "does the age-LR schedule actually help predator convergence?" can be answered directionally in a 12-hour v10-fast run rather than a 4-day v10 run. We then re-validate the winning configuration on a v10 run.

**v10-fast speed combo** (pulled into the branched config; left out of v10):

| Knob | v10 (paper-faithful) | v10-fast | Expected speedup | Cost |
|---|---|---|---|---|
| `ppo_epochs` | 10 | 5 | ~halves PPO update cost | slower per-lifetime fitting |
| `minibatch_size` | 256 | 512 | small (fewer launch overheads) | none in theory |
| `policy_hidden_size` | 64 | 32 | 2-4× cheaper forward+backward | lower capacity (likely fine on axis-1's 4-D reward, **not** on axis-2's 333-D obs) |
| `world_size` | 880 | 800 | smaller pairwise-distance fields, modest | deviates from paper geometry; close to proven `baseline_med_ddb_ddm` (800²) which ran 1.35M stably |
| `prey_initial / cap` | 125 / 375 | 125 / **400** | LV peaks unclipped → more representative ecology + more genetic turnover at peaks | density 6.25e-4 = **1.28× paper**; novel territory for stability |
| `predator_initial / cap` | 9 / 40 | 9 / 40 | unchanged | density 6.25e-5 ≈ 1.15× paper; pred cap rarely binds in any of our runs |

**Why caps go UP at 800² rather than scaling isometrically down:** v8 progress logs show prey cap binds frequently (~24% of recent windows pinned at or near 375). Cap-binding suppresses births at LV peaks, which is when genetic turnover should be at its highest. For v10-fast we want **uncapped LV cycles** — let prey overshoot equilibrium, observe natural crashes, see the full ecology. The hot-density bet (1.28× paper for prey) trades known-stable scaffolding for richer dynamics; risk-bounded because v10-fast wins must be re-validated on v10 anyway.

The compute cost of a higher prey cap at smaller world is roughly: equilibrium prey at 800² density-isometric ≈ 240; with cap=400 we expect peaks of maybe ~340 vs current v8 peaks pinned at 375. So actual per-step compute is approximately the same as v8 even with the higher cap, just with the cap no longer biting.

**Caveats:**
- DDB+DDM at the current settings was validated at 0.96-1.00× paper density. 1.28× is novel; extinction risk is low but not zero.
- If v10-fast crashes early (e.g., extinction at <500K), fall back to density-isometric 300/30 which we know is stable.

Combined expected wall-clock: **~2× faster** than v10. (Less aggressive than my original 2-3× estimate; the higher prey cap eats some of the smaller-world savings.) Iteration speed > paper-absolute numbers for v10-fast; we re-validate winning ideas on full v10 afterwards.

Quality cost on axis-1 should be small (4-D reward is overprovisioned by the 64-hidden 2-layer MLP), but this is exactly the kind of assumption v10-fast is good for testing.

**Knobs we did NOT pull even into v10-fast** (recorded so future-us can find them):

| Knob | Current | Candidate | Why we skipped |
|---|---|---|---|
| `policy_n_hidden_layers` | 2 | 1 | already shrinking hidden_size; doubling up risks too-aggressive capacity drop |
| `n_proximity_sensors` | 32 | 16 | changes observation semantics, not just compute — would need separate sensor-resolution ablation |
| `n_physics_iter` (hardcoded `5` in [src/environment.py:28](../src/environment.py#L28)) | 5 | 3 | tunneling risk at high velocity; deviates from emevo physics fidelity |

**Do not apply v10-fast knobs to axis-2** without re-validation — the 333-D social-obs setup probably needs the larger policy.

**Tier ladder framing (for future organization).** v10/v10-fast naturally extends to a 3-tier ladder of compute-vs-fidelity tradeoffs:

| Tier | Name | World | Caps (prey/pred) | Policy | PPO | Use case | Approx wall-clock for 1M steps |
|---|---|---|---|---|---|---|---|
| **L2** | v10-fast | 800² | 400 / 40 | hidden=32 | epochs=5, mb=512 | overnight directional answers | ~10-12h |
| **L3** | v10 | 880² | 375 / 40 | hidden=64 | epochs=10, mb=256 | paper-comparable with intentional fixes | ~24-36h |
| **L4** | baseline-faithful | 960² | 450 / 50 | hidden=64 | epochs=10, mb=256 | true K&D reproduction (mouth widening optional flag) | ~36-48h |

All three tiers carry the v10 changes (mouth widening, age-keyed LR, death-age logging) **except** L4-baseline-faithful which keeps the original mouth = `[0]` setting unless we explicitly flag it on for a "true paper but with the catch fix" comparison run.

We considered an L1 "super fast" (600²/200/25 with hidden=16, epochs=3) but rejected it: at that scale the policy is so degraded that results may not transfer up the ladder, defeating the iteration purpose. The "smol" 600² scaffold was never validated for DDB+DDM stability either. **Cheaper than L2 should be done via shorter `total_steps` on L2, not a degraded geometry.**

**Next action: launch L2 (v10-fast).** The "appropriately telling, appropriately fast" tier — gives directional answers on the v10 changes within a single overnight cycle, which is the right cadence for early validation before committing to the L3 paper-comparable run.

### 15.20 v10 — three diagnostic analyses + revised L1/L2/L3 ladder + final knobs (2026-05-02)

Before locking the v10-L2 config from §15.19, ran three checkpoint-based analyses against v8 step-2.26M to validate the proposed knob cuts. Two of the original L2 cuts turned out to be wrong; one of the "knobs we didn't pull" turned out to be the most useful lever. The L2 spec was revised; an L1 tier was added.

Analysis script: `/tmp/v8_analysis/analysis.py` and `analysis_2_3.py` (one-shot, not committed). Methodology summarized below.

**Analysis #1 — counterfactual PPO.** Loaded the saved rollout buffer for the 5 oldest active predators. Ran one PPO update under L3 hyperparams (epochs=10, mb=256) and L2 hyperparams (epochs=5, mb=512). Compared resulting policies via symmetric KL on the same observations.

| Metric | Value | Interpretation |
|---|---|---|
| AVG KL(pre, L3) | 0.0030 | One full L3 PPO update is small (already-converged agents) |
| AVG KL(pre, L2) | 0.0019 | L2 update is ~64% the magnitude of L3 |
| AVG sym-KL(L3, L2) | 0.0010 | After both updates, policies are ~34% of L3-update-distance apart |

**Conclusion:** the L2 PPO cuts are *borderline* — at the cusp of "safe" (<25%) and "risky" (>50%). Acceptable for mature agents; can't bound effect on early-life agents (where it matters most for v10).

**Analysis #2 — SVD of policy W_in.** Decomposed the (205, 64) input layer of 10 active mature predators (top-aged) and 10 prey. Question: would `hidden=32` retain the meaningful capacity?

| | Predators | Prey |
|---|---|---|
| Effective rank @ 95% energy (median) | **53** | **52** |
| Top-32 SVs energy | 77.6% | 80.4% |
| Top-48 SVs energy | 92.4% | 93.3% |

The spectrum decays *smoothly* — there is no knee at any rank. Cutting to hidden=32 throws away ~22% of the policy's variance.

**Conclusion:** **hidden=32 is too aggressive for L2. Use hidden=48 (92% capacity preserved) or keep hidden=64.** This was the most surprising finding — directly contradicted §15.19's L2 spec.

**Analysis #3 — recent reward-rate vs age.** For each active agent, computed mean reward in their rollout buffer (last ~1024 steps of life). Plotted vs age.

- Pearson r(age, mean_reward) = **−0.08** (predators), **+0.02** (prey) — essentially zero.
- Predators in 5–20K age band: mean_r=0.31 → 100K+ band: mean_r=0.16. Older predators do *worse* on average.

**Conclusion:** within-lifetime PPO does not measurably improve hunt rate over the observed lifespan distribution. The within-lifetime training mismatch (§15.19 motivation #2) exists, but PPO updates aren't the rate-limiter. **Cuts to ppo_epochs/mb are *empirically* low-stakes** even though analysis #1 was borderline.

**Implications for v10-L2 spec:**
- Keep `policy_hidden_size = 48` (was 32 in §15.19) — analysis #2 forced this.
- Keep `ppo_epochs = 5` and `minibatch_size = 256` (instead of 512) — gives 20 grad-steps/fire (vs L3's 40) for safer middle ground.
- Surrender the speedup expectation from policy/PPO knobs; focus on population.

**Population is the dominant lever (CPU bench, scripts/bench_l2_vs_l3.py).** The `sim_step_core` cost scales as max_agents² for the contact matrix; PPO is only 1-3% of CPU wall-clock. So PPO cuts barely move the needle on total speed. The big lever is `prey_cap × predator_cap` — i.e., max_agents.

| Tier | max_agents | sim_step | eff sps | h/1M (CPU) | speedup vs L3 |
|---|---|---|---|---|---|
| L3 | 415 | 69ms | 14.0 | 19.8h | 1.00× |
| L2 | 335 | 47ms | 20.9 | 13.3h | **1.49×** |
| L1 | 220 | 28ms | 35.6 | 7.8h | **2.54×** |

**Revised tier ladder (final v10 spec):**

| Knob | L3 (axis1_residual) | **L2 (axis1_residual_fast)** | **L1 (axis1_residual_mini)** |
|---|---|---|---|
| `world_size` | 880 | **750** | 600 |
| `prey_cap` | 375 | **300** | 200 |
| `predator_cap` | 40 | **35** | 20 |
| `prey_initial` | 125 | 90 | 60 |
| `predator_initial` | 9 | 7 | 4 |
| `policy_hidden_size` | 64 | **48** | 32 |
| `n_physics_iter` | 5 | **4** (config-driven) | 3 |
| `rollout_steps` | 1024 | **1024** | 1024 |
| `minibatch_size` | 256 | **256** | 256 |
| `ppo_epochs` | 10 | **5** | 5 |
| `ddb_pred_threshold` | 10 | **10** | 8 |
| `ddb_prey_threshold` | 100 | **100** | 60 |
| `ddm_pred_threshold` | 10 | **10** | 8 |
| `ddb_max_boost` | 50 | **75** | 100 |
| `ddb_boost_distribution_alpha` | 0.5 | 0.5 | **0.3** |
| Mouth widening | flag-controlled | yes | yes |
| Age-keyed LR | flag-controlled | yes | yes |
| Death-age ring | flag-controlled | yes | yes |

**Threshold philosophy at smaller pop**: keep T values *high* (don't scale down proportionally with cap). The cost of an LV crash at small pop is *higher*, so the safety net should fire more aggressively, not less. Lever the breeding *rate* (`ddb_max_boost`) instead, which only fires at low pop and is safe at healthy pop.

**Why we stopped at L1 even though L2 only gives 49% speedup**: max_agents = 220 is already below the paper's natural-selection floor (~550 paper). Going smaller would produce dynamics that don't transfer up the ladder at all. L1 is the deepest defensible tier; L2 is the sweet spot of "real speedup + paper-shaped dynamics + intact policy/physics fidelity."

**Implementation work delivered alongside the configs:**
- `configs/axis1_residual_fast.yaml` (L2), `configs/axis1_residual_mini.yaml` (L1)
- `n_physics_iter` plumbed through config (was hardcoded constant) — `src/jax_sim.py:142`
- `policy_hidden_size = 48` validated to compile cleanly (Flax's Dense layer is shape-agnostic, no test changes needed)
- v10 mechanism additions (mouth widening flag-only, age-keyed LR, death-age ring buffer in SimState + lifecycle + progress logging)
- `scripts/bench_l2_vs_l3.py` for one-shot wall-clock A/B benchmarking on any device

**Deferred features (post-launch):** "rate-based alpha" — distributing the breeding boost on per-agent catch-rate or feed-rate instead of instantaneous energy. Real fitness signal but requires new SimState fields (`agent_lifetime_catches`, `agent_lifetime_feedings`) and checkpoint format bump. Will only revisit if v10 results show the energy-share alpha is the bottleneck.

**Next action:** check on v8/L3 (still running, currently ~22.9% complete with predator weights showing strong K&D-aligned signal: `pred_w_pred=−2.97, pred_w_prey=+4.37`); then launch L1 for cheap iteration on v10 mechanism wins.

### 15.21 v10 — three follow-up analyses on v8 (2026-05-02 evening)

After §15.20 locked the L1/L2/L3 spec, v8 progressed from step 2.26M → 2.40M (140K-step window). Three additional analyses on the v8 checkpoints to test whether v10's two main mechanism interventions (mouth widening, age-keyed LR) are still well-motivated. The headline: **one motivation got weaker, one got stronger, and the L2 spec ended up unchanged.**

Analysis script: `/tmp/v8_analysis/three_analyses.py` (one-shot, not committed).

**Mid-window status update.** Predator macros over the 140K window:
- Population: 11 → 9 (cap=40, never bound).
- Max age: **214,026 → 319,897** (same individual; +100K of life — strongly above §15.19's 95th-pct=101K).
- Cum catches: +1,170 in 140K (= 84/10K, healthy and steady).
- `w_pred=−2.97, w_prey=+4.37` — clean K&D-aligned signal (was the strongest evidence selection is working).

Prey weights also showed (slow, tail-driven) K&D-aligned drift: prey `mean(w_pred): −1.11 → −1.32` (~19% more avoidance-leaning). Median essentially flat (−1.18 → −1.12). Most-fearful prey doubled in magnitude (min −13 → −28). Consistent with selection pressure being real but slow at L3 timescales.

**Analysis A — Tactile-bin distribution of near-miss prey (one-snapshot proxy for mouth-widening lift).**

For each active predator, found all active prey within `buf × sum_radii`. Computed each prey's tactile-bin assignment (per `src/jax_food.py:149-154`). Question: what fraction sit in the v8 mouth `[0]` vs the v10 widened mouth `[0, 1, 17]`?

| Buffer | Near-pairs | In `[0]` | In `[0, 1, 17]` | Lift | Top bins |
|---|---|---|---|---|---|
| 1.5× sum_radii (≈36u, near-contact) | 12 | 0 (0%) | 0 (0%) | none | 7, 4, 8 |
| 2.5× | 29 | 0 | 0 | none | 7, 12, 14 |
| 4.0× (≈96u, approach range) | 88 | 3 (3.4%) | 4 (4.5%) | ×1.33 (+1 pair) | 12, 7, 14, 9, 8 |

**Surprise:** prey are **not** in the predator's front arc when close. Top occupied bins are 7-14 (sides + behind). Two possible explanations: (a) prey have learned to avoid the front arc — selection pressure showing up as positional behavior; (b) predators move past prey laterally rather than turning toward them. We can't disentangle from a single snapshot, but the directional read is unambiguous: at this stage of the run, the **mouth size is not the active bottleneck — orientation is**.

**Caveat:** snapshot misses the *moment of approach*. A rollout-buffer version (counting tactile-prey-channel firings in mouth bins vs not at moments of high signal) would be stronger. Deferred — not load-bearing for the L2 launch decision since the conclusion is "mouth widening is cheap, keep it, but expect modest lift," not "drop it."

**Analysis B — Survival by birth-cohort (lower-bound death-age via agent_id diff).**

Compared `agent_ids` between t_old=2.26M and t_new=2.40M. Predators alive only at t_old were dead by t_new; predators alive at both survived the 140K window. Birth-step computed as `step − age` from the t_old checkpoint.

Predator cohort survival rates:

| Birth window | n alive at t_old | survived 140K |
|---|---|---|
| 2.0–2.1M (oldest) | 4 | **25%** |
| 2.1–2.2M | 2 | 50% |
| 2.2–2.26M (most recent) | 5 | **0%** |

**Recent-birth predators are getting massacred.** 9 of 11 predators alive at t_old died in the 140K window (= 64/M-step death rate). Median lower-bound death-age was 35,907; the long tail (max=214K) is a few survivors carrying the population.

**Why this matters for v10.** §15.20 Analysis #3 (`r(age, mean_reward) = −0.08`) sampled top-aged predators — by definition, the surviving long-tail. The dying-young population was invisible to it. Cohort analysis B shows that population is the modal case, not an outlier. The aggressive age-keyed LR schedule (`1e-3 → 3e-4` over 30K steps) is exactly tuned for predators that die in the 5K–30K age band.

**Implication:** an earlier candidate to relax — "drop age-keyed LR at L2 since §15.20 Analysis #3 showed PPO updates don't drive reward in mature predators" — is **withdrawn**. We can't generalize the mature-predator finding to the dying-young population, which is who the schedule targets. Keep it on at L2.

**Analysis C — Catch-rate vs age (per-predator spike-detection on rollout rewards).**

For each active predator, counted `reward[t] > mean + 3σ` events in their last-rollout window as a proxy for catch events. Pearson r(age, spike_rate) = +0.106; r(age, mean_reward) = −0.168 (cross-checks §15.20's −0.08 directionally, magnitude noisier with n=9).

| Age band | n | mean spike rate /1k step | mean reward |
|---|---|---|---|
| 0 – 20K | 4 | 0.49 | +0.068 |
| 60K – 150K | 3 | 0.65 | +0.113 |
| 150K – 1M | 2 | 0.49 | −0.008 |

Underpowered with n=9 — spike rate quantizes to {0, 0.98, 1.95}/1k. Directional read: weak hint that mid-age predators catch slightly more but the very-old ones see reward fall (their genome's `w_eat` may have drifted negative — possible explanation for the long-lived predator's mean_reward = -0.19 outlier). Not load-bearing for L2 design; flagged as a possible follow-up if multi-checkpoint catch tracking gets added.

**Net L2 spec update (post-Analysis A/B/C):** **none — keep L2 as locked in §15.20.** The two changes that were on the table to relax were `ddb_max_boost: 75 → 60` and disabling age-keyed LR at L2. Both withdrawn:
- `ddb_max_boost`: breeding rate is one of the most mechanistically clean LV-stabilizing levers (only fires at low pop) and Analysis B's high turnover (64 predator deaths/M-step) confirms scaffolds are still load-bearing. The 75 vs 50 difference only matters when pred-pop ≤ 10, which is exactly when it should fire harder.
- Age-keyed LR: motivation was *strengthened* by Analysis B, not weakened.

**The one knob worth tuning *if* L2 results suggest it** (parametrically, not via the boolean flag): the schedule values themselves. The schedule is already dynamic via `lr_schedule_initial` / `lr_schedule_final` / `lr_schedule_decay_steps` — setting initial=final disables it without touching the flag. Gentler variants for future ablation: `5e-4 → 3e-4 over 30K` (1.67× boost vs current 3.3×), or `1e-3 → 3e-4 over 60K` (same magnitude, slower decay). Don't apply these proactively; only if L2 + age-LR shows policy instability.

**Next action:** wait for v8 to reach ~3M steps before launching L2 (gives one more checkpoint diff to validate the cohort survival pattern and confirm `w_pred` keeps deepening). Then launch L1 first as the cheap-iteration tier per §15.20's plan.

### 15.22 v8 extinction post-mortem + DDB/DDM retune (2026-05-03)

v8 (axis-1 residual on L3) went extinct between step 3.7M and 3.8M. Predator pop trajectory: 12 (3.50M) → 6 (3.60M) → 7 (3.70M) → **0** (3.80M). The system has been frozen since — prey at cap=375 with mean energy 427, no births, no deaths. This is the second post-§15-reset extinction (v7 §15.15 was the first), now happening on the run that was supposed to "fix" v7.

**What killed the run.** Cohort trajectory across `step_03500000.npz` → `step_04000000.npz` (one-time analysis at `/tmp/v8_analysis/bracket_extinction.py`):

| step | n_pred | n_prey | pred E (min/med/max) | catches added |
|------|--------|--------|----------------------|---------------|
| 3.50M | 12 | 283 | 5.8 / 68.1 / 122.1 | — |
| 3.60M |  6 | 215 | 24.8 / 35.3 / 50.1 | +701 |
| 3.70M |  7 | 221 | 19.8 / 33.2 / 60.4 | +619 |
| 3.80M |  0 | 360 | — | +472 |

The crash window was 100K steps (3.70M → 3.80M). Pop fell from 7 to 0 with median predator energy already at 33 — well below the breeding cliff edge.

**Why DDB+DDM didn't rescue it — the sigmoid math.** Given `P_birth = kappa_eff / (1 + exp(zeta_eff − β·E))` with β=0.4, zeta_b_pred=100, kappa_b=1e-3:

The "cliff edge" of the breeding sigmoid sits at `E = zeta_eff / β`. With T=10 (the v8 setting), at the death-spiral pop of 7 the cliff was at:
- factor = 49/(49+100) = 0.329
- zeta_eff = 100 × 0.329 = 32.9
- cliff edge = 32.9 / 0.4 = **82** energy

Median predator energy was 33 — almost 50 below the cliff. Even with the kappa boost (3× at this pop, capped at 50× by `ddb_max_boost`), the sigmoid floored P_birth at ~1e−10 per step. **`ddb_max_boost` was multiplying a vanishing sigmoid, not overcoming it.**

The error in §15.20's threshold philosophy: keeping T high "to fire more aggressively at small pop" was the right intuition, but T=10 was already too low — the scaffold doesn't meaningfully engage until pop ≤ 3, by which time energies have already collapsed. The recovery window had closed before DDB could help.

**Conceptual fix (which lever does what):**

| Lever | What it controls | Operating regime |
|---|---|---|
| `T` (DDB threshold) | *When* the scaffold engages — half-engagement at pop=T | The main lever. T=10 misses pop=7; T=20 catches it. |
| `zeta_eff = zeta · factor` | Where the sigmoid cliff sits | Slides smoothly with factor. |
| `kappa_eff = kappa / factor` | Max breeding rate when sigmoid saturates | Normally only 1.5–5× at moderate pops; only relevant at pop ≤ 2. |
| ~~`ddb_max_boost`~~ | Cap on the rate boost | **Removed in this section** — it was an arbitrary clip on the smooth `1/factor` curve. Doesn't bite at the operating point and doesn't help recovery. |
| `ddb_floor` | Floor on the factor | Almost never relevant. Stays 0. |

**The retune (§15.22):**

1. **T bumps, tier-graduated:**

| Tier | T_pred (was→is) | T_prey (was→is) | T_ddm_pred | T_ddm_prey (NEW) |
|------|---|---|---|---|
| L3 | 10 → **20** | 100 → **200** | 10 → **20** | **200** |
| L2 | 10 → **17** | 100 → **170** | 10 → **17** | **170** |
| L1 |  8 → **12** |  60 → **120** |  8 → **12** | **120** |

T_pred = 20 (L3) catches the death spiral: at pop=7, zeta_eff = 100 × 0.109 = 10.9, cliff edge at E=27 — *below* the actual energy of survivors. They breed.

2. **DDM extended to prey symmetrically.** `ddm_prey_threshold` is a new config key that scales `c_b` (prey passive metabolic cost) by `factor(N_prey, T_ddm_prey)`. Same formula as the predator side. Closes the asymmetry that DDM was predator-only — there was no principled reason for that.

3. **`ddb_max_boost` removed as a cap.** Replaced with the natural floor `factor ≥ kappa_b`, which gives `P_birth ≤ 1` for any sensible config without clipping the recovery curve. At integer pops with sane T, the natural floor never bites — the rate is just `kappa_b / factor` smoothly. Legacy configs that still set `ddb_max_boost` are silently ignored. (Note: not a literal "remove the cap" — there's still a kappa_b floor, but it's at 1/kappa_b ≈ 1000× and serves the math, not arbitrary tuning.)

4. **Code changes:** `src/jax_lifecycle.py` `_batch_birth_prob_jax` removes the gate-on-max_boost block; `update_energies_jax` now requires `prey_count` and applies the symmetric DDM. Caller in `src/jax_sim.py` passes both counts. Tests updated in `tests/test_ddb_ddm.py`.

**What this does NOT change:**

- The DDB+DDM mechanism itself. Same `factor(N) = N²/(N²+T²)` saturation curve. The fix is a tuning + scope (DDM-for-prey) change, not a formula change. Path A in spirit; only "Path B" element is the symmetry extension and removal of the arbitrary cap.
- `ddb_boost_distribution_alpha` (the energy-weighted-share parameter). L1 stays at 0.3, L2/L3 at 0.5. With higher T, the alpha redistribution engages at higher pops too — but the share formula is unchanged.
- The v10 mechanism additions (mouth widening, age-keyed LR, death-age ring). All independent of this retune.

**What we're explicitly *not* doing here:**

- *Hardcoded extinction floor* (parthenogenesis at pop=1, genome-bank for pop=0). The proposal was discussed; rejected on the principle that the existing levers should work if tuned correctly. T=20 should obviate the need. If T=20 still produces a death spiral, hardcoded floors come back on the table.
- *Linear interpolation rate-boost formula* (`kappa × (1 + (max_boost−1) × (1−factor))`). Considered as Path B; rejected — at moderate pops it's wildly more aggressive than `1/factor` (21× at pop=12 vs the desired ~2×). Current `1/factor` shape is what we want.

**Validation plan before relaunch:**

- Run the test suite (266 fast tests pass, including 14 in `test_ddb_ddm.py` covering the new behavior). Done.
- Local 10K-step smoke on Mac CPU before any cloud launch. (Pending — to be run before next launch.)
- Watch for the failure mode in the first 200K steps: if predator pop oscillates above ~10 and energies stay > 50, T=20 is doing its job. If pop drops below 7 with energies < 30 (the death-spiral signature), the retune wasn't enough.

**Naming cleanup deferred** to a follow-up commit. The `ddb_*` and `ddm_*` config keys carry awkward project-internal names; better candidates are `density_breeding_threshold_*`, `density_metabolism_threshold_*`, etc. Done as a separate clarity pass to keep this commit focused.

**Next action:** the v8 process is dead but still occupying the GPU VM doing nothing (looping with prey at cap, no births/deaths). Stop it before relaunching anything. Then local smoke on the new L1/L2/L3 configs, then GCS launch.

### 15.23 Config-key rename for the density-dependent scaffolds (2026-05-03)

The `ddb_*` and `ddm_*` config-key prefixes are project-internal acronyms that aren't self-explanatory to a reader. Renamed to descriptive keys; new names tell you what the knob does without needing the §15.x story. Old names still read as fallback for backward compat with archived configs.

| Old key | New key | What it controls |
|---------|---------|------------------|
| `ddb_pred_threshold` | `density_breeding_threshold_pred` | Pop at which DDB engages at half-strength (predator) |
| `ddb_prey_threshold` | `density_breeding_threshold_prey` | Pop at which DDB engages at half-strength (prey) |
| `ddm_pred_threshold` | `density_metabolism_threshold_pred` | Pop at which DDM engages at half-strength (predator) |
| `ddm_prey_threshold` | `density_metabolism_threshold_prey` | Pop at which DDM engages at half-strength (prey) |
| `ddb_floor` / `ddm_floor` | `density_factor_floor` | Minimum value of the saturation factor |
| `ddb_boost_distribution_alpha` | `breeding_share_alpha` | Power-law exponent for energy-weighted boost share |

**What stayed the same:**
- Paper-faithful Greek names (`kappa_b`, `beta_b`, `zeta_b_pred`, `zeta_b_prey`) — these match K&D's notation and changing them would create unnecessary documentation drift.
- `stability_mechanism` — already self-explanatory.
- `ddb_max_boost` (removed in §15.22, not renamed).
- The legacy `ddb_boost_distribution` string knob ("uniform" / "energy_weighted") — still works, still maps to alpha.

**Backward compat strategy.** All readers in `src/jax_lifecycle.py` use a chain of `config.get(NEW, config.get(OLD, default))`. New configs use new names; archived configs keep working. Test `test_new_config_keys_match_legacy_keys` in `tests/test_ddb_ddm.py` pins the equivalence (asserts birth probabilities are identical when only the key names differ).

**What this enables.** Reading a fresh config (e.g., when handing off to a collaborator) no longer requires the §15.x history to understand. `density_breeding_threshold_pred: 20` is interpretable on its own; `ddb_pred_threshold: 20` is not.

**Configs updated:** L1/L2/L3 (axis1_residual_mini, axis1_residual_fast, axis1_residual). Other configs in `configs/` and `configs/archive/` keep the legacy names and continue to work. Tests in `tests/test_ddb_ddm.py` keep using legacy names (validates fallback path).

### 15.24 Axis-2 obs redesign — approach-angle + speed; configs reorg (2026-05-03)

While propagating §15.22 to axis 2 we caught the axis-2 observation encoding doing something less direct than we'd thought. Replacing it with a more direct encoding, plus reorganizing the configs into a per-axis × per-tier layout while we were at it.

**The encoding problem.** Old `proximity_encoding: "distance_and_heading"` reported `sin(α), cos(α)` per bin per species where `α = other_heading − my_heading` — i.e., the egocentric *body orientation* of the closest agent in that bin. To answer "is this thing moving toward me or away from me," the policy had to *combine* (bin position) + (body orientation in my frame). The information was there, but indirectly. Walking through it concretely: I face east, a prey sits east of me. If they face west, that's "approaching me" — represented as `(sin, cos) = (0, −1)`. If I rotate to face north and they're still east of me but still facing west, the encoding changes to `(1, 0)` — same physical scenario, different signature. The agent has to learn the joint compositional rule.

**The new encoding.** `proximity_encoding: "distance_approach_speed"` (default for all axis-2 configs from now on):

```
α = other_heading − bearing_from_other_to_me     # NOT − my_heading
sin_approach, cos_approach = sin(α), cos(α)
speed = |velocity_xy| of the closest agent
```

Now `cos_approach = +1` directly encodes "moving toward me" regardless of my own orientation; `cos_approach = −1` directly encodes "moving away from me"; `sin_approach = ±1` directly encodes "moving perpendicular to the bearing line." No policy composition needed. Plus the speed channel disambiguates "stationary fearful prey" from "fleeing-fast prey" — they were identical in the old encoding (both have whatever heading they're holding).

Per bin per species: 4 channels (was 3). Total per bin including food + wall: **10** (was 8). With 32 bins + 72 tactile + 5 self-state, `obs_dim = 397` (was 333). About 19% growth, manageable at our current `policy_hidden_size = 64`.

The legacy `"distance_and_heading"` encoding is preserved in the code path for backward compat with archived configs/runs. Only new axis-2 + axis-12 configs use the new default.

**Configs reorg.** Took the opportunity to reorganize `configs/` while making changes anyway. Old layout was inconsistent (mini/fast/(none) suffix for axis1, `_smol` for axis2, no axis12). New layout:

```
configs/
    axis1/{tiny,small,med,full}.yaml
    axis2/{tiny,small,med,full}.yaml
    axis12/{tiny,small,med,full}.yaml
    archive/...
    runtime/...
    baseline_faithful.yaml
```

Tier names are size-progression (`tiny`/`small`/`med`/`full` map roughly to L1/L2/L3/L4 from prior notation). Each axis directory has a `README.md` explaining the mechanism and tier table. Existing runs that referenced old paths (e.g., `axis1_residual.yaml` for v8) were renamed via `git mv` so blame/history is preserved.

Naming also tightened in `experiment_name` field:
- `axis1` modifier was `residual` → now `residual_reward_mlp` (clearer that it's the reward genome, not the policy)
- `axis2` modifier was `aligned` → now `social_heading` (clearer that it's the social/proximity obs)
- `axis12` modifier is `residual_reward_mlp_social_heading` (verbose but explicit; the GCS run path encodes both axes)

Folder names stay `axis1`/`axis2`/`axis12` since "axis" vocabulary is established throughout findings.md and project conversations.

**What stayed the same:**
- The DDB+DDM scaffolds and their `density_*` config keys (§15.22 / §15.23 unchanged).
- The v10 mechanism additions (mouth widening, age-keyed LR, death-age ring) — all carry through to every tier of every axis.
- `baseline_faithful.yaml` — kept untouched as the K&D-pure reference.

**Implementation:**
- `src/observations.py` adds `_per_channel_encoding_with_speed` and extends `_single_proximity_agents_with_heading` with `use_approach_angle`/`include_speed` static booleans. `compute_all_observations` parses the new encoding name and routes through the right path.
- All 12 new/updated tier configs use `proximity_encoding: "distance_approach_speed"` (axis 2 + axis 12) or omit (axis 1 keeps default `"distance_only"`).
- 267 fast tests pass; backward compat with `"distance_and_heading"` exercised by archived configs that still use it.

**What is *not* validated yet:**
- Whether the more-direct encoding actually accelerates evolution vs the indirect one. That's the empirical question axis 2 answers. We'd need a controlled comparison — same seed, same scaffolds, only `proximity_encoding` differs. Could be a follow-up axis-2 ablation.
- Whether the speed channel matters independently of the approach-angle math. Could split the `distance_approach_speed` encoding into two for the ablation: approach without speed, and full. Deferred.

**Next action:** launch axis1/tiny on GCP (the v8-extinction-driven retune wants empirical validation) and pair with axis2/tiny and axis12/tiny if there's GPU budget. Don't relaunch v8.

### 15.25 v8 axis1 deep post-mortem — predators got smart, then starved (2026-05-03)

§15.22 identified the breeding cliff as the proximate failure. This section is a fuller forensic pass on the v8 axis1 run (`gs://evo-reward-ckpts/results/axis1_residual/seed_0/2026-05-01T2203Z`, 478 metric points spanning step 10K → 4.78M, plus 25 sampled checkpoints across the lifecycle). The headline: **evolution worked. Predators got measurably smarter for 3M steps. They went extinct anyway, because their reproductive cliff sat above the energy levels they could actually attain.** Two pieces of folklore I'd been carrying are also corrected here.

**Predator evolution trajectory** (per-checkpoint inspection, all 25 sampled steps; abbreviated):

| step | n | E_max | w_act | w_prey | w_pred | residual L1 | age_median | age_max |
|------|---|-------|-------|--------|--------|-------------|------------|---------|
| 500K | 13 | 124 | −1.79 | +0.38 | −2.19 | 2.97 | 49K | 376K |
| 1M | 10 | 90 | +0.50 | +1.66 | −2.07 | 4.17 | 75K | 437K |
| 2M | 11 | 73 | +7.93 | +4.55 | −2.54 | 5.05 | 66K | 121K |
| 3M | 8 | 83 | **+11.66** | **+6.71** | −2.83 | 6.60 | 32K | 82K |
| 3.5M | 12 | 122 | +10.10 | +6.94 | −2.87 | 7.00 | 16K | **490K** |
| 3.7M | 7 | 60 | +9.95 | +7.39 | −1.08 | 6.62 | 30K | **690K** |
| 3.76M | 5 | 38 | +10.13 | +7.17 | −2.02 | 6.35 | 6K | 171K |

The reward-genome means show clear directed evolution: `w_act` swung −1.8 → +12 (predators *enjoy* moving), `w_prey` 0.4 → +7.4 (predators *enjoy* being near prey). Residual MLP L1 norm tripled from 2.97 → 7.00. By 3M, the cohort had a converged "active hunter" profile and the residual was carrying real signal. **Through the entire 100K-step death window (3.7M → 3.8M), the *surviving* predators still had this fully-evolved genome** — they didn't get dumber. They just got fewer.

Prey were stressed throughout the late phase: prey mean energy held at 15-18 from 1M onward despite N_prey ~ 250-360. The moment predators went extinct (3.80M), prey energy released: 25 → 167 → 297 → 403 → 497 → 572 (final). So predators were *effective* hunters right up to extinction; they weren't dying because they couldn't catch prey on average.

**The age² death hazard does not exist.** The hazard formula in [src/lifecycle.py:26](src/lifecycle.py#L26) and [src/jax_lifecycle.py:143](src/jax_lifecycle.py#L143) is the K&D Gompertz form:

```
h(t, e) = kappa_h · [1 − 1/(1 + alpha_e · exp(−beta_h · E))] · alpha_t · exp(beta_t · age)
```

Both terms are *exponential* in their argument, multiplied. Earlier writeups (and one SMS to Gil) called the age term `age²`. That was wrong — it's `exp(beta_t · age)`, Gompertz mortality. Also wrong: that age was the dominant death driver. Concrete numbers at v8's `alpha_t_pred=2e-7`, `beta_t_pred=4e-6`:

| operating point | hazard / step | per 100K steps |
|---|---|---|
| young + healthy (E=80, age=0) | 4.5e−18 | 4.5e−13 |
| young + low-energy (E=20, age=0) | 7.3e−13 | 7.3e−8 |
| Methuselah + low-energy (E=20, age=690K) | 1.2e−11 | 1.2e−6 |
| starving (E=1, age=10K) | 3.4e−11 | 3.4e−6 |
| 5M-yr-old + low-energy (E=20, 5M) | 3.6e−4 | ~certainty |

The 690K-old Methuselah in v8 had an age-multiplier of `exp(2.76) ≈ 16×` baseline. Multiplicatively meaningful, but `16 × 7e−13 ≈ 1e−11` per step is still vanishing — cumulative death-by-hazard probability over 100K steps is ~1 in 10⁶. The hazard formula in our parameter regime essentially *never fires*. Both terms hover near their floors.

**So how do predators actually die?** [src/jax_lifecycle.py:366](src/jax_lifecycle.py#L366):

```python
dead_mask = sim_state.is_active & ((sim_state.energies < 0) | (death_randoms < h_all))
```

Two paths — but in our regime, path 1 (`energies < 0` → instant death) handles essentially all deaths. Predators starve. The hazard sigmoid is dormant.

**Policy is per-individual and reset at birth.** [src/jax_evolution.py:154-155](src/jax_evolution.py#L154):

```python
# Child policy: fresh initialization
child_params, child_opt = init_policy(k4, config)
```

Every offspring inherits the (mutated) reward genome but receives a **brand-new PPO policy**. The motor learning — chase trajectories, turn timing, the actual hunting craft — dies with the parent. This is one of the most consequential implementation details in the whole sim. It means:

- A predator's ability to *catch prey* is a function of its lifetime PPO updates, not its inheritance.
- The reward genome encodes "what to want" (chasing, eating). PPO encodes "how to act." Only one of these is heritable.
- Genome convergence (which v8 had: `w_pred std` 0.67 → 0.49 by 3.76M) makes things worse — when veterans die, the freshly-initialized offspring have no diversity in reward signal either, so PPO gradients across the new cohort are highly correlated. They all explore the same way, all fail the same way.

**The Methuselah pipeline — and the PPO-experience accounting that makes it fragile.** Tracking `age_max` across the 25 sampled checkpoints reveals that the v8 cohort never had a *bench* of veterans — only **one Methuselah at a time**:

| step | n_pred | age_max | age_median | Methuselah lineage |
|---|---|---|---|---|
| 500K | 13 | 376K | 49K | M#1 (born ~step 124K) |
| 1M | 10 | 437K | 75K | M#2 (born ~step 563K) — M#1 already died |
| 2M | 11 | **121K** | 66K | M#3 (born ~step 1.88M) — M#2 already died |
| 3M | 8 | **82K** | 32K | M#4 (born ~step 2.92M) — M#3 already died |
| 3.5M | 12 | 490K | 16K | M#5 (born ~step 3.01M) |
| 3.7M | 7 | **690K** | 30K | same M#5, still alive |
| 3.76M | 5 | 171K | 6K | M#6 (born ~step 3.59M) — M#5 already died |

Methuselahs cycle through the run — each lives 300K-700K steps, dies *also from `E<0`* (even great hunters can't outrun a long bad-catch stretch), then a new individual slowly accumulates age. The cohort is small enough (n=7-13) that the reservoir of "next-oldest" is always shallow. When a Methuselah dies during a bad-luck window, the rest of the >30K-age cohort tends to die in the *same* window (same prey-encounter stochasticity), and the freshly-born juveniles inherit the genome but have to start motor learning from zero.

The lifetime-PPO-update count is the key quantity. Per-agent PPO fires every `rollout_steps` (1024 in v8/v10), so:

| individual lifetime | PPO updates | with `ppo_epochs=10` (v8 L3) |
|---|---|---|
| Methuselah (690K steps) | ~674 | ~6.7K gradient passes |
| veteran (100K steps) | ~98 | ~980 |
| typical adult (30K steps) | ~29 | ~290 |
| juvenile (6K steps — median at extinction) | ~6 | ~60 |

**Methuselah had ~100× more gradient signal than the offspring inheriting his (mutated) genome.** When he died, the survivors had single-digit PPO fires trying to catch prey through a stochastically thinned prey field. They couldn't. Pop went 5 → 3 → 0 in a 30K-step span.

This also recasts the v10 age-keyed LR schedule introduced in §15.19/§15.20 (`lr_schedule_initial: 1e-3 → lr_schedule_final: 3e-4 over 30K steps`). v8 ran with constant LR throughout. The v10 schedule front-loads gradient magnitude into each agent's first 30K steps — exactly the band where a juvenile is otherwise locked out by tiny update counts. It doesn't fix the fresh-policy-at-birth problem, but it shortens the time a new cohort needs to become functional hunters. Combined with §15.22's T=20 (which lets more births fire while veterans are still alive), the goal is to move from "1 Methuselah holding the population up" to "a few overlapping veterans + faster-maturing juveniles."

**The unified extinction narrative:**

1. Predators evolved a "smart hunter" genome (positive `w_act`, positive `w_prey`, negative `w_pred`) by 2M-3M steps.
2. Long-lived individuals' PPO policies converged on effective hunting (residual L1 norm 6-7, prey energy depressed throughout).
3. **But** the operating energy distribution (predator E hovering 30-80, max 120) sat below the v8 breeding cliff (E=82 at pop=7 with T=10). Births were rare luck events.
4. Pop ratcheted down 13 → 10 → 8 → 7 across 3M steps despite occasional births. Genome diversity collapsed in parallel (low-pop selection bottleneck).
5. At 3.7M-3.76M, a bad catch window dropped E_max from 122 → 38. Maintenance metabolism (`d_b·t`) drained the surviving predators' energies. They starved — not from age-hazard, but from `E < 0`.
6. Each starving veteran took its trained policy with it. The few births happening in the final window produced freshly-initialized juveniles who had no chance of catching enough prey to survive *or* breed.
7. By 3.79M only 2 predators left, both with E < 3. Extinction at 3.80M.

**Implications for the §15.22 fix (T=10 → T=20).** Drops the breeding cliff from E=82 (unreachable) to E=27 (well below typical operating energy) and simultaneously cuts metabolic bleed 3× at the spiral point — both effects ride the same `N²/(N²+T²)` factor curve. The reframing from this section: the deeper reason it works isn't just "births fire" but "**T=20 keeps the experienced cohort alive long enough that the population always carries enough PPO-trained hunters to ride out a bad-catch window.**" v8's failure mode was being reduced to single-digit-PPO-update juveniles holding the population alone; T=20 should prevent that by making pop=7 a routine recoverable state rather than a one-Methuselah-from-extinction state.

**Implications for "make them live longer" as a separate intervention** (the conversation with Gil that motivated this analysis). The intuition is sound — long lifespan is structurally load-bearing because PPO is per-individual. But the right knob is *not* `beta_t_pred`. The age-hazard is already at the floor; halving or zeroing `beta_t_pred` changes essentially nothing (Methuselah goes from 1.2e−11/step to 6e−13/step — both negligible vs the `E<0` cliff). The right knobs to extend predator lifespan run roughly in this order:

1. **Push T higher still.** `predator_d_b` is *already* DDM-scaled by `N²/(N²+T²)` ([src/jax_lifecycle.py:107](src/jax_lifecycle.py#L107)). Bumping T from 10→20 doesn't only lower the breeding cliff — at the spiral operating point of pop=7 it also cuts metabolic bleed from 1.32e−3/step to 0.44e−3/step (3× reduction). DDB and DDM share the same factor curve, so a single T-bump moves both. With current §15.22 T=20 we expect both effects already.
2. **Raise `predator_eta`** (energy yield per catch). Pumps energy *in* faster — complements DDM pumping less *out*. Doesn't break species-pop independence; clean knob.
3. **Lower `density_factor_floor` to 0** (currently 0.3 in some configs). Lets `d_b_eff → 0` at extreme low pop, giving lone survivors maximum runway between catches. Cheap to try.
4. **Lower the `predator_d_b` baseline (4e-3 → e.g. 2e-3).** Last-resort — globally cheaper metabolism breaks K&D fidelity but is the most direct longevity knob if symmetric-T reductions aren't enough.

Note: `predator_d_a` (action energy cost, 5e-5 × ‖action‖) is *not* DDM-scaled, but it's ~30× smaller than even the floored `d_b_eff` at the spiral point, so it's not the rate-limiter and not worth scaling.

We shouldn't pull any of these yet — axis1/tiny with the §15.22 T=20 retune is the cleanest first ablation. T=20 *should* obviate the need (it cuts metabolism 3× and drops the breeding cliff to E=27 simultaneously). But if it doesn't, the order above is the priority list, with the most paper-faithful knobs first.

**What this updates in our mental model:**

| Belief | Status |
|---|---|
| Death hazard is `α + β·age²` | ❌ — it's Gompertz `α · exp(β·age)`, multiplied by an energy term |
| Old predators die from age | ❌ — in our regime, age hazard is ~1e−11/step at any age below ~5M |
| Predators went extinct because they couldn't learn to hunt | ❌ — they hunted effectively until the last 100K steps; prey energy was depressed throughout |
| Predators went extinct because the breeding cliff was too high | ✓ — directly confirmed by inspecting energy distributions vs `zeta_eff/β` |
| Long-lived predators are structurally important | ✓ — confirmed by the "policy fresh at birth" code path and the cohort-survival data showing veteran PPO policies are non-fungible |
| The cohort had a "bench" of multiple experienced hunters | ❌ — there was only one Methuselah at a time. Pipeline failure mode: when the active Methuselah died in a bad-catch window, the >30K-age cohort tended to die in the same window, leaving only juveniles |
| v10's age-keyed LR schedule (added in §15.19/§15.20) is just a stabilization hack | ❌ — it specifically targets the "single-digit PPO fires per juvenile" failure mode. Front-loads gradient magnitude into the band where lifetime updates are scarce. Shortens the maturation gap between Methuselah-death and offspring-becoming-functional |
| `ddb_max_boost` was the relevant knob | ❌ — it was clipping a smooth `1/factor` curve nowhere near the death-spiral operating point. Removed in §15.22. |

**Process notes** (for future post-mortems):

- `metrics.npz` (the time-series of `prey_population`, `predator_population`, `*_mean_energy`, `*_mean_w_*`, `*_std_w_*`) is the **first** thing to look at on any extincted run. It's compact (~40 KB), gives 478-point coverage of the whole run, and immediately localizes the spiral window without needing to download checkpoints.
- `progress.json` at the run root carries final cumulative counters (`cum_catches`, `cum_deaths`, `next_agent_id`) and final population. Read first.
- For deep cohort-level analysis, `scripts/inspect_checkpoint.py` against a 3-5 checkpoint sequence covering the spiral window is sufficient. The diff-mode (two checkpoints) shows cohort survival cleanly.
- Don't repeat my mistakes: verify the actual code before writing a story about it. The "age²" detail came from misremembering the K&D paper without re-reading [src/lifecycle.py:53](src/lifecycle.py#L53).

**Next action (no change to plan):** axis1/tiny is running on GCP (run tag `2026-05-03T2017Z` on `evo-reward-gpu`). At step 10K it had pred=16, predE healthy, w_pred mean already drifting +0.40 — early evolution active. Watch for whether the v8 ratcheting pattern (steady predator pop decline despite births) repeats; if predator pop holds above ~10 with E_max > 50 through 1M-2M steps, the §15.22 retune is doing its job.

### 15.26 axis1/tiny extinction at 60K → cold-start scaffolds (2026-05-04)

The §15.25 axis1/tiny run (`2026-05-03T2017Z`) finished its predator population at step **~60K** — far worse than v8's 4M. metrics.npz pulled from GCS (340 points, 0 → 3.4M):

| step | pred | predE | pw_eat |
|---|---|---|---|
| 10K | 16 | 80.6 | −0.08 |
| 30K | 13 | 53.1 | −0.23 |
| 50K | 4 | 8.6 | +0.05 |
| 60K | **0** | 0 | — |

For the next 3.34M steps the prey-only economy ran with prey at cap (200) and mean E ramping from 67 → 482. Cumulative final state: 563 catches, 855 total agents born (vs v8's 29571 / 31123 — two orders of magnitude less ecology).

The configs are not directly comparable to v8 (axis1/tiny is L1 — `world=600`, `prey_cap=200`, `predator_cap=20`, `predator_initial=4`; v8 was L3 — `world=960`, `predator_cap=50`, `predator_initial=11`). The §15.22 T retune is also tier-graduated (L1 uses T=12, not T=20). But the *qualitative* failure is the same one §15.25 diagnosed in v8: a cold-start cohort starves before it learns to hunt. L1 just collapses faster because:

1. **4 starting predators with random policy + ~10K-step lifetime ⇒ ~10 PPO updates per individual.** Two PPO steps is nothing; the founder cohort dies still random.
2. **Per-agent fresh policy at birth.** Offspring inherit the (mutated) reward genome but get a brand-new MLP — confirmed at [src/jax_evolution.py:154-155](src/jax_evolution.py#L154-L155). Every newborn restarts from random.
3. **Mean-field collapse.** Same hunting environment for everyone; no genome variation in hunting yet (random policy = random hunting); E drops in lockstep with N.

**Why the existing scaffolds didn't bite.** DDB/DDM relieve density-based suppression but *not* energy-based suppression — and the energy-based gate is what's actually firing in a cold-start collapse:

| step | N | factor | DDM (`d_b_eff/d_b`) | E-cliff = `zeta·factor/β` | predE | E/cliff |
|---|---|---|---|---|---|---|
| 10K | 16 | 0.64 | 0.64× | 160 | 80 | 0.50 |
| 30K | 13 | 0.54 | 0.54× | 135 | 53 | 0.39 |
| 50K | 4 | 0.10 | 0.10× | 25 | 8.6 | 0.34 |

The cliff lowers as N drops, but predE drops in lockstep — `E/E_cliff` stays roughly constant the whole way down (~0.35). Selection effects (low-E individuals dying first → survivors having higher E) don't kick in because random-policy predators all hunt equally badly. By the time scaffold relief is large, predators are already starving below the floor it lowered to.

So the scaffolds correctly handle a stable population riding density swings (the v8 case) but not a population that hasn't bootstrapped hunting at all.

**Three fixes added** (commit `cae5e35`):

1. **`predator_initial` bumps across tiers** — L1: 4→10, L2: 9→20, L3: 11→30. More starters means each individual lifetime contributes a smaller fraction of the cohort's PPO budget, but the *cohort's* aggregate updates over its lifespan dominate, and at ~30 starters losing one to bad luck is a survivable shock rather than a 25% population blow.
2. **`predator_e_initial` 100→150** — gives the founder cohort ~50% more steps before the first individuals hit `E<0`. Per-species initial energy was unwired in the JAX path; [src/jax_state.py:194-200](src/jax_state.py#L194-L200) now reads `prey_e_initial` and `predator_e_initial` separately rather than the single `initial_energy` (which no config sets).
3. **Emergency breeding clause** — when own-species count drops below `emergency_breeding_n_*`, `P_birth` is floored at `kappa_b · max(0, 1 − N/N_em)` regardless of the energy gate. Implemented at [src/jax_lifecycle.py:295-318](src/jax_lifecycle.py#L295-L318) inside `_batch_birth_prob_jax`. Default disabled (N_em=0); tier configs set `emergency_breeding_n_pred=3, emergency_breeding_n_prey=10`. This decouples low-N breeding from the energy gate that DDB does *not* relieve, giving a cohort about to extinct one last shot at recovery even when starvation has crashed mean E below the cliff. The tail is small and rate-limited (max kappa_b = 1e-3/agent/step at N=0).

These are tier-config-keyed knobs, not changes to the K&D math at any operating point above N_emergency. For populations stably above the threshold the formula is bit-identical.

**Local smoke (axis1/tiny, 10K steps on Mac):** predators climbed from 10 → 20 (cap) within ~5K steps, sat at cap with median E ~125, regular catches+births. No cold-start collapse. Three orders of magnitude better than the L1 run we just killed.

**L2 launch:** axis1/med now running on `evo-reward-gpu` (us-west1-a), tmux session `axis1med`, run tag `2026-05-04T0714Z`. At step 870K (~8.5%, 8.2h elapsed):
- pred 17-19, prey 290-316, predE median 50-61
- Δcatches/10K: 100-130. Δbirths/10K: 100-130. Replays uploading.
- Reward weights actively evolving: pred_w_act → −5.4 (action aversion), pred_w_pred → +5.4 (clustering), pred_w_eat → +0.4-0.7. Larger swings than v8 saw — possibly because the surviving cohort is bigger, possibly drift; clip is at 100 so safe.

**What's still uncertain.** Whether the larger reward-weight swings reflect authentic learning or scaffold-enabled drift; whether the L2 run will hit a v8-style late-extinction window or sustain the population; whether L1 with the same fixes can recover (we haven't relaunched it yet).

**Implementation pointers (one-liner each):**
- Predator slot count + initial E: configs/axis1/{tiny,med,full}.yaml — `predator_initial`, `predator_e_initial`, `emergency_breeding_n_*` lines.
- Per-species initial E read: [src/jax_state.py:194-200](src/jax_state.py#L194-L200).
- Emergency breeding clause: [src/jax_lifecycle.py:295-318](src/jax_lifecycle.py#L295-L318).
- v10/tiny analysis workspace (kept locally): `/tmp/v10_analysis/{metrics.npz,progress.json}`.

### 15.27 axis1/med v2 retune — scaffold was too soft, lazy-clusterer attractor (2026-05-07)

After 7.1M steps the §15.22+§15.26 axis1/med run produced a stable but **wrong** evolutionary attractor. Reward weights had tightly converged on a "lazy clusterer" phenotype that defeats the experiment's purpose:

- `pred w_act = -25.86 ± 1.33` (predators penalize moving)
- `pred w_prey = +13.57 ± 1.46` (reward being near prey)
- `pred w_pred = +19.87 ± 2.48` (cluster with other predators)
- `pred w_eat = -5.59 ± 2.95` (linear coefficient negative; MLP residual contributes only +0.59 standardized → full-reward β still −1.95)

**Diagnosis: it's a selection problem, not a learning problem.** The std on the converged weights is single-digit on weights of magnitude 13–25 — selection is firing strongly, just on the wrong attractor. Two-tier population structure makes the failure mode legible:

- 15–18 alive predators with mean age 100K (~100 PPO updates each) — long-lived elite.
- ~256 deaths/10K-step window with median age 13K (~13 PPO updates) — scaffold-spawned newborns dying fast.
- Oldest predator has been alive for 359K steps and growing (336+ PPO updates).

The 7× gap between living-elite and dying-cohort age is the signature of scaffold-protected drift. The DDB+DDM scaffold (T_pred=20) plus emergency-breeding clause keep population at ~18 regardless of fitness — so any genome that stays alive long enough to reproduce wins. "Don't move, cluster, want prey nearby" wins because it minimizes metabolic cost; passive contact with prey provides enough food (290 prey scattered around 18 predators in world=880 makes incidental collisions sufficient).

**Why LR doesn't fix it.** Co-author Gil floated `lr_pred_multiplier` ([fe59b46](src/jax_ppo.py)) as the fix, hypothesis being that few-PPO-updates predators can't learn complex policies → reward genome gets "tuned to dumb policies." Two failures in that chain: (a) policies don't inherit — each newborn re-rolls PPO from scratch with the parent's reward genome, so even smart-elite policies don't propagate; (b) faster LR makes the policy converge to whatever the genome rewards, faster — if the genome says don't-move, LR=∞ converges to don't-move instantly. The bug is in *what* gets reproduced (genomes), not in *how fast* policies adapt within a life. We checked: prediction "selection can't distinguish dumb policies → high std drift" is false; observed std is tight, opposite of that prediction.

**The retune (§15.27):** five coupled changes to the scaffold's selection sharpness, none to genome / hazard / physics.

| Knob | v8 | §15.22 | §15.27 (this) | Why |
|------|------|---------|------|-----|
| `density_breeding_threshold_pred` | 10 | 20 | **15** | Pull back partway toward v8. At pop=18: factor 0.45→0.59 (boost 2.2×→1.7×). At pop=10: factor 0.20→0.31 (boost 5×→3.2×). Less relief at the operating point where lazy-clusterer is currently stable. |
| `density_breeding_threshold_prey` | 100 | 200 | **150** | Symmetric. |
| `density_metabolism_threshold_pred` | 10 | 20 | **15** | Symmetric — DDM relief shrinks proportionally. |
| `density_metabolism_threshold_prey` | — | 200 | **150** | Keep symmetric DDM (§15.22 addition). |
| `breeding_share_alpha` | 0.5 | 0.5 | **0.75** | Sharpens energy-weighted share of births. With pred E spread 12/45/92, top-quartile predator now gets ~3× the breeding share of bottom-quartile (was ~1.7× at α=0.5). The actual selection knob — biases reproduction toward high-E predators (proxy for hunting skill). |

**What stays:**
- `predator_initial: 20` and `predator_e_initial: 150` — cold-start cohort protections (§15.26) are mostly free, no reason to roll back.
- `emergency_breeding_n_pred: 3` / `n_prey: 10` — extinction-prevention floor. Rare to fire at steady-state but provides a safety net if §15.27 retune over-tightens.
- K&D-faithful Gompertz age hazard. Never bites at typical lifetimes (age multiplier ≈ 1.06 at age=13K), but kept for paper alignment. Gil's separate `baseline/med_constant_age*` configs ([93542b6](configs/baseline/), [fe59b46](configs/baseline/)) are alternative ablations for this branch of the design space.
- v10 age-keyed LR schedule (`1e-3 → 3e-4` over 30K steps). Doesn't address selection but cheap to keep.
- `reward_type: linear_plus_mlp_residual` — still the axis-1 question. The MLP residual *is* being utilized (mean L1=14.75, contributes 5.5% of reward variance), it just hasn't compensated for the lazy linear coefficients.

**Predictions for the v2 run:**
1. Pred pop oscillates lower (10–14) than v1 (16–20) because the scaffold protects less aggressively. If pop drops below 5 frequently, emergency-breeding floor takes over and we know to back off α or T.
2. Reward genome std stays tight (selection is sharp), but the converged values shift: `w_act` should be less extreme (closer to 0 than -25), `w_prey`/`w_eat` more positive, `w_pred` lower (less clustering pressure once high-E predators dominate births).
3. Median death age stays around 10–20K (still mostly newborn churn) but mean *living* age stabilizes lower than 100K — old elites turn over faster.

**Risks worth flagging:**
- **α=0.75 could create a lineage bottleneck at low N.** If only 5 predators are alive and one has 2–3× the energy of the rest, ~70% of births come from that lineage. After 200K steps the population is one predator's clone. Mitigation: emergency-breeding clause kicks in at N≤3 and uses uniform breeding (no α weighting), so true bottlenecks get rescued. Watch for `next_agent_id - prior_id` in the 100–200 range per 100K steps but lineage ancestry concentrating.
- **Lower T + higher α together is a double tightening.** If v8's 4M-step extinction was because relief was too small *and* selection couldn't converge, doubling down on selection could trigger another extinction window. Emergency-breeding floor is the safety net, but if the run goes extinct around 4–6M, we know §15.27 was too aggressive.
- **The lazy attractor is a local optimum in genome space.** Tightening selection might just produce a *different* local optimum (e.g., aggressive-but-cluster-feeding) rather than the textbook "predators chase, prey flee" phenotype. If this run also converges weird, the issue is the fitness landscape's topology, not the scaffold knobs.

**Diff vs Gil's parallel ablation.** Gil's `med_constant_age_predlr.yaml` ([fe59b46](configs/baseline/med_constant_age_predlr.yaml)) tests the LR-multiplier hypothesis on a constant-age baseline. The two retunes are orthogonal: §15.27 changes selection pressure, Gil's changes within-life policy plasticity. Running both gives 4-cell ablation evidence (scaffold-tight × LR-bumped, scaffold-tight × LR-flat, scaffold-loose × LR-bumped, scaffold-loose × LR-flat). Each cell is one seed for now; seeds can be added if a cell looks interesting.

**Cross-tier sweep applied at the same time.** §15.27 lands on all four axis-1 tiers (tiny/small/med/full), unifying the selection knobs and bringing food/prey density to paper-faithful values. Two reasons to do them together: (a) the lazy-clusterer attractor is plausibly tier-independent (it's a fitness-landscape phenomenon, not a scale-specific one), so the same intervention should apply at every scale we care about; (b) `small` had been left in a partial-update state — got §15.22 but never got §15.26's cold-start scaffolds — so this is also a parity catchup.

**§15.27 unified values across tiers:**

| Knob | tiny (L1) | small (L2) | med (L3) | full | Notes |
|---|---|---|---|---|---|
| `density_breeding_threshold_pred` | 12 → **10** | 17 → **14** | 20 → **15** | 22 → **17** | ~25% pullback per tier |
| `density_breeding_threshold_prey` | 120 → **100** | 170 → **140** | 200 → **150** | 220 → **170** | symmetric |
| `density_metabolism_threshold_pred` | 12 → **10** | 17 → **14** | 20 → **15** | 22 → **17** | symmetric |
| `density_metabolism_threshold_prey` | 120 → **100** | 170 → **140** | 200 → **150** | 220 → **170** | symmetric |
| `breeding_share_alpha` | 0.3 → **0.75** | 0.5 → **0.75** | 0.5 → **0.75** | 0.5 → **0.75** | unified, big bump on tiny |

**§15.26 catchup applied to small (was missing):**
- `predator_e_initial`: 100 → **150** (more PPO updates before starvation)
- `predator_initial`: 7 → **14** (cold-start cohort doubled like other tiers)
- `emergency_breeding_n_pred`: NEW = **3** (uniform-share floor at near-extinction)
- `emergency_breeding_n_prey`: NEW = **10**

**Density normalization to paper-faithful values.** Audit found med was at +4% food density and +7% food regen vs paper (`world=960²`, `food_max=600`, `food_max_regen=10`); tiny was at +7% food density and roughly +29% food regen vs density-isometric. The lazy-clusterer phenotype catches prey at ~1 per 1500 steps per predator at current med rates — that's right at energy break-even. Even a ~5% reduction in food/prey supply could push lazy below break-even. So density normalization isn't just consistency cleanup; it's a complementary intervention to the selection retune.

| Tier | food_max | food_max_regen_per_step | prey_cap |
|---|---|---|---|
| tiny | 250 → **235** | 5 → **4** | 200 → **180** |
| small | 350 → **365** | 7 → **6** | 300 → **275** |
| med | 525 → **505** | 9 → **8** | 375 (paper) |
| full | 600 (paper) | 10 (paper) | 450 (paper) |

Note small's food_max went *up* (it was below density-isometric); the rest came down. Targets are `world² × 6.51e-4` for food_max and `10 × (world/960)²` for regen, both matching paper.

**Implementation pointers:**
- Config diff: [configs/axis1/med.yaml:79-93](configs/axis1/med.yaml#L79-L93) — five selection values changed in the stability block; food density block also touched.
- Cross-tier diff: [configs/axis1/tiny.yaml](configs/axis1/tiny.yaml), [configs/axis1/small.yaml](configs/axis1/small.yaml), [configs/axis1/full.yaml](configs/axis1/full.yaml) — same selection sweep, density-normalized.
- Selection-sharpness math: `breeding_share_alpha` consumed at [src/jax_lifecycle.py](src/jax_lifecycle.py) — `alpha_to_share` function.
- Verification probe used: pulled the alive-pred ages directly from `state.ages[active & species==1]` on the latest checkpoint via `analysis/checkpoint_explorer.load`.
- Reward-nonlinearity probe used: `analysis/reward_nonlinearity_population.py --species predator` against `step_07000000.npz`. PNG saved to `analysis/reward_nonlinearity_population_predator_step7M.png`.

### 15.28 axis1/small extinction at step ~80K → scaffold-aware birth-energy bonus (2026-05-07)

The §15.27 retune launched on axis1/small went extinct at step ~80K — far worse than predicted (the §15.27 risk note flagged a 4–6M extinction window if over-tightened, but it actually fired at 80K). Trajectory pulled from `~/phase1a_small.log` on the GCP VM:

| step | pred | pred E median | catch rate per pred per 10K |
|---|---|---|---|
| 10K | 31 | 102 | 10.1 (above break-even) |
| 20K | 26 | 68 | 6.3 (just below) |
| 30K | 15 | 58 | 5.8 (well below) |
| ~80K | 0 | — | — |
| 100K | 0 | n/a | population frozen, prey at cap |

**Diagnosis: emergency_breeding clause is structurally crippled by `energy_share_ratio = 0.4`.**

The clause fires correctly at N ≤ n_emerg_pred, setting `P_birth = kappa_b * (1 - N/n_emerg)`. At N=1 that's 6.67e-4/step (~99% chance of at least one birth in 10K steps). But:

```python
child_energy = parent_energy * energy_share_ratio   # 0.4
```

In the death-spiral regime, parent E is already low (≤ 15). Newborn gets `parent_E * 0.4 = 4–6 energy`, lifetime ≈ 900–1400 steps before starvation, can't breed in that window, dies. Rescue is born starving.

Compounded by the cascade speed: pred=15 at step 30K dropped to pred=0 by step ~80K. With per-predator energy decay of ~11/10K, all 15 predators starved on similar timelines (synchronized cohort effect from the cold-start). The window where pop sat at N ≤ 3 (when emergency is active) was probably under 5K steps — not enough to recover even if newborns *were* viable.

**Fix: scaffold-aware additive birth-energy bonus.**

Two new config knobs (default 0.0 → paper-faithful):
- `birth_energy_bonus_global`: float, always added at birth.
- `birth_energy_bonus_emergency`: float, added linearly as scaffold engages.

Bonus formula at birth (per parent, per child):
```python
factor = ddb_factor(species_count, threshold)   # 1 at high pop, 0 at full scaffold
bonus  = bonus_global + bonus_emergency * (1 - factor)
child_E       = parent_E * 0.4 + bonus           # K&D + bonus
parent_E_post = parent_E * 0.6 + bonus           # K&D + bonus
```

Both child and parent get the *same* additive lift. The "extra" energy comes from the environment (paper-faithful conservation broken in service of extinction prevention — same philosophical move as DDB+DDM).

**Why additive, not multiplicative.** A multiplier doesn't help when the source pool is tiny: parent E=10 with 2× multiplier still gives child only 8. Additive bonus injects fixed energy independent of parent E, which is exactly what the death-spiral regime needs.

**Default config values for active axis runs (axis1/2/12, all four tiers each):**
- `birth_energy_bonus_global = 0.0` → no effect at high pop (factor ≈ 1).
- `birth_energy_bonus_emergency = 50.0` → at full scaffold engagement (factor = 0), both parent and child get +50. Parent E=10 → parent_post = 6 + 50 = 56, child = 4 + 50 = 54. Both clearly viable.

**Implementation:**
- [src/jax_evolution.py:105-108](src/jax_evolution.py#L105-L108): `spawn_offspring_jax` accepts `energy_bonus` kwarg, defaults to 0.
- [src/jax_evolution.py:152-154](src/jax_evolution.py#L152-L154): `child_energy = parent_E * share + bonus`.
- [src/jax_lifecycle.py:505-528](src/jax_lifecycle.py#L505-L528): `process_births_and_deaths_jax` reads the two bonus knobs, computes per-species DDB factors, threads bonus through the scan body.
- [src/jax_lifecycle.py:574-580](src/jax_lifecycle.py#L574-L580): parent retains `(1 - share) * E + bonus` instead of bare `(1 - share) * E`.
- All 12 axis configs (axis1/2/12 × tiny/small/med/full) get `birth_energy_bonus_global = 0` and `birth_energy_bonus_emergency = 50`.
- Tests: 272 fast tests still pass (no regressions in DDB/DDM, phase0, evolution, lifecycle).

**Predictions for the next axis1/small run:**
1. The pop=15→0 cascade we just saw should be arrested at N ≈ 5–10. As scaffold engages (factor drops below ~0.5), each birth injects +25–50 energy into both parent and child, giving newborns time to learn to hunt before starving.
2. Total energy in the system slowly grows when scaffold is engaged — at pop=10 with T=14, factor ≈ 0.34, bonus ≈ +33. So every birth adds ~66 energy (split between parent and child) to total system energy. Not a runaway — at high pop, factor → 1, bonus → 0.
3. If population stabilizes at 5–10 predators with reward weights still drifting, that's the success signal. If it grows to predator_cap=35 and stays there, the bonus is over-tuned and we should reduce emergency from 50 → 25.

**What this does NOT change:**
- Paper-faithful K&D mechanics at high population (factor ≈ 1, bonus ≈ 0).
- The §15.27 selection retune (T values, alpha=0.75) — those still apply.
- Reward genome / hazard / physics — untouched.

**Risks worth flagging:**
- **Total system energy can grow unboundedly** if pop sits at scaffold-engaged levels for long. At pop=10 with bonus=33 per birth and ~80 births/10K, that's ~2640 energy injected per 10K steps. Mitigated by predator_cap and the fact that the bonus drops to 0 once pop reaches the threshold. Still worth monitoring — if mean E shoots up, the bonus is too large.
- **Newborn energy can exceed `energy_capacity = 1000`** in extreme cases. Currently no clamp. With bonus=50 and parent_E ≈ 100, child_E = 90 — well below cap. Only a problem if bonus is set to >500.
- **Parent's gain `+ bonus` could fight against the K&D depletion logic.** A parent with E=200 paying child 80, then getting +50 bonus, ends with E=170. So they actually *gain* a little energy at high pop with global_bonus > 0. This isn't catastrophic but technically non-paper-faithful even at high pop. Mitigation: keep `birth_energy_bonus_global = 0` in production configs.

Bonus is a complement to §15.27's selection retune, not a replacement. Selection still drives evolution toward hunting genomes; the bonus prevents the cohort from dying before evolution can act.

**§15.28b — T values pulled tighter (2026-05-08).** First §15.28 launch on axis1/small over-energized at cap within 20K steps because the (1-factor) gradient stays large at high pop when T is high relative to caps. Pulled T values back across all tiers to shrink high-pop bonus contribution and tighten selection a touch more:

| Tier  | T_pred (§15.27 → §15.28b) | T_prey | factor at cap before/after |
|-------|---------------------------|--------|---------------------------|
| tiny  | 10 → 9                    | 100 → 90  | 0.80 → 0.83 |
| small | 14 → 12                   | 140 → 120 | 0.86 → 0.895 |
| med   | 15 → 13                   | 150 → 130 | 0.85 → 0.89 |
| full  | 17 → 15                   | 170 → 150 | 0.87 → 0.90 |

**Result on axis1/small at step 2.1M (the second relaunch):** the run successfully escaped the lazy-clusterer attractor and converged on an active hunting phenotype:
- `pred_w_act = +4.2 ± 0.5` (rewards moving — flipped from −25.86)
- `pred_w_prey = +9.5 ± 0.7` (strong, less extreme than lazy +13.57)
- `pred_w_pred = −2.5 ± 0.7` (territorial — flipped from +19.87 cluster)
- `pred_w_eat = +0.5 ± 0.5` (mildly positive — flipped from −5.59)
- Predator pop oscillates 10–20 (well below cap=35), prey 180–270, classic LV cycle.
- Median pred death age 40K-42K (~40 PPO updates) — 3× the lazy regime, 12× the §15.27 starvation cohort.
- pred E median 55-90 — sustainable, not over-energized.

This is the result we wanted from §15.27. The retune chain (selection tightening + density normalize + per-species birth-energy bonus + T pullback) successfully produced the textbook predator-prey dynamics: predators chase, predator-predator clustering negatively rewarded, real LV oscillation.

**§15.28c — propagated to baseline configs (2026-05-08).** axis1 ran the experimental sequence; baseline (K&D linear reward, no MLP residual) was deliberately left in §15.22-only state during the iteration so it wouldn't interfere with Gil's parallel ablation runs (`baseline/med_constant_age*`, `baseline/med_linear_age*`). With the axis1/small results now showing the §15.27/§15.28/§15.28b stack works, propagated the same parameters to `baseline/{tiny,small,med,full}.yaml` so axis1-vs-baseline A/B comparisons share identical scaffold/density and the only difference is the reward genome (linear vs `linear_plus_mlp_residual`).

**Files updated for §15.28c (4 files):**
- `configs/baseline/tiny.yaml` — same values as `axis1/tiny.yaml`
- `configs/baseline/small.yaml` — same values as `axis1/small.yaml`
- `configs/baseline/med.yaml` — same values as `axis1/med.yaml`
- `configs/baseline/full.yaml` — same values as `axis1/full.yaml`

**Files deliberately NOT touched (Gil's experimental ablations):**
- `configs/baseline/med_constant_age.yaml`, `med_constant_age_predlr.yaml`
- `configs/baseline/med_linear_age.yaml`, `med_linear_age_predlr_r256.yaml`

These are Gil's parallel design-grid cells (constant/linear age hazard × default/boosted predator LR). They expect specific scaffold values for clean comparison to each other; rolling the new scaffold under them mid-experiment would invalidate the ablation. Gil can pick up the new scaffold values when he next re-runs.

### 15.29 axis1/small under §15.28b: hunting phenotype, ambush drift, lineage bottleneck (2026-05-08)

The §15.28b launch on axis1/small (run tag `2026-05-08T0407Z`) is the first axis1 run that produced a non-pathological evolutionary outcome. Detailed observations and analysis follow. **Key takeaway:** the §15.27/§15.28/§15.28b stack works — predators escape the lazy-clusterer attractor and converge on a hunting phenotype — but lineage diversity is bottlenecked by sigmoid + alpha stacking, and the population subsequently drifts toward an "ambush" attractor that's energetically similar to lazy but behaviorally distinct (territorial, not clustering). All numbers below from `metrics.npz` (270 datapoints over 2.7M steps) and per-agent checkpoint inspection at step 2.70M.

**Phenotype trajectory across 2.9M steps:**

| step | pred | prey | predE | w_eat | w_act | w_prey | w_pred | regime |
|---|---|---|---|---|---|---|---|---|
| 10K | 34 | 199 | 130 | -0.09 | +0.02 | +0.14 | -0.01 | bootstrap (cap) |
| 110K | 13 | 178 | 52 | +0.29 | -1.21 | +0.50 | -0.46 | settling |
| 510K | 11 | 189 | 71 | +0.89 | **+14.33** | +0.89 | +0.75 | "eager" — moving heavily rewarded |
| 1010K | 11 | 224 | 60 | -3.11 | +1.91 | +4.04 | -0.98 | mid-evolution |
| 1510K | 12 | 224 | 52 | -1.29 | +1.47 | **+11.99** | +0.40 | active hunter, prey-locked |
| 2010K | 11 | 225 | 53 | +0.20 | +4.33 | +9.52 | **-2.41** | active hunter, territorial |
| 2510K | 12 | 224 | 51 | +0.06 | **-10.86** | +9.49 | -1.81 | ambush emerging |
| 2700K | 12 | 249 | 73 | +0.81 | -8.37 | +11.10 | -2.65 | ambush stable |
| 2900K | 11 | 201 | 43 | +2.30 | -8.31 | +9.20 | -0.27 | ambush + pos eat |

The interesting transitions:
- Cold-start cohort (predator_initial=14 + e_initial=150 + bonus) overshot to predator_cap=35 by step 10K, then settled to 10-13 by step 110K and stayed there for the rest of the run.
- Around step 510K, w_act briefly spiked to +14 ("eager" predators that overweight movement reward). This was transient — std collapsed within a few hundred K and the population settled.
- Active hunter phenotype emerged around 1.5-2M with `w_act = +4`, `w_prey = +9`, `w_pred = -2`, `w_eat ≈ 0`. Tight stds (0.4-1.7).
- Ambush phenotype emerged 2.0-2.5M as `w_act` flipped from +4 to -10. Critically, `w_pred` stayed *negative* (territorial), unlike the §15.27 lazy clusterer's `w_pred = +19.87`.

**Comparison of three attractors (this codebase has now produced all three):**

| Weight | Lazy clusterer (§15.27, 7M) | Active hunter (§15.28b, 1.74M) | Ambush (§15.28b, 2.70M) |
|---|---|---|---|
| `w_eat` | **−5.59** | -0.32 | +0.81 |
| `w_act` | -25.86 | **+4.25** | -8.37 |
| `w_prey` | +13.57 | +8.57 | +11.10 |
| `w_pred` | **+19.87 (cluster)** | -1.93 | -2.65 (territorial) |
| pop pred | 18 (over-protected) | 14 | 12 |
| pred E median | 45 | 64 | 73 |
| catch rate / pred / 10K | 6.4 (passive) | 6.5 (active) | 4-5 (ambush) |
| pred death p50 (PPO updates) | 13 | 37 | 49 |

Behaviorally distinct mechanisms despite some surface-level similarity (e.g., ambush and lazy both have negative `w_act`):
- **Lazy:** cluster in dense prey area, ignore eating events (gaslighting reward), survive on accidental prey-bumps.
- **Active hunter:** patrol, chase, catch — high movement, moderate clustering aversion.
- **Ambush:** position alone in dense prey area, sit, reward catches when they happen.

The territorial signature (`w_pred = -2.65`) is the load-bearing differentiator from lazy. The fittest predators (highest E) are *more* territorial than median (top-2 had `w_pred ≤ -4`), so selection is actively pushing against cluster — different from §15.27 where cluster was reinforced by selection.

**Lineage bottleneck analysis (per-agent inspection at step 2.70M, n=12 alive predators):**

Two factors stack to produce per-step birth probability:

```
P_birth = (alpha-share) × (sigmoid)
        = [kappa_b * boost * N * share_i] × [1 / (1 + exp(zeta_eff − beta_b * E))]
```

At our operating point (pop=12, T_pred=12, factor=0.5, zeta_eff=50, beta_b=0.4):

| Pred | E | share (α=0.75) | sigmoid denom | P_birth/step | E[births in 10K] |
|---|---|---|---|---|---|
| 1 | 114.7 | 20.5% | 1/62.6 | 7.79e-5 | **0.54** |
| 2 | 112.3 | 19.2% | 1/162 | 2.89e-5 | **0.25** |
| 3 | 103.5 | 15.0% | 1/5,432 | 6.65e-7 | 0.01 |
| 4 | 93.5 | 11.1% | 1/2.1e7 | tiny | ~0 |
| ... | | | | | |
| 12 | 14.0 | 0.0% | 1/3.6e19 | 4.6e-25 | 0 |

The **K&D sigmoid is extremely steep** below the cliff (`E_cliff = 125`):
- `exp(beta_b)` = `exp(0.4) ≈ 1.49` per unit E → ~2× per 2 units of E lost.
- Predator at E=103.5 vs E=114.7 has ~90× worse sigmoid factor, even with similar alpha share.

**Combined breeding dominance:**
- **Pred 1 alone: 73% of all predator births.**
- Pred 2: 27%.
- Pred 3: 0.6%.
- Pred 4-12: negligible.

Effective breeders ≈ 1-2 individuals at any given time. **Population is essentially descendants of 1-3 ancestors over the last 1M steps**, with mutation noise adding the tight sub-population variance we see (std 0.5-1.4 on weights of magnitude 4-11).

**Per-agent reward weight distribution at step 2.70M:**

```
                    age      E      w_eat    w_act    w_prey   w_pred
Top by AGE:
  Methuselah        99,260  112.3  +0.83    -8.86    +11.13   -2.46     mainstream ambush
  2nd-oldest        94,803   19.5  +1.00    -8.89    +10.70   -2.50     mainstream
  3rd-oldest        91,998   33.9  +0.44    -8.45    +10.74   +5.26     CLUSTER OUTLIER

Top by ENERGY (the fittest = future breeders):
  E=114.7           42,380  114.7  +0.71    -6.66    +11.28   -4.39     STRONG TERRITORIAL
  E=112.3           99,260  112.3  +0.83    -8.86    +11.13   -2.46
  E=103.5              775  103.5  +1.83    -9.69    +10.59   +0.19     newborn (~1 PPO upd)

Median predator    37,201   78.2  +0.77    -8.20    +11.17   -3.44
```

Two sub-populations on `w_pred`:
- ~80% mainstream territorial (`w_pred ≈ -2 to -5`)
- ~20% cluster outliers (`w_pred > 0`) — relics of the lazy lineage that haven't been selected out.

**The fittest predators are MORE territorial than median.** Both top-E predators have `w_pred ≤ -4`. This means selection is currently disadvantaging cluster genomes — the population is moving *away* from the lazy attractor on this dimension, even as `w_act` drifts toward it.

**Population stability:**

Predator pop has been at 10-13 since step 110K (2.6M steps of steady-state). Prey 180-260 in clean LV oscillation. Speed steady at ~58 sps. No drift toward extinction or cap. This is the most stable predator-prey dynamic this codebase has produced.

**Lifespan progression:**

| metric | §15.27 lazy (7M) | §15.28b (1.74M) | §15.28b (2.70M) |
|---|---|---|---|
| pred death p25 | 13K | 20.7K | 24.8K |
| pred death p50 | 13K | 38.3K | 49.7K |
| pred death p75 | 100K (elite) | 91.5K | 113K |
| pred death max | 720K | 568K | 568K |

Median dying predator at 2.70M gets ~48 PPO updates — **3.7× the lazy regime's 13** and significantly more useful policy training time per agent.

**Prey side: weights remain unconverged (this is robust across runs).**

| step | prey_w_eat | prey_w_act | prey_w_prey | prey_w_pred |
|---|---|---|---|---|
| 110K | -0.41 ±1.9 | -0.14 ±2.1 | -1.18 ±2.5 | +0.15 ±2.0 |
| 1010K | +5.09 ±5.2 | +3.62 ±6.2 | -0.43 ±4.4 | +1.24 ±5.6 |
| 2010K | +4.06 ±5.2 | +5.32 ±7.1 | +2.90 ±5.3 | +3.38 ±5.8 |
| 2900K | +4.25 ±6.6 | +7.46 ±6.0 | +2.22 ±4.6 | +5.41 ±5.9 |

Prey have evolved sensible-positive on `eat` (+4.25) and `act` (+7.46), mild flocking on `prey` (+2.22), but **the wrong sign on `pred` (+5.41)** — they're not evolving fear. Stds 4-7 throughout (vs predator stds 0.5-1.4) means the population is heterogeneous, not converged.

Why prey aren't evolving fear (consistent across runs):
1. Predator sparsity — 5-6% of population means most prey never encounter one
2. Ambush predators don't chase — being in "predator territory" is essentially random
3. Predation accounts for ~5% of all prey deaths (rest is starvation/age) — selection signal on `w_pred` is weak
4. Same `mutation_scale = 0.4` as predators, but selection pressure on prey is order-of-magnitude weaker, so weights drift faster than they're pinned

**Spatial position effect (anecdotal observation):**
The user observed that "predator being in the middle of the map versus on the edge or corner probably constitutes like 80% of its success." Predators near map center see prey from 360°, edge predators only see 180°. This compounds the lineage-lottery effect: the highest-E predator (whose lineage dominates breeding) is partly winning via spatial luck, not just hunting skill. The population is converging not just on a phenotype but on the *spatial luck* of the dominant ancestor.

**Drift trajectory of `pred_w_act`:**
- 1.94M: +4.13 ±0.36
- 2.10M: +2.31 ±5.68 (std blows up from newborn cohort with init weights)
- 2.13M: -0.58 ±7.13
- 2.55M: -10.86 ±0.65 (drift complete, std re-collapsed)
- 2.70M-now: -8.3 to -8.6 (stable)

The drift was real, not just sample noise. Initial active-hunter selection was Pareto-optimal, but as prey densities rose to 220-270 and the energy bonus paid metabolic bills, selection pressure on movement weakened. Predators that moved less had similar fitness without the active cost — gradient pulled `w_act` negative until the Pareto-optimum re-equilibrated at ambush.

**Open questions / proposed levers:**

1. **`predator_d_b` ↑ + `predator_d_a` ↓** (anti-lazy energetics). Make sitting more expensive, moving cheaper. Risk: extinction if too aggressive. Conservative proposal: `d_b 4e-3 → 4.5e-3`, `d_a 5e-5 → 4e-5`. Breaks paper alignment on K&D Appendix A energy parameters — should be documented as deliberate departure if applied.
2. **Sigmoid softening: `beta_b` ↓ + `zeta_b_pred` ↓ together.** Reduce lineage bottleneck by spreading breeding across more predators. Discussed but deferred for future ablation.
3. **`breeding_share_alpha` 0.75 → 0.7.** Slightly less concentrated on top-E. Discussed but deferred.
4. **Mutation rate.** Considered raising but pushed back: the bottleneck is *which* lineages reproduce, not mutation magnitude. Increasing mutation just makes the same dominant lineage noisier.
5. **Prey-side fear evolution.** Open. Possible levers: smaller predator mouth (axis-2 territory), faster prey, prey-fear observation channels. None implemented yet.

**Methodological limitations:**
- **Single seed.** All §15.28b conclusions come from one seed (axis1/small with `seed=0`). Some observations (lineage bottleneck on top-1 individual; specific phenotype attractor reached) are seed-luck-sensitive. For paper-claim, need 3-5 seeds at minimum.
- **Single tier.** Only axis1/small was run. Whether the same retune produces equivalent dynamics on tiny/med/full is untested.
- **Spatial dynamics not analyzed quantitatively.** "Middle vs edge" effect is anecdotal — would benefit from a heat map of catch density vs map position.

**Status of the run:**
At step ~2.9M of 10.24M (~28% complete). Run is healthy and producing stable LV dynamics. User has chosen to wait and observe whether the ambush attractor stabilizes or drifts further. Possible decisions ahead:
- If `w_act` stops drifting and stabilizes around -8: ambush is the equilibrium, document and move on.
- If `w_act` drifts further negative toward -25: applying `d_b`/`d_a` retune would push back toward active hunting before extinction.
- If population starts oscillating wider or losing predators: something else (maybe lineage extinction event) is happening.

**Implementation pointers:**
- Live run: `evo-reward-gpu` GCP VM, tmux session `axis1small`, run tag `2026-05-08T0407Z`.
- Analysis script: ad-hoc `~/scaffold_analysis.py` on VM (compares α=0.75 vs α=0.5 share concentration, computes per-agent P_birth).
- Per-agent inspection: `analysis/checkpoint_explorer.load` reads `state.ages`, `state.energies`, `state.reward_weights[i]`.
- Phenotype trajectory pulled from `metrics.npz` (270 datapoints over 2.7M steps).
- Prior comparison data: `analysis/reward_nonlinearity_population_predator_step7M.png` (lazy attractor MLP probe).
