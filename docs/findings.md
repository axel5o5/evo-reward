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
