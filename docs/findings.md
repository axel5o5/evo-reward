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

## 14. Open questions worth follow-up

1. **`axis2-aligned` on mouth_smol** (highest priority once bin-aligned encoding ships) — does kinematics-co-located-with-position prevent trophic collapse?
2. **Axis 1 retry** — try mut=0.03 (in flight as v3) or smaller MLP (hidden=4) to see if the bottleneck-vs-noise tradeoff has a stable middle ground.
3. **Multi-seed at mouth_smol linear** — seeds 1, 2, 3 to 1M each. Seed 1 reached step 730K with fear -3.88 before we paused; seed 2/3 still untouched. Confirms §10 generalizes.
4. **mouth_smol past 1M** — does LV oscillation stay stable or drift? Need 5M run.
5. **`zeta_b_pred = 150`** — still untested. May be redundant if mouth_smol works, but useful as orthogonal validation.
6. **Initial fear bias** — non-zero mean for `prey_w_pred`. Skips the slow fear-evolution phase. Strong intervention.
7. **Audit emevo's catch geometry one more time** — D28 fixed shared credit, but per-catch energy formula or contact resolution might still differ.
