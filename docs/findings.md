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

## 13. Axis 2 redesign — `n_kin` / `n_other` as separate dials

**The old `social_obs: "position_only" | "position_heading_velocity"` flag is being replaced** with two integers that decompose social information by *target type*:

```yaml
n_social_kin:   0   # number of nearest conspecifics to track [heading, speed]
n_social_other: 0   # number of nearest other-species to track [heading, speed]
```

Each tracked agent contributes 2 dims, so `obs_dim = 205 + 2*(n_kin + n_other)`. Backward compatible:

| Old flag | Equivalent new | obs_dim |
|---|---|---|
| `position_only` | `n_kin=0, n_other=0` | 205 |
| `position_heading_velocity` (axis 2 v1) | `n_kin=5, n_other=0` | 215 |

**The redesigned axis 2 program — three single-seed runs on mouth_smol (linear genome, 2M each):**

| Run | `n_kin` | `n_other` | obs_dim | What it isolates |
|---|---|---|---|---|
| `axis2-cross-1` | 0 | 1 | 207 | Cross-species perception alone (no flocking pathway). Tests: does seeing one threat's kinematics let prey preempt predator surges? |
| `axis2-both-1` | 3 | 1 | 213 | Flocking + cross-species. Tests: does kin info enable flocking to actually *limit* predator success when paired with cross-species awareness? |
| `axis2-cross-1-asym` | 0/0 (pred) / 0/1 (prey) | — | mixed | Asymmetric: prey gets cross-species, predator at baseline. Tests: is the surge-prevention effect prey-specific? |

**Design choices:**
- **N_kin = 3 (not 1) when active.** Flocking is an averaging dynamic; a single-nearest signal isn't enough for it to emerge. Real flock alignment requires summary statistics over multiple neighbors.
- **N_other = 1.** A single nearest threat/prey is already informationally rich (distance + bearing from proximity sensors, heading + speed from social slot). Higher N adds noise without obvious benefit.
- **Cross-species filter.** Implementation flips the species mask in `_single_social_obs` from `same_species` to `same_species != obs_species` based on the `n_other` pathway.

**What the experiment program tells us:**

| If `cross-1` survives 2M cleanly | Cross-species perception alone is the lever — prey can preempt predators when given direct kinematic visibility. |
| If `cross-1` also trophic-collapses | The collapse isn't about perceptual access; either it's about flocking concentrating prey, or about predator over-success unrelated to either species' obs. |
| If `cross-1` survives but `both-1` collapses | Flocking *interacts badly* with cross-species perception — the kin-driven concentration overrides the cross-species-driven avoidance. |
| If both survive but `cross-1-asym` collapses | The predator's upgrade is the active ingredient — predators benefit asymmetrically from social info. |

**Out of scope but worth flagging:** the v1 social obs has additional design quirks beyond same-species filtering — heading is absolute world-frame (not relative to observer), speed drops direction, top-N picks closest with no aggregation. These are reasonable choices but other designs exist (relative bearing, mean-of-N, distance-weighted). For now, holding everything else constant and varying only the species filter / N gives the cleanest experimental signal.

## 14. Open questions worth follow-up

1. **`axis2-cross-1` on mouth_smol** (highest priority once n_kin/n_other ships) — does cross-species perception alone prevent trophic collapse?
2. **`axis2-both-1` on mouth_smol** — does flocking + cross-species coexist, or does flocking still drive concentration?
3. **Axis 1 retry** — try mut=0.03 (intermediate) or smaller MLP (hidden=4) to see if the bottleneck-vs-noise tradeoff has a stable middle ground.
4. **Multi-seed at mouth_smol linear** — seeds 1, 2, 3 to 1M each. Seed 1 reached step 730K with fear -3.88 before we paused; seed 2/3 still untouched. Confirms §10 generalizes.
5. **mouth_smol past 1M** — does LV oscillation stay stable or drift? Need 5M run.
6. **`zeta_b_pred = 150`** — still untested. May be redundant if mouth_smol works, but useful as orthogonal validation.
7. **Initial fear bias** — non-zero mean for `prey_w_pred`. Skips the slow fear-evolution phase. Strong intervention.
8. **Audit emevo's catch geometry one more time** — D28 fixed shared credit, but per-catch energy formula or contact resolution might still differ.
