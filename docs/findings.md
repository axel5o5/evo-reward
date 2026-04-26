# Findings — What We've Learned About the Kanagawa-Doya Replication

Cross-cutting insights from Phase 1a parameter exploration. Pairs with:
- [`experiments-log.md`](experiments-log.md) — chronological run-by-run table
- [`params-playbook.md`](params-playbook.md) — parameter tuning reference
- [`emevo-diff.md`](emevo-diff.md) — code-level deviations from upstream

Last updated: 2026-04-26.

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

## 7. eta=0.50 is the cleanest baseline for axis experiments

For Phase 2 axis comparisons (MLP reward, social obs, temporal reward, LSTM policy), we need a baseline that survives long enough to express the variant's effect. Choices:
- 0.45: too fragile for cross-cycle comparisons
- 0.50: survives ~670K — long enough for 2 cycles
- 0.55: also dies, but at slightly larger overshoots
- 0.60 (paper): dies in cycle 1

Use **eta=0.50** for axis runs, accepting that:
- It's a known deviation from paper's eta=0.6
- It will still extinct beyond ~670K
- Cross-axis comparisons should target 500K runs (well within survival window)

## 8. All four axis configs are mechanically working

Smoke-tested at 20K steps each (2026-04-26):
- `axis1_mlp_reward` — MLP reward genome
- `axis2_social_obs` — neighbor heading+speed in obs vector
- `axis3_temporal_reward` — k-step temporal context for reward
- `axis4_lstm_policy` — LSTM policy network

All compile clean, run at ~30 sps, populations behave normally. Slight per-axis differences at step 20K (axis3 had pred=30 vs ~24 elsewhere — possibly the temporal reward giving predators an advantage at chaining catches, but could be noise within one seed).

## 9. Open questions worth follow-up

1. **Multi-seed at eta=0.50** — does any seed survive past 670K? Direct test of whether seed 0 is just unlucky.
2. **Small mouth (paper variant)** — `predator_mouth_range = [0]` instead of `[0,1,17]`. Paper-explicit condition. Smaller catch arc → harder hunting → less overshoot.
3. **`zeta_b_pred` 100→150** — direct attack on the runaway breeding mechanism. Major deviation from paper but targeted at the specific failure mode.
4. **Initial fear bias** — non-zero mean for `prey_w_pred` in initial reward weights. Skips the slow fear-evolution phase. Strong intervention.
5. **Audit emevo's catch geometry one more time** — D28 fixed shared credit, but per-catch energy formula or contact resolution might still differ.
