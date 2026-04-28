# Experiments Log — Phase 1a Replication of Kanagawa & Doya 2025

Chronological run-by-run table for every Phase 1a run. Add a new row per run.

For cross-cutting analysis and what we've learned, see:
- [`findings.md`](findings.md) — what's been concluded across runs
- [`params-playbook.md`](params-playbook.md) — parameter tuning reference
- [`emevo-diff.md`](emevo-diff.md) — code-level deviations from emevo

**Paper anchor:** Kanagawa & Doya, *Evolution of Fear and Social Rewards in Prey Predator Relationship*, arXiv:2507.09992v2 (Feb 2026). Default condition: medium mouth, `n=0.5` food regen. Paper Table 1 reports prey avg 349, predator avg 23 across 5 default-condition seeds, 10.24M steps each, ~40h on A100.

**Our compute:** GCE L4 GPU VM, ~25-30 sps typical → ~110-120h for full 10M.

---

## Run-by-run log

| Run tag | Date | Code @ SHA | Seed | Max steps | Outcome | One-line note |
|---|---|---|---|---|---|---|
| `phase1a-v2/v3/v4` | pre-04-20 | pre-D18 | 0 | ~100K | Extinct @ ~80K | Monotonic pred collapse; catches barely happened (D18 bug) |
| `phase1a-v5` | 2026-04-22 | `50abb22` (D18-D26) | 0 | 410K | Extinct @ 410K | Two complete cycles; died of over-evolved fear. See [phase1a-v5-analysis.md](phase1a-v5-analysis.md) |
| `phase1a-v7` | 2026-04-22 | `1067319` (+D27 sensor=120) | 0 | ~110K | Extinct @ ~110K | Fear couldn't evolve; opposite failure of v5 |
| `exp_v8_no_cooldown` | 2026-04-23 | `b9188c8` (pred_eat_interval=1) | 0 | ~420K | Extinct @ ~420K | Cooldown=1 not the fix |
| `baseline/d19/seed0` | 2026-04-23 | `94fc91c` (D28-30 NOT yet) | 0 | 220K+ | Extinct @ ~100K | Paper-faithful D18-D27 |
| `baseline/d30/seed0` | 2026-04-24 | `6445d3c` (+D28/29/30) | 0 | 220K+ | Extinct @ ~60K | First full stack; seed variance not regression |
| `d28a` / `d28b` | 2026-04-24 | `1aa8114` | 0 | 80K | Extinct @ 70-80K | (max,pre) (mean,pre). **Invalid — pre-D31** |
| `d31a` / `d31b` | 2026-04-24 | `94fc91c` (+D31) | 0 | 80K | Extinct @ 70-80K | (max,pre) (mean,pre); peak 36-38 @ 20K |
| `d31c` | 2026-04-24 | same | 0 | 80K | Pred=3 @ 80K | (max,post). First fear evolution: prey_w_pred=-0.53 |
| `d31d` | 2026-04-24 | same | 0 | 80K | Pred=2 @ 80K | (mean,post) emevo-faithful. Nominal Phase 1a baseline |
| `emevo_smoke_rect` | 2026-04-24 | emevo `f87a880` | 0 | 102K | Pred=3 @ 102K | Emevo's own code on rect config. Same crash as us |
| `emevo_smoke_square` | 2026-04-24 | same | 0 | 112K | Pred=1 @ 112K | Emevo on correct paper geometry. Still extinct |
| `tune_eta_0.45/seed0` | 2026-04-24 | `94fc91c` (D31) | 0 | 150K | **Survived** | First non-extinct run. Pred=5 at 150K, fragile |
| `tune_eta_0.50/seed0` | 2026-04-24 | same | 0 | 150K | Survived | Pred=18 at 150K, less fragile than 0.45 |
| `tune_eta_0.50/seed0/extend` | 2026-04-25 | same | 0 | 500K | Survived | Two complete LV cycles, period ~270K |
| `tune_eta_0.55/seed0` | 2026-04-25 | same | 0 | 150K | Survived | Pred=17 at 150K, single-cycle view |
| `tune_eta_0.50/seed0/1M` | 2026-04-26 | same | 0 | 1M | Extinct @ ~670K | 3 cycles damped (peaks 24→25→18). No fear |
| `tune_eta_0.55/seed0/1M` | 2026-04-26 | same | 0 | 870K (killed) | Extinct @ ~720K | 4 cycles, runaway peak to cap=50. **Fear evolved to -16** |
| `axis1_smoke` | 2026-04-26 | `779d465` (eta=0.50) | 0 | 20K | OK | **No-op** — `reward_type: mlp` flag silently ignored. Linear-genome baseline run mislabeled. See `docs/todo/wire-mlp-temporal-through-jax-sim.md` |
| `axis2_smoke` | 2026-04-26 | same | 0 | 20K | OK | Social obs (genuinely wired). prey=330 pred=26 — clean |
| `axis3_smoke` | 2026-04-26 | same | 0 | 20K | OK | **No-op** — `obs_buffer` allocated but never updated; `compute_temporal_reward` never called. Baseline-equivalent |
| `axis4_smoke` | 2026-04-26 | same | 0 | 20K | OK | **No-op** — runner imports only `build_ppo_update_fn` (MLP); LSTM PPO never called. Baseline-equivalent |
| `axis2_real_500k/seed0` | 2026-04-26 | same | 0 | 310K (killed) | Extinct @ ~100K | **REAL test** of social_obs (the only wired axis). Extinct on eta=0.50 substrate. |
| `axis1_real_1M/seed0` | 2026-04-27 | `1c0723e` (mouth_smol substrate) | 0 | 80K (killed) | Killed | **No-op** — same axis1 flag bug; killed once realized. Was tracking baseline-equivalent. |
| `sweep_mouth_smol_1M/seed0` | 2026-04-27 | `c35bfab` (+sweep configs) | 0 | 1M | **SURVIVED** | First 1M completion. eta=0.50 + mouth=`[0]`. Fear evolved to -1.97 sustained |
| `sweep_mouth_smol_1M_seed1` | 2026-04-27 | same | 1 | 730K (paused) | alive at pause | Step 730K, prey 449 / pred 21, fear `prey_w_pred = -3.16 ± 7.88`. Paused to free VM for axis1 |
| `axis1_mouth_smol_1M/seed0` | 2026-04-28 | `dc67fc6` (`reward_type: mlp` wired, mouth_smol substrate) | 0 | 1M | Extinct @ ~380K | Pred crashed to 1 @ 140K, recovered to 30, second bottleneck @ 360K extincts. `mlp_mutation_scale=0.01` → diversity-recovery too slow vs linear's 0.4. See findings.md §11 |
| `axis1_mouth_smol_1M_mut08/seed0` | 2026-04-28 | `d93d664` (mut_scale 0.01→0.08) | 0 | 370K (killed) | Extinct @ ~300K | Diversity *did* appear (`pred_w` ±0.06-0.11), but pred peak only ~15 — too many non-viable offspring → smaller stable pop → first downturn extincts. Counter-arg from §11 confirmed |

Weight keys: `w_eat / w_act / w_prey / w_pred`. "Fear evolved" = `|prey_w_pred|` > 0.3 sustained.

---

## Deviation fixes chronologically (D-series)

Each row: the bug, what we did, the commit SHA, and whether it touched dynamics materially.

| D# | SHA | Title | Impact on dynamics |
|---|---|---|---|
| D18 | `3d2c711` | Predator catch: contact + mouth-bin + cooldown (was radial 40-80, no cooldown) | **Massive** — catches happen at all |
| D19 | `da6ad94` | Slot↔body fix + phyjax2d contact plumbing | **Massive** — predators lived in prey-sized bodies pre-fix |
| D20 | `b39d084` | Deactivate caught prey (were re-catchable forever) | **Medium** — closes a predator-free-energy loophole |
| D21 | `5d8278e` | Cumulative event counters in progress.json | Observability only |
| D22 | `a2d8279` | Paper-text alignment: `beta_t_prey`, `zeta_b_prey`, `food_radius` | **Medium** — shifts equilibria |
| D23 | `c69e89c` | `rollout_dones=True` on death (GAE correctness) | **Small** — semantic fix |
| D24 | `bd1b961` | Energy cost uses act_ratio-scaled action norm (was raw; undercharged predators ~49%) | **Large** — predator energy economy |
| D25 | `ad5aa69` | Rectangular world support | Config plumbing |
| D26 | `50abb22` | Tactile bin indexing 90° off (mouth arc pointed 90° right of heading) | **Massive** — root cause of pre-v5 extinctions |
| D27 | `1067319` | Sensor range 200→120 (paper Appendix A) | **Large** — fear signal strength |
| D28 | `6445d3c` | Predator energy credit: shared, not deduped | **Medium** — upper-tail predator energy |
| D29 | same | Sensor reward agg: `mean` not `max` (emevo default) | **Medium** — fear/chase gradient magnitude |
| D30 | same | Reward from post-physics obs (was pre-step) | **Small** — temporal credit alignment |
| D31 | `94fc91c` | **Proximity sensor** convention 90° off (same class as D26) | **Massive** — prey couldn't see predators in front |

Full deviation catalog with code-level details: [emevo-diff.md](emevo-diff.md).

---

## Open questions (current)

See [`findings.md` §10](findings.md) for the full list. Top priorities (post-mouth_smol):

1. **Multi-seed at mouth_smol** — does seed 1, 2, 3 also survive 1M? Confirms mouth_smol generalizes.
2. **Re-run all 4 axes on mouth_smol substrate** — replaces eta=0.50 baseline that extincts before axes can express their effect.
3. **mouth_smol past 1M** — does the LV oscillation stay stable or eventually drift? 5M run tells us.

---

## Artifacts

- Replays: `gs://evo-reward-replays-public/<experiment>/seed_<N>/<run_tag>/...`
- Checkpoints: `gs://evo-reward-ckpts/results/...`
- Local logs: VM `~/<run>.log`
- Deep analyses: [phase1a-v5-analysis.md](phase1a-v5-analysis.md)
- Tooling: [`scripts/emevo_repro/`](../scripts/emevo_repro/) (emevo reproduction harness, compare_logs.py)
