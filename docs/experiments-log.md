# Experiments Log — Phase 1a Replication of Kanagawa & Doya 2025

Running log of every Phase 1a run, the code state it was launched against, outcome, and what we learned. Add a row per run; link to deep-dive postmortems as separate `docs/*-analysis.md` files when warranted.

**Paper anchor:** Kanagawa & Doya, *Evolution of Fear and Social Rewards in Prey Predator Relationship*, arXiv:2507.09992v2 (Feb 2026). Default condition: medium mouth, `n=0.5` food regen. Paper reports **prey avg 349, predator avg 23** (Table 1) across 5 seeds, 10.24M steps each, ~40h on A100.

**Our compute:** GCE L4 GPU VM, ~27 sps typical → ~160h for full 10M.

---

## Run-by-run log

| Run tag | Date | Code @ SHA | Seed | Max steps | Outcome | One-line note |
|---|---|---|---|---|---|---|
| `phase1a-v2/v3/v4` | pre-04-20 | pre-D18 | 0 | ~100K | Extinct @ ~80K | Monotonic pred collapse; catches barely happened (D18 bug) |
| `phase1a-v5` | 2026-04-22 | `50abb22` (D18-D26) | 0 | ran to 410K | Extinct @ **410K**, first real LV oscillation | Two complete cycles; died of over-evolved fear. See [phase1a-v5-analysis.md](phase1a-v5-analysis.md) |
| `phase1a-v7` | 2026-04-22 | `1067319` (+D27 sensor=120) | 0 | ran to ~110K | Extinct @ ~110K | Fear couldn't evolve at all; opposite failure of v5 |
| `exp_v8_no_cooldown` | 2026-04-23 | `b9188c8` (pred_eat_interval=1) | 0 | ran to 420K | Extinct @ ~420K | Cooldown=1 tested; not the fix. Prey_w_pred flatlined |
| `baseline/d19/seed0` | 2026-04-23 | `94fc91c` (D28/29/30 NOT yet) | 0 | ran to 220K+ | Extinct @ ~100K | Paper-faithful D18-D27. Same single-cycle crash |
| `baseline/d19/seed1` | 2026-04-24 | same | 1 | killed early | N/A | Aborted to unblock D28-D30 work |
| `baseline/d30/seed0` | 2026-04-24 | `6445d3c` (+D28/29/30) | 0 | ran to 220K+ | Extinct @ **60K** (faster!) | First full stack; earlier collapse than d19. Seed variance, not regression |
| `d28a` | 2026-04-24 | `1aa8114` (ablation infra) | 0 | 80K | Extinct @ 70K | (max,pre_step) D28 only. **Invalid — pre-D31 proximity bug** |
| `d28b` | 2026-04-24 | same | 0 | 80K | Extinct @ 80K | (mean,pre_step). **Invalid — pre-D31** |
| `d31a` | 2026-04-24 | `94fc91c` (+D31) | 0 | 80K | Extinct @ 70K | (max,pre_step). pred peak 38@20K |
| `d31b` | 2026-04-24 | same | 0 | 80K | Extinct @ 80K | (mean,pre_step). pred peak 36@20K |
| `d31c` | 2026-04-24 | same | 0 | 80K | Pred=3@80K (not extinct yet, declining) | (max,post_step). **First run where fear evolved (prey_w_pred=-0.53)** |
| `d31d` | 2026-04-24 | same | 0 | 80K | running… | (mean,post_step) — nominal emevo-faithful combo |

Weight keys: `w_eat / w_act / w_prey / w_pred`. "Fear evolved" = |prey_w_pred| > 0.3 sustained.

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
| D31 | `94fc91c` | **Proximity sensor** convention 90° off (same class as D26; we fixed D26 but missed D31) | **Massive** — prey couldn't see predators in front |

---

## Diagnosis timeline

### Pre-D18 (through 2026-04-20)
Monotonic predator collapse every run. Catches barely registered because the "radial distance 40-80" catch mechanic was incorrect.

### D18-D26 era (2026-04-22)
D18 unlocked catches. D19 fixed that predators had prey-sized physics bodies (invariant slot↔body mismatch since JAX rewrite). Sequence of D20-D26 cleaned mechanical issues. Culminated in `phase1a-v5`, the first run with real LV oscillation — two complete cycles before extinction at step 410K via over-evolved fear (prey `w_pred=-1.14`, predators couldn't catch anything).

### Cooldown hypothesis (v7, v8)
- **v7** (sensor=120, D27): fear couldn't evolve, extinct @ 110K. Wrong direction.
- **v8** (cooldown=1): didn't save predators either. Cooldown wasn't the missing piece.

### D28/D29/D30 emevo-source audit (2026-04-23)
External thread identified 3 discrepancies between our code and `cf_predator.py`:
- **D28 shared credit** (biggest lever): emevo credits every predator contacting a prey; we deduped to one
- **D29 mean vs max**: emevo defaults `sensor_agg_type="mean"`; we hardcoded `max`
- **D30 post-step reward obs**: emevo uses `obs_t1.sensor`; we used pre-step

Fixed all three. But d30 seed 0 extincted at step 60K — **earlier** than d19. Triggered the 2×2 ablation.

### Knife-edge predator-breeding diagnosis (2026-04-23)
Math pass on birth-probability function:
- `b_pred(E) = 1e-3 / (1 + exp(100 - 0.4·E))`
- Saturates only above E≈250; at E=200 prob is 2e-12 (effectively zero)
- Our observed predator **max** E only crosses 250 briefly in the first 30K steps
- Once prey crashes and per-catch energy drops, no predator climbs back above 250 → **extinction irreversible**

This is structural, not a bug. Paper survives by either:
(a) stable cycles where pred max stays > 240 during troughs
(b) seed-selection bias (they dropped some extinction seeds, e.g. 5/13 kept for n=0.6)

### 2×2 ablation (2026-04-24, first pass — invalidated)
d28a through d28d across (sensor_agg × reward_obs_timing). d28a and d28b completed; all 3 started runs died 60-80K. **Invalidated mid-sweep by D31 discovery** — all were training under mis-framed proximity stimuli.

### D31: proximity sensor 90° bug (2026-04-24)
Same class as D26 (which was previously the "single most likely cause of extinction"). Tactile-bin convention was fixed in D26; proximity-sensor convention was missed. Prey couldn't see predators approaching from in front. Predators couldn't see prey in front of their mouth.

### 2×2 ablation (2026-04-24, second pass on D31-fixed code)
- **d31a** (max, pre_step): pred peak 38 @ 20K, extinct @ 70K
- **d31b** (mean, pre_step): pred peak 36 @ 20K, extinct @ 80K
- **d31c** (max, post_step): pred peak 35 @ 30K, **pred=3 at 80K** — first run where fear actually evolved (prey_w_pred=-0.53)
- **d31d** (mean, post_step): in progress

D31 helped — d31c shows real evolutionary signal — but no cell has yet sustained predators past a single LV cycle. Same overshoot-then-crash pattern.

---

## Emerging hypothesis (as of 2026-04-24)

**We overshoot paper's equilibrium on the first peak, then can't recover.**

Evidence:
- Paper Table 1: default pred avg = **23**
- Our runs: pred peaks at **30-42**, then crashes to 0
- Paper oscillation period: ~200-400K steps (Figure 6 detail)
- Our runs: single cycle, first peak at 20-30K, crash by 60-80K
- Paper's predator breeding needs E > 240; knife-edge — once violated, no recovery

Candidate mechanisms:
1. **Early catch efficiency too high** → predators accumulate too fast → overshoot → prey crash → predators starve
2. **Seed 0 happens to be unlucky**; seeds 1-4 might survive
3. **Subtle parameter mismatch we haven't found** — anything that slightly increases per-catch energy or slightly lowers breeding threshold would prevent overshoot

### Paper verification of seed variance

- Paper default `n=0.5`: 5 seeds shown, reports all survived
- Paper `n=0.6`: "5 out of 13 runs" used — **62% extinction rate in their own data**
- Paper `n=0.4`: 5 out of 6
- Paper large-mouth: 1 of 6 extinct, excluded from analysis

So seed variance / extinction is real in the paper too. But their default clearly works more often than ours.

---

## Open questions

1. Does seed 1, 2, 3 survive on D31 code? (Haven't tested multi-seed on D31 yet)
2. Is there a remaining discrepancy in predator catch mechanics, energy transfer formula, or something we haven't audited?
3. Is there a way to lower "peak predator overshoot" that maps to what paper does naturally? E.g., does the paper's predator policy learn more slowly because of PPO hyperparams we differ on?

---

## Next experiments planned

- **Seed variance on D31**: run seeds 1, 2, 3 on `baseline_faithful.yaml` (`d31d` combo) — see if paper-default behavior reproduces for some subset
- **Parameter sensitivity**: ±20% on `predator_eta` and `zeta_b_pred` to test overshoot hypothesis
- **Specific emevo commit pin**: cross-check src/ against emevo `a777689` (gecco2026 tip just before arxiv submission)

---

## Artifacts

- Replays: `gs://evo-reward-replays-public/baseline_faithful/seed_0/{v5,v7,d19,d30,d28a-d,d31a-d}/...`
- Checkpoints: `gs://evo-reward-ckpts/results/...`
- Local logs: VM `~/phase1a.log`, `~/ablation.log`
- Deep analyses: [phase1a-v5-analysis.md](phase1a-v5-analysis.md)
- Deviation catalog: [emevo-diff.md](emevo-diff.md)
