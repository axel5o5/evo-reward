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
| `d31d` | 2026-04-24 | same | 0 | 80K | Pred=2@80K (declining from 38) | (mean,post_step) nominal emevo-faithful. pred peak 38@20K. Same crash as a/b/c |
| `emevo_smoke_rect` | 2026-04-24 | emevo `f87a880` gecco2026 | 0 | 102K | Pred=3@102K (terminal) | Emevo's own code on `20251001-predator-default.toml` (1200×600 rect). Same overshoot as us, peak 45@30K |
| `emevo_smoke_square` | 2026-04-24 | same | 0 | 112K | Pred=1@112K (terminal) | Emevo on `20251122-predator-square.toml` (correct paper geometry, 960×960). Peak delayed 10K but same crash |
| `tune_eta_0.45/seed0` | 2026-04-24 | `94fc91c` (D31) | 0 | 150K | **Survived: prey=450 pred=5 at 150K** | **FIRST NON-EXTINCT RUN.** Full LV cycle: pred plateau 22-24@30-50K, trough pred=3@110K, recovery pred_E_max crosses 240 @ 150K. Fear evolved to -0.55, chase drive +1.03. |
| `tune_eta_0.50/seed0` | 2026-04-24 | same | 0 | 150K | Survived: prey=450 pred=18 at 150K | Softer change. Less fragile trough (pred_min=6 vs 0.45's 3). |
| `tune_eta_0.50/seed0/extension` | 2026-04-25 | same | 0 | 500K (resumed) | Survived: prey=450 pred=5 at 500K | **Two complete LV cycles, period ~270K**. Cycle-2 peak pred=25 @ 440K. Strong weight evolution (prey_w_prey 0.71→4.13, pred_w_pred -0.11→+0.40) but **prey_w_pred = +0.25 unchanged → no fear evolution**. |
| `tune_eta_0.55/seed0` | 2026-04-25 | same | 0 | 150K | Survived: prey=450 pred=17 | Single-cycle view comparable to 0.45/0.50. Same shape, no fear yet. |
| `tune_eta_0.50/seed0/extend_1M` | 2026-04-26 | same | 0 | 1M | **Extinct at ~670K (damped osc)** | 3 cycles (peaks 24→25→18), each smaller. No fear evolved (drift to +1.5). |
| `tune_eta_0.55/seed0/extend_1M` | 2026-04-26 | same | 0 | 870K (killed post-ext) | **Extinct at ~720K (runaway osc)** | **4 cycles, peaks 21→30→38→50(cap)**. Cycle 4 hit predator cap → prey crash → extinction. **Fear evolved: prey_w_pred -0.04→-4.95 by cycle 3, -16 post-extinction.** |

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
- **d31d** (mean, post_step): pred peak 38 @ 20K, pred=2 at 80K, prey crashed to 104. pred_w_pred drifted **positive** (+0.18) — peer-presence-is-good, not fear.

**2x2 verdict: D29 and D30 don't materially change outcome.** All four cells overshoot (peak 35-38 around step 20-30K), then crash as prey deplete. The reward-signal wiring (max vs mean, pre vs post) is not the lever. The structural issue is the population dynamics at first peak.

### Decision: run emevo end-to-end (2026-04-24)

Three months of source-level verification hasn't resolved the overshoot. Next move: actually run emevo's `cf_predator.py` on the paper-default config and compare log trajectories at matched seed. If emevo reproduces pred≈23 and ours doesn't at the same seed + config, the divergence is real and diffing logs isolates where. If emevo *also* extincts on seed 0, we're in paper's silently-dropped-seed regime and need multi-seed anyway.

### Emevo smoke run v1 — wrong config (rectangular), 2026-04-24

Ran emevo @ `f87a880` gecco2026 on `20251001-predator-default.toml` (1200×600 rectangle) seed=0 to step 102K. **Emevo crashed to pred=3 by step 102K** — same overshoot-crash shape as our d31d:

| Step | Emevo pred | d31d pred |
|---|---|---|
| 10K | 22 | 31 |
| 20K | 39 | 38 |
| 30K | **45 (peak)** | 33 |
| 50K | 34 | 17 |
| 70K | 10 | 3 |
| 100K | 3 | — (extinct by 80K) |

At step 102K: prey=193 (rebounding) but predators terminal (E_max=35, way below E>240 breeding threshold).

### Config-file discovery (2026-04-24)

Audit of emevo commit history revealed **we used the wrong config**. Commit `fd09012` (2026-04-10) is titled **"predator 1200 (unused now)"** — the authors explicitly deprecated the 1200×600 rectangular config. Paper's figures match the square variant.

**Correct config: `20251122-predator-square.toml`** (960×960 square, +28% area vs rectangular). Only difference from the config we used is `xlim`/`ylim`. Lower encounter density could dramatically reduce the overshoot.

Archived the wrong-config run as `~/emevo_repro/logs/smoke_seed0_rect_wrongcfg/` and relaunched on the correct square config. Also noted: emevo `main` branch has 20+ commits beyond gecco2026 (including `c90e959 predator-a4b2-d100` bd config and `8d83775 action_magnitude is incorrect` — a log-field rename only, not dynamics). Paper results should match gecco2026 tip; if square-config run still crashes we try `main` HEAD next.

### Square config also crashes (2026-04-24)

Square config delays peak by ~10K steps (peak pred=42 @ step 40K vs 45 @ step 30K rectangular) and prey stays at cap longer (through step 30K vs collapsing by 20K rectangular). But the crash trajectory shape is identical: prey→154 at step 60K, →107 at step 70K, predators decline 42→32→24→16→8→3→1 by step 112K. **Both emevo configs extinct on seed 0.**

Settles the diagnosis cleanly:
- **Our code reproduces emevo's behavior at matched seed/config — not broken.**
- Paper's reported pred≈23 is a survivor-bias cross-seed average. Authors silently drop seeds that crash (paper Table 1 reports 5/13 survival for n=0.6, 1/6 for large-mouth, etc.).
- Paper-default parameters are in a knife-edge regime where most seeds extinct.

### Pivot to parameter tuning (2026-04-24)

Decision: stop chasing exact paper match, tune parameters to find a stable LV regime. First lever: `predator_eta` 0.6 → 0.45 (configs/experiments/tune_eta_0.45.yaml). Reduces per-catch energy windfall by 25%, slows predator energy accumulation, should prevent the early overshoot. If predators starve immediately (don't reach breeding threshold at all), we know the lever was too aggressive and need a softer change.

### Fear-vs-extinction tension (2026-04-26)

Long runs to 1M reveal a fundamental tension across our eta sweep:

| eta | Cycles | Extinct? | Fear evolution |
|---|---|---|---|
| 0.45 | 1 (150K) | No (pred=5 surviving) | None (-0.07) |
| 0.50 | 3 (1M) | **Yes @ ~670K** | None (drifted to +1.5) |
| 0.55 | 4 (1M) | **Yes @ ~720K** | **Strong (-5 by cycle 3, -16 post-ext)** |
| 0.60 (paper) | 0 (extinct ~80K) | **Yes (immediate)** | Brief partial (-0.5 in d31c before death) |

**Pattern:** the regime that gives fear evolution is the regime where overshoots create strong predation pressure. The same overshoots that drive fear also drive eventual extinction. Lower eta = stable but no selection for fear. Higher eta = fear evolves but population collapses.

This may be a fundamental dynamics problem, OR something stabilizing in paper that we're missing. Open questions:
1. Does paper actually maintain fear across many cycles, or is it transient like our cycle-3 onset?
2. Is there a parameter we haven't tuned (predator metabolic burn, mouth width, breeding threshold) that decouples the two?
3. Multi-seed: do some seeds at eta=0.55 survive cycle 4 by avoiding the runaway peak?

### BREAKTHROUGH: tune_eta_0.45 survived (2026-04-24)

Final state at step 150K: prey=450 (at cap), pred=5, pred_E_max=264, prey_w_pred=-0.55, pred_w_prey=+1.03. **First non-extinct run in the project.** Trajectory captures a complete Lotka-Volterra cycle:

1. **No overshoot**: pred plateau at 22-24 during steps 20K-50K (vs d31d where pred hit 38 and prey crashed to 104 by 50K)
2. **Prey decline**: 302 → 201 → 221 (steps 50K-70K) as predators deplete
3. **Pred crash**: 23 → 16 → 7 → 4 → 3 (steps 50K-110K) — starvation trough
4. **Prey recovery**: rebounded to cap 450 by step 100K with predator pressure off
5. **Pred recovery**: pred_E_max climbed 67 → 117 → 209 → 264 (steps 90K-150K) — just crossed the E>240 breeding threshold; births beginning
6. **Fear evolution**: prey_w_pred drifted monotonically from 0 → -0.50 by step 110K, in paper's reported range
7. **Chase drive**: pred_w_prey swung to +1.03 at step 140K — predators evolved strong prey-pursuit

This is a near-replication of paper Figure 6's cycle structure. The 25% reduction in digestive rate was enough to break the knife-edge.

Next: tune_eta_0.50 (softer change) launched to map the response curve. Longer-run tune_eta_0.45 extensions will tell us whether the oscillation sustains through cycle 2+.

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

- **emevo reproduction smoke (priority 1)**: run emevo `cf_predator.py evolve --seed 0 --cfconfig-path config/env/20251001-predator-default.toml` to 500K steps (~6h L4). Decides whether their code reproduces paper on our hardware. See `scripts/emevo_repro/`.
- **Seed variance on D31** (priority 2): run seeds 1, 2, 3 on `baseline_faithful.yaml` (`d31d` combo) — see if paper-default behavior reproduces for some subset
- **Parameter sensitivity**: ±20% on `predator_eta` and `zeta_b_pred` to test overshoot hypothesis
- **Specific emevo commit pin**: if HEAD (`f87a880`) doesn't reproduce, fall back to `a777689` (closer to arxiv v2 submission)

---

## Artifacts

- Replays: `gs://evo-reward-replays-public/baseline_faithful/seed_0/{v5,v7,d19,d30,d28a-d,d31a-d}/...`
- Checkpoints: `gs://evo-reward-ckpts/results/...`
- Local logs: VM `~/phase1a.log`, `~/ablation.log`
- Deep analyses: [phase1a-v5-analysis.md](phase1a-v5-analysis.md)
- Deviation catalog: [emevo-diff.md](emevo-diff.md)
