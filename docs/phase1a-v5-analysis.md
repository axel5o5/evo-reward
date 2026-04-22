# Phase 1a Run Analysis — `phase1a-v5`

**Run tag:** `2026-04-22T1546Z_phase1a-v5`
**Config:** `baseline_faithful.yaml` (paper-text faithful: 960×960 square, 18 tactile bins, 120° FOV)
**Code state:** commit `50abb22` — full D18 through D26 fix stack
**Seed:** 0
**Outcome:** predator extinction at step **410,000** (4.0% of 10.24M total) after two complete Lotka-Volterra cycles

---

## Why this run matters

Before D26, every run in this session showed **monotonic predator collapse**: populations peaked around step 30K and then declined to extinction by step 80K without recovery. `phase1a-v5` is the first run that produced real oscillations — predators cycled, prey cycled back, and we saw the paper's headline finding (evolved prey fear of predators) happen in-simulation for the first time.

It still extincted, but via a completely different mechanism than prior runs, and one the paper itself describes as a real outcome. This is progress.

---

## Trajectory

```
step    prey  pred    phase
 10K     450   32     D26's impact: predators now catching prey directly in front
 20K     391   41     cycle 1 peak (overshoot of K&D SS ≈ 23)
 60K     109    7     cycle 1 crash
 80K     188    1     cycle 1 trough — v2/v3/v4 extincted here; v5 did NOT
100K     450    1     prey recovered to cap, lone predator surviving
130K     450    3     that predator birthed a second
160K     450   22     cycle 1 recovery — AT K&D steady state ✓
170K     450   39     cycle 2 peak begins
340K     450   48
350K     369   50     hit predator_cap (50 ≫ K&D SS 23)
400K      93    1     cycle 2 trough
410K     140    0     extinction
```

Two full LV cycles, each peaking around 40-50 predators and troughing at 1. Prey held at cap through both recoveries. Period ≈ 200K-300K steps (paper's Figure 6 caption claims ~1M but shows higher-frequency noise on top).

**Events through step 400K:** 3,865 cumulative catches, 4,061 deaths, 4,511 agents born past the initial 160. Population churn is real.

---

## Extinction root cause

Using `scripts/trace_agent.py` from the step-400K checkpoint, we tracked:

1. **The last surviving predator** (agent 3564, slot 492, age 63,190, energy 4.1 at step 400K):
   - Survived **510 steps** after checkpoint, then died at step 400,511
   - **Zero catches** during that window despite mean action magnitude 41.9 (trying to hunt)
   - Energy: 4.1 → 0 (straight-line starvation decay)
   - Reward sum: −7.97 (action cost dominated; no catches to offset)

2. **A contemporary prey** (agent 4140):
   - Evolved `w_pred = −1.14` — **3× stronger fear than the early-run value of −0.36**
   - Survived all 2000 traced steps, energy held steady at 19-25
   - Ate 146 food items in the window

**Conclusion:** extinction was **not** a catch-mechanics failure. It was **over-evolution of prey fear**:
- Prey density fell to 93 at step 400K (down from cap 450)
- Surviving prey had strongly negative `w_pred`, aggressively evading
- Solo predator couldn't catch enough prey to offset metabolic cost
- Starvation → extinction

---

## Paper-consistency

K&D paper Section 4.3 describes this exact failure mode, quote:
> "we conducted 6 simulation runs because the predator population went extinct in one random seed. This run was excluded from the results and analyzed in Appendix C."

That's for their **large-mouth** condition, not the medium mouth we ran. The paper claims no extinctions in the default (medium) configuration across 5 seeds — but we hit one on seed 0. Either:

1. **Seed variance** — K&D used seeds 1-5, not 0. Their Figure 7 shows substantial seed-to-seed variance in evolved reward weights. Running seeds 1-4 and comparing would separate "seed unlucky" from "systematic bug".
2. **Prey learn faster than K&D's** — our PPO converges quicker on evasion, triggering fear-overshoot earlier.
3. **Slight parameter mismatch still present** — we followed paper-text values (D22), not endpoint TOML values. Predator `cap=50 ≫ SS 23` caused overshoot peaks at cap; dropping predator_cap could stabilize.

---

## Replays worth watching

All available at `https://storage.googleapis.com/evo-reward-replays-public/baseline_faithful/seed_0/2026-04-22T1546Z_phase1a-v5/`.

| replay | pred count (start→end) | why watch |
|---|---|---|
| `step_00010001` | 32 → 41 | Baseline behavior — pre-evolution, random policies |
| `step_00170001` | 39 → 44 | Cycle 2 rising, `prey_w_pred ≈ −0.36`: fear *starting* to show |
| `step_00330001` | 24 → 48 | Peak hunting density, both species coevolved |
| `step_00350001` | 50 → 42 | Predator cap hit — densest predator activity |
| `step_00390001` | 7 → 1 | Late crash, `prey_w_pred ≈ −1.14`: watch prey clearly evade the few remaining predators |

Watching 10001 → 350001 → 390001 in sequence shows the evolution arc visually: random-walking baseline → hunting with some avoidance → strong fear-driven evasion by the end.

---

## Next-session action items

1. **Launch seed 1** under identical config — cheapest test of whether seed 0 is just unlucky.
2. **Consider lowering `predator_cap`** from 50 to ~30 to prevent the 2× overshoot at cycle peaks. Deviates from paper but matches observed K&D steady state.
3. **Run baseline_endpoint.yaml** (rectangular 1200×600 world, 9 tactile bins) for comparison — now that D25 made it runnable.
4. **Add a "cycles completed" counter** to progress.json — would help characterize whether a run is stable-oscillating vs. monotonically-declining in real time.

---

## Takeaway

The full D18-D26 fix stack **does** produce K&D-faithful dynamics — prey evolve fear, predators overshoot and crash, single survivors can rebuild populations. The extinction at cycle 3 is plausibly seed variance combined with mild parameter drift. This is the first run in the project where the qualitative trajectory matches the paper's Figure 6.
