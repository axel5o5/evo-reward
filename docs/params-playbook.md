# Parameter Tuning Playbook

Reference for each tunable simulation parameter — paper value, our value, role, and observed sensitivity from Phase 1a runs. Read alongside [`findings.md`](findings.md) for the cross-cutting story.

Last updated: 2026-04-26.

---

## How to read this

Each parameter has:
- **Paper value** — Kanagawa & Doya (2025), arXiv:2507.09992v2, default condition
- **Our default** — what `configs/baseline_faithful.yaml` uses
- **Role** — what this controls in the simulation
- **Observed sensitivity** — what we've learned about how the system responds when this changes
- **Tuning notes** — when to consider adjusting it

Parameters with strong evidence for adjustment are marked ⭐. Parameters we have no direct evidence on are marked ❓.

---

## Energy economy

### `predator_eta` ⭐
- **Paper:** 0.6
- **Our current best:** 0.50
- **Role:** Digestive rate — fraction of prey energy a predator absorbs per catch. Scales every catch's energy gain.
- **Observed sensitivity:** **Massive.** This is the single highest-impact lever we've found.
  - 0.45 → fragile single-cycle survival, no fear
  - 0.50 → 3 LV cycles to step ~670K, no fear evolution
  - 0.55 → 4 cycles, fear evolves to -16 by cycle 3-4, extinct ~720K
  - 0.60 (paper) → immediate overshoot, extinct ~80K on seed 0
- **Tuning notes:** Use 0.50 as the baseline for Phase 2 work. Higher values give more paper-like behavior at the cost of longer-run extinction. Lower values are safe but evolve no fear.

### `predator_basic_ec` (predator_d_b in our config) ❓
- **Paper:** 4e-3
- **Our current:** 4e-3
- **Role:** Predator metabolic burn per step. Energy bleed independent of action.
- **Observed sensitivity:** Untested directly. Hypothesized that raising it could shorten predator survival in troughs (BAD) but also dampen overshoot peaks (good). Net effect unclear.
- **Tuning notes:** Not recommended without specific hypothesis. Single change has both tuning effects.

### `predator_force_ec` (predator_d_a) ❓
- **Paper:** 5e-5
- **Our current:** 5e-5
- **Role:** Predator action-cost coefficient (energy spent per unit action norm).
- **Observed sensitivity:** Untested. D24 fixed an act_ratio scaling bug here — predators were undercharged ~49% pre-fix.

### `prey_basic_ec` (prey_c_b)
- **Paper:** 1e-4
- **Our current:** 1e-4
- **Role:** Prey metabolic burn. 40× lower than predators.
- **Observed sensitivity:** Untested.

### `energy_capacity`
- **Paper:** 1000
- **Our current:** 1000
- **Role:** Hard ceiling on individual energy. Predators rarely approach (max ~250-300 in observed runs). Prey can saturate at this in predator-extinct end-states (we've seen prey_E_mean=280 post-extinction).
- **Tuning notes:** Lowering would flatten prey energy peaks and might dampen prey rebounds that fuel predator overshoots. Untested.

---

## Birth & death

### `zeta_b_pred` (predator birth delay) ⭐
- **Paper:** 100
- **Our current:** 100
- **Role:** Predator birth delay. Birth probability `b(E) = kappa_b / (1 + exp(zeta_b - beta_b·E))` — saturates only when `E >> zeta_b`. Effectively the breeding-energy threshold (predator must have E >> 100 / 0.4 ≈ 250 to have meaningful birth probability).
- **Observed sensitivity:** Predator overshoot mechanism. As predators evolve better hunting, more cross the breeding threshold each cycle, peaks grow. This is the runaway driver.
- **Tuning notes:** Raising to 150 would directly attack the runaway. Hypothesis: prevents cycle-N peaks from growing as predators evolve. Significant deviation from paper.

### `zeta_b_prey` (prey birth delay)
- **Paper:** 10
- **Our current:** 10
- **Role:** Prey breed at much lower energy threshold (~25 effective). This is what enables prey to rebound to cap quickly during predator troughs.

### `kappa_b` (birth probability scale)
- **Paper:** 1e-3
- **Our current:** 1e-3
- **Role:** Maximum per-step birth probability when E is well above threshold.
- **Tuning notes:** Lowering would slow all reproduction, similar effect to raising zeta. Less surgical than zeta_b_pred.

### `beta_b` (birth slope)
- **Paper:** 0.4
- **Our current:** 0.4
- **Role:** How sharply birth-prob saturates with E. Higher = sharper threshold.

### Hazard params (`alpha_t_prey`, `alpha_t_pred`, `beta_t_*`, `kappa_h`, `alpha_e`, `beta_h`)
- All match paper. D22 alignment fixed three of these.
- **Observed sensitivity:** Untested individually.

---

## Population caps

### `n_max_predators` ⭐
- **Paper:** 50
- **Our current:** 50
- **Role:** Hard cap on predator population. Slot-based — slots `[450, 500)` reserved for predators.
- **Observed sensitivity:** Cycle-4 in 0.55 hit this cap (pred=50) → mass-catch event → prey crash → predator starvation. The cap itself can become an extinction driver via runaway peaks slamming into it.
- **Tuning notes:** Lowering to 30 would prevent the runaway-into-cap pattern, at the cost of less peak predation pressure. Significant deviation.

### `n_max_preys`
- **Paper:** 450
- **Our current:** 450
- **Role:** Prey cap. Frequently hit during predator troughs.

### `n_initial_predators`
- **Paper:** 10
- **Our current:** 10
- **Role:** Starting predator count.
- **Tuning notes:** Lowering to 5 would reduce initial predation pressure, give prey more establishment time, may reduce cycle-1 overshoot. Untested.

---

## Catch mechanics

### `predator_mouth_range` ⭐
- **Paper:** `[0, 1, 17]` (medium mouth, 60° arc — three 20° tactile bins)
- **Our current:** `[0, 1, 17]`
- **Role:** Which tactile bins constitute the predator's "mouth." Catches require contact via these bins.
- **Observed sensitivity:** Paper has explicit small/medium/large variants. Paper reports 1/6 large-mouth survival vs 5/5 medium-default — large mouth is more extinction-prone (faster catches → bigger overshoots).
- **Tuning notes:** Tightening to `[0]` (single bin, 20° arc) is a paper-explicit variant ("small mouth"). Each catch is harder, dampens overshoots. Worth testing as a paper-aligned alternative to eta tuning.

### `predator_eat_interval`
- **Paper:** 10
- **Our current:** 10
- **Role:** Cooldown between catches per predator. v8 ablation tested cooldown=1 — didn't fix extinction.

### `n_tactile_bins`
- **Paper:** 18
- **Our current:** 18
- **Role:** Number of tactile sensor bins around predator perimeter (each 20°).

---

## Sensors & perception

### `proximity_max_range`
- **Paper:** 120 (Appendix A)
- **Our current:** 120
- **Role:** How far prey/predator can see via proximity sensors.
- **Observed sensitivity:** D27 narrowed from 200 → 120. v7 (sensor=120 only, pre-D28) had fear unable to evolve, extinct 110K.

### `proximity_fov_deg`
- **Paper:** 120°
- **Our current:** 120°

### `n_proximity_sensors`
- **Paper:** 32
- **Our current:** 32

### `sensor_agg_type`
- **Paper / emevo default:** "mean"
- **Our current:** "mean"
- **Role:** How sensor readings across bins are aggregated for reward computation. D29 changed our default from "max" to match emevo. Modest effect, not the lever.

### `reward_obs_timing`
- **Paper / emevo default:** "post_step"
- **Our current:** "post_step"
- **Role:** Whether reward uses pre- or post-physics observation. D30 changed our default to "post_step". Small effect.

---

## World geometry

### `world_size` / `xlim` / `ylim` ⭐
- **Paper:** 960 × 960 (square)
- **Our current:** 960 × 960
- **Observed sensitivity:** Switching from 1200 × 600 (rectangular) to 960 × 960 (square) delays cycle-1 crash by ~10K-15K steps. Doesn't prevent extinction but lowers density. **The rectangular config is explicitly deprecated in upstream emevo (commit fd09012).**
- **Tuning notes:** Always use square. Settled.

### `food_max`, `food_initial`, `food_growth_rate`, `food_max_regen_per_step`
- All match paper. Paper varies `food_growth_rate` (`n` parameter) across 0.4 / 0.5 / 0.6 conditions; n=0.5 is our default. Paper reports 5/13 survival at n=0.6.

---

## Mutation & evolution

### `mutation_scale`
- **Paper:** 0.4 (StudentT scale)
- **Our current:** 0.4
- **Role:** Per-axis mutation noise on reward weights at birth.

### `weight_clip`
- **Paper:** 100 (Section 4.2) — but emevo uses 10
- **Our current:** 100
- **Role:** Bounds reward weights to ±value. v9 ablation tested 10 (emevo-aligned). Didn't change extinction.

### `reward_weights_init_std`
- **Paper:** 0.1
- **Our current:** 0.1

### `reward_weights_init_mean` ⭐ (potential lever)
- **Paper:** 0.0 (zero-mean Gaussian)
- **Our current:** 0.0
- **Role:** Mean of initial reward weights. All zero by default — agents start neutral on every reward axis.
- **Tuning notes:** Setting `prey_w_pred` initial mean to -0.5 would give prey starting fear bias. Skips the slow evolution phase. **Significant deviation from paper.** Untested.

---

## PPO hyperparameters

These all match paper (Section 4.1). We have not tuned any of these and have low priority to do so unless we suspect optimization issues.

| Param | Paper | Notes |
|---|---|---|
| `lr` | 3e-4 | Adam learning rate |
| `gamma` | 0.999 | Reward discount |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_epsilon` | 0.2 | PPO clip |
| `entropy_coef` | 0.001 | Entropy bonus |
| `rollout_steps` | 1024 | Rollout length per epoch |
| `minibatch_size` | 256 | PPO update minibatch |
| `ppo_epochs` | 10 | Updates per rollout |

---

## Recommended tuning ranking (2026-04-26)

If you have one experiment slot:

1. **Multi-seed on `predator_eta=0.50`**, seeds 1-3 — cheapest information, paper-aligned methodology.
2. **`predator_mouth_range = [0]`** — paper variant, lowers catch rate, dampens overshoot.
3. **`zeta_b_pred = 150`** — direct attack on runaway breeding. Major deviation but targeted.
4. **`reward_weights_init_mean[prey_w_pred] = -0.5`** — skip the slow fear-evolution phase. Strong intervention.
5. **`n_max_predators = 30`** — cap the runaway peak. Less paper-aligned, less surgical than #3.

Untested levers worth eventually trying:
- `predator_basic_ec`, `predator_force_ec` (energy bleed)
- `n_initial_predators` (starting density)
- `energy_capacity` (saturation ceiling)
