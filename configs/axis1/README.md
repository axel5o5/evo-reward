# axis1 — Evolved residual reward MLP

## What this axis tests

Whether evolution can find *nonlinear* reward structure beyond K&D's fixed linear coefficients `[1.0, 0.01, 0.1, 0.1]` for the four reward terms `[n_eaten, motor_norm, max_s_prey, max_s_pred]`.

## Mechanism (the diff vs `baseline_faithful.yaml`)

`reward_type: "linear_plus_mlp_residual"` — the per-step reward becomes:

```
r(s) = r_linear(s) + r_mlp_residual(s)
```

- `r_linear` is the standard K&D linear reward (with evolved per-agent weights, like our normal axis-0 setup).
- `r_mlp_residual` is a tiny MLP `input(4) → Dense(4, tanh) → Dense(1, linear)` (25 params total per agent), zero-initialized so the residual contributes 0 at birth — the system starts as exact linear K&D and only deviates if mutation finds useful nonlinear structure.

Mutation parameters (`residual_mutation_scale`, `residual_weight_clip`) control how aggressively the residual evolves between generations.

## Two questions axis 1 answers

1. **Q1 (utilization):** does the residual get used at all? Measured by per-agent L1 norm growing above zero. Answer: **yes** — see findings.md §15.18 (L1 grew from 0 → ~3 by step 440K, → ~7 by step 2.9M). 100% of agents end up using it.
2. **Q2 (structure):** does the residual encode structure that linear *cannot* express, or does it just make the gradient steeper? Open. Designed in [docs/proposals/axis1-residual-analysis.md](../../docs/proposals/axis1-residual-analysis.md).

## Tiers

| File | Scale | Use |
|---|---|---|
| [`tiny.yaml`](tiny.yaml)   | 600² / 200/20 caps | Cheap iteration on residual mechanism |
| [`small.yaml`](small.yaml) | 750² / 300/35 caps | Overnight runs |
| [`med.yaml`](med.yaml)     | 880² / 375/40 caps | Production runs (was the v8 substrate) |
| [`full.yaml`](full.yaml)   | 960² / 450/50 caps | Paper-faithful K&D scale |

All tiers carry the v10 mechanism additions (mouth widening `[0,1,17]`, age-keyed LR schedule, death-age ring buffer) and the §15.22 retune of DDB/DDM thresholds.

## Ancestry

This is the post-§15.18 successor to `axis1_mlp_reward` (which was a full MLP reward, no linear baseline). That earlier design extincted at step 380K — the residual approach (additive on top of linear) is the rescue. See findings.md §11 and §15 for the full history.
