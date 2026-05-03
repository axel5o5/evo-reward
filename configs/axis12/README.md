# axis12 — Combined: residual reward MLP + social heading obs

## What this axis tests

Whether stacking axis 1 (evolved nonlinear reward residual) + axis 2 (richer approach-angle/speed observation) produces effects that are additive, super-additive, or interfering.

## Mechanism

The union of [axis1](../axis1/README.md) and [axis2](../axis2/README.md):

- `reward_type: "linear_plus_mlp_residual"` (from axis 1) + the residual MLP genome params
- `proximity_encoding: "distance_approach_speed"` (from axis 2) + obs_dim = 397, 10 channels/bin

All other physics, scaffolding (DDB+DDM with §15.22 retune), and v10 mechanism additions are identical to the per-axis configs. Only `reward_type` and `proximity_encoding` differ from `baseline_faithful.yaml`.

## Tiers

| File | Scale | Use |
|---|---|---|
| [`tiny.yaml`](tiny.yaml)   | 600² / 200/20 caps | Cheap iteration |
| [`small.yaml`](small.yaml) | 750² / 300/35 caps | Overnight runs |
| [`med.yaml`](med.yaml)     | 880² / 375/40 caps | Production |
| [`full.yaml`](full.yaml)   | 960² / 450/50 caps | Paper-faithful K&D scale |

## Comparing axes

To compare axis effects, run all three at the same tier with the same seed:

```bash
python scripts/run_experiment_jax.py --config configs/axis1/med.yaml  --seed 0
python scripts/run_experiment_jax.py --config configs/axis2/med.yaml  --seed 0
python scripts/run_experiment_jax.py --config configs/axis12/med.yaml --seed 0
# baseline (no axis interventions): use baseline_faithful.yaml
```

Differences in evolution trajectory at matched step counts isolate per-axis effects.
