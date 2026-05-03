# configs/baseline/

K&D-linear scaffolded controls — the matched A/B reference for the axis1, axis2, and axis12 configs at each tier.

## Why this directory exists

Each axis config (`axis1/`, `axis2/`, `axis12/`) builds on top of v10's DDB+DDM stability scaffolds at a particular tier (tiny / small / med / full). To answer "did the axis modification help?" you need a control run with the **same** scaffolds, **same** world size, **same** population caps, and **same** PPO settings — differing **only** in the axis-under-test mechanism. That's what this directory provides.

```
configs/baseline/
    tiny.yaml   # baseline_kd_linear_tiny  — control for axis*/tiny
    small.yaml  # baseline_kd_linear_small — control for axis*/small
    med.yaml    # baseline_kd_linear_med   — control for axis*/med  (production tier)
    full.yaml   # baseline_kd_linear_full  — control for axis*/full (paper scale)
```

Each file is structurally identical to the same-tier `axis1/<tier>.yaml`, with two changes:
- `reward_type: "linear"` (no residual MLP)
- residual MLP params (`residual_*`) removed

For axis-2 comparisons the additional implicit diff is the K&D-faithful `obs_dim: 205` (vs axis-2's `obs_dim: 333` and `proximity_encoding: distance_approach_speed`).

## A/B usage

| Compare | Pure effect |
|---|---|
| `axis1/<tier>` − `baseline/<tier>` | residual reward MLP only |
| `axis2/<tier>` − `baseline/<tier>` | social heading obs only |
| `axis12/<tier>` − `baseline/<tier>` | combined |

Always pair the same tier; cross-tier comparisons confound scale with mechanism.

## How this differs from `baseline_faithful.yaml`

- [`../baseline_faithful.yaml`](../baseline_faithful.yaml) is the **paper-pure** K&D reference — 960², no stability scaffolds, paper-exact PPO and mouth. Use this when you want to reproduce K&D directly.
- Files here are **scaffolded controls** — they include v10 DDB+DDM scaffolds because the axis runs do, and a fair comparison needs them. Use these when you want to attribute an effect to a specific axis modification.

Both serve different purposes; both are kept.
