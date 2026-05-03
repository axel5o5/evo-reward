# configs/

Science configs organized by axis × tier. For full project context start at [docs/CURRENT_STATE.md](../docs/CURRENT_STATE.md).

## Layout

```
configs/
    axis1/    # axis 1: residual reward MLP (evolved nonlinear addition to K&D's linear reward)
    axis2/    # axis 2: social heading obs (richer per-bin proximity encoding §15.24)
    axis12/   # axis 1+2 combined
    archive/  # superseded / deprecated
    runtime/  # ops overlay (checkpoint cadence, log intervals)
    baseline_faithful.yaml  # K&D-pure reference
```

## Tier convention (every axis has these four)

| Tier | World | Caps (prey/pred) | Hidden | Wall-clock for 1M | Use case |
|---|---|---|---|---|---|
| `tiny`  | 600² | 200 / 20 | 32–48 | ~3–5 h GPU | sanity-check + cheap iteration; below paper's selection floor |
| `small` | 750² | 300 / 35 | 48–64 | ~6–8 h GPU | overnight runs; intermediate fidelity |
| `med`   | 880² | 375 / 40 | 64    | ~12–18 h GPU | production tier you'd cite in a paper |
| `full`  | 960² | 450 / 50 | 64    | ~24–36 h GPU | paper-faithful K&D scale |

Promote from `tiny` upward only after the mechanism shows signal. Tier T values are graduated per §15.22 (T_pred 12/17/20/22; T_prey 120/170/200/220 for tiny/small/med/full).

## Axis convention

Each axis directory has its own `README.md` explaining the mechanism. Brief summary:

| Axis | Mechanism | What changes vs K&D baseline |
|---|---|---|
| `axis1` | Evolved residual reward MLP | `reward_type = linear_plus_mlp_residual` — 25-param MLP added to K&D's linear reward, zero-init, mutates with `residual_mutation_scale` |
| `axis2` | Social heading observation | `proximity_encoding = distance_approach_speed` — per-bin distance + approach-angle (cos=+1 directly toward me) + speed magnitude. obs_dim = 397 |
| `axis12` | Combined | Both: residual reward + social heading obs |

Filenames inside an axis folder are just the tier (`tiny.yaml`, `small.yaml`, `med.yaml`, `full.yaml`). `experiment_name` inside the file encodes axis + mechanism + tier explicitly so GCS run paths are unambiguous (e.g. `axis1_residual_reward_mlp_tiny`).

## Paper reference

[`baseline_faithful.yaml`](baseline_faithful.yaml) is the K&D-pure reference (linear reward, K&D-faithful proximity encoding, 960² scale). Untouched; do not mutate without a documented reason.

## Runtime

[`runtime/`](runtime/) holds the ops overlay configs (checkpoint cadence, log interval, etc.). They overlay the science config at run time and are device-specific (`mac.yaml`, `gcp_l4.yaml`, etc.).

## Archive

[`archive/`](archive/) holds superseded and deferred configs. Indexed chronologically in [`archive/README.md`](archive/README.md).

---

## Adding a new config

1. Pick the right axis folder (`axis1/`, `axis2/`, `axis12/`, or create a new axis directory).
2. Pick the right tier (or add a new one — but the four canonical tiers cover most needs).
3. Inherit scaffold/world settings from the existing same-tier config in another axis as a starting point.
4. Use `food_growth_rate_at_960sq` (scale-relative) for any non-960² world — the resolver in [src/config_utils.py](../src/config_utils.py) handles area scaling.
5. Set `replay_bucket: "evo-reward-replays-public"` to upload replays.
6. Update the parent axis's `README.md` if you've added a tier.

## Promoting / archiving

When a config moves status:
- **Active → Archive:** `git mv configs/<axis>/<tier>.yaml configs/archive/<date>_<descriptor>.yaml`. Update [`archive/README.md`](archive/README.md) chronologically with a one-line outcome.
- **Archive → Active:** reverse. Update both READMEs.

Downstream lists that may also need updates:
- `ARCHIVE_POLICY` in [scripts/archive_prune.py](../scripts/archive_prune.py).
- `ACTIVE_EXPS` in [dashboard/site/src/lib/replayNaming.ts](../dashboard/site/src/lib/replayNaming.ts).
