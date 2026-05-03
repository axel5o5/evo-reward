# axis2 — Social heading observation

## What this axis tests

Whether richer per-bin proximity-sensor encoding (specifically, *direct* approach-toward-me information + speed) helps prey/predator policies learn evasion and hunting faster than K&D's winner-take-all distance-only encoding.

## Mechanism (the diff vs `baseline_faithful.yaml`)

`proximity_encoding: "distance_approach_speed"` — every proximity bin reports four values per species (instead of just one):

| Channel | What | Why |
|---|---|---|
| `distance` | edge-to-edge distance to closest of-species agent in this bin (clipped to `proximity_max_range`, normalized to [0,1]) | same as K&D |
| `sin_approach`, `cos_approach` | `sin(α)`, `cos(α)` where `α = other_heading − bearing_from_other_to_me` | **`cos_approach = +1`** directly encodes "moving toward me." `cos_approach = −1` is "moving away." `sin_approach = ±1` is "perpendicular to bearing line." |
| `speed` | `|velocity_xy|` of the closest agent, normalized by `max_motor_norm` | distinguishes a stationary prey from a fast-fleeing one |

Total channels per bin: 4 (prey) + 4 (predator) + 1 (food) + 1 (wall) = **10**. With 32 bins: `obs_dim = 32 × 10 + 72 (tactile) + 5 (self) = 397`.

## Why approach-angle (and not just heading)

The earlier "distance_and_heading" encoding (still selectable for backward compat) reports `sin/cos` of `other_heading − my_heading` — i.e., the egocentric *body orientation* of the seen agent, NOT its approach angle relative to me. The information needed to derive "approaching me" is implicitly there (combine bin position + heading), but the policy has to *learn* the combination. The new encoding makes "approaching me" a directly observable channel, expected to speed convergence.

See findings.md §15.24 for the math derivation, the v8 motivation, and the decision to make this default.

## Tiers

| File | Scale | Use |
|---|---|---|
| [`tiny.yaml`](tiny.yaml)   | 600² / 200/20 caps | Cheap iteration |
| [`small.yaml`](small.yaml) | 750² / 300/35 caps | Overnight runs |
| [`med.yaml`](med.yaml)     | 880² / 375/40 caps | Production |
| [`full.yaml`](full.yaml)   | 960² / 450/50 caps | Paper-faithful K&D scale |

All tiers carry the v10 mechanism additions and §15.22 scaffold retune.

## Ancestry

The original axis-2 design (findings.md §12) used `social_obs: position_heading_velocity` — a separate "neighbor list" social block. It made it to 1M steps and then trophic-collapsed. §13 redesigned to bin-aligned heading channels (the "distance_and_heading" predecessor of the current encoding). §15.24 added approach-angle math + speed magnitude — the version live in these configs.
