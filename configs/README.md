# configs/

Science configs for the experiments. Layout reflects what's currently active vs. moved-on-from.

For full project context start at [docs/CURRENT_STATE.md](../docs/CURRENT_STATE.md).

## Active

Configs for runs we're launching now or queueing next.

| File | Status |
|---|---|
| [axis1_residual.yaml](axis1_residual.yaml) | **Live (v8)** — residual reward genome; running on `evo-reward-gpu`. |
| [axis2_aligned_smol.yaml](axis2_aligned_smol.yaml) | **Queued** — bin-aligned heading observation. Filename is historical (`experiment_name: axis2_aligned`, med-large scale). |

## Paper reference

| File | Status |
|---|---|
| [baseline_faithful.yaml](baseline_faithful.yaml) | Paper-faithful K&D config. Untouched reference; do not mutate without a documented reason. |

## Runtime

| Path | Status |
|---|---|
| [runtime/](runtime/) | Ops config (checkpoint cadence, log interval) — overlays the science config at run time. |

## Archive

[archive/](archive/) holds superseded and deferred configs. Not active going forward, but kept for git diffs, comparison runs, and historical context. See [archive/README.md](archive/README.md) for per-file status.

---

## Adding a new config

1. Drop it in `configs/` (top level) only if it's an active run target.
2. Inherit scaffold/world settings from `axis1_residual.yaml` unless you have a deliberate reason to deviate (document the deviation in a comment).
3. Use `food_growth_rate_at_960sq` (scale-relative) instead of `food_growth_rate` (absolute) for non-960² worlds — the resolver in [src/config_utils.py](../src/config_utils.py) handles the area scaling. `baseline_faithful.yaml` is the only config that uses the absolute form.
4. Set `replay_bucket: "evo-reward-replays-public"` if you want replays uploaded.

## Promoting / archiving

When a config moves status:
- **Active → Archive:** `git mv configs/<name>.yaml configs/archive/`, then update [archive/README.md](archive/README.md) with a one-line outcome (extincted, superseded, deferred, etc.).
- **Archive → Active:** the reverse. Update both READMEs.
