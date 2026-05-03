# Analysis quickstart — pulling and inspecting saved runs

A short guide for collaborators (or future-you) who want to inspect a run's results without setting up the training stack from scratch. Last updated 2026-05-03.

## Access

- **Checkpoints** (`gs://evo-reward-ckpts/`) — private. Ask Axel for IAM `objectViewer` on your Google account.
- **Replays** (`gs://evo-reward-replays-public/`) — already public; no auth needed.

Install the Google Cloud SDK once (`brew install --cask google-cloud-sdk`), then `gcloud auth login`.

## Bucket layout

```
gs://evo-reward-ckpts/results/<exp_name>/seed_<n>/<run_tag>/
    progress.json           — live status, refreshed every interval (~50 KB)
    metrics.npz             — time-series of aggregates (~few MB)
    checkpoints/
        step_NNNNNNNN.npz   — full SimState pytree (~125 MB each)
    replays/                — viz-only (cross-mirrored to public bucket)
```

The currently-running axis-1 run is at:
```
gs://evo-reward-ckpts/results/axis1_residual/seed_0/2026-05-01T2203Z/
```

## Pulling data

```bash
RUN=gs://evo-reward-ckpts/results/axis1_residual/seed_0/2026-05-01T2203Z

# Quick look — latest progress.json (population, weights, energies)
gsutil cp $RUN/progress.json /tmp/

# List checkpoints, then grab the ones you want
gsutil ls $RUN/checkpoints/
gsutil cp $RUN/checkpoints/step_02900000.npz /tmp/
gsutil cp $RUN/checkpoints/step_02400000.npz /tmp/
```

Checkpoints are ~125 MB each — pull selectively, not the whole directory.

## Setting up the repo

You need the codebase to load checkpoints (the loader uses `init_simstate` as a pytree template).

```bash
git clone <repo-url> evo-reward && cd evo-reward
uv pip install -e .         # or: pip install -e .
```

## Running the inspection script

[`scripts/inspect_checkpoint.py`](../scripts/inspect_checkpoint.py) handles the most common things you'll want to look at — population, ages, reward weights, residual MLP utilization, and (in two-checkpoint mode) cohort survival between saves.

```bash
# Snapshot of one checkpoint
python scripts/inspect_checkpoint.py /tmp/step_02900000.npz

# Diff two checkpoints (drift + cohort survival)
python scripts/inspect_checkpoint.py /tmp/step_02400000.npz /tmp/step_02900000.npz
```

The script prints, per active species at each checkpoint:
- `n` (count), age distribution (median/p75/max), energy stats
- Mean/std of all four reward weights `[w_eat, w_act, w_prey, w_pred]`
- Residual MLP L1 norm (sum of |params| per agent — see §15.18 for trajectory)

In two-checkpoint mode it also prints:
- Drift in each weight mean
- Drift in residual L1 mean
- Cohort survival (how many of the agents alive at the older save are still alive at the newer one)

### Backward compatibility caveat

v8-era checkpoints (the current axis-1 run) were saved before v10 added four
death-age ring fields to `SimState`. The script handles this transparently —
it strips those leaves from the v10 template when unflattening — so the same
script works against both eras. If you see `assert n == len(v8_idxs)`, the
checkpoint is from yet another era and the loader needs another conditional.

## What to read for context

- [`docs/CURRENT_STATE.md`](CURRENT_STATE.md) — where things stand right now, what's running, what's queued.
- [`docs/findings.md`](findings.md) — historical notes. Most relevant for current analysis:
    - **§15.18** — first evidence of residual MLP utilization (axis-1 Q1)
    - **§15.19** — v10 design block (mouth widening, age-keyed LR, death-age logging)
    - **§15.20** — three diagnostic analyses + revised L1/L2/L3 ladder
    - **§15.21** — three follow-up analyses on v8 (cohort survival, mouth bin, catch-rate)

The §15.20-§15.21 analyses are the closest precedent for this kind of inspection work and are worth skimming before designing your own.

## Going beyond the script

The inspection script covers ~80% of the recurring questions. For the deeper analyses (tactile-bin near-miss, SVD of policy `W_in`, counterfactual PPO), the entry points are:

- **Positions/headings** — `state.phyjax_stated["circle"].p.xy` and `state.phyjax_stated["circle"].p.angle`. The tactile-bin convention is in [`src/jax_food.py:143-155`](../src/jax_food.py#L143-L155) (with the π/2 offset for emevo's heading convention).
- **Per-agent rollout** — `state.rollout_obs[i]` (1024 × 205), `rollout_actions[i]` (1024 × 2), `rollout_rewards[i]` (1024,) — the last 1024 step tuples for each agent. Note: this is a circular buffer; see `state.rollout_ptrs[i]` for the next-write index.
- **Policy params** — `state.params["params"]["Dense_0"]["kernel"]` is the input-layer weight matrix, shape (max_agents, 205, hidden_size). Useful for SVD of policy capacity.
- **Residual MLP params** — `state.reward_mlp_params["params"]["Dense_0"]` and `Dense_1` (hidden=4 by default → 25 params per agent total).

For history of how particular metrics were used, search [`docs/findings.md`](findings.md) for the metric name or the section reference.
