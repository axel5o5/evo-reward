"""
jax_metrics.py
--------------
Time-series metrics persistence for the JAX runner.

The legacy `src/metrics.py::MetricsLog` was built for the pre-JAX runner
that operated on WorldState + agent lists. This module is its direct
analogue for SimState, with the same field names so downstream tools
(`scripts/validate_replication.py`, the dashboard) work unchanged.

On-disk format:
  results/<experiment_name>/seed_<N>/metrics.npz
  keys: see JaxMetrics dataclass fields below; values are numpy arrays.

Call `record(log, sim_state)` every `log_interval_steps` and
`save(log, path)` every `checkpoint_interval_steps`. On `--resume`,
`load(path)` restores prior history so the saved time series is
continuous across restarts.
"""

import os
from dataclasses import dataclass, field, asdict

import jax.numpy as jnp
import numpy as np


@dataclass
class JaxMetrics:
    """Accumulated time-series data for a single seed's run.

    Field names match `src/metrics.py::MetricsLog` for compat with
    `scripts/validate_replication.py` and the dashboard loaders.
    """
    steps: list = field(default_factory=list)
    prey_population: list = field(default_factory=list)
    predator_population: list = field(default_factory=list)
    prey_mean_energy: list = field(default_factory=list)
    predator_mean_energy: list = field(default_factory=list)

    # Reward-weight stats — prey
    prey_mean_w_eat: list = field(default_factory=list)
    prey_mean_w_act: list = field(default_factory=list)
    prey_mean_w_prey: list = field(default_factory=list)
    prey_mean_w_pred: list = field(default_factory=list)
    prey_std_w_eat: list = field(default_factory=list)
    prey_std_w_act: list = field(default_factory=list)
    prey_std_w_prey: list = field(default_factory=list)
    prey_std_w_pred: list = field(default_factory=list)

    # Reward-weight stats — predator
    pred_mean_w_eat: list = field(default_factory=list)
    pred_mean_w_act: list = field(default_factory=list)
    pred_mean_w_prey: list = field(default_factory=list)
    pred_mean_w_pred: list = field(default_factory=list)
    pred_std_w_eat: list = field(default_factory=list)
    pred_std_w_act: list = field(default_factory=list)
    pred_std_w_prey: list = field(default_factory=list)
    pred_std_w_pred: list = field(default_factory=list)

    # Polynomial reward genome (reward_type=linear_plus_poly). 10 params each:
    # 4 quadratics [eat^2, act^2, prey^2, pred^2] + 6 pairwise interactions
    # [eat*act, eat*prey, eat*pred, act*prey, act*pred, prey*pred]. Per-species
    # mean+std at every log step. Stored even when reward_type != linear_plus_poly
    # (sim_state.reward_poly_params is allocated unconditionally), so the lists
    # will just be near-zero on non-poly runs — load-compatible across configs.
    prey_mean_poly: list = field(default_factory=list)  # shape (T, 10)
    prey_std_poly: list = field(default_factory=list)
    pred_mean_poly: list = field(default_factory=list)
    pred_std_poly: list = field(default_factory=list)


# Reward-weight layout: [w_eat, w_act, w_prey, w_pred]
_W_INDEX = {"eat": 0, "act": 1, "prey": 2, "pred": 3}


def record(log: JaxMetrics, sim_state) -> None:
    """Compute per-species stats from sim_state and append to log."""
    prey_mask = (sim_state.species == 0) & sim_state.is_active
    pred_mask = (sim_state.species == 1) & sim_state.is_active
    n_prey = int(jnp.sum(prey_mask))
    n_pred = int(jnp.sum(pred_mask))

    log.steps.append(int(sim_state.step))
    log.prey_population.append(n_prey)
    log.predator_population.append(n_pred)

    log.prey_mean_energy.append(
        float(jnp.mean(sim_state.energies[prey_mask])) if n_prey else 0.0
    )
    log.predator_mean_energy.append(
        float(jnp.mean(sim_state.energies[pred_mask])) if n_pred else 0.0
    )

    _record_reward_stats(log, sim_state.reward_weights[prey_mask], "prey", n_prey)
    _record_reward_stats(log, sim_state.reward_weights[pred_mask], "pred", n_pred)

    _record_poly_stats(log, sim_state.reward_poly_params[prey_mask], "prey", n_prey)
    _record_poly_stats(log, sim_state.reward_poly_params[pred_mask], "pred", n_pred)


def _record_reward_stats(log: JaxMetrics, weights, prefix: str, n: int) -> None:
    """Append mean and std for each reward-weight axis under log.{prefix}_...

    If n == 0 (no agents of this species), append zeros (matches legacy
    metrics.py behavior).
    """
    if n > 0:
        means = jnp.mean(weights, axis=0)
        stds = jnp.std(weights, axis=0)
        for name, idx in _W_INDEX.items():
            getattr(log, f"{prefix}_mean_w_{name}").append(float(means[idx]))
            getattr(log, f"{prefix}_std_w_{name}").append(float(stds[idx]))
    else:
        for name in _W_INDEX:
            getattr(log, f"{prefix}_mean_w_{name}").append(0.0)
            getattr(log, f"{prefix}_std_w_{name}").append(0.0)


def _record_poly_stats(log: JaxMetrics, poly_weights, prefix: str, n: int) -> None:
    """Append per-coordinate mean/std for the 10-dim poly genome.

    Stored as length-10 lists so each row of the saved (T, 10) array is one
    log step. On empty species, append zeros for shape consistency.
    """
    if n > 0:
        means = jnp.mean(poly_weights, axis=0)
        stds = jnp.std(poly_weights, axis=0)
        getattr(log, f"{prefix}_mean_poly").append(np.asarray(means).tolist())
        getattr(log, f"{prefix}_std_poly").append(np.asarray(stds).tolist())
    else:
        getattr(log, f"{prefix}_mean_poly").append([0.0] * 10)
        getattr(log, f"{prefix}_std_poly").append([0.0] * 10)


def save(log: JaxMetrics, path: str) -> None:
    """Atomic-write log to `path` as compressed .npz."""
    arrays = {k: np.asarray(v) for k, v in asdict(log).items()}
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def load(path: str) -> JaxMetrics:
    """Load `path` into a fresh JaxMetrics with Python lists (appendable).

    When a field was added after the run started (e.g. poly tracking added
    mid-flight), it'll be absent from the saved .npz. We backfill with NaN
    of the correct length so the lists stay index-aligned with `steps`.
    """
    data = np.load(path)
    log = JaxMetrics()
    for k in asdict(log):
        if k in data.files:
            setattr(log, k, data[k].tolist())
    n_steps = len(log.steps)
    if n_steps:
        for field_name in ("prey_mean_poly", "prey_std_poly",
                           "pred_mean_poly", "pred_std_poly"):
            cur = getattr(log, field_name)
            if len(cur) < n_steps:
                pad = [[float("nan")] * 10] * (n_steps - len(cur))
                setattr(log, field_name, pad + cur)
    return log


def metrics_path(out_dir: str, experiment_name: str, seed: int,
                 run_tag: str = "") -> str:
    """Canonical path for a seed's metrics.npz. Respects run_tag layout."""
    base = os.path.join(out_dir, experiment_name, f"seed_{seed}")
    if run_tag:
        base = os.path.join(base, run_tag)
    return os.path.join(base, "metrics.npz")
