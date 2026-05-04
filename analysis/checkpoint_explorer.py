"""
checkpoint_explorer.py
----------------------
Interactive helpers for loading and inspecting saved SimState checkpoints.

A SimState checkpoint is `np.savez`'d as a flat list of leaves (`leaf_0` ..
`leaf_N`) plus a signature key — the field names are not stored. To get
back named, navigable JAX arrays you have to reconstruct the pytree from
a template SimState built with a config matching the run's shapes. This
module wraps that loader, adds `gs://` fetching, and exposes a few
small helpers for poking at the loaded state.

Typical use, in a notebook or REPL:

    from analysis.checkpoint_explorer import load, summary, agents_table, flat, leaf_guide

    s = load("ckpts/axis1_residual/step_00360000.npz")
    # or directly from the bucket:
    s = load("gs://evo-reward-ckpts/results/axis1_residual/seed_0/"
             "2026-05-01T2203Z/checkpoints/step_00360000.npz")

    summary(s)                  # one-screen population / weight stats
    s.is_active                 # (max_agents,) bool
    s.policy_params             # nested dict — Dense_0..3, log_std
    s.reward_weights            # (max_agents, 4) — w_eat, w_act, w_prey, w_pred

    rows = agents_table(s)      # dict of 1-D arrays, one row per active agent
    everything = flat(s)        # dict[".path['k']": ndarray] for ad-hoc digging
    leaf_guide()                # print the leaf-index → field-name reference

The default template config is `configs/axis1/med.yaml` (prey_cap 375,
predator_cap 40, food_max 525, obs_dim 205, hidden 64, rollout 1024).
Override with `config=` for runs with different shapes.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.jax_state import SimState, init_simstate
from src.config_utils import resolve_scale_dependent_params


DEFAULT_CONFIG = "configs/axis1/med.yaml"

# v10 added a death-age ring buffer; pre-v10 saves are 4 leaves shorter.
_V10_ONLY_FIELDS = {
    "death_age_ring_prey", "death_age_ring_pred",
    "death_age_idx_prey", "death_age_idx_pred",
}

_CACHE_ROOT = Path(
    os.environ.get(
        "EVO_REWARD_CACHE_DIR",
        Path.home() / ".cache" / "evo-reward" / "checkpoints",
    )
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_config(config) -> dict:
    if isinstance(config, dict):
        cfg = dict(config)
    else:
        path = config or DEFAULT_CONFIG
        if not os.path.isabs(path):
            path = str(_PROJECT_ROOT / path)
        with open(path) as f:
            cfg = yaml.safe_load(f)
    resolve_scale_dependent_params(cfg)
    return cfg


def _gs_to_local(uri: str) -> str:
    """Download a gs:// blob into the LRU cache and return the local path.

    Idempotent — reuses any existing cached copy. Shells out to `gsutil`
    so we ride the user's existing gcloud credentials, matching the
    pattern in scripts/replay_cache.py.
    """
    assert uri.startswith("gs://"), uri
    dst = _CACHE_ROOT / uri[len("gs://"):]
    if dst.exists():
        return str(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gsutil", "cp", uri, str(dst)], check=True)
    return str(dst)


def _build_template(config: dict) -> SimState:
    return init_simstate(config, jax.random.PRNGKey(int(config.get("seed", 0))))


def load(path: str, config=None) -> SimState:
    """Load a saved checkpoint as a fully reconstructed SimState pytree.

    `path` is either a local file path or a `gs://` URI; gs:// URIs are
    downloaded into ~/.cache/evo-reward/checkpoints/ on first access.

    `config` controls the template SimState used for tree-unflatten — pass
    a path or a parsed dict to override the default. The template's
    per-agent shapes (prey_cap, predator_cap, hidden size, rollout_steps,
    obs_dim) must match what the run was using; otherwise unflatten fails.

    Pre-v10 checkpoints (those missing the death-age ring) are
    transparently backfilled with the template's ring fields.
    """
    if path.startswith("gs://"):
        path = _gs_to_local(path)

    cfg = _read_config(config)
    template = _build_template(cfg)
    full_leaves, treedef = jtu.tree_flatten(template)
    paths = jtu.tree_flatten_with_path(template)[0]

    data = np.load(path, allow_pickle=False)
    n = sum(1 for k in data.files if k.startswith("leaf_"))
    loaded = [data[f"leaf_{i}"] for i in range(n)]

    if n == len(full_leaves):
        merged = loaded
    else:
        kept = [i for i, (p, _) in enumerate(paths)
                if getattr(p[0], "name", None) not in _V10_ONLY_FIELDS]
        if n != len(kept):
            raise ValueError(
                f"{path}: leaf count {n} matches neither the current schema "
                f"({len(full_leaves)}) nor the v8 schema ({len(kept)}). "
                f"Pass config= for the run's actual tier/axis."
            )
        merged = list(full_leaves)
        for src_i, leaf in enumerate(loaded):
            merged[kept[src_i]] = leaf

    return jtu.tree_unflatten(treedef, [jnp.asarray(l) for l in merged])


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def flat(state: SimState) -> dict:
    """Map every leaf in `state` to its keypath string (e.g.
    `.policy_params['params']['Dense_0']['kernel']`) → numpy array.
    """
    leaves = jtu.tree_flatten_with_path(state)[0]
    return {jtu.keystr(p): np.asarray(l) for p, l in leaves}


def leaf_index(state: SimState) -> list:
    """Return [(i, path_str, shape, dtype), ...] for every leaf in `state`."""
    leaves = jtu.tree_flatten_with_path(state)[0]
    return [
        (i, jtu.keystr(p), tuple(np.asarray(l).shape), np.asarray(l).dtype)
        for i, (p, l) in enumerate(leaves)
    ]


def leaf_guide(config=None, state: SimState | None = None) -> None:
    """Print the leaf-index → field-name reference for the current schema.

    Use when you're working with the raw .npz keys (`leaf_42`, etc.) and
    want to know which SimState field that corresponds to. Pass `state`
    if you've already loaded one — otherwise a fresh template is built.
    """
    s = state if state is not None else _build_template(_read_config(config))
    for i, path, shape, dtype in leaf_index(s):
        print(f"leaf_{i:>3}  {str(shape):20}  {str(dtype):8}  {path}")


def residual_l1_per_agent(state: SimState):
    """L1 norm of the residual reward MLP genome per agent slot, or None
    for runs without a residual genome (linear reward type)."""
    leaves = jtu.tree_leaves(state.reward_mlp_params)
    if not leaves:
        return None
    n = int(np.asarray(state.is_active).shape[0])
    out = np.zeros(n)
    for leaf in leaves:
        out += np.abs(np.asarray(leaf).reshape(leaf.shape[0], -1)).sum(axis=1)
    return out


def ppo_updates_per_agent(state: SimState, config=None) -> np.ndarray:
    """Number of PPO updates each agent has received since its birth.

    `policy_opt_states[0].count` is the per-agent Adam step counter, reset
    to 0 at birth (init_policy is called fresh in handle_birth_jax). One PPO
    trigger fires `ppo_epochs * (rollout_steps / minibatch_size)` Adam
    steps, so dividing the count by that product gives the number of PPO
    update events the agent has lived through. Values for inactive slots
    are still returned (they reflect the previous occupant's life).
    """
    cfg = _read_config(config)
    n_minibatches = cfg["rollout_steps"] // cfg["minibatch_size"]
    adam_per_event = cfg["ppo_epochs"] * n_minibatches
    counts = np.asarray(state.policy_opt_states[0].count)
    return counts // adam_per_event


def agents_table(state: SimState, active_only: bool = True) -> dict:
    """Per-agent fields gathered into a dict of 1-D numpy arrays.

    Columns: slot, is_active, species, agent_id, parent_id, age, energy,
    x, y, angle, w_eat, w_act, w_prey, w_pred, residual_l1 (NaN for
    linear-only runs). Drop-in for `pandas.DataFrame(agents_table(s))`.
    """
    is_active = np.asarray(state.is_active)
    rw = np.asarray(state.reward_weights)

    circle = state.phyjax_stated.get("circle") \
        if hasattr(state.phyjax_stated, "get") else state.phyjax_stated.circle
    xy = np.asarray(circle.p.xy)
    angle = np.asarray(circle.p.angle)

    l1 = residual_l1_per_agent(state)
    if l1 is None:
        l1 = np.full(rw.shape[0], np.nan)

    cols = {
        "slot": np.arange(rw.shape[0]),
        "is_active": is_active,
        "species": np.asarray(state.species),
        "agent_id": np.asarray(state.agent_ids),
        "parent_id": np.asarray(state.parent_ids),
        "age": np.asarray(state.ages),
        "energy": np.asarray(state.energies),
        "x": xy[:, 0],
        "y": xy[:, 1],
        "angle": angle,
        "w_eat": rw[:, 0],
        "w_act": rw[:, 1],
        "w_prey": rw[:, 2],
        "w_pred": rw[:, 3],
        "residual_l1": l1,
        "ppo_updates": ppo_updates_per_agent(state),
    }
    if active_only:
        return {k: v[is_active] for k, v in cols.items()}
    return cols


def summary(state: SimState) -> None:
    """Print a one-screen summary: step + cumulative counters, then per-species
    population, age, energy, reward-weight, and residual-L1 stats."""
    is_active = np.asarray(state.is_active)
    species = np.asarray(state.species)
    ages = np.asarray(state.ages)
    energies = np.asarray(state.energies)
    rw = np.asarray(state.reward_weights)

    print(f"step={int(state.step):,}  next_id={int(state.next_agent_id):,}")
    print(f"cum_catches={int(state.cum_catches):,}  "
          f"cum_deaths={int(state.cum_deaths):,}  "
          f"cum_feedings={int(state.cum_feedings):,}")

    l1 = residual_l1_per_agent(state)

    for sp_name, sp in [("Predators", 1), ("Prey", 0)]:
        m = is_active & (species == sp)
        n = int(m.sum())
        if n == 0:
            print(f"\n{sp_name}: n=0")
            continue
        a = ages[m]
        e = energies[m]
        w = rw[m]
        print(f"\n{sp_name} (n={n})")
        print(f"  age:    median={int(np.median(a)):>7,d}  "
              f"p75={int(np.percentile(a,75)):>7,d}  max={int(a.max()):>7,d}")
        print(f"  energy: min={e.min():>5.1f}  mean={e.mean():>5.1f}  "
              f"max={e.max():>5.1f}")
        for j, name in enumerate(["w_eat ", "w_act ", "w_prey", "w_pred"]):
            print(f"  {name}: mean={w[:,j].mean():+.3f}  std={w[:,j].std():.3f}")
        if l1 is not None:
            l1_sp = l1[m]
            print(f"  residual L1: mean={l1_sp.mean():.3f}  "
                  f"median={np.median(l1_sp):.3f}  max={l1_sp.max():.3f}")
