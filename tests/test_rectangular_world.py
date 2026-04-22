"""
test_rectangular_world.py
-------------------------
D25 regression: the simulation must run with a rectangular world
(e.g. emevo's 1200×600 predator TOML) without any hard-coded square
assumption breaking. Also pins the new `world_bounds` helper's
back-compat behavior for three config shapes:
  * explicit rect: world_size_x + world_size_y
  * explicit tuple: world_size = [x, y]
  * scalar (legacy): world_size = N
"""

import jax
import jax.numpy as jnp
import pytest
import yaml
from pathlib import Path

from src.environment import world_bounds, _build_physics
from src.jax_state import init_simstate
from src.jax_sim import build_sim_step


ROOT = Path(__file__).resolve().parents[1]


def _load(path):
    with open(ROOT / path) as f:
        return yaml.safe_load(f) or {}


# ---- world_bounds helper semantics --------------------------------------


def test_world_bounds_explicit_rectangle():
    cfg = {"world_size_x": 1200, "world_size_y": 600}
    assert world_bounds(cfg) == (1200.0, 600.0)


def test_world_bounds_explicit_rectangle_overrides_scalar():
    cfg = {"world_size": 999, "world_size_x": 1200, "world_size_y": 600}
    assert world_bounds(cfg) == (1200.0, 600.0)


def test_world_bounds_tuple_in_world_size():
    cfg = {"world_size": [1200, 600]}
    assert world_bounds(cfg) == (1200.0, 600.0)


def test_world_bounds_scalar_world_size():
    """Legacy path: single-number world_size means square world."""
    cfg = {"world_size": 960}
    assert world_bounds(cfg) == (960.0, 960.0)


# ---- end-to-end with the endpoint config --------------------------------


@pytest.fixture
def endpoint_config():
    cfg = _load("configs/baseline_endpoint.yaml")
    # Overlay the default runtime so keys like ppo_* and rollout_steps exist.
    cfg.update(_load("configs/runtime/default.yaml"))
    return cfg


def test_endpoint_config_bounds(endpoint_config):
    x, y = world_bounds(endpoint_config)
    assert (x, y) == (1200.0, 600.0)


def test_endpoint_init_simstate_places_agents_within_rectangle(endpoint_config):
    """Initial agent positions must fall inside [0, 1200] × [0, 600]."""
    state = init_simstate(endpoint_config, jax.random.PRNGKey(0))
    pos = state.phyjax_stated.get("circle").p.xy
    active_pos = pos[state.is_active]
    assert active_pos.shape[0] > 0
    assert float(active_pos[:, 0].min()) >= 0.0
    assert float(active_pos[:, 0].max()) <= 1200.0
    assert float(active_pos[:, 1].min()) >= 0.0
    assert float(active_pos[:, 1].max()) <= 600.0
    # And sanity: someone uses the long axis
    assert float(active_pos[:, 0].max()) > 600.0, (
        "no agent spawned in x > 600 — we may be accidentally clamping to "
        "the smaller axis"
    )


def test_endpoint_food_positions_within_rectangle(endpoint_config):
    state = init_simstate(endpoint_config, jax.random.PRNGKey(0))
    fp = state.food_positions
    assert float(fp[:, 0].max()) <= 1200.0
    assert float(fp[:, 1].max()) <= 600.0
    assert float(fp[:, 0].max()) > 600.0, "food not spawning in long axis"


def test_endpoint_sim_step_runs(endpoint_config):
    """sim_step_core must complete a few steps without shape / dtype errors.
    This catches plumbing regressions (e.g., the old world_size scalar was
    used inside JIT'd functions that would crash on a rectangular config)."""
    space, _ = _build_physics(endpoint_config)
    step_fn, _ = build_sim_step(endpoint_config, space)
    state = init_simstate(endpoint_config, jax.random.PRNGKey(0))
    for _ in range(5):
        state = step_fn(state)
    jax.block_until_ready(state.step)
    assert int(state.step) == 5


# ---- baseline_faithful stays exactly as before ---------------------------


@pytest.fixture
def faithful_config():
    cfg = _load("configs/baseline_faithful.yaml")
    cfg.update(_load("configs/runtime/default.yaml"))
    return cfg


def test_faithful_config_still_square(faithful_config):
    """Sanity: paper-faithful config keeps 960×960 square world after the
    D25 helper is introduced."""
    assert world_bounds(faithful_config) == (960.0, 960.0)


def test_faithful_init_unchanged(faithful_config):
    """D25 is purely additive: square-world init must yield the same
    populations as before."""
    state = init_simstate(faithful_config, jax.random.PRNGKey(0))
    n_prey = int(((state.species == 0) & state.is_active).sum())
    n_pred = int(((state.species == 1) & state.is_active).sum())
    assert n_prey == faithful_config["prey_initial"]
    assert n_pred == faithful_config["predator_initial"]
