"""
test_sac_sim_step.py
--------------------
End-to-end smoke test for src/jax_sim_sac.py — the SAC-aware
simulation step. Verifies that the full pipeline:

    init_simstate() + init_sacstate()
      → repeated sim_step_core_sac(sim_state, sac_state)
      → optional sac_runtime["step_update"](sac_state, ...)

runs without errors against a real config (configs/baseline/tiny.yaml)
and that state evolves as expected:

  - sim_state.step counter advances.
  - sac_state.replay_size grows for active slots, stays at 0 for inactive.
  - After enough warmup steps to clear sac_replay_min_size,
    sac_runtime["step_update"] succeeds and shifts at least one
    actor/critic param tensor.

Uses the baseline `tiny` tier (180 prey cap + 20 pred cap = 200 slots)
which is the smallest production config. Test still takes a few seconds
because each step does a full physics solve + neural-net forward pass
for 200 slots.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.config_utils import resolve_scale_dependent_params
from src.jax_state import init_simstate
from src.sac_state import init_sacstate
from src.sac_runtime import build_sac_runtime
from src.jax_sim_sac import build_sim_step_sac
from src.environment import _build_physics


def _tiny_config():
    """Load configs/baseline/tiny.yaml + override SAC knobs to keep the
    test cheap (tiny replay, small warmup)."""
    with open(_ROOT / "configs" / "baseline" / "tiny.yaml") as f:
        cfg = yaml.safe_load(f)
    # Resolve scale-dependent fields the same way the runner does.
    resolve_scale_dependent_params(cfg)

    # SAC-specific overrides.
    cfg["learner_type"] = "sac"
    cfg["sac_replay_capacity"] = 64           # small ring → fast test
    cfg["sac_replay_min_size"] = 16
    cfg["sac_minibatch_size"] = 16
    cfg["sac_target_tau"] = 0.05
    cfg["sac_initial_log_alpha"] = -2.0
    # No age-keyed LR schedule for the test — keep updates deterministic-ish.
    cfg.setdefault("lr_schedule_enable", False)
    return cfg


def _build_states(cfg, rng_key):
    sim_key, sac_key = jax.random.split(rng_key)
    sim_state = init_simstate(cfg, sim_key)
    sac_state = init_sacstate(cfg, sac_key)
    space, _ = _build_physics(
        cfg, n_agent_slots=cfg["prey_cap"] + cfg["predator_cap"],
    )
    sim_step = build_sim_step_sac(cfg, space)
    runtime = build_sac_runtime(cfg)
    return sim_state, sac_state, sim_step, runtime


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sim_step_runs_and_advances_state():
    """Five sim_step_core_sac calls on a real config — no errors, state moves."""
    cfg = _tiny_config()
    sim_state, sac_state, sim_step, _runtime = _build_states(cfg, jax.random.PRNGKey(0))

    initial_step = int(sim_state.step)
    initial_size_total = int(sac_state.replay_size.sum())
    assert initial_step == 0
    assert initial_size_total == 0

    for _ in range(5):
        sim_state, sac_state = sim_step(sim_state, sac_state)

    assert int(sim_state.step) == initial_step + 5
    # All initially-active agents should have written something each step.
    active_count = int(sim_state.is_active.sum())
    # Replays grow by 1 per active-pre-step. After 5 steps, ≈ 5×active
    # write events total. Slight slack for early deaths.
    grown = int(sac_state.replay_size.sum())
    assert grown > 0
    assert grown <= 5 * active_count, (
        f"replay sum {grown} > 5 × {active_count}; ring write overran"
    )


def test_replay_only_grows_for_active_slots():
    """Inactive slots (initial reserves) shouldn't accumulate replay entries."""
    cfg = _tiny_config()
    sim_state, sac_state, sim_step, _runtime = _build_states(cfg, jax.random.PRNGKey(1))

    pre_active = np.asarray(sim_state.is_active).copy()
    for _ in range(3):
        sim_state, sac_state = sim_step(sim_state, sac_state)

    sizes = np.asarray(sac_state.replay_size)
    # Any slot that was inactive at start AND remained inactive (no birth
    # filled it during the 3 steps) must have replay_size == 0.
    post_active = np.asarray(sim_state.is_active)
    never_active = (~pre_active) & (~post_active)
    assert (sizes[never_active] == 0).all(), (
        f"inactive slot wrote replay: "
        f"{int((sizes[never_active] != 0).sum())} violators"
    )


def test_update_shifts_actor_after_warmup():
    """After enough sim steps to clear sac_replay_min_size, calling
    runtime['step_update'] should change at least one actor param."""
    cfg = _tiny_config()
    sim_state, sac_state, sim_step, runtime = _build_states(cfg, jax.random.PRNGKey(2))

    # Run sim steps until at least one agent's replay is past min_size.
    min_size = cfg["sac_replay_min_size"]
    steps_run = 0
    while int(sac_state.replay_size.max()) < min_size and steps_run < 50:
        sim_state, sac_state = sim_step(sim_state, sac_state)
        steps_run += 1
    assert int(sac_state.replay_size.max()) >= min_size, (
        f"replay didn't warm up after {steps_run} steps"
    )

    # Snapshot the actor params for the slot with the fullest replay,
    # then apply one update and verify that slot's actor changed.
    target_slot = int(jnp.argmax(sac_state.replay_size))
    actor_pre = jax.tree_util.tree_map(
        lambda x: np.asarray(x[target_slot]).copy(), sac_state.actor_params,
    )

    sac_state = runtime["step_update"](
        sac_state, sim_state.is_active, sim_state.ages, sim_state.species,
        jax.random.PRNGKey(99),
    )

    actor_post = jax.tree_util.tree_map(
        lambda x: np.asarray(x[target_slot]), sac_state.actor_params,
    )
    pre_leaves = jax.tree_util.tree_leaves(actor_pre)
    post_leaves = jax.tree_util.tree_leaves(actor_post)
    any_changed = any(
        not np.array_equal(a, b) for a, b in zip(pre_leaves, post_leaves)
    )
    assert any_changed, (
        f"slot {target_slot}: actor params unchanged after step_update "
        f"(replay_size at update = {int(sac_state.replay_size[target_slot])})"
    )


if __name__ == "__main__":
    test_sim_step_runs_and_advances_state();    print("ok: sim step advances")
    test_replay_only_grows_for_active_slots();  print("ok: inactive slots untouched")
    test_update_shifts_actor_after_warmup();    print("ok: update shifts actor post-warmup")
    print("\nAll SAC sim_step (chunk-3) tests passed.")
