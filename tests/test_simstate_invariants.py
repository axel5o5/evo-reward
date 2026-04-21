"""
test_simstate_invariants.py
---------------------------
Structural invariants for a freshly initialised SimState.

These tests would have caught the D19 slot bug (predators inhabiting
prey-sized physics bodies) immediately — the mismatch between
`state.radii` and `space.shaped.circle.radius` is a 1-line assertion.
"""

import jax
import pytest

from src.jax_state import init_simstate
from src.environment import _build_physics

# Reuse the same small_config fixture layout used by test_phase0.
from tests.test_phase0 import config, small_config  # noqa: F401


def _state_and_space(cfg):
    space, _ = _build_physics(cfg)
    state = init_simstate(cfg, jax.random.PRNGKey(0))
    return state, space


def test_initial_predators_in_predator_slot_range(small_config):
    """Every active species==1 slot must lie in [prey_cap, max_agents)."""
    state, _ = _state_and_space(small_config)
    prey_cap = small_config["prey_cap"]
    pred_active = (state.species == 1) & state.is_active
    pred_slots = [i for i, a in enumerate(pred_active) if bool(a)]
    assert len(pred_slots) == small_config["predator_initial"]
    for slot in pred_slots:
        assert slot >= prey_cap, f"predator in prey slot {slot}"


def test_initial_prey_in_prey_slot_range(small_config):
    """Every active species==0 slot must lie in [0, prey_cap)."""
    state, _ = _state_and_space(small_config)
    prey_cap = small_config["prey_cap"]
    prey_active = (state.species == 0) & state.is_active
    prey_slots = [i for i, a in enumerate(prey_active) if bool(a)]
    assert len(prey_slots) == small_config["prey_initial"]
    for slot in prey_slots:
        assert slot < prey_cap, f"prey in predator slot {slot}"


def test_simstate_radii_match_physics_radii_per_slot(small_config):
    """Per-slot radii in SimState must match the phyjax2d body radii.

    Physics body radii are bound to slot index at builder time. If
    state.radii[i] disagrees with the phyjax body at slot i, collision
    detection, mass, and inertia will all be wrong for that slot.
    This is *the* invariant the D19 bug violated.
    """
    state, space = _state_and_space(small_config)
    phys_radii = space.shaped.circle.radius
    assert state.radii.shape == phys_radii.shape
    mismatch = [
        i
        for i in range(state.radii.shape[0])
        if float(state.radii[i]) != float(phys_radii[i])
    ]
    assert mismatch == [], f"radii mismatches at slots: {mismatch[:10]}"


def test_inactive_slot_counts_match_caps(small_config):
    """Active prey ∈ [0, n_prey), inactive prey ∈ [n_prey, prey_cap),
    active predator ∈ [prey_cap, prey_cap+n_pred), inactive predator
    ∈ [prey_cap+n_pred, max_agents)."""
    state, _ = _state_and_space(small_config)
    n_prey = small_config["prey_initial"]
    n_pred = small_config["predator_initial"]
    prey_cap = small_config["prey_cap"]
    max_agents = prey_cap + small_config["predator_cap"]

    for i in range(n_prey):
        assert bool(state.is_active[i])
        assert int(state.species[i]) == 0
    for i in range(n_prey, prey_cap):
        assert not bool(state.is_active[i])
        assert int(state.species[i]) == 0
    for i in range(prey_cap, prey_cap + n_pred):
        assert bool(state.is_active[i])
        assert int(state.species[i]) == 1
    for i in range(prey_cap + n_pred, max_agents):
        assert not bool(state.is_active[i])
        assert int(state.species[i]) == 1


def test_act_ratio_matches_species_not_slot(small_config):
    """act_ratio scales force by (pred_r/prey_r)^2 for predators.
    With D19 this must follow species, not slot index."""
    state, _ = _state_and_space(small_config)
    prey_cap = small_config["prey_cap"]
    prey_r = small_config["prey_radius"]
    pred_r = small_config["predator_radius"]
    expected_ratio = (pred_r ** 2) / (prey_r ** 2)
    for i in range(state.act_ratio.shape[0]):
        ratio = float(state.act_ratio[i, 0])
        if i < prey_cap:
            assert abs(ratio - 1.0) < 1e-5, f"prey slot {i} ratio={ratio}"
        else:
            assert abs(ratio - expected_ratio) < 1e-4, f"pred slot {i} ratio={ratio}"


def test_predator_eat_timer_initial_zero(small_config):
    """All slots start at 0 (ready to eat) regardless of species."""
    state, _ = _state_and_space(small_config)
    assert state.predator_eat_timer.shape == state.is_active.shape
    assert int(state.predator_eat_timer.sum()) == 0
