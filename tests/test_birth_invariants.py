"""
test_birth_invariants.py
------------------------
Species→slot invariants preserved across process_births_and_deaths_jax.

D19 made the birth dispatcher species-aware: prey parents spawn into
[0, prey_cap); predator parents spawn into [prey_cap, max_agents).
These tests force a deterministic birth and verify the new slot lives
in the parent's species range and still agrees with the phyjax2d body
radius at that slot.
"""

import jax
import jax.numpy as jnp
import pytest

from src.jax_state import init_simstate
from src.jax_lifecycle import process_births_and_deaths_jax
from src.environment import _build_physics

from tests.test_phase0 import config, small_config  # noqa: F401


def _force_birth_for_slot(state, slot, energy=800.0):
    """Crank a single slot's energy high enough that the sigmoid birth
    probability is ~1. Everyone else gets energy 0 so death hazard
    doesn't also clobber them."""
    energies = jnp.zeros_like(state.energies).at[slot].set(energy)
    # Keep ages at 0 so the age-based hazard term is tiny.
    ages = jnp.zeros_like(state.ages)
    return state.replace(energies=energies, ages=ages)


def _forced_birth_config(cfg):
    """Boost kappa_b so a birth is guaranteed at high energy in a single
    call (normal kappa_b=1e-3 makes it probabilistic over many steps).
    This tests the *dispatch logic*, not the rate — the rate is pinned
    elsewhere in test_params_match_emevo.py."""
    cfg = dict(cfg)
    cfg["kappa_b"] = 1.0
    return cfg


def _run_births(state, config, rng_seed=0):
    state = state.replace(rng_key=jax.random.PRNGKey(rng_seed))
    return process_births_and_deaths_jax(state, config)


def test_prey_parent_spawns_in_prey_slot_range(small_config):
    """A prey parent's child must land at a slot < prey_cap."""
    state = init_simstate(small_config, jax.random.PRNGKey(0))
    prey_parent = 0  # slot 0 is prey
    state = _force_birth_for_slot(state, prey_parent)
    new_state = _run_births(state, _forced_birth_config(small_config), rng_seed=1)

    prey_cap = small_config["prey_cap"]
    newly_active = new_state.is_active & ~state.is_active
    new_slots = [i for i, a in enumerate(newly_active) if bool(a)]
    assert len(new_slots) >= 1, "expected at least one birth"
    for slot in new_slots:
        assert slot < prey_cap, f"prey offspring in predator slot {slot}"
        assert int(new_state.species[slot]) == 0


def test_predator_parent_spawns_in_predator_slot_range(small_config):
    """A predator parent's child must land at a slot >= prey_cap."""
    state = init_simstate(small_config, jax.random.PRNGKey(0))
    pred_parent = small_config["prey_cap"]  # first predator slot
    assert int(state.species[pred_parent]) == 1
    state = _force_birth_for_slot(state, pred_parent)
    new_state = _run_births(state, _forced_birth_config(small_config), rng_seed=7)

    prey_cap = small_config["prey_cap"]
    newly_active = new_state.is_active & ~state.is_active
    new_slots = [i for i, a in enumerate(newly_active) if bool(a)]
    assert len(new_slots) >= 1, "expected at least one predator birth"
    for slot in new_slots:
        assert slot >= prey_cap, f"predator offspring in prey slot {slot}"
        assert int(new_state.species[slot]) == 1


def test_birth_preserves_physics_radius_invariant(small_config):
    """After a birth the (state.radii, physics-body-radius) pair
    at the newborn slot must still agree. If we ever place a
    predator offspring in a prey-sized body, this fires."""
    space, _ = _build_physics(small_config)
    state = init_simstate(small_config, jax.random.PRNGKey(0))

    # Force one prey and one predator birth on the same step.
    prey_parent = 0
    pred_parent = small_config["prey_cap"]
    energies = jnp.zeros_like(state.energies)
    energies = energies.at[prey_parent].set(800.0)
    energies = energies.at[pred_parent].set(800.0)
    state = state.replace(
        energies=energies, ages=jnp.zeros_like(state.ages)
    )
    new_state = _run_births(state, _forced_birth_config(small_config), rng_seed=11)

    phys_radii = space.shaped.circle.radius
    newly_active = new_state.is_active & ~state.is_active
    new_slots = [i for i, a in enumerate(newly_active) if bool(a)]
    assert len(new_slots) >= 2, "expected both species to give birth"
    for slot in new_slots:
        assert float(new_state.radii[slot]) == float(phys_radii[slot]), (
            f"newborn slot {slot}: simstate radius "
            f"{float(new_state.radii[slot])} != physics "
            f"{float(phys_radii[slot])}"
        )


def test_species_free_slots_are_disjoint(small_config):
    """Free (inactive) slots must split cleanly: prey slots and predator
    slots never overlap. Any overlap would let the dispatcher pick a
    wrong-sized slot."""
    state = init_simstate(small_config, jax.random.PRNGKey(0))
    prey_cap = small_config["prey_cap"]
    inactive = ~state.is_active
    prey_free = [i for i in range(state.is_active.shape[0])
                 if bool(inactive[i]) and i < prey_cap]
    pred_free = [i for i in range(state.is_active.shape[0])
                 if bool(inactive[i]) and i >= prey_cap]
    assert set(prey_free).isdisjoint(pred_free)
    # Sanity: counts consistent with fresh init.
    assert len(prey_free) == prey_cap - small_config["prey_initial"]
    assert len(pred_free) == (
        small_config["predator_cap"] - small_config["predator_initial"]
    )
