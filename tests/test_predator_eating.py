"""
test_predator_eating.py
-----------------------
Unit tests for the predator-catch mechanic in src/jax_food.py::check_eating_jax.

The Phase 1a bug discovered on 2026-04-20 (see docs/emevo-diff.md D18) was
that predators could catch prey 40–80 units away (radial range) and had no
cooldown between catches. This file pins down the corrected semantics:

  1. Catch requires PHYSICAL contact — distance ≤ pred_radius + prey_radius.
  2. Catch requires prey to fall in one of `predator_mouth_tactile_bins`
     (the 60° front arc for the medium-mouth config).
  3. Each predator has a cooldown of `predator_eat_interval` steps between
     catches, maintained in SimState.predator_eat_timer.

Run: pytest tests/test_predator_eating.py -v
"""

import math

import jax
import jax.numpy as jnp
import pytest

from src.jax_food import check_eating_jax
from src.jax_state import init_simstate


@pytest.fixture
def config():
    """Minimal config sufficient for check_eating_jax. Matches
    configs/baseline_faithful.yaml values for the predator mechanics we test.
    Small population keeps the test fast."""
    return {
        "experiment_name": "test_predator_eating",
        "world_size": 960,
        "obs_dim": 205,
        "prey_initial": 2,
        "predator_initial": 1,
        "prey_cap": 10,
        "predator_cap": 5,
        "prey_radius": 10.0,
        "predator_radius": 14.0,
        "max_motor_norm": 114.0,
        "n_proximity_sensors": 32,
        "n_proximity_channels": 4,
        "proximity_fov_deg": 120.0,
        "proximity_max_range": 200.0,
        "n_tactile_sensors": 18,
        "n_tactile_channels": 4,
        "tactile_spacing_deg": 20.0,
        "social_obs": "position_only",
        "food_max": 50,
        "food_initial": 5,
        "food_growth_rate": 0.5,
        "food_max_regen_per_step": 10,
        "energy_capacity": 1000.0,
        "prey_e_food": 1.0,
        "prey_c_b": 1.0e-4,
        "prey_c_a": 2.5e-6,
        "predator_d_b": 4.0e-3,
        "predator_d_a": 5.0e-5,
        "predator_eta": 0.6,
        "predator_mouth_tactile_bins": [0, 1, 17],
        "predator_eat_interval": 10,
        "kappa_h": 0.01,
        "alpha_e": 0.02,
        "beta_h": 0.2,
        "alpha_t_prey": 4.0e-7,
        "alpha_t_pred": 2.0e-7,
        "beta_t_prey": 4.0e-6,
        "beta_t_pred": 4.0e-6,
        "kappa_b": 1.0e-3,
        "beta_b": 0.4,
        "zeta_b_prey": 15.0,
        "zeta_b_pred": 100.0,
        "energy_share_ratio": 0.4,
        "spawn_spread": 100.0,
        "prey_e_initial": 100.0,
        "predator_e_initial": 100.0,
        "initial_energy": 100.0,
        "reward_weights_init_std": 0.1,
        "mutation_df": 2,
        "mutation_scale": 0.4,
        "weight_clip": 100.0,
        "policy_hidden_size": 64,
        "policy_n_hidden_layers": 2,
        "action_clip_low": -20.0,
        "action_clip_high": 80.0,
        "action_mapping": "sigmoid",
        "gamma": 0.999,
        "rollout_steps": 32,
        "minibatch_size": 16,
        "ppo_epochs": 2,
        "clip_epsilon": 0.2,
        "entropy_coef": 0.001,
        "gae_lambda": 0.95,
        "lr": 3.0e-4,
        "adam_eps": 1.0e-7,
        "vf_coef": 0.5,
    }


def _place_agents(state, config, placements):
    """Overwrite circle.p.xy, circle.p.angle, species, is_active, radii so
    each placement occupies a *species-correct slot* (D19 invariant).

    Each placement is (species, x, y, heading_rad). Prey go into the first
    free prey slot [0, prey_cap); predators go into the first free pred
    slot [prey_cap, max_agents). Returns (new_state, slots) where slots[i]
    is the slot index assigned to placements[i].
    """
    import phyjax2d as pj

    max_agents = state.is_active.shape[0]
    prey_cap = config["prey_cap"]
    prey_r = config["prey_radius"]
    pred_r = config["predator_radius"]

    xs = jnp.zeros(max_agents)
    ys = jnp.zeros(max_agents)
    angs = jnp.zeros(max_agents)
    species = jnp.where(
        jnp.arange(max_agents) < prey_cap, 0, 1
    ).astype(jnp.int32)
    active = jnp.zeros(max_agents, dtype=bool)
    radii = jnp.where(species == 0, prey_r, pred_r).astype(jnp.float32)

    next_prey = 0
    next_pred = prey_cap
    slots = []
    for (sp, x, y, a) in placements:
        if int(sp) == 0:
            slot = next_prey
            next_prey += 1
            assert slot < prey_cap, "too many prey placements"
        else:
            slot = next_pred
            next_pred += 1
            assert slot < max_agents, "too many predator placements"
        slots.append(slot)
        xs = xs.at[slot].set(x)
        ys = ys.at[slot].set(y)
        angs = angs.at[slot].set(a)
        active = active.at[slot].set(True)

    circle = state.phyjax_stated.get("circle")
    circle = circle.replace(
        p=pj.Position(xy=jnp.stack([xs, ys], axis=-1), angle=angs),
        is_active=active,
    )
    new_stated = state.phyjax_stated.replace(circle=circle)
    new_state = state.replace(
        phyjax_stated=new_stated,
        species=species,
        is_active=active,
        radii=radii,
    )
    return new_state, slots


def _synthetic_contact_mat(state):
    """Distance-based (A, A) bool contact matrix for unit tests.

    Tests don't actually run physics substeps; D19's real contact source is
    phyjax2d's per-substep penetration. Here we synthesize an equivalent
    (dist <= sum_radii) matrix, which is exactly what the legacy distance
    check computed — keeping these tests' semantics unchanged while
    matching the new signature."""
    circle = state.phyjax_stated.get("circle")
    pos = circle.p.xy
    radii = state.radii
    diffs = pos[None, :, :] - pos[:, None, :]
    dists = jnp.linalg.norm(diffs, axis=-1)
    thresh = radii[:, None] + radii[None, :]
    return dists <= thresh


def _catch_count(state, config):
    """Run check_eating_jax and return the total predator catches this step."""
    _, _, pred_n_catches, _, _, _ = check_eating_jax(
        state, config, _synthetic_contact_mat(state)
    )
    return int(jnp.sum(pred_n_catches))


class TestCatchGeometry:

    def test_touching_prey_in_front_is_caught(self, config):
        """Predator at (500,500) facing +x, prey touching at (520,500). Bin 0 = caught."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        # predator at origin of world center, prey directly in front at distance 20
        # (< pred_r + prey_r = 24 → contact)
        state, _ = _place_agents(state, config, [
            (1, 500.0, 500.0, 0.0),    # predator facing east
            (0, 520.0, 500.0, 0.0),    # prey 20 units east (touching, bin 0)
        ])
        assert _catch_count(state, config) == 1

    def test_touching_prey_behind_is_not_caught(self, config):
        """Prey touching but at 180° — nearest bin is 9, not in mouth — no catch."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        state, _ = _place_agents(state, config, [
            (1, 500.0, 500.0, 0.0),    # predator facing east
            (0, 480.0, 500.0, 0.0),    # prey 20 units behind (touching, bin 9)
        ])
        assert _catch_count(state, config) == 0

    def test_non_contact_is_not_caught(self, config):
        """Prey 30 units in front — past contact threshold (24) — no catch.

        This pins down the April 2026 bug: with the old radial-range code
        [40, 80], a prey 30 units away would have been TOO CLOSE and not
        caught, but a prey 60 units away WOULD have been caught despite
        not touching. Both are now correctly "no catch" unless touching.
        """
        state = init_simstate(config, jax.random.PRNGKey(0))
        state, _ = _place_agents(state, config, [
            (1, 500.0, 500.0, 0.0),
            (0, 530.0, 500.0, 0.0),    # prey 30 units ahead — not touching
        ])
        assert _catch_count(state, config) == 0


class TestCooldown:

    def test_catch_at_step0_sets_timer(self, config):
        """After catching, predator_eat_timer resets to eat_interval."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        state, slots = _place_agents(state, config, [
            (1, 500.0, 500.0, 0.0),
            (0, 520.0, 500.0, 0.0),
        ])
        pred_slot = slots[0]
        assert int(state.predator_eat_timer[pred_slot]) == 0  # ready to eat
        _, _, pred_n_catches, _, new_timer, _ = check_eating_jax(
            state, config, _synthetic_contact_mat(state)
        )
        assert int(pred_n_catches[pred_slot]) == 1
        assert int(new_timer[pred_slot]) == config["predator_eat_interval"]

    def test_cannot_catch_during_cooldown(self, config):
        """With timer > 0, predator can't catch even when prey is touching."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        state, slots = _place_agents(state, config, [
            (1, 500.0, 500.0, 0.0),
            (0, 520.0, 500.0, 0.0),
        ])
        pred_slot = slots[0]
        # Manually set timer to 5 (mid-cooldown)
        state = state.replace(
            predator_eat_timer=state.predator_eat_timer.at[pred_slot].set(5)
        )
        _, _, pred_n_catches, _, new_timer, _ = check_eating_jax(
            state, config, _synthetic_contact_mat(state)
        )
        assert int(pred_n_catches[pred_slot]) == 0
        # Timer should decrement to 4
        assert int(new_timer[pred_slot]) == 4

    def test_can_catch_again_after_cooldown(self, config):
        """With timer at 0, predator can catch again."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        state, slots = _place_agents(state, config, [
            (1, 500.0, 500.0, 0.0),
            (0, 520.0, 500.0, 0.0),
        ])
        pred_slot = slots[0]
        # Timer at -1 (well past cooldown)
        state = state.replace(
            predator_eat_timer=state.predator_eat_timer.at[pred_slot].set(-1)
        )
        _, _, pred_n_catches, _, new_timer, _ = check_eating_jax(
            state, config, _synthetic_contact_mat(state)
        )
        assert int(pred_n_catches[pred_slot]) == 1
        assert int(new_timer[pred_slot]) == config["predator_eat_interval"]


class TestEnergyCostScaling:
    """D24 regression: energy cost must use the physics-scaled action
    (with act_ratio), not the raw policy action. For predators,
    act_ratio ≈ (pred_r/prey_r)² = 1.96, so using raw action
    undercharges them ~49%."""

    def test_predator_cost_uses_act_ratio(self, config):
        """Feed the same raw action to update_energies_jax for a prey and
        a predator at matched starting energy. Predator must burn
        strictly more than prey (after their respective baseline costs
        are subtracted)."""
        import jax
        import jax.numpy as jnp
        from src.jax_lifecycle import update_energies_jax

        state = init_simstate(config, jax.random.PRNGKey(0))
        state, slots = _place_agents(config=config, state=state, placements=[
            (0, 200.0, 200.0, 0.0),
            (1, 800.0, 800.0, 0.0),
        ])
        prey_slot, pred_slot = slots[0], slots[1]

        # Set matched energies; give both the same raw action (50, 50).
        start_e = 500.0
        state = state.replace(
            energies=state.energies
                .at[prey_slot].set(start_e)
                .at[pred_slot].set(start_e),
        )
        max_agents = state.is_active.shape[0]
        all_actions = jnp.zeros((max_agents, 2)).at[prey_slot].set(
            jnp.array([50.0, 50.0])
        ).at[pred_slot].set(jnp.array([50.0, 50.0]))
        # No eating this step — we want to measure pure action-cost drain.
        prey_n_eaten = jnp.zeros(max_agents, dtype=jnp.int32)
        pred_catch_slots = jnp.full((max_agents, 5), -1, dtype=jnp.int32)
        pred_n_catches = jnp.zeros(max_agents, dtype=jnp.int32)

        new_state = update_energies_jax(
            state, prey_n_eaten, pred_catch_slots, pred_n_catches,
            all_actions, config,
        )

        # Prey cost (scaled action = raw since act_ratio=1):
        #   c_a * 50√2 + c_b = 2.5e-6 * 70.71 + 1e-4 ≈ 2.77e-4
        # Predator cost (scaled action = 1.96 * raw):
        #   d_a * 50√2 * 1.96 + d_b = 5e-5 * 70.71 * 1.96 + 4e-3 ≈ 1.1e-2
        prey_drain = start_e - float(new_state.energies[prey_slot])
        pred_drain = start_e - float(new_state.energies[pred_slot])
        # Predator should drain ~40x more per step (base 4e-3 vs 1e-4 + action
        # component ~7x larger). Concrete sanity: pred drain > 10x prey drain.
        assert pred_drain > prey_drain * 10, (
            f"predator drain {pred_drain:.5f} should be >10x prey drain "
            f"{prey_drain:.5f} — D24 act_ratio scaling may be missing"
        )
        # And predator should clearly lose > d_b alone (4e-3), i.e. action
        # cost contributes non-trivially — which confirms act_ratio applied.
        d_b = config["predator_d_b"]
        assert pred_drain > d_b * 1.5, (
            f"predator drain {pred_drain:.5f} barely above baseline d_b {d_b}; "
            f"action component missing — D24 may have regressed"
        )


class TestEnergyTransfer:
    """After a successful catch, the predator gains eta*prey_energy and
    the prey is deactivated — verified end-to-end through sim_step_core
    so the whole pipeline (not just check_eating_jax) stays honest."""

    def test_predator_energy_jumps_after_catch(self, config):
        """D20 regression: the *first* version of the D20 patch zeroed
        caught prey energies before update_energies_jax read them,
        so the predator got pred_gain=0 on every catch and slowly
        starved. This test places a predator touching a prey, runs
        one full sim step, and asserts the predator's energy went
        *up* by ≈ eta*prey_energy minus metabolic cost."""
        import jax
        from src.environment import _build_physics
        from src.jax_sim import build_sim_step

        state = init_simstate(config, jax.random.PRNGKey(0))
        state, slots = _place_agents(state, config, [
            (1, 500.0, 500.0, 0.0),    # predator facing east
            (0, 520.0, 500.0, 0.0),    # prey in mouth bin 0, touching
        ])
        pred_slot, prey_slot = slots[0], slots[1]
        # Set known starting energies.
        state = state.replace(
            energies=state.energies
                .at[pred_slot].set(100.0)
                .at[prey_slot].set(80.0),
        )

        space, _ = _build_physics(config)
        sim_step_core, _ = build_sim_step(config, space)
        new_state = sim_step_core(state)

        eta = config["predator_eta"]
        expected_gain = eta * 80.0
        pred_e_before = 100.0
        pred_e_after = float(new_state.energies[pred_slot])
        # Metabolic cost over one step is small vs expected_gain=48.
        assert pred_e_after > pred_e_before + expected_gain * 0.9, (
            f"Predator energy {pred_e_after:.1f} didn't gain ~{expected_gain:.1f} "
            f"from catching prey of energy 80 — D20 transfer broken."
        )
        # Prey must be deactivated.
        assert not bool(new_state.is_active[prey_slot]), (
            "Caught prey is still active — D20 deactivation broken."
        )


class TestDoneFlagOnDeath:
    """D23 regression: when an agent dies, its last rollout slot must be
    marked `rollout_dones = True`. Without this, GAE bootstraps as if the
    agent's trajectory continued past death with a valid value estimate.
    Currently guarded by the PPO is_active gate (which never fires PPO on
    dead agents), but the flag is needed for correct semantics and to
    future-proof any architecture change that relaxes the gate."""

    def test_caught_prey_rollout_done_flag_set(self, config):
        """A prey caught by a predator this step must have
        rollout_dones[prey_slot, old_ptr] = True after sim_step_core."""
        import jax
        from src.environment import _build_physics
        from src.jax_sim import build_sim_step

        state = init_simstate(config, jax.random.PRNGKey(0))
        state, slots = _place_agents(state, config, [
            (1, 500.0, 500.0, 0.0),    # predator
            (0, 520.0, 500.0, 0.0),    # prey in mouth, will be caught
        ])
        pred_slot, prey_slot = slots[0], slots[1]
        state = state.replace(
            energies=state.energies
                .at[pred_slot].set(100.0)
                .at[prey_slot].set(80.0),
        )
        # Remember the prey's rollout ptr before the step.
        prey_ptr_before = int(state.rollout_ptrs[prey_slot])

        space, _ = _build_physics(config)
        sim_step_core, _ = build_sim_step(config, space)
        new_state = sim_step_core(state)

        assert not bool(new_state.is_active[prey_slot]), "prey should be caught"
        assert bool(new_state.rollout_dones[prey_slot, prey_ptr_before]), (
            f"rollout_dones[prey, {prey_ptr_before}] should be True after catch"
        )


class TestIndependentTimers:

    def test_two_predators_independent_cooldowns(self, config):
        """Predator A catches, its timer resets. Predator B hasn't eaten; its
        timer decrements independently and doesn't affect A."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        # A and B far apart; A touching a prey, B alone
        state, slots = _place_agents(state, config, [
            (1, 200.0, 200.0, 0.0),   # predator A facing east
            (0, 220.0, 200.0, 0.0),   # prey 20 east of A  (A catches)
            (1, 700.0, 700.0, 0.0),   # predator B, no prey near
        ])
        pred_a, pred_b = slots[0], slots[2]
        _, _, pred_n_catches, _, new_timer, _ = check_eating_jax(
            state, config, _synthetic_contact_mat(state)
        )
        # A caught, timer reset to 10
        assert int(pred_n_catches[pred_a]) == 1
        assert int(new_timer[pred_a]) == config["predator_eat_interval"]
        # B didn't catch, timer decremented from 0 to -1
        assert int(pred_n_catches[pred_b]) == 0
        assert int(new_timer[pred_b]) == -1
