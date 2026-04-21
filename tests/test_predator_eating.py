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
    """Overwrite circle.p.xy, circle.p.angle, species, is_active, radii for
    the first len(placements) slots. Each placement is
    (species, x, y, heading_rad). Inactive slots get pushed to (0, 0).
    """
    import phyjax2d as pj

    max_agents = state.is_active.shape[0]
    prey_r = config["prey_radius"]
    pred_r = config["predator_radius"]

    xs = jnp.zeros(max_agents)
    ys = jnp.zeros(max_agents)
    angs = jnp.zeros(max_agents)
    species = jnp.zeros(max_agents, dtype=jnp.int32)
    active = jnp.zeros(max_agents, dtype=bool)
    radii = jnp.where(
        jnp.zeros(max_agents, dtype=jnp.int32) == 0, prey_r, pred_r
    )  # placeholder, overwritten below

    for i, (sp, x, y, a) in enumerate(placements):
        xs = xs.at[i].set(x)
        ys = ys.at[i].set(y)
        angs = angs.at[i].set(a)
        species = species.at[i].set(int(sp))
        active = active.at[i].set(True)
        radii = radii.at[i].set(prey_r if sp == 0 else pred_r)

    circle = state.phyjax_stated.get("circle")
    circle = circle.replace(
        p=pj.Position(xy=jnp.stack([xs, ys], axis=-1), angle=angs),
        is_active=active,
    )
    new_stated = state.phyjax_stated.replace(circle=circle)
    return state.replace(
        phyjax_stated=new_stated,
        species=species,
        is_active=active,
        radii=radii,
    )


def _catch_count(state, config):
    """Run check_eating_jax and return the total predator catches this step."""
    _, _, pred_n_catches, _, _ = check_eating_jax(state, config)
    return int(jnp.sum(pred_n_catches))


class TestCatchGeometry:

    def test_touching_prey_in_front_is_caught(self, config):
        """Predator at (500,500) facing +x, prey touching at (520,500). Bin 0 = caught."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        # predator at origin of world center, prey directly in front at distance 20
        # (< pred_r + prey_r = 24 → contact)
        state = _place_agents(state, config, [
            (1, 500.0, 500.0, 0.0),    # predator facing east
            (0, 520.0, 500.0, 0.0),    # prey 20 units east (touching, bin 0)
        ])
        assert _catch_count(state, config) == 1

    def test_touching_prey_behind_is_not_caught(self, config):
        """Prey touching but at 180° — nearest bin is 9, not in mouth — no catch."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        state = _place_agents(state, config, [
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
        state = _place_agents(state, config, [
            (1, 500.0, 500.0, 0.0),
            (0, 530.0, 500.0, 0.0),    # prey 30 units ahead — not touching
        ])
        assert _catch_count(state, config) == 0


class TestCooldown:

    def test_catch_at_step0_sets_timer(self, config):
        """After catching, predator_eat_timer resets to eat_interval."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        state = _place_agents(state, config, [
            (1, 500.0, 500.0, 0.0),
            (0, 520.0, 500.0, 0.0),
        ])
        assert int(state.predator_eat_timer[0]) == 0  # ready to eat
        _, _, pred_n_catches, _, new_timer = check_eating_jax(state, config)
        assert int(pred_n_catches[0]) == 1
        assert int(new_timer[0]) == config["predator_eat_interval"]

    def test_cannot_catch_during_cooldown(self, config):
        """With timer > 0, predator can't catch even when prey is touching."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        state = _place_agents(state, config, [
            (1, 500.0, 500.0, 0.0),
            (0, 520.0, 500.0, 0.0),
        ])
        # Manually set timer to 5 (mid-cooldown)
        state = state.replace(
            predator_eat_timer=state.predator_eat_timer.at[0].set(5)
        )
        _, _, pred_n_catches, _, new_timer = check_eating_jax(state, config)
        assert int(pred_n_catches[0]) == 0
        # Timer should decrement to 4
        assert int(new_timer[0]) == 4

    def test_can_catch_again_after_cooldown(self, config):
        """With timer at 0, predator can catch again."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        state = _place_agents(state, config, [
            (1, 500.0, 500.0, 0.0),
            (0, 520.0, 500.0, 0.0),
        ])
        # Timer at -1 (well past cooldown)
        state = state.replace(
            predator_eat_timer=state.predator_eat_timer.at[0].set(-1)
        )
        _, _, pred_n_catches, _, new_timer = check_eating_jax(state, config)
        assert int(pred_n_catches[0]) == 1
        assert int(new_timer[0]) == config["predator_eat_interval"]


class TestIndependentTimers:

    def test_two_predators_independent_cooldowns(self, config):
        """Predator A catches, its timer resets. Predator B hasn't eaten; its
        timer decrements independently and doesn't affect A."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        # A and B far apart; A touching a prey, B alone
        state = _place_agents(state, config, [
            (1, 200.0, 200.0, 0.0),   # predator A at (200,200) facing east
            (0, 220.0, 200.0, 0.0),   # prey 20 east of A  (A catches)
            (1, 700.0, 700.0, 0.0),   # predator B at (700,700), no prey near
        ])
        _, _, pred_n_catches, _, new_timer = check_eating_jax(state, config)
        # A caught, timer reset to 10
        assert int(pred_n_catches[0]) == 1
        assert int(new_timer[0]) == config["predator_eat_interval"]
        # B didn't catch, timer decremented from 0 to -1
        assert int(pred_n_catches[2]) == 0
        assert int(new_timer[2]) == -1
