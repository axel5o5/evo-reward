"""
test_tactile_bin_indexing.py
----------------------------
D26 regression: tactile bin assignment must follow phyjax2d/emevo's
convention — heading=0 means agent's forward direction is WORLD +y,
bins are 0-indexed starting at "directly forward" and rotating
counter-clockwise 2π/n_bins per bin.

Pre-D26, our code (a) omitted the π/2 offset required by the phyjax2d
heading convention and (b) didn't subtract heading at all in the
tactile observation pipeline. The effect: `predator_mouth_tactile_bins
= [0, 1, 17]` (supposed to be the 60° front arc) actually pointed 90°
to the agent's right, so predators physically could not catch prey
they were facing. Likely root cause of predator extinction in
phase1a-v{2,3,4}.

This test pins the correct convention in two places:
  * check_eating_jax catch classification
  * observations._single_tactile observation classification

Both must agree with emevo's `get_relative_angle` + `_search_bin`.
"""

import math

import jax
import jax.numpy as jnp
import pytest


# Reuse the small-population config from the predator-eat tests.
from tests.test_predator_eating import config, _place_agents, _synthetic_contact_mat
from src.jax_state import init_simstate


# ─── check_eating_jax / mouth bin ─────────────────────────────────────────


def _run_catch(config, state):
    from src.jax_food import check_eating_jax
    _, _, pred_n_catches, _, _, _ = check_eating_jax(
        state, config, _synthetic_contact_mat(state)
    )
    return int(jnp.sum(pred_n_catches))


def test_prey_in_front_is_bin_0_heading_0(config):
    """heading=0 → forward=+y. A prey 20 units north of pred is bin 0,
    which IS in predator_mouth_tactile_bins → caught."""
    state = init_simstate(config, jax.random.PRNGKey(0))
    state, _ = _place_agents(state, config, [
        (1, 500.0, 500.0, 0.0),    # pred heading=0 → forward=+y
        (0, 500.0, 520.0, 0.0),    # prey 20 north → directly in front
    ])
    assert _run_catch(config, state) == 1


def test_prey_to_right_heading_0_is_not_in_mouth(config):
    """heading=0 → forward=+y. A prey 20 units east of pred is bin 13
    (world angle 0, minus π/2 for heading offset = 3π/2 = 270° → bin 13).
    Bin 13 is NOT in mouth → NOT caught. Pre-D26 this was bin 0 and
    would have been caught — that was the off-by-90° bug."""
    state = init_simstate(config, jax.random.PRNGKey(0))
    state, _ = _place_agents(state, config, [
        (1, 500.0, 500.0, 0.0),    # heading=0 → forward=+y
        (0, 520.0, 500.0, 0.0),    # prey 20 east → to right (NOT in front)
    ])
    assert _run_catch(config, state) == 0


def test_prey_in_front_rotates_with_heading(config):
    """Keep prey position fixed, rotate predator's heading so the prey
    is directly in front — must be caught from every orientation."""
    for heading_deg in (0, 45, 90, 135, 180, 225, 270, 315):
        heading_rad = math.radians(heading_deg)
        # Place prey exactly `dist` units away in the pred's forward direction.
        # Per phyjax2d, forward = (-sin(heading), cos(heading)).
        dist = 20.0
        prey_x = 500.0 - math.sin(heading_rad) * dist
        prey_y = 500.0 + math.cos(heading_rad) * dist
        state = init_simstate(config, jax.random.PRNGKey(0))
        state, _ = _place_agents(state, config, [
            (1, 500.0, 500.0, heading_rad),
            (0, prey_x, prey_y, 0.0),
        ])
        assert _run_catch(config, state) == 1, (
            f"heading={heading_deg}° → prey should be in front and caught, "
            f"but catch count was 0 (pre-D26 bug pattern: mouth 90° off)"
        )


# ─── tactile observation bin ──────────────────────────────────────────────


def test_tactile_food_in_front_lights_bin_0(config):
    """With heading=0 → forward=+y, a food item in contact at +y from
    the agent must light tactile bin 0 of the food channel, regardless
    of whether the agent is at world +x or +y orientation."""
    from src.observations import _bin_in_agent_frame

    n_bins = config["n_tactile_sensors"]
    # Prey at origin, heading=0. Food contact at +y → world angle π/2.
    bin_idx = int(_bin_in_agent_frame(jnp.float32(math.pi / 2.0), jnp.float32(0.0), n_bins))
    assert bin_idx == 0, f"prey-front-food should be bin 0 but got {bin_idx}"


def test_tactile_to_right_of_heading_0_is_bin_13(config):
    """A contact at world +x (east) with heading=0 (forward=+y) is 90°
    to the agent's right. That's bin 13 for n=18 (270° in rotated frame,
    /20° = 13)."""
    from src.observations import _bin_in_agent_frame
    n_bins = config["n_tactile_sensors"]
    bin_idx = int(_bin_in_agent_frame(jnp.float32(0.0), jnp.float32(0.0), n_bins))
    # 270° / 20° = 13
    assert bin_idx == 13, f"east contact with north heading should be bin 13 but got {bin_idx}"


def test_tactile_rotates_with_heading(config):
    """Bin index of a WORLD-fixed contact direction should rotate with
    the observer's heading. A contact at world +y is bin 0 from
    heading=0; from heading=π/2 (observer facing west), the same
    contact is 90° right of forward, i.e. bin 13."""
    from src.observations import _bin_in_agent_frame
    n_bins = config["n_tactile_sensors"]
    angle_world = jnp.float32(math.pi / 2.0)  # contact at world +y

    b0 = int(_bin_in_agent_frame(angle_world, jnp.float32(0.0), n_bins))
    b_west = int(_bin_in_agent_frame(angle_world, jnp.float32(math.pi / 2.0), n_bins))
    assert b0 == 0, f"contact at +y with heading 0 should be bin 0, got {b0}"
    # heading=π/2 means agent's forward direction is world -x (west).
    # Contact at world +y is then to agent's right (90° cw from forward).
    # So bin = 13 (for n=18, 2π·13/18 = 260°, center 270°).
    assert b_west == 13, (
        f"same +y contact with heading=π/2 should be bin 13 but got {b_west}"
    )
