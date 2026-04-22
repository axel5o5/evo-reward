"""
test_paper_anchored.py
----------------------
Tests whose assertions are anchored against EXTERNAL sources (the K&D 2025
paper, phyjax2d source comments, or emevo TOMLs at a pinned URL) rather
than against our code's current behavior.

**Rationale.** D18-D26 were all semantic-mismatch bugs: code compiled and
ran, but encoded a different model than the paper/emevo described. Tests
that just verify "the code does what we coded" can't catch that — they
ratify whatever we shipped. Tests here are written the other way around:
each one quotes its source in the docstring and fails if our
implementation ever disagrees with it.

Categories:
  1. Convention pinning — phyjax2d heading convention, bin-0 direction,
     force-direction rules. If phyjax2d ever changes, we catch it.
  2. Transformation invariants — rotation/translation/Markov properties
     that MUST hold regardless of implementation detail. These would
     have caught D26 directly.
  3. Fuzz invariants — random scenes + universal claims. Catch geometry
     bugs on lots of orientations simultaneously.
  4. Paper-claim tests — each assertion cites a paper section.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from src.jax_state import init_simstate
from src.jax_food import check_eating_jax
from src.environment import _build_physics
from src.jax_sim import build_sim_step

from tests.test_predator_eating import (
    config as small_config,  # small test-speed config
    _place_agents,
    _synthetic_contact_mat,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def paper_config():
    """Real `baseline_faithful.yaml` (paper-text-faithful). Paper-claim
    assertions must run against the actual config, not a shrunk test
    fixture — otherwise a param regression in the real config slips past
    a passing small-config test."""
    with open(ROOT / "configs/baseline_faithful.yaml") as f:
        cfg = yaml.safe_load(f) or {}
    with open(ROOT / "configs/runtime/default.yaml") as f:
        cfg.update(yaml.safe_load(f) or {})
    return cfg


# Alias so old "config" fixture references still resolve for the
# transformation-invariance tests (which use small config for speed).
config = small_config


# ════════════════════════════════════════════════════════════════════════
# 1. CONVENTION PINNING
#    Source: phyjax2d `get_relative_angle` docstring + emevo cf_predator
#    convention. These tests import from phyjax2d directly — if the
#    library's convention ever changes, these break loudly.
# ════════════════════════════════════════════════════════════════════════


def test_phyjax2d_heading_convention_pinned():
    """phyjax2d source: `get_relative_angle` subtracts π/2 with the comment
    'our angle starts from 0.5π (90 degree)'. That means: agent at
    heading=0 has its local +y axis pointing to world +y.

    Source URL at time of writing:
      https://raw.githubusercontent.com/oist/emevo/gecco2026/src/emevo/environments/circle_foraging.py
      (imported from `phyjax2d`)

    Construct a minimal space via phyjax2d.SpaceBuilder + zeros_state
    rather than instantiating State by hand, so the test stays robust
    to phyjax2d adding / renaming State fields.
    """
    import phyjax2d as pj
    from phyjax2d import Position, get_relative_angle

    builder = pj.SpaceBuilder(
        gravity=(0.0, 0.0),
        dt=0.1,
        linear_damping=0.8,
        angular_damping=0.6,
        n_velocity_iter=6,
        n_position_iter=2,
    )
    builder.add_circle(radius=10.0, density=1.0, friction=0.2, elasticity=0.4)
    builder.add_circle(radius=10.0, density=1.0, friction=0.2, elasticity=0.4)
    space = builder.build()
    stated = space.zeros_state()

    circle = stated.circle
    # A at origin heading=0, B at (0, 10).
    circle = circle.replace(
        p=Position(
            angle=jnp.array([0.0, 0.0]),
            xy=jnp.array([[0.0, 0.0], [0.0, 10.0]]),
        ),
        is_active=jnp.array([True, True]),
    )
    stated = stated.replace(circle=circle)

    rel = float(get_relative_angle(stated.circle, stated.circle)[0, 1])
    # rel[0, 1] = relative angle of B from A's frame.
    # B is directly in front of A per phyjax2d's forward=+y convention
    # → rel_angle should be 0 (mod 2π). phyjax2d's own `+ 3·TWO_PI - π/2`
    # construction can produce rel ≈ 2π instead of 0 due to fp rounding;
    # either is semantically "directly forward".
    TWO_PI = 2.0 * math.pi
    wrapped = min(abs(rel), abs(rel - TWO_PI))
    assert wrapped < 1e-4, (
        f"phyjax2d heading convention changed? Expected rel_angle ≈ 0 "
        f"(mod 2π) for agent heading=0 with target at +y, got {rel}"
    )


def test_bin_0_is_directly_forward():
    """Paper Figure 3 caption + emevo `mouth_range='front'`: the mouth
    is the forward arc. For n_tactile_bins=18 with 20° spacing, the
    forward arc covers bins {0, 1, n-1} = {0, 1, 17}. Bin 0's *lower*
    boundary is 0° (directly forward). Verify our helper classifies
    a contact angle equal to the agent's heading + π/2 (i.e. directly
    along the local +y axis) as bin 0."""
    from src.observations import _bin_in_agent_frame

    n_bins = 18
    for heading_deg in (-180, -90, -45, 0, 45, 90, 135, 180):
        heading_rad = math.radians(heading_deg)
        # Forward direction in world = (-sin θ, cos θ); its world angle = θ + π/2
        forward_world_angle = heading_rad + math.pi / 2.0
        bin_idx = int(
            _bin_in_agent_frame(
                jnp.float32(forward_world_angle),
                jnp.float32(heading_rad),
                n_bins,
            )
        )
        # rel=0 sits exactly on the bin 0 ↔ bin (n-1) boundary. Either
        # resolution is valid convention; the test is that it's NOT in
        # the middle (which is where the pre-D26 bug put it — bin 4-5).
        assert bin_idx in (0, n_bins - 1), (
            f"heading {heading_deg}°: forward-direction contact should be "
            f"bin 0 or bin {n_bins - 1} (front arc boundary), got bin "
            f"{bin_idx}. Pre-D26 pattern (mouth 90° rotated) would produce "
            f"bin 4-5 here."
        )


def test_heading_force_direction_convention():
    """Emevo applies force as `f = [0, f_raw]` in local frame with the
    comment/implication that local +y is forward. After rotation by the
    body's heading θ, world force direction is (-sin θ, cos θ). Verify
    by stepping a single agent through phyjax2d and observing its
    velocity direction after one step.

    Source: our `jax_sim.py::physics_step` applies `f1 = [0, f1_raw]`
    via `circle.apply_force_local(...)`. Paper Section 3 says agents
    move by applying force to two rear points.
    """
    import phyjax2d as pj

    builder = pj.SpaceBuilder(
        gravity=(0.0, 0.0),
        dt=0.1,
        linear_damping=0.8,
        angular_damping=0.6,
        n_velocity_iter=6,
        n_position_iter=2,
    )
    builder.add_circle(radius=10.0, density=1.0, friction=0.2, elasticity=0.4)
    space = builder.build()

    for heading_deg in (0, 45, 90, 135):
        heading = math.radians(heading_deg)
        # Put the agent at (500, 500) with the chosen heading.
        stated = space.zeros_state()
        circle = stated.circle
        circle = circle.replace(
            p=pj.Position(
                angle=jnp.array([heading]),
                xy=jnp.array([[500.0, 500.0]]),
            ),
            is_active=jnp.array([True]),
        )
        stated = stated.replace(circle=circle)

        # Apply a large forward force at the center.
        f = jnp.array([[0.0, 100.0]])  # local +y
        p_center = jnp.array([[0.0, 0.0]])
        circle = stated.circle.apply_force_local(p_center, f)
        stated = stated.replace(circle=circle)

        # Step once.
        solver = space.init_solver()
        stated, solver, _ = pj.step(space, stated, solver)

        v = stated.circle.v.xy[0]
        # Expected world direction: (-sin θ, cos θ)
        expected_dir = jnp.array([-math.sin(heading), math.cos(heading)])
        # Compare unit vectors (magnitude depends on damping etc.)
        v_norm = float(jnp.linalg.norm(v))
        if v_norm < 1e-3:
            continue  # damping swallowed the impulse — not diagnostic
        v_unit = v / v_norm
        cos_error = float(jnp.dot(v_unit, expected_dir))
        assert cos_error > 0.99, (
            f"heading {heading_deg}°: local +y force produced world velocity "
            f"direction with cos-similarity {cos_error:.3f} to (-sin θ, cos θ). "
            f"Heading/force convention broken."
        )


# ════════════════════════════════════════════════════════════════════════
# 2. TRANSFORMATION INVARIANTS
#    Things that MUST hold under rotation / translation / repeated calls,
#    regardless of implementation. Would have caught D26 directly — a
#    90°-rotated mouth breaks rotation invariance.
# ════════════════════════════════════════════════════════════════════════


def _rotate_around(point, center, theta):
    dx, dy = point[0] - center[0], point[1] - center[1]
    return (
        center[0] + dx * math.cos(theta) - dy * math.sin(theta),
        center[1] + dx * math.sin(theta) + dy * math.cos(theta),
    )


def _catch_count(state, config):
    _, _, pred_n_catches, _, _, _ = check_eating_jax(
        state, config, _synthetic_contact_mat(state)
    )
    return int(jnp.sum(pred_n_catches))


def test_catch_rotation_invariant(config):
    """If a (predator, prey) geometry produces a catch at one orientation,
    rotating the entire scene by any θ (including the predator's heading)
    must still produce a catch. This is a PROPERTY of the semantics —
    nothing about our implementation should depend on absolute world
    direction, only the relative geometry.

    Pre-D26 this failed because the mouth was fixed in world frame:
    rotating the scene moved the prey out of the mouth's world-frame arc.
    """
    center = (500.0, 500.0)
    pred_base = (500.0, 500.0)
    prey_base = (500.0, 520.0)  # directly north → in front per phyjax2d
    pred_heading_base = 0.0

    for theta in (0.0, math.pi / 6, math.pi / 4, math.pi / 3,
                  math.pi / 2, math.pi, 1.5 * math.pi, 2 * math.pi - 0.1):
        pred_pos = _rotate_around(pred_base, center, theta)
        prey_pos = _rotate_around(prey_base, center, theta)
        pred_heading = pred_heading_base + theta

        state = init_simstate(config, jax.random.PRNGKey(0))
        state, _ = _place_agents(state, config, [
            (1, pred_pos[0], pred_pos[1], pred_heading),
            (0, prey_pos[0], prey_pos[1], 0.0),
        ])
        assert _catch_count(state, config) == 1, (
            f"rotation θ={math.degrees(theta):.1f}° breaks catch invariance; "
            f"pre-D26 pattern (mouth fixed in world frame)"
        )


def test_catch_translation_invariant(config):
    """Shifting the entire scene by (dx, dy) within world bounds must not
    change catch outcome. Catches only depend on relative geometry.
    """
    pred_offset = (0.0, 0.0)
    prey_offset = (0.0, 20.0)

    for (dx, dy) in [(0, 0), (100, 0), (-80, 0), (0, 120), (150, -80)]:
        state = init_simstate(config, jax.random.PRNGKey(0))
        state, _ = _place_agents(state, config, [
            (1, 500 + dx + pred_offset[0], 500 + dy + pred_offset[1], 0.0),
            (0, 500 + dx + prey_offset[0], 500 + dy + prey_offset[1], 0.0),
        ])
        assert _catch_count(state, config) == 1, (
            f"translation ({dx}, {dy}) broke catch invariance — catch "
            f"semantics should only depend on relative geometry"
        )


def test_sim_step_deterministic(config):
    """Same SimState + same JIT'd sim_step_core → same next state. This
    is the Markov property; JAX makes it nearly automatic but a
    non-deterministic RNG read outside `rng_key` would break it.
    """
    space, _ = _build_physics(config)
    step_fn, _ = build_sim_step(config, space)

    state0 = init_simstate(config, jax.random.PRNGKey(42))
    next_a = step_fn(state0)
    next_b = step_fn(state0)

    # Compare a few representative leaves from the two runs.
    for field in ("step", "next_agent_id", "cum_catches", "cum_deaths"):
        va = int(getattr(next_a, field))
        vb = int(getattr(next_b, field))
        assert va == vb, f"{field}: deterministic step broke ({va} vs {vb})"

    # Positions must match exactly (same seed, same JIT'd step).
    pa = np.asarray(next_a.phyjax_stated.get("circle").p.xy)
    pb = np.asarray(next_b.phyjax_stated.get("circle").p.xy)
    np.testing.assert_allclose(pa, pb, rtol=0, atol=0,
                                err_msg="positions differ between repeated calls")


# ════════════════════════════════════════════════════════════════════════
# 3. FUZZ INVARIANTS
#    Universal claims across random inputs. Catches geometry bugs on
#    many orientations simultaneously.
# ════════════════════════════════════════════════════════════════════════


def test_prey_directly_behind_is_never_caught(config):
    """Universal claim: a prey positioned directly BEHIND a predator
    (along -forward vector) is never caught, regardless of heading,
    distance within contact range, or absolute position.

    Source: paper Figure 3 (mouth is a forward arc) + emevo
    `predator_mouth_range = [0, 1, 17]` (front arc, bins 9 through 16
    covering the rear are not in mouth).
    """
    rng = np.random.default_rng(seed=0)
    for _ in range(25):
        heading = float(rng.uniform(-math.pi, math.pi))
        pred_x = float(rng.uniform(200, 800))
        pred_y = float(rng.uniform(200, 800))
        dist = float(rng.uniform(15, 22))  # within contact
        # "Directly behind" in phyjax2d frame = along (+sin θ, -cos θ)
        prey_x = pred_x + math.sin(heading) * dist
        prey_y = pred_y - math.cos(heading) * dist

        state = init_simstate(config, jax.random.PRNGKey(0))
        state, _ = _place_agents(state, config, [
            (1, pred_x, pred_y, heading),
            (0, prey_x, prey_y, 0.0),
        ])
        assert _catch_count(state, config) == 0, (
            f"heading={math.degrees(heading):.1f}° dist={dist:.1f}: "
            f"prey directly behind was caught — mouth geometry broken"
        )


def test_cum_counters_only_increase(config):
    """Cumulative counters (catches, deaths, feedings) are monotonically
    non-decreasing across sim steps. No implementation detail should
    ever cause them to go backwards."""
    space, _ = _build_physics(config)
    step_fn, _ = build_sim_step(config, space)

    state = init_simstate(config, jax.random.PRNGKey(1))
    prev = {
        "catches": int(state.cum_catches),
        "deaths": int(state.cum_deaths),
        "feedings": int(state.cum_feedings),
        "next_id": int(state.next_agent_id),
    }
    for _ in range(50):
        state = step_fn(state)
    jax.block_until_ready(state.step)
    new = {
        "catches": int(state.cum_catches),
        "deaths": int(state.cum_deaths),
        "feedings": int(state.cum_feedings),
        "next_id": int(state.next_agent_id),
    }
    for k, v in new.items():
        assert v >= prev[k], f"cum_{k} decreased from {prev[k]} to {v}"


# ════════════════════════════════════════════════════════════════════════
# 4. PAPER-CLAIM TESTS
#    Each test's docstring names the paper section it enforces. Tests
#    here would have surfaced config drift we've had (e.g. zeta=15 vs 10).
# ════════════════════════════════════════════════════════════════════════


def test_predator_catch_energy_gain_in_paper_range(config):
    """Paper Section 3: 'Since we use 0.6 [digestive rate] and prey is
    expected to maintain 10 to 20 energy units to survive, predators
    gain 6 to 10 energy units per predation event.'

    Verify the lower end of that claim: with prey at e=10 (low-survival
    prey), a catch should yield ~6 energy (minus the predator's one-step
    metabolic cost, which is small).
    """
    space, _ = _build_physics(config)
    step_fn, _ = build_sim_step(config, space)

    state = init_simstate(config, jax.random.PRNGKey(0))
    state, slots = _place_agents(state, config, [
        (1, 500.0, 500.0, 0.0),
        (0, 500.0, 520.0, 0.0),
    ])
    pred_slot, prey_slot = slots[0], slots[1]
    state = state.replace(
        energies=state.energies
            .at[pred_slot].set(100.0)
            .at[prey_slot].set(10.0),  # paper's low-end prey energy
    )
    pred_e_before = float(state.energies[pred_slot])
    new_state = step_fn(state)
    pred_e_after = float(new_state.energies[pred_slot])
    gain = pred_e_after - pred_e_before

    # Paper claim: 6 <= gain <= 10. Allow a small negative margin for
    # the one-step metabolic cost (d_b ≈ 4e-3 + d_a·||action||·act_ratio
    # ~= 0.01/step max).
    assert 5.0 <= gain <= 10.0, (
        f"predator catch gain {gain:.2f} outside paper's 6-10 range; "
        f"with prey_e=10 and eta=0.6 expected ~6."
    )


def test_predator_catch_high_prey_energy(config):
    """Same paper claim, upper end: with prey at e=20, catch gain ≈ 12
    (0.6 × 20 = 12). Paper says '6 to 10' but this is stated for the
    *survival band* of prey; catching a high-energy prey can exceed the
    top. We test that the eta multiplier is correct, not that gain is
    bounded above by 10.
    """
    space, _ = _build_physics(config)
    step_fn, _ = build_sim_step(config, space)

    state = init_simstate(config, jax.random.PRNGKey(0))
    state, slots = _place_agents(state, config, [
        (1, 500.0, 500.0, 0.0),
        (0, 500.0, 520.0, 0.0),
    ])
    pred_slot, prey_slot = slots[0], slots[1]
    state = state.replace(
        energies=state.energies
            .at[pred_slot].set(100.0)
            .at[prey_slot].set(20.0),
    )
    new_state = step_fn(state)
    gain = float(new_state.energies[pred_slot]) - 100.0

    eta = config["predator_eta"]
    expected = eta * 20.0  # paper's Equation 1: sum_k eta · e_k
    assert abs(gain - expected) < 1.0, (
        f"catch gain {gain:.2f} ≠ eta·prey_e = {expected:.2f} (within 1.0 "
        f"metabolic-cost tolerance). Paper Equation 1 not matched."
    )


def test_prey_birth_threshold_near_30_energy(paper_config):
    """Paper Section 3: '30 energy units are required to increase the
    birth probability for prey agents'. Equivalently, at e < 30 the birth
    probability should be well below max; at e ≥ 30 it should be near max.

    Paper uses b(e) = κ_b / (1 + exp(ζ - β_b·e)). With the paper's
    Table 3 values (κ_b=1e-3, β_b=0.4, ζ=10), half-max is at
    e = ζ/β_b = 25, and the curve is in its upper saturation by e=30.
    """
    from src.jax_lifecycle import _batch_birth_prob_jax

    kappa_b = paper_config["kappa_b"]
    species = jnp.zeros(4, dtype=jnp.int32)  # all prey
    energies = jnp.array([10.0, 20.0, 30.0, 50.0])
    probs = _batch_birth_prob_jax(energies, species, paper_config)

    p10, p20, p30, p50 = [float(x) for x in probs]

    assert p10 < kappa_b * 0.1, (
        f"prey birth prob at e=10 is {p10:.3e} (>10% of max). Paper says "
        f"30 energy units are required — at 10 should be essentially zero."
    )
    assert p30 >= kappa_b * 0.5, (
        f"prey birth prob at e=30 is {p30:.3e} (<50% of max). Paper says "
        f"30 units is enough to 'increase birth probability' — at 30 "
        f"should be well past half-max. ζ may be too high (D22 regression?)."
    )
    assert p10 < p20 < p30 < p50, "birth prob not monotone in energy"


def test_predator_birth_threshold_near_260_energy(paper_config):
    """Paper Section 3: 'predators need 260 or more' energy units for
    reproduction. With κ_b=1e-3, β_b=0.4, ζ=100, half-max is at
    e = ζ/β_b = 250; by e=260 we expect upper-saturation.
    """
    from src.jax_lifecycle import _batch_birth_prob_jax

    kappa_b = paper_config["kappa_b"]
    species = jnp.ones(3, dtype=jnp.int32)  # all predators
    energies = jnp.array([100.0, 260.0, 500.0])
    probs = _batch_birth_prob_jax(energies, species, paper_config)

    p100, p260, p500 = [float(x) for x in probs]

    assert p100 < kappa_b * 0.05, (
        f"predator birth prob at e=100 is {p100:.3e}. Paper says 260+ "
        f"required — at 100 should be essentially zero."
    )
    assert p260 >= kappa_b * 0.5, (
        f"predator birth prob at e=260 is {p260:.3e} (<50% of max). Paper "
        f"says 260 is enough to reproduce — predator ζ may be too high."
    )
    assert p100 < p260 < p500


def test_prey_initial_population_matches_paper(paper_config):
    """Paper Section 4: 'The initial populations are set to 150 prey and
    10 predators.' Pinned here so a runtime overlay or a copy-paste
    mistake in the science config can't silently change this."""
    assert paper_config["prey_initial"] == 150
    assert paper_config["predator_initial"] == 10


def test_world_bounds_match_paper_appendix_a(paper_config):
    """Paper Appendix A: 'The simulation environment is a square domain
    measuring 960×960 units.'"""
    from src.environment import world_bounds
    x, y = world_bounds(paper_config)
    assert x == 960.0 and y == 960.0, (
        f"world bounds are ({x}, {y}) but paper Appendix A says "
        f"square 960×960. If you're running the endpoint config "
        f"(1200×600), this test isn't meant for that."
    )
