"""
test_poly_reward.py
-------------------
Unit tests for the polynomial residual reward genome (Axis 1 v11).

Covers:
  - Zero-init poly genome → reward = pure K&D linear part.
  - Hand-picked weights produce expected reward (one quadratic, one
    interaction term).
  - init_poly_genome + mutate_poly_genome_jax produces non-zero, bounded
    values.
  - compute_poly_reward vmaps correctly over a batch of agents.

Run: pytest tests/test_poly_reward.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.reward import (
    init_poly_genome,
    compute_poly_reward,
)
from src.jax_evolution import mutate_poly_genome_jax


# Fixed linear coefficients applied inside compute_poly_reward.
LINEAR_COEFS = jnp.array([1.0, 0.01, 0.1, 0.1])


@pytest.fixture
def stimuli():
    """An arbitrary non-zero stimulus vector — covers all 4 dimensions."""
    return jnp.array([2.0, 30.0, 0.5, 0.25], dtype=jnp.float32)


@pytest.fixture
def linear_genome():
    """A representative non-zero linear genome (4 weights)."""
    return jnp.array([1.0, -0.5, 2.0, -1.5], dtype=jnp.float32)


# ─── Zero-init: poly residual contributes 0 ─────────────────────────────────

def test_init_poly_genome_zeros():
    """init_poly_genome must return shape (10,) of zeros."""
    g = init_poly_genome(jax.random.PRNGKey(0), {})
    assert g.shape == (10,)
    assert jnp.all(g == 0.0)
    assert g.dtype == jnp.float32


def test_zero_poly_genome_gives_pure_linear(stimuli, linear_genome):
    """With poly_genome = zeros, compute_poly_reward should equal the
    K&D linear reward exactly: sum(w_lin_i * x_i * coefs_i)."""
    poly = jnp.zeros((10,), dtype=jnp.float32)
    expected = float(jnp.sum(linear_genome * stimuli * LINEAR_COEFS))
    r = float(compute_poly_reward(linear_genome, poly, stimuli))
    np.testing.assert_allclose(r, expected, rtol=1e-5)


# ─── Specific weights: quadratic and interaction ─────────────────────────────

def test_single_quadratic_term():
    """Set w_sq_1 = 2.0 (others zero) and verify the residual = 2 * x_1^2."""
    linear = jnp.zeros(4, dtype=jnp.float32)            # kill linear part
    stim = jnp.array([0.0, 3.0, 0.0, 0.0], dtype=jnp.float32)  # x_1 = 3
    poly = jnp.zeros((10,), dtype=jnp.float32).at[1].set(2.0)  # w_sq_1 = 2
    r = float(compute_poly_reward(linear, poly, stim))
    np.testing.assert_allclose(r, 2.0 * 3.0 ** 2, rtol=1e-5)


def test_single_interaction_term():
    """Set w_xy_02 = 4.0 (the (0,2) pair → index 5) and verify the
    residual = 4 * x_0 * x_2 with everything else zero.

    Pair ordering: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
                     4      5      6      7      8      9
    """
    linear = jnp.zeros(4, dtype=jnp.float32)
    stim = jnp.array([2.0, 0.0, 5.0, 0.0], dtype=jnp.float32)
    poly = jnp.zeros((10,), dtype=jnp.float32).at[5].set(4.0)  # w_xy_(0,2) = 4
    r = float(compute_poly_reward(linear, poly, stim))
    np.testing.assert_allclose(r, 4.0 * 2.0 * 5.0, rtol=1e-5)


def test_all_pairs_indexed_correctly():
    """Sanity-check every pair slot: set one weight at a time, verify the
    correct (i, j) product is selected. Catches index-order bugs."""
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    linear = jnp.zeros(4, dtype=jnp.float32)
    stim = jnp.array([2.0, 3.0, 5.0, 7.0], dtype=jnp.float32)
    for k, (i, j) in enumerate(pairs):
        poly = jnp.zeros((10,), dtype=jnp.float32).at[4 + k].set(1.0)
        expected = float(stim[i] * stim[j])
        r = float(compute_poly_reward(linear, poly, stim))
        np.testing.assert_allclose(r, expected, rtol=1e-5,
            err_msg=f"pair slot {k} (i={i},j={j}) gave {r}, expected {expected}")


def test_full_decomposition(stimuli, linear_genome):
    """A hand-rolled reward should match compute_poly_reward for non-trivial
    poly genome (both quadratic and interaction parts non-zero)."""
    poly = jnp.array(
        [0.1, -0.2, 0.05, 0.3,                # 4 quadratic
         0.4, -0.1, 0.2, 0.0, -0.5, 0.15],    # 6 interaction
        dtype=jnp.float32,
    )

    # Manual computation.
    lin = float(jnp.sum(linear_genome * stimuli * LINEAR_COEFS))
    sq = float(jnp.sum(poly[:4] * stimuli * stimuli))
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    inter = 0.0
    for k, (i, j) in enumerate(pairs):
        inter += float(poly[4 + k]) * float(stimuli[i]) * float(stimuli[j])
    expected = lin + sq + inter

    r = float(compute_poly_reward(linear_genome, poly, stimuli))
    np.testing.assert_allclose(r, expected, rtol=1e-5)


# ─── Mutation: produces non-zero, bounded values ────────────────────────────

def test_mutate_poly_genome_nonzero_and_bounded():
    """Mutating a zero-init poly genome should produce non-zero entries
    that stay within the clip bound."""
    config = {
        "poly_mutation_scale": 0.05,
        "poly_weight_clip": 5.0,
        "mutation_df": 2.0,
    }
    parent = init_poly_genome(jax.random.PRNGKey(0), config)
    assert jnp.all(parent == 0.0)

    child = mutate_poly_genome_jax(parent, jax.random.PRNGKey(42), config)
    assert child.shape == (10,)
    # At least one entry must have moved (default scale=0.05 over 10 draws
    # makes "all zero" essentially impossible).
    assert float(jnp.max(jnp.abs(child))) > 1e-6
    # All entries must respect the clip bound.
    assert jnp.all(jnp.abs(child) <= 5.0 + 1e-5)
    assert jnp.all(jnp.isfinite(child))


def test_mutate_poly_genome_respects_clip_bound():
    """A parent already at the clip bound plus a positive delta must
    still satisfy |child| ≤ clip."""
    config = {
        "poly_mutation_scale": 0.5,    # large scale to provoke clipping
        "poly_weight_clip": 1.0,       # tight clip
        "mutation_df": 2.0,
    }
    parent = jnp.full((10,), 1.0, dtype=jnp.float32)
    for seed in range(20):
        child = mutate_poly_genome_jax(parent, jax.random.PRNGKey(seed), config)
        assert jnp.all(child <= 1.0 + 1e-5)
        assert jnp.all(child >= -1.0 - 1e-5)


def test_mutate_poly_genome_deterministic():
    """Same key + same parent + same config → same child."""
    config = {"poly_mutation_scale": 0.05, "poly_weight_clip": 5.0, "mutation_df": 2.0}
    parent = jnp.zeros((10,), dtype=jnp.float32)
    key = jax.random.PRNGKey(7)
    a = mutate_poly_genome_jax(parent, key, config)
    b = mutate_poly_genome_jax(parent, key, config)
    assert jnp.array_equal(a, b)


# ─── vmap compatibility ─────────────────────────────────────────────────────

def test_compute_poly_reward_vmappable():
    """compute_poly_reward must work under jax.vmap over a batch of agents
    — that's exactly how jax_sim's reward dispatch calls it."""
    N = 5
    rng = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(rng, 3)
    linear_batch = jax.random.normal(k1, (N, 4))
    poly_batch = jax.random.normal(k2, (N, 10)) * 0.1
    stim_batch = jax.random.uniform(k3, (N, 4), minval=0.0, maxval=2.0)

    r_batch = jax.vmap(compute_poly_reward)(linear_batch, poly_batch, stim_batch)
    assert r_batch.shape == (N,)
    assert jnp.all(jnp.isfinite(r_batch))

    # Each batch element should match a single-call result.
    for i in range(N):
        r_i = float(compute_poly_reward(linear_batch[i], poly_batch[i], stim_batch[i]))
        np.testing.assert_allclose(float(r_batch[i]), r_i, rtol=1e-5)


def test_compute_poly_reward_jit_compatible():
    """Sanity check: the function should JIT cleanly."""
    f = jax.jit(compute_poly_reward)
    linear = jnp.array([1.0, 0.0, 0.0, 0.0])
    poly = jnp.zeros(10)
    stim = jnp.array([3.0, 0.0, 0.0, 0.0])
    r = float(f(linear, poly, stim))
    # K&D linear: 1.0 * 1.0 * 3.0 = 3.0
    np.testing.assert_allclose(r, 3.0, rtol=1e-5)
