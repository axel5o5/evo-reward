"""
test_jax_evolution.py
---------------------
Unit tests for src/jax_evolution.py — Student's t mutation and offspring spawning.

These cover the evolution path that turns a parent's reward genome into a
clipped, mutated child and writes a fresh agent into a SimState slot.
Silent bugs in this module (wrong slot writes, missing zeroed fields,
unbounded weights) would quietly corrupt a population mid-run.

Run: pytest tests/test_jax_evolution.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats as scipy_stats

from src.jax_evolution import (
    mutate_genome_jax,
    sample_students_t,
    spawn_offspring_jax,
)
from src.jax_state import init_simstate


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    """Minimal config sized for fast tests. Matches baseline_faithful semantics."""
    return {
        "experiment_name": "test_evo",
        "world_size": 960,
        "total_steps": 1000,
        "obs_dim": 205,
        "prey_initial": 4,
        "predator_initial": 2,
        "prey_cap": 10,
        "predator_cap": 4,
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
        "food_initial": 10,
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
        "minibatch_size": 8,
        "ppo_epochs": 2,
        "clip_epsilon": 0.2,
        "entropy_coef": 0.001,
        "gae_lambda": 0.95,
        "lr": 3.0e-4,
        "adam_eps": 1.0e-7,
        "vf_coef": 0.5,
        "checkpoint_interval_steps": 25_000,
        "log_interval_steps": 10_000,
        "seed": 0,
    }


# ─── sample_students_t ────────────────────────────────────────────────────────

class TestSampleStudentsT:

    def test_shape_and_dtype(self):
        key = jax.random.PRNGKey(0)
        s = sample_students_t(key, (4,), df=2.0, scale=0.4)
        assert s.shape == (4,)
        assert s.dtype == jnp.float32

    def test_deterministic_for_same_key(self):
        key = jax.random.PRNGKey(42)
        a = sample_students_t(key, (100,))
        b = sample_students_t(key, (100,))
        assert jnp.array_equal(a, b)

    def test_scale_is_linear(self):
        """Doubling `scale` doubles every sample."""
        key = jax.random.PRNGKey(7)
        a = sample_students_t(key, (256,), df=2.0, scale=0.4)
        b = sample_students_t(key, (256,), df=2.0, scale=0.8)
        np.testing.assert_allclose(np.asarray(b), 2.0 * np.asarray(a), rtol=1e-5)

    def test_matches_scipy_t_df2(self):
        """Empirical CDF should be close to scipy.stats.t(df=2) within a loose tolerance.

        We test the raw (scale=1.0) distribution against t(df=2). The KS statistic
        for n=20_000 samples from the correct distribution is < 0.02 with high
        probability; we allow 0.05 to keep the test robust on CI.
        """
        key = jax.random.PRNGKey(0)
        samples = np.asarray(sample_students_t(key, (20_000,), df=2.0, scale=1.0))
        # t(df=2) has infinite variance, so trim the tails to get a stable stat.
        trimmed = samples[(samples > -50) & (samples < 50)]
        ks_stat, _ = scipy_stats.kstest(trimmed, scipy_stats.t(df=2).cdf)
        assert ks_stat < 0.05, f"KS stat {ks_stat:.3f} — distribution doesn't match t(df=2)"

    def test_heavier_tailed_than_normal(self):
        """Student's t(df=2) produces more extreme samples than N(0, scale)
        at matched scale. Concrete check: P(|x| > 3*scale) should exceed the
        normal value (~0.27%) by a comfortable margin."""
        key = jax.random.PRNGKey(3)
        scale = 0.4
        samples = np.asarray(sample_students_t(key, (50_000,), df=2.0, scale=scale))
        frac_extreme = np.mean(np.abs(samples) > 3 * scale)
        assert frac_extreme > 0.05, (
            f"Expected heavy-tailed distribution; only {frac_extreme:.3%} "
            "of samples exceeded 3*scale"
        )


# ─── mutate_genome_jax ────────────────────────────────────────────────────────

class TestMutateGenome:

    def test_shape_preserved(self, config):
        parent = jnp.array([1.0, -2.0, 3.0, -4.0])
        child = mutate_genome_jax(parent, jax.random.PRNGKey(0), config)
        assert child.shape == parent.shape
        assert child.dtype == parent.dtype

    def test_finite(self, config):
        parent = jnp.array([0.0, 0.0, 0.0, 0.0])
        child = mutate_genome_jax(parent, jax.random.PRNGKey(1), config)
        assert jnp.all(jnp.isfinite(child))

    def test_deterministic(self, config):
        parent = jnp.array([0.5, -0.5, 1.0, -1.0])
        key = jax.random.PRNGKey(99)
        a = mutate_genome_jax(parent, key, config)
        b = mutate_genome_jax(parent, key, config)
        assert jnp.array_equal(a, b)

    def test_clip_bound_respected(self, config):
        """A parent already at +clip_val plus a positive delta must stay ≤ clip_val."""
        clip = config["weight_clip"]
        parent = jnp.full((4,), clip)
        # Sample 50 mutations; none should exceed the clip bound.
        for i in range(50):
            child = mutate_genome_jax(parent, jax.random.PRNGKey(i), config)
            assert jnp.all(child <= clip + 1e-5)
            assert jnp.all(child >= -clip - 1e-5)

    def test_nonzero_scale_produces_change(self, config):
        """With default scale=0.4 on a 4-vector, it's overwhelmingly likely
        at least one component moves by more than 1e-6."""
        parent = jnp.zeros(4)
        child = mutate_genome_jax(parent, jax.random.PRNGKey(5), config)
        assert jnp.max(jnp.abs(child - parent)) > 1e-6


# ─── spawn_offspring_jax ──────────────────────────────────────────────────────

class TestSpawnOffspring:
    """Integration tests: spawn writes to every SimState field consistently."""

    def test_new_slot_becomes_active(self, config):
        state = init_simstate(config, jax.random.PRNGKey(0))
        n_initial = config["prey_initial"] + config["predator_initial"]
        new_slot = n_initial  # first inactive slot
        assert not bool(state.is_active[new_slot])

        new_state = spawn_offspring_jax(
            state, parent_slot=0, new_slot=new_slot,
            rng_key=jax.random.PRNGKey(1), config=config,
        )
        assert bool(new_state.is_active[new_slot])

    def test_child_inherits_species(self, config):
        state = init_simstate(config, jax.random.PRNGKey(0))
        prey_cap = config["prey_cap"]
        n_prey = config["prey_initial"]
        n_pred = config["predator_initial"]

        # D19 layout: prey slots [0, prey_cap), predator slots [prey_cap, max).
        # Use first inactive slot in each species range as the spawn target.
        prey_new_slot = n_prey                  # first inactive prey slot
        pred_new_slot = prey_cap + n_pred       # first inactive predator slot

        # Prey parent (slot 0 is prey)
        prey_child = spawn_offspring_jax(
            state, 0, prey_new_slot, jax.random.PRNGKey(1), config,
        )
        assert int(prey_child.species[prey_new_slot]) == int(state.species[0]) == 0

        # Predator parent: first predator slot is prey_cap (not prey_initial)
        pred_parent_slot = prey_cap
        pred_child = spawn_offspring_jax(
            state, pred_parent_slot, pred_new_slot, jax.random.PRNGKey(2), config,
        )
        assert int(pred_child.species[pred_new_slot]) == 1

    def test_child_energy_is_parent_share(self, config):
        state = init_simstate(config, jax.random.PRNGKey(0))
        n_initial = config["prey_initial"] + config["predator_initial"]
        parent_e = float(state.energies[0])

        new_state = spawn_offspring_jax(
            state, 0, n_initial, jax.random.PRNGKey(1), config,
        )
        child_e = float(new_state.energies[n_initial])
        expected = parent_e * config["energy_share_ratio"]
        np.testing.assert_allclose(child_e, expected, rtol=1e-6)

    def test_child_identity_and_parent_link(self, config):
        state = init_simstate(config, jax.random.PRNGKey(0))
        n_initial = config["prey_initial"] + config["predator_initial"]
        parent_slot = 3
        parent_id = int(state.agent_ids[parent_slot])
        expected_child_id = int(state.next_agent_id)

        new_state = spawn_offspring_jax(
            state, parent_slot, n_initial, jax.random.PRNGKey(1), config,
        )
        assert int(new_state.agent_ids[n_initial]) == expected_child_id
        assert int(new_state.parent_ids[n_initial]) == parent_id
        assert int(new_state.next_agent_id) == expected_child_id + 1
        assert int(new_state.ages[n_initial]) == 0

    def test_child_genome_mutated_and_clipped(self, config):
        state = init_simstate(config, jax.random.PRNGKey(0))
        n_initial = config["prey_initial"] + config["predator_initial"]

        # Force parent genome to the clip boundary so we can verify clipping.
        clip = config["weight_clip"]
        parent_genome = jnp.array([clip, -clip, clip, -clip])
        state = state.replace(
            reward_weights=state.reward_weights.at[0].set(parent_genome),
        )
        new_state = spawn_offspring_jax(
            state, 0, n_initial, jax.random.PRNGKey(11), config,
        )
        child_genome = new_state.reward_weights[n_initial]
        assert child_genome.shape == (4,)
        assert jnp.all(jnp.abs(child_genome) <= clip + 1e-5)

    def test_child_position_within_bounds(self, config):
        state = init_simstate(config, jax.random.PRNGKey(0))
        n_initial = config["prey_initial"] + config["predator_initial"]
        world_size = config["world_size"]

        for seed in range(8):
            new_state = spawn_offspring_jax(
                state, 0, n_initial, jax.random.PRNGKey(seed), config,
            )
            child_pos = new_state.phyjax_stated.get("circle").p.xy[n_initial]
            assert jnp.all(child_pos >= 0.0)
            assert jnp.all(child_pos <= world_size)

    def test_rollout_buffers_zeroed_at_new_slot(self, config):
        state = init_simstate(config, jax.random.PRNGKey(0))
        n_initial = config["prey_initial"] + config["predator_initial"]

        # Dirty the buffers at new_slot so "zeroed" means something.
        dirty = lambda arr: arr.at[n_initial].set(jnp.ones_like(arr[n_initial]))
        state = state.replace(
            rollout_obs=dirty(state.rollout_obs),
            rollout_actions=dirty(state.rollout_actions),
            rollout_log_probs=dirty(state.rollout_log_probs),
            rollout_rewards=dirty(state.rollout_rewards),
            rollout_values=dirty(state.rollout_values),
            rollout_dones=state.rollout_dones.at[n_initial].set(True),
            rollout_ptrs=state.rollout_ptrs.at[n_initial].set(999),
            obs_buffer=dirty(state.obs_buffer),
            lstm_hidden=dirty(state.lstm_hidden),
            rollout_init_hidden=dirty(state.rollout_init_hidden),
        )

        new_state = spawn_offspring_jax(
            state, 0, n_initial, jax.random.PRNGKey(1), config,
        )
        assert jnp.all(new_state.rollout_obs[n_initial] == 0.0)
        assert jnp.all(new_state.rollout_actions[n_initial] == 0.0)
        assert jnp.all(new_state.rollout_log_probs[n_initial] == 0.0)
        assert jnp.all(new_state.rollout_rewards[n_initial] == 0.0)
        assert jnp.all(new_state.rollout_values[n_initial] == 0.0)
        assert not jnp.any(new_state.rollout_dones[n_initial])
        assert int(new_state.rollout_ptrs[n_initial]) == 0
        assert jnp.all(new_state.obs_buffer[n_initial] == 0.0)
        assert jnp.all(new_state.lstm_hidden[n_initial] == 0.0)
        assert jnp.all(new_state.rollout_init_hidden[n_initial] == 0.0)

    def test_child_physics_velocity_is_zero(self, config):
        state = init_simstate(config, jax.random.PRNGKey(0))
        n_initial = config["prey_initial"] + config["predator_initial"]

        new_state = spawn_offspring_jax(
            state, 0, n_initial, jax.random.PRNGKey(1), config,
        )
        circle = new_state.phyjax_stated.get("circle")
        assert jnp.all(circle.v.xy[n_initial] == 0.0)
        assert float(circle.v.angle[n_initial]) == 0.0
        assert bool(circle.is_active[n_initial])

    def test_other_slots_untouched(self, config):
        """Spawning into slot N must not disturb any other slot."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        n_initial = config["prey_initial"] + config["predator_initial"]
        new_slot = n_initial

        new_state = spawn_offspring_jax(
            state, 0, new_slot, jax.random.PRNGKey(1), config,
        )
        for slot in [0, 1, n_initial - 1]:
            assert bool(new_state.is_active[slot]) == bool(state.is_active[slot])
            assert int(new_state.species[slot]) == int(state.species[slot])
            assert float(new_state.energies[slot]) == float(state.energies[slot])
            assert jnp.array_equal(
                new_state.reward_weights[slot], state.reward_weights[slot]
            )

    def test_no_nan_or_inf_anywhere(self, config):
        """Full spawn shouldn't introduce NaN/Inf in any float field."""
        import jax.tree_util as jtu
        state = init_simstate(config, jax.random.PRNGKey(0))
        n_initial = config["prey_initial"] + config["predator_initial"]

        new_state = spawn_offspring_jax(
            state, 0, n_initial, jax.random.PRNGKey(1), config,
        )

        leaves = jtu.tree_leaves(new_state)
        for leaf in leaves:
            arr = jnp.asarray(leaf)
            if jnp.issubdtype(arr.dtype, jnp.floating):
                assert jnp.all(jnp.isfinite(arr)), (
                    f"Non-finite value in float leaf with shape {arr.shape}"
                )
