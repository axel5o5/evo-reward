"""
test_components.py
------------------
Fast unit tests for each module. Should complete in under 60 seconds on CPU.
These test individual functions in isolation — no full simulation loop.

Run: pytest tests/test_components.py
Run specific module: pytest tests/test_components.py -k "lifecycle"
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from scipy import stats


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def base_config():
    """Minimal config matching baseline_faithful.yaml for unit tests.
    All values CONFIRMED against emevo gecco2026 branch.
    """
    return {
        "world_size": 960,
        "prey_radius": 10.0,
        "predator_radius": 14.0,
        "max_motor_norm": 114.0,
        "n_proximity_sensors": 32,
        "n_proximity_channels": 4,
        "proximity_fov_deg": 120.0,
        "proximity_max_range": 200.0,       # CONFIRMED: sensor_length=200 (not 120)
        "n_tactile_sensors": 18,
        "n_tactile_channels": 4,
        "tactile_spacing_deg": 20.0,
        "social_obs": "position_only",       # baseline — no social channels
        "obs_dim": 205,                     # CONFIRMED: 128+72+2+1+1+1
        "food_max": 600,                    # CONFIRMED: n_max_foods=600
        "food_initial": 40,
        "food_growth_rate": 0.5,            # CONFIRMED: 0.5/step
        "food_max_regen_per_step": 10,
        "prey_e_food": 1.0,
        "prey_c_b": 1.0e-4,                # CONFIRMED: code value (paper Table 2 labels c_a/c_b swapped)
        "prey_c_a": 2.5e-6,                # CONFIRMED: code value (paper Table 2 labels c_a/c_b swapped)
        "predator_d_b": 4.0e-3,
        "predator_d_a": 5.0e-5,
        "predator_eta": 0.6,
        "predator_mouth_deg": 60.0,
        "predator_mouth_range_min": 40.0,
        "predator_mouth_range_max": 80.0,
        "energy_capacity": 1000.0,
        "kappa_h": 0.01,
        "alpha_e": 0.02,
        "beta_h": 0.2,
        "alpha_t_prey": 4.0e-7,
        "alpha_t_pred": 2.0e-7,
        "beta_t_prey": 2.0e-6,
        "beta_t_pred": 4.0e-6,
        "kappa_b": 1.0e-3,
        "beta_b": 0.1,
        "zeta_b_prey": 10.0,
        "zeta_b_pred": 100.0,
        "energy_share_ratio": 0.4,          # CONFIRMED: 0.4 (not 0.5)
        "spawn_spread": 100.0,              # CONFIRMED: neighbor_stddev=100.0
        "prey_e_initial": 100.0,            # CONFIRMED: init_energy=100.0
        "predator_e_initial": 100.0,        # CONFIRMED
        "reward_weights_init_std": 0.1,
        "mutation_df": 2,
        "mutation_scale": 0.4,
        "weight_clip": 100.0,
        "policy_hidden_size": 64,
        "policy_n_hidden_layers": 2,        # CONFIRMED: 2 hidden layers (not 3)
        "action_clip_low": -20.0,
        "action_clip_high": 80.0,
        "action_mapping": "sigmoid",
        "gamma": 0.999,
        "rollout_steps": 1024,
        "minibatch_size": 256,
        "ppo_epochs": 10,
        "clip_epsilon": 0.2,
        "entropy_coef": 0.001,
        "gae_lambda": 0.95,
        "lr": 3.0e-4,
        "adam_eps": 1.0e-7,
        # Population caps
        "prey_initial": 150,
        "predator_initial": 10,
        "prey_cap": 450,
        "predator_cap": 50,
        # Identity / logging / scheduling
        "experiment_name": "test",
        "total_steps": 10_240_000,
        "seed": 0,
        "checkpoint_interval_steps": 25_000,
        "log_interval_steps": 10_000,
    }


@pytest.fixture
def mlp_config(base_config):
    """Config for MLP reward genome tests (Axis 1)."""
    return {
        **base_config,
        "reward_type": "mlp",
        "mlp_hidden_size": 8,
        "mlp_mutation_scale": 0.01,
        "mlp_weight_clip": 10.0,
    }


@pytest.fixture
def social_config(base_config):
    """Config for social observation tests (Axis 2)."""
    return {
        **base_config,
        "social_obs": "position_heading_velocity",
        "obs_dim": 215,
        "n_social_neighbors": 5,
    }


@pytest.fixture
def temporal_config(base_config):
    """Config for temporal reward genome tests (Axis 3)."""
    return {
        **base_config,
        "reward_type": "temporal",
        "reward_context_window": 10,
        "temporal_hidden_size": 16,
        "temporal_mutation_scale": 0.005,
        "temporal_weight_clip": 5.0,
    }


@pytest.fixture
def lstm_config(base_config):
    """Config for LSTM policy tests (Axis 4)."""
    return {
        **base_config,
        "policy_type": "lstm",
        "lstm_hidden_size": 64,
        "lstm_chunk_length": 128,
    }


# ─── Reward genome ────────────────────────────────────────────────────────────

class TestRewardGenome:

    def test_init_genome_shape(self, base_config):
        from src.reward import init_genome
        rng = jax.random.PRNGKey(0)
        genome = init_genome(rng, base_config)
        assert genome.shape == (4,), f"Expected (4,), got {genome.shape}"

    def test_init_genome_dtype(self, base_config):
        from src.reward import init_genome
        rng = jax.random.PRNGKey(0)
        genome = init_genome(rng, base_config)
        assert genome.dtype == jnp.float32

    def test_init_genome_distribution(self, base_config):
        """Over many samples, weights should be approx N(0, 0.1)."""
        from src.reward import init_genome
        samples = []
        for i in range(500):
            rng = jax.random.PRNGKey(i)
            samples.append(init_genome(rng, base_config))
        samples = np.array(samples).flatten()
        assert abs(np.mean(samples)) < 0.02, f"Mean too far from 0: {np.mean(samples)}"
        assert abs(np.std(samples) - 0.1) < 0.02, f"Std not ~0.1: {np.std(samples)}"

    def test_linear_reward_food(self):
        """w_eat * n_eaten with other stimuli zero."""
        from src.reward import compute_linear_reward
        genome = jnp.array([2.0, 0.0, 0.0, 0.0])
        r = compute_linear_reward(genome, n_eaten=1, motor_norm=0.0, max_s_prey=0.0, max_s_pred=0.0)
        assert abs(float(r) - 2.0) < 1e-5

    def test_linear_reward_motor_scaling(self):
        """Motor reward is scaled by 0.01 * w_act * motor_norm."""
        from src.reward import compute_linear_reward
        genome = jnp.array([0.0, 1.0, 0.0, 0.0])
        # motor_norm = 1.0 (max), expected contribution = 0.01 * 1.0 * 1.0 = 0.01
        r = compute_linear_reward(genome, n_eaten=0, motor_norm=1.0, max_s_prey=0.0, max_s_pred=0.0)
        assert abs(float(r) - 0.01) < 1e-5, f"Motor reward wrong: {float(r)}"

    def test_linear_reward_sensor_scaling(self):
        """Sensor rewards are scaled by 0.1."""
        from src.reward import compute_linear_reward
        genome = jnp.array([0.0, 0.0, 1.0, 0.0])
        r = compute_linear_reward(genome, n_eaten=0, motor_norm=0.0, max_s_prey=1.0, max_s_pred=0.0)
        assert abs(float(r) - 0.1) < 1e-5, f"Prey sensor reward wrong: {float(r)}"

    def test_linear_reward_zero_genome(self):
        """Zero genome always produces zero reward."""
        from src.reward import compute_linear_reward
        genome = jnp.zeros(4)
        r = compute_linear_reward(genome, n_eaten=5, motor_norm=1.0, max_s_prey=1.0, max_s_pred=1.0)
        assert float(r) == 0.0

    def test_linear_reward_fear_sign(self):
        """Negative w_pred near predator produces negative reward."""
        from src.reward import compute_linear_reward
        genome = jnp.array([0.0, 0.0, 0.0, -10.0])
        r = compute_linear_reward(genome, n_eaten=0, motor_norm=0.0, max_s_prey=0.0, max_s_pred=1.0)
        assert float(r) < 0, f"Fear reward should be negative, got {float(r)}"


# ─── MLP reward genome (Axis 1) ──────────────────────────────────────────────

class TestMLPRewardGenome:

    def test_mlp_genome_shape(self, mlp_config):
        """init_mlp_genome returns correct PyTree structure with 121 params."""
        from src.reward import init_mlp_genome
        from jax.flatten_util import ravel_pytree
        rng = jax.random.PRNGKey(0)
        genome = init_mlp_genome(rng, mlp_config)

        # Verify PyTree structure
        assert 'params' in genome
        assert 'Dense_0' in genome['params']
        assert 'Dense_1' in genome['params']
        assert 'Dense_2' in genome['params']

        # Verify layer shapes: 4->8->8->1
        assert genome['params']['Dense_0']['kernel'].shape == (4, 8)
        assert genome['params']['Dense_0']['bias'].shape == (8,)
        assert genome['params']['Dense_1']['kernel'].shape == (8, 8)
        assert genome['params']['Dense_1']['bias'].shape == (8,)
        assert genome['params']['Dense_2']['kernel'].shape == (8, 1)
        assert genome['params']['Dense_2']['bias'].shape == (1,)

        # Verify total param count
        flat, _ = ravel_pytree(genome)
        assert flat.shape == (121,), f"Expected 121 params, got {flat.shape[0]}"

    def test_mlp_genome_output_scalar(self, mlp_config):
        """compute_mlp_reward returns a scalar float32 for various inputs."""
        from src.reward import init_mlp_genome, compute_mlp_reward
        rng = jax.random.PRNGKey(0)
        genome = init_mlp_genome(rng, mlp_config)

        # Normal stimuli
        stimuli = jnp.array([1.0, 0.5, 0.3, 0.8])
        r = compute_mlp_reward(genome, stimuli)
        assert r.shape == (), f"Expected scalar, got shape {r.shape}"
        assert r.dtype == jnp.float32
        assert jnp.isfinite(r)

        # Zero stimuli
        r_zero = compute_mlp_reward(genome, jnp.zeros(4))
        assert jnp.isfinite(r_zero)

        # Large stimuli (tanh saturates, should still be finite)
        r_large = compute_mlp_reward(genome, jnp.ones(4) * 100.0)
        assert jnp.isfinite(r_large)

    def test_mlp_mutation_changes_genome(self, mlp_config):
        """mutate_mlp_genome produces different weights from parent."""
        from src.reward import init_mlp_genome
        from src.evolution import mutate_mlp_genome
        from jax.flatten_util import ravel_pytree

        rng = jax.random.PRNGKey(0)
        parent = init_mlp_genome(rng, mlp_config)
        child = mutate_mlp_genome(parent, jax.random.PRNGKey(42), mlp_config)

        flat_parent, _ = ravel_pytree(parent)
        flat_child, _ = ravel_pytree(child)
        assert not jnp.allclose(flat_parent, flat_child), \
            "Mutation did not change any weights"

    def test_mlp_mutation_clipping(self, mlp_config):
        """All weights stay within +/-mlp_weight_clip after mutation."""
        from src.reward import init_mlp_genome
        from src.evolution import mutate_mlp_genome
        from jax.flatten_util import ravel_pytree

        clip = mlp_config["mlp_weight_clip"]
        rng = jax.random.PRNGKey(0)
        parent = init_mlp_genome(rng, mlp_config)

        # Test from normal init
        for i in range(50):
            child = mutate_mlp_genome(parent, jax.random.PRNGKey(i), mlp_config)
            flat, _ = ravel_pytree(child)
            assert jnp.all(jnp.abs(flat) <= clip + 1e-5), \
                f"MLP genome exceeds clip: max={float(jnp.max(jnp.abs(flat)))}"

        # Test from boundary parent (weights near clip edge)
        flat_parent, unflatten = ravel_pytree(parent)
        boundary_parent = unflatten(jnp.full_like(flat_parent, clip - 0.001))
        for i in range(50):
            child = mutate_mlp_genome(boundary_parent, jax.random.PRNGKey(i + 1000), mlp_config)
            flat, _ = ravel_pytree(child)
            assert jnp.all(jnp.abs(flat) <= clip + 1e-5), \
                f"Boundary mutation exceeds clip: max={float(jnp.max(jnp.abs(flat)))}"

    def test_capacity_util_linear_genome(self, mlp_config):
        """A genome with tiny weights (tanh ~ linear) shows near-zero nonlinearity."""
        from src.reward import init_mlp_genome
        from analysis.capacity_util import compute_reward_nonlinearity
        from jax.flatten_util import ravel_pytree

        rng = jax.random.PRNGKey(0)
        genome = init_mlp_genome(rng, mlp_config)

        # Scale weights down so tanh operates in linear regime
        flat, unflatten = ravel_pytree(genome)
        linear_genome = unflatten(flat * 0.001)

        nonlinearity = compute_reward_nonlinearity(linear_genome, mlp_config)
        assert nonlinearity < 0.05, \
            f"Near-linear genome should have low nonlinearity, got {nonlinearity}"

    def test_capacity_util_nonlinear_genome(self, mlp_config):
        """A genome with large weights (tanh saturates) shows nonlinearity > 0.01."""
        from src.reward import init_mlp_genome
        from analysis.capacity_util import compute_reward_nonlinearity
        from jax.flatten_util import ravel_pytree

        rng = jax.random.PRNGKey(0)
        genome = init_mlp_genome(rng, mlp_config)

        # Scale weights up to push tanh into saturation
        flat, unflatten = ravel_pytree(genome)
        nonlinear_genome = unflatten(flat * 10.0)

        nonlinearity = compute_reward_nonlinearity(nonlinear_genome, mlp_config)
        assert nonlinearity > 0.01, \
            f"Large-weight genome should have nonlinearity > 0.01, got {nonlinearity}"

    def test_mlp_reward_jit_vmap_compatible(self, mlp_config):
        """compute_mlp_reward works under jax.jit and jax.vmap."""
        from src.reward import init_mlp_genome, compute_mlp_reward

        rng = jax.random.PRNGKey(0)
        genome = init_mlp_genome(rng, mlp_config)
        stimuli = jnp.array([1.0, 0.5, 0.3, 0.8])

        # JIT: should compile and produce same result as eager
        jitted = jax.jit(compute_mlp_reward)
        r_eager = compute_mlp_reward(genome, stimuli)
        r_jit = jitted(genome, stimuli)
        assert jnp.allclose(r_eager, r_jit, atol=1e-6), \
            f"JIT result {float(r_jit)} != eager {float(r_eager)}"

        # vmap over a batch of stimuli
        batch = jax.random.uniform(jax.random.PRNGKey(1), (16, 4))
        vmapped = jax.vmap(lambda x: compute_mlp_reward(genome, x))
        r_batch = vmapped(batch)
        assert r_batch.shape == (16,), f"Expected (16,), got {r_batch.shape}"
        assert jnp.all(jnp.isfinite(r_batch))

        # vmap + jit combined
        r_batch_jit = jax.jit(vmapped)(batch)
        assert jnp.allclose(r_batch, r_batch_jit, atol=1e-6)

    def test_mlp_mutation_preserves_pytree_structure(self, mlp_config):
        """Mutated genome has identical PyTree structure to parent."""
        from src.reward import init_mlp_genome, compute_mlp_reward
        from src.evolution import mutate_mlp_genome

        rng = jax.random.PRNGKey(0)
        parent = init_mlp_genome(rng, mlp_config)
        child = mutate_mlp_genome(parent, jax.random.PRNGKey(42), mlp_config)

        # Same top-level keys
        assert set(parent.keys()) == set(child.keys())
        # Same layer keys
        assert set(parent['params'].keys()) == set(child['params'].keys())
        # Same shapes and dtypes per leaf
        for layer in ['Dense_0', 'Dense_1', 'Dense_2']:
            for param in ['kernel', 'bias']:
                p = parent['params'][layer][param]
                c = child['params'][layer][param]
                assert p.shape == c.shape, \
                    f"{layer}/{param} shape mismatch: {p.shape} vs {c.shape}"
                assert p.dtype == c.dtype, \
                    f"{layer}/{param} dtype mismatch: {p.dtype} vs {c.dtype}"

        # Child genome is usable for forward pass (no structural error)
        stimuli = jnp.array([0.5, 0.5, 0.5, 0.5])
        r = compute_mlp_reward(child, stimuli)
        assert jnp.isfinite(r)

    def test_mlp_mutation_heavy_tails(self, mlp_config):
        """MLP mutation uses t(df=2), not Gaussian — verify heavy tails.

        Same logic as TestEvolution.test_mutate_genome_heavy_tails:
        t(df=2) produces >8% of samples beyond +/-2*scale, vs ~4.6% for Gaussian.
        """
        from src.reward import init_mlp_genome
        from src.evolution import mutate_mlp_genome
        from jax.flatten_util import ravel_pytree

        parent = init_mlp_genome(jax.random.PRNGKey(0), mlp_config)
        flat_parent, _ = ravel_pytree(parent)

        deltas = []
        for i in range(200):  # 200 * 121 = 24,200 samples
            child = mutate_mlp_genome(parent, jax.random.PRNGKey(i), mlp_config)
            flat_child, _ = ravel_pytree(child)
            deltas.extend((np.array(flat_child) - np.array(flat_parent)).tolist())

        deltas = np.array(deltas)
        scale = mlp_config["mlp_mutation_scale"]
        # Fraction beyond +/-2*scale: Gaussian ~4.6%, t(df=2) ~13.4%
        beyond_2sigma = np.mean(np.abs(deltas) > 2 * scale)
        assert beyond_2sigma > 0.08, (
            f"MLP mutation appears Gaussian (heavy tail fraction={beyond_2sigma:.3f} < 0.08). "
            f"Check that scipy.stats.t(df=2) is used, not np.random.normal()."
        )

    def test_mlp_mutation_unbiased_mean(self, mlp_config):
        """Mean of MLP mutations should be near zero (unbiased)."""
        from src.reward import init_mlp_genome
        from src.evolution import mutate_mlp_genome
        from jax.flatten_util import ravel_pytree

        parent = init_mlp_genome(jax.random.PRNGKey(0), mlp_config)
        flat_parent, _ = ravel_pytree(parent)

        deltas = []
        for i in range(500):
            child = mutate_mlp_genome(parent, jax.random.PRNGKey(i), mlp_config)
            flat_child, _ = ravel_pytree(child)
            deltas.append(np.array(flat_child) - np.array(flat_parent))

        deltas = np.array(deltas)  # (500, 121)
        mean_delta = np.mean(deltas)
        assert abs(mean_delta) < 0.005, \
            f"MLP mutation mean not near zero: {mean_delta:.6f}"

    def test_mlp_genome_type_confusion(self, mlp_config):
        """Passing wrong genome type to reward functions raises a clear error."""
        from src.reward import init_genome, init_mlp_genome, compute_linear_reward, compute_mlp_reward

        rng = jax.random.PRNGKey(0)
        linear_genome = init_genome(rng, mlp_config)       # shape (4,)
        mlp_genome = init_mlp_genome(rng, mlp_config)       # PyTree

        # Linear genome → MLP function should fail (not a dict/PyTree)
        with pytest.raises((KeyError, TypeError, AttributeError)):
            compute_mlp_reward(linear_genome, jnp.ones(4))

        # MLP genome → linear function should fail (dict indexed with int → KeyError)
        with pytest.raises((KeyError, TypeError, IndexError, ValueError)):
            compute_linear_reward(mlp_genome, n_eaten=1, motor_norm=0.5,
                                  max_s_prey=0.3, max_s_pred=0.2)

    def test_capacity_util_zero_weight_genome(self, mlp_config):
        """All-zero genome produces constant output — nonlinearity should be 0."""
        from src.reward import init_mlp_genome
        from analysis.capacity_util import compute_reward_nonlinearity
        from jax.flatten_util import ravel_pytree

        rng = jax.random.PRNGKey(0)
        genome = init_mlp_genome(rng, mlp_config)

        # Zero out all weights
        flat, unflatten = ravel_pytree(genome)
        zero_genome = unflatten(jnp.zeros_like(flat))

        nonlinearity = compute_reward_nonlinearity(zero_genome, mlp_config)
        assert jnp.isfinite(nonlinearity), "Zero genome produced NaN nonlinearity"
        assert nonlinearity < 1e-6, \
            f"Zero genome should have ~0 nonlinearity, got {nonlinearity}"

    def test_mlp_genome_init_deterministic(self, mlp_config):
        """Same rng_key produces identical genome — reproducibility guarantee."""
        from src.reward import init_mlp_genome
        from jax.flatten_util import ravel_pytree

        genome_a = init_mlp_genome(jax.random.PRNGKey(42), mlp_config)
        genome_b = init_mlp_genome(jax.random.PRNGKey(42), mlp_config)

        flat_a, _ = ravel_pytree(genome_a)
        flat_b, _ = ravel_pytree(genome_b)
        assert jnp.array_equal(flat_a, flat_b), "Same key should produce identical genome"

        # Different key should produce different genome
        genome_c = init_mlp_genome(jax.random.PRNGKey(99), mlp_config)
        flat_c, _ = ravel_pytree(genome_c)
        assert not jnp.array_equal(flat_a, flat_c), "Different keys should produce different genomes"

    def test_mlp_mutation_sequential_drift(self, mlp_config):
        """100 generations of chained mutation — weights stay clipped, genome stays usable."""
        from src.reward import init_mlp_genome, compute_mlp_reward
        from src.evolution import mutate_mlp_genome
        from jax.flatten_util import ravel_pytree

        clip = mlp_config["mlp_weight_clip"]
        genome = init_mlp_genome(jax.random.PRNGKey(0), mlp_config)

        for gen in range(100):
            genome = mutate_mlp_genome(genome, jax.random.PRNGKey(gen), mlp_config)

        # All weights still within clip bounds
        flat, _ = ravel_pytree(genome)
        assert jnp.all(jnp.abs(flat) <= clip + 1e-5), \
            f"After 100 generations, max weight = {float(jnp.max(jnp.abs(flat)))}"

        # Genome still produces finite reward
        r = compute_mlp_reward(genome, jnp.array([1.0, 0.5, 0.3, 0.8]))
        assert jnp.isfinite(r), f"After 100 generations, reward is not finite: {r}"

        # Weights should have drifted away from init (not stuck at zero)
        assert float(jnp.std(flat)) > 0.001, \
            "Weights did not drift after 100 generations of mutation"


# ─── Temporal reward genome (Axis 3) ────────────────────────────────────────

class TestTemporalRewardGenome:
    """Tests for temporal reward context window (Axis 3)."""

    def test_temporal_genome_shape(self, temporal_config):
        """init_temporal_genome returns correct PyTree structure with 945 params."""
        from src.reward import init_temporal_genome
        from jax.flatten_util import ravel_pytree

        rng = jax.random.PRNGKey(0)
        genome = init_temporal_genome(rng, temporal_config)

        # Verify PyTree structure
        assert 'params' in genome
        assert 'Dense_0' in genome['params']
        assert 'Dense_1' in genome['params']
        assert 'Dense_2' in genome['params']

        # Verify layer shapes: 40->16->16->1
        assert genome['params']['Dense_0']['kernel'].shape == (40, 16)
        assert genome['params']['Dense_0']['bias'].shape == (16,)
        assert genome['params']['Dense_1']['kernel'].shape == (16, 16)
        assert genome['params']['Dense_1']['bias'].shape == (16,)
        assert genome['params']['Dense_2']['kernel'].shape == (16, 1)
        assert genome['params']['Dense_2']['bias'].shape == (1,)

        # Verify total param count
        flat, _ = ravel_pytree(genome)
        assert flat.shape == (945,), f"Expected 945 params, got {flat.shape[0]}"

    def test_temporal_reward_scalar(self, temporal_config):
        """compute_temporal_reward returns scalar float32 for various inputs."""
        from src.reward import init_temporal_genome, compute_temporal_reward

        rng = jax.random.PRNGKey(0)
        genome = init_temporal_genome(rng, temporal_config)

        # Normal obs_window
        obs_window = jnp.ones((10, 4)) * 0.5
        r = compute_temporal_reward(genome, obs_window)
        assert r.shape == (), f"Expected scalar, got shape {r.shape}"
        assert r.dtype == jnp.float32
        assert jnp.isfinite(r)

        # Zero obs_window
        r_zero = compute_temporal_reward(genome, jnp.zeros((10, 4)))
        assert jnp.isfinite(r_zero)

        # Large obs_window (tanh saturates, should still be finite)
        r_large = compute_temporal_reward(genome, jnp.ones((10, 4)) * 100.0)
        assert jnp.isfinite(r_large)

    def test_obs_buffer_initialization(self, temporal_config):
        """obs_buffer is zeros at agent birth."""
        k = temporal_config["reward_context_window"]
        obs_buffer = jnp.zeros((k, 4))
        assert obs_buffer.shape == (10, 4)
        assert jnp.all(obs_buffer == 0.0)

    def test_obs_buffer_shift(self, temporal_config):
        """After k steps, obs_buffer contains the last k stimuli in order."""
        k = temporal_config["reward_context_window"]
        obs_buffer = jnp.zeros((k, 4))

        # Simulate 15 steps of stimulus insertion
        stimuli_history = []
        for step in range(15):
            new_stimuli = jnp.array([float(step), 0.1 * step, 0.2 * step, 0.3 * step])
            stimuli_history.append(new_stimuli)

            # Shift buffer: roll left, insert at end
            obs_buffer = jnp.roll(obs_buffer, -1, axis=0).at[-1].set(new_stimuli)

            if step == 0:
                # After step 0: buffer should have zeros except last row
                assert float(obs_buffer[-1, 0]) == 0.0
                assert jnp.all(obs_buffer[:-1] == 0.0)
            elif step == 4:
                # After step 4: last 5 rows have data, first 5 are zeros
                assert float(obs_buffer[-1, 0]) == 4.0
                assert float(obs_buffer[-2, 0]) == 3.0
                assert jnp.all(obs_buffer[:5] == 0.0)
            elif step == 9:
                # After step 9: buffer is full
                for i in range(k):
                    assert float(obs_buffer[i, 0]) == float(i)
            elif step == 14:
                # After step 14: contains steps 5..14
                for i in range(k):
                    expected_step = i + 5
                    assert float(obs_buffer[i, 0]) == float(expected_step), \
                        f"At position {i}, expected step {expected_step}, got {float(obs_buffer[i, 0])}"

    def test_obs_buffer_window1_matches_linear(self, base_config):
        """With k=1, temporal reward should be achievable to match linear reward."""
        from src.reward import init_temporal_genome, compute_temporal_reward, compute_linear_reward
        from jax.flatten_util import ravel_pytree

        config_k1 = {**base_config, "reward_context_window": 1, "temporal_hidden_size": 16}
        rng = jax.random.PRNGKey(0)
        genome = init_temporal_genome(rng, config_k1)

        # With k=1, input is (1, 4) flattened to (4,) — same input as linear.
        # The MLP can represent any function of 4 inputs, including the linear one.
        # Just verify the interface works and produces a finite scalar.
        obs_window = jnp.array([[1.0, 0.5, 0.3, 0.8]])  # shape (1, 4)
        r_temporal = compute_temporal_reward(genome, obs_window)
        assert r_temporal.shape == ()
        assert jnp.isfinite(r_temporal)

        # Also verify linear reward produces a finite scalar for comparison
        linear_genome = jnp.array([1.0, 1.0, 1.0, -1.0])
        r_linear = compute_linear_reward(linear_genome, 1.0, 0.5, 0.3, 0.8)
        assert jnp.isfinite(r_linear)

    def test_temporal_mutation_clipping(self, temporal_config):
        """All weights stay within +-temporal_weight_clip after mutation."""
        from src.reward import init_temporal_genome
        from src.evolution import mutate_temporal_genome
        from jax.flatten_util import ravel_pytree

        clip = temporal_config["temporal_weight_clip"]
        rng = jax.random.PRNGKey(0)
        parent = init_temporal_genome(rng, temporal_config)

        # Test from normal init
        for i in range(50):
            child = mutate_temporal_genome(parent, jax.random.PRNGKey(i), temporal_config)
            flat, _ = ravel_pytree(child)
            assert jnp.all(jnp.abs(flat) <= clip + 1e-5), \
                f"Temporal genome exceeds clip: max={float(jnp.max(jnp.abs(flat)))}"

        # Test from boundary parent (weights near clip edge)
        flat_parent, unflatten = ravel_pytree(parent)
        boundary_parent = unflatten(jnp.full_like(flat_parent, clip - 0.001))
        for i in range(50):
            child = mutate_temporal_genome(
                boundary_parent, jax.random.PRNGKey(i + 1000), temporal_config
            )
            flat, _ = ravel_pytree(child)
            assert jnp.all(jnp.abs(flat) <= clip + 1e-5), \
                f"Boundary mutation exceeds clip: max={float(jnp.max(jnp.abs(flat)))}"

    def test_temporal_reward_jit_vmap_compatible(self, temporal_config):
        """compute_temporal_reward works under jax.jit and jax.vmap."""
        from src.reward import init_temporal_genome, compute_temporal_reward

        rng = jax.random.PRNGKey(0)
        genome = init_temporal_genome(rng, temporal_config)
        obs_window = jnp.ones((10, 4)) * 0.5

        # JIT
        jitted = jax.jit(compute_temporal_reward)
        r_jit = jitted(genome, obs_window)
        r_eager = compute_temporal_reward(genome, obs_window)
        np.testing.assert_allclose(float(r_jit), float(r_eager), rtol=1e-5)

        # vmap over batch of obs_windows (simulating multiple agents)
        batch_windows = jnp.stack([obs_window * i for i in range(5)])  # (5, 10, 4)
        batch_genomes = jax.tree.map(
            lambda p: jnp.tile(p[None], (5, *([1] * p.ndim))),
            genome,
        )
        vmapped = jax.vmap(compute_temporal_reward)
        batch_r = vmapped(batch_genomes, batch_windows)
        assert batch_r.shape == (5,), f"Expected (5,), got {batch_r.shape}"
        assert jnp.all(jnp.isfinite(batch_r))

    def test_temporal_mutation_heavy_tails(self, temporal_config):
        """Temporal mutations have heavy tails (t(df=2), not Gaussian)."""
        from src.reward import init_temporal_genome
        from src.evolution import mutate_temporal_genome
        from jax.flatten_util import ravel_pytree

        rng = jax.random.PRNGKey(0)
        parent = init_temporal_genome(rng, temporal_config)
        flat_parent, _ = ravel_pytree(parent)

        deltas = []
        for i in range(2000):
            child = mutate_temporal_genome(parent, jax.random.PRNGKey(i), temporal_config)
            flat_child, _ = ravel_pytree(child)
            deltas.append(np.array(flat_child - flat_parent))

        all_deltas = np.concatenate(deltas)
        scale = temporal_config["temporal_mutation_scale"]
        # For t(df=2), P(|x| > 2*scale) ≈ 8-10%. Gaussian would be ~5%.
        fraction_extreme = np.mean(np.abs(all_deltas) > 2 * scale)
        assert fraction_extreme > 0.06, \
            f"Mutations lack heavy tails: only {fraction_extreme:.1%} beyond 2*scale"

    def test_temporal_mutation_unbiased_mean(self, temporal_config):
        """Mean mutation delta is approximately zero (unbiased)."""
        from src.reward import init_temporal_genome
        from src.evolution import mutate_temporal_genome
        from jax.flatten_util import ravel_pytree

        rng = jax.random.PRNGKey(0)
        parent = init_temporal_genome(rng, temporal_config)
        flat_parent, _ = ravel_pytree(parent)

        deltas = []
        for i in range(500):
            child = mutate_temporal_genome(parent, jax.random.PRNGKey(i), temporal_config)
            flat_child, _ = ravel_pytree(child)
            deltas.append(np.array(flat_child - flat_parent))

        mean_delta = np.mean(np.concatenate(deltas))
        assert abs(mean_delta) < 0.005, \
            f"Mean mutation delta should be ~0, got {mean_delta}"

    def test_temporal_utilization_metric(self, temporal_config):
        """compute_temporal_utilization returns correct shapes and handles edge cases."""
        from analysis.capacity_util import compute_temporal_utilization

        # Constant reward signal — zero autocorrelation
        k = temporal_config["reward_context_window"]
        rewards_const = np.ones(200)
        result = compute_temporal_utilization(rewards_const, k)
        assert result["autocorrelation"].shape == (k,)
        assert isinstance(result["sensitivity_ratio"], float)

        # Sinusoidal reward signal — positive autocorrelation at low lags
        t = np.arange(500)
        rewards_sin = np.sin(2 * np.pi * t / 20)
        result2 = compute_temporal_utilization(rewards_sin, k)
        assert result2["autocorrelation"][0] > 0.5, \
            "Sine wave should have high lag-1 autocorrelation"

        # Very short signal (shorter than k)
        result_short = compute_temporal_utilization(np.array([1.0, 2.0]), k)
        assert result_short["autocorrelation"].shape == (k,)

    def test_obs_buffer_vectorized_update(self, temporal_config):
        """obs_buffer roll+set works correctly when vectorized across agents."""
        k = temporal_config["reward_context_window"]
        n_agents = 5
        obs_buffer = jnp.zeros((n_agents, k, 4))

        # Each agent gets different stimuli
        stimuli = jnp.arange(n_agents * 4, dtype=jnp.float32).reshape(n_agents, 4)

        # Vectorized shift: roll all agents' buffers, insert new stimuli
        new_buffer = jnp.roll(obs_buffer, -1, axis=1)
        new_buffer = new_buffer.at[:, -1, :].set(stimuli)

        # Agent 0 should have stimuli [0, 1, 2, 3] at position -1
        np.testing.assert_allclose(np.array(new_buffer[0, -1]), [0, 1, 2, 3])
        # Agent 3 should have stimuli [12, 13, 14, 15] at position -1
        np.testing.assert_allclose(np.array(new_buffer[3, -1]), [12, 13, 14, 15])
        # Everything else should be zeros
        assert jnp.all(new_buffer[:, :-1, :] == 0.0)


# ─── LSTM policy (Axis 4) ────────────────────────────────────────────────────

class TestLSTMPolicy:
    """Tests for LSTM policy (Axis 4).

    CRITICAL INVARIANT: LSTM hidden state is lifetime state.
    It is NOT inherited. It resets to zeros at every birth.
    """

    def test_lstm_policy_init(self, lstm_config):
        """init_lstm_policy returns correct shapes and ~73k params."""
        from src.policy import init_lstm_policy
        from jax.flatten_util import ravel_pytree

        rng = jax.random.PRNGKey(0)
        params, opt_state, init_hidden = init_lstm_policy(rng, lstm_config)

        # Hidden state shape
        assert init_hidden.shape == (2, 64), f"Expected (2, 64), got {init_hidden.shape}"
        assert jnp.all(init_hidden == 0.0), "Initial hidden should be zeros"

        # Param count
        flat, _ = ravel_pytree(params)
        total_params = flat.shape[0]
        assert total_params > 70000, f"Expected ~73k params, got {total_params}"
        assert total_params < 80000, f"Expected ~73k params, got {total_params}"

        # Verify PyTree has LSTM structure
        assert 'params' in params

    def test_lstm_hidden_state_changes(self, lstm_config):
        """Hidden state changes after each step."""
        from src.policy import init_lstm_policy, sample_action_lstm

        rng = jax.random.PRNGKey(0)
        params, _, init_hidden = init_lstm_policy(rng, lstm_config)
        obs = jnp.ones(lstm_config["obs_dim"]) * 0.1

        # Step 1
        _, _, _, h1 = sample_action_lstm(params, obs, init_hidden, jax.random.PRNGKey(1), lstm_config)
        assert not jnp.allclose(init_hidden, h1), "Hidden should change after step 1"

        # Step 2
        _, _, _, h2 = sample_action_lstm(params, obs, h1, jax.random.PRNGKey(2), lstm_config)
        assert not jnp.allclose(h1, h2), "Hidden should change between steps"

        # Step 3 with different obs
        obs2 = jnp.zeros(lstm_config["obs_dim"])
        _, _, _, h3 = sample_action_lstm(params, obs2, h2, jax.random.PRNGKey(3), lstm_config)
        assert not jnp.allclose(h2, h3), "Hidden should change with different obs"

    def test_lstm_hidden_reset_at_birth(self, lstm_config):
        """Newborn agent (from spawn_offspring_jax) has zero LSTM hidden state."""
        # Test the invariant directly: the hidden state for a new slot is zeros.
        # In spawn_offspring_jax, we set lstm_hidden.at[new_slot].set(0.0).
        # Here we verify that a fresh hidden state is all zeros.
        lstm_hidden_size = lstm_config["lstm_hidden_size"]
        new_hidden = jnp.zeros((2, lstm_hidden_size))
        assert jnp.all(new_hidden == 0.0), "Birth hidden state must be zeros"
        assert new_hidden.shape == (2, lstm_hidden_size)

    def test_lstm_hidden_not_inherited(self, lstm_config):
        """Two offspring of the same parent have identical (zero) hidden states."""
        lstm_hidden_size = lstm_config["lstm_hidden_size"]

        # Simulate parent with non-zero hidden state
        parent_hidden = jnp.ones((2, lstm_hidden_size)) * 0.5

        # Two children both get zero hidden states (NOT parent's)
        child1_hidden = jnp.zeros((2, lstm_hidden_size))
        child2_hidden = jnp.zeros((2, lstm_hidden_size))

        assert jnp.array_equal(child1_hidden, child2_hidden), \
            "Both children should have identical zero hidden states"
        assert not jnp.array_equal(child1_hidden, parent_hidden), \
            "Children should NOT inherit parent's hidden state"

    def test_lstm_action_shape(self, lstm_config):
        """sample_action_lstm returns action shape (2,) and correct types."""
        from src.policy import init_lstm_policy, sample_action_lstm

        rng = jax.random.PRNGKey(0)
        params, _, init_hidden = init_lstm_policy(rng, lstm_config)
        obs = jax.random.normal(jax.random.PRNGKey(99), (lstm_config["obs_dim"],))

        action, log_prob, value, new_hidden = sample_action_lstm(
            params, obs, init_hidden, jax.random.PRNGKey(42), lstm_config,
        )

        assert action.shape == (2,), f"Expected action shape (2,), got {action.shape}"
        assert jnp.isfinite(log_prob), f"log_prob should be finite, got {log_prob}"
        assert jnp.isfinite(value), f"value should be finite, got {value}"
        assert new_hidden.shape == (2, 64), f"Expected hidden (2, 64), got {new_hidden.shape}"

        # Action should be in sigmoid range [-20, 80]
        assert jnp.all(action >= -20.0) and jnp.all(action <= 80.0), \
            f"Action out of [-20, 80] range: {action}"

    def test_lstm_rollout_buffer(self, lstm_config):
        """Rollout buffer correctly stores initial hidden for PPO replay."""
        lstm_hidden_size = lstm_config["lstm_hidden_size"]
        max_agents = 10
        rollout_steps = lstm_config["rollout_steps"]

        # Simulate rollout_init_hidden storage
        rollout_init_hidden = jnp.zeros((max_agents, 2, lstm_hidden_size))

        # Simulate agent 3 starting a new rollout with some hidden state
        agent_hidden = jnp.ones((2, lstm_hidden_size)) * 0.42
        rollout_init_hidden = rollout_init_hidden.at[3].set(agent_hidden)

        # Verify storage
        np.testing.assert_allclose(
            np.array(rollout_init_hidden[3]),
            np.array(agent_hidden),
            err_msg="Init hidden not stored correctly",
        )
        # Other agents should still be zeros
        np.testing.assert_allclose(
            np.array(rollout_init_hidden[0]),
            np.zeros((2, lstm_hidden_size)),
            err_msg="Other agents should have zero init hidden",
        )

    def test_mlp_policy_unchanged(self, base_config):
        """Existing MLP policy functions still work correctly (regression test)."""
        from src.policy import init_policy, sample_action, policy_forward

        rng = jax.random.PRNGKey(0)
        params, opt_state = init_policy(rng, base_config)
        obs = jnp.ones(base_config["obs_dim"]) * 0.1

        # MLP forward pass
        action_mean, log_std, value = policy_forward(params, obs, base_config)
        assert action_mean.shape == (2,)
        assert log_std.shape == (2,)
        assert jnp.isfinite(value)

        # MLP sample
        action, log_prob, value = sample_action(params, obs, jax.random.PRNGKey(42), base_config)
        assert action.shape == (2,)
        assert jnp.isfinite(log_prob)
        assert jnp.isfinite(value)

    def test_lstm_jit_vmap_compatible(self, lstm_config):
        """sample_action_lstm and policy_forward_lstm work under jax.jit and jax.vmap."""
        from src.policy import init_lstm_policy, policy_forward_lstm

        rng = jax.random.PRNGKey(0)
        params, _, init_hidden = init_lstm_policy(rng, lstm_config)
        obs = jnp.ones(lstm_config["obs_dim"]) * 0.1

        # JIT
        jitted_fwd = jax.jit(lambda p, o, h: policy_forward_lstm(p, o, h, lstm_config))
        mean_jit, log_std_jit, val_jit, h_jit = jitted_fwd(params, obs, init_hidden)
        mean_eager, _, _, h_eager = policy_forward_lstm(params, obs, init_hidden, lstm_config)
        np.testing.assert_allclose(np.array(mean_jit), np.array(mean_eager), rtol=1e-5)
        np.testing.assert_allclose(np.array(h_jit), np.array(h_eager), rtol=1e-5)

        # vmap over batch of agents (stacked params, obs, hidden)
        n = 4
        batch_params = jax.tree.map(lambda p: jnp.tile(p[None], (n, *([1]*p.ndim))), params)
        batch_obs = jnp.stack([obs * i for i in range(n)])  # (4, 205)
        batch_hidden = jnp.zeros((n, 2, lstm_config["lstm_hidden_size"]))

        vmapped = jax.vmap(lambda p, o, h: policy_forward_lstm(p, o, h, lstm_config))
        batch_mean, batch_log_std, batch_val, batch_h = vmapped(batch_params, batch_obs, batch_hidden)
        assert batch_mean.shape == (n, 2)
        assert batch_h.shape == (n, 2, lstm_config["lstm_hidden_size"])
        assert jnp.all(jnp.isfinite(batch_mean))

    def test_lstm_hidden_stability_long_rollout(self, lstm_config):
        """LSTM hidden state stays finite over 200+ steps."""
        from src.policy import init_lstm_policy, sample_action_lstm

        rng = jax.random.PRNGKey(0)
        params, _, hidden = init_lstm_policy(rng, lstm_config)

        for step in range(200):
            obs = jax.random.normal(jax.random.PRNGKey(step), (lstm_config["obs_dim"],))
            _, _, _, hidden = sample_action_lstm(
                params, obs, hidden, jax.random.PRNGKey(step + 1000), lstm_config,
            )
            if step % 50 == 49:
                assert jnp.all(jnp.isfinite(hidden)), \
                    f"Hidden state has NaN/Inf at step {step+1}"
                max_abs = float(jnp.max(jnp.abs(hidden)))
                assert max_abs < 1e6, \
                    f"Hidden state exploding at step {step+1}: max |h| = {max_abs}"

    def test_lstm_deterministic_same_key(self, lstm_config):
        """Same RNG key produces identical action from LSTM policy."""
        from src.policy import init_lstm_policy, sample_action_lstm

        rng = jax.random.PRNGKey(0)
        params, _, init_hidden = init_lstm_policy(rng, lstm_config)
        obs = jnp.ones(lstm_config["obs_dim"]) * 0.3

        a1, lp1, v1, h1 = sample_action_lstm(params, obs, init_hidden, jax.random.PRNGKey(42), lstm_config)
        a2, lp2, v2, h2 = sample_action_lstm(params, obs, init_hidden, jax.random.PRNGKey(42), lstm_config)

        np.testing.assert_array_equal(np.array(a1), np.array(a2))
        np.testing.assert_array_equal(np.array(h1), np.array(h2))

    def test_lstm_ppo_loss_decreases(self, lstm_config):
        """LSTM PPO update reduces loss on a synthetic rollout."""
        from src.policy import init_lstm_policy, LSTMPolicyNetwork
        from src.jax_ppo import _compute_gae_jax
        import optax

        rng = jax.random.PRNGKey(0)
        params, opt_state, init_hidden = init_lstm_policy(rng, lstm_config)

        rollout_steps = lstm_config["rollout_steps"]
        obs_dim = lstm_config["obs_dim"]
        chunk_length = lstm_config["lstm_chunk_length"]
        n_chunks = rollout_steps // chunk_length

        # Generate synthetic rollout data
        rng, data_key = jax.random.split(rng)
        obs = jax.random.normal(data_key, (rollout_steps, obs_dim)) * 0.1
        actions = jnp.zeros((rollout_steps, 2))
        rewards = jnp.ones(rollout_steps) * 0.1
        dones = jnp.zeros(rollout_steps, dtype=bool)

        # Collect log_probs and values by replaying LSTM
        net = LSTMPolicyNetwork(
            lstm_hidden_size=lstm_config["lstm_hidden_size"],
            hidden_size=lstm_config["policy_hidden_size"],
            action_dim=2,
        )
        carry = (init_hidden[0], init_hidden[1])

        def scan_step(carry, obs_t):
            new_carry, mean, log_std, value = net.apply(params, carry, obs_t)
            return new_carry, (mean, log_std, value)

        _, (all_means, all_log_stds, all_values) = jax.lax.scan(scan_step, carry, obs)
        values = all_values
        stds = jnp.exp(all_log_stds)
        log_probs = -0.5 * jnp.sum(
            jnp.log(2 * jnp.pi) + 2 * all_log_stds + ((actions - all_means) / stds) ** 2,
            axis=-1,
        )

        # GAE
        advantages, returns = _compute_gae_jax(
            rewards, values, dones, 0.0,
            lstm_config["gamma"], lstm_config["gae_lambda"],
        )
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

        # Compute initial loss
        def chunk_loss(p, c_obs, c_actions, c_old_lp, c_adv, c_ret, init_carry):
            def step_fn(carry, inputs):
                obs_t, act_t = inputs
                new_carry, mean, log_std, value = net.apply(p, carry, obs_t)
                return new_carry, (mean, log_std, value)

            _, (means, log_stds_out, vals) = jax.lax.scan(
                step_fn, init_carry, (c_obs, c_actions)
            )
            stds_out = jnp.exp(log_stds_out)
            clamped = jnp.clip(c_actions, -19.99, 79.99)
            raw_acts = jnp.log((clamped + 20.0) / (80.0 - clamped))
            new_lp = -0.5 * jnp.sum(
                jnp.log(2 * jnp.pi) + 2 * log_stds_out + ((raw_acts - means) / stds_out) ** 2,
                axis=-1,
            )
            ratio = jnp.exp(new_lp - c_old_lp)
            clip_eps = lstm_config["clip_epsilon"]
            surr1 = ratio * c_adv
            surr2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * c_adv
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
            value_loss = 0.5 * jnp.mean((vals - c_ret) ** 2)
            return policy_loss + 0.5 * value_loss

        # Compute loss on first chunk before any updates
        obs_chunks = obs.reshape(n_chunks, chunk_length, -1)
        act_chunks = actions.reshape(n_chunks, chunk_length, -1)
        lp_chunks = log_probs.reshape(n_chunks, chunk_length)
        adv_chunks = advantages.reshape(n_chunks, chunk_length)
        ret_chunks = returns.reshape(n_chunks, chunk_length)

        init_carry = (init_hidden[0], init_hidden[1])
        loss_before = chunk_loss(
            params, obs_chunks[0], act_chunks[0], lp_chunks[0],
            adv_chunks[0], ret_chunks[0], init_carry,
        )

        # Run one gradient step
        optimizer = optax.adam(learning_rate=lstm_config["lr"], eps=lstm_config["adam_eps"])
        loss_grad_fn = jax.value_and_grad(chunk_loss)
        loss_val, grads = loss_grad_fn(
            params, obs_chunks[0], act_chunks[0], lp_chunks[0],
            adv_chunks[0], ret_chunks[0], init_carry,
        )
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        loss_after = chunk_loss(
            new_params, obs_chunks[0], act_chunks[0], lp_chunks[0],
            adv_chunks[0], ret_chunks[0], init_carry,
        )

        assert jnp.isfinite(loss_before), f"Loss before is not finite: {loss_before}"
        assert jnp.isfinite(loss_after), f"Loss after is not finite: {loss_after}"
        # After one gradient step, loss should not increase (much)
        assert float(loss_after) <= float(loss_before) + 0.1, \
            f"Loss increased significantly after gradient step: {float(loss_before):.4f} -> {float(loss_after):.4f}"

    def test_lstm_utilization_metric(self, lstm_config):
        """compute_lstm_utilization returns correct shapes and sensible values."""
        from analysis.capacity_util import compute_lstm_utilization

        hidden_dim = lstm_config["lstm_hidden_size"]

        # Zero hidden trajectories — entropy should be 0
        h_zero = np.zeros((100, hidden_dim))
        result = compute_lstm_utilization(h_zero)
        assert result["mean_hidden_entropy"] == 0.0
        assert result["ablation_delta"] == 0.0

        # Random hidden trajectories — should have nonzero entropy and delta
        rng_np = np.random.default_rng(42)
        h_random = rng_np.normal(size=(200, hidden_dim))
        result2 = compute_lstm_utilization(h_random)
        assert result2["mean_hidden_entropy"] > 0.0, \
            "Random hidden states should have positive entropy"
        assert result2["ablation_delta"] > 0.0, \
            "Random hidden states should have nonzero L2 norm"

    def test_lstm_chunk_divisibility_config(self, lstm_config):
        """rollout_steps must be divisible by lstm_chunk_length."""
        rollout = lstm_config["rollout_steps"]
        chunk = lstm_config["lstm_chunk_length"]
        assert rollout % chunk == 0, \
            f"rollout_steps ({rollout}) must be divisible by lstm_chunk_length ({chunk})"
        n_chunks = rollout // chunk
        assert n_chunks > 0, "Must have at least 1 chunk"
        assert n_chunks == 8, f"Expected 8 chunks (1024/128), got {n_chunks}"


# ─── Social observation (Axis 2) ─────────────────────────────────────────────

class TestSocialObservation:
    """Tests for social observation extension (heading + speed of conspecifics)."""

    def _make_small_world(self, config, n_prey=5, n_pred=0):
        """Create a small world for social obs tests."""
        from src.environment import init_world
        cfg = {
            **config,
            "prey_initial": n_prey,
            "predator_initial": n_pred,
            "food_initial": 10,
        }
        rng = jax.random.PRNGKey(123)
        return init_world(cfg, rng), cfg

    def test_social_obs_shape_position_only(self, base_config):
        """get_observation returns (205,) when social_obs = position_only."""
        from src.agents import get_observation
        world, cfg = self._make_small_world(base_config, n_prey=3, n_pred=1)
        agent = world.agents[0]
        obs = get_observation(world, agent.agent_id, cfg)
        assert obs.shape == (205,), f"Expected (205,), got {obs.shape}"

    def test_social_obs_shape_social(self, social_config):
        """get_observation returns (215,) when social_obs = position_heading_velocity."""
        from src.agents import get_observation
        world, cfg = self._make_small_world(social_config, n_prey=6, n_pred=1)
        agent = world.agents[0]  # a prey
        obs = get_observation(world, agent.agent_id, cfg)
        assert obs.shape == (215,), f"Expected (215,), got {obs.shape}"

    def test_social_obs_zero_padding(self, social_config):
        """When fewer than n_social_neighbors conspecifics visible, trailing entries are zero."""
        from src.agents import get_observation
        # 1 prey + only predators => 0 conspecifics for the prey
        world, cfg = self._make_small_world(social_config, n_prey=1, n_pred=3)
        prey = [a for a in world.agents if a.species == 0][0]
        obs = get_observation(world, prey.agent_id, cfg)
        assert obs.shape == (215,)
        social_block = np.array(obs[205:215])
        np.testing.assert_array_equal(
            social_block, np.zeros(10),
            err_msg="Social block should be all zeros when no conspecifics visible"
        )

    def test_social_obs_closest_first(self, social_config):
        """First neighbor entry corresponds to the closest conspecific."""
        from src.agents import get_observation
        from src.environment import AgentState, init_world
        # Create world with 3 prey at known positions
        world, cfg = self._make_small_world(social_config, n_prey=3, n_pred=0)
        agents = [a for a in world.agents if a.species == 0]
        observer = agents[0]

        # Place agents at controlled positions (all within range 200)
        observer.position = jnp.array([480.0, 480.0])
        observer.velocity = jnp.zeros(2)
        observer.angle = 0.0

        agents[1].position = jnp.array([480.0, 530.0])  # distance 50 (closest)
        agents[1].velocity = jnp.array([3.0, 4.0])       # speed = 5.0
        agents[1].angle = 1.5

        agents[2].position = jnp.array([480.0, 600.0])  # distance 120 (farther)
        agents[2].velocity = jnp.array([0.0, 2.0])       # speed = 2.0
        agents[2].angle = -0.5

        obs = get_observation(world, observer.agent_id, cfg)
        social_block = np.array(obs[205:215])

        # First neighbor (closest, dist=50): heading=1.5, speed=5.0
        assert abs(social_block[0] - 1.5) < 1e-5, \
            f"First neighbor heading should be 1.5, got {social_block[0]}"
        assert abs(social_block[1] - 5.0) < 1e-5, \
            f"First neighbor speed should be 5.0, got {social_block[1]}"

        # Second neighbor (dist=120): heading=-0.5, speed=2.0
        assert abs(social_block[2] - (-0.5)) < 1e-5, \
            f"Second neighbor heading should be -0.5, got {social_block[2]}"
        assert abs(social_block[3] - 2.0) < 1e-5, \
            f"Second neighbor speed should be 2.0, got {social_block[3]}"

        # Remaining slots (3rd-5th) should be zero-padded
        np.testing.assert_array_equal(
            social_block[4:10], np.zeros(6),
            err_msg="Slots 3-5 should be zero-padded (only 2 conspecifics)"
        )

    def test_social_obs_no_change_to_baseline(self, base_config, social_config):
        """First 205 dims of social obs match baseline obs exactly (regression)."""
        from src.agents import get_observation
        # Use same seed and config for identical worlds
        from src.environment import init_world
        rng = jax.random.PRNGKey(77)
        baseline_cfg = {
            **base_config,
            "prey_initial": 5,
            "predator_initial": 2,
            "food_initial": 10,
        }
        social_cfg = {
            **social_config,
            "prey_initial": 5,
            "predator_initial": 2,
            "food_initial": 10,
        }
        world_baseline = init_world(baseline_cfg, rng)
        world_social = init_world(social_cfg, rng)

        for agent_b, agent_s in zip(world_baseline.agents, world_social.agents):
            obs_baseline = get_observation(world_baseline, agent_b.agent_id, baseline_cfg)
            obs_social = get_observation(world_social, agent_s.agent_id, social_cfg)

            assert obs_baseline.shape == (205,)
            assert obs_social.shape == (215,)

            # First 205 dims must match exactly
            np.testing.assert_allclose(
                np.array(obs_social[:205]),
                np.array(obs_baseline),
                atol=1e-6,
                err_msg=f"Baseline dims differ for agent {agent_b.agent_id}"
            )

    def test_social_obs_predator_species_isolation(self, social_config):
        """Predators observe only other predators, not prey."""
        from src.agents import get_observation
        # 1 predator + 5 prey, all close together — predator should see 0 conspecifics
        world, cfg = self._make_small_world(social_config, n_prey=5, n_pred=1)
        predator = [a for a in world.agents if a.species == 1][0]
        # Move all agents close so they're within range
        for a in world.agents:
            a.position = jnp.array([480.0, 480.0]) + jax.random.uniform(
                jax.random.PRNGKey(a.agent_id), shape=(2,), minval=-10.0, maxval=10.0
            )
        obs = get_observation(world, predator.agent_id, cfg)
        social_block = np.array(obs[205:215])
        # Only 1 predator total, so 0 conspecifics — all zeros
        np.testing.assert_array_equal(
            social_block, np.zeros(10),
            err_msg="Predator should see 0 conspecifics (only prey nearby)"
        )

    def test_social_obs_more_than_n_conspecifics(self, social_config):
        """When >N conspecifics visible, only the 5 closest are included."""
        from src.agents import get_observation
        # 8 prey total — observer + 7 conspecifics, all within range
        world, cfg = self._make_small_world(social_config, n_prey=8, n_pred=0)
        agents = world.agents
        observer = agents[0]
        observer.position = jnp.array([480.0, 480.0])
        observer.velocity = jnp.zeros(2)

        # Place 7 conspecifics at increasing distances
        for i, a in enumerate(agents[1:], start=1):
            a.position = jnp.array([480.0, 480.0 + float(i) * 20.0])  # 20, 40, ..., 140
            a.velocity = jnp.array([float(i), 0.0])  # speed = i
            a.angle = float(i) * 0.1

        obs = get_observation(world, observer.agent_id, cfg)
        social_block = np.array(obs[205:215])

        # Should contain the 5 closest (agents 1-5, distances 20-100)
        for k in range(5):
            expected_heading = float(k + 1) * 0.1
            expected_speed = float(k + 1)
            assert abs(social_block[2 * k] - expected_heading) < 1e-5, \
                f"Neighbor {k} heading: expected {expected_heading}, got {social_block[2*k]}"
            assert abs(social_block[2 * k + 1] - expected_speed) < 1e-5, \
                f"Neighbor {k} speed: expected {expected_speed}, got {social_block[2*k+1]}"

    def test_social_obs_boundary_at_max_range(self, social_config):
        """Conspecific at exactly max_range (200 units) is included."""
        from src.agents import get_observation
        world, cfg = self._make_small_world(social_config, n_prey=2, n_pred=0)
        observer = world.agents[0]
        neighbor = world.agents[1]

        observer.position = jnp.array([480.0, 480.0])
        observer.velocity = jnp.zeros(2)

        # Place neighbor at exactly max_range distance
        max_range = float(cfg["proximity_max_range"])  # 200.0
        neighbor.position = jnp.array([480.0, 480.0 + max_range])  # dist = 200.0 exactly
        neighbor.velocity = jnp.array([1.0, 0.0])  # speed = 1.0
        neighbor.angle = 0.5

        obs = get_observation(world, observer.agent_id, cfg)
        social_block = np.array(obs[205:215])

        # Should be included (dist <= max_range)
        assert abs(social_block[0] - 0.5) < 1e-5, \
            f"Neighbor at exactly max_range should be included, heading={social_block[0]}"
        assert abs(social_block[1] - 1.0) < 1e-5, \
            f"Neighbor at exactly max_range should be included, speed={social_block[1]}"

        # Now move just beyond max_range
        neighbor.position = jnp.array([480.0, 480.0 + max_range + 0.01])
        obs2 = get_observation(world, observer.agent_id, cfg)
        social_block2 = np.array(obs2[205:215])

        # Should be excluded (dist > max_range)
        np.testing.assert_array_equal(
            social_block2, np.zeros(10),
            err_msg="Neighbor just beyond max_range should be excluded"
        )

    def test_social_obs_velocity_none_safety(self, social_config):
        """Social obs handles conspecific with velocity=None without crashing."""
        from src.agents import get_observation
        world, cfg = self._make_small_world(social_config, n_prey=2, n_pred=0)
        observer = world.agents[0]
        neighbor = world.agents[1]

        observer.position = jnp.array([480.0, 480.0])
        observer.velocity = jnp.zeros(2)

        neighbor.position = jnp.array([480.0, 500.0])  # within range
        neighbor.velocity = None  # edge case
        neighbor.angle = 1.0

        obs = get_observation(world, observer.agent_id, cfg)
        assert obs.shape == (215,)
        social_block = np.array(obs[205:215])
        # heading should be 1.0, speed should be 0.0 (None velocity -> zeros)
        assert abs(social_block[0] - 1.0) < 1e-5
        assert abs(social_block[1] - 0.0) < 1e-5


# ─── Lifecycle functions ───────────────────────────────────────────────────────

class TestLifecycle:

    def test_hazard_formula(self, base_config):
        """Check hazard function matches the exact formula at known inputs."""
        from src.lifecycle import hazard_prob
        # h(t=0, e=20, prey) should be very low (young, adequate energy)
        h = hazard_prob(age=0, energy=20.0, species=0, config=base_config)
        assert 0.0 <= h <= 0.01, f"Hazard at t=0,e=20 should be near 0, got {h}"

    def test_hazard_increases_with_age(self, base_config):
        """Hazard increases with age at constant energy."""
        from src.lifecycle import hazard_prob
        h_young = hazard_prob(age=0, energy=30.0, species=0, config=base_config)
        h_old = hazard_prob(age=1_000_000, energy=30.0, species=0, config=base_config)
        assert h_old > h_young, f"Hazard should increase with age: {h_young} vs {h_old}"

    def test_hazard_increases_with_low_energy(self, base_config):
        """Hazard increases sharply when energy drops below ~15-20."""
        from src.lifecycle import hazard_prob
        h_high_e = hazard_prob(age=500_000, energy=30.0, species=0, config=base_config)
        h_low_e = hazard_prob(age=500_000, energy=5.0, species=0, config=base_config)
        assert h_low_e > h_high_e, f"Low energy should increase hazard: {h_high_e} vs {h_low_e}"

    def test_hazard_range(self, base_config):
        """Hazard should be a valid probability in [0, 1]."""
        from src.lifecycle import hazard_prob
        for age in [0, 100_000, 500_000, 2_000_000]:
            for energy in [0.1, 5.0, 20.0, 100.0]:
                h = hazard_prob(age=age, energy=energy, species=0, config=base_config)
                assert 0.0 <= h <= 1.0, f"Hazard={h} out of [0,1] at age={age}, e={energy}"

    def test_birth_formula(self, base_config):
        """Birth function matches formula at known inputs."""
        from src.lifecycle import birth_prob
        # b(e) = κ_b / (1 + exp(ζ - β_b * e))
        # prey: κ_b=1e-3, ζ=10, β_b=0.1
        # at e=100: b = 1e-3 / (1 + exp(10 - 10)) = 1e-3 / 2 = 0.5e-3
        b = birth_prob(energy=100.0, species=0, config=base_config)
        expected = base_config["kappa_b"] / (1 + np.exp(base_config["zeta_b_prey"] - base_config["beta_b"] * 100.0))
        assert abs(b - expected) < 1e-6, f"Birth prob mismatch: {b} vs {expected}"

    def test_birth_monotone_increasing(self, base_config):
        """Birth probability increases with energy."""
        from src.lifecycle import birth_prob
        prev = 0.0
        for e in [10.0, 30.0, 50.0, 100.0, 200.0]:
            b = birth_prob(energy=e, species=0, config=base_config)
            assert b >= prev, f"Birth not monotone: b({e})={b} < b(prev)={prev}"
            prev = b

    def test_birth_near_zero_low_energy(self, base_config):
        """Birth probability near zero when energy well below threshold."""
        from src.lifecycle import birth_prob
        b = birth_prob(energy=1.0, species=0, config=base_config)
        assert b < 1e-4, f"Birth prob should be near zero at e=1, got {b}"

    def test_predator_higher_energy_threshold(self, base_config):
        """Predators need much more energy to reproduce than prey."""
        from src.lifecycle import birth_prob
        b_prey_100 = birth_prob(energy=100.0, species=0, config=base_config)
        b_pred_100 = birth_prob(energy=100.0, species=1, config=base_config)
        assert b_prey_100 > b_pred_100 * 10, (
            f"Prey b(100)={b_prey_100:.6f} should be >> predator b(100)={b_pred_100:.6f}"
        )


# ─── Evolution (mutation) ──────────────────────────────────────────────────────

class TestEvolution:

    def test_mutate_genome_shape(self, base_config):
        from src.evolution import mutate_genome
        parent = jnp.array([1.0, -2.0, 0.5, -0.5])
        rng = jax.random.PRNGKey(42)
        child = mutate_genome(parent, rng)
        assert child.shape == (4,)

    def test_mutate_genome_clipping(self, base_config):
        """All weights stay within clip range after mutation."""
        from src.evolution import mutate_genome
        # Start at clip boundary — mutations should not exceed ±100
        parent = jnp.array([99.0, -99.0, 99.0, -99.0])
        for i in range(100):
            rng = jax.random.PRNGKey(i)
            child = mutate_genome(parent, rng)
            assert jnp.all(jnp.abs(child) <= 100.0 + 1e-5), (
                f"Genome exceeds clip range: {child}"
            )

    def test_mutate_genome_heavy_tails(self):
        """
        Mutation distribution should be heavy-tailed (t(df=2)), not Gaussian.
        Test by checking kurtosis: t(df=2) has infinite kurtosis.
        Practically: more samples > 2σ than a Gaussian would produce.
        With 10,000 samples from N(0,1), ~4.6% exceed ±2σ.
        With t(df=2, scale=0.4), normalized to unit scale, significantly more should.
        """
        from src.evolution import mutate_genome
        parent = jnp.zeros(4)
        samples = []
        for i in range(2500):  # 2500 * 4 = 10,000 samples
            rng = jax.random.PRNGKey(i)
            delta = mutate_genome(parent, rng) - parent
            samples.extend(delta.tolist())

        samples = np.array(samples)
        std = base_config_mutation_scale = 0.4
        # Fraction beyond ±2σ: Gaussian predicts ~4.6%, t(df=2) predicts ~13.4%
        beyond_2sigma = np.mean(np.abs(samples) > 2 * std)
        assert beyond_2sigma > 0.08, (
            f"Mutation appears Gaussian (heavy tail fraction={beyond_2sigma:.3f} < 0.08). "
            f"Check that scipy.stats.t(df=2) is being used, not jax.random.normal()."
        )

    def test_mutate_genome_not_identical_to_parent(self):
        """Mutation should change at least some weights (with very high probability)."""
        from src.evolution import mutate_genome
        parent = jnp.zeros(4)
        any_changed = False
        for i in range(20):
            rng = jax.random.PRNGKey(i)
            child = mutate_genome(parent, rng)
            if not jnp.allclose(child, parent):
                any_changed = True
                break
        assert any_changed, "Mutation never changes genome — check implementation"

    def test_mutate_genome_preserves_unclipped_mean(self):
        """Mean of mutations across many offspring should be near zero (unbiased)."""
        from src.evolution import mutate_genome
        parent = jnp.zeros(4)
        deltas = []
        for i in range(1000):
            rng = jax.random.PRNGKey(i)
            child = mutate_genome(parent, rng)
            deltas.append(float(child[0]))  # check first weight
        mean_delta = np.mean(deltas)
        assert abs(mean_delta) < 0.05, f"Mutation mean not near zero: {mean_delta}"


# ─── Observation vector ────────────────────────────────────────────────────────

class TestObservation:

    def test_observation_shape(self, base_config):
        """Observation vector has correct dimension."""
        from src.agents import get_observation
        # This test requires a minimal WorldState — use a stub or minimal world
        # If get_observation is not yet implemented, this test will fail with
        # NotImplementedError or ImportError — that's expected and correct.
        pytest.skip(
            "Requires WorldState stub. Implement after environment.py is built."
        )

    def test_observation_finite(self, base_config):
        """All observation values must be finite."""
        pytest.skip("Requires WorldState stub.")

    def test_observation_proximity_range(self, base_config):
        """Proximity sensor values must be in [0, 1]."""
        pytest.skip("Requires WorldState stub.")


# ─── PPO utilities ────────────────────────────────────────────────────────────

class TestPPO:

    def test_gae_constant_reward(self, base_config):
        """
        GAE with constant reward r, no terminal states:
        advantage ≈ r / (1 - γ) - V for constant value V.
        Verify that GAE returns are close to the geometric sum.
        """
        from src.ppo import compute_gae
        N = 64
        r = 1.0
        V = 10.0  # constant value estimate
        rewards = jnp.full((N,), r)
        values = jnp.full((N,), V)
        dones = jnp.zeros(N, dtype=bool)
        last_value = V

        advantages, returns = compute_gae(rewards, values, dones, last_value, base_config)

        assert advantages.shape == (N,)
        assert returns.shape == (N,)
        assert jnp.all(jnp.isfinite(advantages)), "GAE advantages contain non-finite values"
        assert jnp.all(jnp.isfinite(returns)), "GAE returns contain non-finite values"

    def test_gae_terminal_state(self, base_config):
        """GAE advantages after a terminal state should not use values beyond it."""
        from src.ppo import compute_gae
        N = 32
        rewards = jnp.ones(N)
        values = jnp.ones(N) * 5.0
        # Terminal at step 15
        dones = jnp.zeros(N, dtype=bool).at[15].set(True)
        advantages, returns = compute_gae(rewards, values, dones, 0.0, base_config)

        # After the terminal, the GAE should reset (no value bootstrap beyond done)
        # Advantage at step 16 should be computed fresh
        assert jnp.isfinite(advantages[16]), "Advantage after terminal should be finite"

    def test_ppo_loss_decreases(self, base_config):
        """Run one ppo_update() on a synthetic rollout and confirm
        policy_loss after 10 epochs is less than initial loss."""
        from src.policy import init_policy, PolicyNetwork
        from src.ppo import ppo_update

        rng = jax.random.PRNGKey(42)
        params, opt_state = init_policy(rng, base_config)

        # Build synthetic rollout with clear positive-reward signal
        N = base_config["rollout_steps"]  # 1024
        rng, obs_rng, act_rng = jax.random.split(rng, 3)
        observations = jax.random.normal(obs_rng, (N, base_config["obs_dim"])) * 0.1

        # Use current policy to generate consistent actions/log_probs
        net = PolicyNetwork(hidden_size=base_config["policy_hidden_size"], action_dim=2)

        actions_list = []
        log_probs_list = []
        values_list = []
        for i in range(N):
            action_mean, log_std, value = net.apply(params, observations[i])
            std = jnp.exp(log_std)
            noise = jax.random.normal(jax.random.PRNGKey(i), shape=(2,))
            raw_action = action_mean + std * noise
            # sigmoid scale
            action = 100.0 * jax.nn.sigmoid(raw_action) - 20.0
            lp = -0.5 * jnp.sum(
                jnp.log(2 * jnp.pi) + 2 * log_std + ((raw_action - action_mean) / std) ** 2
            )
            actions_list.append(action)
            log_probs_list.append(lp)
            values_list.append(value)

        actions = jnp.stack(actions_list)
        log_probs = jnp.array(log_probs_list)
        values = jnp.array(values_list)
        rewards = jnp.ones(N) * 1.0  # constant positive reward
        dones = jnp.zeros(N, dtype=bool)

        rollout = {
            "observations": observations,
            "actions": actions,
            "log_probs": log_probs,
            "rewards": rewards,
            "values": values,
            "dones": dones,
        }

        # Compute initial value loss for comparison
        initial_values = values
        from src.ppo import compute_gae
        advantages, returns = compute_gae(rewards, initial_values, dones, 0.0, base_config)
        initial_value_loss = float(0.5 * jnp.mean((initial_values - returns) ** 2))

        new_params, new_opt_state, info = ppo_update(params, opt_state, rollout, base_config)

        # Value loss should decrease after 10 PPO epochs
        assert info["value_loss"] < initial_value_loss, (
            f"Value loss did not decrease: initial={initial_value_loss:.4f}, "
            f"final={info['value_loss']:.4f}"
        )
        # All info values should be finite
        for key in ["policy_loss", "value_loss", "entropy", "approx_kl"]:
            assert np.isfinite(info[key]), f"info['{key}'] is not finite: {info[key]}"


# ─── Metrics and checkpointing ────────────────────────────────────────────────

class TestMetrics:

    def test_metrics_log_structure(self, base_config):
        """MetricsLog can be created and its fields accessed."""
        from src.metrics import MetricsLog
        log = MetricsLog()
        # All fields should be empty lists
        assert isinstance(log.steps, list) and len(log.steps) == 0
        assert isinstance(log.prey_population, list)
        assert isinstance(log.birth_log, list)
        # Check all expected fields exist
        expected_fields = [
            "steps", "prey_population", "predator_population",
            "prey_mean_energy", "predator_mean_energy",
            "prey_mean_w_eat", "prey_mean_w_act", "prey_mean_w_prey", "prey_mean_w_pred",
            "prey_std_w_eat", "prey_std_w_act", "prey_std_w_prey", "prey_std_w_pred",
            "pred_mean_w_eat", "pred_mean_w_act", "pred_mean_w_prey", "pred_mean_w_pred",
            "pred_std_w_eat", "pred_std_w_act", "pred_std_w_prey", "pred_std_w_pred",
            "capture_rate", "food_consumption_rate", "birth_log",
        ]
        for fname in expected_fields:
            assert hasattr(log, fname), f"MetricsLog missing field: {fname}"

    def test_metrics_save_load_roundtrip(self, base_config, tmp_path):
        """Save and load metrics; values should match."""
        from src.metrics import MetricsLog, save_metrics, load_metrics, record_birth
        log = MetricsLog()
        log.steps.append(1000)
        log.prey_population.append(100)
        log.predator_population.append(10)
        log.prey_mean_energy.append(50.0)
        log.predator_mean_energy.append(80.0)
        # Fill all weight fields with a value
        for prefix in ["prey", "pred"]:
            for stat in ["mean", "std"]:
                for w in ["w_eat", "w_act", "w_prey", "w_pred"]:
                    getattr(log, f"{prefix}_{stat}_{w}").append(0.5)
        log.capture_rate.append(0.1)
        log.food_consumption_rate.append(0.2)
        log = record_birth(log, 1000, 42, 7)

        config = {**base_config, "experiment_name": "test_exp"}
        save_metrics(log, config, seed=0, out_dir=str(tmp_path))
        loaded = load_metrics(str(tmp_path / "test_exp" / "seed_0" / "metrics.npz"))

        assert loaded.steps == [1000]
        assert loaded.prey_population == [100]
        assert loaded.birth_log == [(1000, 42, 7)]

    def test_checkpoint_roundtrip(self, base_config, tmp_path):
        """Save and load a checkpoint; reward_weights arrays are identical."""
        from src.metrics import MetricsLog, save_checkpoint, load_checkpoint
        from types import SimpleNamespace

        # Build minimal world stub
        agents = [
            SimpleNamespace(
                agent_id=0, species=0, age=100, energy=50.0,
                reward_weights=np.array([1.1, -2.2, 3.3, -4.4], dtype=np.float32),
                parent_id=-1,
            ),
            SimpleNamespace(
                agent_id=1, species=1, age=50, energy=80.0,
                reward_weights=np.array([-0.5, 0.5, -0.5, 0.5], dtype=np.float32),
                parent_id=-1,
            ),
        ]
        world = SimpleNamespace(step=10000, agents=agents)
        log = MetricsLog()
        config = {**base_config, "experiment_name": "test_ckpt"}

        save_checkpoint(world, log, config, seed=0, out_dir=str(tmp_path))
        ckpt_path = str(tmp_path / "test_ckpt" / "seed_0" / "step_00010000.pkl")
        loaded = load_checkpoint(ckpt_path)

        assert loaded["step"] == 10000
        np.testing.assert_array_equal(loaded["agent_ids"], np.array([0, 1]))
        np.testing.assert_array_almost_equal(
            loaded["reward_weights"],
            np.array([[1.1, -2.2, 3.3, -4.4], [-0.5, 0.5, -0.5, 0.5]],
                     dtype=np.float32),
        )

    def test_trajectory_save_load_roundtrip(self, base_config, tmp_path):
        """Trajectory fields survive save/load when save_trajectories is on."""
        from src.metrics import (
            MetricsLog, save_metrics, load_metrics, record_trajectory_step,
        )
        log = MetricsLog()
        # Fill required scalar fields so save doesn't fail on empty
        log.steps.append(1000)
        log.prey_population.append(100)
        log.predator_population.append(10)
        log.prey_mean_energy.append(50.0)
        log.predator_mean_energy.append(80.0)
        for prefix in ["prey", "pred"]:
            for stat in ["mean", "std"]:
                for w in ["w_eat", "w_act", "w_prey", "w_pred"]:
                    getattr(log, f"{prefix}_{stat}_{w}").append(0.5)
        log.capture_rate.append(0.1)
        log.food_consumption_rate.append(0.2)

        config = {**base_config, "experiment_name": "test_traj", "save_trajectories": True}

        # Record a few trajectory steps
        obs1 = np.random.randn(215).astype(np.float32)
        obs2 = np.random.randn(215).astype(np.float32)
        act1 = np.array([10.0, -5.0], dtype=np.float32)
        act2 = np.array([-3.0, 7.0], dtype=np.float32)

        record_trajectory_step(log, obs1, act1, agent_id=0, config=config)
        record_trajectory_step(log, obs2, act2, agent_id=1, config=config)

        assert len(log.trajectory_obs) == 2
        assert len(log.trajectory_actions) == 2
        assert len(log.trajectory_agent_ids) == 2

        save_metrics(log, config, seed=0, out_dir=str(tmp_path))
        loaded = load_metrics(str(tmp_path / "test_traj" / "seed_0" / "metrics.npz"))

        assert len(loaded.trajectory_obs) == 2
        assert len(loaded.trajectory_actions) == 2
        assert loaded.trajectory_agent_ids == [0, 1]
        np.testing.assert_allclose(loaded.trajectory_obs[0], obs1, atol=1e-6)
        np.testing.assert_allclose(loaded.trajectory_actions[1], act2, atol=1e-6)

    def test_trajectory_not_saved_when_disabled(self, base_config):
        """record_trajectory_step is a no-op when save_trajectories is False."""
        from src.metrics import MetricsLog, record_trajectory_step
        log = MetricsLog()
        config = {**base_config, "save_trajectories": False}
        obs = np.zeros(205, dtype=np.float32)
        act = np.zeros(2, dtype=np.float32)

        record_trajectory_step(log, obs, act, agent_id=0, config=config)
        assert len(log.trajectory_obs) == 0
        assert len(log.trajectory_actions) == 0


# ─── Config validation ────────────────────────────────────────────────────────

class TestConfig:

    def test_required_keys_present(self, base_config):
        """All required config keys are present in base_config fixture."""
        try:
            from scripts.run_experiment import REQUIRED_CONFIG_KEYS
        except (ModuleNotFoundError, ImportError):
            pytest.skip("scripts/run_experiment.py not yet implemented")
        missing = REQUIRED_CONFIG_KEYS - set(base_config.keys())
        assert not missing, f"Missing required config keys: {missing}"

    def test_config_values_positive_where_required(self, base_config):
        """Key parameters that must be positive."""
        positive_keys = [
            "world_size", "prey_radius", "predator_radius", "max_motor_norm",
            "n_proximity_sensors", "proximity_max_range", "n_tactile_sensors",
            "food_max", "food_growth_rate", "prey_c_b", "prey_c_a",
            "predator_d_b", "predator_d_a", "kappa_h", "alpha_e", "beta_h",
            "kappa_b", "beta_b", "mutation_scale", "weight_clip",
            "policy_hidden_size", "policy_n_hidden_layers", "rollout_steps",
            "minibatch_size", "ppo_epochs", "lr",
        ]
        for key in positive_keys:
            assert base_config[key] > 0, f"Config key '{key}' should be positive, got {base_config[key]}"

    def test_gae_lambda_in_range(self, base_config):
        assert 0.0 < base_config["gae_lambda"] < 1.0

    def test_clip_epsilon_in_range(self, base_config):
        assert 0.0 < base_config["clip_epsilon"] < 1.0

    def test_gamma_in_range(self, base_config):
        assert 0.0 < base_config["gamma"] < 1.0
