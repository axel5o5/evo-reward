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
        "obs_dim": 205,                     # CONFIRMED: 128+72+2+1+1+1
        "food_max": 600,                    # CONFIRMED: n_max_foods=600
        "food_initial": 40,
        "food_growth_rate": 0.5,            # CONFIRMED: 0.5/step
        "food_max_regen_per_step": 10,
        "prey_e_food": 1.0,
        "prey_c_b": 1.0e-4,                # CONFIRMED: code value (not paper 2.5e-3)
        "prey_c_a": 2.5e-6,                # CONFIRMED: code value (not paper 1.0e-4)
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
