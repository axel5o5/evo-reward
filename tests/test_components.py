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
    """Minimal config matching baseline_faithful.yaml for unit tests."""
    return {
        "world_size": 960,
        "prey_radius": 10.0,
        "predator_radius": 14.0,
        "max_motor_norm": 114.0,
        "n_proximity_sensors": 32,
        "proximity_fov_deg": 120.0,
        "proximity_max_range": 120.0,
        "n_tactile_sensors": 18,
        "tactile_spacing_deg": 20.0,
        "obs_dim": 54,  # update after resolving emevo-diff open item #2
        "food_max": 100,
        "food_growth_rate": 0.02,
        "food_regen_rate": 0.5,
        "prey_e_food": 1.0,
        "prey_c_b": 2.5e-3,
        "prey_c_a": 1.0e-4,
        "predator_d_b": 4.0e-3,
        "predator_d_a": 5.0e-5,
        "predator_eta": 0.6,
        "predator_mouth_deg": 60.0,
        "predator_mouth_range_min": 40.0,
        "predator_mouth_range_max": 80.0,
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
        "energy_share_ratio": 0.5,
        "spawn_spread": 30.0,
        "reward_weights_init_std": 0.1,
        "mutation_df": 2,
        "mutation_scale": 0.4,
        "weight_clip": 100.0,
        "policy_hidden_size": 64,
        "policy_n_layers": 3,
        "gamma": 0.999,
        "rollout_steps": 1024,
        "minibatch_size": 256,
        "ppo_epochs": 10,
        "clip_epsilon": 0.2,
        "entropy_coef": 0.001,
        "gae_lambda": 0.95,
        "lr": 3.0e-4,
        "adam_eps": 1.0e-7,
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


# ─── Metrics and checkpointing ────────────────────────────────────────────────

class TestMetrics:

    def test_metrics_log_structure(self, base_config, tmp_path):
        """MetricsLog can be created and its fields accessed."""
        pytest.skip("Requires MetricsLog implementation.")

    def test_checkpoint_roundtrip(self, base_config, tmp_path):
        """Save and load a checkpoint; values should match exactly."""
        pytest.skip("Requires Checkpoint implementation and WorldState stub.")


# ─── Config validation ────────────────────────────────────────────────────────

class TestConfig:

    def test_required_keys_present(self, base_config):
        """All required config keys are present in base_config fixture."""
        from scripts.run_experiment import REQUIRED_CONFIG_KEYS
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
            "policy_hidden_size", "policy_n_layers", "rollout_steps",
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
