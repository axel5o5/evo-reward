"""
test_vectorized_obs.py
----------------------
Correctness tests for vectorized observation computation (src/observations.py)
compared against the original per-agent reference implementation.
"""

import math
import sys
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import init_world, extract_obs_state, step_physics
from src.agents import get_observation
from src.observations import compute_all_observations


@pytest.fixture
def config():
    with open("configs/baseline_faithful.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg["prey_initial"] = 10
    cfg["predator_initial"] = 3
    cfg["food_initial"] = 20
    cfg["seed"] = 42
    cfg["experiment_name"] = "test_vec_obs"
    cfg["total_steps"] = 100
    cfg["checkpoint_interval_steps"] = 100
    cfg["log_interval_steps"] = 100
    return cfg


@pytest.fixture
def world_and_config(config):
    rng = jax.random.PRNGKey(42)
    world = init_world(config, rng)
    # Do one physics step to give agents non-trivial positions
    actions = {a.agent_id: jnp.array([10.0, -5.0]) for a in world.agents}
    world = step_physics(world, actions, config)
    return world, config


class TestVectorizedVsReference:
    """Gold standard: vectorized output matches per-agent reference."""

    def test_observations_match(self, world_and_config):
        """Compare vectorized obs against per-agent loop for all agents."""
        world, config = world_and_config

        # Reference: per-agent loop
        ref_obs = []
        for agent in world.agents:
            obs = get_observation(world, agent.agent_id, config)
            ref_obs.append(np.array(obs))
        ref_obs = np.stack(ref_obs)

        # Vectorized: single call
        obs_state = extract_obs_state(world, config)
        vec_obs_all = compute_all_observations(obs_state, config)

        # Gather active agents in same order
        slots = [world.physics["agent_id_to_slot"][a.agent_id] for a in world.agents]
        vec_obs = np.array(vec_obs_all[jnp.array(slots)])

        np.testing.assert_allclose(vec_obs, ref_obs, atol=1e-3, rtol=1e-3)

    def test_proximity_sensors_match(self, world_and_config):
        """Proximity sensors (first 128 dims) match between methods."""
        world, config = world_and_config

        ref_obs = []
        for agent in world.agents:
            obs = get_observation(world, agent.agent_id, config)
            ref_obs.append(np.array(obs[:128]))
        ref_prox = np.stack(ref_obs)

        obs_state = extract_obs_state(world, config)
        vec_obs_all = compute_all_observations(obs_state, config)
        slots = [world.physics["agent_id_to_slot"][a.agent_id] for a in world.agents]
        vec_prox = np.array(vec_obs_all[jnp.array(slots), :128])

        np.testing.assert_allclose(vec_prox, ref_prox, atol=1e-3, rtol=1e-3)

    def test_tactile_sensors_match(self, world_and_config):
        """Tactile sensors (dims 128-199) match between methods."""
        world, config = world_and_config

        ref_obs = []
        for agent in world.agents:
            obs = get_observation(world, agent.agent_id, config)
            ref_obs.append(np.array(obs[128:200]))
        ref_tact = np.stack(ref_obs)

        obs_state = extract_obs_state(world, config)
        vec_obs_all = compute_all_observations(obs_state, config)
        slots = [world.physics["agent_id_to_slot"][a.agent_id] for a in world.agents]
        vec_tact = np.array(vec_obs_all[jnp.array(slots), 128:200])

        np.testing.assert_allclose(vec_tact, ref_tact, atol=1e-3, rtol=1e-3)

    def test_scalar_obs_match(self, world_and_config):
        """Velocity, angle, angular velocity, energy (dims 200-204) match."""
        world, config = world_and_config

        ref_obs = []
        for agent in world.agents:
            obs = get_observation(world, agent.agent_id, config)
            ref_obs.append(np.array(obs[200:205]))
        ref_scalars = np.stack(ref_obs)

        obs_state = extract_obs_state(world, config)
        vec_obs_all = compute_all_observations(obs_state, config)
        slots = [world.physics["agent_id_to_slot"][a.agent_id] for a in world.agents]
        vec_scalars = np.array(vec_obs_all[jnp.array(slots), 200:205])

        np.testing.assert_allclose(vec_scalars, ref_scalars, atol=1e-5)


class TestJITCompilation:
    """Verify the vectorized function compiles and runs under JIT."""

    def test_jit_compiles(self, world_and_config):
        """compute_all_observations should JIT-compile without error."""
        world, config = world_and_config
        obs_state = extract_obs_state(world, config)
        # First call triggers compilation
        result = compute_all_observations(obs_state, config)
        assert result.shape == (world.physics["max_agents"], config["obs_dim"])
        assert not jnp.any(jnp.isnan(result))

    def test_second_call_fast(self, world_and_config):
        """Second call should reuse compiled function (no recompilation)."""
        import time
        world, config = world_and_config
        obs_state = extract_obs_state(world, config)

        # Warmup
        _ = compute_all_observations(obs_state, config)

        # Timed call
        t0 = time.time()
        for _ in range(10):
            _ = compute_all_observations(obs_state, config)
        t1 = time.time()
        ms_per_call = (t1 - t0) / 10 * 1000
        # Should be < 500ms per call on CPU (vs 330ms for old loop at 150 agents)
        assert ms_per_call < 500, f"Vectorized obs too slow: {ms_per_call:.1f}ms"


class TestEdgeCases:
    """Edge cases for sensor computation."""

    def test_inactive_agents_zeroed(self, world_and_config):
        """Observations for inactive agent slots should be all zeros."""
        world, config = world_and_config
        obs_state = extract_obs_state(world, config)
        result = compute_all_observations(obs_state, config)

        # Find inactive slots
        is_active = np.array(obs_state["is_active"])
        inactive_mask = ~is_active
        if inactive_mask.any():
            inactive_obs = np.array(result[inactive_mask])
            assert np.all(inactive_obs == 0.0)

    def test_obs_dim_correct(self, world_and_config):
        """Output has correct obs_dim."""
        world, config = world_and_config
        obs_state = extract_obs_state(world, config)
        result = compute_all_observations(obs_state, config)
        assert result.shape[1] == config["obs_dim"]

    def test_no_nan_in_output(self, world_and_config):
        """No NaN values in observation output."""
        world, config = world_and_config
        obs_state = extract_obs_state(world, config)
        result = compute_all_observations(obs_state, config)
        assert not jnp.any(jnp.isnan(result))


class TestSocialObsVectorized:
    """Verify vectorized social observation matches Python per-agent reference."""

    @pytest.fixture
    def social_config(self, config):
        """Extend baseline config with social observation settings."""
        return {
            **config,
            "social_obs": "position_heading_velocity",
            "obs_dim": 215,
            "n_social_neighbors": 5,
        }

    @pytest.fixture
    def social_world(self, social_config):
        rng = jax.random.PRNGKey(42)
        world = init_world(social_config, rng)
        # Do one physics step to give agents non-trivial velocities
        actions = {a.agent_id: jnp.array([10.0, -5.0]) for a in world.agents}
        world = step_physics(world, actions, social_config)
        return world, social_config

    def test_social_obs_vectorized_matches_reference(self, social_world):
        """Vectorized social obs (dims 205-214) matches per-agent Python loop."""
        world, config = social_world

        # Reference: per-agent Python loop
        ref_obs = []
        for agent in world.agents:
            obs = get_observation(world, agent.agent_id, config)
            ref_obs.append(np.array(obs))
        ref_obs = np.stack(ref_obs)

        # Vectorized: single call
        obs_state = extract_obs_state(world, config)
        vec_obs_all = compute_all_observations(obs_state, config)

        # Gather active agents in same order
        slots = [world.physics["agent_id_to_slot"][a.agent_id] for a in world.agents]
        vec_obs = np.array(vec_obs_all[jnp.array(slots)])

        assert ref_obs.shape[1] == 215, f"Expected 215 dims, got {ref_obs.shape[1]}"
        assert vec_obs.shape[1] == 215, f"Expected 215 dims, got {vec_obs.shape[1]}"

        # Baseline dims (0-204) should match closely
        np.testing.assert_allclose(
            vec_obs[:, :205], ref_obs[:, :205], atol=1e-3, rtol=1e-3,
            err_msg="Baseline dims (0-204) differ between vectorized and reference"
        )

        # Social dims (205-214) should match closely
        # Note: tie-breaking order may differ for equidistant neighbors,
        # so we compare sorted neighbor pairs instead of raw order
        for i in range(len(world.agents)):
            ref_social = ref_obs[i, 205:215].reshape(5, 2)
            vec_social = vec_obs[i, 205:215].reshape(5, 2)

            # Separate real entries (non-zero) from padding
            ref_real = ref_social[np.any(ref_social != 0, axis=1)]
            vec_real = vec_social[np.any(vec_social != 0, axis=1)]

            assert ref_real.shape == vec_real.shape, \
                f"Agent {i}: different number of visible neighbors: " \
                f"ref={ref_real.shape[0]}, vec={vec_real.shape[0]}"

            if ref_real.shape[0] > 0:
                # Sort by heading to handle tie-breaking differences
                ref_sorted = ref_real[np.argsort(ref_real[:, 0])]
                vec_sorted = vec_real[np.argsort(vec_real[:, 0])]
                np.testing.assert_allclose(
                    vec_sorted, ref_sorted, atol=1e-3, rtol=1e-3,
                    err_msg=f"Agent {i}: social obs entries differ"
                )

    def test_social_obs_shape_215(self, social_world):
        """Output shape is (max_agents, 215) with social config."""
        world, config = social_world
        obs_state = extract_obs_state(world, config)
        result = compute_all_observations(obs_state, config)
        assert result.shape == (world.physics["max_agents"], 215)

    def test_social_obs_inactive_zeroed(self, social_world):
        """Inactive agent slots have all-zero social obs."""
        world, config = social_world
        obs_state = extract_obs_state(world, config)
        result = compute_all_observations(obs_state, config)

        is_active = np.array(obs_state["is_active"])
        inactive_mask = ~is_active
        if inactive_mask.any():
            inactive_social = np.array(result[inactive_mask, 205:215])
            assert np.all(inactive_social == 0.0), \
                "Inactive slots should have zero social obs"
