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
