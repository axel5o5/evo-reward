"""
test_phase0.py
--------------
Integration gate tests for Phase 0. All must pass before Phase 1a.
Tests validate the full module integration:
  - Physics (world bounds, agent movement)
  - Food regeneration
  - Sensors (contact, range)
  - Energy dynamics (idle drain, starvation death)
  - Birth probability
  - Offspring policy independence
  - Checkpoint roundtrip

Run: pytest tests/test_phase0.py -v
"""

import math
import os
import pytest
import numpy as np
import jax
import jax.numpy as jnp


@pytest.fixture
def config():
    """Full config matching baseline_faithful.yaml."""
    return {
        "experiment_name": "test_phase0",
        "world_size": 960,
        "total_steps": 10_240_000,
        "obs_dim": 205,
        "prey_initial": 150,
        "predator_initial": 10,
        "prey_cap": 450,
        "predator_cap": 50,
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
        "food_max": 600,
        "food_initial": 40,
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
        "rollout_steps": 1024,
        "minibatch_size": 256,
        "ppo_epochs": 10,
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


@pytest.fixture
def small_config(config):
    """Small-population config for faster integration tests."""
    cfg = dict(config)
    cfg["prey_initial"] = 10
    cfg["predator_initial"] = 2
    cfg["prey_cap"] = 20
    cfg["predator_cap"] = 5
    cfg["food_initial"] = 10
    return cfg


# ─── World Bounds ────────────────────────────────────────────────────────────

class TestWorldBounds:

    def test_agents_within_bounds_after_init(self, small_config):
        """All agents placed inside [0, 960] x [0, 960] at initialization."""
        from src.environment import init_world
        rng = jax.random.PRNGKey(42)
        world = init_world(small_config, rng)
        ws = small_config["world_size"]
        for a in world.agents:
            px, py = float(a.position[0]), float(a.position[1])
            assert 0 <= px <= ws, f"Agent {a.agent_id} x={px} out of bounds"
            assert 0 <= py <= ws, f"Agent {a.agent_id} y={py} out of bounds"

    def test_agents_within_bounds_after_stepping(self, small_config):
        """Agents stay inside [0, 960] x [0, 960] after physics steps."""
        from src.environment import init_world, step_physics
        rng = jax.random.PRNGKey(42)
        world = init_world(small_config, rng)
        ws = small_config["world_size"]

        for step_i in range(20):
            # Random actions (large forces to test boundary clamping)
            rng, act_key = jax.random.split(rng)
            actions = {}
            for a in world.agents:
                actions[a.agent_id] = jax.random.uniform(
                    act_key, shape=(2,), minval=-20.0, maxval=80.0
                )
                rng, act_key = jax.random.split(rng)

            world = step_physics(world, actions, small_config)

            for a in world.agents:
                px, py = float(a.position[0]), float(a.position[1])
                assert 0 <= px <= ws + 1, f"Step {step_i}: agent {a.agent_id} x={px} escaped"
                assert 0 <= py <= ws + 1, f"Step {step_i}: agent {a.agent_id} y={py} escaped"


# ─── Food Regeneration ───────────────────────────────────────────────────────

class TestFoodRegeneration:

    def test_food_approaches_capacity(self, config):
        """Starting from 0 food, food_internal approaches 600 within expected steps.

        Growth rate = 0.5/step, capacity = 600.
        After 1200 steps: food_internal = min(0 + 0.5*1200, 600) = 600.
        Actual food items should be close to 600 (limited by max_regen_per_step=10).
        """
        from src.environment import WorldState
        from src.lifecycle import regenerate_food

        world = WorldState(
            step=0,
            agents=[],
            food_internal=0.0,
            food_positions=jnp.zeros((0, 2)),
            rng_key=jax.random.PRNGKey(0),
        )

        for _ in range(1400):
            world = regenerate_food(world, config)

        # Internal counter should be at capacity
        assert abs(world.food_internal - 600.0) < 1.0, (
            f"food_internal should be ~600, got {world.food_internal}"
        )
        # Actual food items should be close to 600
        # (limited by max_regen_per_step=10, starting from 0, 1400 steps > 600/10=60 needed)
        n_food = len(world.food_positions)
        assert n_food >= 580, f"Expected ~600 food items, got {n_food}"


# ─── Sensor Contact ──────────────────────────────────────────────────────────

class TestSensorContact:

    def test_food_proximity_at_contact(self, config):
        """Agent adjacent to food reads proximity = 1.0 in the food channel."""
        from src.environment import (
            AgentState, WorldState, compute_proximity_sensors, CHANNEL_FOOD,
        )

        # Place agent at (480, 480). With phyjax2d convention, angle=0 means
        # forward is world +y.
        agent = AgentState(
            agent_id=0, species=0,
            position=jnp.array([480.0, 480.0]),
            velocity=jnp.zeros(2),
            angle=0.0, ang_vel=0.0,
            energy=100.0,
        )

        # Place food directly in front (+y), at contact distance.
        # Contact center distance for prey-food is prey_r + food_r = 14.
        food_pos = jnp.array([[480.0, 494.0]])

        world = WorldState(
            step=0,
            agents=[agent],
            food_positions=food_pos,
            rng_key=jax.random.PRNGKey(0),
        )

        proximity = compute_proximity_sensors(agent, world, config)
        # proximity shape: (32, 4)
        # Food channel is CHANNEL_FOOD=2
        food_readings = proximity[:, CHANNEL_FOOD]
        max_food = float(jnp.max(food_readings))
        assert max_food >= 0.9, (
            f"Food at contact should read ~1.0, got max={max_food}"
        )

    def test_sensor_range_beyond_max(self, config):
        """Agent > 200 units from all objects reads all proximity sensors = -1.0
        (i.e., nothing detected beyond max range)."""
        from src.environment import (
            AgentState, WorldState, compute_proximity_sensors,
        )

        # Place agent in center, no other agents, food far away
        agent = AgentState(
            agent_id=0, species=0,
            position=jnp.array([480.0, 480.0]),
            velocity=jnp.zeros(2),
            angle=0.0, ang_vel=0.0,
            energy=100.0,
        )

        # Place food very far away straight ahead (+y), outside max range=200.
        food_pos = jnp.array([[480.0, 480.0 + 250.0]])

        world = WorldState(
            step=0,
            agents=[agent],
            food_positions=food_pos,
            rng_key=jax.random.PRNGKey(0),
        )

        proximity = compute_proximity_sensors(agent, world, config)
        # All readings should be -1.0 (nothing detected) except possibly walls
        # For non-wall channels at center of 960x960 world, all walls are >200 away
        non_wall = proximity[:, :3]  # prey, predator, food channels
        max_non_wall = float(jnp.max(non_wall))
        assert max_non_wall <= 0.0, (
            f"No objects within range, expected all <= 0.0, got max={max_non_wall}"
        )


# ─── Energy Dynamics ─────────────────────────────────────────────────────────

class TestEnergyDynamics:

    def test_energy_decrease_idle(self, config):
        """Agent loses c_b = 1e-4 per step with no motor output and no food."""
        from src.lifecycle import update_energy_prey

        e0 = 100.0
        c_b = config["prey_c_b"]
        e1 = update_energy_prey(e0, n_eaten=0, action_norm=0.0, config=config)
        expected = e0 - c_b
        assert abs(e1 - expected) < 1e-8, (
            f"Idle energy loss: expected {expected}, got {e1}"
        )

    def test_death_starvation(self, config):
        """Agent with e < 0 is removed in next process_births_and_deaths call."""
        from src.environment import AgentState, WorldState
        from src.lifecycle import process_births_and_deaths

        # Agent with negative energy
        agent = AgentState(
            agent_id=42, species=0,
            position=jnp.array([480.0, 480.0]),
            velocity=jnp.zeros(2),
            angle=0.0, ang_vel=0.0,
            energy=-0.001,
            reward_weights=jnp.zeros(4),
        )

        world = WorldState(
            step=100,
            agents=[agent],
            food_positions=jnp.zeros((0, 2)),
            rng_key=jax.random.PRNGKey(0),
        )

        world, dead_ids, born_ids = process_births_and_deaths(world, jax.random.PRNGKey(1), config)
        assert 42 in dead_ids, f"Agent with e<0 should die, dead_ids={dead_ids}"
        assert len(world.agents) == 0, f"Dead agent should be removed"


# ─── Birth Probability ──────────────────────────────────────────────────────

class TestBirthProbability:

    def test_birth_formula_known_values(self, config):
        """b(e) formula correct at known values."""
        from src.lifecycle import birth_prob

        # Prey at e=100: b = κ_b / (1 + exp(ζ - β_b*e))
        # = 1e-3 / (1 + exp(15 - 0.4*100)) = 1e-3 / (1 + exp(-25)) ≈ 1e-3
        b100 = birth_prob(100.0, species=0, config=config)
        expected = config["kappa_b"] / (1 + math.exp(config["zeta_b_prey"] - config["beta_b"] * 100))
        assert abs(b100 - expected) < 1e-8, f"b(100) = {b100}, expected {expected}"

        # Prey at e=200 still ≈ κ_b (sigmoid saturated further toward max)
        b200 = birth_prob(200.0, species=0, config=config)
        expected200 = config["kappa_b"] / (1 + math.exp(config["zeta_b_prey"] - config["beta_b"] * 200))
        assert abs(b200 - expected200) < 1e-8, f"b(200) = {b200}, expected {expected200}"

        # Predator at e=100: exponent = 100 - 0.4*100 = 60, b ≈ 0
        b_pred = birth_prob(100.0, species=1, config=config)
        assert b_pred < 1e-20, f"Predator b(100) should be ~0, got {b_pred}"


# ─── Offspring Fresh Policy ──────────────────────────────────────────────────

class TestOffspringPolicy:

    def test_offspring_fresh_policy(self, config):
        """Child policy_params != parent policy_params after spawn."""
        from src.environment import AgentState
        from src.evolution import spawn_offspring
        from src.policy import init_policy

        rng = jax.random.PRNGKey(0)
        parent_params, _ = init_policy(rng, config)

        parent = AgentState(
            agent_id=0, species=0,
            position=jnp.array([480.0, 480.0]),
            velocity=jnp.zeros(2),
            angle=0.0, ang_vel=0.0,
            energy=200.0,
            reward_weights=jnp.array([1.0, -0.5, 0.5, -1.0]),
            policy_params=parent_params,
        )

        rng, spawn_key = jax.random.split(rng)
        child = spawn_offspring(parent, new_id=1, rng_key=spawn_key, config=config)

        # Child policy should be None (fresh init needed) or different from parent
        if child.policy_params is not None:
            # Compare flat parameter arrays
            import jax.tree_util as jtu
            parent_flat = jtu.tree_leaves(parent.policy_params)
            child_flat = jtu.tree_leaves(child.policy_params)
            all_same = all(
                jnp.allclose(p, c) for p, c in zip(parent_flat, child_flat)
            )
            assert not all_same, "Child policy_params should differ from parent"
        # child.policy_params == None is also valid (fresh init will happen later)

    def test_offspring_genome_mutated(self, config):
        """Child reward_weights differ from parent (mutated)."""
        from src.environment import AgentState
        from src.evolution import spawn_offspring

        parent = AgentState(
            agent_id=0, species=0,
            position=jnp.array([480.0, 480.0]),
            velocity=jnp.zeros(2),
            angle=0.0, ang_vel=0.0,
            energy=200.0,
            reward_weights=jnp.array([1.0, -0.5, 0.5, -1.0]),
        )

        rng = jax.random.PRNGKey(42)
        child = spawn_offspring(parent, new_id=1, rng_key=rng, config=config)

        assert not jnp.allclose(child.reward_weights, parent.reward_weights), (
            "Child genome should be mutated from parent"
        )


# ─── Checkpoint Roundtrip ───────────────────────────────────────────────────

class TestCheckpointRoundtrip:

    def test_save_load_reward_weights_identical(self, config, tmp_path):
        """save_checkpoint → load_checkpoint → reward_weights are identical."""
        from src.metrics import MetricsLog, save_checkpoint, load_checkpoint
        from types import SimpleNamespace

        rw1 = np.array([1.5, -2.3, 0.7, -0.9], dtype=np.float32)
        rw2 = np.array([-0.1, 0.2, -0.3, 0.4], dtype=np.float32)

        agents = [
            SimpleNamespace(
                agent_id=0, species=0, age=500, energy=75.0,
                reward_weights=rw1, parent_id=-1,
            ),
            SimpleNamespace(
                agent_id=1, species=1, age=200, energy=120.0,
                reward_weights=rw2, parent_id=-1,
            ),
        ]
        world = SimpleNamespace(step=50000, agents=agents)
        log = MetricsLog()

        save_checkpoint(world, log, config, seed=0, out_dir=str(tmp_path))
        ckpt_path = str(tmp_path / "test_phase0" / "seed_0" / "step_00050000.pkl")
        loaded = load_checkpoint(ckpt_path)

        np.testing.assert_array_almost_equal(
            loaded["reward_weights"][0], rw1,
            err_msg="Agent 0 reward_weights mismatch after roundtrip",
        )
        np.testing.assert_array_almost_equal(
            loaded["reward_weights"][1], rw2,
            err_msg="Agent 1 reward_weights mismatch after roundtrip",
        )
        assert loaded["step"] == 50000
