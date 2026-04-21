"""
test_jax_metrics.py
-------------------
Unit tests for src/jax_metrics.py — the JaxMetrics dataclass and its
save/load/record helpers.

These are load-bearing for scripts/validate_replication.py: if metrics
aren't recorded correctly, the Phase 1a gate check silently operates
on wrong data.

Run: pytest tests/test_jax_metrics.py -v
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src import jax_metrics
from src.jax_metrics import JaxMetrics
from src.jax_state import init_simstate


@pytest.fixture
def config():
    """Small config copied from tests/test_checkpoint_jax.py for isolation."""
    return {
        "experiment_name": "test_metrics",
        "world_size": 960,
        "total_steps": 10_240_000,
        "obs_dim": 205,
        "prey_initial": 10,
        "predator_initial": 2,
        "prey_cap": 20,
        "predator_cap": 5,
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
        "food_max": 100,
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
        "rollout_steps": 64,
        "minibatch_size": 16,
        "ppo_epochs": 2,
        "clip_epsilon": 0.2,
        "entropy_coef": 0.001,
        "gae_lambda": 0.95,
        "lr": 3.0e-4,
        "adam_eps": 1.0e-7,
        "vf_coef": 0.5,
    }


class TestRecord:

    def test_first_record_populates_all_fields(self, config):
        """A fresh log recorded once has a single sample per field."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        log = JaxMetrics()

        jax_metrics.record(log, state)

        for f in log.__dataclass_fields__:
            assert len(getattr(log, f)) == 1, f"Field {f} has wrong length"

    def test_populations_match_state(self, config):
        state = init_simstate(config, jax.random.PRNGKey(0))
        log = JaxMetrics()
        jax_metrics.record(log, state)

        expected_prey = int(jnp.sum((state.species == 0) & state.is_active))
        expected_pred = int(jnp.sum((state.species == 1) & state.is_active))
        assert log.prey_population[-1] == expected_prey
        assert log.predator_population[-1] == expected_pred

    def test_reward_stats_match_direct_computation(self, config):
        """prey_mean_w_pred and prey_std_w_pred match jnp.mean/std directly."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        log = JaxMetrics()
        jax_metrics.record(log, state)

        prey_mask = (state.species == 0) & state.is_active
        prey_w = state.reward_weights[prey_mask]
        expected_mean_wpred = float(jnp.mean(prey_w[:, 3]))
        expected_std_wpred = float(jnp.std(prey_w[:, 3]))

        assert abs(log.prey_mean_w_pred[-1] - expected_mean_wpred) < 1e-6
        assert abs(log.prey_std_w_pred[-1] - expected_std_wpred) < 1e-6

    def test_multiple_records_grow_lists(self, config):
        state = init_simstate(config, jax.random.PRNGKey(0))
        log = JaxMetrics()
        for _ in range(5):
            jax_metrics.record(log, state)
        assert len(log.steps) == 5
        assert len(log.prey_mean_w_pred) == 5

    def test_zero_population_records_zeros(self, config):
        """When a species has no active agents, stats are zero (not NaN)."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        # Kill all predators
        new_active = state.is_active & (state.species == 0)
        state = state.replace(is_active=new_active)

        log = JaxMetrics()
        jax_metrics.record(log, state)
        assert log.predator_population[-1] == 0
        assert log.predator_mean_energy[-1] == 0.0
        assert log.pred_mean_w_pred[-1] == 0.0
        assert log.pred_std_w_pred[-1] == 0.0


class TestSaveLoadRoundtrip:

    def test_save_load_empty_log(self, tmp_path):
        log = JaxMetrics()
        path = str(tmp_path / "metrics.npz")
        jax_metrics.save(log, path)
        loaded = jax_metrics.load(path)
        for f in log.__dataclass_fields__:
            assert getattr(loaded, f) == [], f"Field {f} not empty after reload"

    def test_save_load_populated_log(self, config, tmp_path):
        state = init_simstate(config, jax.random.PRNGKey(0))
        log = JaxMetrics()
        for _ in range(3):
            jax_metrics.record(log, state)

        path = str(tmp_path / "metrics.npz")
        jax_metrics.save(log, path)
        loaded = jax_metrics.load(path)

        for f in log.__dataclass_fields__:
            assert getattr(loaded, f) == getattr(log, f), f"Field {f} mismatch"

    def test_save_is_atomic(self, config, tmp_path):
        """No stray .tmp file remains after save completes."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        log = JaxMetrics()
        jax_metrics.record(log, state)
        path = str(tmp_path / "metrics.npz")
        jax_metrics.save(log, path)
        leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []

    def test_load_then_append_roundtrip(self, config, tmp_path):
        """Load preserves list-mutability so record() can continue appending."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        log = JaxMetrics()
        for _ in range(3):
            jax_metrics.record(log, state)

        path = str(tmp_path / "metrics.npz")
        jax_metrics.save(log, path)

        resumed = jax_metrics.load(path)
        for _ in range(2):
            jax_metrics.record(resumed, state)
        assert len(resumed.steps) == 5


class TestValidateReplicationCompat:

    def test_metrics_npz_has_keys_validate_expects(self, config, tmp_path):
        """Smoke test: saved metrics.npz exposes the keys
        scripts/validate_replication.py looks for."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        log = JaxMetrics()
        jax_metrics.record(log, state)

        path = str(tmp_path / "metrics.npz")
        jax_metrics.save(log, path)
        loaded = np.load(path)

        required = [
            "prey_mean_w_pred", "prey_mean_w_prey", "prey_mean_w_eat",
            "pred_mean_w_prey",
            "prey_population", "predator_population",
        ]
        for key in required:
            assert key in loaded.files, f"Missing key validate_replication reads: {key}"
