"""
test_jax_ppo_update.py
----------------------
Integration tests for src/jax_ppo.py — the full per-agent conditional PPO update.

Only `_compute_gae_jax` was previously covered. The complete `build_ppo_update_fn`
pipeline (loss, gradient, Adam step, multi-epoch/minibatch loop, per-slot `lax.cond`)
is the core learning path and was untested. A silent bug in the loss, optimizer
wiring, or conditional update would quietly corrupt every training run.

These tests are marked `slow` because JIT compilation of the update graph takes
~20–40s on CPU the first time. Keep the config small.

Run: pytest tests/test_jax_ppo_update.py -v -m slow
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest

from src.jax_ppo import _compute_gae_jax, build_ppo_update_fn
from src.jax_state import init_simstate


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    """Tiny config so JIT compilation and the update itself are fast."""
    return {
        "experiment_name": "test_ppo",
        "world_size": 960,
        "total_steps": 1000,
        "obs_dim": 205,
        "prey_initial": 2,
        "predator_initial": 1,
        "prey_cap": 4,
        "predator_cap": 2,
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
        "food_max": 20,
        "food_initial": 5,
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
        "policy_hidden_size": 32,
        "policy_n_hidden_layers": 2,
        "action_clip_low": -20.0,
        "action_clip_high": 80.0,
        "action_mapping": "sigmoid",
        "gamma": 0.99,
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


def _fill_synthetic_rollout(state, config, rng):
    """Populate SimState rollout buffers with plausible synthetic data.

    Actions are sampled inside the network's sigmoid-scaled range [-20, 80]
    so the inverse mapping inside the PPO loss stays finite.
    """
    rollout_steps = config["rollout_steps"]
    obs_dim = config["obs_dim"]
    n_slots = state.rollout_obs.shape[0]

    k_obs, k_act, k_lp, k_rew, k_val = jax.random.split(rng, 5)

    obs = jax.random.normal(k_obs, (n_slots, rollout_steps, obs_dim)) * 0.1
    # Actions bounded safely inside (-20, 80) — outer clip in loss is (-19.99, 79.99).
    raw = jax.random.normal(k_act, (n_slots, rollout_steps, 2))
    actions = 100.0 * jax.nn.sigmoid(raw) - 20.0
    log_probs = -0.5 * jax.random.normal(k_lp, (n_slots, rollout_steps)) ** 2
    rewards = jax.random.normal(k_rew, (n_slots, rollout_steps)) * 0.1
    values = jax.random.normal(k_val, (n_slots, rollout_steps)) * 0.1
    dones = jnp.zeros((n_slots, rollout_steps), dtype=bool)

    return state.replace(
        rollout_obs=obs,
        rollout_actions=actions,
        rollout_log_probs=log_probs,
        rollout_rewards=rewards,
        rollout_values=values,
        rollout_dones=dones,
    )


def _params_equal(a, b, atol=0.0):
    """True if every leaf in two param pytrees is elementwise equal."""
    leaves_a = jtu.tree_leaves(a)
    leaves_b = jtu.tree_leaves(b)
    if len(leaves_a) != len(leaves_b):
        return False
    for la, lb in zip(leaves_a, leaves_b):
        if la.shape != lb.shape:
            return False
        if atol == 0.0:
            if not bool(jnp.all(la == lb)):
                return False
        else:
            if not bool(jnp.all(jnp.abs(la - lb) <= atol)):
                return False
    return True


# ─── GAE (already partially covered — add numerical regression) ───────────────

class TestComputeGAE:

    def test_zero_rewards_zero_values_gives_zero_advantages(self):
        rewards = jnp.zeros(8)
        values = jnp.zeros(8)
        dones = jnp.zeros(8, dtype=bool)
        adv, ret = _compute_gae_jax(rewards, values, dones, 0.0, gamma=0.99, gae_lambda=0.95)
        assert jnp.all(adv == 0.0)
        assert jnp.all(ret == 0.0)

    def test_returns_equal_advantages_plus_values(self):
        rewards = jnp.array([1.0, 0.5, -0.2, 0.3, 0.0, 1.0, -0.5, 0.2])
        values = jnp.array([0.1, 0.2, 0.3, 0.1, 0.0, -0.1, 0.2, 0.1])
        dones = jnp.zeros(8, dtype=bool)
        adv, ret = _compute_gae_jax(rewards, values, dones, 0.0, gamma=0.99, gae_lambda=0.95)
        np.testing.assert_allclose(np.asarray(ret), np.asarray(adv + values), rtol=1e-5)

    def test_done_breaks_advantage_propagation(self):
        """A `done` at timestep t means the advantage at t ignores values beyond."""
        rewards = jnp.array([0.0, 0.0, 1.0, 0.0])
        values = jnp.array([0.0, 0.0, 0.0, 10.0])  # large V after done
        dones_with = jnp.array([False, False, True, False])
        dones_without = jnp.array([False, False, False, False])

        adv_with, _ = _compute_gae_jax(rewards, values, dones_with, 0.0, 0.99, 0.95)
        adv_without, _ = _compute_gae_jax(rewards, values, dones_without, 0.0, 0.99, 0.95)
        # The first timestep advantage differs because done truncates bootstrap
        assert float(adv_with[0]) != float(adv_without[0])


# ─── Full PPO update ──────────────────────────────────────────────────────────

@pytest.mark.slow
class TestPPOUpdate:

    def _setup(self, config, seed=0):
        """Build an initial SimState with synthetic rollouts filled."""
        state = init_simstate(config, jax.random.PRNGKey(seed))
        state = _fill_synthetic_rollout(state, config, jax.random.PRNGKey(seed + 1))
        return state

    def _apply(self, config, state, rng_seed=99):
        """Run one full PPO update pass over the SimState."""
        update_fn = build_ppo_update_fn(config)
        n_slots = state.rollout_obs.shape[0]
        rng_keys = jax.random.split(jax.random.PRNGKey(rng_seed), n_slots)
        return update_fn(
            state.policy_params,
            state.policy_opt_states,
            state.rollout_obs,
            state.rollout_actions,
            state.rollout_log_probs,
            state.rollout_rewards,
            state.rollout_values,
            state.rollout_dones,
            state.rollout_ptrs,
            state.is_active,
            rng_keys,
        )

    def test_ptr_resets_after_update(self, config):
        """Active agents with a full buffer get their ptr reset to 0."""
        state = self._setup(config)
        rollout_steps = config["rollout_steps"]
        # Mark agent 0's buffer as full, others as partial.
        ptrs = jnp.array([rollout_steps, 5, 10, 0, 0, 0], dtype=jnp.int32)
        # Truncate to actual number of slots.
        n_slots = state.rollout_obs.shape[0]
        ptrs = ptrs[:n_slots]
        state = state.replace(rollout_ptrs=ptrs)

        _, _, new_ptrs = self._apply(config, state)
        # Active slot 0 with full buffer → reset to 0
        assert int(new_ptrs[0]) == 0
        # Slots with partial buffers → unchanged
        for i in range(1, n_slots):
            assert int(new_ptrs[i]) == int(ptrs[i])

    def test_params_change_for_updating_agent(self, config):
        """Running a PPO update must change the updating agent's params."""
        state = self._setup(config)
        n_slots = state.rollout_obs.shape[0]
        ptrs = jnp.zeros(n_slots, dtype=jnp.int32).at[0].set(config["rollout_steps"])
        state = state.replace(rollout_ptrs=ptrs)

        old_params = state.policy_params
        new_params, _, _ = self._apply(config, state)

        # Slot 0: params must differ.
        slot0_old = jtu.tree_map(lambda p: p[0], old_params)
        slot0_new = jtu.tree_map(lambda p: p[0], new_params)
        assert not _params_equal(slot0_old, slot0_new), (
            "PPO update ran but slot-0 params are unchanged"
        )

        # Slot 1 (not updating): params must be unchanged.
        slot1_old = jtu.tree_map(lambda p: p[1], old_params)
        slot1_new = jtu.tree_map(lambda p: p[1], new_params)
        assert _params_equal(slot1_old, slot1_new), (
            "Non-updating slot's params mutated"
        )

    def test_inactive_agent_not_updated(self, config):
        """An inactive slot must keep its params even with a full buffer."""
        state = self._setup(config)
        n_slots = state.rollout_obs.shape[0]
        n_active = int(jnp.sum(state.is_active))
        assert n_active < n_slots, "Test assumes there is at least one inactive slot"

        inactive_slot = n_active  # first inactive slot
        ptrs = jnp.zeros(n_slots, dtype=jnp.int32).at[inactive_slot].set(
            config["rollout_steps"]
        )
        state = state.replace(rollout_ptrs=ptrs)

        old_params = state.policy_params
        new_params, _, new_ptrs = self._apply(config, state)

        inactive_old = jtu.tree_map(lambda p: p[inactive_slot], old_params)
        inactive_new = jtu.tree_map(lambda p: p[inactive_slot], new_params)
        assert _params_equal(inactive_old, inactive_new)
        # ptr for inactive slot stays at the sentinel value
        assert int(new_ptrs[inactive_slot]) == config["rollout_steps"]

    def test_no_nan_or_inf_in_updated_params(self, config):
        """Gradient step must not produce NaN/Inf parameters."""
        state = self._setup(config)
        n_slots = state.rollout_obs.shape[0]
        ptrs = jnp.full(n_slots, config["rollout_steps"], dtype=jnp.int32)
        state = state.replace(rollout_ptrs=ptrs)

        new_params, new_opt, _ = self._apply(config, state)

        for leaf in jtu.tree_leaves(new_params):
            assert jnp.all(jnp.isfinite(leaf)), "Non-finite value in updated params"
        for leaf in jtu.tree_leaves(new_opt):
            arr = jnp.asarray(leaf)
            if jnp.issubdtype(arr.dtype, jnp.floating):
                assert jnp.all(jnp.isfinite(arr)), "Non-finite in optimizer state"

    def test_update_is_deterministic(self, config):
        """Same inputs + same RNG → same output."""
        state = self._setup(config)
        n_slots = state.rollout_obs.shape[0]
        ptrs = jnp.full(n_slots, config["rollout_steps"], dtype=jnp.int32)
        state = state.replace(rollout_ptrs=ptrs)

        p1, _, _ = self._apply(config, state, rng_seed=42)
        p2, _, _ = self._apply(config, state, rng_seed=42)
        assert _params_equal(p1, p2), "PPO update is non-deterministic with fixed RNG"
