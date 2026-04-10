"""
jax_sim.py
----------
Unified JIT-compiled simulation step function.

sim_step(sim_state, space) → sim_state

One call = one complete simulation step matching AGENTS.md order:
  1. Observations  2. Policy sample  3. Rollout write
  4. Physics       5. Eating         6. Rewards
  7. Energy        8. Birth/death    9. Food regen
  10. PPO update

All pure JAX. Same code runs on CPU and GPU.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import phyjax2d as pj

from src.policy import PolicyNetwork
from src.observations import _build_obs_fn
from src.jax_food import check_eating_jax, remove_eaten_food_jax, regenerate_food_jax
from src.jax_lifecycle import update_energies_jax, process_births_and_deaths_jax
from src.jax_ppo import build_ppo_update_fn
from src.environment import N_PHYSICS_ITER, CHANNEL_PREY, CHANNEL_PREDATOR


def build_sim_step(config, space):
    """Build the JIT-compiled simulation step function.

    Returns two functions:
        sim_step_core(sim_state) → sim_state  — steps 1-9, JIT-compiled
        ppo_update_fn — for step 10, called from Python for ready agents only

    PPO is separated because vmap+lax.cond over all 500 slots is too expensive
    on CPU (traces both branches for all slots). Instead, the runner checks
    which agents have full buffers and calls PPO only for those.
    On GPU, you can optionally inline PPO into sim_step for full JIT.
    """
    max_agents = config["prey_cap"] + config["predator_cap"]
    food_max = config["food_max"]
    obs_dim = config["obs_dim"]
    rollout_steps = config["rollout_steps"]
    n_sensors = config["n_proximity_sensors"]
    n_channels = config.get("n_proximity_channels", 4)
    F_max = config["max_motor_norm"]

    # Pre-build the JIT-compiled observation function
    obs_fn = _build_obs_fn(config, max_agents, food_max)

    # Pre-build the policy network for action sampling
    net = PolicyNetwork(hidden_size=config["policy_hidden_size"], action_dim=2)

    # Pre-build the PPO update function (used outside JIT by the runner)
    ppo_update_fn = build_ppo_update_fn(config)

    # Physics stepper
    @jax.jit
    def physics_step(stated, solver, act_p1, act_p2, f1, f2):
        circle = stated.get("circle")
        circle = circle.apply_force_local(act_p1, f1)
        circle = circle.apply_force_local(act_p2, f2)
        stated = stated.replace(circle=circle)

        def body(carry, _):
            st, sol = carry
            st, sol, _contact = pj.step(space, st, sol)
            return (st, sol), None

        (stated, solver), _ = jax.lax.scan(body, (stated, solver), None, length=N_PHYSICS_ITER)
        return stated, solver

    @jax.jit
    def sim_step_core(sim_state):
        """Steps 1-9 of the simulation. Pure JAX, JIT-compiled.

        PPO (step 10) is handled separately by the runner.
        """

        # === 1. Observations (vectorized) ===
        circle = sim_state.phyjax_stated.get("circle")
        obs_state = {
            "positions": circle.p.xy,
            "angles": circle.p.angle,
            "velocities_xy": circle.v.xy,
            "velocities_ang": circle.v.angle,
            "is_active": sim_state.is_active,
            "species": sim_state.species,
            "radii": sim_state.radii,
            "energies": sim_state.energies,
            "food_positions": sim_state.food_positions,
            "food_active": sim_state.food_active,
            "max_agents": max_agents,
        }
        all_obs = obs_fn(obs_state)  # (max_agents, obs_dim)

        # === 2. Policy sample (vmap over all agents) ===
        rng, sample_key = jax.random.split(sim_state.rng_key)
        all_rngs = jax.random.split(sample_key, max_agents)

        def sample_one(params, obs, rng_k):
            mean, log_std, value = net.apply(params, obs)
            std = jnp.exp(log_std)
            noise = jax.random.normal(rng_k, shape=mean.shape)
            raw_action = mean + std * noise
            log_prob = -0.5 * jnp.sum(
                jnp.log(2 * jnp.pi) + 2 * log_std + ((raw_action - mean) / std) ** 2
            )
            action = 100.0 * jax.nn.sigmoid(raw_action) - 20.0
            return action, log_prob, value

        all_actions, all_log_probs, all_values = jax.vmap(sample_one)(
            sim_state.policy_params, all_obs, all_rngs,
        )
        all_actions = jnp.where(sim_state.is_active[:, None], all_actions, 0.0)

        # === 3. Write to rollout buffers ===
        ptrs = sim_state.rollout_ptrs
        safe_ptrs = jnp.clip(ptrs, 0, rollout_steps - 1)
        agent_idx = jnp.arange(max_agents)

        new_rollout_obs = sim_state.rollout_obs.at[agent_idx, safe_ptrs].set(all_obs)
        new_rollout_actions = sim_state.rollout_actions.at[agent_idx, safe_ptrs].set(all_actions)
        new_rollout_log_probs = sim_state.rollout_log_probs.at[agent_idx, safe_ptrs].set(all_log_probs)
        new_rollout_values = sim_state.rollout_values.at[agent_idx, safe_ptrs].set(all_values)

        # === 4. Physics step ===
        action_arr = all_actions * sim_state.act_ratio
        f1_raw = action_arr[:, 0:1]
        f2_raw = action_arr[:, 1:2]
        f1 = jnp.concatenate([jnp.zeros_like(f1_raw), f1_raw], axis=1)
        f2 = jnp.concatenate([jnp.zeros_like(f2_raw), f2_raw], axis=1)

        new_stated, new_solver = physics_step(
            sim_state.phyjax_stated, sim_state.phyjax_solver,
            sim_state.act_p1, sim_state.act_p2, f1, f2,
        )

        sim_state = sim_state.replace(
            phyjax_stated=new_stated, phyjax_solver=new_solver,
            rollout_obs=new_rollout_obs, rollout_actions=new_rollout_actions,
            rollout_log_probs=new_rollout_log_probs, rollout_values=new_rollout_values,
        )

        # === 5. Check eating ===
        prey_n_eaten, pred_catch_slots, pred_n_catches, food_eaten_mask = check_eating_jax(
            sim_state, config
        )
        sim_state = remove_eaten_food_jax(sim_state, food_eaten_mask)

        # === 6. Compute rewards (vectorized) ===
        prox_all = all_obs[:, :n_sensors * n_channels].reshape(max_agents, n_sensors, n_channels)
        max_s_prey = jnp.max(jnp.clip(prox_all[:, :, CHANNEL_PREY], 0.0), axis=1)
        max_s_pred = jnp.max(jnp.clip(prox_all[:, :, CHANNEL_PREDATOR], 0.0), axis=1)
        motor_norms = jnp.linalg.norm(all_actions, axis=1) / F_max

        n_eaten_reward = jnp.where(
            sim_state.species == 0,
            prey_n_eaten.astype(jnp.float32),
            pred_n_catches.astype(jnp.float32),
        )

        coefs = jnp.array([1.0, 0.01, 0.1, 0.1])
        stimuli = jnp.stack([n_eaten_reward, motor_norms, max_s_prey, max_s_pred], axis=1)
        all_rewards = jnp.sum(sim_state.reward_weights * stimuli * coefs, axis=1)
        all_rewards = jnp.where(sim_state.is_active, all_rewards, 0.0)

        new_rollout_rewards = sim_state.rollout_rewards.at[agent_idx, safe_ptrs].set(all_rewards)
        new_rollout_dones = sim_state.rollout_dones.at[agent_idx, safe_ptrs].set(False)
        new_ptrs = jnp.where(sim_state.is_active, ptrs + 1, ptrs)

        sim_state = sim_state.replace(
            rollout_rewards=new_rollout_rewards, rollout_dones=new_rollout_dones,
            rollout_ptrs=new_ptrs,
        )

        # === 7. Update energies ===
        sim_state = update_energies_jax(
            sim_state, prey_n_eaten, pred_catch_slots, pred_n_catches,
            all_actions, config,
        )

        # === 8. Process births and deaths ===
        sim_state = process_births_and_deaths_jax(sim_state, config)

        # === 9. Regenerate food ===
        sim_state = regenerate_food_jax(sim_state, config)

        sim_state = sim_state.replace(rng_key=rng, step=sim_state.step + 1)
        return sim_state

    return sim_step_core, ppo_update_fn
