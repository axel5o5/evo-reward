"""
run_experiment.py
-----------------
Main simulation loop for evo-reward. Ties all modules together.

Entrypoint: python scripts/run_experiment.py --config configs/baseline_faithful.yaml --seed 0
Also: --max-steps N (for smoke tests without running to 10M)

Simulation step order (from AGENTS.md):
  1. Get observations for all agents
  2. Sample actions → write obs/action/logprob/value to rollout buffer
  3. step_physics()
  4. check_eating()
  5. Compute rewards → write to rollout buffer
  6. update_energies()
  7. process_births_and_deaths()
  8. regenerate_food()
  9. ppo_update() for agents whose rollout buffer is full
  10. log_step() and save_checkpoint() on interval

Performance: policy forward passes are batched via vmap for speed.
Physics via JIT-compiled phyjax2d.
Observations computed once per step, reused for reward stimuli.
"""

import argparse
import math
import os
import sys
import time

import yaml
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import (
    init_world, step_physics, check_eating, remove_eaten_food,
    map_actions, sync_physics_after_population_change, sync_physics_food,
    get_sensor_readings, CHANNEL_PREY, CHANNEL_PREDATOR,
)
from src.agents import get_observation
from src.reward import compute_linear_reward
from src.lifecycle import update_energies, process_births_and_deaths, regenerate_food
from src.policy import init_policy, sample_action, PolicyNetwork, policy_forward
from src.ppo import ppo_update
from src.metrics import MetricsLog, log_step, record_birth, save_checkpoint, save_metrics


# Required config keys (referenced by test_components.py)
REQUIRED_CONFIG_KEYS = {
    "experiment_name", "world_size", "total_steps", "obs_dim",
    "prey_initial", "predator_initial", "prey_cap", "predator_cap",
    "prey_radius", "predator_radius", "max_motor_norm",
    "n_proximity_sensors", "n_proximity_channels", "proximity_fov_deg",
    "proximity_max_range", "n_tactile_sensors", "n_tactile_channels",
    "tactile_spacing_deg",
    "food_max", "food_initial", "food_growth_rate", "food_max_regen_per_step",
    "energy_capacity", "prey_e_food", "prey_c_b", "prey_c_a",
    "predator_d_b", "predator_d_a", "predator_eta",
    "predator_mouth_deg", "predator_mouth_range_min", "predator_mouth_range_max",
    "kappa_h", "alpha_e", "beta_h",
    "alpha_t_prey", "alpha_t_pred", "beta_t_prey", "beta_t_pred",
    "kappa_b", "beta_b", "zeta_b_prey", "zeta_b_pred",
    "energy_share_ratio", "spawn_spread",
    "reward_weights_init_std", "mutation_df", "mutation_scale", "weight_clip",
    "policy_hidden_size", "policy_n_hidden_layers",
    "action_clip_low", "action_clip_high", "action_mapping",
    "gamma", "rollout_steps", "minibatch_size", "ppo_epochs",
    "clip_epsilon", "entropy_coef", "gae_lambda", "lr", "adam_eps",
    "checkpoint_interval_steps", "log_interval_steps", "seed",
}


# ---------------------------------------------------------------------------
# Batched policy operations
# ---------------------------------------------------------------------------

_batched_sample_cache = {}

# Stacked-params cache: avoid rebuilding jtu.tree_map stack every step.
# Rebuilt only when the set of alive agents changes (birth/death). When PPO
# updates one agent's params, we update just that agent's slot.
_param_stack_cache = {
    "padded_n": 0,
    "agent_ids": (),
    "params": None,
    "slots": {},
}


def _get_stacked_params(agent_order):
    """Return (padded_stacked_params, padded_n) for the current agent set.

    Caches the stacked pytree and only rebuilds from scratch when the set of
    agent IDs changes (birth/death). On most steps (no pop change) this is a
    cache hit costing ~0ms vs the 50ms of jtu.tree_map from scratch.
    """
    n = len(agent_order)
    pn = _next_pow2(n)
    ids = tuple(a.agent_id for a in agent_order)

    c = _param_stack_cache
    if c["padded_n"] == pn and c["agent_ids"] == ids:
        return c["params"], pn

    # Population changed — rebuild from scratch
    dummy = jtu.tree_map(lambda x: jnp.zeros_like(x), agent_order[0].policy_params)
    all_p = [a.policy_params for a in agent_order] + [dummy] * (pn - n)
    c["params"] = jtu.tree_map(lambda *xs: jnp.stack(xs), *all_p)
    c["padded_n"] = pn
    c["agent_ids"] = ids
    c["slots"] = {a.agent_id: i for i, a in enumerate(agent_order)}
    return c["params"], pn


def _update_param_slot(agent_id, new_params):
    """Update one agent's slot in the stacked params cache after PPO update."""
    c = _param_stack_cache
    slot = c["slots"].get(agent_id)
    if slot is None or c["params"] is None:
        return
    c["params"] = jtu.tree_map(
        lambda stack, new: stack.at[slot].set(new),
        c["params"], new_params
    )


def _next_pow2(n):
    """Round up to next power of 2 (minimum 2)."""
    return 2 ** math.ceil(math.log2(max(n, 2)))


def _get_batched_sampler(config, n_agents):
    """Get or create JIT-compiled batched action sampler.

    Pads to the next power of 2 so that population fluctuations (births/deaths)
    don't trigger constant JAX recompilation. Without this, each unique n_agents
    triggers a ~5-10s recompile; with this, at most log2(max_pop) compilations
    happen across the entire run (e.g., 64 → 128 → 256 for pop range 45-151).

    Returns (sampler_fn, padded_n).
    """
    padded_n = _next_pow2(n_agents)

    if padded_n in _batched_sample_cache:
        return _batched_sample_cache[padded_n], padded_n

    net = PolicyNetwork(hidden_size=config["policy_hidden_size"], action_dim=2)

    @jax.jit
    def _sample(stacked_params, all_obs, all_rngs):
        def _single(p, o, k):
            mean, log_std, value = net.apply(p, o)
            std = jnp.exp(log_std)
            noise = jax.random.normal(k, shape=mean.shape)
            raw_action = mean + std * noise
            log_prob = -0.5 * jnp.sum(
                jnp.log(2 * jnp.pi) + 2 * log_std + ((raw_action - mean) / std) ** 2
            )
            action = 100.0 * jax.nn.sigmoid(raw_action) - 20.0
            return action, log_prob, value
        return jax.vmap(_single)(stacked_params, all_obs, all_rngs)

    _batched_sample_cache[padded_n] = _sample
    return _sample, padded_n


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

def make_rollout_buffer(config):
    """Create an empty rollout buffer dict (NumPy arrays for fast writes).

    We use NumPy here instead of JAX arrays because we write to the buffer
    every step via indexed assignment. JAX's immutable .at[].set() creates
    a new array on each write (~23ms/step for 45 agents); NumPy assignment
    is in-place and ~20x faster. ppo_update receives jnp.array() views.
    """
    N = config["rollout_steps"]
    obs_dim = config["obs_dim"]
    return {
        "observations": np.zeros((N, obs_dim), dtype=np.float32),
        "actions": np.zeros((N, 2), dtype=np.float32),
        "log_probs": np.zeros(N, dtype=np.float32),
        "rewards": np.zeros(N, dtype=np.float32),
        "values": np.zeros(N, dtype=np.float32),
        "dones": np.zeros(N, dtype=bool),
        "ptr": 0,
    }


def ensure_agent_initialized(agent, config, rng_key):
    """Ensure agent has policy params, opt state, and rollout buffer."""
    if agent.policy_params is None:
        rng_key, init_key = jax.random.split(rng_key)
        agent.policy_params, agent.policy_opt_state = init_policy(init_key, config)
    if agent.rollout is None:
        agent.rollout = make_rollout_buffer(config)
    return agent, rng_key


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _extract_stimuli_from_obs(obs, config):
    """Extract reward-relevant stimuli from a pre-computed observation vector.

    Avoids redundant sensor computation by reading directly from obs.
    """
    n_sensors = config["n_proximity_sensors"]
    n_channels = config.get("n_proximity_channels", 4)

    # Proximity sensors are at obs[0:128], shape (32, 4) flattened row-major
    prox = obs[:n_sensors * n_channels].reshape(n_sensors, n_channels)
    prey_readings = prox[:, CHANNEL_PREY]
    pred_readings = prox[:, CHANNEL_PREDATOR]

    # Clip >= 0 before aggregation
    max_s_prey = float(jnp.max(jnp.clip(prey_readings, 0.0)))
    max_s_pred = float(jnp.max(jnp.clip(pred_readings, 0.0)))

    return {
        "n_eaten": 0,
        "motor_norm": 0.0,
        "max_s_prey": max_s_prey,
        "max_s_pred": max_s_pred,
    }


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_experiment(config, seed, max_steps=None, out_dir="results"):
    """Main simulation loop."""
    total_steps = max_steps if max_steps is not None else config["total_steps"]
    rollout_steps = config["rollout_steps"]
    log_interval = config["log_interval_steps"]
    ckpt_interval = config["checkpoint_interval_steps"]
    F_max = config["max_motor_norm"]

    # Initialize world
    rng_key = jax.random.PRNGKey(seed)
    rng_key, init_key = jax.random.split(rng_key)
    world = init_world(config, init_key)

    # Initialize all agents' policies and rollout buffers
    for agent in world.agents:
        agent, rng_key = ensure_agent_initialized(agent, config, rng_key)

    # Metrics
    log = MetricsLog()
    log = log_step(log, world, config)

    # Rolling counters for ecological metrics
    food_eaten_total = 0
    prey_caught_total = 0
    steps_since_log = 0

    start_time = time.time()

    n_prey_init = sum(1 for a in world.agents if a.species == 0)
    n_pred_init = sum(1 for a in world.agents if a.species == 1)
    print(f"Starting experiment: {config.get('experiment_name', 'unnamed')}")
    print(f"  Seed: {seed}, Steps: {total_steps}")
    print(f"  Prey: {n_prey_init}, Predators: {n_pred_init}")
    print(f"  Food: {len(world.food_positions) if world.food_positions is not None else 0}")
    print()

    # Warmup physics JIT
    warmup_actions = {a.agent_id: jnp.zeros(2) for a in world.agents}
    world = step_physics(world, warmup_actions, config)
    world.step -= 1

    for step in range(total_steps):
        n_agents = len(world.agents)
        if n_agents == 0:
            print(f"Step {step}: all agents extinct. Stopping.")
            break

        agent_order = list(world.agents)

        # === 1. Get observations for all agents ===
        obs_list = []
        for agent in agent_order:
            obs = get_observation(world, agent.agent_id, config)
            obs_list.append(obs)
        all_obs = jnp.stack(obs_list)

        # === 2. Sample actions (batched) ===
        rng_key, sample_key = jax.random.split(rng_key)
        all_rngs = jax.random.split(sample_key, n_agents)

        # Get padded stacked params (cached; only rebuilt on population change)
        padded_params, padded_n = _get_stacked_params(agent_order)
        sampler, _ = _get_batched_sampler(config, n_agents)

        pad = padded_n - n_agents
        padded_obs = jnp.concatenate(
            [all_obs, jnp.zeros((pad, all_obs.shape[1]))], axis=0
        ) if pad > 0 else all_obs
        padded_rngs = jnp.concatenate(
            [all_rngs, jax.random.split(sample_key, pad)], axis=0
        ) if pad > 0 else all_rngs

        _all_actions, _all_log_probs, _all_values = sampler(padded_params, padded_obs, padded_rngs)
        all_actions = _all_actions[:n_agents]
        all_log_probs = _all_log_probs[:n_agents]
        all_values = _all_values[:n_agents]

        # Write to rollout buffers (NumPy in-place: ~1ms vs ~23ms for JAX .at[].set())
        actions = {}
        for i, agent in enumerate(agent_order):
            actions[agent.agent_id] = all_actions[i]
            buf = agent.rollout
            ptr = buf["ptr"]
            buf["observations"][ptr] = np.asarray(all_obs[i])
            buf["actions"][ptr] = np.asarray(all_actions[i])
            buf["log_probs"][ptr] = float(all_log_probs[i])
            buf["values"][ptr] = float(all_values[i])

        # === 3. Step physics (actions already sigmoid-scaled) ===
        world = step_physics(world, actions, config)

        # === 4. Check eating events ===
        eating_events, food_eaten_indices = check_eating(world, config)
        world = remove_eaten_food(world, food_eaten_indices)
        sync_physics_food(world, config)

        # Count ecological metrics
        for aid, val in eating_events.items():
            agent_sp = None
            for a in agent_order:
                if a.agent_id == aid:
                    agent_sp = a.species
                    break
            if agent_sp == 0 and isinstance(val, int):
                food_eaten_total += val
            elif agent_sp == 1 and isinstance(val, list):
                prey_caught_total += len(val)

        # === 5. Compute rewards (reuse pre-computed observations) ===
        for i, agent in enumerate(agent_order):
            aid = agent.agent_id
            # Extract stimuli from pre-computed obs (avoids redundant sensor scan)
            stimuli = _extract_stimuli_from_obs(all_obs[i], config)

            ev = eating_events.get(aid, 0)
            if agent.species == 0:
                stimuli["n_eaten"] = ev if isinstance(ev, int) else 0
            else:
                stimuli["n_eaten"] = len(ev) if isinstance(ev, list) else 0

            action = actions[aid]
            motor_norm = float(jnp.linalg.norm(action)) / F_max
            stimuli["motor_norm"] = motor_norm

            r = float(compute_linear_reward(
                agent.reward_weights,
                stimuli["n_eaten"], stimuli["motor_norm"],
                stimuli["max_s_prey"], stimuli["max_s_pred"],
            ))

            buf = agent.rollout
            ptr = buf["ptr"]
            buf["rewards"][ptr] = r
            buf["dones"][ptr] = False
            buf["ptr"] = ptr + 1

        # === 6. Update energies ===
        world = update_energies(world, eating_events, actions, config)

        # === 7. Process births and deaths ===
        rng_key, bd_key = jax.random.split(rng_key)
        world, dead_ids, born_ids = process_births_and_deaths(world, bd_key, config)

        newborns = []
        for agent in world.agents:
            if agent.agent_id in born_ids:
                agent, rng_key = ensure_agent_initialized(agent, config, rng_key)
                newborns.append(agent)

        sync_physics_after_population_change(world, dead_ids, newborns, config)

        for cid in born_ids:
            for a in world.agents:
                if a.agent_id == cid:
                    log = record_birth(log, world.step, cid, a.parent_id)
                    break

        # === 8. Regenerate food ===
        world = regenerate_food(world, config)
        sync_physics_food(world, config)

        # === 9. PPO update for agents with full rollout buffers ===
        for agent in world.agents:
            if agent.rollout is None:
                continue
            if agent.rollout["ptr"] >= rollout_steps:
                agent.policy_params, agent.policy_opt_state, _info = ppo_update(
                    agent.policy_params, agent.policy_opt_state,
                    agent.rollout, config,
                )
                agent.rollout = make_rollout_buffer(config)
                # Update this agent's slot in the stacked params cache
                _update_param_slot(agent.agent_id, agent.policy_params)

        # === 10. Log and checkpoint ===
        steps_since_log += 1
        if (step + 1) % log_interval == 0:
            log = log_step(log, world, config)
            if log.capture_rate and steps_since_log > 0:
                log.capture_rate[-1] = prey_caught_total / steps_since_log
                log.food_consumption_rate[-1] = food_eaten_total / steps_since_log
            food_eaten_total = 0
            prey_caught_total = 0
            steps_since_log = 0

            n_prey = sum(1 for a in world.agents if a.species == 0)
            n_pred = sum(1 for a in world.agents if a.species == 1)
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            prey_agents = [a for a in world.agents if a.species == 0]
            print(f"Step {step+1:>8d}/{total_steps} | "
                  f"prey={n_prey:>3d} pred={n_pred:>2d} | "
                  f"food={len(world.food_positions) if world.food_positions is not None else 0:>3d} | "
                  f"{steps_per_sec:.1f} steps/s | "
                  f"elapsed={elapsed:.0f}s", end="")
            if prey_agents:
                prey_w = np.array([np.array(a.reward_weights) for a in prey_agents])
                std_w = np.std(prey_w, axis=0)
                print(f" | std(w_pred)={std_w[3]:.3f}", end="")
            print()

        if (step + 1) % ckpt_interval == 0:
            save_checkpoint(world, log, config, seed, out_dir)

    # Final save
    save_metrics(log, config, seed, out_dir)
    save_checkpoint(world, log, config, seed, out_dir)

    elapsed = time.time() - start_time
    print(f"\nDone. {total_steps} steps in {elapsed:.1f}s ({total_steps/elapsed:.1f} steps/s)")

    return world, log


def main():
    parser = argparse.ArgumentParser(description="Run evo-reward experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override total_steps (for smoke tests)")
    parser.add_argument("--out-dir", default="results", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    config["seed"] = args.seed

    run_experiment(config, args.seed, max_steps=args.max_steps, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
