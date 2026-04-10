"""
run_experiment_jax.py
---------------------
Experiment runner using the fully-JAX SimState architecture.

sim_step_core (steps 1-9) runs inside JIT. PPO (step 10) runs from Python
only for agents with full rollout buffers, avoiding vmap+lax.cond overhead.

Usage:
  python scripts/run_experiment_jax.py --config configs/baseline_faithful.yaml --seed 0
  python scripts/run_experiment_jax.py --config configs/baseline_faithful.yaml --seed 0 --max-steps 50000
"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import _build_physics
from src.jax_state import SimState, init_simstate
from src.jax_sim import build_sim_step
from src.jax_ppo import build_ppo_update_fn


def run_experiment_jax(config, seed, max_steps=None, out_dir="results"):
    """Main simulation loop using SimState + JIT-compiled sim_step_core."""
    total_steps = max_steps if max_steps is not None else config["total_steps"]
    rollout_steps = config["rollout_steps"]
    log_interval = config.get("log_interval_steps", 10_000)
    ckpt_interval = config.get("checkpoint_interval_steps", 25_000)
    max_agents = config["prey_cap"] + config["predator_cap"]

    # Build physics space and sim_step
    space, _ = _build_physics(config)
    sim_step_core, ppo_update_fn = build_sim_step(config, space)

    # Initialize state
    rng_key = jax.random.PRNGKey(seed)
    sim_state = init_simstate(config, rng_key)

    n_prey = int(jnp.sum((sim_state.species == 0) & sim_state.is_active))
    n_pred = int(jnp.sum((sim_state.species == 1) & sim_state.is_active))
    n_food = int(jnp.sum(sim_state.food_active))

    print(f"Starting experiment (JAX runner): {config.get('experiment_name', 'unnamed')}")
    print(f"  Seed: {seed}, Steps: {total_steps}")
    print(f"  Prey: {n_prey}, Predators: {n_pred}, Food: {n_food}")
    print(f"  Max agents: {max_agents}, Rollout steps: {rollout_steps}")
    print()

    # Warmup JIT
    print("Compiling sim_step_core (first call)...")
    t_compile_start = time.time()
    sim_state = sim_step_core(sim_state)
    jax.block_until_ready(sim_state.step)
    t_compile_end = time.time()
    print(f"  Compiled in {t_compile_end - t_compile_start:.1f}s")
    print()

    start_time = time.time()

    for step in range(1, total_steps):
        # --- Steps 1-9: JIT-compiled core ---
        sim_state = sim_step_core(sim_state)

        # --- Step 10: PPO update (Python-side, only for ready agents) ---
        ptrs = np.array(sim_state.rollout_ptrs)
        active = np.array(sim_state.is_active)
        ready_mask = (ptrs >= rollout_steps) & active
        ready_slots = np.where(ready_mask)[0]

        if len(ready_slots) > 0:
            rng, ppo_key = jax.random.split(sim_state.rng_key)
            ppo_rngs = jax.random.split(ppo_key, max_agents)

            new_params, new_opt, new_ptrs = ppo_update_fn(
                sim_state.policy_params, sim_state.policy_opt_states,
                sim_state.rollout_obs, sim_state.rollout_actions,
                sim_state.rollout_log_probs, sim_state.rollout_rewards,
                sim_state.rollout_values, sim_state.rollout_dones,
                sim_state.rollout_ptrs, sim_state.is_active, ppo_rngs,
            )

            sim_state = sim_state.replace(
                policy_params=new_params,
                policy_opt_states=new_opt,
                rollout_ptrs=new_ptrs,
                rng_key=rng,
            )

        # --- Logging ---
        if (step + 1) % log_interval == 0:
            jax.block_until_ready(sim_state.step)
            n_prey = int(jnp.sum((sim_state.species == 0) & sim_state.is_active))
            n_pred = int(jnp.sum((sim_state.species == 1) & sim_state.is_active))
            n_food = int(jnp.sum(sim_state.food_active))
            elapsed = time.time() - start_time
            sps = (step + 1) / elapsed

            # Reward weight stats
            prey_active = (sim_state.species == 0) & sim_state.is_active
            if jnp.any(prey_active):
                prey_w = sim_state.reward_weights[prey_active]
                std_w = float(jnp.std(prey_w[:, 3]))
            else:
                std_w = 0.0

            print(f"Step {step+1:>8d}/{total_steps} | "
                  f"prey={n_prey:>3d} pred={n_pred:>2d} | "
                  f"food={n_food:>3d} | "
                  f"{sps:.1f} steps/s | "
                  f"elapsed={elapsed:.0f}s | "
                  f"std(w_pred)={std_w:.3f}")

        # --- Checkpointing ---
        if (step + 1) % ckpt_interval == 0:
            _save_checkpoint_jax(sim_state, config, seed, out_dir)

    # Final save
    jax.block_until_ready(sim_state.step)
    _save_checkpoint_jax(sim_state, config, seed, out_dir)

    elapsed = time.time() - start_time
    print(f"\nDone. {total_steps} steps in {elapsed:.1f}s ({total_steps/elapsed:.1f} steps/s)")

    return sim_state


def _save_checkpoint_jax(sim_state, config, seed, out_dir):
    """Save SimState checkpoint."""
    exp_name = config.get("experiment_name", "unnamed")
    ckpt_dir = os.path.join(out_dir, exp_name, f"seed_{seed}", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    step = int(sim_state.step)
    path = os.path.join(ckpt_dir, f"step_{step:08d}.npz")

    # Save key arrays (not full pytree — too large)
    active = np.array(sim_state.is_active)
    np.savez_compressed(
        path,
        step=step,
        is_active=active,
        species=np.array(sim_state.species),
        energies=np.array(sim_state.energies),
        ages=np.array(sim_state.ages),
        reward_weights=np.array(sim_state.reward_weights),
        food_active=np.array(sim_state.food_active),
        food_internal=float(sim_state.food_internal),
    )


def main():
    parser = argparse.ArgumentParser(description="Run evo-reward experiment (JAX)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_experiment_jax(config, args.seed, args.max_steps, args.out_dir)


if __name__ == "__main__":
    main()
