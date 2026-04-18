"""
run_experiment_jax.py
---------------------
Experiment runner using the fully-JAX SimState architecture.

sim_step_core (steps 1-9) runs inside JIT. PPO (step 10) runs from Python
only for agents with full rollout buffers, avoiding vmap+lax.cond overhead.

Usage:
  python scripts/run_experiment_jax.py --config configs/baseline_faithful.yaml --seed 0
  python scripts/run_experiment_jax.py --config configs/baseline_faithful.yaml --seed 0 --max-steps 50000
  python scripts/run_experiment_jax.py --config configs/baseline_faithful.yaml --seed 0 --resume
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
from src.jax_checkpoint import (
    save_simstate,
    load_simstate,
    find_latest_checkpoint,
    rotate_checkpoints,
    checkpoint_path,
)


CHECKPOINTS_TO_KEEP = 3


def run_experiment_jax(config, seed, max_steps=None, out_dir="results",
                       resume=False, resume_from=None):
    """Main simulation loop using SimState + JIT-compiled sim_step_core."""
    total_steps = max_steps if max_steps is not None else config["total_steps"]
    rollout_steps = config["rollout_steps"]
    log_interval = config.get("log_interval_steps", 10_000)
    ckpt_interval = config.get("checkpoint_interval_steps", 100_000)
    max_agents = config["prey_cap"] + config["predator_cap"]
    exp_name = config.get("experiment_name", "unnamed")

    ckpt_dir = os.path.join(out_dir, exp_name, f"seed_{seed}", "checkpoints")

    # Build physics space and sim_step
    space, _ = _build_physics(config)
    sim_step_core, ppo_update_fn = build_sim_step(config, space)

    # Initialize state (also serves as the deserialization template)
    rng_key = jax.random.PRNGKey(seed)
    sim_state = init_simstate(config, rng_key)

    # Resume handling
    latest = find_latest_checkpoint(ckpt_dir)
    if resume or resume_from is not None:
        path = resume_from or latest
        if path is None:
            sys.exit(
                f"--resume requested but no checkpoint found in {ckpt_dir}. "
                f"Run without --resume to start fresh."
            )
        sim_state = load_simstate(path, sim_state)
        print(f"Resumed from {path} at step {int(sim_state.step)}")
    elif latest is not None:
        sys.exit(
            f"Checkpoints already exist in {ckpt_dir}.\n"
            f"  Use --resume to continue from {latest}, or\n"
            f"  delete the checkpoint dir to start fresh."
        )

    n_prey = int(jnp.sum((sim_state.species == 0) & sim_state.is_active))
    n_pred = int(jnp.sum((sim_state.species == 1) & sim_state.is_active))
    n_food = int(jnp.sum(sim_state.food_active))

    print(f"Starting experiment (JAX runner): {exp_name}")
    print(f"  Seed: {seed}, Steps: {total_steps}")
    print(f"  Prey: {n_prey}, Predators: {n_pred}, Food: {n_food}")
    print(f"  Max agents: {max_agents}, Rollout steps: {rollout_steps}")
    print()

    # Warmup JIT (always advances step by 1, whether fresh or resumed)
    print("Compiling sim_step_core (first call)...")
    t_compile_start = time.time()
    sim_state = sim_step_core(sim_state)
    jax.block_until_ready(sim_state.step)
    t_compile_end = time.time()
    print(f"  Compiled in {t_compile_end - t_compile_start:.1f}s")
    print()

    start_time = time.time()
    start_step = int(sim_state.step)

    for step in range(start_step, total_steps):
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
            prey_mask = (sim_state.species == 0) & sim_state.is_active
            pred_mask = (sim_state.species == 1) & sim_state.is_active
            n_prey = int(jnp.sum(prey_mask))
            n_pred = int(jnp.sum(pred_mask))
            n_food = int(jnp.sum(sim_state.food_active))

            elapsed = time.time() - start_time
            steps_done = (step + 1) - start_step
            sps = steps_done / elapsed if elapsed > 0 else 0.0

            # Energy — mean across living agents (sanity check for population health)
            any_active = bool(jnp.any(sim_state.is_active))
            mean_energy = (
                float(jnp.mean(sim_state.energies[sim_state.is_active]))
                if any_active else 0.0
            )

            # Reward weights — mean ± std of the two science-critical weights
            # across prey (w_pred = fear, w_prey = social affiliation).
            # reward_weights layout: [w_eat, w_act, w_prey, w_pred]
            if n_prey > 0:
                prey_w = sim_state.reward_weights[prey_mask]
                wpd_mean = float(jnp.mean(prey_w[:, 3]))
                wpd_std = float(jnp.std(prey_w[:, 3]))
                wpy_mean = float(jnp.mean(prey_w[:, 2]))
                wpy_std = float(jnp.std(prey_w[:, 2]))
            else:
                wpd_mean = wpd_std = wpy_mean = wpy_std = 0.0

            print(
                f"Step {step+1:>8d}/{total_steps} | "
                f"prey={n_prey:>3d} pred={n_pred:>2d} food={n_food:>3d} | "
                f"E={mean_energy:>5.1f} | "
                f"w_pred={wpd_mean:+.2f}±{wpd_std:.2f} "
                f"w_prey={wpy_mean:+.2f}±{wpy_std:.2f} | "
                f"{sps:.1f} sps | "
                f"{elapsed:.0f}s"
            )

        # --- Checkpointing ---
        if (step + 1) % ckpt_interval == 0:
            _save_checkpoint_jax(sim_state, out_dir, exp_name, seed)

    # Final save
    jax.block_until_ready(sim_state.step)
    _save_checkpoint_jax(sim_state, out_dir, exp_name, seed)

    elapsed = time.time() - start_time
    steps_done = total_steps - start_step
    print(f"\nDone. {steps_done} steps this invocation in {elapsed:.1f}s "
          f"({steps_done/elapsed:.1f} steps/s). Final step: {total_steps}")

    return sim_state


def _save_checkpoint_jax(sim_state, out_dir, exp_name, seed):
    """Save full SimState and rotate old checkpoints."""
    step = int(sim_state.step)
    path = checkpoint_path(out_dir, exp_name, seed, step)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_simstate(sim_state, path)
    rotate_checkpoints(os.path.dirname(path), keep=CHECKPOINTS_TO_KEEP)


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUNTIME_CONFIG = os.path.join(_REPO_ROOT, "configs", "runtime", "default.yaml")


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Run evo-reward experiment (JAX)")
    parser.add_argument("--config", required=True,
                        help="Path to YAML science config (physics, reward, PPO params)")
    parser.add_argument("--runtime", default=DEFAULT_RUNTIME_CONFIG,
                        help="Path to YAML runtime config (checkpoint / log cadence). "
                             "Defaults to configs/runtime/default.yaml.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                        help="Override checkpoint_interval_steps from --runtime yaml")
    parser.add_argument("--log-interval", type=int, default=None,
                        help="Override log_interval_steps from --runtime yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in the seed's output dir")
    parser.add_argument("--resume-from", default=None,
                        help="Resume from an explicit checkpoint path (overrides --resume auto-detect)")
    args = parser.parse_args()

    # Science config is the base; runtime overlays (runtime wins on conflict);
    # CLI flags win over runtime. Science configs should not define ops keys
    # anymore — but if they do, runtime still takes precedence.
    config = _load_yaml(args.config)
    runtime = _load_yaml(args.runtime)
    config.update(runtime)

    if args.checkpoint_interval is not None:
        config["checkpoint_interval_steps"] = args.checkpoint_interval
    if args.log_interval is not None:
        config["log_interval_steps"] = args.log_interval

    run_experiment_jax(
        config, args.seed, args.max_steps, args.out_dir,
        resume=args.resume, resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
