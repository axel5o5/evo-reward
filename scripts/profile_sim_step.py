"""
profile_sim_step.py
-------------------
Temporary diagnostic: time each phase of sim_step_core individually.

Approach: build each sub-function as its own @jit'd function, warm the
JIT, then loop calling them in sequence with block_until_ready() between
phases so we get per-phase wall-time.

Breaking JIT fusion like this slows the whole sim down, so the absolute
numbers are inflated vs production. But the *ratio* between phases
tells us where time actually goes in the fused graph. That's what we
want for deciding which optimization to target.

Usage:
  python scripts/profile_sim_step.py --steps 50

Print format per phase: mean ms/step, std, total share of step time.
"""

import argparse
import os
import sys
import time

os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_enable_fast_math=true --xla_cpu_use_thunk_runtime=true",
)

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from functools import partial

from src.environment import _build_physics, N_PHYSICS_ITER, CHANNEL_PREY, CHANNEL_PREDATOR
from src.jax_state import init_simstate
from src.observations import _build_obs_fn
from src.jax_food import check_eating_jax, remove_eaten_food_jax, regenerate_food_jax
from src.jax_lifecycle import update_energies_jax, process_births_and_deaths_jax
from src.policy import PolicyNetwork
import phyjax2d as pj


def build_phase_fns(config, space):
    max_agents = config["prey_cap"] + config["predator_cap"]
    food_max = config["food_max"]
    rollout_steps = config["rollout_steps"]
    n_sensors = config["n_proximity_sensors"]
    n_channels = config.get("n_proximity_channels", 4)
    F_max = config["max_motor_norm"]

    obs_fn = _build_obs_fn(config, max_agents, food_max)
    net = PolicyNetwork(hidden_size=config["policy_hidden_size"], action_dim=2)

    @jax.jit
    def phase_obs(sim_state):
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
        return obs_fn(obs_state)

    @jax.jit
    def phase_policy(sim_state, all_obs):
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

        a, lp, v = jax.vmap(sample_one)(sim_state.policy_params, all_obs, all_rngs)
        a = jnp.where(sim_state.is_active[:, None], a, 0.0)
        return a, lp, v, rng

    @jax.jit
    def phase_physics(sim_state, all_actions):
        action_arr = all_actions * sim_state.act_ratio
        f1_raw = action_arr[:, 0:1]
        f2_raw = action_arr[:, 1:2]
        f1 = jnp.concatenate([jnp.zeros_like(f1_raw), f1_raw], axis=1)
        f2 = jnp.concatenate([jnp.zeros_like(f2_raw), f2_raw], axis=1)

        circle = sim_state.phyjax_stated.get("circle")
        circle = circle.apply_force_local(sim_state.act_p1, f1)
        circle = circle.apply_force_local(sim_state.act_p2, f2)
        stated = sim_state.phyjax_stated.replace(circle=circle)
        solver = sim_state.phyjax_solver

        def body(carry, _):
            st, sol = carry
            st, sol, _c = pj.step(space, st, sol)
            return (st, sol), None

        (stated, solver), _ = jax.lax.scan(body, (stated, solver), None, length=N_PHYSICS_ITER)
        return sim_state.replace(phyjax_stated=stated, phyjax_solver=solver)

    @jax.jit
    def phase_eating(sim_state):
        return check_eating_jax(sim_state, config)

    @jax.jit
    def phase_energy_births_food(sim_state, prey_n_eaten, pred_catch_slots, pred_n_catches,
                                 all_actions, food_eaten_mask):
        sim_state = remove_eaten_food_jax(sim_state, food_eaten_mask)
        sim_state = update_energies_jax(
            sim_state, prey_n_eaten, pred_catch_slots, pred_n_catches, all_actions, config
        )
        sim_state = process_births_and_deaths_jax(sim_state, config)
        sim_state = regenerate_food_jax(sim_state, config)
        return sim_state

    return phase_obs, phase_policy, phase_physics, phase_eating, phase_energy_births_food


def time_phase(fn, args, n_iter=20):
    """Run fn(*args) n_iter times, block after each, return mean/std ms."""
    # Warmup
    result = fn(*args)
    jax.block_until_ready(result if not isinstance(result, tuple) else result[0])

    samples = []
    for _ in range(n_iter):
        t0 = time.time()
        result = fn(*args)
        jax.block_until_ready(result if not isinstance(result, tuple) else result[0])
        samples.append((time.time() - t0) * 1000.0)
    return float(np.mean(samples)), float(np.std(samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_faithful.yaml")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Advance sim to build up realistic population")
    parser.add_argument("--iter", type=int, default=20,
                        help="Number of timing iterations per phase")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Overlay default runtime so config has the ops keys
    with open("configs/runtime/default.yaml") as f:
        config.update(yaml.safe_load(f))

    space, _ = _build_physics(config)
    sim_state = init_simstate(config, jax.random.PRNGKey(0))

    # Warm up sim to realistic population — use the real sim_step_core
    from src.jax_sim import build_sim_step
    sim_step_core, _ = build_sim_step(config, space)
    print(f"Warming up {args.warmup_steps} steps to build population...")
    for _ in range(args.warmup_steps):
        sim_state = sim_step_core(sim_state)
    jax.block_until_ready(sim_state.step)
    n_active = int(jnp.sum(sim_state.is_active))
    n_prey = int(jnp.sum((sim_state.species == 0) & sim_state.is_active))
    n_pred = int(jnp.sum((sim_state.species == 1) & sim_state.is_active))
    print(f"Population at profile time: {n_active} active ({n_prey} prey, {n_pred} predator)\n")

    phase_obs, phase_policy, phase_physics, phase_eating, phase_ebd = build_phase_fns(config, space)

    # Time each phase
    print("Timing phases (ms/step, mean ± std over {} iters):".format(args.iter))
    print("-" * 72)
    phase_times = []

    # Phase: observations
    mean, std = time_phase(phase_obs, (sim_state,), args.iter)
    phase_times.append(("observations", mean, std))

    all_obs = phase_obs(sim_state)
    jax.block_until_ready(all_obs)

    # Phase: policy sample
    mean, std = time_phase(phase_policy, (sim_state, all_obs), args.iter)
    phase_times.append(("policy sample", mean, std))

    all_actions, _, _, _ = phase_policy(sim_state, all_obs)
    jax.block_until_ready(all_actions)

    # Phase: physics
    mean, std = time_phase(phase_physics, (sim_state, all_actions), args.iter)
    phase_times.append(("physics (phyjax2d)", mean, std))

    sim_state_post_phys = phase_physics(sim_state, all_actions)
    jax.block_until_ready(sim_state_post_phys.step)

    # Phase: eating (this is the O(N^2) one)
    mean, std = time_phase(phase_eating, (sim_state_post_phys,), args.iter)
    phase_times.append(("eating (O(N^2) catch)", mean, std))

    prey_n_eaten, pred_catch_slots, pred_n_catches, food_eaten_mask = phase_eating(sim_state_post_phys)

    # Phase: energy + births/deaths + food regen
    mean, std = time_phase(
        phase_ebd,
        (sim_state_post_phys, prey_n_eaten, pred_catch_slots, pred_n_catches, all_actions, food_eaten_mask),
        args.iter,
    )
    phase_times.append(("energy + births + food regen", mean, std))

    # Full step for comparison
    mean_full, std_full = time_phase(sim_step_core, (sim_state,), args.iter)

    print(f"  {'phase':<32}  mean (ms)   ±std   share%")
    total_phased = sum(m for _, m, _ in phase_times)
    for name, m, s in phase_times:
        share = 100.0 * m / total_phased if total_phased > 0 else 0
        print(f"  {name:<32}  {m:>7.2f}   {s:>5.2f}   {share:>5.1f}%")
    print("-" * 72)
    print(f"  {'sum of phases (broken JIT)':<32}  {total_phased:>7.2f}")
    print(f"  {'full sim_step_core (fused)':<32}  {mean_full:>7.2f}   {std_full:>5.2f}")
    print(f"  {'fusion speedup':<32}  {total_phased/mean_full:>7.2f}x")


if __name__ == "__main__":
    main()
