"""
bench_l2_vs_l3.py
-----------------
Side-by-side timing benchmark for v10-L2 (axis1_residual_fast.yaml) vs v10-L3
(axis1_residual.yaml). Runs each config for N steps, fires at least one full
PPO cycle, and reports steps/sec and per-block breakdown (sim_step vs PPO).

The first call to sim_step_core triggers JIT compilation (~30-90s). We do a
short warmup, then time a longer measurement window so JIT overhead doesn't
contaminate the rate.

Usage:
  python3 scripts/bench_l2_vs_l3.py
  python3 scripts/bench_l2_vs_l3.py --steps 4096 --warmup 200
"""

import argparse
import os
import sys
import time

# CPU/GPU XLA flags (mirror runner)
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_enable_fast_math=true --xla_cpu_use_thunk_runtime=true",
)

import jax
import jax.numpy as jnp
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import _build_physics
from src.jax_state import init_simstate
from src.jax_sim import build_sim_step
from src.jax_ppo import build_ppo_update_fn
from src.config_utils import resolve_scale_dependent_params


def time_config(config_path: str, n_steps: int, warmup: int, label: str):
    print(f"\n{'='*70}\n{label}: {config_path}\n{'='*70}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = resolve_scale_dependent_params(config)

    print(f"  policy_hidden={config['policy_hidden_size']} "
          f"ppo_epochs={config['ppo_epochs']} "
          f"minibatch={config['minibatch_size']} "
          f"world={config['world_size']} "
          f"caps={config['prey_cap']}/{config['predator_cap']} "
          f"max_agents={config['prey_cap'] + config['predator_cap']}")

    space, _ = _build_physics(config)
    sim_step_core, ppo_update_fn = build_sim_step(config, space)

    state = init_simstate(config, jax.random.PRNGKey(config["seed"]))
    rollout_steps = config["rollout_steps"]

    # JIT compile
    print(f"  Compiling sim_step_core...", flush=True)
    t0 = time.time()
    state = sim_step_core(state)
    jax.block_until_ready(state.step)
    t_compile_sim = time.time() - t0
    print(f"    sim_step compiled in {t_compile_sim:.1f}s")

    # Warmup more steps
    for _ in range(warmup):
        state = sim_step_core(state)
    jax.block_until_ready(state.step)

    # Force-fill all rollout buffers so PPO fires
    state = state.replace(
        rollout_ptrs=jnp.full(state.rollout_ptrs.shape, rollout_steps, dtype=jnp.int32)
    )

    # Compile + time a single PPO fire
    max_agents = state.policy_params["params"]["Dense_0"]["kernel"].shape[0]
    rng_key, ppo_key = jax.random.split(state.rng_key)
    ppo_rngs = jax.random.split(ppo_key, max_agents)
    print(f"  Compiling + timing PPO update...", flush=True)
    t0 = time.time()
    new_params, new_opt, new_ptrs = ppo_update_fn(
        state.policy_params, state.policy_opt_states,
        state.rollout_obs, state.rollout_actions,
        state.rollout_log_probs, state.rollout_rewards,
        state.rollout_values, state.rollout_dones,
        state.rollout_ptrs, state.is_active, ppo_rngs,
        state.ages,
    )
    jax.block_until_ready(jax.tree_util.tree_leaves(new_params)[0])
    t_ppo_first = time.time() - t0
    print(f"    PPO compile + 1 call: {t_ppo_first:.2f}s")

    # Apply the result and time a 2nd PPO call (post-compile)
    state = state.replace(
        policy_params=new_params, policy_opt_states=new_opt, rollout_ptrs=new_ptrs,
        rng_key=rng_key,
    )
    state = state.replace(
        rollout_ptrs=jnp.full(state.rollout_ptrs.shape, rollout_steps, dtype=jnp.int32)
    )
    rng_key, ppo_key = jax.random.split(state.rng_key)
    ppo_rngs = jax.random.split(ppo_key, max_agents)
    t0 = time.time()
    new_params, new_opt, new_ptrs = ppo_update_fn(
        state.policy_params, state.policy_opt_states,
        state.rollout_obs, state.rollout_actions,
        state.rollout_log_probs, state.rollout_rewards,
        state.rollout_values, state.rollout_dones,
        state.rollout_ptrs, state.is_active, ppo_rngs,
        state.ages,
    )
    jax.block_until_ready(jax.tree_util.tree_leaves(new_params)[0])
    t_ppo_post = time.time() - t0
    print(f"    PPO post-compile call: {t_ppo_post*1000:.1f}ms")

    state = state.replace(
        policy_params=new_params, policy_opt_states=new_opt, rollout_ptrs=new_ptrs,
        rng_key=rng_key,
    )

    # Time pure sim_step_core for n_steps
    t0 = time.time()
    for _ in range(n_steps):
        state = sim_step_core(state)
    jax.block_until_ready(state.step)
    t_sim_block = time.time() - t0
    sps_pure_sim = n_steps / t_sim_block
    print(f"\n  Pure sim_step_core ({n_steps} steps): {t_sim_block:.2f}s = {sps_pure_sim:.1f} sps")
    print(f"    per-step: {t_sim_block/n_steps*1000:.2f}ms")

    # Effective sps once PPO is amortized over rollout_steps cycle:
    # Every rollout_steps sim steps, all agents fire one PPO update simultaneously.
    # The runner spreads this work via PPO_CHECK_EVERY but the total work is the same.
    effective_total = t_sim_block + t_ppo_post * (n_steps / rollout_steps)
    eff_sps = n_steps / effective_total
    ppo_share = (t_ppo_post * (n_steps / rollout_steps)) / effective_total
    print(f"  Effective sps (sim + amortized PPO): {eff_sps:.1f} sps")
    print(f"    PPO share of wall: {ppo_share*100:.1f}%")
    print(f"    1M-step extrapolation: {1_000_000 / eff_sps / 60:.1f} min")

    return {
        "label": label,
        "sps_pure_sim": sps_pure_sim,
        "eff_sps": eff_sps,
        "ppo_post_ms": t_ppo_post * 1000,
        "ppo_share": ppo_share,
        "step_ms": t_sim_block / n_steps * 1000,
        "compile_sim_s": t_compile_sim,
        "compile_ppo_s": t_ppo_first,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2048,
                        help="Sim steps in the timing window (default 2x rollout_steps)")
    parser.add_argument("--warmup", type=int, default=200)
    args = parser.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    L1 = os.path.join(repo, "configs", "axis1_residual_mini.yaml")
    L2 = os.path.join(repo, "configs", "axis1_residual_fast.yaml")
    L3 = os.path.join(repo, "configs", "axis1_residual.yaml")

    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"Steps per measurement: {args.steps}, warmup: {args.warmup}")

    r3 = time_config(L3, args.steps, args.warmup, "v10-L3 (axis1_residual)")
    r2 = time_config(L2, args.steps, args.warmup, "v10-L2 (axis1_residual_fast)")
    r1 = time_config(L1, args.steps, args.warmup, "v10-L1 (axis1_residual_mini)")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'metric':<26} {'L3':>10} {'L2':>10} {'L1':>10} "
          f"{'L2/L3':>8} {'L1/L3':>8} {'L1/L2':>8}")
    rows = [
        ("sim_step time / step",  "step_ms",      "ms"),
        ("PPO post-compile",      "ppo_post_ms",  "ms"),
        ("pure sim sps",          "sps_pure_sim", "sps"),
        ("effective sps",         "eff_sps",      "sps"),
    ]
    for label, key, unit in rows:
        print(f"{label:<26} {r3[key]:>9.1f}{unit} {r2[key]:>9.1f}{unit} {r1[key]:>9.1f}{unit} "
              f"{r2[key]/r3[key]:>8.2f}x {r1[key]/r3[key]:>8.2f}x {r1[key]/r2[key]:>8.2f}x")
    print(f"{'PPO % of wall':<26} {r3['ppo_share']*100:>9.1f}%  "
          f"{r2['ppo_share']*100:>9.1f}%  {r1['ppo_share']*100:>9.1f}%")

    print()
    for r, base in [(r2, r3), (r1, r3), (r1, r2)]:
        ratio = r["eff_sps"] / base["eff_sps"]
        delta_pct = (ratio - 1) * 100
        print(f"  {r['label']:<32} vs {base['label']:<32}: "
              f"{delta_pct:+6.1f}% ({ratio:.2f}x eff sps)")
    print()
    for r in (r3, r2, r1):
        per_1m = 1_000_000 / r["eff_sps"] / 60
        print(f"  {r['label']:<32}: {per_1m:>6.0f}min per 1M steps  "
              f"({per_1m/60:.1f}h)")


if __name__ == "__main__":
    main()
