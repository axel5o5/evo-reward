"""
benchmark_vmap_seeds.py
-----------------------
Measure how throughput scales when running multiple independent seeds in
parallel on a single GPU via jax.vmap over the seed axis.

The question this answers: how much GPU headroom do we have at N=1, and
how many parallel seeds can we batch before per-seed throughput collapses?

For each N in --seeds, we:
  1. Build N independent SimStates (one rng_key per seed) and stack them
     along a new leading axis (jax.tree_util.tree_map + jnp.stack).
  2. JIT a vmap'd sim_step_core that maps over that leading axis.
  3. Warm up, then time `iter` steps with block_until_ready.
  4. Sample nvidia-smi util + memory at peak.

Output: one row per N showing total_steps/sec, steps/sec/seed, GPU util%,
GPU mem used. The interesting comparison is steps/sec/seed at N=1 vs N=8 —
if it barely drops, we're far from saturated and can run many seeds free.

Usage:
  python scripts/benchmark_vmap_seeds.py --config configs/baseline_faithful.yaml
  python scripts/benchmark_vmap_seeds.py --seeds 1,2,4,8,16 --warmup 100 --iter 200

Notes:
- PPO is excluded from the benchmark (it fires every PPO_CHECK_EVERY steps
  and runs from Python, not part of the JIT'd hot loop). sim_step_core is
  the hot path that dominates wall time during a real run.
- Set N too high and you'll OOM; the script catches OOMErrors per N and
  reports them so the rest of the sweep still completes.
"""
import argparse
import os
import subprocess
import sys
import time

# Match run_experiment_jax.py XLA flags so the benchmark reflects production.
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


def stack_states(states):
    """Stack a list of SimState pytrees along a new leading axis."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


def gpu_snapshot():
    """Sample nvidia-smi for util + memory. Returns dict or None if unavailable."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=2.0,
        )
        if out.returncode != 0:
            return None
        # Take the first GPU's row. If we ever go multi-GPU, rework this.
        row = out.stdout.strip().splitlines()[0].split(",")
        return {
            "util_pct": int(row[0]),
            "mem_util_pct": int(row[1]),
            "mem_used_mb": int(row[2]),
            "mem_total_mb": int(row[3]),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError, ValueError):
        return None


def benchmark_n(config, space, n_seeds, warmup, n_iter):
    """Time vmap'd sim_step_core for a given N. Returns dict of metrics."""
    sim_step_core, _ = build_sim_step(config, space)

    # Build N independent initial states and stack them.
    states = [init_simstate(config, jax.random.PRNGKey(s)) for s in range(n_seeds)]
    batched = stack_states(states) if n_seeds > 1 else states[0]

    if n_seeds > 1:
        step_fn = jax.jit(jax.vmap(sim_step_core))
    else:
        step_fn = jax.jit(sim_step_core)

    # Warmup: trigger compile + populate sim a bit so we're not timing first-step
    # JIT compilation. Block once at the end so XLA finishes async dispatch.
    for _ in range(warmup):
        batched = step_fn(batched)
    jax.block_until_ready(batched.step)

    # Time the steady-state loop.
    gpu_before = gpu_snapshot()
    t0 = time.time()
    for _ in range(n_iter):
        batched = step_fn(batched)
    jax.block_until_ready(batched.step)
    elapsed = time.time() - t0
    gpu_after = gpu_snapshot()

    total_sps = n_iter / elapsed
    per_seed_sps = total_sps  # each step advances all N seeds simultaneously
    # Total seed-steps/sec: useful for "work done per second" comparisons.
    seed_steps_sps = total_sps * n_seeds

    return {
        "n_seeds": n_seeds,
        "elapsed_s": elapsed,
        "steps_per_sec": total_sps,
        "per_seed_steps_per_sec": per_seed_sps,
        "seed_steps_per_sec": seed_steps_sps,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_faithful.yaml")
    parser.add_argument("--seeds", default="1,2,4,8",
                        help="Comma-separated N values to test")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Steps to run before timing (covers JIT compile)")
    parser.add_argument("--iter", type=int, default=200,
                        help="Steps to time per N")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    runtime_yaml = "configs/runtime/default.yaml"
    if os.path.exists(runtime_yaml):
        with open(runtime_yaml) as f:
            config.update(yaml.safe_load(f))

    space, _ = _build_physics(config)
    n_list = [int(x) for x in args.seeds.split(",") if x.strip()]

    print(f"Config: {args.config}")
    print(f"JAX devices: {jax.devices()}")
    snap = gpu_snapshot()
    if snap:
        print(f"GPU at start: util={snap['util_pct']}% mem={snap['mem_used_mb']}/{snap['mem_total_mb']} MB")
    print(f"Seeds to test: {n_list}, warmup={args.warmup}, iter={args.iter}\n")

    results = []
    for n in n_list:
        print(f"--- N={n} ---")
        try:
            r = benchmark_n(config, space, n, args.warmup, args.iter)
            results.append(r)
            g = r["gpu_after"] or {}
            print(
                f"  total {r['steps_per_sec']:.1f} steps/s | "
                f"per-seed {r['per_seed_steps_per_sec']:.1f} steps/s | "
                f"seed-steps {r['seed_steps_per_sec']:.1f}/s | "
                f"GPU util {g.get('util_pct', '?')}% | "
                f"mem {g.get('mem_used_mb', '?')}/{g.get('mem_total_mb', '?')} MB"
            )
        except Exception as e:  # OOM, compile errors, etc.
            print(f"  FAILED: {type(e).__name__}: {e}")
            results.append({"n_seeds": n, "error": str(e)})

    # Summary table — the actual artifact someone wants to read.
    print("\n=== Summary ===")
    print(f"  {'N':>3}  {'total sps':>10}  {'per-seed sps':>13}  {'seed-steps/s':>13}  {'GPU%':>5}  {'mem MB':>10}")
    baseline = None
    for r in results:
        if "error" in r:
            print(f"  {r['n_seeds']:>3}  ERROR: {r['error'][:60]}")
            continue
        g = r["gpu_after"] or {}
        per_seed = r["per_seed_steps_per_sec"]
        if baseline is None:
            baseline = per_seed
        ratio = per_seed / baseline if baseline > 0 else 0.0
        print(
            f"  {r['n_seeds']:>3}  {r['steps_per_sec']:>10.1f}  "
            f"{per_seed:>13.1f}  {r['seed_steps_per_sec']:>13.1f}  "
            f"{g.get('util_pct', 0):>4}%  "
            f"{g.get('mem_used_mb', 0):>5}/{g.get('mem_total_mb', 0):<5}  "
            f"({ratio:.2f}x per-seed vs N=1)"
        )

    print(
        "\nReading the table: per-seed sps near baseline = headroom (run many seeds free).\n"
        "Per-seed sps falling fast = saturated (parallel seeds cost wall time per seed).\n"
        "Seed-steps/s rising near-linearly with N = the regime we want."
    )


if __name__ == "__main__":
    main()
