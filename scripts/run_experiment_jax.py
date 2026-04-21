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
import datetime
import json
import os
import sys
import time

# Enable CPU-friendly XLA optimizations (harmless on GPU — flags are ignored
# by the non-CPU backend). Must be set before `import jax` so XLA picks them
# up at client init. Override by exporting XLA_FLAGS before invoking.
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_enable_fast_math=true --xla_cpu_use_thunk_runtime=true",
)

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
from src import jax_metrics
from scripts.replay_recorder import ReplayRecorder


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
    metrics_file = jax_metrics.metrics_path(out_dir, exp_name, seed)

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
        # Also restore time-series metrics so the saved trajectory is continuous.
        if os.path.exists(metrics_file):
            metrics_log = jax_metrics.load(metrics_file)
            print(f"Restored metrics history: {len(metrics_log.steps)} log points")
        else:
            metrics_log = jax_metrics.JaxMetrics()
            print("Warning: checkpoint found but no metrics.npz — starting fresh metrics")
    elif latest is not None:
        sys.exit(
            f"Checkpoints already exist in {ckpt_dir}.\n"
            f"  Use --resume to continue from {latest}, or\n"
            f"  delete the checkpoint dir to start fresh."
        )
    else:
        metrics_log = jax_metrics.JaxMetrics()

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

    # Replay recorder — off unless replay_record_interval_steps is set in config.
    # Bucket defaults to EVO_REWARD_REPLAYS_BUCKET (or the upload module's
    # DEFAULT_BUCKET). Set config["replay_bucket"] to "" to disable uploads
    # and only write locally.
    replay_local_root = os.path.join(out_dir, exp_name, f"seed_{seed}", "replays")
    replay_bucket_cfg = config.get("replay_bucket")
    replay_bucket = (
        replay_bucket_cfg if replay_bucket_cfg is not None
        else os.environ.get("EVO_REWARD_REPLAYS_BUCKET")
    )
    replay_bucket = replay_bucket or None  # empty string → disabled
    recorder = ReplayRecorder(config, exp_name, seed, replay_local_root, bucket=replay_bucket)
    if recorder.enabled:
        print(f"Replay recorder: every {recorder.interval:,} steps, "
              f"capturing last {recorder.length} frames, "
              f"{'quantized' if recorder.quantize else 'raw'}, "
              f"bucket={replay_bucket or '(local only)'}, "
              f"retention={recorder.retention_policy}")
        print()

    start_time = time.time()
    start_step = int(sim_state.step)

    # PPO readiness is batched every PPO_CHECK_EVERY steps to amortize the
    # host<->device sync (np.array(sim_state.rollout_ptrs) blocks the Python
    # loop on GPU completion). The tradeoff: when an agent's rollout fills
    # (ptrs reaches rollout_steps=1024), subsequent sim_step_core calls clip
    # the write index to slot rollout_steps-1 (jax_sim.py:123), so the final
    # slot may get overwritten by up to (PPO_CHECK_EVERY - 1) later observations
    # before PPO fires. At PPO_CHECK_EVERY=10 and rollout_steps=1024 that's
    # ~0.1% of the rollout. Override with env var for exact-timing comparison.
    PPO_CHECK_EVERY = int(os.environ.get("EVO_PPO_CHECK_EVERY", "10"))

    def _maybe_fire_ppo(state):
        """Pull rollout_ptrs+is_active, fire PPO for any ready agents."""
        ptrs = np.array(state.rollout_ptrs)
        active = np.array(state.is_active)
        ready_mask = (ptrs >= rollout_steps) & active
        ready_slots = np.where(ready_mask)[0]
        if len(ready_slots) == 0:
            return state
        rng, ppo_key = jax.random.split(state.rng_key)
        ppo_rngs = jax.random.split(ppo_key, max_agents)
        new_params, new_opt, new_ptrs = ppo_update_fn(
            state.policy_params, state.policy_opt_states,
            state.rollout_obs, state.rollout_actions,
            state.rollout_log_probs, state.rollout_rewards,
            state.rollout_values, state.rollout_dones,
            state.rollout_ptrs, state.is_active, ppo_rngs,
        )
        return state.replace(
            policy_params=new_params,
            policy_opt_states=new_opt,
            rollout_ptrs=new_ptrs,
            rng_key=rng,
        )

    # Progress file lands next to checkpoints so the gcs-sync sidecar picks
    # it up for free. Dashboard monitor reads this via the GCS API so it
    # can show training progress without SSHing the VM.
    progress_file = os.path.join(out_dir, exp_name, f"seed_{seed}", "progress.json")

    def _log_progress(state, step):
        """Persist time-series metrics and print a one-line progress summary."""
        jax.block_until_ready(state.step)

        # Authoritative values go into the persisted metrics log first.
        jax_metrics.record(metrics_log, state)

        # Extract what we need for the progress line from the just-appended entry.
        n_prey = metrics_log.prey_population[-1]
        n_pred = metrics_log.predator_population[-1]
        n_food = int(jnp.sum(state.food_active))
        elapsed = time.time() - start_time
        steps_done = (step + 1) - start_step
        sps = steps_done / elapsed if elapsed > 0 else 0.0
        any_active = bool(jnp.any(state.is_active))
        mean_energy = (
            float(jnp.mean(state.energies[state.is_active]))
            if any_active else 0.0
        )
        # All 8 reward-weight trajectories (means + stds).
        # Surfaces the full Phase 1a gate, including the new predator w_pred
        # criterion (strongest K&D finding) and prey w_eat.
        py_eat_m,  py_eat_s  = metrics_log.prey_mean_w_eat[-1],  metrics_log.prey_std_w_eat[-1]
        py_act_m,  py_act_s  = metrics_log.prey_mean_w_act[-1],  metrics_log.prey_std_w_act[-1]
        py_prey_m, py_prey_s = metrics_log.prey_mean_w_prey[-1], metrics_log.prey_std_w_prey[-1]
        py_pred_m, py_pred_s = metrics_log.prey_mean_w_pred[-1], metrics_log.prey_std_w_pred[-1]
        pd_eat_m,  pd_eat_s  = metrics_log.pred_mean_w_eat[-1],  metrics_log.pred_std_w_eat[-1]
        pd_act_m,  pd_act_s  = metrics_log.pred_mean_w_act[-1],  metrics_log.pred_std_w_act[-1]
        pd_prey_m, pd_prey_s = metrics_log.pred_mean_w_prey[-1], metrics_log.pred_std_w_prey[-1]
        pd_pred_m, pd_pred_s = metrics_log.pred_mean_w_pred[-1], metrics_log.pred_std_w_pred[-1]
        print(
            f"Step {step+1:>8d}/{total_steps} | "
            f"prey={n_prey:>3d} pred={n_pred:>2d} food={n_food:>3d} | "
            f"E={mean_energy:>5.1f} | "
            f"prey_w eat={py_eat_m:+.2f}±{py_eat_s:.2f} act={py_act_m:+.2f}±{py_act_s:.2f} "
            f"prey={py_prey_m:+.2f}±{py_prey_s:.2f} pred={py_pred_m:+.2f}±{py_pred_s:.2f} | "
            f"pred_w eat={pd_eat_m:+.2f}±{pd_eat_s:.2f} act={pd_act_m:+.2f}±{pd_act_s:.2f} "
            f"prey={pd_prey_m:+.2f}±{pd_prey_s:.2f} pred={pd_pred_m:+.2f}±{pd_pred_s:.2f} | "
            f"{sps:.1f} sps | "
            f"{elapsed:.0f}s"
        )

        # Mirror the same values to progress.json. Atomic replace (write +
        # rename) so the gcs-sync sidecar never catches a half-written file.
        progress_payload = {
            "updated_at": datetime.datetime.now(datetime.timezone.utc)
                .isoformat(timespec="seconds"),
            "experiment_name": exp_name,
            "seed": int(seed),
            "step": int(step + 1),
            "total_steps": int(total_steps),
            "sps": float(sps),
            "elapsed_seconds": float(elapsed),
            "population": {
                "prey": int(n_prey),
                "pred": int(n_pred),
                "food": int(n_food),
                "mean_energy": float(mean_energy),
            },
            "reward_weights": {
                "prey": {
                    "eat":  [float(py_eat_m),  float(py_eat_s)],
                    "act":  [float(py_act_m),  float(py_act_s)],
                    "prey": [float(py_prey_m), float(py_prey_s)],
                    "pred": [float(py_pred_m), float(py_pred_s)],
                },
                "pred": {
                    "eat":  [float(pd_eat_m),  float(pd_eat_s)],
                    "act":  [float(pd_act_m),  float(pd_act_s)],
                    "prey": [float(pd_prey_m), float(pd_prey_s)],
                    "pred": [float(pd_pred_m), float(pd_pred_s)],
                },
            },
        }
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        tmp = progress_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(progress_payload, f, indent=2)
        os.replace(tmp, progress_file)

    for step in range(start_step, total_steps):
        # --- Steps 1-9: JIT-compiled core ---
        sim_state = sim_step_core(sim_state)

        # --- Replay recording (zero-cost outside the window) ---
        # Pass step+1 as a plain Python int so the recorder's boundary check
        # doesn't force a host sync on sim_state.step every tick.
        recorder.step(sim_state, step + 1)

        # --- Step 10: PPO update (batched) ---
        if (step + 1) % PPO_CHECK_EVERY == 0:
            sim_state = _maybe_fire_ppo(sim_state)

        # --- Logging ---
        if (step + 1) % log_interval == 0:
            _log_progress(sim_state, step)

        # --- Checkpointing ---
        if (step + 1) % ckpt_interval == 0:
            # Fire any pending PPO so the checkpoint reflects the correct
            # post-update state (ptrs reset etc.) rather than a mid-batch
            # stale view.
            sim_state = _maybe_fire_ppo(sim_state)
            _save_checkpoint_jax(sim_state, out_dir, exp_name, seed)
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            jax_metrics.save(metrics_log, metrics_file)

    # Final flush: fire any pending PPO, then save
    sim_state = _maybe_fire_ppo(sim_state)
    jax.block_until_ready(sim_state.step)
    _save_checkpoint_jax(sim_state, out_dir, exp_name, seed)
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    jax_metrics.save(metrics_log, metrics_file)

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
