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
import subprocess
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
    AsyncCheckpointWriter,
    save_simstate,
    load_simstate,
    find_latest_checkpoint,
    rotate_checkpoints,
    checkpoint_path,
    list_run_tags,
    run_dir,
)
from src import jax_metrics
from scripts.replay_recorder import ReplayRecorder


CHECKPOINTS_TO_KEEP = 3


# psutil is optional — present on most envs but not in requirements.txt. If
# it's missing, we just skip host-side metrics (gpu still works on its own).
try:
    import psutil as _psutil
except ImportError:
    _psutil = None


def _sample_system_metrics():
    """Probe nvidia-smi + (optional) psutil for live GPU/host telemetry.

    Returns (gpu, host) where each is a dict or None. Called at every log
    interval, not per step — subprocess latency (~30ms) is fine at that
    cadence and the values get serialized into progress.json so the
    dashboard can show GPU saturation without SSHing the VM.
    """
    gpu = None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=2.0,
        )
        if out.returncode == 0 and out.stdout.strip():
            row = out.stdout.strip().splitlines()[0].split(",")
            gpu = {
                "util_pct": int(row[0]),
                "mem_util_pct": int(row[1]),
                "mem_used_mb": int(row[2]),
                "mem_total_mb": int(row[3]),
            }
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError, ValueError):
        gpu = None

    host = None
    if _psutil is not None:
        try:
            # interval=None reads since last call — non-blocking, but the first
            # call returns 0.0 by definition. Acceptable for log-interval cadence.
            cpu_pct = _psutil.cpu_percent(interval=None)
            vm = _psutil.virtual_memory()
            host = {
                "cpu_pct": float(cpu_pct),
                "ram_pct": float(vm.percent),
                "ram_used_mb": int(vm.used / 1e6),
                "ram_total_mb": int(vm.total / 1e6),
            }
        except Exception:
            host = None
    return gpu, host


def run_experiment_jax(config, seed, max_steps=None, out_dir="results",
                       resume=False, resume_from=None, run_tag=""):
    """Main simulation loop using SimState + JIT-compiled sim_step_core.

    `run_tag` isolates this run's checkpoints, replays, metrics, and progress
    under <out_dir>/<exp>/seed_<N>/<run_tag>/. Empty tag = legacy untagged
    layout, still supported for old runs but not recommended for new ones."""
    total_steps = max_steps if max_steps is not None else config["total_steps"]
    rollout_steps = config["rollout_steps"]
    log_interval = config.get("log_interval_steps", 10_000)
    from src.save_schedule import parse_schedule, interval_at
    ckpt_schedule = parse_schedule(
        config.get("checkpoint_interval_steps"), default_interval=100_000
    )
    max_agents = config["prey_cap"] + config["predator_cap"]
    exp_name = config.get("experiment_name", "unnamed")

    run_root = run_dir(out_dir, exp_name, seed, run_tag)
    ckpt_dir = os.path.join(run_root, "checkpoints")
    metrics_file = jax_metrics.metrics_path(out_dir, exp_name, seed, run_tag)

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
    replay_local_root = os.path.join(run_root, "replays")
    replay_bucket_cfg = config.get("replay_bucket")
    replay_bucket = (
        replay_bucket_cfg if replay_bucket_cfg is not None
        else os.environ.get("EVO_REWARD_REPLAYS_BUCKET")
    )
    replay_bucket = replay_bucket or None  # empty string → disabled
    recorder = ReplayRecorder(
        config, exp_name, seed, replay_local_root,
        bucket=replay_bucket, run_tag=run_tag,
    )
    if recorder.enabled:
        print(f"Replay recorder: every {recorder.interval:,} steps, "
              f"capturing last {recorder.length} frames, "
              f"{'quantized' if recorder.quantize else 'raw'}, "
              f"bucket={replay_bucket or '(local only)'}, "
              f"retention={recorder.retention_policy}")
        print()

    # Async checkpoint writer overlaps disk I/O with GPU compute. The host
    # snapshot still happens on the caller thread (np.asarray forces the
    # device→host barrier) but savez_compressed + atomic rename + rotation
    # run on a single background worker. Set EVO_SYNC_CHECKPOINT=1 to fall
    # back to the old inline sync path for A/B comparison.
    sync_ckpt = os.environ.get("EVO_SYNC_CHECKPOINT", "0") == "1"
    ckpt_writer = None if sync_ckpt else AsyncCheckpointWriter()
    print(f"Checkpoint writer: {'sync (legacy)' if sync_ckpt else 'async (background)'}")
    print()

    start_time = time.time()
    start_step = int(sim_state.step)

    # Next step at which to fire a checkpoint. Computed from the schedule so
    # resumed runs land on the same boundaries fresh runs would.
    next_ckpt_at = start_step + interval_at(ckpt_schedule, start_step)
    next_ckpt_at -= start_step % interval_at(ckpt_schedule, start_step)

    # D21: snapshot of cumulative counters at last log — used to derive
    # per-interval event counts without storing a per-step history.
    # Also remember prev_next_agent_id to derive per-interval births.
    _prev_log_state = {
        "step": start_step,
        "cum_catches": int(sim_state.cum_catches),
        "cum_deaths": int(sim_state.cum_deaths),
        "cum_feedings": int(sim_state.cum_feedings),
        "next_agent_id": int(sim_state.next_agent_id),
    }
    # How many consecutive log intervals with no catches / births — used
    # to gate the warning lines (one-shot on threshold, not on every line).
    _consec_no_catches = [0]
    _consec_no_births = [0]

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
            state.ages, state.species,
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
    progress_file = os.path.join(run_root, "progress.json")

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
        # D21: per-interval event counts derived from the cumulative counters.
        cum_catches = int(state.cum_catches)
        cum_deaths = int(state.cum_deaths)
        cum_feedings = int(state.cum_feedings)
        cur_next_id = int(state.next_agent_id)
        interval_catches = cum_catches - _prev_log_state["cum_catches"]
        interval_deaths = cum_deaths - _prev_log_state["cum_deaths"]
        interval_feedings = cum_feedings - _prev_log_state["cum_feedings"]
        interval_births = cur_next_id - _prev_log_state["next_agent_id"]

        # D21: energy-band stats — min/max expose saturation/starvation that
        # the population mean would hide (prey ~100 + pred ~991 → mean 189).
        is_active_np = np.asarray(state.is_active)
        species_np = np.asarray(state.species)
        energies_np = np.asarray(state.energies)
        def _band(mask):
            e = energies_np[mask]
            if e.size == 0:
                return (0.0, 0.0, 0.0)
            return (float(e.min()), float(e.mean()), float(e.max()))
        prey_e_band = _band(is_active_np & (species_np == 0))
        pred_e_band = _band(is_active_np & (species_np == 1))

        # v10: death-age percentiles from per-species ring buffers. Skipped
        # cleanly when the ring is the (1,) sentinel "disabled" shape.
        def _death_age_stats(ring):
            ring_np = np.asarray(ring)
            if ring_np.size <= 1:
                return None
            valid = ring_np[ring_np >= 0]
            if valid.size == 0:
                return None
            return {
                "n": int(valid.size),
                "p25": float(np.percentile(valid, 25)),
                "p50": float(np.percentile(valid, 50)),
                "p75": float(np.percentile(valid, 75)),
                "max": int(valid.max()),
            }
        prey_death_ages = _death_age_stats(state.death_age_ring_prey)
        pred_death_ages = _death_age_stats(state.death_age_ring_pred)

        _prev_log_state["step"] = step + 1
        _prev_log_state["cum_catches"] = cum_catches
        _prev_log_state["cum_deaths"] = cum_deaths
        _prev_log_state["cum_feedings"] = cum_feedings
        _prev_log_state["next_agent_id"] = cur_next_id

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
            f"E={mean_energy:>5.1f} "
            f"(prey {prey_e_band[0]:.0f}/{prey_e_band[1]:.0f}/{prey_e_band[2]:.0f} "
            f"pred {pred_e_band[0]:.0f}/{pred_e_band[1]:.0f}/{pred_e_band[2]:.0f}) | "
            f"Δ catch={interval_catches} death={interval_deaths} "
            f"birth={interval_births} feed={interval_feedings} | "
            f"prey_w eat={py_eat_m:+.2f}±{py_eat_s:.2f} act={py_act_m:+.2f}±{py_act_s:.2f} "
            f"prey={py_prey_m:+.2f}±{py_prey_s:.2f} pred={py_pred_m:+.2f}±{py_pred_s:.2f} | "
            f"pred_w eat={pd_eat_m:+.2f}±{pd_eat_s:.2f} act={pd_act_m:+.2f}±{pd_act_s:.2f} "
            f"prey={pd_prey_m:+.2f}±{pd_prey_s:.2f} pred={pd_pred_m:+.2f}±{pd_pred_s:.2f} | "
            f"{sps:.1f} sps | "
            f"{elapsed:.0f}s"
        )

        # v10: one-line death-age summary (median is the headline; p25/p75
        # show spread). Only print when the ring has at least one entry.
        if pred_death_ages is not None or prey_death_ages is not None:
            def _fmt(d, name):
                if d is None:
                    return f"{name}=n/a"
                return (f"{name} p25/50/75={d['p25']:.0f}/{d['p50']:.0f}/{d['p75']:.0f} "
                        f"(n={d['n']})")
            print(f"  death-age: {_fmt(prey_death_ages, 'prey')} | "
                  f"{_fmt(pred_death_ages, 'pred')}")

        # D21: warning lines — only print on transitions so logs don't spam.
        cap = float(config.get("energy_capacity", 1000.0))
        if interval_catches == 0 and n_pred > 0:
            _consec_no_catches[0] += 1
        else:
            _consec_no_catches[0] = 0
        if interval_births == 0:
            _consec_no_births[0] += 1
        else:
            _consec_no_births[0] = 0
        if _consec_no_catches[0] == 2:
            print(
                f"  ⚠ no predator catches for {2 * log_interval:,} consecutive steps — "
                f"check contact detection (see docs/emevo-diff.md D18–D20)"
            )
        if _consec_no_births[0] == 5:
            print(
                f"  ⚠ zero births for {5 * log_interval:,} consecutive steps — "
                f"evolution stalled; check energy/birth dynamics"
            )
        if pred_e_band[0] > cap * 0.95 and n_pred > 0:
            print(
                f"  ⚠ predator energy saturated (min={pred_e_band[0]:.0f} ≥ {cap*0.95:.0f}) — "
                f"predators may be eating without metabolic cost"
            )

        # Live GPU/host telemetry — populated when nvidia-smi (and optionally
        # psutil) are available on the VM. Either or both may be None on dev.
        gpu_metrics, host_metrics = _sample_system_metrics()

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
            # D21 visibility blocks: event deltas + energy bands per species.
            "events_last_interval": {
                "catches":  int(interval_catches),
                "deaths":   int(interval_deaths),
                "births":   int(interval_births),
                "feedings": int(interval_feedings),
                "interval_steps": int(log_interval),
            },
            "events_cumulative": {
                "catches":  cum_catches,
                "deaths":   cum_deaths,
                "feedings": cum_feedings,
                "next_agent_id": cur_next_id,
            },
            "energy_stats": {
                "prey": {"min": prey_e_band[0], "mean": prey_e_band[1], "max": prey_e_band[2]},
                "pred": {"min": pred_e_band[0], "mean": pred_e_band[1], "max": pred_e_band[2]},
            },
            # v10: death-age distribution (per-species ring of recent death ages).
            # null when the ring buffer is disabled or empty.
            "death_age_stats": {
                "prey": prey_death_ages,
                "pred": pred_death_ages,
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
            "gpu": gpu_metrics,
            "host": host_metrics,
        }
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        tmp = progress_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(progress_payload, f, indent=2)
        os.replace(tmp, progress_file)

    try:
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
            if (step + 1) >= next_ckpt_at:
                # Fire any pending PPO so the checkpoint reflects the correct
                # post-update state (ptrs reset etc.) rather than a mid-batch
                # stale view.
                sim_state = _maybe_fire_ppo(sim_state)
                _save_checkpoint_jax(
                    sim_state, out_dir, exp_name, seed, run_tag, ckpt_writer,
                )
                os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
                jax_metrics.save(metrics_log, metrics_file)
                next_ckpt_at = (step + 1) + interval_at(ckpt_schedule, step + 1)

        # Final flush: fire any pending PPO, then save
        sim_state = _maybe_fire_ppo(sim_state)
        jax.block_until_ready(sim_state.step)
        _save_checkpoint_jax(
            sim_state, out_dir, exp_name, seed, run_tag, ckpt_writer,
        )
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        jax_metrics.save(metrics_log, metrics_file)
    finally:
        # Block on any in-flight async write before returning. Surfaces
        # disk errors that the worker would otherwise swallow.
        if ckpt_writer is not None:
            ckpt_writer.close()

    elapsed = time.time() - start_time
    steps_done = total_steps - start_step
    print(f"\nDone. {steps_done} steps this invocation in {elapsed:.1f}s "
          f"({steps_done/elapsed:.1f} steps/s). Final step: {total_steps}")

    return sim_state


def _save_checkpoint_jax(sim_state, out_dir, exp_name, seed, run_tag="", writer=None):
    """Save full SimState and rotate old checkpoints.

    When `writer` is an AsyncCheckpointWriter, the host snapshot is taken
    synchronously and the disk write + rotation runs on a background
    thread. When `writer` is None, falls back to the inline sync path
    (preserves the old behavior under EVO_SYNC_CHECKPOINT=1).
    """
    step = int(sim_state.step)
    path = checkpoint_path(out_dir, exp_name, seed, step, run_tag)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if writer is not None:
        writer.submit(
            sim_state, path,
            rotate_dir=os.path.dirname(path), rotate_keep=CHECKPOINTS_TO_KEEP,
        )
    else:
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
    parser.add_argument("--run-tag", default=None,
                        help="Isolates this run's checkpoints, replays, metrics, and "
                             "progress under <out_dir>/<exp>/seed_<N>/<run_tag>/. "
                             "Default: auto-generate a UTC timestamp (e.g. "
                             "'2026-04-21T1447Z'). Pass '' for the legacy untagged layout.")
    parser.add_argument("--run-name", default=None,
                        help="Optional human-readable suffix appended to the auto-timestamp "
                             "run_tag, e.g. 'post-d19' → '2026-04-21T1447Z_post-d19'. "
                             "Ignored when --run-tag is given explicitly.")
    args = parser.parse_args()

    # Science config is the base; runtime overlays (runtime wins on conflict);
    # CLI flags win over runtime. Science configs should not define ops keys
    # anymore — but if they do, runtime still takes precedence.
    config = _load_yaml(args.config)
    runtime = _load_yaml(args.runtime)
    config.update(runtime)

    from src.config_utils import resolve_scale_dependent_params
    resolve_scale_dependent_params(config)

    if args.checkpoint_interval is not None:
        config["checkpoint_interval_steps"] = args.checkpoint_interval
    if args.log_interval is not None:
        config["log_interval_steps"] = args.log_interval

    run_tag = _resolve_run_tag(
        out_dir=args.out_dir,
        exp_name=config.get("experiment_name", "unnamed"),
        seed=args.seed,
        explicit=args.run_tag,
        run_name=args.run_name,
        resume=args.resume or args.resume_from is not None,
    )
    print(f"Run tag: {run_tag or '(legacy untagged)'}")

    run_experiment_jax(
        config, args.seed, args.max_steps, args.out_dir,
        resume=args.resume, resume_from=args.resume_from,
        run_tag=run_tag,
    )


def _resolve_run_tag(*, out_dir: str, exp_name: str, seed: int,
                     explicit: str | None, run_name: str | None,
                     resume: bool) -> str:
    """Pick the run_tag for this invocation.

    Priority:
      1. `--run-tag` explicit (including empty string → legacy layout)
      2. `--resume`: reuse the most recent existing run_tag for this
         (exp, seed); error if none exist.
      3. fresh: UTC timestamp, optionally suffixed with `--run-name`.
    """
    if explicit is not None:
        return explicit

    if resume:
        existing = list_run_tags(out_dir, exp_name, seed)
        if existing:
            return existing[-1]
        # Fall back to legacy untagged — older runs predate run_tag.
        return ""

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    return f"{ts}_{run_name}" if run_name else ts


if __name__ == "__main__":
    main()
