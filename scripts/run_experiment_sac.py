"""
run_experiment_sac.py
---------------------
Minimal training loop for an evo-reward run with SAC instead of PPO.
Parallel to scripts/run_experiment_jax.py — same config + runtime YAML
format, but uses src/jax_sim_sac.py for the sim step and runs one SAC
gradient update per env-step (no rollout-trigger threshold).

Usage:
    python scripts/run_experiment_sac.py \\
        --config configs/baseline/tiny_sac.yaml \\
        --seed 0 \\
        --max-steps 10000

Scope deviations from run_experiment_jax.py:
  - No `metrics.npz` time-series; only the basic progress.json that the
    dashboard monitor reads (step / population / sps / log_alpha mean).
  - No GCS upload sidecar — bring your own (the launcher script's
    gcs-sync tmux session works unchanged if the results layout matches).
  - --resume works for both states. Each checkpoint writes a pair:
        step_NNNNNNNN.npz       — SimState (PPO-compatible leaf layout)
        step_NNNNNNNN_sac.npz   — SacState (separate file because the
                                  schemas are independent)
    The resume path picks the latest step number that has BOTH files.

The results layout matches the PPO runner so existing tooling
(progress.json discovery, checkpoint browsing) continues to work:
    <out_dir>/<experiment_name>/seed_<seed>/<run_tag>/
        progress.json
        checkpoints/step_<NNNNNNNN>.npz
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.config_utils import resolve_scale_dependent_params
from src.jax_state import init_simstate
from src.sac_state import init_sacstate, save_sac_state, load_sac_state
from src.sac_runtime import build_sac_runtime
from src.jax_sim_sac import build_sim_step_sac
from src.environment import _build_physics
from scripts.replay_recorder import ReplayRecorder


DEFAULT_RUNTIME = _REPO_ROOT / "configs" / "runtime" / "default.yaml"


def _load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_run_tag(out_dir: str, exp: str, seed: int, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%MZ")


def _save_simstate_checkpoint(sim_state, path: Path) -> None:
    """Flatten SimState to leaf arrays and np.savez. Mirrors the PPO
    runner's layout so analysis/checkpoint_explorer.py can read these
    too — it rebuilds a template SimState via init_simstate to unflatten."""
    leaves = jtu.tree_leaves(sim_state)
    np.savez(path, **{f"leaf_{i}": np.asarray(l) for i, l in enumerate(leaves)})


def _load_simstate_checkpoint(path: Path, config: dict, seed: int):
    """Reverse of _save_simstate_checkpoint — build a fresh template
    via init_simstate, then unflatten the saved leaves."""
    template = init_simstate(config, jax.random.PRNGKey(seed))
    full_leaves, treedef = jtu.tree_flatten(template)
    data = np.load(str(path), allow_pickle=False)
    n = sum(1 for k in data.files if k.startswith("leaf_"))
    if n != len(full_leaves):
        raise ValueError(
            f"sim checkpoint {path}: leaf count {n} != template {len(full_leaves)}. "
            f"Config must match the run that produced it."
        )
    loaded = [jnp.asarray(data[f"leaf_{i}"]) for i in range(n)]
    return jtu.tree_unflatten(treedef, loaded)


def _find_latest_resume_pair(ckpt_dir: Path) -> tuple[int, Path, Path] | None:
    """Look in ckpt_dir for the highest-numbered step where both a
    step_NNNNNNNN.npz AND step_NNNNNNNN_sac.npz exist. Returns
    (step, sim_path, sac_path) or None."""
    if not ckpt_dir.is_dir():
        return None
    sims = {}
    sacs = {}
    for p in ckpt_dir.iterdir():
        name = p.name
        if not name.startswith("step_") or not name.endswith(".npz"):
            continue
        if name.endswith("_sac.npz"):
            try:
                step = int(name[len("step_"):-len("_sac.npz")])
            except ValueError:
                continue
            sacs[step] = p
        else:
            try:
                step = int(name[len("step_"):-len(".npz")])
            except ValueError:
                continue
            sims[step] = p
    paired = sorted(set(sims.keys()) & set(sacs.keys()))
    if not paired:
        return None
    s = paired[-1]
    return s, sims[s], sacs[s]


def main():
    ap = argparse.ArgumentParser(description="Run evo-reward SAC training")
    ap.add_argument("--config", required=True,
                    help="Path to YAML science config (must set learner_type: sac)")
    ap.add_argument("--runtime", default=str(DEFAULT_RUNTIME),
                    help=f"Runtime YAML (cadence, log levels). Default: {DEFAULT_RUNTIME}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Stop after this many env-steps (overrides config.total_steps)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--run-tag", default=None,
                    help="Subdir under <out_dir>/<exp>/seed_<N>/. Default: UTC timestamp.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the latest paired (sim, sac) checkpoint in "
                         "the seed's output dir.")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    runtime_cfg = _load_yaml(args.runtime)
    cfg.update(runtime_cfg)
    resolve_scale_dependent_params(cfg)

    if cfg.get("learner_type", "ppo") != "sac":
        print(f"warning: learner_type is {cfg.get('learner_type', 'ppo')!r}; "
              f"this runner ignores that and uses SAC unconditionally.",
              file=sys.stderr)

    total_steps = int(args.max_steps if args.max_steps is not None
                      else cfg["total_steps"])
    log_interval = int(cfg.get("log_interval_steps", 10_000))
    checkpoint_interval = int(cfg.get("checkpoint_interval_steps", 50_000))

    exp_name = cfg.get("experiment_name", "unnamed_sac")
    run_tag = _resolve_run_tag(args.out_dir, exp_name, args.seed, args.run_tag)
    run_root = Path(args.out_dir) / exp_name / f"seed_{args.seed}" / run_tag
    ckpt_dir = run_root / "checkpoints"
    run_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    progress_file = run_root / "progress.json"

    print(f"=== SAC run ===")
    print(f"  experiment: {exp_name}")
    print(f"  seed:       {args.seed}")
    print(f"  run_tag:    {run_tag}")
    print(f"  out:        {run_root}")
    print(f"  total_steps:{total_steps:,}")
    print(f"  log/ckpt:   every {log_interval:,} / {checkpoint_interval:,} steps")
    print()

    # ---- Build state + step + runtime ----
    start_step = 0
    resume_pair = _find_latest_resume_pair(ckpt_dir) if args.resume else None
    if resume_pair is not None:
        step_loaded, sim_path, sac_path = resume_pair
        print(f"Resuming from step {step_loaded:,}:")
        print(f"  sim → {sim_path.name}")
        print(f"  sac → {sac_path.name}")
        sim_state = _load_simstate_checkpoint(sim_path, cfg, args.seed)
        sac_state = load_sac_state(sac_path, cfg)
        start_step = int(sim_state.step)
    else:
        if args.resume:
            print("--resume requested but no paired (sim, sac) checkpoint found; "
                  "starting fresh.")
        print("Initializing SimState...")
        sim_state = init_simstate(cfg, jax.random.PRNGKey(args.seed))
        print("Initializing SacState (this allocates the per-agent replay ring)...")
        sac_state = init_sacstate(cfg, jax.random.PRNGKey(args.seed + 1))

    max_agents = cfg["prey_cap"] + cfg["predator_cap"]
    space, _ = _build_physics(cfg, n_agent_slots=max_agents)

    print("JIT-compiling sim_step_core_sac (first step will be slow)...")
    sim_step = build_sim_step_sac(cfg, space)
    runtime = build_sac_runtime(cfg)

    # ---- Replay recorder (dashboard-compatible) ----
    # sim_step_core_sac writes rollout_actions/rollout_ptrs each step
    # (passive, not used for learning) so the existing recorder can read
    # per-frame actions exactly like under PPO. Setting
    # replay_record_interval_steps=0 in the config disables it.
    replay_root = run_root / "replays"
    recorder = ReplayRecorder(
        cfg, exp_name, args.seed, replay_root,
        bucket=cfg.get("replay_bucket") or None,
        run_tag=run_tag,
    )
    if recorder.enabled:
        print(f"  replay recorder: every {recorder.schedule[0][1]:,} steps, "
              f"length {recorder.length:,} frames")
    else:
        print("  replay recorder: disabled (set replay_record_interval_steps > 0 to enable)")

    # ---- Main loop ----
    start_time = time.time()
    rng = jax.random.PRNGKey(args.seed + 2)

    def write_progress(step: int):
        is_active_np = np.asarray(sim_state.is_active)
        species_np = np.asarray(sim_state.species)
        elapsed = time.time() - start_time
        sps = (step + 1) / elapsed if elapsed > 0 else 0.0
        n_prey = int((is_active_np & (species_np == 0)).sum())
        n_pred = int((is_active_np & (species_np == 1)).sum())
        n_food = int(jnp.sum(sim_state.food_active))
        log_alpha = np.asarray(sac_state.log_alpha)
        active_mask = is_active_np
        active_alpha_mean = (float(np.exp(log_alpha[active_mask]).mean())
                             if active_mask.any() else 0.0)
        replay_sizes = np.asarray(sac_state.replay_size)
        replay_mean = float(replay_sizes[active_mask].mean()) if active_mask.any() else 0.0

        payload = {
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "experiment_name": exp_name,
            "seed": args.seed,
            "learner_type": "sac",
            "step": int(step + 1),
            "total_steps": int(total_steps),
            "sps": round(sps, 3),
            "elapsed_seconds": elapsed,
            "population": {
                "prey": n_prey,
                "pred": n_pred,
                "food": n_food,
            },
            "sac": {
                "alpha_mean_active": active_alpha_mean,
                "replay_size_mean_active": replay_mean,
                "replay_size_max": int(replay_sizes.max()),
            },
            # cum_* counters from SimState — exposed so the dashboard
            # progress card can show per-interval catches/deaths/feedings
            # like the PPO runner does.
            "events_cumulative": {
                "catches": int(sim_state.cum_catches),
                "deaths": int(sim_state.cum_deaths),
                "feedings": int(sim_state.cum_feedings),
                "next_agent_id": int(sim_state.next_agent_id),
            },
        }
        # Atomic write: tmp + replace, so the dashboard never reads a
        # partially-flushed file.
        tmp_path = progress_file.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, progress_file)

        print(
            f"Step {step+1:>8d}/{total_steps} | "
            f"prey={n_prey:>3d} pred={n_pred:>2d} food={n_food:>3d} | "
            f"α={active_alpha_mean:.3f} replay≈{replay_mean:.0f} | "
            f"{sps:.1f} sps | {elapsed:.0f}s"
        )

    # Initial progress so dashboard sees the run as alive even pre-warmup.
    write_progress(step=-1)

    for step in range(start_step, total_steps):
        rng, sub = jax.random.split(rng)
        sim_state, sac_state = sim_step(sim_state, sac_state)
        sac_state = runtime["step_update"](
            sac_state, sim_state.is_active, sim_state.ages, sim_state.species, sub,
        )

        # Replay capture / flush. No-op when disabled.
        recorder.step(sim_state, step_after=step + 1)

        if (step + 1) % log_interval == 0:
            write_progress(step)

        if (step + 1) % checkpoint_interval == 0:
            sim_ckpt = ckpt_dir / f"step_{step+1:08d}.npz"
            sac_ckpt = ckpt_dir / f"step_{step+1:08d}_sac.npz"
            print(f"  checkpoint → {sim_ckpt.name} + {sac_ckpt.name}")
            jax.block_until_ready(sim_state.step)
            # Write sim first, then sac. _find_latest_resume_pair only
            # picks step N when BOTH files exist, so a crash mid-write
            # leaves an unpaired sim_ckpt that gets ignored next run.
            _save_simstate_checkpoint(sim_state, sim_ckpt)
            save_sac_state(sac_state, sac_ckpt)

        # Hard stop if everyone died (avoids burning compute on dead runs).
        if not bool(jnp.any(sim_state.is_active)):
            print(f"[stop] all agents dead at step {step+1}, exiting")
            write_progress(step)
            break

    print(f"\n=== done. final step={int(sim_state.step):,} ===")


if __name__ == "__main__":
    main()
