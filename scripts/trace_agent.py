"""trace_agent.py
-----------------
Post-hoc per-agent trace: load a checkpoint, run `sim_step_core` forward,
and record everything one specific agent did (or experienced) over the
next N steps.

Relies on the simulation being deterministic — SimState.rng_key drives
all randomness, so replaying from a saved state reproduces the original
trajectory bit-for-bit. We don't need to instrument training; we can
investigate any agent after the fact as long as we have a checkpoint
from before the question starts.

Usage:
  # By agent_id (preferred — stable identity across the agent's life).
  python scripts/trace_agent.py \
      --checkpoint results/.../checkpoints/step_00100000.npz \
      --config configs/baseline_faithful.yaml \
      --agent-id 257 --steps 5000 --out traces/

  # By slot (useful when you want to watch the same physical body as
  # it dies + is reused for a new agent).
  python scripts/trace_agent.py --checkpoint ... --slot 453 --steps 1000

  # Full obs (big — only for deep debugging a short window).
  python scripts/trace_agent.py --checkpoint ... --agent-id 42 --steps 500 \
      --include-obs

Design:
  * Per-step log records position, angle, velocity, energy, age, action,
    log_prob, value, reward, done, is_active, global cum_catches/feedings.
  * Action/reward/log_prob/value are read from the rollout buffer AFTER
    each step — `sim_step_core` just wrote them there at `rollout_ptrs - 1`
    for that slot. Zero code duplication; zero extra host-sync cost.
  * Trace stops early if the agent dies AND we're tracking by agent_id.
    Tracking by slot continues even after the body dies + is reborn
    (useful for slot-level analysis; the `agent_id` column will change).
  * Size: ~40 bytes/step without obs, ~860 bytes/step with obs.
    10,000 steps ≈ 400 KB thin / 8.6 MB with obs. Negligible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# XLA flags must be set before `import jax` — match the training runner.
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_enable_fast_math=true",
)

import jax
import jax.numpy as jnp

from src.environment import _build_physics
from src.jax_checkpoint import load_simstate
from src.jax_sim import build_sim_step
from src.jax_state import init_simstate


# ─── helpers ───────────────────────────────────────────────────────────────


def _find_slot_by_agent_id(sim_state, agent_id: int) -> int | None:
    """Return the slot holding the given agent_id, or None if not found
    (agent has already died or never existed in this checkpoint)."""
    ids = np.asarray(sim_state.agent_ids)
    active = np.asarray(sim_state.is_active)
    for i, (a, live) in enumerate(zip(ids, active)):
        if live and int(a) == agent_id:
            return i
    return None


def _read_rollout_at(sim_state, slot: int, ptr_offset: int = -1):
    """Read the entry at `rollout_ptrs[slot] + ptr_offset` (default: the
    just-written entry). Handles wrap-around cleanly."""
    rollout_steps = sim_state.rollout_obs.shape[1]
    ptr = int(sim_state.rollout_ptrs[slot])
    idx = (ptr + ptr_offset) % rollout_steps
    return {
        "obs": np.asarray(sim_state.rollout_obs[slot, idx]),
        "action": np.asarray(sim_state.rollout_actions[slot, idx]),
        "log_prob": float(sim_state.rollout_log_probs[slot, idx]),
        "value": float(sim_state.rollout_values[slot, idx]),
        "reward": float(sim_state.rollout_rewards[slot, idx]),
        "done": bool(sim_state.rollout_dones[slot, idx]),
    }


def _snapshot_agent(sim_state, slot: int, *, include_obs: bool) -> dict:
    """Capture one step's state for a slot — fields that are 'frozen' at
    the time of the call, not dependent on the rollout buffer (position,
    energy, etc.)."""
    circle = sim_state.phyjax_stated.get("circle")
    out = {
        "slot": slot,
        "agent_id": int(sim_state.agent_ids[slot]),
        "species": int(sim_state.species[slot]),
        "is_active": bool(sim_state.is_active[slot]),
        "position": np.asarray(circle.p.xy[slot]),
        "angle": float(circle.p.angle[slot]),
        "velocity_xy": np.asarray(circle.v.xy[slot]),
        "velocity_ang": float(circle.v.angle[slot]),
        "energy": float(sim_state.energies[slot]),
        "age": int(sim_state.ages[slot]),
        "cum_catches": int(sim_state.cum_catches),
        "cum_deaths": int(sim_state.cum_deaths),
        "cum_feedings": int(sim_state.cum_feedings),
    }
    rb = _read_rollout_at(sim_state, slot, ptr_offset=-1)
    out["reward"] = rb["reward"]
    out["log_prob"] = rb["log_prob"]
    out["value"] = rb["value"]
    out["done"] = rb["done"]
    out["action"] = rb["action"]
    if include_obs:
        out["obs"] = rb["obs"]
    return out


def _stack_trace(snapshots: list[dict], *, include_obs: bool) -> dict:
    """Turn list-of-dicts into dict-of-arrays suitable for np.savez."""
    keys = [
        "slot", "agent_id", "species", "is_active", "age",
        "angle", "velocity_ang", "energy",
        "reward", "log_prob", "value", "done",
        "cum_catches", "cum_deaths", "cum_feedings",
    ]
    arrays: dict[str, np.ndarray] = {}
    for k in keys:
        arrays[k] = np.array([s[k] for s in snapshots])
    arrays["position"] = np.stack([s["position"] for s in snapshots])
    arrays["velocity_xy"] = np.stack([s["velocity_xy"] for s in snapshots])
    arrays["action"] = np.stack([s["action"] for s in snapshots])
    if include_obs:
        arrays["obs"] = np.stack([s["obs"] for s in snapshots])
    return arrays


def _summarize(trace: dict, stop_reason: str) -> dict:
    """Human-readable synopsis of a trace."""
    n = len(trace["slot"])
    if n == 0:
        return {"n_steps": 0, "stop_reason": stop_reason}
    energy = trace["energy"]
    alive_count = int(trace["is_active"].sum())
    catches = int(trace["cum_catches"][-1] - trace["cum_catches"][0])
    feedings = int(trace["cum_feedings"][-1] - trace["cum_feedings"][0])
    return {
        "n_steps": int(n),
        "stop_reason": stop_reason,
        "steps_alive": alive_count,
        "species": "predator" if int(trace["species"][0]) == 1 else "prey",
        "agent_id_first": int(trace["agent_id"][0]),
        "agent_id_last": int(trace["agent_id"][-1]),
        "energy": {
            "first": float(energy[0]),
            "last": float(energy[-1]),
            "min": float(energy.min()),
            "mean": float(energy.mean()),
            "max": float(energy.max()),
        },
        "action_magnitude_mean": float(
            np.linalg.norm(trace["action"], axis=1).mean()
        ),
        "reward_mean": float(trace["reward"].mean()),
        "reward_sum": float(trace["reward"].sum()),
        "global_events_during_window": {
            "catches": catches,
            "feedings": feedings,
            "died_this_window": int((trace["is_active"] == False).sum() > 0),
        },
    }


# ─── core ──────────────────────────────────────────────────────────────────


def run_trace(
    checkpoint: str,
    config: dict,
    *,
    n_steps: int,
    agent_id: int | None,
    slot: int | None,
    include_obs: bool,
) -> tuple[dict, dict]:
    """Return (trace_arrays, summary)."""
    # Build a template SimState to load into.
    template = init_simstate(config, jax.random.PRNGKey(int(config.get("seed", 0))))
    sim_state = load_simstate(checkpoint, template)

    # Resolve target slot.
    if agent_id is not None and slot is not None:
        raise ValueError("Pass either --agent-id or --slot, not both.")
    if agent_id is not None:
        s = _find_slot_by_agent_id(sim_state, agent_id)
        if s is None:
            raise ValueError(
                f"agent_id {agent_id} is not in any active slot at "
                f"this checkpoint. It may have died before step "
                f"{int(sim_state.step)}."
            )
        slot = s
        track_mode = "agent_id"
    else:
        if slot is None:
            raise ValueError("Must pass --agent-id or --slot")
        track_mode = "slot"

    # Build the simulation step.
    space, _ = _build_physics(config)
    step_fn, _ = build_sim_step(config, space)

    # Capture the starting state before the first step (so the trace's
    # [0] entry reflects the checkpoint exactly, not the state after
    # one forward step).
    snapshots: list[dict] = []
    # First snapshot: no action/reward has happened yet at the checkpoint
    # boundary; the rollout buffer entry at `ptr-1` is whatever the
    # LAST step of the previous run wrote. We include it for continuity
    # but the `reward`/`action`/`done` fields at index 0 are from the
    # step that CREATED this checkpoint, not from this trace window.
    snapshots.append(_snapshot_agent(sim_state, slot, include_obs=include_obs))
    start_step = int(sim_state.step)
    start_agent_id = int(sim_state.agent_ids[slot])

    stop_reason = "max_steps"
    for _ in range(n_steps):
        sim_state = step_fn(sim_state)
        # Short-circuit if tracking by agent_id and the agent died.
        if track_mode == "agent_id":
            if not bool(sim_state.is_active[slot]):
                # Could have been caught (D20 path) or starved (process_
                # births_and_deaths path). Either way, this agent_id is
                # gone from the simulation — record one final snapshot
                # and stop.
                snapshots.append(_snapshot_agent(sim_state, slot, include_obs=include_obs))
                stop_reason = "agent_died"
                break
        snapshots.append(_snapshot_agent(sim_state, slot, include_obs=include_obs))

    jax.block_until_ready(sim_state.step)
    end_step = int(sim_state.step)

    trace = _stack_trace(snapshots, include_obs=include_obs)
    summary = _summarize(trace, stop_reason)
    summary.update({
        "checkpoint": checkpoint,
        "track_mode": track_mode,
        "start_sim_step": start_step,
        "end_sim_step": end_step,
        "slot": slot,
        "start_agent_id": start_agent_id,
    })
    return trace, summary


def main():
    parser = argparse.ArgumentParser(description="Per-agent post-hoc trace.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a saved SimState .npz.")
    parser.add_argument("--config", required=True,
                        help="Science config YAML (e.g. configs/baseline_faithful.yaml).")
    parser.add_argument("--runtime", default="configs/runtime/default.yaml",
                        help="Runtime config YAML.")
    parser.add_argument("--steps", type=int, default=5000,
                        help="How many sim_step_core calls to run (max).")
    parser.add_argument("--agent-id", type=int, default=None,
                        help="Track this specific agent_id. Stops early if it dies.")
    parser.add_argument("--slot", type=int, default=None,
                        help="Track a fixed slot. Continues even if the slot is "
                             "reused for a new agent.")
    parser.add_argument("--include-obs", action="store_true",
                        help="Also log the full observation vector per step. ~20× bigger.")
    parser.add_argument("--out", default="traces",
                        help="Output directory; file is named by agent_id/slot.")
    args = parser.parse_args()

    # Load configs.
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    with open(args.runtime) as f:
        cfg.update(yaml.safe_load(f) or {})

    trace, summary = run_trace(
        args.checkpoint, cfg,
        n_steps=args.steps,
        agent_id=args.agent_id,
        slot=args.slot,
        include_obs=args.include_obs,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"agent_{args.agent_id}" if args.agent_id is not None
           else f"slot_{args.slot}")
    stem = f"{Path(args.checkpoint).stem}_{tag}"
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.summary.json"

    np.savez_compressed(npz_path, **trace)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"wrote {npz_path}  ({npz_path.stat().st_size / 1e3:.1f} KB)")
    print(f"wrote {json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
