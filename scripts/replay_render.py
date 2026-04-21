"""replay_render.py
-------------------
Load a SimState checkpoint, re-run sim_step_core for N steps, and emit a
compact binary trajectory the React dashboard can scrub through.

Output layout (written under `dashboard/site/public/replays/`):
  <exp>/seed_<N>/step_<start:08d>/frames.bin   — concatenated float32/uint8 arrays
  <exp>/seed_<N>/step_<start:08d>/meta.json    — section offsets + static fields

Replay determinism note:
  Re-running sim_step_core with frozen policy params exactly reproduces the
  training trajectory until the next PPO update would fire (rollout_ptrs ≥
  rollout_steps). After that point, training would have updated the policy;
  our replay does not, so trajectories diverge. For any replay shorter than
  `rollout_steps` (default 1024) from the checkpointed rollout_ptrs, this
  divergence is zero — and for rendering ecosystem dynamics it doesn't matter
  even past that point (still a plausible forward roll from the same state).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_enable_fast_math=true --xla_cpu_use_thunk_runtime=true",
)

import jax
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import _build_physics
from src.jax_state import init_simstate
from src.jax_sim import build_sim_step
from src.jax_checkpoint import load_simstate


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PUBLIC_REPLAYS = REPO_ROOT / "dashboard" / "site" / "public" / "replays"


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def render_trajectory(
    config: dict,
    ckpt_path: str | Path,
    n_steps: int,
    seed_for_template: int = 0,
) -> dict:
    """Load checkpoint, step forward n_steps, return trajectory arrays + metadata."""
    max_agents = config["prey_cap"] + config["predator_cap"]
    food_max = config["food_max"]
    world_size = config["world_size"]

    space, _ = _build_physics(config)
    sim_step_core, _ = build_sim_step(config, space)

    template = init_simstate(config, jax.random.PRNGKey(seed_for_template))
    sim_state = load_simstate(str(ckpt_path), template)

    start_step = int(sim_state.step)
    print(f"loaded checkpoint at step {start_step}")

    # Pre-allocate numpy output buffers (avoids dict-of-list append in hot loop).
    pos = np.empty((n_steps, max_agents, 2), dtype=np.float32)
    angle = np.empty((n_steps, max_agents), dtype=np.float32)
    energy = np.empty((n_steps, max_agents), dtype=np.float32)
    alive = np.empty((n_steps, max_agents), dtype=np.uint8)
    food_pos = np.empty((n_steps, food_max, 2), dtype=np.float32)
    food_active = np.empty((n_steps, food_max), dtype=np.uint8)
    step_nums = np.empty((n_steps,), dtype=np.int32)

    # First call compiles — print progress separately so the user sees the wait.
    print(f"compiling sim_step_core…")
    t0 = time.time()
    sim_state = sim_step_core(sim_state)
    jax.block_until_ready(sim_state.step)
    print(f"  compiled in {time.time() - t0:.1f}s")

    # We already stepped once above to trigger JIT — capture that as frame 0
    # and run n_steps-1 more. This way n_steps is honored and frame 0 reflects
    # a one-tick advance from the raw checkpoint (matches how training sees it).
    def capture(t: int, st) -> None:
        circle = st.phyjax_stated.get("circle")
        pos[t] = np.asarray(circle.p.xy)
        angle[t] = np.asarray(circle.p.angle)
        energy[t] = np.asarray(st.energies)
        alive[t] = np.asarray(st.is_active, dtype=np.uint8)
        food_pos[t] = np.asarray(st.food_positions)
        food_active[t] = np.asarray(st.food_active, dtype=np.uint8)
        step_nums[t] = int(st.step)

    print(f"rolling {n_steps} steps…")
    t0 = time.time()
    capture(0, sim_state)
    for t in range(1, n_steps):
        sim_state = sim_step_core(sim_state)
        capture(t, sim_state)
    elapsed = time.time() - t0
    print(f"  rolled {n_steps} steps in {elapsed:.1f}s ({n_steps / max(elapsed, 1e-6):.1f} sps)")

    species = np.asarray(sim_state.species).astype(np.int32)
    radii = np.asarray(sim_state.radii).astype(np.float32)

    return {
        "pos": pos,
        "angle": angle,
        "energy": energy,
        "alive": alive,
        "food_pos": food_pos,
        "food_active": food_active,
        "step_nums": step_nums,
        "species": species,
        "radii": radii,
        "start_step": start_step,
        "world_size": float(world_size),
        "max_agents": int(max_agents),
        "food_max": int(food_max),
        "n_frames": int(n_steps),
    }


def write_trajectory(traj: dict, out_dir: Path) -> Path:
    """Serialize trajectory to frames.bin + meta.json. Returns the dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Order is the contract with JS; do not reorder without updating the loader.
    sections = [
        ("pos", traj["pos"], "float32"),
        ("angle", traj["angle"], "float32"),
        ("energy", traj["energy"], "float32"),
        ("alive", traj["alive"], "uint8"),
        ("food_pos", traj["food_pos"], "float32"),
        ("food_active", traj["food_active"], "uint8"),
        ("step_nums", traj["step_nums"], "int32"),
        ("species", traj["species"].astype(np.int32), "int32"),
        ("radii", traj["radii"].astype(np.float32), "float32"),
    ]

    bin_path = out_dir / "frames.bin"
    offsets = {}
    with open(bin_path, "wb") as f:
        for name, arr, dtype in sections:
            arr = np.ascontiguousarray(arr).astype(dtype, copy=False)
            buf = arr.tobytes()
            offsets[name] = {
                "offset": f.tell(),
                "length": len(buf),
                "dtype": dtype,
                "shape": list(arr.shape),
            }
            f.write(buf)

    meta = {
        "version": 1,
        "start_step": traj["start_step"],
        "n_frames": traj["n_frames"],
        "max_agents": traj["max_agents"],
        "food_max": traj["food_max"],
        "world_size": traj["world_size"],
        "sections": offsets,
        "frames_bin": "frames.bin",
        "frames_bin_size": bin_path.stat().st_size,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    size_mb = bin_path.stat().st_size / 1e6
    print(f"wrote {bin_path} ({size_mb:.1f} MB) + meta.json")
    return out_dir


def update_index(replays_root: Path) -> None:
    """Walk replays_root and rebuild the index.json the React page reads."""
    replays_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for meta_path in sorted(replays_root.rglob("meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        rel = meta_path.parent.relative_to(replays_root)
        parts = rel.parts
        if len(parts) < 3:
            continue
        exp = parts[0]
        seed = int(parts[1].removeprefix("seed_"))
        entries.append({
            "exp": exp,
            "seed": seed,
            "start_step": meta["start_step"],
            "n_frames": meta["n_frames"],
            "path": rel.as_posix(),
            "size_bytes": meta.get("frames_bin_size", 0),
        })
    index_path = replays_root / "index.json"
    index_path.write_text(json.dumps({"replays": entries}, indent=2))
    print(f"updated {index_path} ({len(entries)} replays)")


def main():
    ap = argparse.ArgumentParser(description="Render a replay trajectory from a SimState checkpoint.")
    ap.add_argument("--checkpoint", required=True, help="Path to step_*.npz")
    ap.add_argument("--config", default="configs/baseline_faithful.yaml",
                    help="Config YAML matching the run that produced the checkpoint")
    ap.add_argument("--steps", type=int, default=1000, help="Number of steps to render")
    ap.add_argument("--exp", required=True, help="Experiment name (for output path)")
    ap.add_argument("--seed", type=int, required=True, help="Seed (for output path)")
    ap.add_argument("--out-root", default=str(DEFAULT_PUBLIC_REPLAYS),
                    help=f"Replays root dir (default {DEFAULT_PUBLIC_REPLAYS})")
    args = ap.parse_args()

    config = load_config(args.config)
    traj = render_trajectory(config, args.checkpoint, args.steps)
    out_dir = Path(args.out_root) / args.exp / f"seed_{args.seed}" / f"step_{traj['start_step']:08d}"
    write_trajectory(traj, out_dir)
    update_index(Path(args.out_root))


if __name__ == "__main__":
    main()
