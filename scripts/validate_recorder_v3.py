"""validate_recorder_v3.py
-------------------------
Smoke check for the recorder v3 → dashboard pipeline.

Spins up an MLP-axis simulator at small scale, records a single replay
window using recorder v3, then prints the meta + decodes one genome row
to confirm structure. Output is dropped into
`dashboard/site/public/replays/test_axis1_mlp_v3/` so the dashboard can
load it via the regular index path.

Run from repo root with the conda env active:

    python scripts/validate_recorder_v3.py

Side effect: appends an entry for `test_axis1_mlp_v3` to
`dashboard/site/public/replays/index.json` so the local dashboard picks
it up. This entry is local-only — revert the index.json change (and
ignore the new test_axis1_mlp_v3/ directory) before committing.
"""
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.replay_recorder import ReplayRecorder
from src.jax_state import init_simstate
from src.jax_sim import build_sim_step
from src.environment import _build_physics


def main():
    repo_root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((repo_root / "configs/axis1_mlp_reward.yaml").read_text())
    runtime = yaml.safe_load((repo_root / "configs/runtime/mac.yaml").read_text())
    cfg.update(runtime)
    # Tiny world so the run completes in seconds.
    cfg["prey_initial"] = 12
    cfg["predator_initial"] = 3
    cfg["prey_cap"] = 24
    cfg["predator_cap"] = 6
    cfg["food_initial"] = 30
    cfg["food_max"] = 80
    cfg["world_size"] = 480
    # Quick window: 200-step interval, 100-step capture.
    cfg["replay_record_interval_steps"] = 200
    cfg["replay_record_length_steps"] = 100
    cfg["total_steps"] = 250
    cfg["seed"] = 0

    out_root = repo_root / "dashboard/site/public/replays/test_axis1_mlp_v3"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    state = init_simstate(cfg, jax.random.PRNGKey(cfg["seed"]))
    space, _ = _build_physics(cfg)
    sim_step, _ppo_update = build_sim_step(cfg, space)

    recorder = ReplayRecorder(
        cfg, "test_axis1_mlp_v3", seed=0, local_out_root=out_root,
    )
    print(f"recorder enabled={recorder.enabled} arch={recorder.genome_arch} dim={recorder.genome_dim}")

    n_steps = 220  # past the first 200-step boundary
    for step in range(1, n_steps + 1):
        state = sim_step(state)
        recorder.step(state, step)

    flushes = sorted(out_root.iterdir())
    if not flushes:
        raise SystemExit("No replay flushed — recorder window math failed.")

    out_dir = flushes[0]
    meta = json.loads((out_dir / "meta.json").read_text())
    print(f"\nflush dir: {out_dir.relative_to(repo_root)}")
    print(f"meta version: {meta['version']}")
    print(f"genome_arch:  {meta['genome_arch']}")
    print(f"genome_dim:   {meta['genome_dim']}")
    print(f"genome_shape: {meta['genome_shape']}")
    print(f"layout entries: {len(meta['genome_layout'])}")
    print(f"sections: {sorted(meta['sections'].keys())}")

    # Decode the genome rows + idmap and verify shape + a forward-pass sanity.
    bin_buf = (out_dir / "frames.bin").read_bytes()
    rows_sec = meta["sections"]["reward_genomes_byid"]
    idmap_sec = meta["sections"]["reward_genomes_idmap"]

    rows = np.frombuffer(
        bin_buf[rows_sec["offset"]: rows_sec["offset"] + rows_sec["length"]],
        dtype=np.float32,
    ).reshape(rows_sec["shape"])
    idmap = np.frombuffer(
        bin_buf[idmap_sec["offset"]: idmap_sec["offset"] + idmap_sec["length"]],
        dtype=np.int32,
    ).reshape(idmap_sec["shape"])

    print(f"\nn_unique_agents: {rows.shape[0]}")
    print(f"genome row 0 stats: mean={rows[0].mean():.4f} std={rows[0].std():.4f}")
    print(f"id range: {idmap.min()}..{idmap.max()}")

    # Build an index entry so the dashboard's index.json picks this up.
    index_path = repo_root / "dashboard/site/public/replays/index.json"
    n_frames = meta["n_frames"]
    bin_size = meta["frames_bin_size"]
    rel_path = f"test_axis1_mlp_v3/{out_dir.name}"
    new_entry = {
        "exp": "test_axis1_mlp_v3",
        "seed": 0,
        "start_step": meta["start_step"],
        "n_frames": n_frames,
        "path": rel_path,
        "size_bytes": bin_size,
    }
    if index_path.exists():
        idx = json.loads(index_path.read_text())
        # Drop any prior entries for this validation experiment so we don't
        # accumulate duplicates across reruns.
        idx["replays"] = [
            r for r in idx.get("replays", []) if r.get("exp") != "test_axis1_mlp_v3"
        ]
        idx["replays"].append(new_entry)
    else:
        idx = {"replays": [new_entry]}
    index_path.write_text(json.dumps(idx, indent=2))
    print(f"\nindex.json updated → {index_path.relative_to(repo_root)}")
    print(f"start dashboard and pick 'test_axis1_mlp_v3' from the replay selector.")


if __name__ == "__main__":
    main()
