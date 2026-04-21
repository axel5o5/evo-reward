"""
jax_checkpoint.py
-----------------
Full-state SimState serialization for resumable training runs.

The legacy _save_checkpoint_jax in scripts/run_experiment_jax.py saved
only 8 of 27 SimState fields. Fine for post-hoc analysis but mid-run
resume was impossible — policy_params, optimizer state, rollout
buffers, and rng_key were all lost.

This module saves the entire SimState pytree by flattening it with
jax.tree_util (which walks through phyjax2d's pytree-registered types
cleanly, unlike flax.serialization's msgpack codec which chokes on
phyjax2d's StateDict). Leaves are stored as numpy arrays in an .npz;
the template supplied at load time reconstructs the original pytree
structure.

On-disk layout:
  <out_dir>/<experiment_name>/seed_<N>/checkpoints/step_<step:08d>.npz

The new format is distinguished from legacy partial checkpoints by the
presence of a signature key (`__evo_reward_ckpt_v1__`). Legacy files
without that signature are rejected with a clear error.
"""

import os
import re

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np


SIGNATURE_KEY = "__evo_reward_ckpt_v1__"

_STEP_RE = re.compile(r"step_(\d+)\.npz$")


def save_simstate(sim_state, path: str) -> None:
    """Serialize the full SimState pytree to `path` atomically.

    Uses jax.tree_util.tree_flatten to extract leaves (handles
    phyjax2d's StateDict and other registered pytree types that flax's
    msgpack codec can't serialize). Writes to `path + ".tmp"` first,
    then os.replace — a crash mid-write never leaves a truncated
    checkpoint in place.
    """
    leaves, _ = jtu.tree_flatten(sim_state)
    arrays = {f"leaf_{i}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
    arrays[SIGNATURE_KEY] = np.array(1, dtype=np.int32)

    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp_path, path)


def load_simstate(path: str, template):
    """Deserialize a checkpoint from `path` using `template` for structure.

    `template` must be a SimState produced by init_simstate(config, rng)
    with a config that matches the saved run — only the pytree structure
    and leaf shapes are used, the values are overwritten.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    data = np.load(path, allow_pickle=False)
    if SIGNATURE_KEY not in data.files:
        raise ValueError(
            f"Checkpoint {path!r} is not in the expected .msgpack format. "
            f"It may be a legacy partial checkpoint (pre-resume support). "
            f"Legacy .npz files only contain 8 of 27 SimState fields and "
            f"cannot be used to resume a run."
        )

    _, treedef = jtu.tree_flatten(template)
    n_leaves = sum(1 for k in data.files if k.startswith("leaf_"))
    jax_leaves = [jnp.asarray(data[f"leaf_{i}"]) for i in range(n_leaves)]
    return jtu.tree_unflatten(treedef, jax_leaves)


def find_latest_checkpoint(ckpt_dir: str):
    """Return path of the highest-step checkpoint in `ckpt_dir`, or None."""
    if not os.path.isdir(ckpt_dir):
        return None
    best_step = -1
    best_path = None
    for name in os.listdir(ckpt_dir):
        m = _STEP_RE.match(name)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best_path = os.path.join(ckpt_dir, name)
    return best_path


def rotate_checkpoints(ckpt_dir: str, keep: int = 3) -> None:
    """Delete all but the `keep` newest step_*.npz files in `ckpt_dir`."""
    if not os.path.isdir(ckpt_dir):
        return
    candidates = []
    for name in os.listdir(ckpt_dir):
        m = _STEP_RE.match(name)
        if m:
            candidates.append((int(m.group(1)), os.path.join(ckpt_dir, name)))
    candidates.sort(key=lambda x: x[0])
    for _, path in candidates[:-keep]:
        os.unlink(path)


def list_run_tags(out_dir: str, experiment_name: str, seed: int) -> list[str]:
    """List existing run_tag directories under <out_dir>/<exp>/seed_<N>/ that
    contain a checkpoints/ subdir. Returns sorted (lexicographic = time-order
    for ISO timestamps) list; empty if the seed dir doesn't exist or only
    holds a legacy untagged layout."""
    seed_dir = os.path.join(out_dir, experiment_name, f"seed_{seed}")
    if not os.path.isdir(seed_dir):
        return []
    tags = []
    for name in os.listdir(seed_dir):
        candidate = os.path.join(seed_dir, name, "checkpoints")
        if os.path.isdir(candidate):
            tags.append(name)
    return sorted(tags)


def run_dir(out_dir: str, experiment_name: str, seed: int, run_tag: str = "") -> str:
    """Directory root for everything written by a single run.

    Layout (with run_tag):    <out_dir>/<exp>/seed_<N>/<run_tag>/
    Layout (legacy, no tag):  <out_dir>/<exp>/seed_<N>/
    The legacy layout is kept readable for resuming older runs.
    """
    base = os.path.join(out_dir, experiment_name, f"seed_{seed}")
    return os.path.join(base, run_tag) if run_tag else base


def checkpoint_path(out_dir: str, experiment_name: str, seed: int, step: int,
                    run_tag: str = "") -> str:
    """Canonical path for a checkpoint at a given step."""
    ckpt_dir = os.path.join(run_dir(out_dir, experiment_name, seed, run_tag), "checkpoints")
    return os.path.join(ckpt_dir, f"step_{step:08d}.npz")
