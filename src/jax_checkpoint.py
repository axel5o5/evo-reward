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
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

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


def _write_npz_atomic(
    arrays: dict, path: str, rotate_dir: Optional[str], rotate_keep: int,
) -> None:
    """Worker-thread payload: compress + atomic rename + rotation.

    Mirrors the on-disk contract of save_simstate exactly — same .tmp +
    os.replace, same rotation semantics. Crucially this touches no JAX
    objects: by the time it runs, `arrays` is already a dict of pure
    numpy arrays in CPU memory.
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp_path, path)
    if rotate_dir is not None:
        rotate_checkpoints(rotate_dir, keep=rotate_keep)


class AsyncCheckpointWriter:
    """Background-thread checkpoint writer that overlaps disk I/O with GPU compute.

    The expensive part of save_simstate is np.savez_compressed: ~500MB
    through GCE persistent disk takes seconds, and the runtime profile
    flags it as ~5-7% of wall clock at the configured 10k-step cadence.

    The host snapshot itself (np.asarray on each pytree leaf) is the
    GPU→host barrier and stays on the caller thread — JAX device buffers
    must not be touched off the main thread, and once asarray returns
    we have an independent numpy copy that's safe to hand off.

    Snapshot semantics: by the time submit() returns, the saved state is
    decoupled from sim_state. The simulation loop is free to mutate
    sim_state immediately. Test coverage in test_checkpoint_jax.py
    asserts this property (test_async_snapshot_decoupled).

    Concurrency: max_workers=1 with explicit wait-on-previous before
    queuing the next submit. Bounds memory at one in-flight 500MB
    snapshot and applies natural backpressure if disk gets slow rather
    than letting writes pile up.

    Crash safety: identical to the sync path. SIGTERM mid-write leaves a
    .tmp file (ignored by find_latest_checkpoint, which only matches
    step_*.npz), and the previous successful checkpoint is intact.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ckpt-writer"
        )
        self._pending: Optional[Future] = None

    def submit(
        self,
        sim_state,
        path: str,
        rotate_dir: Optional[str] = None,
        rotate_keep: int = 3,
    ) -> None:
        """Snapshot to host arrays sync, then queue the disk write.

        Blocks if a previous write is still in flight — keeps at most
        one outstanding write so we never accumulate queued snapshots.
        """
        # Sync GPU→host transfer. After this loop, `arrays` is pure CPU
        # memory and JAX is free to mutate or donate the device buffers.
        leaves, _ = jtu.tree_flatten(sim_state)
        arrays = {f"leaf_{i}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
        arrays[SIGNATURE_KEY] = np.array(1, dtype=np.int32)

        # Wait for the previous write before queuing the next. .result()
        # surfaces any exception raised on the worker thread.
        if self._pending is not None:
            self._pending.result()

        self._pending = self._executor.submit(
            _write_npz_atomic, arrays, path, rotate_dir, rotate_keep,
        )

    def wait(self) -> None:
        """Block until any in-flight write completes."""
        if self._pending is not None:
            self._pending.result()
            self._pending = None

    def close(self) -> None:
        """Flush pending write and shut down the worker thread."""
        try:
            self.wait()
        finally:
            self._executor.shutdown(wait=True)


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
