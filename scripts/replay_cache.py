"""replay_cache.py
-----------------
Fetch checkpoints from gs://evo-reward-ckpts into a local LRU cache.

Cache layout mirrors the bucket layout:
  ~/.cache/evo-reward/checkpoints/<experiment>/seed_<N>/step_<step:08d>.npz

On each fetch we touch the file's atime so the eviction sweep picks the
oldest-accessed files first when cache size exceeds EVO_REWARD_CACHE_GB
(default 2 GB).
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


BUCKET = "evo-reward-ckpts"
PROJECT_ID = "evo-reward"
STEP_RE = re.compile(r"step_(\d+)\.npz$")
RESULT_PREFIX = "results/"


def cache_root() -> Path:
    root = os.environ.get("EVO_REWARD_CACHE_DIR")
    if root:
        return Path(root).expanduser()
    return Path.home() / ".cache" / "evo-reward" / "checkpoints"


def cache_cap_bytes() -> int:
    gb = float(os.environ.get("EVO_REWARD_CACHE_GB", "2.0"))
    return int(gb * 1e9)


def cache_path(exp: str, seed: int, step: int) -> Path:
    return cache_root() / exp / f"seed_{seed}" / f"step_{step:08d}.npz"


def gcs_key(exp: str, seed: int, step: int) -> str:
    return f"{RESULT_PREFIX}{exp}/seed_{seed}/checkpoints/step_{step:08d}.npz"


@dataclass
class CheckpointRef:
    exp: str
    seed: int
    step: int
    size_bytes: int
    updated_iso: str
    gcs_key: str


def _client():
    from google.cloud import storage
    return storage.Client(project=PROJECT_ID)


def list_checkpoints(exp: str | None = None, seed: int | None = None) -> list[CheckpointRef]:
    """Enumerate step_*.npz blobs in GCS. Filter by exp/seed if given."""
    client = _client()
    b = client.bucket(BUCKET)
    prefix = RESULT_PREFIX if exp is None else f"{RESULT_PREFIX}{exp}/"
    if exp is not None and seed is not None:
        prefix = f"{RESULT_PREFIX}{exp}/seed_{seed}/checkpoints/"

    out: list[CheckpointRef] = []
    for blob in b.list_blobs(prefix=prefix):
        m = STEP_RE.search(blob.name)
        if not m:
            continue
        parts = blob.name.split("/")
        try:
            i = parts.index("results")
            b_exp = parts[i + 1]
            b_seed = int(parts[i + 2].removeprefix("seed_"))
        except (ValueError, IndexError):
            continue
        if exp is not None and b_exp != exp:
            continue
        if seed is not None and b_seed != seed:
            continue
        out.append(CheckpointRef(
            exp=b_exp,
            seed=b_seed,
            step=int(m.group(1)),
            size_bytes=blob.size or 0,
            updated_iso=blob.updated.isoformat(timespec="seconds") if blob.updated else "",
            gcs_key=blob.name,
        ))
    out.sort(key=lambda r: (r.exp, r.seed, r.step))
    return out


def resolve_step(exp: str, seed: int, step: int | str) -> CheckpointRef:
    """Turn a step spec ('latest' or int) into a concrete CheckpointRef."""
    refs = list_checkpoints(exp, seed)
    if not refs:
        raise FileNotFoundError(f"No checkpoints for {exp}/seed_{seed} in gs://{BUCKET}/")
    if step == "latest":
        return refs[-1]
    step_i = int(step)
    for r in refs:
        if r.step == step_i:
            return r
    raise FileNotFoundError(
        f"No checkpoint at step {step_i} for {exp}/seed_{seed}. "
        f"Available: {[r.step for r in refs]}"
    )


def touch_atime(path: Path) -> None:
    """Refresh atime (for LRU) while preserving mtime."""
    import time
    try:
        st = path.stat()
    except FileNotFoundError:
        return
    os.utime(path, (time.time(), st.st_mtime))


def cache_size_bytes(root: Path | None = None) -> int:
    root = root or cache_root()
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*.npz"))


def evict_lru(cap_bytes: int | None = None, root: Path | None = None) -> int:
    """Delete oldest-atime .npz files until total size ≤ cap. Returns bytes freed."""
    cap = cap_bytes if cap_bytes is not None else cache_cap_bytes()
    root = root or cache_root()
    if not root.exists():
        return 0
    files = [(p.stat().st_atime, p.stat().st_size, p) for p in root.rglob("*.npz")]
    total = sum(s for _, s, _ in files)
    if total <= cap:
        return 0
    files.sort(key=lambda t: t[0])  # oldest atime first
    freed = 0
    for _atime, size, p in files:
        if total - freed <= cap:
            break
        try:
            p.unlink()
            freed += size
        except FileNotFoundError:
            pass
    return freed


def fetch_checkpoint(exp: str, seed: int, step: int | str = "latest",
                     force: bool = False) -> Path:
    """Download checkpoint to cache (no-op if already cached). Returns local path."""
    ref = resolve_step(exp, seed, step)
    dst = cache_path(ref.exp, ref.seed, ref.step)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not force:
        touch_atime(dst)
        return dst

    client = _client()
    blob = client.bucket(BUCKET).blob(ref.gcs_key)
    tmp = dst.with_suffix(".npz.tmp")
    print(f"  fetching gs://{BUCKET}/{ref.gcs_key} → {dst}")
    blob.download_to_filename(str(tmp))
    os.replace(tmp, dst)
    touch_atime(dst)

    freed = evict_lru()
    if freed > 0:
        print(f"  cache LRU-evicted {freed / 1e6:.1f} MB")
    return dst


def clean_cache(root: Path | None = None) -> int:
    """Delete everything in the cache root. Returns bytes freed."""
    root = root or cache_root()
    if not root.exists():
        return 0
    freed = cache_size_bytes(root)
    for p in root.rglob("*.npz"):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    return freed


if __name__ == "__main__":
    print(f"cache root: {cache_root()}")
    print(f"cache size: {cache_size_bytes() / 1e6:.1f} MB / {cache_cap_bytes() / 1e9:.1f} GB cap")
    sys.exit(0)
