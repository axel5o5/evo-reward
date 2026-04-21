"""replay_upload.py
-------------------
GCS-side operations for the public replays bucket.

Bucket layout mirrors the local layout:
  gs://<bucket>/<exp>/seed_<N>/step_<start:08d>/
      meta.json
      frames.bin

Top-level index at gs://<bucket>/index.json lists every replay currently in
the bucket. The dashboard fetches from this URL:
  https://storage.googleapis.com/<bucket>/index.json

Make the bucket public-read + CORS-enabled so the browser can fetch directly.
See scripts/setup_replays_bucket.sh (or docs/replays-bucket.md) for the one-time
gsutil commands.
"""
from __future__ import annotations

import io
import json
import os
import re
from dataclasses import asdict
from pathlib import Path

from scripts.replay_retention import ReplayRef, apply_policy


DEFAULT_BUCKET = os.environ.get("EVO_REWARD_REPLAYS_BUCKET", "evo-reward-replays-public")
PROJECT_ID = "evo-reward"

# gs://<bucket>/<exp>/seed_<N>/step_<start:08d>/{meta.json,frames.bin}
_PATH_RE = re.compile(r"^([^/]+)/seed_(\d+)/step_(\d+)/meta\.json$")


def _client():
    from google.cloud import storage
    return storage.Client(project=PROJECT_ID)


def _bucket(name: str | None = None):
    return _client().bucket(name or DEFAULT_BUCKET)


def upload_replay(local_dir: Path, exp: str, seed: int, start_step: int,
                  bucket: str | None = None) -> str:
    """Upload meta.json + frames.bin from `local_dir` to the bucket. Returns remote prefix."""
    b = _bucket(bucket)
    prefix = f"{exp}/seed_{seed}/step_{start_step:08d}"
    for name in ("meta.json", "frames.bin"):
        src = local_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing {src}")
        blob = b.blob(f"{prefix}/{name}")
        # Content-type helps the dashboard fetch; frames.bin is just bytes.
        blob.content_type = "application/json" if name.endswith(".json") else "application/octet-stream"
        # Short-ish cache so updates to the same path aren't sticky. Replays
        # are immutable by start_step, so a long TTL would also be fine.
        blob.cache_control = "public, max-age=300"
        blob.upload_from_filename(str(src))
    return prefix


def list_remote_replays(bucket: str | None = None) -> list[ReplayRef]:
    """Scan the bucket, return a ReplayRef per discovered meta.json."""
    b = _bucket(bucket)
    out: list[ReplayRef] = []
    for blob in b.list_blobs():
        m = _PATH_RE.match(blob.name)
        if not m:
            continue
        exp, seed_s, start_s = m.group(1), m.group(2), m.group(3)
        # Pair the meta with the sibling frames.bin to get the full size.
        frames = b.blob(f"{exp}/seed_{seed_s}/step_{start_s}/frames.bin")
        size = (blob.size or 0) + (frames.size or 0 if frames.exists() else 0)
        out.append(ReplayRef(
            exp=exp,
            seed=int(seed_s),
            start_step=int(start_s),
            size_bytes=size,
            remote_path=f"{exp}/seed_{seed_s}/step_{start_s}",
        ))
    out.sort(key=lambda r: (r.exp, r.seed, r.start_step))
    return out


def delete_replay(exp: str, seed: int, start_step: int,
                  bucket: str | None = None) -> int:
    """Delete both blobs for a replay. Returns number of blobs removed."""
    b = _bucket(bucket)
    prefix = f"{exp}/seed_{seed}/step_{start_step:08d}"
    removed = 0
    for name in ("meta.json", "frames.bin"):
        blob = b.blob(f"{prefix}/{name}")
        try:
            blob.delete()
            removed += 1
        except Exception:
            # Blob may already be gone; keep going.
            pass
    return removed


def rebuild_index(bucket: str | None = None,
                  index_path: str = "index.json") -> int:
    """Walk the bucket, rewrite index.json with every discovered replay.

    Called after any upload/delete. Browsers read this file via the public
    GCS URL; keep it small and deterministic.
    """
    refs = list_remote_replays(bucket)
    entries = []
    for r in refs:
        # Re-read each meta to pull n_frames for the listing.
        b = _bucket(bucket)
        meta_blob = b.blob(f"{r.remote_path}/meta.json")
        try:
            meta = json.loads(meta_blob.download_as_bytes())
        except Exception:
            continue
        entries.append({
            "exp": r.exp,
            "seed": r.seed,
            "start_step": r.start_step,
            "n_frames": meta.get("n_frames", 0),
            "path": r.remote_path,
            "size_bytes": r.size_bytes,
        })

    payload = json.dumps({"replays": entries}, indent=2).encode("utf-8")
    index_blob = _bucket(bucket).blob(index_path)
    index_blob.content_type = "application/json"
    # index.json is the live lookup — don't cache aggressively or new uploads
    # won't show up until the TTL expires.
    index_blob.cache_control = "no-cache"
    index_blob.upload_from_file(io.BytesIO(payload), size=len(payload))
    return len(entries)


def prune(policy: str, policy_config: dict, bucket: str | None = None,
          dry_run: bool = False) -> list[ReplayRef]:
    """Apply retention policy; delete losers from GCS. Returns refs that were
    (or would be) deleted. Index is rebuilt after a real delete.
    """
    refs = list_remote_replays(bucket)
    to_delete = apply_policy(policy, refs, policy_config)
    if dry_run:
        return to_delete
    for r in to_delete:
        delete_replay(r.exp, r.seed, r.start_step, bucket=bucket)
    if to_delete:
        rebuild_index(bucket)
    return to_delete
