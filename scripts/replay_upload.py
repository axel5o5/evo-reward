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

# Optional default tag for uploaded replays, e.g. "post_d18_fix". An empty
# string yields the legacy layout with no tag segment.
DEFAULT_RUN_TAG = os.environ.get("EVO_REPLAY_RUN_TAG", "")

# Layouts:
#   legacy (untagged): gs://<bucket>/<exp>/seed_<N>/step_<start:08d>/{meta,frames}
#   tagged:            gs://<bucket>/<exp>/seed_<N>/<tag>/step_<start:08d>/{meta,frames}
# Tag must be a single path segment without slashes. See docs/emevo-diff.md D18
# for the motivating use case (pre- vs post-bugfix runs).
_PATH_RE = re.compile(
    r"^([^/]+)/seed_(\d+)/(?:([^/]+)/)?step_(\d+)/meta\.json$"
)


def _prefix(exp: str, seed: int, start_step: int, run_tag: str = "") -> str:
    """Build the GCS object-key prefix for a replay (no bucket, no trailing '/')."""
    step_part = f"step_{start_step:08d}"
    if run_tag:
        return f"{exp}/seed_{seed}/{run_tag}/{step_part}"
    return f"{exp}/seed_{seed}/{step_part}"


def _client():
    from google.cloud import storage
    return storage.Client(project=PROJECT_ID)


def _bucket(name: str | None = None):
    return _client().bucket(name or DEFAULT_BUCKET)


def upload_replay(local_dir: Path, exp: str, seed: int, start_step: int,
                  bucket: str | None = None,
                  run_tag: str | None = None) -> str:
    """Upload meta.json + frames.bin from `local_dir` to the bucket.

    If `run_tag` is None, DEFAULT_RUN_TAG (env var EVO_REPLAY_RUN_TAG) is used.
    An empty tag preserves the legacy untagged path layout.
    Returns remote prefix (without bucket scheme).
    """
    tag = DEFAULT_RUN_TAG if run_tag is None else run_tag
    b = _bucket(bucket)
    prefix = _prefix(exp, seed, start_step, tag)
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
    """Scan the bucket, return a ReplayRef per discovered meta.json.

    Tolerates both the legacy untagged layout and the tagged layout:
      <exp>/seed_<N>/step_<NNN>/meta.json
      <exp>/seed_<N>/<tag>/step_<NNN>/meta.json
    """
    b = _bucket(bucket)
    out: list[ReplayRef] = []
    for blob in b.list_blobs():
        m = _PATH_RE.match(blob.name)
        if not m:
            continue
        exp, seed_s, tag, start_s = m.group(1), m.group(2), m.group(3) or "", m.group(4)
        remote_path = _prefix(exp, int(seed_s), int(start_s), tag)
        frames = b.blob(f"{remote_path}/frames.bin")
        size = (blob.size or 0) + (frames.size or 0 if frames.exists() else 0)
        out.append(ReplayRef(
            exp=exp,
            seed=int(seed_s),
            start_step=int(start_s),
            size_bytes=size,
            remote_path=remote_path,
            run_tag=tag,
        ))
    out.sort(key=lambda r: (r.exp, r.seed, r.run_tag, r.start_step))
    return out


def delete_replay(exp: str, seed: int, start_step: int,
                  bucket: str | None = None,
                  run_tag: str = "") -> int:
    """Delete both blobs for a replay. Returns number of blobs removed."""
    b = _bucket(bucket)
    prefix = _prefix(exp, seed, start_step, run_tag)
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
            "run_tag": r.run_tag,
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
          dry_run: bool = False,
          exp: str | None = None, seed: int | None = None,
          run_tag: str | None = None) -> list[ReplayRef]:
    """Apply retention policy; delete losers from GCS. Returns refs that were
    (or would be) deleted. Index is rebuilt after a real delete.

    When `exp` / `seed` / `run_tag` are provided, the policy is applied only
    to that subset so one run's retention pass can't evict another run's
    milestones. `run_tag=""` matches legacy untagged replays explicitly.
    """
    refs = list_remote_replays(bucket)
    scoped = [
        r for r in refs
        if (exp is None or r.exp == exp)
        and (seed is None or r.seed == seed)
        and (run_tag is None or r.run_tag == run_tag)
    ]
    to_delete = apply_policy(policy, scoped, policy_config)
    if dry_run:
        return to_delete
    for r in to_delete:
        delete_replay(r.exp, r.seed, r.start_step, bucket=bucket, run_tag=r.run_tag)
    if to_delete:
        rebuild_index(bucket)
    return to_delete
