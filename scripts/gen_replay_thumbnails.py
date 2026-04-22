#!/usr/bin/env python3
"""gen_replay_thumbnails.py — inline per-replay sparklines into index.json.

Walks every replay referenced by `dashboard/site/public/replays/index.json`,
decodes `alive` + `species` from its `frames.bin`, computes downsampled
prey/predator counts, and writes them back into the same index.json as an
optional `sparkline` field per entry. The ReplaySelector picks this up to
render inline SVG thumbnails next to each timeline dot.

Plan says "emit sparkline.json alongside the replay directory". We instead
inline into index.json so the selector doesn't need an extra HTTP fetch per
visible replay — the selector already loads index.json on mount. Graceful
degradation still holds: older index.json entries without `sparkline` just
render without a thumbnail.

Usage:
  python scripts/gen_replay_thumbnails.py
  python scripts/gen_replay_thumbnails.py --points 40  # fewer samples per replay
  python scripts/gen_replay_thumbnails.py --dry-run    # report what would change
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Target number of samples per sparkline. 60 gives a decent 60×18 inline SVG
# without bloating index.json. With uint8 counts that's ~120 bytes/replay
# before JSON overhead — cheap.
DEFAULT_POINTS = 60

REPO_ROOT = Path(__file__).resolve().parent.parent
REPLAYS_ROOT = REPO_ROOT / "dashboard" / "site" / "public" / "replays"


def decode_counts(replay_dir: Path) -> tuple[list[int], list[int], int] | None:
    """Return (prey_counts, pred_counts, n_frames) for one replay dir, or None on error."""
    meta_path = replay_dir / "meta.json"
    bin_path = replay_dir / "frames.bin"
    if not meta_path.exists() or not bin_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    sections = meta.get("sections", {})
    alive_sec = sections.get("alive")
    species_sec = sections.get("species")
    if alive_sec is None or species_sec is None:
        return None
    n_frames = meta["n_frames"]
    max_agents = meta["max_agents"]

    buf = bin_path.read_bytes()
    alive = np.frombuffer(
        buf, dtype=np.uint8, count=n_frames * max_agents, offset=alive_sec["offset"]
    ).reshape(n_frames, max_agents)
    species = np.frombuffer(
        buf, dtype=np.int32, count=max_agents, offset=species_sec["offset"]
    )

    is_pred = species == 1
    pred = (alive & is_pred[None, :].astype(np.uint8)).sum(axis=1).astype(int)
    prey = alive.sum(axis=1).astype(int) - pred
    return prey.tolist(), pred.tolist(), n_frames


def downsample(series: list[int], n_points: int) -> list[int]:
    """Evenly-spaced max over buckets. Max preserves LV peaks better than mean."""
    if len(series) <= n_points:
        return list(series)
    arr = np.asarray(series)
    # np.array_split handles uneven division without dropping tail samples.
    buckets = np.array_split(arr, n_points)
    return [int(b.max()) for b in buckets]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=DEFAULT_POINTS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--index",
        type=Path,
        default=REPLAYS_ROOT / "index.json",
        help="Path to index.json (default: dashboard/site/public/replays/index.json)",
    )
    args = ap.parse_args()

    if not args.index.exists():
        print(f"error: index not found at {args.index}", file=sys.stderr)
        return 1
    index = json.loads(args.index.read_text())
    replays = index.get("replays", [])
    if not replays:
        print("no replays listed in index; nothing to do")
        return 0

    changed = 0
    skipped = 0
    base = args.index.parent
    for entry in replays:
        rel = entry["path"].replace("\\", "/")
        replay_dir = base / rel
        decoded = decode_counts(replay_dir)
        if decoded is None:
            print(f"skip  {rel}  (meta/bin missing or malformed)")
            skipped += 1
            continue
        prey, pred, _ = decoded
        sparkline = {
            "prey": downsample(prey, args.points),
            "pred": downsample(pred, args.points),
        }
        if entry.get("sparkline") != sparkline:
            entry["sparkline"] = sparkline
            changed += 1
            print(f"ok    {rel}  ({args.points}pt)")
        else:
            print(f"same  {rel}")

    if args.dry_run:
        print(f"\ndry-run: {changed} would change, {skipped} skipped")
        return 0
    if changed:
        args.index.write_text(json.dumps(index, indent=2) + "\n")
        print(f"\nwrote {args.index} ({changed} changed, {skipped} skipped)")
    else:
        print(f"\nno changes ({skipped} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
