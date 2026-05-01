"""archive_summary.py
--------------------
Walk the public replays bucket, summarize every run from its checkpoints,
and emit a compact archive that justifies later deletion.

For each `<exp>/seed_<N>/[<run_tag>/]step_<NNN>/` checkpoint we pull only the
`alive`, `species`, and `step_nums` slices from frames.bin (a few MB instead
of the full ~80MB) and compute population, birth/death, and extinction stats.
Per-run summaries roll up across checkpoints and land in
`archive/runs/<exp>__seed_<N>__<run_tag>.json` plus a top-level
`archive/SUMMARY.md` table for human reading.

Config-free: nothing here depends on prey/predator radius, mouth bins, or
energy capacity, so it works across the whole history without git
archaeology to recover the exact config a given run used.

Usage:
  python scripts/archive_summary.py --out archive/
  python scripts/archive_summary.py --exp baseline_faithful --out archive/
  python scripts/archive_summary.py --dry-run

Re-runs are incremental: per-run JSON files that already exist are skipped
unless --force is passed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DEFAULT_BUCKET = os.environ.get("EVO_REWARD_REPLAYS_BUCKET", "evo-reward-replays-public")

# Live runs that must NOT be touched while in progress. Each entry is a path
# prefix relative to the bucket root.
LIVE_RUN_PREFIXES = (
    "axis1_residual/seed_0/2026-05-01T1646Z/",
)

_DTYPE_SIZE = {"float32": 4, "int32": 4, "uint16": 2, "uint8": 1}
_NP_DTYPE = {"float32": np.float32, "int32": np.int32, "uint16": np.uint16, "uint8": np.uint8}

_PATH_RE = re.compile(
    r"^([^/]+)/seed_(\d+)/(?:([^/]+)/)?step_(\d+)/meta\.json$"
)


@dataclass(frozen=True)
class CheckpointRef:
    exp: str
    seed: int
    run_tag: str
    start_step: int
    prefix: str  # without bucket, without trailing slash, e.g. "exp/seed_0/tag/step_00010001"


# ─── GCS plumbing ─────────────────────────────────────────────────────────


def _client():
    from google.cloud import storage
    try:
        return storage.Client()
    except Exception:
        return storage.Client.create_anonymous_client()


def _is_live(prefix: str) -> bool:
    return any(prefix.startswith(p) for p in LIVE_RUN_PREFIXES)


def discover(bucket_name: str, exp_filter: str | None = None) -> list[CheckpointRef]:
    """List every checkpoint in the bucket. Skips live-run prefixes."""
    bucket = _client().bucket(bucket_name)
    refs: list[CheckpointRef] = []
    iter_prefix = f"{exp_filter}/" if exp_filter else None
    for blob in bucket.list_blobs(prefix=iter_prefix):
        m = _PATH_RE.match(blob.name)
        if not m:
            continue
        if _is_live(blob.name):
            continue
        exp, seed, run_tag, start = m.group(1), int(m.group(2)), (m.group(3) or ""), int(m.group(4))
        refs.append(CheckpointRef(
            exp=exp, seed=seed, run_tag=run_tag, start_step=start,
            prefix=blob.name[: -len("/meta.json")],
        ))
    return refs


def _section(meta: dict, raw: bytes, name: str) -> np.ndarray:
    s = meta["sections"][name]
    arr = np.frombuffer(raw, dtype=_NP_DTYPE[s["dtype"]])
    return arr.reshape(s["shape"])


def load_minimal(bucket_name: str, prefix: str) -> dict:
    """Download just the bytes we need: meta.json + the alive/species/step_nums slices."""
    bucket = _client().bucket(bucket_name)
    meta = json.loads(bucket.blob(f"{prefix}/meta.json").download_as_text())
    frames = bucket.blob(f"{prefix}/frames.bin")

    out = {"meta": meta}
    for name in ("alive", "species", "step_nums"):
        s = meta["sections"][name]
        # download_as_bytes uses inclusive end byte.
        raw = frames.download_as_bytes(start=s["offset"], end=s["offset"] + s["length"] - 1)
        out[name] = _section(meta, raw, name)
    return out


# ─── per-checkpoint + per-run summaries ───────────────────────────────────


def summarize_checkpoint(data: dict) -> dict:
    alive = data["alive"]            # (T, N) uint8
    species = data["species"]        # (N,) int32
    step_nums = data["step_nums"]    # (T,) int32
    n_frames = alive.shape[0]

    prey_slots = np.where(species == 0)[0]
    pred_slots = np.where(species == 1)[0]

    prey_pop = alive[:, prey_slots].sum(axis=1).astype(np.int32)
    pred_pop = alive[:, pred_slots].sum(axis=1).astype(np.int32)

    flip_on = (alive[:-1] == 0) & (alive[1:] == 1)
    flip_off = (alive[:-1] == 1) & (alive[1:] == 0)

    return {
        "start_step": int(step_nums[0]),
        "end_step": int(step_nums[-1]),
        "n_frames": int(n_frames),
        "prey_first": int(prey_pop[0]),  "prey_last": int(prey_pop[-1]),
        "prey_min":   int(prey_pop.min()), "prey_max": int(prey_pop.max()),
        "pred_first": int(pred_pop[0]),  "pred_last": int(pred_pop[-1]),
        "pred_min":   int(pred_pop.min()), "pred_max": int(pred_pop.max()),
        "births": int(flip_on.sum()),
        "deaths": int(flip_off.sum()),
    }


def summarize_run(checkpoints: list[dict]) -> dict:
    """checkpoints assumed sorted by start_step ascending."""
    if not checkpoints:
        return {
            "final_step": None, "n_checkpoints": 0,
            "extinct": False, "extinction_step": None, "extinct_species": "none",
            "peak_prey": 0, "peak_pred": 0,
        }

    final = checkpoints[-1]

    # A species counts as extinct iff its final-frame count is 0 in the LAST
    # checkpoint. The extinction_step is the start_step of the earliest
    # checkpoint where that species's last-frame count was 0 and stayed 0
    # through the rest of the run (so we don't latch on a transient zero).
    extinct_species = "none"
    extinction_step: int | None = None

    for sp in ("prey", "pred"):
        last_key = f"{sp}_last"
        if final[last_key] != 0:
            continue
        first_zero: int | None = None
        for ck in checkpoints:
            if ck[last_key] == 0 and first_zero is None:
                first_zero = ck["start_step"]
            elif ck[last_key] != 0:
                first_zero = None  # recovered, reset the search
        if first_zero is not None:
            extinct_species = sp
            extinction_step = first_zero
            break

    return {
        "final_step": int(final["end_step"]),
        "n_checkpoints": len(checkpoints),
        "extinct": extinct_species != "none",
        "extinction_step": extinction_step,
        "extinct_species": extinct_species,
        "peak_prey": max(c["prey_max"] for c in checkpoints),
        "peak_pred": max(c["pred_max"] for c in checkpoints),
    }


# ─── orchestration ────────────────────────────────────────────────────────


def _run_key(ref: CheckpointRef) -> tuple[str, int, str]:
    return (ref.exp, ref.seed, ref.run_tag)


def _run_filename(key: tuple[str, int, str]) -> str:
    exp, seed, tag = key
    tag_part = tag if tag else "untagged"
    return f"{exp}__seed_{seed}__{tag_part}.json"


def process_run(
    bucket_name: str,
    key: tuple[str, int, str],
    refs: list[CheckpointRef],
    workers: int,
) -> dict:
    """Download each checkpoint's minimal slices and produce a run record."""
    refs = sorted(refs, key=lambda r: r.start_step)
    checkpoints: list[dict] = [None] * len(refs)  # type: ignore[list-item]

    def _do(i: int, ref: CheckpointRef) -> tuple[int, dict]:
        data = load_minimal(bucket_name, ref.prefix)
        return i, summarize_checkpoint(data)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_do, i, r) for i, r in enumerate(refs)]
            for fut in as_completed(futs):
                i, summary = fut.result()
                checkpoints[i] = summary
    else:
        for i, r in enumerate(refs):
            _, summary = _do(i, r)
            checkpoints[i] = summary

    summary = summarize_run(checkpoints)
    exp, seed, tag = key
    return {
        "exp": exp,
        "seed": seed,
        "run_tag": tag,
        "summary": summary,
        "checkpoints": checkpoints,
    }


def write_summary_md(out_dir: Path, runs: list[dict]) -> None:
    rows = sorted(
        runs,
        key=lambda r: (r["exp"], r["seed"], r["run_tag"]),
    )
    lines = [
        "# Archive summary",
        "",
        "Per-run roll-up of population and extinction stats from public replays.",
        "Per-checkpoint detail lives in `runs/<exp>__seed_<N>__<tag>.json`.",
        "",
        "| exp | seed | run_tag | final_step | ckpts | extinct | extinct@ | species | peak_prey | peak_pred |",
        "|-----|------|---------|-----------:|------:|:-------:|---------:|:-------:|----------:|----------:|",
    ]
    for r in rows:
        s = r["summary"]
        ext = "yes" if s["extinct"] else "no"
        ext_step = f"{s['extinction_step']:,}" if s["extinction_step"] is not None else "—"
        final = f"{s['final_step']:,}" if s["final_step"] is not None else "—"
        tag = r["run_tag"] or "(untagged)"
        lines.append(
            f"| {r['exp']} | {r['seed']} | {tag} | {final} | "
            f"{s['n_checkpoints']} | {ext} | {ext_step} | "
            f"{s['extinct_species']} | {s['peak_prey']} | {s['peak_pred']} |"
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def write_summary_json(out_dir: Path, runs: list[dict]) -> None:
    """Compact roll-up the dashboard fetches. Mirrors the .md columns,
    plus the per-checkpoint trajectory so the panel can show a sparkline
    of prey/pred over the run without a second round-trip."""
    rows = []
    for r in sorted(runs, key=lambda r: (r["exp"], r["seed"], r["run_tag"])):
        s = r["summary"]
        rows.append({
            "exp": r["exp"],
            "seed": r["seed"],
            "run_tag": r["run_tag"],
            "final_step": s["final_step"],
            "n_checkpoints": s["n_checkpoints"],
            "extinct": s["extinct"],
            "extinction_step": s["extinction_step"],
            "extinct_species": s["extinct_species"],
            "peak_prey": s["peak_prey"],
            "peak_pred": s["peak_pred"],
            "trajectory": [
                {
                    "step": c["start_step"],
                    "prey": c["prey_last"],
                    "pred": c["pred_last"],
                }
                for c in r["checkpoints"]
            ],
        })
    (out_dir / "SUMMARY.json").write_text(json.dumps({"runs": rows}, indent=2))


# ─── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--exp", default=None, help="Limit to a single experiment folder.")
    parser.add_argument("--out", default="archive", help="Output directory.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel checkpoint downloads per run.")
    parser.add_argument("--force", action="store_true",
                        help="Re-process runs even if their JSON already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List discovered runs/checkpoints and exit.")
    parser.add_argument("--upload", action="store_true",
                        help="After computing, upload SUMMARY.json to "
                             "gs://<bucket>/summary.json so the dashboard can fetch it.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovering checkpoints in gs://{args.bucket}/" +
          (f"{args.exp}/" if args.exp else ""), file=sys.stderr)
    refs = discover(args.bucket, args.exp)

    by_run: dict[tuple[str, int, str], list[CheckpointRef]] = {}
    for r in refs:
        by_run.setdefault(_run_key(r), []).append(r)

    print(f"Found {len(refs)} checkpoints across {len(by_run)} runs.", file=sys.stderr)
    if args.dry_run:
        for key in sorted(by_run):
            exp, seed, tag = key
            n = len(by_run[key])
            tag_part = tag or "(untagged)"
            print(f"  {exp}/seed_{seed}/{tag_part}: {n} checkpoints", file=sys.stderr)
        return 0

    all_runs: list[dict] = []
    for i, key in enumerate(sorted(by_run), 1):
        target = runs_dir / _run_filename(key)
        if target.exists() and not args.force:
            all_runs.append(json.loads(target.read_text()))
            print(f"[{i}/{len(by_run)}] cached: {target.name}", file=sys.stderr)
            continue
        exp, seed, tag = key
        n = len(by_run[key])
        print(f"[{i}/{len(by_run)}] {exp}/seed_{seed}/{tag or '(untagged)'}: "
              f"{n} ckpts...", file=sys.stderr, end=" ", flush=True)
        record = process_run(args.bucket, key, by_run[key], args.workers)
        target.write_text(json.dumps(record, indent=2))
        all_runs.append(record)
        s = record["summary"]
        ext = f"extinct@{s['extinction_step']:,} ({s['extinct_species']})" if s["extinct"] else "alive"
        print(f"final={s['final_step']:,} {ext}", file=sys.stderr)

    write_summary_md(out_dir, all_runs)
    write_summary_json(out_dir, all_runs)
    print(f"Wrote {out_dir}/SUMMARY.md + SUMMARY.json ({len(all_runs)} runs).",
          file=sys.stderr)

    if args.upload:
        # Shell out to gsutil so we use the user's existing gcloud creds
        # rather than the read-only anonymous client _client() falls back to.
        target = f"gs://{args.bucket}/summary.json"
        src = str(out_dir / "SUMMARY.json")
        subprocess.run(
            ["gsutil", "-h", "Content-Type:application/json",
             "-h", "Cache-Control:public,max-age=60",
             "cp", src, target],
            check=True,
        )
        print(f"Uploaded → {target}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
