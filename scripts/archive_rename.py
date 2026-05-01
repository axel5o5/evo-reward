"""archive_rename.py
-------------------
Prefix every non-date-prefixed run_tag in gs://evo-reward-replays-public/
with the run's ISO start date so paths sort chronologically and the
"old vs new" question is answerable from the path alone. The tagged
hierarchy (`<exp>/seed_<N>/<run_tag>/`) stays — we just normalize the
run_tag value.

Examples:
  exp_tune_eta_0.45/seed_0/tune_eta_045/         → exp_tune_eta_0.45/seed_0/2026-04-24_tune_eta_045/
  baseline_faithful/seed_0/d19/                  → baseline_faithful/seed_0/2026-04-19_d19/
  baseline_faithful/seed_0/<step_NNN>/ (untagged) → baseline_faithful/seed_0/<date>/<step_NNN>/

Date is derived from the earliest meta.json blob's GCS create_time. Runs
already date-prefixed (`2026-04-21T2159Z_phase1a-v2`, `2026-04-30T1806Z`,
etc.) are left alone.

Side effects:
  - GCS rename via `gsutil -m mv -r` per run.
  - Local `archive/runs/<exp>__seed_<N>__<old_tag>.json` files are
    renamed to match the new tag, with the embedded `run_tag` field
    rewritten so the panel reads the new name.
  - ARCHIVE_POLICY in scripts/archive_prune.py needs a manual update
    to the new tag names — the script prints the substitutions to do.
  - index.json + summary.json have to be rebuilt + re-uploaded after.

Run --dry-run first. There is no undo for a rename: the old paths are
deleted as part of `gsutil mv`.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from scripts.replay_upload import _PATH_RE


DEFAULT_BUCKET = "evo-reward-replays-public"

# Skip these runs entirely (currently-running training).
EXCLUDED_PREFIXES = (
    "axis1_residual/seed_0/2026-05-01T1800Z/",
)

# A run_tag is "already date-prefixed" if it starts with YYYY-MM-DD.
_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _is_excluded(exp: str, seed: int, run_tag: str) -> bool:
    p = f"{exp}/seed_{seed}/" + (f"{run_tag}/" if run_tag else "")
    return any(p.startswith(e) for e in EXCLUDED_PREFIXES)


def _needs_rename(run_tag: str) -> bool:
    if not run_tag:
        return True  # untagged → give it a date
    return not _DATE_PREFIX_RE.match(run_tag)


def _new_tag(iso_date: str, old_tag: str) -> str:
    return iso_date if not old_tag else f"{iso_date}_{old_tag}"


def discover_renames(bucket_name: str) -> list[tuple[str, int, str, str]]:
    """Return [(exp, seed, old_tag, new_tag), ...] sorted by old path."""
    from google.cloud import storage
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)

    earliest: dict[tuple[str, int, str], datetime] = {}
    for blob in bucket.list_blobs():
        m = _PATH_RE.match(blob.name)
        if not m:
            continue
        exp = m.group(1)
        seed = int(m.group(2))
        tag = m.group(3) or ""
        key = (exp, seed, tag)
        t = blob.time_created
        if key not in earliest or t < earliest[key]:
            earliest[key] = t

    plan = []
    for key in sorted(earliest):
        exp, seed, old_tag = key
        if _is_excluded(exp, seed, old_tag):
            continue
        if not _needs_rename(old_tag):
            continue
        iso = earliest[key].strftime("%Y-%m-%d")
        plan.append((exp, seed, old_tag, _new_tag(iso, old_tag)))
    return plan


def _print_plan(plan: list[tuple[str, int, str, str]], bucket: str) -> None:
    if not plan:
        print("\n  Nothing to rename — every run is already date-prefixed.",
              file=sys.stderr)
        return
    print(f"\n=== rename plan ({len(plan)} runs) ===\n", file=sys.stderr)
    for exp, seed, old_tag, new_tag in plan:
        old = f"gs://{bucket}/{exp}/seed_{seed}/" + (f"{old_tag}/" if old_tag else "")
        new = f"gs://{bucket}/{exp}/seed_{seed}/{new_tag}/"
        print(f"  {old}\n    → {new}", file=sys.stderr)


def _gsutil_mv_run(bucket: str, exp: str, seed: int, old_tag: str, new_tag: str) -> None:
    """Move every step_NNN dir under the old run path to the new path."""
    base = f"gs://{bucket}/{exp}/seed_{seed}"
    if old_tag:
        # Tagged → tagged: gsutil mv -r handles the directory rename.
        src = f"{base}/{old_tag}"
        dst = f"{base}/{new_tag}"
        subprocess.run(["gsutil", "-m", "mv", "-r", src, dst], check=True)
    else:
        # Untagged → tagged: source is the seed_N dir itself, but we only
        # want the step_* subdirs (not anything else that might live there).
        # Move them into a new tag dir one wildcard at a time.
        src = f"{base}/step_*"
        dst = f"{base}/{new_tag}/"
        subprocess.run(["gsutil", "-m", "mv", "-r", src, dst], check=True)


def _rename_archive_files(plan: list[tuple[str, int, str, str]],
                          archive_runs_dir: Path) -> list[tuple[str, str]]:
    """Rename local archive/runs/<key>.json files + rewrite their run_tag.
    Returns [(old_basename, new_basename), ...]."""
    renamed = []
    for exp, seed, old_tag, new_tag in plan:
        old_name = f"{exp}__seed_{seed}__{old_tag or 'untagged'}.json"
        new_name = f"{exp}__seed_{seed}__{new_tag}.json"
        old_path = archive_runs_dir / old_name
        new_path = archive_runs_dir / new_name
        if not old_path.exists():
            print(f"  WARN: {old_path} not found; skipping local rename", file=sys.stderr)
            continue
        record = json.loads(old_path.read_text())
        record["run_tag"] = new_tag
        new_path.write_text(json.dumps(record, indent=2))
        old_path.unlink()
        renamed.append((old_name, new_name))
    return renamed


def _print_policy_substitutions(plan: list[tuple[str, int, str, str]]) -> None:
    """Show sed-style substitutions for archive_prune.py's ARCHIVE_POLICY."""
    if not plan:
        return
    print("\n=== update scripts/archive_prune.py ARCHIVE_POLICY ===\n", file=sys.stderr)
    for exp, seed, old_tag, new_tag in plan:
        if old_tag:
            print(
                f'  ("{exp}", {seed}, "{old_tag}")  →  '
                f'("{exp}", {seed}, "{new_tag}")',
                file=sys.stderr,
            )
        else:
            print(
                f'  ("{exp}", {seed}, "")  →  '
                f'("{exp}", {seed}, "{new_tag}")',
                file=sys.stderr,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True,
                   help="(default) Print the plan and exit.")
    g.add_argument("--execute", action="store_true",
                   help="Actually move blobs + rename archive files.")
    args = parser.parse_args()

    print(f"Listing gs://{args.bucket}/ ...", file=sys.stderr)
    plan = discover_renames(args.bucket)
    _print_plan(plan, args.bucket)

    if not args.execute:
        if plan:
            _print_policy_substitutions(plan)
            print(f"\nDry-run only. Re-run with --execute to apply.", file=sys.stderr)
        return 0

    if not plan:
        return 0

    print(f"\nExecuting {len(plan)} renames via `gsutil -m mv -r` ...", file=sys.stderr)
    for i, (exp, seed, old_tag, new_tag) in enumerate(plan, 1):
        print(f"  [{i}/{len(plan)}] {exp}/seed_{seed}/{old_tag or '(untagged)'} "
              f"→ {new_tag}", file=sys.stderr)
        _gsutil_mv_run(args.bucket, exp, seed, old_tag, new_tag)

    print("\nRenaming local archive/runs/*.json ...", file=sys.stderr)
    renamed = _rename_archive_files(plan, Path("archive/runs"))
    for old, new in renamed:
        print(f"  {old} → {new}", file=sys.stderr)

    _print_policy_substitutions(plan)
    print("\nNext steps:", file=sys.stderr)
    print("  1. Apply ARCHIVE_POLICY substitutions above to scripts/archive_prune.py",
          file=sys.stderr)
    print("  2. Run scripts/replay_upload.py rebuild-index — or use the rebuild step",
          file=sys.stderr)
    print("  3. Run `python scripts/archive_summary.py --upload` to refresh summary.json",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
