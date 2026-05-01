"""archive_prune.py
------------------
Apply the per-run retention plan to gs://evo-reward-replays-public/ and
free disk by dropping superseded checkpoints. The summary archive
(scripts/archive_summary.py output) is the documentation that justifies
this — anything pruned here is still represented in archive/SUMMARY.md
and the dashboard's Archived Runs panel.

Workflow:
  1. python scripts/archive_prune.py --dry-run     # preview deletions
  2. <review the plan>
  3. python scripts/archive_prune.py --execute     # actually delete + rebuild index
  4. python scripts/archive_summary.py --upload    # refresh summary.json

Per-run policies:
  keep_all        — leave every checkpoint (recent / showcase runs)
  keep_sparse_N   — thin to ~N evenly-spaced checkpoints (default N=10)
  keep_last_only  — drop everything except the most recent checkpoint
                    (so the final state is still scrubbable)

Edit ARCHIVE_POLICY below to change category assignments. Any run on the
bucket that isn't in ARCHIVE_POLICY (and isn't on EXCLUDED_PREFIXES) is
treated as an error — the script aborts rather than guess.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Iterable

from scripts.replay_retention import ReplayRef, apply_policy
from scripts.replay_upload import (
    DEFAULT_BUCKET,
    _PATH_RE,
    _prefix,
)


def _anon_list(bucket_name: str) -> list[ReplayRef]:
    """Like replay_upload.list_remote_replays() but uses an anonymous client
    so the dry-run path doesn't require ADC. Read-only — for execute we still
    rely on the auth'd client in delete_replay/rebuild_index."""
    from google.cloud import storage
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    out: list[ReplayRef] = []
    blobs_by_prefix: dict[str, dict] = {}
    for blob in bucket.list_blobs():
        m = _PATH_RE.match(blob.name)
        if m:
            exp, seed_s, tag, start_s = m.group(1), m.group(2), m.group(3) or "", m.group(4)
            remote_path = _prefix(exp, int(seed_s), int(start_s), tag)
            d = blobs_by_prefix.setdefault(remote_path, {"meta": 0, "frames": 0})
            d["meta"] = blob.size or 0
            d["exp"] = exp
            d["seed"] = int(seed_s)
            d["tag"] = tag
            d["start"] = int(start_s)
        elif blob.name.endswith("/frames.bin"):
            prefix = blob.name[: -len("/frames.bin")]
            d = blobs_by_prefix.setdefault(prefix, {"meta": 0, "frames": 0})
            d["frames"] = blob.size or 0
    for prefix, d in blobs_by_prefix.items():
        if "exp" not in d:
            continue  # frames.bin without meta — skip
        out.append(ReplayRef(
            exp=d["exp"], seed=d["seed"], start_step=d["start"],
            size_bytes=d["meta"] + d["frames"],
            remote_path=prefix, run_tag=d["tag"],
        ))
    out.sort(key=lambda r: (r.exp, r.seed, r.run_tag, r.start_step))
    return out


# Run identifiers we never touch — currently the live recording in progress.
# Each entry is a path prefix relative to the bucket root.
EXCLUDED_PREFIXES = (
    "axis1_residual/seed_0/2026-05-01T1800Z/",  # axis-1 v3 (live)
)


# Per-run policy assignments. Keys are (exp, seed, run_tag) tuples; tag
# may be empty string for legacy untagged runs.
#
# keep_all       — every checkpoint stays (recent / current-config runs)
# keep_sparse    — thin to ~10 evenly-spaced checkpoints
# keep_last_only — only the highest-numbered checkpoint stays
ARCHIVE_POLICY: dict[str, list[tuple[str, int, str]]] = {
    # Most recent / actively meaningful — full scrubbability.
    "keep_all": [
        ("axis2_aligned",         0, "2026-04-30T1806Z"),
        ("axis2_social_obs",      0, "2026-04-28_axis2_mouth_smol_1M"),
        ("axis2_social_obs",      1, "2026-04-29_axis2_mouth_smol_2M_seed1"),
        ("baseline_med_ddb",      0, "2026-04-30_baseline_med_ddb_2M"),
        ("baseline_med_ddb_ddm",  0, "2026-04-30_baseline_med_ddb_ddm_2M"),
        ("baseline_smol_ddb",     0, "2026-04-29_baseline_smol_ddb_2M"),
    ],
    # Long runs we still want to scrub through, but thinned to ~10 ckpts.
    "keep_sparse": [
        ("axis1_mlp_reward",      0, "2026-04-28_axis1_mouth_smol_1M"),
        ("axis1_residual",        0, "2026-05-01T1646Z"),  # T=4 v3 — diversity-loss case study
        ("exp_sweep_mouth_smol",  0, "2026-04-27_sweep_mouth_smol_1M"),
        ("exp_sweep_mouth_smol",  1, "2026-04-27_sweep_mouth_smol_1M_seed1"),
        ("exp_tune_eta_0.50",     0, "2026-04-25_tune_eta_050"),
        ("exp_tune_eta_0.55",     0, "2026-04-25_tune_eta_055"),
        ("exp_v8_no_cooldown",    0, "2026-04-23T1558Z_v8-no-cooldown-seed0"),
    ],
    # Historical / superseded — keep just the final checkpoint so the end
    # state is still playable, but drop the trajectory.
    "keep_last_only": [
        ("axis1_mlp_reward",      0, "2026-04-29_axis1_mouth_smol_1M_mut03"),
        ("axis1_mlp_reward",      0, "2026-04-28_axis1_mouth_smol_1M_mut08"),
        ("baseline_faithful",     0, "2026-04-21"),  # was legacy untagged
        ("baseline_faithful",     0, "2026-04-21T1935Z_post-d19"),
        ("baseline_faithful",     0, "2026-04-21T2159Z_phase1a-v2"),
        ("baseline_faithful",     0, "2026-04-21T2319Z_phase1a-v3"),
        ("baseline_faithful",     0, "2026-04-22T1417Z_phase1a-v4"),
        ("baseline_faithful",     0, "2026-04-22T1546Z_phase1a-v5"),
        ("baseline_faithful",     0, "2026-04-23T0400Z"),
        ("baseline_faithful",     0, "2026-04-23T1008Z"),
        ("baseline_faithful",     0, "2026-04-23_d19"),
        ("baseline_faithful",     0, "2026-04-24_d28a"),
        ("baseline_faithful",     0, "2026-04-24_d28b"),
        ("baseline_faithful",     0, "2026-04-24_d30"),
        ("baseline_faithful",     0, "2026-04-24_d31a"),
        ("baseline_faithful",     0, "2026-04-24_d31b"),
        ("baseline_faithful",     0, "2026-04-24_d31c"),
        ("baseline_faithful",     0, "2026-04-24_d31d"),
        ("baseline_faithful",     0, "2026-04-21_pre_d18_fix"),
        ("baseline_faithful",     1, "2026-04-22T2328Z_phase1a-v7-seed1-sensor120"),
        ("baseline_faithful",     1, "2026-04-24_d19"),
        ("exp_tune_eta_0.45",     0, "2026-04-24_tune_eta_045"),
    ],
}


SPARSE_N = 10  # evenly_spaced n_keep for the keep_sparse bucket


def _is_excluded(ref: ReplayRef) -> bool:
    return any(ref.remote_path.startswith(p.rstrip("/")) for p in EXCLUDED_PREFIXES)


def _policy_for(key: tuple[str, int, str]) -> str | None:
    for name, members in ARCHIVE_POLICY.items():
        if key in members:
            return name
    return None


def _format_size(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def plan(bucket: str) -> tuple[
    dict[tuple[str, int, str], list[ReplayRef]],   # by-run keep
    dict[tuple[str, int, str], list[ReplayRef]],   # by-run delete
    list[ReplayRef],                                 # excluded refs
    list[tuple[str, int, str]],                      # unmatched run keys
]:
    """Build the keep/delete plan without touching the bucket."""
    refs = _anon_list(bucket)

    by_run: dict[tuple[str, int, str], list[ReplayRef]] = defaultdict(list)
    excluded: list[ReplayRef] = []
    for r in refs:
        if _is_excluded(r):
            excluded.append(r)
            continue
        by_run[(r.exp, r.seed, r.run_tag)].append(r)

    keep: dict[tuple[str, int, str], list[ReplayRef]] = {}
    delete: dict[tuple[str, int, str], list[ReplayRef]] = {}
    unmatched: list[tuple[str, int, str]] = []

    for key, group in by_run.items():
        policy = _policy_for(key)
        if policy is None:
            unmatched.append(key)
            continue
        ordered = sorted(group, key=lambda r: r.start_step)
        if policy == "keep_all":
            to_delete: list[ReplayRef] = []
        elif policy == "keep_sparse":
            to_delete = apply_policy("evenly_spaced", ordered, {"n_keep": SPARSE_N})
        elif policy == "keep_last_only":
            to_delete = apply_policy("last_n", ordered, {"keep_last_n": 1})
        else:
            raise AssertionError(f"unknown policy: {policy}")
        delete[key] = to_delete
        keep[key] = [r for r in ordered if r not in to_delete]

    return keep, delete, excluded, unmatched


def _print_plan(
    keep: dict[tuple[str, int, str], list[ReplayRef]],
    delete: dict[tuple[str, int, str], list[ReplayRef]],
    excluded: list[ReplayRef],
    unmatched: list[tuple[str, int, str]],
) -> None:
    total_keep_bytes = 0
    total_del_bytes = 0
    print("\n=== prune plan ===\n", file=sys.stderr)
    for key in sorted(keep.keys()):
        exp, seed, tag = key
        kept = keep[key]
        gone = delete[key]
        kept_bytes = sum(r.size_bytes for r in kept)
        gone_bytes = sum(r.size_bytes for r in gone)
        total_keep_bytes += kept_bytes
        total_del_bytes += gone_bytes
        policy = _policy_for(key)
        tag_part = tag or "(untagged)"
        print(
            f"  {exp}/seed_{seed}/{tag_part}",
            f"[{policy}]",
            f"keep={len(kept)} ({_format_size(kept_bytes)})",
            f"delete={len(gone)} ({_format_size(gone_bytes)})",
            file=sys.stderr,
        )

    if excluded:
        ex_bytes = sum(r.size_bytes for r in excluded)
        print(f"\n  excluded (live runs, untouched): {len(excluded)} ckpts "
              f"({_format_size(ex_bytes)})", file=sys.stderr)

    print(
        f"\n  TOTAL keep:   {_format_size(total_keep_bytes)}\n"
        f"  TOTAL delete: {_format_size(total_del_bytes)}",
        file=sys.stderr,
    )

    if unmatched:
        print("\n  UNMATCHED runs (script will refuse to run):", file=sys.stderr)
        for key in unmatched:
            exp, seed, tag = key
            print(f"    {exp}/seed_{seed}/{tag or '(untagged)'}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True,
                   help="(default) Print the plan and exit without deleting.")
    g.add_argument("--execute", action="store_true",
                   help="Actually delete blobs + rebuild index.json.")
    args = parser.parse_args()

    print(f"Listing gs://{args.bucket}/ ...", file=sys.stderr)
    keep, delete, excluded, unmatched = plan(args.bucket)
    _print_plan(keep, delete, excluded, unmatched)

    if unmatched:
        print("\nERROR: unmatched runs above — add them to ARCHIVE_POLICY or "
              "EXCLUDED_PREFIXES before running.", file=sys.stderr)
        return 2

    if not args.execute:
        print("\nDry-run only. Re-run with --execute to apply.", file=sys.stderr)
        return 0

    paths_to_delete: list[str] = []
    for refs in delete.values():
        for r in refs:
            # Trailing /** so gsutil expands each line into the contained blobs;
            # gsutil's -I (stdin) mode doesn't apply -r per-line.
            paths_to_delete.append(f"gs://{args.bucket}/{r.remote_path}/**")

    if not paths_to_delete:
        print("\nNothing to delete.", file=sys.stderr)
        return 0

    print(f"\nExecuting: removing {len(paths_to_delete)} step dirs via "
          f"`gsutil -m rm -I` ...", file=sys.stderr)
    proc = subprocess.run(
        ["gsutil", "-m", "rm", "-I"],
        input="\n".join(paths_to_delete) + "\n",
        text=True, check=True,
    )
    print(f"  done. {len(paths_to_delete)} step dirs removed.", file=sys.stderr)

    print("Rebuilding index.json ...", file=sys.stderr)
    n = _rebuild_index_anon(args.bucket)
    print(f"  index.json now lists {n} replays.", file=sys.stderr)

    print("\nNext: run `python scripts/archive_summary.py --upload` to refresh "
          "the dashboard's archive panel.", file=sys.stderr)
    return 0


def _rebuild_index_anon(bucket_name: str) -> int:
    """Rebuild index.json the same way replay_upload.rebuild_index does, but
    read with an anonymous client (no ADC needed) and upload via gsutil."""
    from google.cloud import storage
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)

    # Re-list post-delete so the index reflects what's actually still there.
    refs = _anon_list(bucket_name)

    entries = []
    for r in refs:
        meta_blob = bucket.blob(f"{r.remote_path}/meta.json")
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

    payload = json.dumps({"replays": entries}, indent=2)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(payload)
        tmp_path = f.name
    subprocess.run(
        ["gsutil",
         "-h", "Content-Type:application/json",
         "-h", "Cache-Control:no-cache",
         "cp", tmp_path, f"gs://{bucket_name}/index.json"],
        check=True,
    )
    return len(entries)


if __name__ == "__main__":
    sys.exit(main())
