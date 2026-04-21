"""replay_retention.py
----------------------
Retention policies for replay files. A policy is a pure function

    policy(refs: list[ReplayRef], **config) -> list[ReplayRef]  # to delete

Given the current set of replays (sorted or not — policies re-sort as needed),
it returns the subset that should be deleted. Nothing is actually removed here;
the upload/CLI layer does that based on the return value.

Adding a new policy: write the function, then register it in POLICIES below.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ReplayRef:
    """Minimal view of a replay for retention decisions."""
    exp: str
    seed: int
    start_step: int
    # Optional fields the caller can fill in — policies should tolerate their absence.
    size_bytes: int = 0
    remote_path: str = ""
    # Run tag — inserted between seed_N and step_NNN when present, e.g.
    # "pre_d18". Empty string = legacy untagged path (bucket root layout).
    run_tag: str = ""


# ----------------------------- policies ---------------------------------------


def policy_last_n(refs: list[ReplayRef], *, keep_last_n: int = 10, **_) -> list[ReplayRef]:
    """Keep the N replays with the highest start_step. Delete the rest."""
    ordered = sorted(refs, key=lambda r: r.start_step)
    return ordered[:-keep_last_n] if len(ordered) > keep_last_n else []


def policy_milestones(
    refs: list[ReplayRef],
    *,
    keep_last_n: int = 10,
    keep_at_steps: list[int] | None = None,
    tolerance: int = 0,
    **_,
) -> list[ReplayRef]:
    """Keep last N + any replay whose start_step is within `tolerance` of a pinned step.

    Deletes everything else. If tolerance > 0, we keep the nearest replay for
    each pinned step (useful when the record interval doesn't land exactly on
    a milestone).
    """
    keep_at_steps = keep_at_steps or []
    ordered = sorted(refs, key=lambda r: r.start_step)
    keep: set[int] = set()

    # Most-recent N
    for r in ordered[-keep_last_n:]:
        keep.add(r.start_step)

    # Pinned milestones: for each target, keep the closest existing replay
    # whose start_step is within `tolerance`. If tolerance is 0, only exact
    # matches survive.
    for target in keep_at_steps:
        best: ReplayRef | None = None
        best_dist = math.inf
        for r in ordered:
            dist = abs(r.start_step - target)
            if dist <= tolerance and dist < best_dist:
                best = r
                best_dist = dist
        if best is not None:
            keep.add(best.start_step)

    return [r for r in ordered if r.start_step not in keep]


def policy_logarithmic(
    refs: list[ReplayRef],
    *,
    keep_last_n: int = 10,
    base: float = 10.0,
    **_,
) -> list[ReplayRef]:
    """Keep last N + approximately one replay per decade of step count.

    Buckets replays by floor(log_base(start_step + 1)). Within each bucket
    keep the *earliest* replay (so the bucket's signature point survives even
    as newer ones pile up); the last-N override protects recent detail.

    Example with base=10 over a 10M-step run, one replay every 100k steps:
      bucket 5 (1e5–1e6): keeps 1    ← earliest in that decade
      bucket 6 (1e6–1e7): keeps 1
      bucket 7 (1e7+):    keeps 1
      plus the 10 most recent
    Total: ~13 replays instead of 100.
    """
    ordered = sorted(refs, key=lambda r: r.start_step)
    keep: set[int] = set()

    for r in ordered[-keep_last_n:]:
        keep.add(r.start_step)

    by_bucket: dict[int, ReplayRef] = {}
    for r in ordered:
        bucket = 0 if r.start_step < 1 else int(math.floor(math.log(r.start_step, base)))
        # Keep the earliest in each bucket (i.e. the first we encounter since
        # `ordered` is ascending).
        if bucket not in by_bucket:
            by_bucket[bucket] = r
    for r in by_bucket.values():
        keep.add(r.start_step)

    return [r for r in ordered if r.start_step not in keep]


# ----------------------------- registry ---------------------------------------


POLICIES: dict[str, Callable[..., list[ReplayRef]]] = {
    "last_n":       policy_last_n,
    "milestones":   policy_milestones,
    "logarithmic":  policy_logarithmic,
}


def apply_policy(
    name: str,
    refs: list[ReplayRef],
    config: dict | None = None,
) -> list[ReplayRef]:
    """Look up `name` in POLICIES and call it with **config. Unknown name → error."""
    if name not in POLICIES:
        raise ValueError(
            f"Unknown retention policy {name!r}. Available: {sorted(POLICIES)}"
        )
    return POLICIES[name](refs, **(config or {}))
