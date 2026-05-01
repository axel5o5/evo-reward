# Replay archiving and pruning

Replays in `gs://evo-reward-replays-public/` accumulate fast — by 2026-05-01
the bucket held 93 GB across 800+ checkpoints, much of it superseded
debugging runs we'd never play back again. This doc records the two-step
archive → prune pattern we use to reclaim storage without losing the
record of what each run showed.

## Principle: document before deleting

A pruned replay's binary is unrecoverable, but its **summary** —
population trajectory, extinction timing, peak counts — is cheap and
small enough to commit to the repo permanently. So the rule is:

> **Never delete a run's binaries until that run's stats are
> committed to `archive/`.**

The archive is the documentation that makes the deletion safe. It also
keeps the dashboard's "Archived runs" panel honest: pruned runs stay
visible there with their trajectory sparkline, just without the option
to play them back.

## Two scripts, in order

### 1. `scripts/archive_summary.py` — capture the record

Walks the bucket, downloads only the `alive` / `species` / `step_nums`
slices of each `frames.bin` (a few MB per checkpoint instead of the
full ~80 MB), and computes per-run rollups:

- final step reached, n_checkpoints
- extinct? if so, which species, at what step
- peak prey, peak pred
- per-checkpoint trajectory (prey / pred counts at each step)

Outputs:

- `archive/runs/<exp>__seed_<N>__<tag>.json` — full per-checkpoint
  detail (committed).
- `archive/SUMMARY.md` — human-readable table of all runs (committed).
- `archive/SUMMARY.json` — machine-readable mirror for the dashboard
  (committed). With `--upload`, also pushed to
  `gs://evo-reward-replays-public/summary.json` for the prod panel.

Re-runs are incremental: if a run's per-run JSON already exists in
`archive/runs/`, it's reused. Use `--force` to re-process.

Config-free by design — we don't try to recover the historical config a
given run used. The metrics derived (population, extinction, births /
deaths) come straight from `alive` and don't depend on prey-radius /
mouth-bin / energy-capacity numbers, which is why this works across
config drift.

### 2. `scripts/archive_prune.py` — reclaim the storage

A separate script with an explicit per-run policy mapping. **Always
dry-run first** (default) and read the plan before passing
`--execute`.

Three policies per run:

- **`keep_all`** — leave every checkpoint. Use for runs you'd
  realistically scrub through again (recent / current-config / your
  showcase results).
- **`keep_sparse`** — thin to ~10 evenly-spaced checkpoints
  (`policy_evenly_spaced` in `replay_retention.py`). Use for long runs
  where you want trajectory scrubbability but don't need every 20k-step
  capture. Default `SPARSE_N=10` is tuned to keep ~1 GB per run.
- **`keep_last_only`** — drop everything except the highest-numbered
  checkpoint. Use for superseded experiments where you only want the
  end state to remain playable. The `archive/runs/<key>.json` still
  records the full trajectory.

Excluded entirely: `EXCLUDED_PREFIXES` at the top of the script. Add
the path of any currently-running training run there before pruning so
its incoming checkpoints aren't touched.

Any run on the bucket that isn't in either `ARCHIVE_POLICY` or
`EXCLUDED_PREFIXES` causes the script to abort. This is deliberate —
the script refuses to guess.

After deletion, it rebuilds `gs://<bucket>/index.json` from a fresh
listing so the dashboard's replay selector reflects what's actually
still there.

## Auth setup

Both scripts read the bucket anonymously (it's public-read), so
`--dry-run` works with no credentials.

For writes (`--execute`, `--upload`):

- We shell out to `gsutil` rather than the Python `storage.Client` so
  the user's existing `gcloud auth login` credentials are picked up
  without needing to set up Application Default Credentials.
- If you do want the Python path: `gcloud auth application-default
  login` once is enough.

## Step-by-step recipe

```bash
# 1. (Optional) refresh the archive snapshot if there are new runs since
#    the last archive_summary.py run.
python scripts/archive_summary.py

# 2. Edit the ARCHIVE_POLICY dict in scripts/archive_prune.py to assign
#    every run on the bucket to keep_all / keep_sparse / keep_last_only.
#    Add the path of any currently-running training to EXCLUDED_PREFIXES.

# 3. Dry-run. Read the totals. The script aborts if any bucket run is
#    unmatched.
python -m scripts.archive_prune --dry-run

# 4. Execute. Deletes the binaries via gsutil and rebuilds index.json.
python -m scripts.archive_prune --execute

# 5. Push the refreshed summary so the dashboard's archive panel
#    reflects the new state.
python scripts/archive_summary.py --upload

# 6. Commit the new archive/ files to the repo.
git add archive/ scripts/
git commit -m "[Archive] Prune <date> — <reason>"
git push
```

## How to decide what goes in which bucket

Useful heuristics from the 2026-05-01 prune (47.5 GB freed):

- **Recent + current config → `keep_all`.** If the run was launched in
  the last few days against a config that's still active, keep it
  fully scrubbable. Examples: `axis2_aligned`, `baseline_*_ddb*`.
- **Long meaningful run → `keep_sparse`.** Hundreds of checkpoints
  where you want the trajectory shape but not every 20k-step frame.
  Examples: `exp_tune_eta_*`, `exp_sweep_mouth_smol`.
- **Old / superseded → `keep_last_only`.** Anything with a date prefix
  in the run_tag from > 1 week ago, anything tagged with a debug
  marker (`d19`, `d28a`, `phase1a-v3`, `pre_d18_fix`), or anything
  whose config no longer exists. The end-state checkpoint is still
  playable but the dense middle is gone. Examples: most of
  `baseline_faithful/seed_0/*`.

Edge cases worth thinking about each time:

- **Just-finished runs you'll cite in writeups.** Demote one or two
  notches harder than the heuristic suggests — a trajectory you'll
  need a figure of is worth `keep_sparse` even if you'd otherwise
  call it superseded.
- **Reproductions of bug fixes (`pre_d18_fix`, `d28a`/`d28b`, etc).**
  These are evidence in `docs/emevo-diff.md`. `keep_last_only` is
  fine — the fix discussion already cites the exact behavior.
- **Currently-running training.** Add to `EXCLUDED_PREFIXES`, not to
  any policy bucket. The bucket scan is read at script start and the
  paths it returns are static within one run; new checkpoints written
  after that won't be in the delete list, but you don't want any
  partial state racing with rebuild_index either.

## Renaming for chronological sort: `scripts/archive_rename.py`

Run tags accumulated three different conventions over time
(`d19`, `phase1a-v5`, `2026-04-24T1646Z_v8`...) which made it hard to
tell at a glance what was old versus new. The fix: prefix every
non-date-prefixed tag with the ISO date of the earliest blob's GCS
`time_created`. The exp/seed structure is untouched — only the
`<run_tag>` segment changes.

```
exp_tune_eta_0.45/seed_0/tune_eta_045/  →  exp_tune_eta_0.45/seed_0/2026-04-24_tune_eta_045/
baseline_faithful/seed_0/d19/            →  baseline_faithful/seed_0/2026-04-23_d19/
baseline_faithful/seed_0/<step_NNN>/     →  baseline_faithful/seed_0/2026-04-21/<step_NNN>/  (untagged → tagged)
```

The script:

1. Lists the bucket anonymously and finds every `(exp, seed, run_tag)`
   tuple whose tag doesn't already start with `YYYY-MM-DD`.
2. Pulls the earliest blob `time_created` per run to derive the date.
3. Prints a dry-run plan (default) — review before executing.
4. With `--execute`: runs `gsutil -m mv -r` per run, renames the local
   `archive/runs/<exp>__seed_<N>__<old>.json` files (and rewrites their
   embedded `run_tag` field), and prints the substitutions to apply to
   `ARCHIVE_POLICY` in `archive_prune.py`.

After execute:

```bash
# Apply the printed (exp, seed, "old") → (exp, seed, "new") edits to
# scripts/archive_prune.py manually. The script does NOT auto-edit it
# to avoid silent code changes.

# Rebuild the bucket-side index from the new paths:
python -c "from scripts.archive_prune import _rebuild_index_anon; \
           _rebuild_index_anon('evo-reward-replays-public')"

# Re-upload summary.json (regenerates from the renamed archive/runs/* files):
python scripts/archive_summary.py --upload

# Sync local mirrors:
gsutil cp gs://evo-reward-replays-public/index.json dashboard/site/public/replays/index.json
cp archive/SUMMARY.json dashboard/site/public/replays/summary.json
```

**Caveat: URL bookmarks break.** The replay page encodes the run via
`?tag=...&exp=...&seed=...&step=...`. Anyone holding a URL with an old
tag value will 404 after the rename. There's no compatibility mapping
on the dashboard side; this is the cost of doing it.

## What the dashboard panel shows

The "Archived runs" panel on the Replay page reads
`/replays/summary.json` (= GCS `summary.json` in prod). It shows every
run that was archived, sortable by final step / extinction step / peak
counts, with a per-run trajectory sparkline. Each row carries a `live`
or `pruned` badge based on whether the run still has at least one
checkpoint in `index.json`.

After a prune that thins runs but doesn't fully delete any of them,
every row reads `live` even though most are now sparser than the
sparkline shows. That's a UI follow-up if it becomes confusing — for
now the sparkline density makes the thinning visually obvious.
