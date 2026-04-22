#!/usr/bin/env python3
"""replay.py — one CLI for the replay pipeline.

Subcommands:
  list          enumerate checkpoints available on GCS
  fetch         download one checkpoint into the local LRU cache
  render        load a cached checkpoint and render a trajectory for the dashboard
  all           fetch + render in one shot
  clean         wipe the local cache
  list-remote   enumerate replays in the public GCS replays bucket
  prune         apply a retention policy to the replays bucket
  delete        delete a single replay from the replays bucket
  backfill      re-simulate from historic checkpoints to seed replay milestones

Examples:
  python scripts/replay.py list --exp baseline_faithful
  python scripts/replay.py all --exp baseline_faithful --seed 0 --step latest --steps 1000
  python scripts/replay.py list-remote
  python scripts/replay.py prune --policy milestones --keep-last-n 10 \\
      --keep-at-steps 0,1000000,5000000,10000000 --tolerance 100000 --dry-run
  python scripts/replay.py delete --exp baseline_faithful --seed 0 --step 99000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import replay_cache


def cmd_list(args):
    refs = replay_cache.list_checkpoints(args.exp, args.seed)
    if not refs:
        print("(no checkpoints found)")
        return
    by_run: dict[tuple[str, int], list] = {}
    for r in refs:
        by_run.setdefault((r.exp, r.seed), []).append(r)
    for (exp, seed), group in sorted(by_run.items()):
        print(f"\n{exp} / seed_{seed}  ({len(group)} checkpoints)")
        for r in group:
            size_mb = r.size_bytes / 1e6
            print(f"  step {r.step:>10,d}  {size_mb:6.1f} MB  updated {r.updated_iso}")


def cmd_fetch(args):
    path = replay_cache.fetch_checkpoint(args.exp, args.seed, args.step, force=args.force)
    print(path)


def cmd_render(args):
    # Heavy imports deferred: render pulls in the full JAX stack.
    from scripts import replay_render

    ckpt = Path(args.checkpoint) if args.checkpoint else replay_cache.fetch_checkpoint(
        args.exp, args.seed, args.step
    )
    config = replay_render.load_config(args.config)
    traj = replay_render.render_trajectory(config, ckpt, args.steps)
    out_dir = Path(args.out_root) / args.exp / f"seed_{args.seed}" / f"step_{traj['start_step']:08d}"
    replay_render.write_trajectory(traj, out_dir)
    replay_render.update_index(Path(args.out_root))


def cmd_all(args):
    ckpt = replay_cache.fetch_checkpoint(args.exp, args.seed, args.step)
    args.checkpoint = str(ckpt)
    cmd_render(args)


def cmd_clean(args):
    freed = replay_cache.clean_cache()
    print(f"cleaned cache, freed {freed / 1e6:.1f} MB from {replay_cache.cache_root()}")


def cmd_prune(args):
    from scripts import replay_upload

    policy_cfg: dict = {}
    if args.keep_last_n is not None:
        policy_cfg["keep_last_n"] = args.keep_last_n
    if args.keep_at_steps:
        policy_cfg["keep_at_steps"] = [int(s) for s in args.keep_at_steps.split(",") if s]
    if args.tolerance is not None:
        policy_cfg["tolerance"] = args.tolerance

    targets = replay_upload.prune(
        args.policy, policy_cfg, bucket=args.bucket, dry_run=args.dry_run,
    )
    if not targets:
        print(f"(nothing to prune under policy '{args.policy}')")
        return
    verb = "would delete" if args.dry_run else "deleted"
    print(f"{verb} {len(targets)} replay(s):")
    for r in targets:
        print(f"  {r.exp}/seed_{r.seed}/step_{r.start_step:08d}  ({r.size_bytes/1e6:.1f} MB)")


def cmd_delete(args):
    from scripts import replay_upload

    if not args.yes:
        confirm = input(
            f"delete gs://{args.bucket or replay_upload.DEFAULT_BUCKET}/"
            f"{args.exp}/seed_{args.seed}/step_{args.step:08d}/ ? [y/N] "
        )
        if confirm.strip().lower() not in ("y", "yes"):
            print("aborted")
            return
    n = replay_upload.delete_replay(args.exp, args.seed, args.step, bucket=args.bucket)
    replay_upload.rebuild_index(args.bucket)
    print(f"removed {n} blob(s)")


def cmd_backfill(args):
    """Simulate forward K steps from one or more historic checkpoints, upload
    replays in the same quantized format the live recorder emits. Used to seed
    the public bucket with "before" snapshots at pinned milestones after the
    fact. Step 0 has no checkpoint — we init_simstate(PRNGKey(seed)) instead."""
    import os
    import tempfile
    from pathlib import Path

    # Heavy imports — only pulled when this command actually runs.
    os.environ.setdefault("XLA_FLAGS",
                          "--xla_cpu_enable_fast_math=true")
    import jax
    import yaml
    from src.environment import _build_physics
    from src.jax_state import init_simstate
    from src.jax_sim import build_sim_step
    from src.jax_checkpoint import load_simstate

    from scripts import replay_cache, replay_upload
    from scripts.replay_recorder import ReplayRecorder

    with open(args.config) as f:
        config = yaml.safe_load(f) or {}

    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    bucket = args.bucket or replay_upload.DEFAULT_BUCKET

    # Reuse compiled sim_step across checkpoints — saves ~15s JIT per target.
    print(f"building sim infra (one-time JIT)…")
    space, _ = _build_physics(config)
    sim_step_core, _ = build_sim_step(config, space)
    template = init_simstate(config, jax.random.PRNGKey(args.seed))

    with tempfile.TemporaryDirectory() as td:
        for start in steps:
            print(f"\n=== backfilling start_step={start:,} (length={args.length}) ===")
            # Load (or synthesize) state at this step
            if start == 0:
                sim_state = init_simstate(config, jax.random.PRNGKey(args.seed))
                print(f"  init state from PRNGKey({args.seed})")
            else:
                ckpt = replay_cache.fetch_checkpoint(args.exp, args.seed, start)
                sim_state = load_simstate(str(ckpt), template)
                print(f"  loaded checkpoint at sim_state.step={int(sim_state.step)}")

            # Recorder with interval set above length so auto-flush never fires;
            # we drive the flush manually at the end. Retention runs as normal.
            cfg = dict(config)
            cfg["replay_record_interval_steps"] = args.length * 100
            cfg["replay_record_length_steps"] = args.length
            recorder = ReplayRecorder(cfg, args.exp, args.seed, td, bucket=bucket)

            t0 = time.time()
            base = int(sim_state.step)
            for i in range(args.length):
                sim_state = sim_step_core(sim_state)
                recorder._capture(sim_state, base + i + 1)
                if (i + 1) % 1000 == 0:
                    sps = (i + 1) / (time.time() - t0)
                    print(f"  captured {i + 1}/{args.length} "
                          f"({sps:.1f} sps, ETA {(args.length - i - 1) / max(sps, 0.1):.0f}s)")
            recorder._flush(sim_state)
            print(f"  flushed in {time.time() - t0:.1f}s total")

    # Final index rebuild — recorder rebuilds once per flush already, but do it
    # once more at the end so a dashboard refresh right after this script
    # finishes sees all N replays atomically.
    n_idx = replay_upload.rebuild_index(bucket)
    print(f"\ndone. bucket index now has {n_idx} replay(s)")


def cmd_list_remote(args):
    from scripts import replay_upload

    refs = replay_upload.list_remote_replays(args.bucket)
    if not refs:
        print(f"(no replays in gs://{args.bucket or replay_upload.DEFAULT_BUCKET}/)")
        return
    by_run: dict[tuple[str, int], list] = {}
    for r in refs:
        by_run.setdefault((r.exp, r.seed), []).append(r)
    for (exp, seed), group in sorted(by_run.items()):
        print(f"\n{exp} / seed_{seed}  ({len(group)} replays)")
        for r in group:
            print(f"  step {r.start_step:>10,d}  {r.size_bytes/1e6:6.1f} MB")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    lst = sub.add_parser("list", help="enumerate checkpoints on GCS")
    lst.add_argument("--exp")
    lst.add_argument("--seed", type=int)
    lst.set_defaults(func=cmd_list)

    fet = sub.add_parser("fetch", help="download a checkpoint to local cache")
    fet.add_argument("--exp", required=True)
    fet.add_argument("--seed", type=int, required=True)
    fet.add_argument("--step", default="latest", help="step number or 'latest'")
    fet.add_argument("--force", action="store_true", help="re-download even if cached")
    fet.set_defaults(func=cmd_fetch)

    # Keep the default as a string so we don't pull in the JAX-heavy
    # replay_render module just to build the arg parser.
    default_out_root = str(
        Path(__file__).resolve().parent.parent
        / "dashboard" / "site" / "public" / "replays"
    )

    ren = sub.add_parser("render", help="render a trajectory for the dashboard")
    ren.add_argument("--exp", required=True)
    ren.add_argument("--seed", type=int, required=True)
    ren.add_argument("--step", default="latest")
    ren.add_argument("--steps", type=int, default=1000, help="number of steps to render")
    ren.add_argument("--checkpoint", help="skip GCS; use this local .npz instead")
    ren.add_argument("--config", default="configs/baseline_faithful.yaml")
    ren.add_argument("--out-root", default=default_out_root)
    ren.set_defaults(func=cmd_render)

    al = sub.add_parser("all", help="fetch + render")
    al.add_argument("--exp", required=True)
    al.add_argument("--seed", type=int, required=True)
    al.add_argument("--step", default="latest")
    al.add_argument("--steps", type=int, default=1000)
    al.add_argument("--config", default="configs/baseline_faithful.yaml")
    al.add_argument("--out-root", default=default_out_root)
    al.set_defaults(func=cmd_all)

    cln = sub.add_parser("clean", help="wipe the local cache")
    cln.set_defaults(func=cmd_clean)

    lsr = sub.add_parser("list-remote", help="enumerate replays in the public GCS bucket")
    lsr.add_argument("--bucket", help="override EVO_REWARD_REPLAYS_BUCKET")
    lsr.set_defaults(func=cmd_list_remote)

    prn = sub.add_parser("prune", help="apply a retention policy to the bucket")
    prn.add_argument("--policy", default="last_n",
                     choices=["last_n", "milestones", "logarithmic"])
    prn.add_argument("--keep-last-n", type=int, default=None)
    prn.add_argument("--keep-at-steps", help="comma-separated step numbers to always keep")
    prn.add_argument("--tolerance", type=int, default=None,
                     help="milestones: max distance from pinned step for a match")
    prn.add_argument("--dry-run", action="store_true")
    prn.add_argument("--bucket")
    prn.set_defaults(func=cmd_prune)

    bfl = sub.add_parser("backfill", help="re-simulate milestones from saved checkpoints")
    bfl.add_argument("--exp", default="baseline_faithful")
    bfl.add_argument("--seed", type=int, default=0)
    bfl.add_argument("--steps", required=True,
                     help="comma-separated start_step values to backfill (0 synthesizes from seed)")
    bfl.add_argument("--length", type=int, default=10_000,
                     help="frames per replay (matches the live recorder default)")
    bfl.add_argument("--config", default="configs/baseline_faithful.yaml")
    bfl.add_argument("--bucket", help="override EVO_REWARD_REPLAYS_BUCKET")
    bfl.set_defaults(func=cmd_backfill)

    dlt = sub.add_parser("delete", help="delete one replay from the bucket")
    dlt.add_argument("--exp", required=True)
    dlt.add_argument("--seed", type=int, required=True)
    dlt.add_argument("--step", type=int, required=True, help="start_step of the replay to delete")
    dlt.add_argument("--yes", "-y", action="store_true", help="skip confirmation")
    dlt.add_argument("--bucket")
    dlt.set_defaults(func=cmd_delete)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
