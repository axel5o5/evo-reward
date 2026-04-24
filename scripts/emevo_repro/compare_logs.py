"""Compare emevo smoke run vs our Phase 1a run at matched steps.

Emevo logs:  <emevo_logdir>/log-*.parquet   (per-slot per-step rows)
Our logs:    <our_results>/metrics.npz      (time-series arrays; schema: src/jax_metrics.py)

Prints side-by-side trajectory at sample steps and writes CSV.

Slot layout (paper default config 20251001-predator-default.toml):
  prey: slots [0, n_max_preys)            = [0, 450)
  pred: slots [n_max_preys, n_max_agents) = [450, 500)

Usage:
    python scripts/emevo_repro/compare_logs.py \\
        --emevo-dir ~/emevo_repro/logs/smoke_seed0 \\
        --ours-metrics results/baseline_faithful/seed_0/d31d/metrics.npz \\
        --out comparison.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

SAMPLE_STEPS = [1_000, 5_000, 10_000, 20_000, 50_000, 100_000,
                200_000, 300_000, 400_000, 500_000]
PREY_SLOT_END = 450   # n_max_preys
PRED_SLOT_END = 500   # n_max_agents


def load_emevo_logs(logdir: Path) -> pl.DataFrame:
    files = sorted(logdir.glob("log-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No log-*.parquet in {logdir}")
    return pl.concat([pl.from_arrow(pq.read_table(f)) for f in files])


def emevo_trajectory(df: pl.DataFrame) -> pl.DataFrame:
    active = df.filter(pl.col("unique_id") > 0)
    prey = active.filter(pl.col("slots") < PREY_SLOT_END)
    pred = active.filter(pl.col("slots") >= PREY_SLOT_END)
    p_stats = prey.group_by("step").agg([
        pl.len().alias("prey_count"),
        pl.col("energy").mean().alias("prey_E_mean"),
    ])
    d_stats = pred.group_by("step").agg([
        pl.len().alias("pred_count"),
        pl.col("energy").mean().alias("pred_E_mean"),
        pl.col("energy").max().alias("pred_E_max"),
    ])
    return p_stats.join(d_stats, on="step", how="full", coalesce=True).sort("step")


def load_our_metrics(path: Path) -> pl.DataFrame:
    data = np.load(path)
    cols = {
        "step":         data["steps"],
        "prey_count":   data["prey_population"],
        "pred_count":   data["predator_population"],
        "prey_E_mean":  data["prey_mean_energy"],
        "pred_E_mean":  data["predator_mean_energy"],
        "prey_w_pred":  data["prey_mean_w_pred"],
        "pred_w_prey":  data["pred_mean_w_prey"],
        "pred_w_pred":  data["pred_mean_w_pred"],
    }
    # pred_E_max isn't in metrics.npz; fill None.
    cols["pred_E_max"] = np.full_like(cols["pred_count"], np.nan, dtype=float)
    return pl.DataFrame(cols)


def nearest_row(df: pl.DataFrame, step: int, tol: int = 2000) -> dict | None:
    if "step" not in df.columns or df.is_empty():
        return None
    candidates = df.filter((pl.col("step") - step).abs() <= tol)
    if candidates.is_empty():
        return None
    idx = int((candidates["step"] - step).abs().arg_min())
    return candidates.row(idx, named=True)


def fmt(x, spec=".1f") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emevo-dir", type=Path, required=True)
    ap.add_argument("--ours-metrics", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("emevo_vs_ours.csv"))
    args = ap.parse_args()

    print(f"Emevo logs: {args.emevo_dir}")
    edf = load_emevo_logs(args.emevo_dir)
    print(f"  {len(edf):,} rows, {edf['step'].n_unique()} unique steps")
    etraj = emevo_trajectory(edf)

    print(f"Ours metrics: {args.ours_metrics}")
    otraj = load_our_metrics(args.ours_metrics)
    print(f"  {len(otraj):,} logged steps")

    cols = ["prey_count", "pred_count", "prey_E_mean", "pred_E_mean", "pred_E_max"]
    hdr = f"\n{'step':>7}"
    for c in cols:
        hdr += f"  {c + ' (emevo)':>18}  {c + ' (ours)':>18}"
    print(hdr)
    print("-" * len(hdr))

    rows_out = []
    for step in SAMPLE_STEPS:
        e = nearest_row(etraj, step) or {}
        o = nearest_row(otraj, step) or {}
        if not e and not o:
            continue
        line = f"{step:>7,}"
        rec = {"step": step}
        for c in cols:
            ev = e.get(c)
            ov = o.get(c)
            line += f"  {fmt(ev):>18}  {fmt(ov):>18}"
            rec[f"{c}_emevo"] = ev
            rec[f"{c}_ours"] = ov
        print(line)
        rows_out.append(rec)

    print(f"\nWrote {args.out}")
    pl.DataFrame(rows_out).write_csv(args.out)


if __name__ == "__main__":
    main()
