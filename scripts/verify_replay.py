"""verify_replay.py
-------------------
Replay sanity check: load a frames.bin + meta.json pair (local or GCS),
count the events that should be happening in a healthy simulation, and
flag obvious pathologies.

This is a *post-hoc* integrity check — it doesn't re-run the sim, just
reads positions/energy/alive from the recorded frames and derives
events by comparing consecutive frames:

  - prey feeding: a food slot goes active→inactive while some prey is
    within prey_radius of its position
  - predator catches: a prey slot goes alive→dead while some predator
    is within (pred_r + prey_r) and the prey falls in the mouth arc
  - births: any slot flipping inactive→active
  - deaths: any slot flipping active→inactive (includes catches + starvation)
  - predator energy band: min/mean/max across all frames
  - population turnover: births and deaths aggregated

Usage:
  python scripts/verify_replay.py --replay path/to/step_NNNNNNNN/
  python scripts/verify_replay.py --gcs baseline_faithful/seed_0/<tag>/step_NNNNNNNN
  python scripts/verify_replay.py --gcs baseline_faithful/seed_0/<tag>/step_NNNNNNNN \\
      --config configs/baseline_faithful.yaml

JSON output goes to stdout; human-readable summary to stderr.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml


# ─── section decoding (mirrors dashboard/site/src/lib/replayLoader.ts) ────

_DTYPE_SIZE = {"float32": 4, "int32": 4, "uint16": 2, "uint8": 1}
_NP_DTYPE = {"float32": np.float32, "int32": np.int32, "uint16": np.uint16, "uint8": np.uint8}


def _load_replay(replay_dir: Path) -> dict:
    """Return a dict of numpy arrays matching the JS loader's ReplayData."""
    meta = json.loads((replay_dir / "meta.json").read_text())
    buf = (replay_dir / "frames.bin").read_bytes()

    def section(name: str) -> np.ndarray:
        s = meta["sections"][name]
        count = s["length"] // _DTYPE_SIZE[s["dtype"]]
        arr = np.frombuffer(buf, dtype=_NP_DTYPE[s["dtype"]],
                            count=count, offset=s["offset"])
        return arr.reshape(s["shape"])

    out = {"meta": meta}
    out["alive"] = section("alive")
    out["angle"] = section("angle")
    out["energy_raw"] = section("energy")
    out["food_active"] = section("food_active")
    out["food_pos_raw"] = section("food_pos")
    out["pos_raw"] = section("pos")
    out["species"] = section("species")
    out["radii"] = section("radii")
    out["step_nums"] = section("step_nums")

    # Dequantize if needed.
    scales = meta.get("scales", {}) or {}
    if meta.get("quantize"):
        out["pos"] = out["pos_raw"].astype(np.float32) * scales["pos"]
        out["food_pos"] = out["food_pos_raw"].astype(np.float32) * scales["food_pos"]
        out["energy"] = out["energy_raw"].astype(np.float32) * scales["energy"]
    else:
        out["pos"] = out["pos_raw"].astype(np.float32)
        out["food_pos"] = out["food_pos_raw"].astype(np.float32)
        out["energy"] = out["energy_raw"].astype(np.float32)
    return out


# ─── event extraction ─────────────────────────────────────────────────────


def _wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _nearest_mouth_bin(dx, dy, heading, n_bins, spacing_rad):
    """Return the nearest tactile bin index for a relative position."""
    angle_to_target = np.arctan2(dy, dx)
    centers = np.arange(n_bins) * spacing_rad
    rel = _wrap_angle(angle_to_target - heading - centers)
    return int(np.argmin(np.abs(rel)))


def analyze(data: dict, config: dict) -> dict:
    """Compute per-replay statistics and event counts."""
    n_frames = data["meta"]["n_frames"]
    max_agents = data["meta"]["max_agents"]
    food_max = data["meta"]["food_max"]
    prey_r = float(config["prey_radius"])
    pred_r = float(config["predator_radius"])
    n_tactile = int(config["n_tactile_sensors"])
    tactile_rad = math.radians(float(config["tactile_spacing_deg"]))
    mouth_bins = set(config.get("predator_mouth_tactile_bins", [0, 1, 17]))

    alive = data["alive"]              # (T, N) uint8
    pos = data["pos"]                  # (T, N, 2)
    angle = data["angle"]              # (T, N)
    energy = data["energy"]            # (T, N)
    food_active = data["food_active"]  # (T, F)
    food_pos = data["food_pos"]        # (T, F, 2)
    species = data["species"]          # (N,) int32

    prey_slots = np.where(species == 0)[0]
    pred_slots = np.where(species == 1)[0]

    # Per-frame population + food + pred-energy stats.
    prey_pop = [int(alive[t, prey_slots].sum()) for t in range(n_frames)]
    pred_pop = [int(alive[t, pred_slots].sum()) for t in range(n_frames)]
    food_count = [int(food_active[t].sum()) for t in range(n_frames)]
    pred_energy_per_frame = []
    for t in range(n_frames):
        mask = alive[t, pred_slots].astype(bool)
        if mask.any():
            pred_energy_per_frame.append(float(energy[t, pred_slots][mask].mean()))
        else:
            pred_energy_per_frame.append(0.0)

    # Birth / death events at frame-pair boundaries.
    births = 0
    deaths = 0
    for t in range(n_frames - 1):
        flip_on = (alive[t] == 0) & (alive[t + 1] == 1)
        flip_off = (alive[t] == 1) & (alive[t + 1] == 0)
        births += int(flip_on.sum())
        deaths += int(flip_off.sum())

    # Feeding events: food_active goes 1→0 at frame t+1; attribute to nearest
    # prey within prey_radius at frame t.
    feedings = 0
    for t in range(n_frames - 1):
        eaten_food = np.where(food_active[t].astype(bool) & ~food_active[t + 1].astype(bool))[0]
        if eaten_food.size == 0:
            continue
        live_prey = prey_slots[alive[t, prey_slots].astype(bool)]
        if live_prey.size == 0:
            continue
        for f in eaten_food:
            d = np.linalg.norm(pos[t, live_prey] - food_pos[t, f], axis=1)
            if (d <= prey_r + 0.5).any():
                feedings += 1

    # Predator catch events: prey slot goes alive 1→0; look for any predator
    # within sum-of-radii and in the mouth arc at frame t.
    catches = 0
    unexplained_deaths = 0
    for t in range(n_frames - 1):
        dead_prey = np.where(
            (alive[t, prey_slots] == 1) & (alive[t + 1, prey_slots] == 0)
        )[0]
        if dead_prey.size == 0:
            continue
        live_pred = pred_slots[alive[t, pred_slots].astype(bool)]
        for k in dead_prey:
            prey_slot = int(prey_slots[k])
            prey_p = pos[t, prey_slot]
            caught = False
            for p in live_pred:
                delta = prey_p - pos[t, int(p)]
                d = np.linalg.norm(delta)
                if d > pred_r + prey_r + 0.5:
                    continue
                nbin = _nearest_mouth_bin(
                    float(delta[0]), float(delta[1]),
                    float(angle[t, int(p)]), n_tactile, tactile_rad,
                )
                if nbin in mouth_bins:
                    caught = True
                    break
            if caught:
                catches += 1
            else:
                unexplained_deaths += 1

    # Per-frame headline.
    prey_eaten_by_pred_pct = (
        catches / deaths * 100.0 if deaths > 0 else 0.0
    )

    return {
        "replay": {
            "start_step": int(data["meta"]["start_step"]),
            "n_frames": int(n_frames),
            "max_agents": int(max_agents),
            "food_max": int(food_max),
        },
        "population": {
            "prey_first": prey_pop[0], "prey_last": prey_pop[-1],
            "prey_min": min(prey_pop), "prey_max": max(prey_pop),
            "pred_first": pred_pop[0], "pred_last": pred_pop[-1],
            "pred_min": min(pred_pop), "pred_max": max(pred_pop),
            "food_min": min(food_count), "food_max": max(food_count),
        },
        "events": {
            "prey_feedings": int(feedings),
            "predator_catches": int(catches),
            "births": int(births),
            "deaths": int(deaths),
            "unexplained_deaths": int(unexplained_deaths),
            "pct_of_deaths_from_catches": round(prey_eaten_by_pred_pct, 1),
        },
        "predator_energy": {
            "min": round(min(pred_energy_per_frame), 1),
            "mean": round(sum(pred_energy_per_frame) / len(pred_energy_per_frame), 1),
            "max": round(max(pred_energy_per_frame), 1),
        },
        "red_flags": _red_flags(catches, feedings, births, pred_energy_per_frame,
                                 pred_pop, config),
    }


def _red_flags(catches, feedings, births, pred_e, pred_pop, config):
    flags = []
    cap = float(config.get("energy_capacity", 1000.0))
    pred_active = any(p > 0 for p in pred_pop)

    if catches == 0 and pred_active:
        flags.append(
            "ZERO_PREDATOR_CATCHES: predators present but no catches detected. "
            "Expect ~0 with D18/D19 pre-fix; expect >0 with fix applied."
        )
    if feedings == 0:
        flags.append("ZERO_PREY_FEEDINGS: no prey eating food across the whole window.")
    if pred_active and pred_e and max(pred_e) > cap * 0.95:
        flags.append(
            f"PREDATOR_ENERGY_SATURATED: peak mean={max(pred_e):.0f} ≥ {cap*0.95:.0f}; "
            "predators may be eating without metabolic cost."
        )
    if births == 0:
        flags.append("ZERO_BIRTHS: no births in this window — evolution cannot happen.")
    return flags


# ─── CLI ──────────────────────────────────────────────────────────────────


def _load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    # Overlay runtime default so fields like n_tactile_sensors etc. are
    # present if the science config is missing them.
    rt_path = Path(__file__).resolve().parents[1] / "configs/runtime/default.yaml"
    if rt_path.exists():
        cfg.update(yaml.safe_load(rt_path.read_text()) or {})
    return cfg


def _fetch_gcs(remote_prefix: str, bucket: str) -> Path:
    """Download meta.json + frames.bin into a tempdir and return its path."""
    from google.cloud import storage
    # Fall back to anonymous client if ADC isn't set up — the replays bucket
    # is public-read, so auth isn't required just to download from it.
    try:
        client = storage.Client()
    except Exception:
        client = storage.Client.create_anonymous_client()
    b = client.bucket(bucket)
    tmp = Path(tempfile.mkdtemp(prefix="verify-replay-"))
    for name in ("meta.json", "frames.bin"):
        blob = b.blob(f"{remote_prefix.strip('/')}/{name}")
        blob.download_to_filename(str(tmp / name))
    return tmp


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--replay", help="Local replay dir (must contain meta.json + frames.bin)")
    src.add_argument("--gcs", help="GCS object prefix, e.g. baseline_faithful/seed_0/<tag>/step_00010001")
    parser.add_argument("--bucket", default="evo-reward-replays-public")
    parser.add_argument("--config", default="configs/baseline_faithful.yaml")
    args = parser.parse_args()

    if args.gcs:
        replay_dir = _fetch_gcs(args.gcs, args.bucket)
        src_name = f"gs://{args.bucket}/{args.gcs}"
    else:
        replay_dir = Path(args.replay)
        src_name = str(replay_dir)

    config = _load_config(args.config)
    data = _load_replay(replay_dir)
    result = analyze(data, config)
    result["source"] = src_name

    # Human summary to stderr — doesn't pollute the machine-readable JSON.
    ev = result["events"]
    pop = result["population"]
    pe = result["predator_energy"]
    rep = result["replay"]
    print(f"--- {src_name}", file=sys.stderr)
    print(f"start_step={rep['start_step']:,} n_frames={rep['n_frames']:,}", file=sys.stderr)
    print(f"prey {pop['prey_min']}–{pop['prey_max']}  "
          f"pred {pop['pred_min']}–{pop['pred_max']}  "
          f"food {pop['food_min']}–{pop['food_max']}", file=sys.stderr)
    print(f"catches={ev['predator_catches']:,}  "
          f"feedings={ev['prey_feedings']:,}  "
          f"births={ev['births']}  deaths={ev['deaths']}  "
          f"({ev['pct_of_deaths_from_catches']:.0f}% of deaths from catches)",
          file=sys.stderr)
    print(f"pred_energy  min={pe['min']:.0f}  mean={pe['mean']:.0f}  max={pe['max']:.0f}", file=sys.stderr)
    for flag in result["red_flags"]:
        print(f"  ⚠ {flag}", file=sys.stderr)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
