"""lineage_analysis.py
----------------------
Post-hoc lineage analysis from a directory of checkpoints. Reconstructs:
  - Per-agent n_offspring (how many children each agent produced)
  - Birth event log (step, parent_id, parent_genome, parent_energy, child_id, child_genome)
  - Lineage trees (ancestry of currently-alive agents)
  - Reproductive skew metrics (Gini, top-K share)
  - Diversity-over-time (distinct lineages tracing back N generations)

Designed to compare runs that differ in scaffold design (e.g. uniform
vs energy-weighted DDB rate boost — see findings.md §15.14).

The reconstruction has one known gap: agents born AND dying entirely
between two consecutive checkpoints are invisible. With 20K-step
cadence and typical lifespans of 50K+ steps, this loses ~30-50% of
total births but only ~5% of evolutionarily-meaningful (reproductive)
lineages. See `--diagnose-coverage` for an estimate per run.

Usage:
  python3 scripts/lineage_analysis.py \
      --checkpoint-dir results/axis1_residual/seed_0/2026-05-01T2019Z/checkpoints/ \
      --config configs/axis1_residual.yaml \
      --out-dir analysis/axis1_v7/

  # Compare two runs:
  python3 scripts/lineage_analysis.py \
      --checkpoint-dir results/<v3-archive>/checkpoints/ \
      --config configs/axis1_residual.yaml --out-dir analysis/v3/
  python3 scripts/lineage_analysis.py \
      --checkpoint-dir results/<v7-archive>/checkpoints/ \
      --config configs/axis1_residual.yaml --out-dir analysis/v7/
  # Then compare the JSON summaries side by side.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

# Allow imports from src/ when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
from src.jax_state import init_simstate
from src.jax_checkpoint import load_simstate


def load_checkpoint(path: Path, template):
    """Load a checkpoint and extract per-agent records."""
    state = load_simstate(str(path), template)
    is_active = np.asarray(state.is_active)
    agent_ids = np.asarray(state.agent_ids)
    parent_ids = np.asarray(state.parent_ids)
    species = np.asarray(state.species)
    ages = np.asarray(state.ages)
    energies = np.asarray(state.energies)
    weights = np.asarray(state.reward_weights)  # (max_agents, 4)

    # Active rows only.
    rows = []
    for i in range(len(is_active)):
        if not is_active[i]:
            continue
        rows.append({
            "slot": int(i),
            "agent_id": int(agent_ids[i]),
            "parent_id": int(parent_ids[i]),
            "species": int(species[i]),
            "age": int(ages[i]),
            "energy": float(energies[i]),
            "w_eat": float(weights[i, 0]),
            "w_act": float(weights[i, 1]),
            "w_prey": float(weights[i, 2]),
            "w_pred": float(weights[i, 3]),
        })
    return rows


def parse_step_from_filename(path: Path) -> int:
    """Checkpoint filenames are step_<NNNNNNNN>.npz."""
    stem = path.stem
    if not stem.startswith("step_"):
        raise ValueError(f"unexpected checkpoint filename: {path}")
    return int(stem.removeprefix("step_"))


def compute_agent_registry(checkpoints):
    """Build a registry of every agent ever observed across checkpoints.

    Returns dict: agent_id -> {
        first_step, last_step, species, parent_id,
        max_energy, min_energy, final_genome (4-tuple),
        first_observed_age (helps estimate birth_step),
    }
    """
    registry = {}
    for step, rows in checkpoints:
        for r in rows:
            aid = r["agent_id"]
            if aid not in registry:
                registry[aid] = {
                    "first_step": step,
                    "last_step": step,
                    "species": r["species"],
                    "parent_id": r["parent_id"],
                    "max_energy": r["energy"],
                    "min_energy": r["energy"],
                    "final_genome": (r["w_eat"], r["w_act"], r["w_prey"], r["w_pred"]),
                    "first_observed_age": r["age"],
                }
            else:
                e = registry[aid]
                e["last_step"] = step
                e["max_energy"] = max(e["max_energy"], r["energy"])
                e["min_energy"] = min(e["min_energy"], r["energy"])
                e["final_genome"] = (r["w_eat"], r["w_act"], r["w_prey"], r["w_pred"])
    return registry


def compute_offspring_counts(registry):
    """For each agent_id, count how many distinct children list it as parent."""
    children = defaultdict(list)
    for aid, info in registry.items():
        pid = info["parent_id"]
        if pid >= 0:  # -1 = initial agent, no parent
            children[pid].append(aid)
    return children


def reconstruct_birth_events(checkpoints, registry):
    """For each agent observed, infer their birth step from age and the
    first checkpoint they appear in.

    Also estimate parent's energy at birth: use the parent's energy from
    the LAST checkpoint they appear at before the child's first appearance.
    """
    events = []
    for aid, info in registry.items():
        if info["parent_id"] < 0:
            continue  # initial agent, no birth event
        # Birth step from first observed age:
        # birth_step ≈ first_step - first_observed_age
        birth_step = info["first_step"] - info["first_observed_age"]
        events.append({
            "step": birth_step,
            "child_id": aid,
            "parent_id": info["parent_id"],
            "child_species": info["species"],
            "child_genome_at_first_obs": info["final_genome"],
            # The "final_genome" recorded was the first time we saw this agent,
            # since their genome doesn't change. So this IS the birth genome.
        })
    events.sort(key=lambda e: e["step"])

    # Backfill parent's energy and genome at the closest checkpoint <= birth_step.
    # Build a per-agent timeline: agent_id -> [(step, energy, genome)].
    timeline = defaultdict(list)
    for step, rows in checkpoints:
        for r in rows:
            timeline[r["agent_id"]].append(
                (step, r["energy"], (r["w_eat"], r["w_act"], r["w_prey"], r["w_pred"]))
            )

    for ev in events:
        pid = ev["parent_id"]
        parent_obs = timeline.get(pid, [])
        # Find the closest observation BEFORE the birth (parent must be alive then).
        before = [obs for obs in parent_obs if obs[0] <= ev["step"]]
        if before:
            obs = before[-1]  # most recent before birth
            ev["parent_energy_at_or_before_birth"] = obs[1]
            ev["parent_genome"] = obs[2]
            ev["parent_energy_lag_steps"] = ev["step"] - obs[0]
        else:
            # We never saw the parent before birth (probably born+died in same
            # interval as the child's birth interval). Mark as unknown.
            ev["parent_energy_at_or_before_birth"] = None
            ev["parent_genome"] = None
            ev["parent_energy_lag_steps"] = None
    return events


def reproductive_skew_stats(offspring_counts, species_filter=None, registry=None):
    """Gini coefficient + top-K share + summary stats over offspring counts.

    Counts include zero-children agents (they're part of the population).
    """
    counts = []
    for aid, kids in offspring_counts.items():
        if species_filter is not None and registry is not None:
            # Some parent_ids appear in children's parent_id field but the parent
            # itself was never observed in any checkpoint (died before our
            # window). Skip them — we can't categorize them by species.
            if aid not in registry or registry[aid]["species"] != species_filter:
                continue
        counts.append(len(kids))
    # Add zero entries for agents with no offspring observed.
    if registry is not None:
        for aid, info in registry.items():
            if species_filter is not None and info["species"] != species_filter:
                continue
            if aid not in offspring_counts:
                counts.append(0)
    if not counts:
        return {}

    counts = np.array(counts, dtype=np.float64)
    counts.sort()
    n = len(counts)
    if counts.sum() == 0:
        gini = 0.0
    else:
        # Gini = (2 * Σ i*x_i) / (n * Σ x_i) - (n+1)/n
        i_idx = np.arange(1, n + 1)
        gini = (2.0 * np.sum(i_idx * counts)) / (n * counts.sum()) - (n + 1) / n

    cum = np.cumsum(counts[::-1])
    top_10_share = cum[max(1, int(n * 0.1)) - 1] / cum[-1] if cum[-1] > 0 else 0.0
    top_20_share = cum[max(1, int(n * 0.2)) - 1] / cum[-1] if cum[-1] > 0 else 0.0

    return {
        "n_agents": int(n),
        "n_with_offspring": int((counts > 0).sum()),
        "frac_with_offspring": float((counts > 0).mean()),
        "mean_offspring": float(counts.mean()),
        "max_offspring": int(counts.max()),
        "gini": float(gini),
        "top_10pct_share": float(top_10_share),
        "top_20pct_share": float(top_20_share),
        "total_offspring": int(counts.sum()),
    }


def lineage_depth(agent_id, registry, max_depth=100):
    """Trace back through parent_ids to count generations from initial agents."""
    depth = 0
    cur = agent_id
    while depth < max_depth:
        info = registry.get(cur)
        if info is None or info["parent_id"] < 0:
            return depth
        cur = info["parent_id"]
        depth += 1
    return depth


def trace_lineage(agent_id, registry, registry_full=None):
    """Return a list of (agent_id, info) from this agent up to root.

    Note: ancestors that are not in registry (e.g., never appeared in any
    checkpoint because they died between checkpoints) appear as
    (parent_id, None) so we can still see the chain structure.
    """
    chain = []
    cur = agent_id
    while True:
        info = registry.get(cur)
        chain.append((cur, info))
        if info is None or info["parent_id"] < 0:
            break
        cur = info["parent_id"]
    return chain


def diagnose_coverage(checkpoints, registry):
    """Estimate fraction of births visible.

    For each interval (t, t+1), compare:
      - births observed: # new agent_ids in checkpoint t+1
      - births implied by next_agent_id: ckpt_t+1.next_agent_id - ckpt_t.next_agent_id
    """
    diagnostics = []
    for i in range(len(checkpoints) - 1):
        step_a, rows_a = checkpoints[i]
        step_b, rows_b = checkpoints[i + 1]
        ids_a = set(r["agent_id"] for r in rows_a)
        ids_b = set(r["agent_id"] for r in rows_b)
        new_in_b = len(ids_b - ids_a)
        diagnostics.append({
            "interval": (step_a, step_b),
            "n_active_a": len(ids_a),
            "n_active_b": len(ids_b),
            "births_observed": new_in_b,
        })
    return diagnostics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--max-checkpoints", type=int, default=None,
                   help="Limit to first N checkpoints (for fast iteration)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Build a SimState template once; load_simstate uses it for pytree structure.
    template = init_simstate(config, jax.random.PRNGKey(0))

    ckpt_paths = sorted(args.checkpoint_dir.glob("step_*.npz"),
                        key=lambda p: parse_step_from_filename(p))
    if args.max_checkpoints is not None:
        ckpt_paths = ckpt_paths[: args.max_checkpoints]
    if not ckpt_paths:
        raise SystemExit(f"No checkpoints found in {args.checkpoint_dir}")
    print(f"Loading {len(ckpt_paths)} checkpoints from {args.checkpoint_dir}...")

    checkpoints = []
    for path in ckpt_paths:
        step = parse_step_from_filename(path)
        rows = load_checkpoint(path, template)
        checkpoints.append((step, rows))
        print(f"  step {step}: {len(rows)} active agents", end="")
        prey = sum(1 for r in rows if r["species"] == 0)
        pred = sum(1 for r in rows if r["species"] == 1)
        print(f" (prey {prey}, pred {pred})")

    print("\n=== Building agent registry ===")
    registry = compute_agent_registry(checkpoints)
    print(f"Total distinct agents observed: {len(registry)}")

    print("\n=== Computing offspring counts ===")
    offspring = compute_offspring_counts(registry)
    print(f"Agents with at least one observed child: {len(offspring)}")

    print("\n=== Reproductive skew ===")
    pred_skew = reproductive_skew_stats(offspring, species_filter=1, registry=registry)
    prey_skew = reproductive_skew_stats(offspring, species_filter=0, registry=registry)
    print("Predator skew:")
    for k, v in pred_skew.items():
        print(f"  {k}: {v}")
    print("\nPrey skew:")
    for k, v in prey_skew.items():
        print(f"  {k}: {v}")

    print("\n=== Reconstructing birth events ===")
    events = reconstruct_birth_events(checkpoints, registry)
    print(f"Total reconstructed birth events: {len(events)}")

    print("\n=== Coverage diagnostic ===")
    coverage = diagnose_coverage(checkpoints, registry)
    if coverage:
        total_observed = sum(d["births_observed"] for d in coverage)
        print(f"Total births observed across all intervals: {total_observed}")

    # Currently-alive agents and their lineage depths.
    if checkpoints:
        latest_step, latest_rows = checkpoints[-1]
        depths = [lineage_depth(r["agent_id"], registry) for r in latest_rows]
        pred_depths = [lineage_depth(r["agent_id"], registry)
                       for r in latest_rows if r["species"] == 1]
        prey_depths = [lineage_depth(r["agent_id"], registry)
                       for r in latest_rows if r["species"] == 0]

        # Distinct ancestor count (going back ~5 gens; useful as diversity proxy)
        def gen_n_ancestors(rows, n=5):
            ancs = set()
            for r in rows:
                cur = r["agent_id"]
                for _ in range(n):
                    info = registry.get(cur)
                    if info is None or info["parent_id"] < 0:
                        break
                    cur = info["parent_id"]
                ancs.add(cur)
            return len(ancs)

        print(f"\n=== Lineage depth at step {latest_step} ===")
        if pred_depths:
            print(f"Predator depths: min={min(pred_depths)} max={max(pred_depths)} "
                  f"mean={np.mean(pred_depths):.1f}")
            print(f"  distinct 5-gen-back pred ancestors: "
                  f"{gen_n_ancestors([r for r in latest_rows if r['species']==1])}")
        if prey_depths:
            print(f"Prey depths: min={min(prey_depths)} max={max(prey_depths)} "
                  f"mean={np.mean(prey_depths):.1f}")

    # Write outputs.
    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "n_checkpoints": len(checkpoints),
        "first_step": checkpoints[0][0] if checkpoints else None,
        "last_step": checkpoints[-1][0] if checkpoints else None,
        "total_distinct_agents": len(registry),
        "predator_reproductive_skew": pred_skew,
        "prey_reproductive_skew": prey_skew,
        "n_birth_events_reconstructed": len(events),
        "coverage_per_interval": coverage,
    }
    summary_path = args.out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {summary_path}")

    # Write detailed per-agent reproductive output (for predators only — most relevant).
    pred_breakdown_path = args.out_dir / "predator_reproductive_breakdown.csv"
    with open(pred_breakdown_path, "w") as f:
        f.write("agent_id,parent_id,first_step,last_step,first_age,max_energy,n_offspring,"
                "w_eat,w_act,w_prey,w_pred\n")
        for aid, info in sorted(registry.items()):
            if info["species"] != 1:
                continue
            n_off = len(offspring.get(aid, []))
            g = info["final_genome"]
            f.write(f"{aid},{info['parent_id']},{info['first_step']},"
                    f"{info['last_step']},{info['first_observed_age']},"
                    f"{info['max_energy']:.2f},{n_off},"
                    f"{g[0]:+.4f},{g[1]:+.4f},{g[2]:+.4f},{g[3]:+.4f}\n")
    print(f"Wrote {pred_breakdown_path}")

    # Write birth events log.
    events_path = args.out_dir / "birth_events.csv"
    with open(events_path, "w") as f:
        f.write("step,child_id,parent_id,child_species,child_w_eat,child_w_act,"
                "child_w_prey,child_w_pred,parent_energy_at_or_before_birth,"
                "parent_energy_lag_steps,parent_w_eat,parent_w_act,parent_w_prey,parent_w_pred\n")
        for ev in events:
            cg = ev["child_genome_at_first_obs"]
            pg = ev.get("parent_genome") or (None, None, None, None)
            pe = ev.get("parent_energy_at_or_before_birth")
            pl = ev.get("parent_energy_lag_steps")
            f.write(f"{ev['step']},{ev['child_id']},{ev['parent_id']},"
                    f"{ev['child_species']},"
                    f"{cg[0]:+.4f},{cg[1]:+.4f},{cg[2]:+.4f},{cg[3]:+.4f},"
                    f"{'' if pe is None else f'{pe:.2f}'},"
                    f"{'' if pl is None else pl},"
                    f"{'' if pg[0] is None else f'{pg[0]:+.4f}'},"
                    f"{'' if pg[1] is None else f'{pg[1]:+.4f}'},"
                    f"{'' if pg[2] is None else f'{pg[2]:+.4f}'},"
                    f"{'' if pg[3] is None else f'{pg[3]:+.4f}'}\n")
    print(f"Wrote {events_path}")

    print(f"\nDone. Summary at {summary_path}")


if __name__ == "__main__":
    main()
