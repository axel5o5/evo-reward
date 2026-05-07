"""
axis2_other_velocity_weights.py
-------------------------------
For an axis2 (bin-aligned heading) checkpoint, ask: do any agents'
MLP policies actually *use* the inputs that encode other creatures'
heading/velocity?

axis2_aligned obs layout (333 dims):
  Per proximity bin (8 chans): [prey_dist, prey_sin_h, prey_cos_h,
                                 pred_dist, pred_sin_h, pred_cos_h,
                                 food_dist, wall_dist]
  → 32 bins × 8 = 256 dims (indices 0..255)
  Tactile: 72 dims (256..327)
  Self:    5 dims  (328=vel_x, 329=vel_y, 330=angle, 331=ang_vel, 332=energy)

The "other-creature velocity" inputs are the heading sin/cos channels —
indices {bin*8 + 1, bin*8 + 2} (prey heading) and {bin*8 + 4, bin*8 + 5}
(predator heading) across all 32 bins.

We compare per-agent input-layer weight magnitude on these channels
against:
  - the corresponding distance channels (known to be useful);
  - other input groups (food, wall, tactile, self);
  - a random-init reference (Flax LeCun-normal: row-norm at init).

Run:
    python analysis/axis2_other_velocity_weights.py \\
        --ckpt ckpts/axis2_aligned/step_00560000.npz \\
        --config configs/archive/axis2_aligned.yaml
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from analysis.checkpoint_explorer import load


DEFAULT_CKPT = "ckpts/axis2_aligned/step_00560000.npz"
DEFAULT_CONFIG = "configs/archive/axis2_aligned.yaml"


# ---------------------------------------------------------------------------
# Obs-index groupings
# ---------------------------------------------------------------------------

def build_index_groups(config: dict) -> dict[str, np.ndarray]:
    """Return obs-index arrays for each semantic group.

    Assumes proximity_encoding == 'distance_and_heading' (8 channels/bin).
    """
    enc = config.get("proximity_encoding", "distance_only")
    if enc != "distance_and_heading":
        raise ValueError(
            f"This script targets proximity_encoding='distance_and_heading'; "
            f"got {enc!r}. The channel layout differs for other encodings."
        )

    n_bins = config["n_proximity_sensors"]              # 32
    n_chans = config["n_proximity_channels"]            # 8
    n_tact = config["n_tactile_sensors"] * config["n_tactile_channels"]  # 72

    bins = np.arange(n_bins)

    def chan(c: int) -> np.ndarray:
        return bins * n_chans + c

    prox_end = n_bins * n_chans                          # 256
    tact_end = prox_end + n_tact                         # 328

    groups = {
        "prey_dist":      chan(0),
        "prey_heading":   np.concatenate([chan(1), chan(2)]),
        "pred_dist":      chan(3),
        "pred_heading":   np.concatenate([chan(4), chan(5)]),
        "food_dist":      chan(6),
        "wall_dist":      chan(7),
        "tactile":        np.arange(prox_end, tact_end),
        "self_vel_xy":    np.array([tact_end, tact_end + 1]),
        "self_other":     np.arange(tact_end + 2, tact_end + 5),
    }
    # sanity
    seen = np.concatenate(list(groups.values()))
    assert seen.size == config["obs_dim"], (
        f"index groups cover {seen.size} dims, obs_dim={config['obs_dim']}"
    )
    assert np.unique(seen).size == seen.size, "overlapping index groups"
    return groups


# ---------------------------------------------------------------------------
# Per-agent input-row norm
# ---------------------------------------------------------------------------

def per_input_norm(kernel: np.ndarray) -> np.ndarray:
    """Row-wise L2 norm of Dense_0 kernel.

    `kernel` shape: (n_agents, obs_dim, hidden). Returns (n_agents, obs_dim).
    The row norm tells how much that input feature drives the first hidden
    layer for that agent (any rotation in hidden space preserves it).
    """
    return np.linalg.norm(kernel, axis=2)


def init_row_norm_reference(obs_dim: int, hidden: int) -> float:
    """Expected per-input row norm at Flax-default init (LeCun-normal).

    Flax's nn.Dense default is lecun_normal: stddev = 1/sqrt(in_features).
    Row norm ≈ sqrt(hidden) * 1/sqrt(in_features).
    """
    return math.sqrt(hidden) / math.sqrt(obs_dim)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def group_stats(row_norm: np.ndarray, idx: np.ndarray) -> dict:
    """Per-agent group statistic: mean of row-norms over `idx`."""
    sub = row_norm[:, idx]
    per_agent_mean = sub.mean(axis=1)
    return {
        "per_agent_mean":  per_agent_mean,
        "pop_mean":        float(per_agent_mean.mean()),
        "pop_p25":         float(np.percentile(per_agent_mean, 25)),
        "pop_p50":         float(np.percentile(per_agent_mean, 50)),
        "pop_p75":         float(np.percentile(per_agent_mean, 75)),
        "pop_max":         float(per_agent_mean.max()),
        "max_agent":       int(np.argmax(per_agent_mean)),
    }


def print_block(species: str, row_norm: np.ndarray, groups: dict,
                ref: float) -> None:
    print(f"\n=== {species}: per-agent mean of input-row L2 norm by channel group "
          f"(n={row_norm.shape[0]}) ===")
    print(f"  random-init reference ≈ {ref:.4f}\n")
    print(f"  {'group':<18} {'pop_mean':>10} {'p25':>9} {'p50':>9} "
          f"{'p75':>9} {'max':>9}  {'×ref(median)':>14}")
    rows = []
    for name, idx in groups.items():
        s = group_stats(row_norm, idx)
        rows.append((name, s))
    # sort by population median descending for readability
    rows.sort(key=lambda kv: -kv[1]["pop_p50"])
    for name, s in rows:
        ratio = s["pop_p50"] / ref
        print(f"  {name:<18} {s['pop_mean']:>10.4f} {s['pop_p25']:>9.4f} "
              f"{s['pop_p50']:>9.4f} {s['pop_p75']:>9.4f} "
              f"{s['pop_max']:>9.4f}  {ratio:>13.2f}×")


def find_high_heading_agents(row_norm: np.ndarray, groups: dict,
                             ref: float, top_k: int = 5) -> list:
    """Top-k agents by combined other-species heading row-norm."""
    # "other-species heading" depends on caller; here just return per-agent
    # stats for both prey-heading and pred-heading.
    return [
        ("prey_heading",
         np.argsort(-row_norm[:, groups["prey_heading"]].mean(axis=1))[:top_k]),
        ("pred_heading",
         np.argsort(-row_norm[:, groups["pred_heading"]].mean(axis=1))[:top_k]),
    ]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_distributions(prey_rn, pred_rn, groups, ref, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    order = ["prey_dist", "prey_heading", "pred_dist", "pred_heading",
             "food_dist", "wall_dist", "tactile", "self_vel_xy", "self_other"]

    for ax, rn, label in [(axes[0], prey_rn, "prey policies"),
                          (axes[1], pred_rn, "predator policies")]:
        data = [rn[:, groups[g]].mean(axis=1) for g in order]
        bp = ax.boxplot(data, labels=order, showfliers=True, whis=(5, 95))
        ax.axhline(ref, color="red", linestyle="--", lw=1,
                   label=f"random-init ref ≈ {ref:.3f}")
        ax.set_ylabel("per-agent mean row-norm of Dense_0[input, :]")
        ax.set_title(f"axis2_aligned — input-layer attention by channel group ({label})")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="upper right", fontsize=8)
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   default=DEFAULT_CKPT)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--top-k",  type=int, default=5,
                    help="Print top-k agents per heading group")
    ap.add_argument("--out",    default=None,
                    help="Output PNG path (default: alongside the ckpt)")
    args = ap.parse_args()

    print(f"Loading {args.ckpt} (config={args.config})…")
    state = load(args.ckpt, config=args.config)

    import yaml
    with open(_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    groups = build_index_groups(cfg)
    obs_dim = cfg["obs_dim"]
    hidden = cfg["policy_hidden_size"]
    ref = init_row_norm_reference(obs_dim, hidden)

    is_active = np.asarray(state.is_active)
    species = np.asarray(state.species)

    kernel = np.asarray(
        state.policy_params["params"]["Dense_0"]["kernel"]
    )  # (max_agents, obs_dim, hidden)
    print(f"Dense_0 kernel shape: {kernel.shape}, obs_dim={obs_dim}, "
          f"hidden={hidden}")

    row_norm = per_input_norm(kernel)  # (max_agents, obs_dim)

    prey_slots = np.where(is_active & (species == 0))[0]
    pred_slots = np.where(is_active & (species == 1))[0]
    print(f"Active prey={prey_slots.size}, predators={pred_slots.size}")

    prey_rn = row_norm[prey_slots]
    pred_rn = row_norm[pred_slots]

    print_block("PREY",      prey_rn, groups, ref)
    print_block("PREDATORS", pred_rn, groups, ref)

    # Headline answer: for prey, focus on pred_heading; for predators, on
    # prey_heading. That is the cross-species "other creature's velocity"
    # info — the whole point of the axis2 encoding.
    print("\n=== Headline: cross-species heading attention vs distance ===")
    for label, rn, slots, head_g, dist_g in [
        ("PREY → predator-heading",
         prey_rn, prey_slots, "pred_heading", "pred_dist"),
        ("PREDATOR → prey-heading",
         pred_rn, pred_slots, "prey_heading", "prey_dist"),
    ]:
        h = rn[:, groups[head_g]].mean(axis=1)
        d = rn[:, groups[dist_g]].mean(axis=1)
        ratio_h = h / ref
        ratio_d = d / ref
        frac_above_2x = float((h > 2 * ref).mean())
        print(f"\n  {label}")
        print(f"    heading row-norm  median={np.median(h):.4f} "
              f"(× ref = {np.median(ratio_h):.2f}), "
              f"max={h.max():.4f} (× ref = {ratio_h.max():.2f})")
        print(f"    dist    row-norm  median={np.median(d):.4f} "
              f"(× ref = {np.median(ratio_d):.2f}), "
              f"max={d.max():.4f} (× ref = {ratio_d.max():.2f})")
        print(f"    fraction of agents with heading row-norm > 2× ref: "
              f"{frac_above_2x:.1%}")

        order_h = np.argsort(-h)[: args.top_k]
        print(f"    top {args.top_k} agents (slot, agent_id, heading-rn, dist-rn):")
        agent_ids = np.asarray(state.agent_ids)
        for j in order_h:
            slot = int(slots[j])
            print(f"      slot={slot:>4}  id={int(agent_ids[slot]):>6}  "
                  f"heading={h[j]:.4f}  dist={d[j]:.4f}")

    out_path = (
        Path(args.out) if args.out
        else Path(args.ckpt).with_suffix("").with_name(
            Path(args.ckpt).stem + "_input_weights.png"
        )
    )
    plot_distributions(prey_rn, pred_rn, groups, ref, out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
