"""
reward_nonlinearity.py
----------------------
Probe how nonlinear an evolved residual-MLP reward genome is, for a single
predator from an axis1_residual run.

Pipeline:
  1. Load a SimState checkpoint, pick one active predator.
  2. Sample N stimulus vectors [n_eaten, motor_norm, s_prey, s_pred] uniformly
     across plausible ranges, run them through compute_residual_reward, and
     also through the linear-only and residual-only branches.
  3. Multiple regression of full reward on:
        - 4 linear terms
        - 4 squares
        - 6 pairwise interactions
     Compare R^2 of {linear-only} vs {linear + nonlinear} fit. The gap is
     the share of the response that genuinely needs nonlinearity to fit.
  4. For each input feature, sweep that feature with the others held at
     midpoints and plot the response (with a degree-3 fit overlaid).

Run from the repo root:
    python analysis/reward_nonlinearity.py
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from analysis.checkpoint_explorer import load, residual_l1_per_agent
from src.reward import compute_residual_reward

CKPT = "ckpts/axis1_residual/step_00360000.npz"
CONFIG = "configs/axis1/med.yaml"
N_SAMPLES = 4000
SEED = 0
FEATURES = ["n_eaten", "motor_norm", "s_prey", "s_pred"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slice_pytree(tree, idx):
    """Take leaf[idx] out of every leaf in a stacked Flax pytree."""
    return jtu.tree_map(lambda x: x[idx], tree)


def _pick_agent(state, species_id: int):
    """Return the slot of the active agent (in given species) with the
    largest residual-MLP L1.

    We want an agent whose MLP genome has actually drifted away from the
    zero init — founders born at t=0 still carry zero residuals because
    mutations only fire at birth, and a zero MLP would make this whole
    nonlinearity probe vacuous.
    """
    is_active = np.asarray(state.is_active)
    species = np.asarray(state.species)
    ages = np.asarray(state.ages)
    l1 = residual_l1_per_agent(state)
    mask = is_active & (species == species_id)
    slots = np.where(mask)[0]
    if slots.size == 0:
        raise RuntimeError(f"no active agents with species={species_id}")
    chosen = slots[int(np.argmax(l1[slots]))]
    return int(chosen), float(l1[chosen]), int(ages[chosen])


def _sample_stimuli(rng):
    """Draw N stimulus vectors covering the input space the MLP was trained on.

    n_eaten: integer count of catches per step. Real distribution is heavily
        skewed to 0 (catches are rare). Sample {0,1,2,3} with weights
        [.85, .12, .025, .005] to stay close to training while still
        exercising the nonzero regime.
    motor_norm: ||a||/F_max — bounded near [0, 1.5]. Use U[0, 1.2] to cover
        the typical envelope (norms slightly exceeding 1 are seen because
        F_max caps the policy output but doesn't hard-clip rewards).
    s_prey, s_pred: aggregated proximity in [0, 1] (clipped). Use U[0, 1].
    """
    n_eaten = rng.choice([0, 1, 2, 3], size=N_SAMPLES,
                         p=[0.85, 0.12, 0.025, 0.005]).astype(np.float32)
    motor = rng.uniform(0.0, 1.2, size=N_SAMPLES).astype(np.float32)
    sp = rng.uniform(0.0, 1.0, size=N_SAMPLES).astype(np.float32)
    sd = rng.uniform(0.0, 1.0, size=N_SAMPLES).astype(np.float32)
    return np.stack([n_eaten, motor, sp, sd], axis=1)


def _design_matrix(X):
    """Build [linear | squares | pairwise interactions], plus column names.

    All columns are mean-centered and unit-scaled so coefficients are
    comparable across heterogeneously scaled features.
    """
    cols = []
    names = []
    # linear
    for j, n in enumerate(FEATURES):
        cols.append(X[:, j])
        names.append(n)
    # squares
    for j, n in enumerate(FEATURES):
        cols.append(X[:, j] ** 2)
        names.append(f"{n}^2")
    # pairwise interactions
    for j, k in combinations(range(4), 2):
        cols.append(X[:, j] * X[:, k])
        names.append(f"{FEATURES[j]}*{FEATURES[k]}")
    M = np.stack(cols, axis=1)
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Mz = (M - mu) / sd
    return Mz, names, mu, sd


def _ols(D, y):
    """OLS via lstsq with a leading bias column. Returns (beta, R^2)."""
    A = np.concatenate([np.ones((D.shape[0], 1)), D], axis=1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return beta, r2, yhat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", choices=["prey", "predator"], default="predator")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--config", default=CONFIG)
    args = ap.parse_args()
    species_id = 0 if args.species == "prey" else 1
    suffix = args.species

    print(f"Loading {args.ckpt} (config={args.config})…")
    state = load(args.ckpt, config=args.config)
    slot, l1, age = _pick_agent(state, species_id)
    aid = int(np.asarray(state.agent_ids)[slot])
    linear_w = np.asarray(state.reward_weights)[slot]
    print(f"Picked {args.species}: slot={slot}  agent_id={aid}  age={age:,}  "
          f"residual L1={l1:.3f}")
    print(f"  Linear weights [w_eat, w_act, w_prey, w_pred] = "
          f"{np.round(linear_w, 3).tolist()}")

    residual_geno = _slice_pytree(state.reward_mlp_params, slot)
    linear_geno = jnp.asarray(linear_w)

    @jax.jit
    def reward_fn(stim):
        return compute_residual_reward(linear_geno, residual_geno, stim)

    @jax.jit
    def residual_only_fn(stim):
        # subtract the linear part to isolate the MLP residual
        coefs = jnp.array([1.0, 0.01, 0.1, 0.1])
        return compute_residual_reward(linear_geno, residual_geno, stim) \
               - jnp.sum(linear_geno * stim * coefs)

    rng = np.random.default_rng(SEED)
    X = _sample_stimuli(rng)
    print(f"\nSampling N={N_SAMPLES} stimulus vectors…")
    y_full = np.asarray(jax.vmap(reward_fn)(jnp.asarray(X)))
    y_resid = np.asarray(jax.vmap(residual_only_fn)(jnp.asarray(X)))
    y_lin = y_full - y_resid

    print(f"  y_full   range [{y_full.min():+.3f}, {y_full.max():+.3f}]  "
          f"std={y_full.std():.3f}")
    print(f"  y_linear range [{y_lin.min():+.3f}, {y_lin.max():+.3f}]  "
          f"std={y_lin.std():.3f}")
    print(f"  y_resid  range [{y_resid.min():+.3f}, {y_resid.max():+.3f}]  "
          f"std={y_resid.std():.3f}")
    print(f"  share of full-reward variance from MLP residual: "
          f"{(y_resid.var()/y_full.var())*100:.1f}%")

    # --- regressions on the FULL reward ---
    D_full, names, _, _ = _design_matrix(X)
    D_lin = D_full[:, :4]                # linear-only
    D_sq = D_full[:, :8]                 # linear + squares
    D_all = D_full                       # linear + squares + interactions

    _, r2_lin, _ = _ols(D_lin, y_full)
    _, r2_sq, _ = _ols(D_sq, y_full)
    beta, r2_all, yhat_all = _ols(D_all, y_full)

    # --- regression on the RESIDUAL (no linear part) ---
    _, r2_resid_lin, _ = _ols(D_lin, y_resid)
    beta_resid, r2_resid_all, _ = _ols(D_all, y_resid)

    print("\n=== Regression R^2 on FULL reward ===")
    print(f"  linear only            R^2 = {r2_lin:.4f}")
    print(f"  + squares              R^2 = {r2_sq:.4f}  "
          f"(Δ = {r2_sq - r2_lin:+.4f})")
    print(f"  + squares + interact   R^2 = {r2_all:.4f}  "
          f"(Δ = {r2_all - r2_sq:+.4f})")
    print(f"  → fraction of variance NOT explained by a linear fit: "
          f"{(1.0 - r2_lin)*100:.2f}%")

    print("\n=== Regression R^2 on RESIDUAL (MLP only) ===")
    print(f"  linear only            R^2 = {r2_resid_lin:.4f}")
    print(f"  + squares + interact   R^2 = {r2_resid_all:.4f}")

    # standardized coefficients on the FULL reward (drop bias column)
    beta_no_bias = beta[1:]
    # standardized coefficients on the RESIDUAL
    beta_resid_no_bias = beta_resid[1:]

    print("\n=== Top standardized coefficients (full reward, |β| sorted) ===")
    idx = np.argsort(-np.abs(beta_no_bias))
    for i in idx[:12]:
        print(f"  {names[i]:30s}  β = {beta_no_bias[i]:+.4f}")

    print("\n=== Top standardized coefficients (residual only) ===")
    idx_r = np.argsort(-np.abs(beta_resid_no_bias))
    for i in idx_r[:12]:
        print(f"  {names[i]:30s}  β = {beta_resid_no_bias[i]:+.4f}")

    # --- single-variable sweeps ---
    out_path = _ROOT / "analysis" / f"reward_nonlinearity_{suffix}.png"
    _plot_sweeps(reward_fn, residual_only_fn, slot, aid, age, linear_w,
                 r2_lin, r2_all, out_path, species_label=args.species)
    print(f"\nWrote {out_path}")


def _plot_sweeps(reward_fn, residual_only_fn, slot, aid, age, linear_w,
                 r2_lin, r2_all, out_path, species_label="predator"):
    """2×4 grid: each column is one feature; rows are full reward (top) and
    residual-only (bottom).

    On every panel:
      - blue solid line = actual signal (MLP for top row, residual for bottom)
      - orange dashed line = best 1-D linear fit
      - green dotted line = cubic fit
    Cubic - linear gap = visual measure of nonlinearity in *that* slice.
    """
    midpoints = np.array([0.0, 0.5, 0.5, 0.5])  # n_eaten=0 dominates the data
    sweep_grids = [
        np.linspace(0.0, 3.0, 200),
        np.linspace(0.0, 1.2, 200),
        np.linspace(0.0, 1.0, 200),
        np.linspace(0.0, 1.0, 200),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for j in range(4):
        grid = sweep_grids[j]
        X = np.tile(midpoints, (grid.size, 1)).astype(np.float32)
        X[:, j] = grid
        y_full = np.asarray(jax.vmap(reward_fn)(jnp.asarray(X)))
        y_resid = np.asarray(jax.vmap(residual_only_fn)(jnp.asarray(X)))

        for row, (ax, y, label) in enumerate(
            [(axes[0, j], y_full, "full reward"),
             (axes[1, j], y_resid, "residual MLP only")]
        ):
            # 1-D linear and cubic fits to whatever slice we're plotting
            lin_coefs = np.polyfit(grid, y, 1)
            y_lin = np.polyval(lin_coefs, grid)
            cub_coefs = np.polyfit(grid, y, 3)
            y_cub = np.polyval(cub_coefs, grid)

            # in-slice R^2 of linear vs cubic — the gap quantifies the
            # nonlinearity along this single axis
            ss_tot = np.sum((y - y.mean()) ** 2) + 1e-30
            r2_lin_slice = 1.0 - np.sum((y - y_lin) ** 2) / ss_tot
            r2_cub_slice = 1.0 - np.sum((y - y_cub) ** 2) / ss_tot

            ax.plot(grid, y, "-", color="C0", lw=2.2, label=label)
            ax.plot(grid, y_lin, "--", color="C1", lw=1.4,
                    label=f"linear fit  R²={r2_lin_slice:.3f}")
            ax.plot(grid, y_cub, ":", color="C2", lw=1.6,
                    label=f"cubic fit  R²={r2_cub_slice:.3f}")
            ax.axvline(midpoints[j], color="gray", lw=0.8, alpha=0.5)
            ax.set_title(f"{FEATURES[j]} — {label}")
            ax.set_xlabel(FEATURES[j])
            ax.set_ylabel("reward")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="best")

    title = (f"{species_label.capitalize()} slot={slot}  agent_id={aid}  age={age:,}  "
             f"linear_w={np.round(linear_w, 2).tolist()}\n"
             f"FULL reward: linear-only R²={r2_lin:.4f}   "
             f"+ squares + interactions R²={r2_all:.4f}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
