"""
reward_nonlinearity_population.py
---------------------------------
Run the per-agent nonlinearity analysis (regress reward outputs on linear
+ square + interaction features) across the entire population of one
species, then average.

For each agent we report:
  - residual_L1             : L1 of the residual MLP genome
  - sigma_full, sigma_resid : reward std with / without the linear K&D part
  - r2_full_lin             : R^2 of a pure-linear fit to the FULL reward
  - r2_full_all             : R^2 of lin + squares + interactions
  - r2_resid_lin            : R^2 of a pure-linear fit to the RESIDUAL only
  - r2_resid_all            : R^2 with squares + interactions on the residual
  - standardized betas      : 14 coefficients

Population-level outputs:
  - mean and quantiles for each per-agent metric
  - mean standardized betas (bar chart, ±1 std)
  - average single-variable sweep curves with p25..p75 band

Run:
    python analysis/reward_nonlinearity_population.py --species prey
    python analysis/reward_nonlinearity_population.py --species predator
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
# Sampling and design matrix (same shape as in reward_nonlinearity.py)
# ---------------------------------------------------------------------------

def _sample_stimuli(rng):
    n_eaten = rng.choice([0, 1, 2, 3], size=N_SAMPLES,
                         p=[0.85, 0.12, 0.025, 0.005]).astype(np.float32)
    motor = rng.uniform(0.0, 1.2, size=N_SAMPLES).astype(np.float32)
    sp = rng.uniform(0.0, 1.0, size=N_SAMPLES).astype(np.float32)
    sd = rng.uniform(0.0, 1.0, size=N_SAMPLES).astype(np.float32)
    return np.stack([n_eaten, motor, sp, sd], axis=1)


def _design_matrix(X):
    cols, names = [], []
    for j, n in enumerate(FEATURES):
        cols.append(X[:, j]); names.append(n)
    for j, n in enumerate(FEATURES):
        cols.append(X[:, j] ** 2); names.append(f"{n}^2")
    for j, k in combinations(range(4), 2):
        cols.append(X[:, j] * X[:, k]); names.append(f"{FEATURES[j]}*{FEATURES[k]}")
    M = np.stack(cols, axis=1)
    sd = M.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Mz = (M - M.mean(axis=0)) / sd
    return Mz, names


def _ols_r2_and_beta(D, y):
    """OLS with bias column; returns (R^2, beta_no_bias)."""
    A = np.concatenate([np.ones((D.shape[0], 1)), D], axis=1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot < 1e-30:
        return float("nan"), beta[1:]
    return 1.0 - ss_res / ss_tot, beta[1:]


# ---------------------------------------------------------------------------
# Population analysis
# ---------------------------------------------------------------------------

def _slice_pytree(tree, idx):
    return jtu.tree_map(lambda x: x[idx], tree)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", choices=["prey", "predator"], default="prey")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--config", default=CONFIG)
    args = ap.parse_args()
    species_id = 0 if args.species == "prey" else 1

    print(f"Loading {args.ckpt} (config={args.config})…")
    state = load(args.ckpt, config=args.config)

    is_active = np.asarray(state.is_active)
    species = np.asarray(state.species)
    ages = np.asarray(state.ages)
    rw = np.asarray(state.reward_weights)
    l1 = residual_l1_per_agent(state)

    slots = np.where(is_active & (species == species_id))[0]
    n_agents = slots.size
    print(f"Found {n_agents} active {args.species}.")

    rng = np.random.default_rng(SEED)
    X = _sample_stimuli(rng)
    D_full, names = _design_matrix(X)
    D_lin = D_full[:, :4]

    # vmap over agents AND samples: for one agent fn, vmap over samples; then
    # vmap that over agents. Result is (n_agents, n_samples) reward matrix.
    coefs_kd = jnp.array([1.0, 0.01, 0.1, 0.1])

    def _per_agent(linear_w, residual_geno, X_jax):
        def _per_sample(stim):
            full = compute_residual_reward(linear_w, residual_geno, stim)
            lin = jnp.sum(linear_w * stim * coefs_kd)
            return full, full - lin    # (full, residual_only)
        return jax.vmap(_per_sample)(X_jax)

    @jax.jit
    def _all_agents(linear_ws, residual_genos, X_jax):
        return jax.vmap(_per_agent, in_axes=(0, 0, None))(
            linear_ws, residual_genos, X_jax
        )

    print(f"Sampling N={N_SAMPLES} stimulus vectors and running through "
          f"all {n_agents} {args.species} reward functions…")
    linear_ws = jnp.asarray(rw[slots])
    residual_genos = _slice_pytree(state.reward_mlp_params, slots)
    y_full_all, y_resid_all = _all_agents(
        linear_ws, residual_genos, jnp.asarray(X)
    )
    y_full_all = np.asarray(y_full_all)    # (n_agents, n_samples)
    y_resid_all = np.asarray(y_resid_all)

    # Per-agent regression stats
    sigma_full = y_full_all.std(axis=1)
    sigma_resid = y_resid_all.std(axis=1)
    r2_full_lin = np.empty(n_agents)
    r2_full_all_arr = np.empty(n_agents)
    r2_resid_lin = np.full(n_agents, np.nan)
    r2_resid_all = np.full(n_agents, np.nan)
    betas_full = np.empty((n_agents, D_full.shape[1]))
    betas_resid = np.full((n_agents, D_full.shape[1]), np.nan)

    for i in range(n_agents):
        r2_full_lin[i], _ = _ols_r2_and_beta(D_lin, y_full_all[i])
        r2_full_all_arr[i], betas_full[i] = _ols_r2_and_beta(D_full, y_full_all[i])
        if sigma_resid[i] > 1e-12:
            r2_resid_lin[i], _ = _ols_r2_and_beta(D_lin, y_resid_all[i])
            r2_resid_all[i], betas_resid[i] = _ols_r2_and_beta(D_full, y_resid_all[i])

    # ----- print population summary -----
    nz = sigma_resid > 1e-12
    n_zero = int((~nz).sum())
    print()
    print(f"=== Population summary ({args.species}, n={n_agents}, "
          f"{n_zero} with zero residual) ===")
    _stat("residual L1",     l1[slots])
    _stat("sigma_full",      sigma_full)
    _stat("sigma_resid",     sigma_resid)
    _stat("var share of MLP residual (%)",
          100.0 * (sigma_resid ** 2) / (sigma_full ** 2 + 1e-30))
    print()
    _stat("R^2 (full reward, linear-only)",          r2_full_lin)
    _stat("R^2 (full reward, lin+sq+int)",           r2_full_all_arr)
    print()
    _stat("R^2 (residual,    linear-only)",          r2_resid_lin[nz])
    _stat("R^2 (residual,    lin+sq+int)",           r2_resid_all[nz])
    nonlin_gain = r2_resid_all[nz] - r2_resid_lin[nz]
    _stat("Δ R^2 (residual,  sq+int over linear)",   nonlin_gain)

    print(f"\n=== Mean standardized β across {nz.sum()} non-zero-residual {args.species} ===")
    print(f"Reported as: mean ± std    (sign convention: + raises reward)")
    mean_b = np.nanmean(betas_resid, axis=0)
    std_b = np.nanstd(betas_resid, axis=0)
    print("\n  -- on residual --")
    order = np.argsort(-np.abs(mean_b))
    for i in order:
        print(f"  {names[i]:30s}  β = {mean_b[i]:+.4f}  ±{std_b[i]:.4f}")

    print("\n  -- on full reward --")
    mean_bf = betas_full.mean(axis=0)
    std_bf = betas_full.std(axis=0)
    order_f = np.argsort(-np.abs(mean_bf))
    for i in order_f:
        print(f"  {names[i]:30s}  β = {mean_bf[i]:+.4f}  ±{std_bf[i]:.4f}")

    out_path = _ROOT / "analysis" / f"reward_nonlinearity_population_{args.species}.png"
    _plot_population(args.species, n_agents, l1[slots],
                     sigma_full, sigma_resid,
                     r2_full_lin, r2_full_all_arr,
                     r2_resid_lin, r2_resid_all,
                     betas_resid, names,
                     y_full_all, y_resid_all, X, out_path)
    print(f"\nWrote {out_path}")


def _stat(name, x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        print(f"  {name:42s}  (no finite values)")
        return
    print(f"  {name:42s}  mean={x.mean():+.4f}  med={np.median(x):+.4f}  "
          f"p25={np.percentile(x,25):+.4f}  p75={np.percentile(x,75):+.4f}  "
          f"min={x.min():+.4f}  max={x.max():+.4f}")


def _plot_population(species, n_agents, l1,
                     sigma_full, sigma_resid,
                     r2_full_lin, r2_full_all,
                     r2_resid_lin, r2_resid_all,
                     betas_resid, names,
                     y_full_all, y_resid_all, X, out_path):
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

    # Row 1: histograms of per-agent stats
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(l1, bins=40, color="C0")
    ax.set_title("residual L1")
    ax.set_xlabel("L1")
    ax.set_ylabel("agents")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    var_share = 100.0 * (sigma_resid ** 2) / (sigma_full ** 2 + 1e-30)
    ax.hist(var_share, bins=40, color="C0")
    ax.set_title("MLP share of full-reward variance (%)")
    ax.set_xlabel("%")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(r2_full_lin, bins=40, color="C0")
    ax.set_title("R² linear-only fit to FULL reward")
    ax.set_xlabel("R²")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 3])
    nz = np.isfinite(r2_resid_lin)
    ax.hist(r2_resid_lin[nz], bins=40, color="C0", alpha=0.6, label="linear")
    ax.hist(r2_resid_all[nz], bins=40, color="C2", alpha=0.6, label="lin+sq+int")
    ax.set_title("R² fit to RESIDUAL only")
    ax.set_xlabel("R²")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 2: average standardized betas on the residual (mean ± std)
    ax = fig.add_subplot(gs[1, :])
    mean_b = np.nanmean(betas_resid, axis=0)
    std_b = np.nanstd(betas_resid, axis=0)
    order = np.argsort(-np.abs(mean_b))
    ordered_names = [names[i] for i in order]
    ax.bar(range(len(order)), mean_b[order], yerr=std_b[order],
           color=["C0" if i < 4 else "C2" if i < 8 else "C1" for i in order],
           capsize=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(ordered_names, rotation=35, ha="right")
    ax.set_ylabel("standardized β  (residual)")
    ax.set_title(f"Mean ±1 std of standardized regression coefficients on RESIDUAL "
                 f"(over {nz.sum()} {species} with non-zero residual). "
                 f"Blue=linear, Green=square, Orange=interaction.")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(alpha=0.3, axis="y")

    # Row 3: per-feature 1-D sweep — population mean and p25..p75 band
    midpoints = np.array([0.0, 0.5, 0.5, 0.5])
    sweep_grids = [
        np.linspace(0.0, 3.0, 100),
        np.linspace(0.0, 1.2, 100),
        np.linspace(0.0, 1.0, 100),
        np.linspace(0.0, 1.0, 100),
    ]
    # Re-sweep all agents through these 1-D grids, residual only.
    for j in range(4):
        grid = sweep_grids[j]
        # Build the (grid_size, 4) sweep input.
        sweep_X = np.tile(midpoints, (grid.size, 1)).astype(np.float32)
        sweep_X[:, j] = grid
        # For each agent, residual at every grid point. We have not stored
        # the per-agent reward fns separately so just re-run the same vmap
        # design-matrix machinery: residual = full - linear with K&D coefs.
        # Because re-running JAX from this scope is cumbersome, approximate
        # by *projecting* via the fitted lin+sq+int model — but that
        # discards the genuine network curvature we want to show. Instead,
        # we use the already-stored y_resid_all and condition: pick the
        # subset of samples in X that are within an epsilon ball of the
        # fixed midpoints in the OTHER 3 features.
        eps = 0.06 if j == 0 else 0.18
        keep = np.ones(X.shape[0], dtype=bool)
        for k in range(4):
            if k == j:
                continue
            keep &= np.abs(X[:, k] - midpoints[k]) < eps
        if keep.sum() < 10:
            # Loosen epsilon if the strict band has too few samples.
            for k in range(4):
                if k == j:
                    continue
                keep &= np.abs(X[:, k] - midpoints[k]) < 0.3
        x_pts = X[keep, j]
        order_x = np.argsort(x_pts)
        x_pts = x_pts[order_x]
        y_pts = y_resid_all[:, keep][:, order_x]   # (n_agents, n_kept)
        med = np.median(y_pts, axis=0)
        p25 = np.percentile(y_pts, 25, axis=0)
        p75 = np.percentile(y_pts, 75, axis=0)
        # smooth the median with a small running mean for the line plot
        if med.size > 5:
            kernel = np.ones(5) / 5.0
            med_s = np.convolve(med, kernel, mode="same")
        else:
            med_s = med

        ax = fig.add_subplot(gs[2, j])
        ax.fill_between(x_pts, p25, p75, alpha=0.25, color="C0",
                        label="p25..p75")
        ax.plot(x_pts, med_s, color="C0", lw=2, label="median")
        ax.axvline(midpoints[j], color="gray", lw=0.8, alpha=0.5)
        ax.set_title(f"residual MLP vs {FEATURES[j]} (others≈midpoint)")
        ax.set_xlabel(FEATURES[j])
        ax.set_ylabel("residual reward")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Population-level reward-nonlinearity analysis: "
        f"{species} (n={n_agents}, ckpt step 360k, axis1_residual)",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
