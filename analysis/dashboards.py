"""
dashboards.py
-------------
Post-hoc plotting functions for evo-reward experiment results.

Three functions, each reading from a completed metrics.npz:
  1. plot_reward_trajectories  — K&D Figure 7 style
  2. plot_population_dynamics  — K&D Figure 6 style
  3. plot_reward_kde           — K&D Figure 8/12 style

Usage:
    from analysis.dashboards import plot_reward_trajectories, plot_population_dynamics, plot_reward_kde
    plot_reward_trajectories("results/baseline_faithful/seed_0/metrics.npz",
                             "results/baseline_faithful/seed_0/plots/reward_trajectories.png")
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_metrics(metrics_path):
    """Load metrics.npz and return dict of arrays."""
    data = np.load(metrics_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _ensure_dir(path):
    """Ensure parent directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_reward_trajectories(metrics_path, output_path):
    """Reproduce K&D Figure 7 style: mean +/- std of each reward weight over time.

    Four subplots (one per weight: w_eat, w_act, w_prey, w_pred), showing
    prey reward weight trajectories as mean +/- 1 std shaded band.
    X-axis: simulation steps. Horizontal dashed line at y=0 for reference.

    Args:
        metrics_path: Path to metrics.npz (single seed) or list of paths (multi-seed).
        output_path: Where to save the figure (e.g., .png or .pdf).
    """
    _ensure_dir(output_path)

    # Support single path or list of paths for multi-seed overlay
    if isinstance(metrics_path, str):
        metrics_path = [metrics_path]

    weight_names = ["w_eat", "w_act", "w_prey", "w_pred"]
    weight_labels = [r"$w_{\mathrm{eat}}$", r"$w_{\mathrm{act}}$",
                     r"$w_{\mathrm{prey}}$", r"$w_{\mathrm{pred}}$"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(metrics_path), 10)))

    for seed_idx, mpath in enumerate(metrics_path):
        m = _load_metrics(mpath)
        steps = m["steps"]
        color = colors[seed_idx]
        label = f"seed {seed_idx}" if len(metrics_path) > 1 else None

        for i, wname in enumerate(weight_names):
            ax = axes[i]
            mean = m[f"prey_mean_{wname}"]
            std = m[f"prey_std_{wname}"]

            ax.plot(steps, mean, color=color, linewidth=1.5, label=label)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)

    for i, (wname, wlabel) in enumerate(zip(weight_names, weight_labels)):
        ax = axes[i]
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_ylabel(f"Prey {wlabel}")
        ax.set_title(f"Prey {wlabel} over time")
        ax.grid(True, alpha=0.3)
        if i >= 2:
            ax.set_xlabel("Simulation step")
        if len(metrics_path) > 1:
            ax.legend(fontsize=8)

    fig.suptitle("Prey Reward Weight Evolution (K&D Fig. 7 style)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reward trajectories plot to {output_path}")


def plot_population_dynamics(metrics_path, output_path):
    """Reproduce K&D Figure 6 style: prey and predator population over time.

    Two subplots stacked vertically: prey population (top), predator population
    (bottom). X-axis: simulation steps. Should show Lotka-Volterra-like
    oscillations if replication succeeds.

    Args:
        metrics_path: Path to metrics.npz (single seed) or list of paths.
        output_path: Where to save the figure.
    """
    _ensure_dir(output_path)

    if isinstance(metrics_path, str):
        metrics_path = [metrics_path]

    fig, (ax_prey, ax_pred) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(metrics_path), 10)))

    for seed_idx, mpath in enumerate(metrics_path):
        m = _load_metrics(mpath)
        steps = m["steps"]
        color = colors[seed_idx]
        label = f"seed {seed_idx}" if len(metrics_path) > 1 else None

        ax_prey.plot(steps, m["prey_population"], color=color, linewidth=1.2, label=label)
        ax_pred.plot(steps, m["predator_population"], color=color, linewidth=1.2, label=label)

    ax_prey.set_ylabel("Prey population")
    ax_prey.set_title("Prey population over time")
    ax_prey.grid(True, alpha=0.3)
    if len(metrics_path) > 1:
        ax_prey.legend(fontsize=8)

    ax_pred.set_ylabel("Predator population")
    ax_pred.set_title("Predator population over time")
    ax_pred.set_xlabel("Simulation step")
    ax_pred.grid(True, alpha=0.3)
    if len(metrics_path) > 1:
        ax_pred.legend(fontsize=8)

    fig.suptitle("Population Dynamics (K&D Fig. 6 style)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved population dynamics plot to {output_path}")


def plot_reward_kde(metrics_path, output_path, step=None):
    """Reproduce K&D Figure 8/12 style: KDE of population reward weight distributions.

    Uses checkpoint data (.pkl) from the same results directory to get per-agent
    reward weights at the requested step. Falls back to the final logged step's
    aggregate stats if no checkpoint is available.

    Two panels: prey scatter (w_prey vs w_pred, colored by energy as a proxy
    for lifetime fitness), and predator equivalent.

    Args:
        metrics_path: Path to metrics.npz (single seed).
        output_path: Where to save the figure.
        step: Which logged step to visualize. If None, uses the final step.
    """
    _ensure_dir(output_path)

    m = _load_metrics(metrics_path)
    steps = m["steps"]

    if step is None:
        step_idx = len(steps) - 1
    else:
        # Find closest logged step
        step_idx = int(np.argmin(np.abs(steps - step)))

    actual_step = int(steps[step_idx])

    # Try to load checkpoint for per-agent data
    metrics_dir = os.path.dirname(metrics_path)
    checkpoint_path = os.path.join(metrics_dir, f"step_{actual_step:08d}.pkl")

    has_checkpoint = os.path.exists(checkpoint_path)

    if has_checkpoint:
        import pickle
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)

        species = ckpt["species"]
        weights = ckpt["reward_weights"]
        energies = ckpt["energies"]

        prey_mask = species == 0
        pred_mask = species == 1

        prey_w = weights[prey_mask]
        prey_e = energies[prey_mask]
        pred_w = weights[pred_mask]
        pred_e = energies[pred_mask]

        fig, (ax_prey, ax_pred) = plt.subplots(1, 2, figsize=(14, 6))

        # Prey: w_prey (idx 2) vs w_pred (idx 3), colored by energy
        if len(prey_w) > 0:
            sc = ax_prey.scatter(prey_w[:, 2], prey_w[:, 3], c=prey_e,
                                 cmap="viridis", alpha=0.7, s=20, edgecolors="none")
            plt.colorbar(sc, ax=ax_prey, label="Energy")
        ax_prey.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_prey.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_prey.set_xlabel(r"$w_{\mathrm{prey}}$")
        ax_prey.set_ylabel(r"$w_{\mathrm{pred}}$")
        ax_prey.set_title(f"Prey (n={len(prey_w)})")
        ax_prey.grid(True, alpha=0.3)

        # Predator: w_prey (idx 2) vs w_pred (idx 3), colored by energy
        if len(pred_w) > 0:
            sc = ax_pred.scatter(pred_w[:, 2], pred_w[:, 3], c=pred_e,
                                 cmap="magma", alpha=0.7, s=20, edgecolors="none")
            plt.colorbar(sc, ax=ax_pred, label="Energy")
        ax_pred.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_pred.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_pred.set_xlabel(r"$w_{\mathrm{prey}}$")
        ax_pred.set_ylabel(r"$w_{\mathrm{pred}}$")
        ax_pred.set_title(f"Predator (n={len(pred_w)})")
        ax_pred.grid(True, alpha=0.3)

    else:
        # Fallback: use aggregate stats to show KDE-like marginal distributions
        fig, (ax_prey, ax_pred) = plt.subplots(1, 2, figsize=(14, 6))

        # For prey: show marginal distributions of w_prey and w_pred using mean/std
        prey_w_prey_mean = m["prey_mean_w_prey"][step_idx]
        prey_w_prey_std = m["prey_std_w_prey"][step_idx]
        prey_w_pred_mean = m["prey_mean_w_pred"][step_idx]
        prey_w_pred_std = m["prey_std_w_pred"][step_idx]

        # Generate synthetic scatter from aggregate stats for visualization
        rng = np.random.default_rng(42)
        n_synthetic = 200
        prey_synth_wprey = rng.normal(prey_w_prey_mean, max(prey_w_prey_std, 0.01), n_synthetic)
        prey_synth_wpred = rng.normal(prey_w_pred_mean, max(prey_w_pred_std, 0.01), n_synthetic)

        ax_prey.scatter(prey_synth_wprey, prey_synth_wpred, alpha=0.4, s=15, c="steelblue")
        ax_prey.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_prey.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_prey.set_xlabel(r"$w_{\mathrm{prey}}$")
        ax_prey.set_ylabel(r"$w_{\mathrm{pred}}$")
        ax_prey.set_title(f"Prey (synthetic from mean/std, n={int(m['prey_population'][step_idx])})")
        ax_prey.grid(True, alpha=0.3)

        # For predator
        pred_w_prey_mean = m["pred_mean_w_prey"][step_idx]
        pred_w_prey_std = m["pred_std_w_prey"][step_idx]
        pred_w_pred_mean = m["pred_mean_w_pred"][step_idx]
        pred_w_pred_std = m["pred_std_w_pred"][step_idx]

        pred_synth_wprey = rng.normal(pred_w_prey_mean, max(pred_w_prey_std, 0.01), n_synthetic)
        pred_synth_wpred = rng.normal(pred_w_pred_mean, max(pred_w_pred_std, 0.01), n_synthetic)

        ax_pred.scatter(pred_synth_wprey, pred_synth_wpred, alpha=0.4, s=15, c="orangered")
        ax_pred.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_pred.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_pred.set_xlabel(r"$w_{\mathrm{prey}}$")
        ax_pred.set_ylabel(r"$w_{\mathrm{pred}}$")
        ax_pred.set_title(f"Predator (synthetic from mean/std, n={int(m['predator_population'][step_idx])})")
        ax_pred.grid(True, alpha=0.3)

    fig.suptitle(f"Reward Weight Distribution at Step {actual_step:,} (K&D Fig. 8/12 style)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reward KDE plot to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate dashboard plots from metrics.npz")
    parser.add_argument("--metrics", required=True, help="Path to metrics.npz")
    parser.add_argument("--output-dir", required=True, help="Directory to save plots")
    parser.add_argument("--step", type=int, default=None, help="Step for KDE plot (default: final)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_reward_trajectories(
        args.metrics,
        os.path.join(args.output_dir, "reward_trajectories.png"),
    )
    plot_population_dynamics(
        args.metrics,
        os.path.join(args.output_dir, "population_dynamics.png"),
    )
    plot_reward_kde(
        args.metrics,
        os.path.join(args.output_dir, "reward_kde.png"),
        step=args.step,
    )
    print(f"\nAll plots saved to {args.output_dir}")
