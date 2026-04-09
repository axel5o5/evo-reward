"""
Generate synthetic metrics.npz for dashboard development.

Produces plausible-looking data that mimics a successful K&D replication:
- w_pred drifts negative (fear emerges)
- w_prey drifts positive (social affiliation)
- w_eat stays positive
- Population oscillates (Lotka-Volterra-like)
- No extinction

Usage:
    python -m dashboard.app.utils.synthetic_data [--output results/synthetic/seed_0/]
"""

import numpy as np
from pathlib import Path


def generate_synthetic_metrics(
    total_steps: int = 10_240_000,
    log_interval: int = 10_000,
    seed: int = 42,
) -> dict:
    """Generate a dict of numpy arrays matching MetricsLog field names."""
    rng = np.random.RandomState(seed)
    n_points = total_steps // log_interval

    steps = np.arange(0, total_steps, log_interval)
    t = np.linspace(0, 1, n_points)  # normalized time

    # --- Population dynamics: Lotka-Volterra-like oscillations ---
    prey_base = 150 + 50 * np.sin(2 * np.pi * t * 8)  # ~8 oscillation cycles
    pred_base = 15 + 8 * np.sin(2 * np.pi * t * 8 + np.pi / 4)  # phase-shifted
    prey_population = np.clip(
        prey_base + rng.normal(0, 10, n_points), 20, 450
    ).astype(int)
    predator_population = np.clip(
        pred_base + rng.normal(0, 3, n_points), 2, 50
    ).astype(int)

    # --- Energy ---
    prey_mean_energy = 30 + 10 * np.sin(2 * np.pi * t * 4) + rng.normal(0, 2, n_points)
    predator_mean_energy = 80 + 30 * np.sin(2 * np.pi * t * 3) + rng.normal(0, 5, n_points)

    # --- Prey reward weight trajectories ---
    # w_eat: drifts positive (agents learn eating is good)
    prey_mean_w_eat = 2.0 * (1 - np.exp(-3 * t)) + rng.normal(0, 0.1, n_points)
    prey_std_w_eat = 0.5 + 0.3 * t + rng.normal(0, 0.05, n_points)

    # w_act: stays near zero with some drift
    prey_mean_w_act = 0.3 * np.sin(2 * np.pi * t * 2) + rng.normal(0, 0.1, n_points)
    prey_std_w_act = 0.4 + rng.normal(0, 0.05, n_points)

    # w_prey: drifts positive (social affiliation emerges)
    prey_mean_w_prey = 1.5 * (1 - np.exp(-2 * t)) + rng.normal(0, 0.15, n_points)
    prey_std_w_prey = 0.6 + 0.4 * t + rng.normal(0, 0.05, n_points)

    # w_pred: drifts NEGATIVE (fear emerges) -- the headline result
    prey_mean_w_pred = -3.0 * (1 - np.exp(-2.5 * t)) + rng.normal(0, 0.2, n_points)
    prey_std_w_pred = 0.8 + 0.5 * t + rng.normal(0, 0.05, n_points)

    # --- Predator reward weight trajectories ---
    pred_mean_w_eat = 3.0 * (1 - np.exp(-4 * t)) + rng.normal(0, 0.1, n_points)
    pred_std_w_eat = 0.4 + 0.2 * t + rng.normal(0, 0.03, n_points)

    pred_mean_w_act = -0.5 * t + rng.normal(0, 0.1, n_points)
    pred_std_w_act = 0.3 + rng.normal(0, 0.03, n_points)

    pred_mean_w_prey = 2.0 * (1 - np.exp(-3 * t)) + rng.normal(0, 0.15, n_points)
    pred_std_w_prey = 0.5 + 0.3 * t + rng.normal(0, 0.05, n_points)

    pred_mean_w_pred = 0.2 * np.sin(2 * np.pi * t) + rng.normal(0, 0.1, n_points)
    pred_std_w_pred = 0.4 + rng.normal(0, 0.03, n_points)

    # --- Ecological metrics ---
    capture_rate = 0.002 + 0.001 * np.sin(2 * np.pi * t * 6) + rng.normal(0, 0.0005, n_points)
    capture_rate = np.clip(capture_rate, 0, None)

    food_consumption_rate = 0.05 + 0.02 * np.sin(2 * np.pi * t * 4) + rng.normal(0, 0.005, n_points)
    food_consumption_rate = np.clip(food_consumption_rate, 0, None)

    # --- Birth log (sparse events) ---
    n_births = int(n_points * 3)  # roughly 3 births per log interval
    birth_steps = np.sort(rng.randint(0, total_steps, n_births))
    birth_child_ids = np.arange(160, 160 + n_births)  # IDs starting after initial pop
    birth_parent_ids = rng.randint(0, 160, n_births)
    birth_log = np.column_stack([birth_steps, birth_child_ids, birth_parent_ids])

    return {
        "steps": steps,
        "prey_population": prey_population,
        "predator_population": predator_population,
        "prey_mean_energy": prey_mean_energy,
        "predator_mean_energy": predator_mean_energy,
        # Prey weights
        "prey_mean_w_eat": prey_mean_w_eat,
        "prey_mean_w_act": prey_mean_w_act,
        "prey_mean_w_prey": prey_mean_w_prey,
        "prey_mean_w_pred": prey_mean_w_pred,
        "prey_std_w_eat": np.abs(prey_std_w_eat),
        "prey_std_w_act": np.abs(prey_std_w_act),
        "prey_std_w_prey": np.abs(prey_std_w_prey),
        "prey_std_w_pred": np.abs(prey_std_w_pred),
        # Predator weights
        "pred_mean_w_eat": pred_mean_w_eat,
        "pred_mean_w_act": pred_mean_w_act,
        "pred_mean_w_prey": pred_mean_w_prey,
        "pred_mean_w_pred": pred_mean_w_pred,
        "pred_std_w_eat": np.abs(pred_std_w_eat),
        "pred_std_w_act": np.abs(pred_std_w_act),
        "pred_std_w_prey": np.abs(pred_std_w_prey),
        "pred_std_w_pred": np.abs(pred_std_w_pred),
        # Ecological
        "capture_rate": capture_rate,
        "food_consumption_rate": food_consumption_rate,
        # Birth log
        "birth_log": birth_log,
    }


def save_synthetic(output_dir: str = "results/synthetic/seed_0/", **kwargs):
    """Generate and save synthetic metrics to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = generate_synthetic_metrics(**kwargs)
    np.savez(out / "metrics.npz", **metrics)
    print(f"Saved synthetic metrics to {out / 'metrics.npz'}")
    print(f"  {len(metrics['steps'])} data points, "
          f"{metrics['steps'][-1]:,} total steps")


if __name__ == "__main__":
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else "results/synthetic/seed_0/"
    save_synthetic(output)
