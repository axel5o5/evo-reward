"""
metrics.py
----------
Logging, checkpointing, and save/load for the evo-reward simulation.

MetricsLog stores all fields from docs/interfaces.md as Python lists.
save_metrics() -> .npz, save_checkpoint() -> .pkl.
Path convention: results/{experiment_name}/seed_{seed}/
"""

import os
import pickle
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MetricsLog:
    """Accumulated metrics during training. All fields are Python lists."""
    # Logged every log_interval_steps
    steps: list = field(default_factory=list)
    prey_population: list = field(default_factory=list)
    predator_population: list = field(default_factory=list)
    prey_mean_energy: list = field(default_factory=list)
    predator_mean_energy: list = field(default_factory=list)
    # Reward weight trajectories — prey
    prey_mean_w_eat: list = field(default_factory=list)
    prey_mean_w_act: list = field(default_factory=list)
    prey_mean_w_prey: list = field(default_factory=list)
    prey_mean_w_pred: list = field(default_factory=list)
    prey_std_w_eat: list = field(default_factory=list)
    prey_std_w_act: list = field(default_factory=list)
    prey_std_w_prey: list = field(default_factory=list)
    prey_std_w_pred: list = field(default_factory=list)
    # Reward weight trajectories — predator
    pred_mean_w_eat: list = field(default_factory=list)
    pred_mean_w_act: list = field(default_factory=list)
    pred_mean_w_prey: list = field(default_factory=list)
    pred_mean_w_pred: list = field(default_factory=list)
    pred_std_w_eat: list = field(default_factory=list)
    pred_std_w_act: list = field(default_factory=list)
    pred_std_w_prey: list = field(default_factory=list)
    pred_std_w_pred: list = field(default_factory=list)
    # Ecological metrics
    capture_rate: list = field(default_factory=list)
    food_consumption_rate: list = field(default_factory=list)
    # Birth log: (step, child_id, parent_id)
    birth_log: list = field(default_factory=list)


def _get_species_agents(world, species: int):
    """Return list of agents matching the given species."""
    return [a for a in world.agents if a.species == species]


def log_step(log: MetricsLog, world, config: dict) -> MetricsLog:
    """Append current step's aggregate metrics. Returns the log (mutated in place)."""
    prey = _get_species_agents(world, 0)
    preds = _get_species_agents(world, 1)

    log.steps.append(world.step)
    log.prey_population.append(len(prey))
    log.predator_population.append(len(preds))

    # Mean energy
    log.prey_mean_energy.append(
        float(np.mean([a.energy for a in prey])) if prey else 0.0
    )
    log.predator_mean_energy.append(
        float(np.mean([a.energy for a in preds])) if preds else 0.0
    )

    # Reward weight stats — prey
    if prey:
        w = np.array([np.array(a.reward_weights) for a in prey])
        means = w.mean(axis=0)
        stds = w.std(axis=0)
    else:
        means = np.zeros(4)
        stds = np.zeros(4)
    log.prey_mean_w_eat.append(float(means[0]))
    log.prey_mean_w_act.append(float(means[1]))
    log.prey_mean_w_prey.append(float(means[2]))
    log.prey_mean_w_pred.append(float(means[3]))
    log.prey_std_w_eat.append(float(stds[0]))
    log.prey_std_w_act.append(float(stds[1]))
    log.prey_std_w_prey.append(float(stds[2]))
    log.prey_std_w_pred.append(float(stds[3]))

    # Reward weight stats — predator
    if preds:
        w = np.array([np.array(a.reward_weights) for a in preds])
        means = w.mean(axis=0)
        stds = w.std(axis=0)
    else:
        means = np.zeros(4)
        stds = np.zeros(4)
    log.pred_mean_w_eat.append(float(means[0]))
    log.pred_mean_w_act.append(float(means[1]))
    log.pred_mean_w_prey.append(float(means[2]))
    log.pred_mean_w_pred.append(float(means[3]))
    log.pred_std_w_eat.append(float(stds[0]))
    log.pred_std_w_act.append(float(stds[1]))
    log.pred_std_w_prey.append(float(stds[2]))
    log.pred_std_w_pred.append(float(stds[3]))

    # Ecological rates default to 0 (caller should update with rolling averages)
    log.capture_rate.append(0.0)
    log.food_consumption_rate.append(0.0)

    return log


def record_birth(log: MetricsLog, step: int, child_id: int, parent_id: int) -> MetricsLog:
    """Append birth event to birth_log."""
    log.birth_log.append((step, child_id, parent_id))
    return log


def _make_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def _build_path(config: dict, seed: int, out_dir: str) -> str:
    """Build the standard output directory path."""
    experiment_name = config.get("experiment_name", "experiment")
    return os.path.join(out_dir, experiment_name, f"seed_{seed}")


def save_checkpoint(world, log: MetricsLog, config: dict, seed: int, out_dir: str) -> None:
    """Save checkpoint as .pkl.

    Path: {out_dir}/{experiment_name}/seed_{seed}/step_{step:08d}.pkl
    """
    dirpath = _build_path(config, seed, out_dir)
    _make_dir(dirpath)

    # Build checkpoint dict
    agents = world.agents
    checkpoint = {
        "step": world.step,
        "config": config,
        "seed": seed,
        "agent_ids": np.array([a.agent_id for a in agents], dtype=np.int64),
        "species": np.array([a.species for a in agents], dtype=np.int64),
        "ages": np.array([a.age for a in agents], dtype=np.int64),
        "energies": np.array([a.energy for a in agents], dtype=np.float32),
        "reward_weights": np.array(
            [np.array(a.reward_weights) for a in agents], dtype=np.float32
        ),
        "parent_ids": np.array([a.parent_id for a in agents], dtype=np.int64),
    }

    filepath = os.path.join(dirpath, f"step_{world.step:08d}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(checkpoint, f)


def save_metrics(log: MetricsLog, config: dict, seed: int, out_dir: str) -> None:
    """Save metrics as .npz.

    Path: {out_dir}/{experiment_name}/seed_{seed}/metrics.npz
    All list fields saved as numpy arrays with field names as keys.
    """
    dirpath = _build_path(config, seed, out_dir)
    _make_dir(dirpath)

    data = {}
    for fname in log.__dataclass_fields__:
        val = getattr(log, fname)
        if fname == "birth_log":
            # List of tuples -> (M, 3) array
            if val:
                data[fname] = np.array(val, dtype=np.int64)
            else:
                data[fname] = np.zeros((0, 3), dtype=np.int64)
        else:
            data[fname] = np.array(val)

    filepath = os.path.join(dirpath, "metrics.npz")
    np.savez(filepath, **data)


def load_checkpoint(path: str) -> dict:
    """Load checkpoint from .pkl. Returns checkpoint dict."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_metrics(path: str) -> MetricsLog:
    """Load metrics from .npz. Returns MetricsLog with lists restored."""
    data = np.load(path, allow_pickle=True)
    log = MetricsLog()
    for fname in log.__dataclass_fields__:
        if fname in data:
            arr = data[fname]
            if fname == "birth_log":
                setattr(log, fname, [tuple(row) for row in arr])
            else:
                setattr(log, fname, arr.tolist())
    return log
