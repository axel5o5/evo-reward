"""
Data loading utilities for the evo-reward dashboard.
Loads metrics.npz and checkpoint .pkl files from results/.
"""

import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RunData:
    """Loaded data from a single experiment run."""
    condition: str
    seed: int
    metrics: dict  # numpy arrays keyed by MetricsLog field names
    config: Optional[dict] = None
    path: Path = field(default_factory=Path)

    @property
    def steps(self) -> np.ndarray:
        return self.metrics.get("steps", np.array([]))

    @property
    def total_steps(self) -> int:
        return int(self.steps[-1]) if len(self.steps) > 0 else 0


def scan_results(results_dir: Path) -> list[dict]:
    """Scan results/ for all available runs. Returns metadata, not full data."""
    runs = []
    if not results_dir.exists():
        return runs

    for metrics_path in sorted(results_dir.glob("*/seed_*/metrics.npz")):
        seed_dir = metrics_path.parent
        condition_dir = seed_dir.parent
        runs.append({
            "condition": condition_dir.name,
            "seed": int(seed_dir.name.replace("seed_", "")),
            "metrics_path": metrics_path,
            "seed_dir": seed_dir,
            "size_bytes": metrics_path.stat().st_size,
        })
    return runs


def load_metrics(metrics_path: Path) -> dict:
    """Load metrics.npz and return as a dict of numpy arrays."""
    data = np.load(metrics_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_run(metrics_path: Path) -> RunData:
    """Load a full run from its metrics.npz path."""
    seed_dir = metrics_path.parent
    condition_dir = seed_dir.parent

    metrics = load_metrics(metrics_path)

    # Try to load config from a checkpoint or config.yaml
    config = None
    config_path = seed_dir / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

    return RunData(
        condition=condition_dir.name,
        seed=int(seed_dir.name.replace("seed_", "")),
        metrics=metrics,
        config=config,
        path=seed_dir,
    )


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load a single checkpoint .pkl file."""
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def list_checkpoints(seed_dir: Path) -> list[Path]:
    """List all checkpoint files in a seed directory, sorted by step."""
    return sorted(seed_dir.glob("step_*.pkl"))
