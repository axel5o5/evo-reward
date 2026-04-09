"""
Config validation and diffing utilities.

Validates config dicts against known schema and range constraints.
Also provides config diff for comparing two configs.
"""

import json
from pathlib import Path
from typing import Any


# Range constraints for known parameters: (min, max, type)
RANGE_HINTS = {
    "world_size": (100, 10000, int),
    "total_steps": (1000, 100_000_000, int),
    "obs_dim": (1, 1000, int),
    "prey_initial": (1, 1000, int),
    "predator_initial": (1, 500, int),
    "prey_cap": (1, 2000, int),
    "predator_cap": (1, 500, int),
    "prey_e_initial": (0.01, 10000.0, float),
    "predator_e_initial": (0.01, 10000.0, float),
    "prey_radius": (1.0, 100.0, float),
    "predator_radius": (1.0, 100.0, float),
    "max_motor_norm": (1.0, 1000.0, float),
    "n_proximity_sensors": (1, 128, int),
    "n_proximity_channels": (1, 16, int),
    "proximity_fov_deg": (10.0, 360.0, float),
    "proximity_max_range": (10.0, 2000.0, float),
    "n_tactile_sensors": (1, 72, int),
    "n_tactile_channels": (1, 16, int),
    "food_max": (1, 10000, int),
    "food_initial": (0, 10000, int),
    "food_growth_rate": (0.0, 100.0, float),
    "food_max_regen_per_step": (1, 1000, int),
    "energy_capacity": (1.0, 100000.0, float),
    "prey_c_b": (0.0, 0.1, float),
    "prey_c_a": (0.0, 0.01, float),
    "predator_d_b": (0.0, 0.1, float),
    "predator_d_a": (0.0, 0.01, float),
    "predator_eta": (0.0, 1.0, float),
    "kappa_h": (0.0, 1.0, float),
    "alpha_e": (0.0, 1.0, float),
    "beta_h": (0.0, 10.0, float),
    "kappa_b": (0.0, 1.0, float),
    "beta_b": (0.0, 10.0, float),
    "zeta_b_prey": (0.0, 1000.0, float),
    "zeta_b_pred": (0.0, 1000.0, float),
    "energy_share_ratio": (0.0, 1.0, float),
    "spawn_spread": (0.0, 1000.0, float),
    "reward_weights_init_std": (0.0, 10.0, float),
    "mutation_df": (1, 100, int),
    "mutation_scale": (0.0, 10.0, float),
    "weight_clip": (0.1, 1000.0, float),
    "policy_hidden_size": (4, 1024, int),
    "policy_n_hidden_layers": (1, 10, int),
    "gamma": (0.0, 1.0, float),
    "rollout_steps": (16, 100000, int),
    "minibatch_size": (8, 10000, int),
    "ppo_epochs": (1, 100, int),
    "clip_epsilon": (0.01, 1.0, float),
    "entropy_coef": (0.0, 1.0, float),
    "gae_lambda": (0.0, 1.0, float),
    "lr": (1e-8, 1.0, float),
    "adam_eps": (1e-12, 1e-3, float),
    "vf_coef": (0.0, 10.0, float),
    "action_clip_low": (-1000.0, 0.0, float),
    "action_clip_high": (0.0, 1000.0, float),
    "checkpoint_interval_steps": (100, 10_000_000, int),
    "log_interval_steps": (100, 10_000_000, int),
    "seed": (0, 999999, int),
}

# Allowed categorical values
CATEGORICALS = {
    "policy_mode": ["independent", "shared"],
    "lifecycle_mode": ["continuous", "generational"],
    "reward_type": ["linear", "mlp"],
    "social_obs": ["position_only", "position_heading_velocity"],
    "policy_type": ["mlp", "lstm"],
    "coevolution_mode": ["concurrent", "alternating"],
    "action_mapping": ["sigmoid", "clip"],
}

# Keys that must always be present
REQUIRED_KEYS = [
    "experiment_name", "policy_mode", "lifecycle_mode", "reward_type",
    "world_size", "total_steps", "obs_dim",
    "prey_initial", "predator_initial",
    "prey_c_b", "prey_c_a", "predator_d_b", "predator_d_a",
    "mutation_df", "mutation_scale",
    "gamma", "rollout_steps", "ppo_epochs", "clip_epsilon", "lr",
    "seed",
]


def validate_config(config: dict) -> list[dict]:
    """
    Validate a config dict against known constraints.

    Returns a list of issues, each a dict with keys:
        key: the config parameter name
        issue: description of the problem
        severity: "error" | "warning"
    """
    issues = []

    # Check required keys
    for key in REQUIRED_KEYS:
        if key not in config:
            issues.append({
                "key": key,
                "issue": f"Required key '{key}' is missing",
                "severity": "error",
            })

    for key, value in config.items():
        # Check categoricals
        if key in CATEGORICALS:
            if value not in CATEGORICALS[key]:
                issues.append({
                    "key": key,
                    "issue": f"Value '{value}' not in allowed values: {CATEGORICALS[key]}",
                    "severity": "error",
                })
            continue

        # Check range hints
        if key in RANGE_HINTS:
            lo, hi, expected_type = RANGE_HINTS[key]

            # Type check (allow int where float expected)
            if expected_type == float and not isinstance(value, (int, float)):
                issues.append({
                    "key": key,
                    "issue": f"Expected numeric, got {type(value).__name__}",
                    "severity": "error",
                })
                continue
            elif expected_type == int and not isinstance(value, int):
                if isinstance(value, float) and value == int(value):
                    pass  # e.g. 2.0 is fine for int
                elif not isinstance(value, (int, float)):
                    issues.append({
                        "key": key,
                        "issue": f"Expected integer, got {type(value).__name__}",
                        "severity": "error",
                    })
                    continue

            # Range check
            if isinstance(value, (int, float)):
                if value < lo or value > hi:
                    issues.append({
                        "key": key,
                        "issue": f"Value {value} outside expected range [{lo}, {hi}]",
                        "severity": "warning",
                    })

        # Check booleans
        if key in ("vf_clip",) and not isinstance(value, bool):
            issues.append({
                "key": key,
                "issue": f"Expected boolean, got {type(value).__name__}",
                "severity": "error",
            })

    # Cross-key consistency checks
    if "minibatch_size" in config and "rollout_steps" in config:
        if config["rollout_steps"] % config["minibatch_size"] != 0:
            issues.append({
                "key": "minibatch_size",
                "issue": f"rollout_steps ({config['rollout_steps']}) not divisible by minibatch_size ({config['minibatch_size']})",
                "severity": "warning",
            })

    if "action_clip_low" in config and "action_clip_high" in config:
        if config["action_clip_low"] >= config["action_clip_high"]:
            issues.append({
                "key": "action_clip_low",
                "issue": "action_clip_low must be less than action_clip_high",
                "severity": "error",
            })

    return issues


def diff_configs(config_a: dict, config_b: dict) -> dict:
    """
    Compare two config dicts and return differences.

    Returns: {key: {"old": value_a, "new": value_b}} for all keys where values differ.
    Keys present in only one config are included with None for the missing side.
    """
    all_keys = sorted(set(list(config_a.keys()) + list(config_b.keys())))
    diffs = {}

    for key in all_keys:
        val_a = config_a.get(key)
        val_b = config_b.get(key)
        if val_a != val_b:
            diffs[key] = {"old": val_a, "new": val_b}

    return diffs


def load_schema_ranges() -> dict:
    """Load range hints from config-schema.json if available."""
    schema_path = Path(__file__).resolve().parent.parent.parent / "site" / "src" / "data" / "config-schema.json"
    if schema_path.exists():
        with open(schema_path) as f:
            return json.load(f)
    return {}
