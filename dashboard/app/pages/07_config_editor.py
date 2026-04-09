"""
Config Editor — Load, edit, validate, and save YAML configs.
Supports diffing against baseline and range validation.
"""

import streamlit as st
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config_validator import (
    validate_config, diff_configs, RANGE_HINTS, CATEGORICALS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

st.set_page_config(page_title="Config Editor", layout="wide")
st.title("Config Editor")
st.caption("Load, edit, validate, and save experiment configurations")

# --- Load config files ---
config_files = sorted(CONFIGS_DIR.glob("*.yaml")) if CONFIGS_DIR.exists() else []
if not config_files:
    st.warning(f"No YAML files found in {CONFIGS_DIR}")
    st.stop()

config_labels = [f.stem for f in config_files]
selected_idx = st.sidebar.selectbox("Load config", range(len(config_files)),
                                     format_func=lambda i: config_labels[i])
config_path = config_files[selected_idx]

with open(config_path) as f:
    config = yaml.safe_load(f)

# --- Diff against baseline toggle ---
show_diff = st.sidebar.toggle("Diff against baseline", value=False)
baseline_config = None
if show_diff:
    baseline_path = CONFIGS_DIR / "baseline_faithful.yaml"
    if baseline_path.exists() and baseline_path != config_path:
        with open(baseline_path) as f:
            baseline_config = yaml.safe_load(f)
        diffs = diff_configs(baseline_config, config)
    else:
        diffs = {}
        if baseline_path == config_path:
            st.sidebar.info("Selected config IS the baseline")
else:
    diffs = {}

# --- Group parameters by section ---
# Infer sections from key prefixes and known groupings
SECTION_ORDER = [
    ("Identity", ["experiment_name", "policy_mode", "lifecycle_mode", "reward_type",
                   "social_obs", "policy_type", "coevolution_mode"]),
    ("World", ["world_size", "total_steps"]),
    ("Observation", ["obs_dim"]),
    ("Population", ["prey_initial", "predator_initial", "prey_cap", "predator_cap",
                     "prey_e_initial", "predator_e_initial"]),
    ("Agent Bodies", ["prey_radius", "predator_radius", "max_motor_norm"]),
    ("Sensors", ["n_proximity_sensors", "n_proximity_channels", "proximity_fov_deg",
                  "proximity_max_range", "n_tactile_sensors", "n_tactile_channels",
                  "tactile_spacing_deg"]),
    ("Food Dynamics", ["food_max", "food_initial", "food_growth_rate", "food_max_regen_per_step"]),
    ("Energy", ["energy_capacity", "prey_e_food", "prey_c_b", "prey_c_a",
                 "predator_d_b", "predator_d_a", "predator_eta"]),
    ("Predator Mouth", ["predator_mouth_deg", "predator_mouth_range_min", "predator_mouth_range_max"]),
    ("Hazard", ["kappa_h", "alpha_e", "beta_h", "alpha_t_prey", "alpha_t_pred",
                 "beta_t_prey", "beta_t_pred"]),
    ("Birth", ["kappa_b", "beta_b", "zeta_b_prey", "zeta_b_pred"]),
    ("Offspring", ["energy_share_ratio", "spawn_spread"]),
    ("Reward Genome", ["reward_weights_init_std", "mutation_df", "mutation_scale", "weight_clip"]),
    ("Policy Network", ["policy_hidden_size", "policy_n_hidden_layers"]),
    ("PPO", ["gamma", "rollout_steps", "minibatch_size", "ppo_epochs", "clip_epsilon",
              "entropy_coef", "gae_lambda", "lr", "adam_eps", "vf_clip", "vf_coef"]),
    ("Action Space", ["action_clip_low", "action_clip_high", "action_mapping"]),
    ("Logging", ["checkpoint_interval_steps", "log_interval_steps"]),
    ("Reproducibility", ["seed"]),
]

# Collect all keys that appear in sections
sectioned_keys = set()
for _, keys in SECTION_ORDER:
    sectioned_keys.update(keys)

# Initialize edited config in session state
if "edited_config" not in st.session_state or st.session_state.get("_loaded_path") != str(config_path):
    st.session_state.edited_config = dict(config)
    st.session_state._loaded_path = str(config_path)

edited = st.session_state.edited_config

# --- Render form widgets by section ---
for section_name, section_keys in SECTION_ORDER:
    present_keys = [k for k in section_keys if k in config]
    if not present_keys:
        continue

    is_diff_section = show_diff and any(k in diffs for k in present_keys)
    header = f"{section_name}" + (" *" if is_diff_section else "")
    st.subheader(header)

    cols = st.columns(min(len(present_keys), 3))
    for i, key in enumerate(present_keys):
        col = cols[i % len(cols)]
        value = edited.get(key, config[key])
        diff_marker = " (changed)" if key in diffs else ""

        with col:
            if key in CATEGORICALS:
                options = CATEGORICALS[key]
                idx = options.index(value) if value in options else 0
                new_val = st.selectbox(f"{key}{diff_marker}", options, index=idx, key=f"edit_{key}")
            elif isinstance(value, bool):
                new_val = st.checkbox(f"{key}{diff_marker}", value=value, key=f"edit_{key}")
            elif isinstance(value, str):
                new_val = st.text_input(f"{key}{diff_marker}", value=value, key=f"edit_{key}")
            elif isinstance(value, int) and key in RANGE_HINTS:
                lo, hi, _ = RANGE_HINTS[key]
                new_val = st.number_input(f"{key}{diff_marker}", min_value=lo, max_value=hi,
                                          value=value, step=1, key=f"edit_{key}")
            elif isinstance(value, float) and key in RANGE_HINTS:
                lo, hi, _ = RANGE_HINTS[key]
                # Use slider for bounded ranges, number_input for very small values
                if lo >= 0 and hi <= 1000 and value >= lo and value <= hi:
                    new_val = st.number_input(f"{key}{diff_marker}", min_value=lo, max_value=hi,
                                              value=value, format="%.6g", key=f"edit_{key}")
                else:
                    new_val = st.number_input(f"{key}{diff_marker}", value=value,
                                              format="%.6g", key=f"edit_{key}")
            elif isinstance(value, (int, float)):
                new_val = st.number_input(f"{key}{diff_marker}", value=value, key=f"edit_{key}")
            else:
                new_val = st.text_input(f"{key}{diff_marker}", value=str(value), key=f"edit_{key}")

            edited[key] = new_val

# Unsectioned keys
unsectioned = [k for k in config if k not in sectioned_keys]
if unsectioned:
    st.subheader("Other")
    for key in unsectioned:
        value = edited.get(key, config[key])
        new_val = st.text_input(key, value=str(value), key=f"edit_{key}")
        edited[key] = new_val

# --- Validation ---
st.divider()
st.header("Validation")

issues = validate_config(edited)
if issues:
    for issue in issues:
        icon = "🔴" if issue["severity"] == "error" else "🟡"
        st.markdown(f"{icon} **{issue['key']}**: {issue['issue']}")
else:
    st.success("All checks passed")

# --- Diff display ---
if show_diff and baseline_config:
    st.divider()
    st.header("Diff against baseline_faithful")
    current_diffs = diff_configs(baseline_config, edited)
    if current_diffs:
        diff_data = [{"Key": k, "Baseline": str(v["old"]), "Current": str(v["new"])}
                     for k, v in current_diffs.items()]
        st.dataframe(diff_data, use_container_width=True)
    else:
        st.info("No differences from baseline")

# --- Save ---
st.divider()
col1, col2 = st.columns([2, 1])
with col1:
    save_name = st.text_input("Save as", value=config_path.stem + "_modified")
with col2:
    if st.button("Save config", type="primary"):
        save_path = CONFIGS_DIR / f"{save_name}.yaml"
        if save_path.exists():
            st.warning(f"File {save_path.name} already exists. Choose a different name.")
        else:
            with open(save_path, "w") as f:
                yaml.dump(dict(edited), f, default_flow_style=False, sort_keys=False)
            st.success(f"Saved to {save_path.relative_to(PROJECT_ROOT)}")
