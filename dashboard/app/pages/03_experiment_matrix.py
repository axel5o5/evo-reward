"""
Experiment Matrix — Grid view of what's been run across conditions and seeds.
"""

import streamlit as st
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loader import scan_results

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"

st.set_page_config(page_title="Experiment Matrix", layout="wide")
st.title("Experiment Matrix")
st.caption("Tracks which conditions have been run and at how many seeds")

# Discover all known conditions from configs
known_conditions = []
if CONFIGS_DIR.exists():
    for yaml_file in sorted(CONFIGS_DIR.glob("*.yaml")):
        known_conditions.append(yaml_file.stem)

# Discover completed runs
runs = scan_results(RESULTS_DIR)
run_map = {}  # {condition: {seed: info}}
for r in runs:
    run_map.setdefault(r["condition"], {})[r["seed"]] = r

# Also include conditions that have results but no config
all_conditions = sorted(set(known_conditions) | set(run_map.keys()))

# Detect running experiments (by .pid files)
running = set()
if RESULTS_DIR.exists():
    for pid_file in RESULTS_DIR.glob("*/seed_*/.pid"):
        cond = pid_file.parent.parent.name
        seed = int(pid_file.parent.name.replace("seed_", ""))
        running.add((cond, seed))

# Build the matrix
MAX_SEEDS = 5
st.markdown("| Condition | " + " | ".join(f"Seed {i}" for i in range(MAX_SEEDS)) + " |")
st.markdown("|" + "---|" * (MAX_SEEDS + 1))

for condition in all_conditions:
    cells = []
    for seed in range(MAX_SEEDS):
        if condition in run_map and seed in run_map[condition]:
            cells.append("done")
        elif (condition, seed) in running:
            cells.append("running")
        else:
            cells.append("--")
    row = f"| `{condition}` | " + " | ".join(cells) + " |"
    st.markdown(row)

st.divider()
st.metric("Total completed runs", len(runs))
st.metric("Known conditions", len(all_conditions))
