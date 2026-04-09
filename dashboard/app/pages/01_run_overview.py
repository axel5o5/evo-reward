"""
Run Overview — Single-run deep dive.
Reproduces K&D Figures 6 (population dynamics), 7 (reward weight trajectories),
and 8/12 (reward weight KDE).
"""

import streamlit as st
import numpy as np
from pathlib import Path

# Deferred imports to avoid issues when streamlit scans pages
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loader import scan_results, load_run

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

st.set_page_config(page_title="Run Overview", layout="wide")
st.title("Run Overview")

# --- Sidebar: select run ---
runs = scan_results(RESULTS_DIR)
if not runs:
    st.warning("No runs found. Generate synthetic data first:\n\n"
               "```python -m dashboard.app.utils.synthetic_data```")
    st.stop()

run_labels = [f"{r['condition']}/seed_{r['seed']}" for r in runs]
selected_idx = st.sidebar.selectbox("Select run", range(len(runs)),
                                     format_func=lambda i: run_labels[i])
run_info = runs[selected_idx]
run = load_run(run_info["metrics_path"])

is_synthetic = run.condition == "synthetic"
if is_synthetic:
    st.info("Viewing **synthetic data** (generated for dashboard development)")

st.sidebar.metric("Total steps", f"{run.total_steps:,}")
st.sidebar.metric("Data points", len(run.steps))

# --- Reward Weight Trajectories (K&D Figure 7) ---
st.header("Reward Weight Trajectories")
st.caption("Reproduces K&D Figure 7 — mean +/- 1 std across population")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
weight_names = ["w_eat", "w_act", "w_prey", "w_pred"]
titles = ["w_eat (food reward)", "w_act (motor cost)",
          "w_prey (conspecific)", "w_pred (predator — fear)"]

for ax, wname, title in zip(axes.flat, weight_names, titles):
    steps = run.metrics["steps"]
    mean = run.metrics[f"prey_mean_{wname}"]
    std = run.metrics[f"prey_std_{wname}"]

    ax.plot(steps, mean, color="steelblue", linewidth=1.5, label="prey mean")
    ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color="steelblue")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("weight value")

axes[1, 0].set_xlabel("simulation steps")
axes[1, 1].set_xlabel("simulation steps")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- Population Dynamics (K&D Figure 6) ---
st.header("Population Dynamics")
st.caption("Reproduces K&D Figure 6 — Lotka-Volterra-like oscillations")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
steps = run.metrics["steps"]

ax1.plot(steps, run.metrics["prey_population"], color="royalblue", linewidth=1)
ax1.set_ylabel("prey count")
ax1.set_title("Prey population")

ax2.plot(steps, run.metrics["predator_population"], color="crimson", linewidth=1)
ax2.set_ylabel("predator count")
ax2.set_xlabel("simulation steps")
ax2.set_title("Predator population")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- Energy ---
st.header("Mean Energy")

fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(steps, run.metrics["prey_mean_energy"], color="royalblue", label="prey", linewidth=1)
ax.plot(steps, run.metrics["predator_mean_energy"], color="crimson", label="predator", linewidth=1)
ax.set_ylabel("mean energy")
ax.set_xlabel("simulation steps")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- Ecological Rates ---
st.header("Ecological Rates")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(steps, run.metrics["capture_rate"], color="crimson", linewidth=1)
    ax.set_title("Capture rate (prey caught/step)")
    ax.set_xlabel("steps")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(steps, run.metrics["food_consumption_rate"], color="forestgreen", linewidth=1)
    ax.set_title("Food consumption rate (food eaten/step)")
    ax.set_xlabel("steps")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
