"""
Comparison — Overlay plots from multiple seeds or conditions.
Statistical summary for Phase 1a success criteria.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loader import scan_results, load_run

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

st.set_page_config(page_title="Comparison", layout="wide")
st.title("Cross-Run Comparison")

# --- Discover runs ---
runs = scan_results(RESULTS_DIR)
if not runs:
    st.warning("No runs found. Generate synthetic data first:\n\n"
               "```python -m dashboard.app.utils.synthetic_data```")
    st.stop()

run_labels = [f"{r['condition']}/seed_{r['seed']}" for r in runs]
selected_indices = st.sidebar.multiselect(
    "Select runs to compare",
    range(len(runs)),
    default=list(range(min(len(runs), 3))),
    format_func=lambda i: run_labels[i],
)

if not selected_indices:
    st.info("Select at least one run from the sidebar.")
    st.stop()

# Load selected runs
loaded_runs = []
for idx in selected_indices:
    run = load_run(runs[idx]["metrics_path"])
    loaded_runs.append(run)

COLORS = ["#4e79a7", "#e15759", "#76b7b2", "#59a14f", "#edc949",
          "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab"]

# --- Reward Weight Trajectories (overlaid) ---
st.header("Reward Weight Trajectories")
st.caption("Mean prey reward weights — one line per selected run")

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
weight_names = ["w_eat", "w_act", "w_prey", "w_pred"]
titles = ["w_eat (food reward)", "w_act (motor cost)",
          "w_prey (conspecific)", "w_pred (predator — fear)"]

for ax, wname, title in zip(axes.flat, weight_names, titles):
    for i, run in enumerate(loaded_runs):
        color = COLORS[i % len(COLORS)]
        steps = run.metrics["steps"]
        mean = run.metrics[f"prey_mean_{wname}"]
        label = f"{run.condition}/s{run.seed}"
        ax.plot(steps, mean, color=color, linewidth=1.2, label=label)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("weight value")

axes[0, 0].legend(fontsize=7)
axes[1, 0].set_xlabel("simulation steps")
axes[1, 1].set_xlabel("simulation steps")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- Population Dynamics (overlaid) ---
st.header("Population Dynamics")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
for i, run in enumerate(loaded_runs):
    color = COLORS[i % len(COLORS)]
    steps = run.metrics["steps"]
    label = f"{run.condition}/s{run.seed}"
    ax1.plot(steps, run.metrics["prey_population"], color=color, linewidth=1, label=label)
    ax2.plot(steps, run.metrics["predator_population"], color=color, linewidth=1, label=label)

ax1.set_ylabel("prey count")
ax1.set_title("Prey population")
ax1.legend(fontsize=7)
ax2.set_ylabel("predator count")
ax2.set_xlabel("simulation steps")
ax2.set_title("Predator population")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- Phase 1a Success Criteria ---
st.header("Phase 1a Success Criteria")
st.caption("Evaluated at the final logged step for each run")

criteria_data = []
for run in loaded_runs:
    final_idx = -1
    final_step = int(run.metrics["steps"][final_idx])
    w_pred = float(run.metrics["prey_mean_w_pred"][final_idx])
    w_prey = float(run.metrics["prey_mean_w_prey"][final_idx])
    w_eat = float(run.metrics["prey_mean_w_eat"][final_idx])
    prey_pop = int(run.metrics["prey_population"][final_idx])
    pred_pop = int(run.metrics["predator_population"][final_idx])

    # Check population oscillation: std/mean > threshold
    prey_cv = float(np.std(run.metrics["prey_population"]) / max(np.mean(run.metrics["prey_population"]), 1))
    no_extinction = prey_pop > 0 and pred_pop > 0

    criteria_data.append({
        "Run": f"{run.condition}/s{run.seed}",
        "Final step": f"{final_step:,}",
        "w_pred < 0": "PASS" if w_pred < 0 else "FAIL",
        "w_pred value": f"{w_pred:.3f}",
        "w_prey > 0": "PASS" if w_prey > 0 else "FAIL",
        "w_prey value": f"{w_prey:.3f}",
        "w_eat > 0": "PASS" if w_eat > 0 else "FAIL",
        "w_eat value": f"{w_eat:.3f}",
        "Pop oscillates": "PASS" if prey_cv > 0.05 else "FAIL",
        "No extinction": "PASS" if no_extinction else "FAIL",
    })

st.dataframe(criteria_data, use_container_width=True)

# Summary
n_runs = len(loaded_runs)
n_pred_neg = sum(1 for c in criteria_data if c["w_pred < 0"] == "PASS")
n_prey_pos = sum(1 for c in criteria_data if c["w_prey > 0"] == "PASS")
n_eat_pos = sum(1 for c in criteria_data if c["w_eat > 0"] == "PASS")
n_no_ext = sum(1 for c in criteria_data if c["No extinction"] == "PASS")

st.subheader("Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("w_pred < 0", f"{n_pred_neg}/{n_runs}", delta="PASS" if n_pred_neg >= 3 else "FAIL")
col2.metric("w_prey > 0", f"{n_prey_pos}/{n_runs}", delta="PASS" if n_prey_pos >= 3 else "FAIL")
col3.metric("w_eat > 0", f"{n_eat_pos}/{n_runs}", delta="PASS" if n_eat_pos == n_runs else "FAIL")
col4.metric("No extinction", f"{n_no_ext}/{n_runs}", delta="PASS" if n_no_ext == n_runs else "FAIL")
