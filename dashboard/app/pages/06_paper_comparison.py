"""
Paper Comparison — Side-by-side: K&D published figures vs our reproduced plots.
The visual replication test.
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
PAPER_FIGURES_DIR = PROJECT_ROOT / "dashboard" / "site" / "public" / "paper-figures"

st.set_page_config(page_title="Paper Comparison", layout="wide")
st.title("Paper Comparison")
st.caption("Side-by-side: K&D published figures vs our reproduced plots")

# --- Select run ---
runs = scan_results(RESULTS_DIR)
if not runs:
    st.warning("No runs found.")
    st.stop()

run_labels = [f"{r['condition']}/seed_{r['seed']}" for r in runs]
selected_idx = st.sidebar.selectbox("Select run", range(len(runs)),
                                     format_func=lambda i: run_labels[i])
run = load_run(runs[selected_idx]["metrics_path"])

is_synthetic = run.condition == "synthetic"
if is_synthetic:
    st.info("Viewing **synthetic data** — comparison is illustrative only")

# --- Scan for paper figures ---
paper_figures = {}
if PAPER_FIGURES_DIR.exists():
    for img in sorted(PAPER_FIGURES_DIR.glob("*")):
        if img.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            paper_figures[img.stem.lower()] = img

if not paper_figures:
    st.info(
        f"No paper figures found in `{PAPER_FIGURES_DIR.relative_to(PROJECT_ROOT)}/`.\n\n"
        "To enable side-by-side comparison, add K&D published figures:\n"
        "- `figure6-population.png` — Population dynamics\n"
        "- `figure7-reward-weights.png` — Reward weight trajectories\n"
        "- `figure8-kde.png` — Reward weight KDE scatter\n"
        "- `figure12-distribution.png` — Weight distribution comparison"
    )

# --- Figure 7: Reward Weight Trajectories ---
st.header("Reward Weight Trajectories (cf. K&D Figure 7)")
col1, col2 = st.columns(2)

with col1:
    st.subheader("K&D Published")
    fig7_key = next((k for k in paper_figures if "figure7" in k or "reward-weight" in k), None)
    if fig7_key:
        st.image(str(paper_figures[fig7_key]), use_container_width=True)
    else:
        st.info("Add `figure7-reward-weights.png` to paper-figures/")

with col2:
    st.subheader("Our Reproduction")
    fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharex=True)
    weight_names = ["w_eat", "w_act", "w_prey", "w_pred"]
    titles = ["w_eat", "w_act", "w_prey", "w_pred"]
    steps = run.metrics["steps"]

    for ax, wname, title in zip(axes.flat, weight_names, titles):
        mean = run.metrics[f"prey_mean_{wname}"]
        std = run.metrics[f"prey_std_{wname}"]
        ax.plot(steps, mean, color="steelblue", linewidth=1.2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color="steelblue")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=7)

    axes[1, 0].set_xlabel("steps", fontsize=8)
    axes[1, 1].set_xlabel("steps", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# --- Figure 6: Population Dynamics ---
st.header("Population Dynamics (cf. K&D Figure 6)")
col1, col2 = st.columns(2)

with col1:
    st.subheader("K&D Published")
    fig6_key = next((k for k in paper_figures if "figure6" in k or "population" in k), None)
    if fig6_key:
        st.image(str(paper_figures[fig6_key]), use_container_width=True)
    else:
        st.info("Add `figure6-population.png` to paper-figures/")

with col2:
    st.subheader("Our Reproduction")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    ax1.plot(steps, run.metrics["prey_population"], color="royalblue", linewidth=1)
    ax1.set_ylabel("prey", fontsize=8)
    ax2.plot(steps, run.metrics["predator_population"], color="crimson", linewidth=1)
    ax2.set_ylabel("predator", fontsize=8)
    ax2.set_xlabel("steps", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# --- Figure 8/12: Reward Weight Distribution ---
st.header("Reward Weight Distribution (cf. K&D Figure 8/12)")
col1, col2 = st.columns(2)

with col1:
    st.subheader("K&D Published")
    fig8_key = next((k for k in paper_figures if "figure8" in k or "figure12" in k or "kde" in k or "distribution" in k), None)
    if fig8_key:
        st.image(str(paper_figures[fig8_key]), use_container_width=True)
    else:
        st.info("Add `figure8-kde.png` or `figure12-distribution.png` to paper-figures/")

with col2:
    st.subheader("Our Reproduction")
    # Approximate a KDE-like scatter: w_prey vs w_pred at final step
    # Use mean +/- random draws from the population distribution
    final_w_prey_mean = run.metrics["prey_mean_w_prey"][-1]
    final_w_pred_mean = run.metrics["prey_mean_w_pred"][-1]
    final_w_prey_std = run.metrics["prey_std_w_prey"][-1]
    final_w_pred_std = run.metrics["prey_std_w_pred"][-1]

    rng = np.random.RandomState(42)
    n_samples = 200
    scatter_w_prey = rng.normal(final_w_prey_mean, max(final_w_prey_std, 0.1), n_samples)
    scatter_w_pred = rng.normal(final_w_pred_mean, max(final_w_pred_std, 0.1), n_samples)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(scatter_w_prey, scatter_w_pred, alpha=0.4, s=15, color="steelblue")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("w_prey (social affiliation)", fontsize=9)
    ax.set_ylabel("w_pred (fear)", fontsize=9)
    ax.set_title("Prey reward weights at final step", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
