"""
Live Monitor — Watch running experiments in real time.
Auto-refreshes every 30 seconds, shows progress and live metrics.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loader import load_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

st.set_page_config(page_title="Live Monitor", layout="wide")
st.title("Live Monitor")

# Auto-refresh toggle
auto_refresh = st.sidebar.toggle("Auto-refresh (30s)", value=True)
if auto_refresh:
    # Streamlit's built-in auto-rerun
    refresh_interval = st.sidebar.slider("Refresh interval (s)", 10, 120, 30)


def find_running_experiments():
    """Find experiments by .pid file or recent metrics.npz modification."""
    experiments = []
    if not RESULTS_DIR.exists():
        return experiments

    for seed_dir in sorted(RESULTS_DIR.glob("*/seed_*")):
        condition = seed_dir.parent.name
        seed = seed_dir.name

        pid_file = seed_dir / ".pid"
        metrics_file = seed_dir / "metrics.npz"
        is_running = False
        pid = None

        # Check PID file
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)  # Check if process exists
                is_running = True
            except (ProcessLookupError, ValueError):
                # Stale PID file — process is dead
                pid_file.unlink(missing_ok=True)
                pid = None

        # Also check recent metrics modification (within last 5 minutes)
        if not is_running and metrics_file.exists():
            mtime = metrics_file.stat().st_mtime
            age_seconds = time.time() - mtime
            if age_seconds < 300:  # Modified in last 5 minutes
                is_running = True

        if is_running or metrics_file.exists():
            experiments.append({
                "condition": condition,
                "seed": seed,
                "seed_dir": seed_dir,
                "pid": pid,
                "is_running": is_running,
                "metrics_path": metrics_file if metrics_file.exists() else None,
            })

    return experiments


experiments = find_running_experiments()

if not experiments:
    st.info("No experiments found (running or completed). "
            "Use the **Launcher** page to start an experiment.")
    st.stop()

running = [e for e in experiments if e["is_running"]]
completed = [e for e in experiments if not e["is_running"]]

if running:
    st.subheader(f"Running ({len(running)})")
else:
    st.info("No experiments currently running")

# --- Display each running experiment ---
for exp in running:
    with st.container():
        st.markdown(f"### {exp['condition']}/{exp['seed']}")
        if exp["pid"]:
            st.caption(f"PID: {exp['pid']}")

        if exp["metrics_path"] and exp["metrics_path"].exists():
            try:
                metrics = load_metrics(exp["metrics_path"])
                steps = metrics.get("steps", np.array([]))

                if len(steps) > 0:
                    current_step = int(steps[-1])
                    n_points = len(steps)

                    # Try to determine total steps from config
                    config_path = exp["seed_dir"] / "config.yaml"
                    total_steps = None
                    if config_path.exists():
                        import yaml
                        with open(config_path) as f:
                            cfg = yaml.safe_load(f)
                        total_steps = cfg.get("total_steps")

                    # Progress
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current step", f"{current_step:,}")
                    col2.metric("Data points", n_points)

                    if total_steps:
                        progress = current_step / total_steps
                        col3.metric("Progress", f"{progress:.1%}")
                        st.progress(min(progress, 1.0))

                        # Estimate time remaining from file modification time
                        mtime = exp["metrics_path"].stat().st_mtime
                        age = time.time() - mtime
                        if age < 600 and n_points > 1:
                            step_interval = int(steps[-1] - steps[-2])
                            if step_interval > 0:
                                steps_remaining = total_steps - current_step
                                # Estimate seconds per step from last interval
                                secs_per_point = age  # rough: time since last write
                                time_remaining_s = (steps_remaining / step_interval) * secs_per_point
                                hours_remaining = time_remaining_s / 3600
                                col4.metric("ETA", f"~{hours_remaining:.1f}h")

                    # Live population counts
                    if "prey_population" in metrics and "predator_population" in metrics:
                        pcol1, pcol2 = st.columns(2)
                        pcol1.metric("Prey population",
                                     int(metrics["prey_population"][-1]),
                                     delta=int(metrics["prey_population"][-1] - metrics["prey_population"][-2])
                                     if len(metrics["prey_population"]) > 1 else None)
                        pcol2.metric("Predator population",
                                     int(metrics["predator_population"][-1]),
                                     delta=int(metrics["predator_population"][-1] - metrics["predator_population"][-2])
                                     if len(metrics["predator_population"]) > 1 else None)

                    # Current mean reward weights
                    weight_names = ["w_eat", "w_act", "w_prey", "w_pred"]
                    wcols = st.columns(4)
                    for wcol, wname in zip(wcols, weight_names):
                        key = f"prey_mean_{wname}"
                        if key in metrics:
                            val = float(metrics[key][-1])
                            wcol.metric(f"prey {wname}", f"{val:.3f}")

                    # Mini plots: last 100 data points
                    show_recent = min(100, n_points)
                    recent_steps = steps[-show_recent:]

                    fig, axes = plt.subplots(1, 3, figsize=(14, 3))

                    # Population
                    if "prey_population" in metrics:
                        axes[0].plot(recent_steps, metrics["prey_population"][-show_recent:],
                                     color="royalblue", linewidth=1, label="prey")
                    if "predator_population" in metrics:
                        axes[0].plot(recent_steps, metrics["predator_population"][-show_recent:],
                                     color="crimson", linewidth=1, label="pred")
                    axes[0].set_title("Population (recent)", fontsize=9)
                    axes[0].legend(fontsize=7)

                    # Reward weights
                    for wname in weight_names:
                        key = f"prey_mean_{wname}"
                        if key in metrics:
                            axes[1].plot(recent_steps, metrics[key][-show_recent:],
                                         linewidth=1, label=wname)
                    axes[1].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
                    axes[1].set_title("Prey reward weights (recent)", fontsize=9)
                    axes[1].legend(fontsize=7)

                    # Energy
                    if "prey_mean_energy" in metrics:
                        axes[2].plot(recent_steps, metrics["prey_mean_energy"][-show_recent:],
                                     color="royalblue", linewidth=1, label="prey")
                    if "predator_mean_energy" in metrics:
                        axes[2].plot(recent_steps, metrics["predator_mean_energy"][-show_recent:],
                                     color="crimson", linewidth=1, label="pred")
                    axes[2].set_title("Mean energy (recent)", fontsize=9)
                    axes[2].legend(fontsize=7)

                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Metrics file exists but no data points logged yet")
            except Exception as e:
                st.error(f"Error loading metrics: {e}")
        else:
            st.info("Waiting for first metrics checkpoint...")

        st.divider()

# --- Completed experiments ---
if completed:
    st.subheader(f"Recent Completed ({len(completed)})")
    for exp in completed[-5:]:  # Show last 5
        metrics_age = ""
        if exp["metrics_path"] and exp["metrics_path"].exists():
            mtime = datetime.fromtimestamp(exp["metrics_path"].stat().st_mtime)
            metrics_age = f" — last updated {mtime.strftime('%Y-%m-%d %H:%M')}"
        st.text(f"  {exp['condition']}/{exp['seed']}{metrics_age}")

# --- Auto-refresh ---
if auto_refresh and running:
    time.sleep(refresh_interval)
    st.rerun()
