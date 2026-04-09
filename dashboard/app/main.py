"""
evo-reward Dashboard — Streamlit entry point.

Launch: streamlit run dashboard/app/main.py
"""

import streamlit as st
from pathlib import Path

# Project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"

st.set_page_config(
    page_title="evo-reward Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("evo-reward Dashboard")
st.markdown(
    "Evolutionary reward structures in predator-prey RL — "
    "replicating and extending Kanagawa & Doya (2025)"
)

# Scan for completed runs
if RESULTS_DIR.exists():
    runs = sorted(RESULTS_DIR.glob("*/seed_*/metrics.npz"))
    st.metric("Completed runs", len(runs))

    if runs:
        st.subheader("Available runs")
        for run_path in runs:
            condition = run_path.parent.parent.name
            seed = run_path.parent.name
            size_mb = run_path.stat().st_size / (1024 * 1024)
            st.text(f"  {condition}/{seed}  ({size_mb:.1f} MB)")
    else:
        st.info("No completed runs yet. Use the Launcher page to start an experiment, "
                "or generate synthetic data for development.")
else:
    st.info("No results/ directory found. Experiments have not been run yet.")

st.divider()
st.caption("See sidebar for: Run Overview, Comparison, Experiment Matrix, "
           "Config Editor, Launcher, Live Monitor")
