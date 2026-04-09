"""
Agent Inspector — Load checkpoint .pkl files and inspect individual agents.
Shows agent table, genome weights as bar chart, parent lineage.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loader import scan_results, load_run, load_checkpoint, list_checkpoints

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

st.set_page_config(page_title="Agent Inspector", layout="wide")
st.title("Agent Inspector")
st.caption("Drill into individual agents from checkpoint data")

# --- Select run ---
runs = scan_results(RESULTS_DIR)
if not runs:
    st.warning("No runs found.")
    st.stop()

run_labels = [f"{r['condition']}/seed_{r['seed']}" for r in runs]
selected_idx = st.sidebar.selectbox("Select run", range(len(runs)),
                                     format_func=lambda i: run_labels[i])
run_info = runs[selected_idx]
seed_dir = run_info["seed_dir"]

# --- List checkpoints ---
checkpoints = list_checkpoints(seed_dir)

if not checkpoints:
    st.info("No checkpoint files found for this run. "
            "Checkpoints are .pkl files saved during training (e.g., step_25000.pkl).\n\n"
            "For synthetic data, checkpoint inspection is not available — "
            "this page requires real experiment checkpoints.")

    # Show what we can from metrics: birth log as a lineage proxy
    run = load_run(run_info["metrics_path"])
    if "birth_log" in run.metrics and len(run.metrics["birth_log"]) > 0:
        st.header("Birth Log (from metrics)")
        birth_log = run.metrics["birth_log"]
        st.caption(f"{len(birth_log)} birth events recorded")

        # Show recent births
        n_show = min(50, len(birth_log))
        st.dataframe(
            [{"Step": int(row[0]), "Child ID": int(row[1]), "Parent ID": int(row[2])}
             for row in birth_log[-n_show:]],
            use_container_width=True,
        )

        # Simple lineage visualization: parent-child histogram
        st.subheader("Most Prolific Parents")
        parent_ids, counts = np.unique(birth_log[:, 2].astype(int), return_counts=True)
        top_n = min(20, len(parent_ids))
        top_idx = np.argsort(counts)[-top_n:]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(range(top_n), counts[top_idx], color="#4e79a7")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([str(parent_ids[i]) for i in top_idx])
        ax.set_xlabel("Number of offspring")
        ax.set_ylabel("Parent ID")
        ax.set_title("Top parents by offspring count")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.stop()

# --- Checkpoint selector ---
ckpt_labels = [p.stem for p in checkpoints]
selected_ckpt_idx = st.sidebar.selectbox("Select checkpoint", range(len(checkpoints)),
                                          format_func=lambda i: ckpt_labels[i])
ckpt_path = checkpoints[selected_ckpt_idx]

st.sidebar.info(f"Loading: {ckpt_path.name}")

try:
    checkpoint = load_checkpoint(ckpt_path)
except Exception as e:
    st.error(f"Failed to load checkpoint: {e}")
    st.stop()

# --- Parse checkpoint data ---
# Expected checkpoint structure: list of agent dicts or a dict with 'agents' key
agents_data = None
if isinstance(checkpoint, dict):
    if "agents" in checkpoint:
        agents_data = checkpoint["agents"]
    elif "state" in checkpoint and "agents" in checkpoint["state"]:
        agents_data = checkpoint["state"]["agents"]
    else:
        st.warning(f"Checkpoint keys: {list(checkpoint.keys())}. "
                   "Expected 'agents' key. Showing raw structure.")
        st.json({k: str(type(v)) for k, v in checkpoint.items()})
        st.stop()
elif isinstance(checkpoint, list):
    agents_data = checkpoint

if agents_data is None:
    st.warning("Could not parse agent data from checkpoint.")
    st.stop()

# --- Agent table ---
st.header(f"Agents at {ckpt_path.stem}")

weight_names = ["w_eat", "w_act", "w_prey", "w_pred"]
table_data = []

for agent in agents_data:
    row = {}
    if isinstance(agent, dict):
        row["ID"] = agent.get("id", "?")
        row["Species"] = agent.get("species", "?")
        row["Age"] = agent.get("age", "?")
        row["Energy"] = f"{agent.get('energy', 0):.1f}" if isinstance(agent.get("energy"), (int, float)) else "?"
        genome = agent.get("reward_weights", agent.get("genome", []))
        for j, wn in enumerate(weight_names):
            row[wn] = f"{genome[j]:.3f}" if j < len(genome) else "?"
        row["Parent"] = agent.get("parent_id", "?")
    table_data.append(row)

if table_data:
    st.dataframe(table_data, use_container_width=True)

    # --- Select agent for detail view ---
    agent_ids = [row.get("ID", i) for i, row in enumerate(table_data)]
    selected_agent_id = st.selectbox("Select agent for detail view", agent_ids)

    # Find the selected agent
    selected_agent = None
    for agent in agents_data:
        if isinstance(agent, dict) and agent.get("id") == selected_agent_id:
            selected_agent = agent
            break

    if selected_agent:
        st.subheader(f"Agent {selected_agent_id} Detail")
        col1, col2 = st.columns(2)

        with col1:
            # Genome weights as bar chart
            genome = selected_agent.get("reward_weights", selected_agent.get("genome", []))
            if genome and len(genome) >= 4:
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ["#59a14f" if v >= 0 else "#e15759" for v in genome[:4]]
                ax.barh(weight_names, genome[:4], color=colors)
                ax.axvline(x=0, color="gray", linewidth=0.8)
                ax.set_xlabel("Weight value")
                ax.set_title(f"Reward genome — Agent {selected_agent_id}")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with col2:
            st.markdown(f"**Species:** {selected_agent.get('species', '?')}")
            st.markdown(f"**Age:** {selected_agent.get('age', '?')}")
            st.markdown(f"**Energy:** {selected_agent.get('energy', '?')}")
            st.markdown(f"**Parent ID:** {selected_agent.get('parent_id', '?')}")
            if "position" in selected_agent:
                pos = selected_agent["position"]
                st.markdown(f"**Position:** ({pos[0]:.1f}, {pos[1]:.1f})")

    # --- Energy trajectory across checkpoints (if multiple available) ---
    if len(checkpoints) > 1 and selected_agent_id is not None:
        st.subheader(f"Agent {selected_agent_id} — Trajectory Across Checkpoints")
        st.caption("Tracking this agent's energy and weights across available checkpoints")

        trajectory_steps = []
        trajectory_energy = []
        trajectory_weights = {wn: [] for wn in weight_names}

        for ckpt_p in checkpoints:
            try:
                ckpt = load_checkpoint(ckpt_p)
                ckpt_agents = None
                if isinstance(ckpt, dict):
                    ckpt_agents = ckpt.get("agents", ckpt.get("state", {}).get("agents"))
                elif isinstance(ckpt, list):
                    ckpt_agents = ckpt

                if ckpt_agents:
                    for a in ckpt_agents:
                        if isinstance(a, dict) and a.get("id") == selected_agent_id:
                            step_str = ckpt_p.stem.replace("step_", "")
                            trajectory_steps.append(int(step_str))
                            trajectory_energy.append(a.get("energy", 0))
                            genome = a.get("reward_weights", a.get("genome", []))
                            for j, wn in enumerate(weight_names):
                                trajectory_weights[wn].append(genome[j] if j < len(genome) else 0)
                            break
            except Exception:
                continue

        if len(trajectory_steps) > 1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax1.plot(trajectory_steps, trajectory_energy, "o-", color="#4e79a7")
            ax1.set_ylabel("Energy")
            ax1.set_title(f"Agent {selected_agent_id} trajectory")

            for wn in weight_names:
                ax2.plot(trajectory_steps, trajectory_weights[wn], "o-", label=wn, markersize=4)
            ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
            ax2.set_ylabel("Weight value")
            ax2.set_xlabel("Step")
            ax2.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(f"Agent {selected_agent_id} only found in {len(trajectory_steps)} checkpoint(s).")
else:
    st.warning("No agent data to display.")
