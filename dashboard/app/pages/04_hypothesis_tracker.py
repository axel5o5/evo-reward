"""
Hypothesis Tracker — H1-H7 from experimental-plan.md as interactive cards.
Each card shows prediction, measurement, which conditions test it, status.
"""

import streamlit as st

st.set_page_config(page_title="Hypothesis Tracker", layout="wide")
st.title("Hypothesis Tracker")
st.caption("H1-H7 from the experimental plan — track status as experiments complete")

HYPOTHESES = [
    {
        "id": "H1",
        "title": "Reward convergence speed",
        "prediction": "Shared policy produces faster reward weight convergence than independent policy, "
                      "because the amortized policy provides a more stable mapping from reward genome to fitness. "
                      "Generational batching produces faster convergence than continuous birth-death.",
        "measurement": "Number of generations/steps until reward weights stabilize within threshold of final values.",
        "conditions": ["Phase 1a (independent/continuous) vs Phase 1b (shared/continuous)"],
        "default_status": "UNTESTED",
    },
    {
        "id": "H2",
        "title": "Intra-species behavioral diversity",
        "prediction": "Independent policy produces greater behavioral diversity within a species than shared policy, "
                      "even among agents with similar reward weights, because individual learning trajectories diverge.",
        "measurement": "Shannon entropy over discretized behavioral space (speed x heading x nearest-neighbor bins). "
                       "Number of distinct behavioral modes in the population.",
        "conditions": ["Phase 1a vs Phase 1b comparison"],
        "default_status": "UNTESTED",
    },
    {
        "id": "H3",
        "title": "Reward weight branching",
        "prediction": "Under shared policy, we predict less branching of reward weights (fewer distinct lineages), "
                      "because the shared policy smooths the reward-fitness landscape. Extreme reward genomes "
                      "are less viable when policy is optimized for the population mean.",
        "measurement": "K&D Figure 8 branching analysis. Reward weight distribution width (std) over time. "
                       "Number of clusters in reward weight space.",
        "conditions": ["Phase 1a (independent) vs Phase 1b (shared)"],
        "default_status": "UNTESTED",
    },
    {
        "id": "H4",
        "title": "Population dynamics and ecological structure",
        "prediction": "Continuous birth-death produces Lotka-Volterra-like oscillations. "
                      "Generational batching produces flat (fixed) population sizes. "
                      "Oscillatory dynamics create temporally varying selection pressure.",
        "measurement": "Population size time series, autocorrelation, spectral analysis.",
        "conditions": ["Continuous vs generational lifecycle comparison"],
        "default_status": "UNTESTED",
    },
    {
        "id": "H5",
        "title": "Qualitative robustness of core result",
        "prediction": "Fear (negative w_pred) and social affiliation (positive w_prey) emerge under ALL FOUR "
                      "architecture combinations. These are robust consequences of predator-prey competitive dynamics.",
        "measurement": "Phase 1a success criteria applied to all four architecture combos. "
                       "If this holds, it's a meaningful robustness result.",
        "conditions": ["All Phase 1 runs (ind/cont, shared/cont, ind/gen, shared/gen)"],
        "default_status": "UNTESTED",
    },
    {
        "id": "H6",
        "title": "Baldwin effect strength",
        "prediction": "Under independent + continuous (K&D faithful), reward functions that enable fast learning "
                      "should be selectively favored. Under shared policy, this pressure is weakened because "
                      "all agents benefit from the shared policy.",
        "measurement": "Correlation between reward genome complexity and agent lifetime. Should be positive "
                       "under independent/continuous, weaker under shared.",
        "conditions": ["Phase 1a vs Phase 1b, measuring learning curve slopes"],
        "default_status": "UNTESTED",
    },
    {
        "id": "H7",
        "title": "Implicit cultural transmission via shared policy",
        "prediction": "Shared policy acts as population-level knowledge transfer. Newborns immediately inherit "
                      "behavioral competence. This should reduce infant mortality relative to independent policy.",
        "measurement": "Mean age at first reproduction, survival rate in first N timesteps of life.",
        "conditions": ["Phase 1a vs Phase 1b, measuring newborn survival rates"],
        "default_status": "UNTESTED",
    },
]

STATUS_OPTIONS = ["UNTESTED", "CONFIRMED", "PARTIALLY CONFIRMED", "FALSIFIED", "INCONCLUSIVE"]
STATUS_COLORS = {
    "UNTESTED": "gray",
    "CONFIRMED": "green",
    "PARTIALLY CONFIRMED": "orange",
    "FALSIFIED": "red",
    "INCONCLUSIVE": "blue",
}

# Initialize session state for hypothesis statuses
if "hypothesis_statuses" not in st.session_state:
    st.session_state.hypothesis_statuses = {
        h["id"]: h["default_status"] for h in HYPOTHESES
    }
if "hypothesis_evidence" not in st.session_state:
    st.session_state.hypothesis_evidence = {h["id"]: "" for h in HYPOTHESES}

# Render each hypothesis as an expandable card
for h in HYPOTHESES:
    hid = h["id"]
    status = st.session_state.hypothesis_statuses[hid]
    status_emoji = {"UNTESTED": "⬜", "CONFIRMED": "✅", "PARTIALLY CONFIRMED": "🟡",
                    "FALSIFIED": "❌", "INCONCLUSIVE": "🔵"}

    with st.expander(f"{status_emoji.get(status, '⬜')} {hid}: {h['title']}", expanded=False):
        st.markdown(f"**Prediction:** {h['prediction']}")
        st.markdown(f"**Measurement:** {h['measurement']}")
        st.markdown(f"**Conditions that test this:** {', '.join(h['conditions'])}")

        col1, col2 = st.columns([1, 3])
        with col1:
            new_status = st.selectbox(
                "Status",
                STATUS_OPTIONS,
                index=STATUS_OPTIONS.index(status),
                key=f"status_{hid}",
            )
            if new_status != status:
                st.session_state.hypothesis_statuses[hid] = new_status

        with col2:
            evidence = st.text_area(
                "Evidence / notes",
                value=st.session_state.hypothesis_evidence[hid],
                key=f"evidence_{hid}",
                height=80,
            )
            st.session_state.hypothesis_evidence[hid] = evidence

# Summary table
st.divider()
st.subheader("Summary")
summary_data = []
for h in HYPOTHESES:
    status = st.session_state.hypothesis_statuses[h["id"]]
    summary_data.append({
        "Hypothesis": f"{h['id']}: {h['title']}",
        "Status": status,
        "Evidence": st.session_state.hypothesis_evidence[h["id"]][:80] or "—",
    })
st.dataframe(summary_data, use_container_width=True)
