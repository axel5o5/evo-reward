# evo-reward

Research repository for **Evolved Reward Structures in Predator-Prey RL: How Agent Capacity Shapes What Evolution Discovers** — a graduate project in Columbia University's Advanced Reinforcement Learning course.

---

## Overview

Kanagawa & Doya (2024, 2025) showed that RL agents in a predator-prey world can *evolve their reward functions* via birth-death selection pressure. Without any hand-design, prey agents reliably develop **fear** (negative reward for predator stimuli) and **social affiliation** (positive reward for conspecific proximity). The inner loop is PPO; the outer loop is natural selection on the reward genome.

Their agents use a **fixed linear reward** over **instantaneous, position-only observations**. This project asks:

> **What reward structures does evolution discover when agents have richer representational capacity?**

The central framing is *capacity utilization*: does evolution exploit expanded representational capacity, or converge on the same simple structures regardless? Both outcomes are treated as scientifically informative.

---

## Four Axes of Extension

The project extends K&D along four independently controlled axes, forming a 2×2×2×2 space of 16 conditions:

| Axis | Baseline (K&D) | Extension |
|------|---------------|-----------|
| **1 — Reward nonlinearity** | Linear `r(t) = w · stimulus(t)` | MLP genome: `r(t) = MLP_θ(stimulus(t))` |
| **2 — Social observation** | Conspecific positions only | + heading and velocity of neighbors |
| **3 — Reward temporal depth** | Instantaneous stimulus | Context window: `r(t) = MLP(obs(t), ..., obs(t−k))` |
| **4 — Policy temporal depth** | Feedforward MLP policy | LSTM policy |

Axes 3 and 4 interact by design: temporal reward without memory provides a richer signal the agent can't act on; memory without temporal reward lets the agent use past experience but doesn't reward temporal patterns. The 2×2 of (temporal reward × LSTM) is a key comparison.

---

## Capacity Utilization Metrics

Each axis has a dedicated metric for whether evolution *used* the added capacity:

- **Axis 1** — MLP residual from best linear approximation (nonlinearity actually exploited?)
- **Axis 2** — Mutual information between heading/velocity observations and evolved reward weights
- **Axis 3** — Autocorrelation structure of the evolved temporal reward signal (prediction-error-like?)
- **Axis 4** — Hidden-state entropy of the LSTM across a trajectory

---

## Architecture

The codebase is config-driven. Two independent flags control the core architectural mode:

```python
config = {
    "policy_mode":    "independent" | "shared",
    "lifecycle_mode": "continuous"  | "generational",
    "reward_type":    "linear"      | "mlp",
    "reward_context_window": 1,          # 1 = instantaneous, >1 = temporal
    "social_obs":     "position_only" | "position_heading_velocity",
    "policy_type":    "mlp"         | "lstm",
    "coevolution_mode": "concurrent" | "alternating",
    "population_size": 80,
    "num_generations": 300,
    "mutation_scale":  0.02,
    ...
}
```

**Default for extension experiments:** `shared` policy + `continuous` birth-death. This preserves the ecological grounding of K&D (no designer-imposed fitness function, energy-driven reproduction, overlapping generations) while amortizing PPO updates from O(N) to O(1) via reward-genome-conditioned policy sharing.

The architectural comparison itself — independent vs. shared policy, continuous vs. generational lifecycle — is formalized as a scientific contribution with seven testable hypotheses (H1–H7) covering reward convergence speed, intra-species behavioral diversity, reward weight branching, population dynamics, Baldwin effect strength, and implicit cultural transmission.

### Codebase structure

```
emevo-ext/
├── configs/                 # One YAML per experiment condition
├── src/
│   ├── environment.py       # JAX 2D world, physics, food, collisions
│   ├── agents.py            # Observation construction, reward computation
│   ├── policy.py            # Policy networks (MLP + LSTM, shared + independent)
│   ├── reward.py            # Reward genome (linear + MLP variants)
│   ├── evolution.py         # Selection, mutation
│   ├── lifecycle.py         # Continuous birth-death vs. generational batching
│   ├── ppo.py               # PPO inner loop
│   └── metrics.py           # All measurement functions
├── analysis/
│   ├── dashboards.py        # Per-run visualization
│   ├── comparison.py        # Cross-condition comparison plots
│   └── capacity_util.py     # Capacity utilization metrics
├── scripts/
│   ├── run_experiment.py
│   ├── run_sweep.py
│   └── analyze_results.py
└── results/                 # results/{condition}/{seed}/
```

---

## Experimental Plan

| Phase | Description | Seed strategy |
|-------|-------------|---------------|
| **0** | Infrastructure, measurement setup, K&D environment | — |
| **1a** | Faithful K&D replication | 1–2 seeds |
| **1b** | Architectural comparison (H1–H7): 2×2 of policy × lifecycle mode | 1 seed per cell |
| **2** | One-axis ablations: each extension axis in isolation | 1 seed (breadth scan) |
| **3** | Key combinations: temporal × LSTM, temporal × social, IBR vs. concurrent coevolution | 1 seed then depth |
| **4** | Full factorial depth on interesting conditions | 5 seeds for final claims |
| **5** | Relaxing simplifications, alternative fitness functions (stretch) | — |

**Decision rules:** Phase 1a must reproduce fear and social affiliation before proceeding. If shared policy breaks the core result (H5 falsified), fall back to independent policies for all extension experiments and write up the falsification as a finding. After the Phase 2 breadth scan, invest depth only where qualitative signal appears.

---

## Key References

- Kanagawa & Doya (2025). *Evolution of fear and social rewards in prey-predator relationship.* arXiv:2507.09992.
- Kanagawa & Doya (2024). *Evolution of rewards for food and motor action by simulating birth and death.* ALIFE 2024.
- Doya (2002). *Metalearning and neuromodulation.* Neural Networks, 15(4–6).
- Du et al. (2019). *LIIR: Learning individual intrinsic reward in multi-agent RL.* NeurIPS 2019.
- Christianos et al. (2021). *Shared experience multi-agent RL.* NeurIPS 2021.
- Hornby (2006). *ALPS: Age-layered population structures.*
- Bredeche & Montanier (2012). *mEDEA: Embodied evolution with implicit fitness.*

---

## Presentations

- [Pitch deck](https://docs.google.com/presentation/d/1gQ6SUbUOerLji3fFnF4-DhIVRfg_QVS6/edit?usp=sharing&ouid=117130514047298135284&rtpof=true&sd=true)

---

## Repo Contents

```
reward-evo/
├── docs/           # Planning documents and writeups
├── papers/         # PDFs of key references
└── src/
    ├── configs/    # One YAML per experiment condition
    ├── src/        # Environment, agents, policy, reward, evolution, PPO
    ├── analysis/   # Visualization and capacity utilization metrics
    └── results/    # Organized by condition and seed
```

---

*Columbia University — Advanced Reinforcement Learning, Spring 2026*
