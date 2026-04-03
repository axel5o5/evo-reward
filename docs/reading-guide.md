# Reading Guide

This is a map of what to read and in what order to get up to speed on the project. You don't need to read every paper cover-to-cover — the goal is to understand what we're doing well enough to contribute ideas and code.

---

## Start here

### 1. Background concepts

**`docs/background.md`**

If you're coming from RL and haven't worked with evolutionary computation or artificial life before, read this first. It covers the two-loop architecture (evolution on reward functions, PPO on policy), how K&D's birth-death simulation works, coevolutionary dynamics and pathologies, the capacity utilization framing, and the Doya connection for temporal reward. ~2000 words, written for this project specifically.

### 2. The paper we're building on

**Kanagawa & Doya (2025), "Evolution of Fear and Social Rewards in Prey-Predator Relationship"**
`docs/papers/evolution-of-fear-and-social-rewards.pdf`

The single most important paper. Focus on:

- **Section 3** — the simulation model, sensors, energy/metabolism, birth-death dynamics, the reward function (Equation 3), and Algorithm 1 (the simulation loop).
- **Section 4.1** — the evolved reward analysis. Figures 7, 8, and 12 are the key results we need to reproduce and extend.
- **Skim the rest** — environmental variations (mouth size, food density, pitfalls) show how ecological conditions shift what evolves.

Their code is at `github.com/oist/emevo` and included in our repo under `emevo/`.

If the 2025 paper feels dense on first read, the 2024 precursor (`docs/papers/evolution-of-rewards-for-food.pdf`) is a simpler single-species version — same framework, fewer moving parts.

### 3. Our project design

**`docs/experimental-plan.md`** — the master plan. Covers architecture modes (shared vs. independent policy, continuous vs. generational lifecycle), seven testable hypotheses (H1–H7) for the architectural comparison, the phased experimental timeline, seed strategy, and decision rules. This is the canonical reference for what we're building and why.

**`docs/full-extension-design-doc.md`** — the technical design for each extension axis. Read at least the baseline description and Extensions 2 (social observation) and 3 (temporal reward), which are highest priority. Each section covers what changes mechanically, what it means for the agent, and what we'd measure.

**`docs/project-description.md`** — one-page summary. Quick skim for the compressed version.

---

## Papers we build on

The K&D papers above are essential. The rest are references — know what they contribute so you can go deeper if a topic comes up.

**The inner-loop algorithm:**

- **Schulman et al. (2017), "Proximal Policy Optimization."** — PPO. The RL algorithm every agent uses to learn behavior within its lifetime. Read if you need a refresher; skim if you've used PPO before.

**Theoretical grounding:**

- **Doya (2002), "Metalearning and neuromodulation."** (`docs/papers/neuromodulated-meta-learning.pdf`) — Maps biological neuromodulators to RL meta-parameters. The theoretical backbone for why temporal reward (Axis 3) should converge on prediction-error-like structures. The background doc covers the key ideas; read the paper if you want the full argument.

- **Singh, Lewis & Barto (2009), "Where Do Rewards Come From?"** — Formalizes the optimal reward framework: reward functions that, when optimized by an RL agent, maximize an external fitness criterion. This is exactly what K&D implement with evolution as the outer loop. Worth reading the first few pages for the framing.

**Closest RL precedent:**

- **Du et al. (2019), LIIR.** (`docs/papers/liir-learning-individual-intrinsic-reward.pdf`) — Learns individual intrinsic reward functions in cooperative MARL via meta-gradients (not evolution). Uses shared policies. Look at Figures 4–5 for how they visualize learned intrinsic rewards — we'll want similar visualizations.

**Emergent behavior from competition:**

- **Bansal et al. (2018), "Emergent Complexity through Multi-Agent Competition."** — Simple competitive self-play produces complex behaviors (wrestling, grappling) nobody programmed. The empirical precedent for emergence from competition.

- **Baker et al. (2020), "Emergent Tool Use from Multi-Agent Autocurricula."** — Hide-and-seek agents going through six emergent phases. Establishes "counting behavioral phase transitions" as a methodology.

**On our architectural choices:**

- **Christianos et al. (2021), "Scaling Multi-Agent RL with Selective Parameter Sharing."** — When does sharing help vs. hurt? Directly relevant to our shared-policy comparison. Their finding: sharing helps homogeneous teams, can hurt heterogeneous settings.

- **Bredeche & Montanier (2012), mEDEA.** — Evolution without explicit fitness in robot swarms. Conceptual ancestor of K&D's birth-death dynamics. Relevant to understanding why our generational batching mode is a meaningful departure.

---

## Before Monday

1. Read `docs/background.md`.
2. Read the K&D 2025 paper (Sections 3–4, Algorithm 1, Figures 7/8/12).
3. Read `docs/experimental-plan.md` (architecture modes section + skim the phases).
4. Browse `github.com/oist/emevo`.
5. Come with thoughts on what would be interesting to measure or interpret in the evolved reward functions and learned policies.
