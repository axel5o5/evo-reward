# Project Understanding: Evolved Reward Structures in Predator-Prey RL
### A human-readable synthesis — for collaborators, teammates, and future-you

*Columbia University — Advanced Reinforcement Learning, Spring 2026*
*Author: Axel. Last updated: April 2026.*

---

## What this document is

This is the document you'd hand to a lab member before a meeting, or read yourself after two weeks away from the project. It explains what we're building and why, how the pieces connect, what the key decisions were and the reasoning behind them, and where things are going. It assumes you have an RL background but no prior exposure to this specific work. It is deliberately not a paper and not a spec — those exist elsewhere. This is the conceptual map.

---

## The core idea in one paragraph

Most RL systems have a fixed reward function designed by the researcher. This project asks: what if the reward function itself evolved? Kanagawa & Doya (2025) showed that when RL agents in a predator-prey world reproduce and die based on their energy levels, natural selection shapes their reward functions over time — without any hand-design. Prey agents reliably evolve *fear* (a negative reward signal for seeing predators) and *social affiliation* (a positive reward signal for being near other prey), because these reward structures lead to survival-promoting behavior. Our project replicates this result and then asks a deeper question: **what reward structures does evolution discover when agents have richer representational capacity?** We extend the system along four axes — nonlinear reward functions, richer social observation, temporal reward depth, and memory-augmented policies — and measure whether evolution actually exploits what we give it. Both outcomes (exploitation and non-exploitation) are scientifically meaningful.

---

## Why this is an RL project, not just an evolution project

The professor flagged an early version of this project as "sounding like neuroevolution, not RL." The distinction matters and is worth understanding clearly.

In pure neuroevolution, evolution optimizes behavior directly — the genome encodes the policy, and selection acts on behavioral outcomes. There's no learning within a lifetime. The mechanism is evolutionary computation, not reinforcement learning.

In this project, the genome encodes the **reward function**, not the policy. Each agent must learn its behavior from scratch during its lifetime using PPO — a standard RL algorithm. What it learns *to do* depends on what it finds *rewarding*, which is determined by its inherited genome. Selection then acts on the reward genome indirectly: agents whose inherited reward function drives good behavior survive longer, accumulate more energy, reproduce more, and spread their reward weights to the next generation.

This is a two-loop system:
- **Inner loop (PPO, within a lifetime):** Standard RL. Policy is learned from scratch using the inherited reward signal.
- **Outer loop (evolution, across generations):** Reward function parameters are mutated and selected. No gradients — only mutation and survival pressure.

The scientific question lives at the intersection: how does the capacity of the inner-loop learner (what the policy can represent) interact with what the outer-loop selector (evolution) discovers? That's an RL question. The reward function is the object of study, and RL is the mechanism by which it's evaluated.

---

## The biological intuition

A gazelle is born with an innate fear response to predator stimuli. This isn't learned — it's encoded in the dopaminergic reward circuitry it inherits from its parents. What the gazelle *does* with that fear — which escape route to take, how fast to run, when to stop — is learned during its lifetime through experience.

The gazelle's genome encodes what it finds rewarding (fear of predators, attraction to food, comfort in the presence of other gazelles). Its individual learning encodes what it has figured out how to do about those rewards. Both are heritable in a biological sense, but reward circuitry evolves much more slowly and is much more conserved across individuals of the same species than learned behavioral strategies.

This project simulates exactly that structure. The genome is the reward function. The lifetime learning is PPO. The environmental pressure is predators trying to eat you.

Kenji Doya (2002) formalized this connection: he argued that the brain's neuromodulatory systems (dopamine, serotonin, acetylcholine, noradrenaline) implement the meta-parameters of reinforcement learning — dopamine encodes prediction error, serotonin controls the discount horizon, and so on. If that's right, then the evolution of reward systems *is* the evolution of RL meta-parameters. This project tests one consequence of that theory: if you give evolution access to richer reward representations, does it discover richer reward structures, including the prediction-error-like signals that Doya argues dopamine encodes?

---

## The simulation world

The environment is a continuous 2D world, 960×960 units, with rigid-body physics. There are three kinds of entities: prey (blue circles, radius 10), predators (red circles, radius 14), and food (green dots).

Each agent has sensors: 32 proximity sensors spread across a 120-degree forward arc, each returning the inverse distance to the nearest object (1.0 at contact, 0.0 when nothing's in range), plus 18 tactile sensors around its body for detecting contact. The policy network takes all sensor readings plus the agent's own heading, speed, angular velocity, and energy level as input — roughly 54 values total — and outputs forces applied to two rear points on its body, producing differential drive motion.

Prey eat food by touching it within their forward range. Each food item gives +1 energy. Predators hunt prey by catching them within a forward mouth range (default: 60 degrees, 40–80 units ahead). A successful catch transfers energy from prey to predator. Everyone metabolizes energy at a baseline rate and burns additional energy proportional to motor activity. The world is an energy economy: you must eat to survive, and you must avoid being eaten.

Birth and death are probabilistic, governed by two functions of age and energy:
- The **hazard function** h(t,e) is the per-step probability of death. It increases with age and increases sharply when energy falls below about 15–20 units. You can live for roughly a million steps in theory, but low energy dramatically shortens your expected lifespan.
- The **birth function** b(e) is the per-step probability of reproduction. It increases sigmoidally with energy. Prey need about 20–30 energy units to start reproducing; predators need 240–260.

There is no designer-imposed fitness function. Reproductive success emerges from behavior: agents that eat efficiently and avoid predators accumulate energy, reproduce more, and spread their reward weights. This is what the artificial life literature calls "implicit fitness" — the environment itself is the fitness function.

---

## The reward genome

Each agent inherits a reward genome from its parent. In the baseline (K&D faithful) version, this is just four scalar weights:

```
r(t) = w_eat  * n_food_eaten(t)
     + 0.01 * w_act  * (||motor_output|| / 114.0)
     + 0.1  * w_prey * max_proximity_to_conspecifics(t)
     + 0.1  * w_pred * max_proximity_to_opposing_species(t)
```

The four weights `[w_eat, w_act, w_prey, w_pred]` are the genome. The 0.01 and 0.1 scaling coefficients are fixed — they normalize the smaller, continuous sensor signals to be comparable in magnitude to the sparse eating events. The genome weights start near zero (sampled from N(0, 0.1)) and drift over evolutionary time.

At reproduction, the child inherits the parent's weights plus heavy-tailed noise: mutations are drawn from a Student's t distribution with 2 degrees of freedom and scale 0.4. This is heavier-tailed than Gaussian — most mutations are small, but occasional large jumps allow evolutionary exploration across the reward landscape. All weights are clipped to ±100 after mutation.

The policy network is NOT inherited. Every newborn starts with a randomly initialized MLP policy and must learn behavioral competence from scratch via PPO. What passes from parent to child is motivation — what the agent finds rewarding — not skill.

---

## What K&D discovered (the result we're replicating)

After running this system for about 10 million steps (roughly 473–501 prey generations), K&D found that:

**Fear evolved.** Prey reliably evolved negative `w_pred` — a negative reward signal for predator proximity. This drives PPO to learn avoidance behavior. Nobody programmed fear; evolution discovered that negative reward for predator stimuli produces survival-promoting behavior.

**Social affiliation evolved.** Prey reliably evolved positive `w_prey` — positive reward for being near other prey. This drives grouping behavior. Evolution discovered that social proximity reduces predation risk (dilution effect) and is therefore worth rewarding.

**These emerged sequentially.** The v2 of the paper added an important finding: social reward tends to evolve *before* fear. An agent needs to be in a social group before the fear signal becomes strongly adaptive — individual fear without group behavior provides weaker protection than group cohesion with moderate fear.

**Reward weight branching.** Within the same population, different lineages evolve different strategies. Some prey develop strong fear with weak social affiliation; others develop strong social affiliation with weak fear. These coexist as alternative evolutionary stable strategies — a computational analog of behavioral polymorphism in nature.

**Population dynamics.** Prey and predator populations show Lotka-Volterra-like oscillations: prey grow when predators are scarce, predators grow when prey are abundant, prey decline as predation pressure rises, predators decline as prey become scarce, and the cycle repeats with a period of roughly one million steps.

---

## Our extensions: the four axes

K&D's reward function operates on instantaneous, position-only observations with a linear, 4-weight genome. We extend this along four axes independently:

**Axis 1 — Reward nonlinearity.** Replace the linear genome with a small MLP. This allows the reward function to be context-dependent: fear might be stronger when energy is low, weaker when surrounded by conspecifics. The question: does evolution exploit nonlinearity, or converge on something approximately linear anyway?

**Axis 2 — Social observation.** Give the policy network access to the heading and velocity of nearby conspecifics, not just their positions. A stationary neighbor versus a fleeing neighbor carry very different information. The reward genome is unchanged — we're enriching what the policy can perceive. The question: does evolved social reward shift when the policy can actually distinguish *what* conspecifics are doing?

**Axis 3 — Reward temporal depth.** Give the reward function a context window: `r(t) = MLP(obs(t), obs(t-1), ..., obs(t-k))`. This allows the reward function to respond to *change* rather than just *state*. A predator that was distant 10 steps ago and is now close is more threatening than one that has been close all along. Doya's theory predicts this should converge toward prediction-error-like structures — reward for "things are getting worse," not just "things are bad."

**Axis 4 — Policy temporal depth.** Replace the feedforward MLP policy with an LSTM. The policy gains memory. Even with an instantaneous reward function, the agent can now remember where the predator was, track its trajectory, and act on context. The interaction with Axis 3 is the interesting case: temporal reward without memory gives the agent a richer signal it can't fully act on; memory without temporal reward lets the agent remember but doesn't reward anticipatory behavior.

Each axis is tested independently first, then in key combinations. The 2×2 of (temporal reward × social observation) is the core experimental matrix.

---

## The framing: capacity utilization

The key insight that makes this project work as science regardless of what we find is the **capacity utilization framing**. We're not asking "does a richer reward function perform better?" That's an engineering question. We're asking "does evolution *use* the richer capacity?" — and we measure this specifically for each axis:

- **Axis 1:** Fit a linear approximation to the MLP reward output; measure the residual. High residual = evolution found nonlinearity useful.
- **Axis 2:** Compute mutual information between conspecific heading/velocity inputs and agent actions. Zero = agent ignores the social channel.
- **Axis 3:** Measure autocorrelation structure of the reward signal across the context window; check whether reward is more sensitive to recent observations than old ones.
- **Axis 4:** Measure hidden-state entropy of the LSTM across trajectories; ablate the hidden state mid-episode and measure the performance drop.

If evolution exploits the added capacity, that tells us simple linear reward is leaving adaptive value on the table — that richer reward structures are genuinely useful and evolution can discover them. If evolution doesn't exploit it, that's equally interesting: it suggests simple reactive reward is a strong attractor, which would explain why many real organisms rely on relatively simple neuromodulatory systems despite having the neural capacity for more complex ones. The result is scientifically meaningful either way.

---

## The codebase architecture

The code is organized around the principle of **build once, configure per-experiment**. Every module is written to support both the K&D-faithful baseline and all four extension axes. The configuration flag system allows switching between modes without code changes:

```python
config = {
    "policy_mode":           "independent" | "shared",
    "lifecycle_mode":        "continuous" | "generational",
    "reward_type":           "linear" | "mlp",
    "reward_context_window": 1,      # 1 = instantaneous, >1 = temporal
    "social_obs":            "position_only" | "position_heading_velocity",
    "policy_type":           "mlp" | "lstm",
    ...
}
```

The eight source modules have a clear division of responsibility:

- **`environment.py`** — the physical world: physics via phyjax2d (Kanagawa's own JAX 2D physics library), sensor geometry, food dynamics, eating/capture detection. Knows nothing about rewards or evolution.

- **`lifecycle.py`** — energy and demographics: the hazard function h(t,e), birth function b(e), energy update equations for prey and predators, birth/death events. This is where the implicit fitness function lives.

- **`agents.py`** — the interface between world and mind: constructs the observation vector from world state, extracts the stimulus scalars the reward equation needs. The observation vector layout is pinned in `docs/interfaces.md` and must not change.

- **`reward.py`** — the genome computation: linear or MLP reward function applied to stimuli. The four-weight linear genome is the baseline. MLP and temporal variants are stubs until Phase 2.

- **`evolution.py`** — reproduction: Student's t(df=2) mutation on reward weights, offspring creation (new position, inherited genome, fresh policy). The mutation distribution is heavy-tailed by design.

- **`policy.py`** — the learned component: 3-layer MLP (or LSTM) mapping observations to actions. Three layers, 64 hidden units, tanh activation. Action output is a Gaussian distribution; the agent samples from it. The value head shares the same trunk.

- **`ppo.py`** — the learning algorithm: Proximal Policy Optimization with Generalized Advantage Estimation. This is standard PPO — the important thing is the exact hyperparameters (γ=0.999, N=1024, 10 epochs, lr=3e-4) which come directly from K&D's Table 4 and must not be changed for the faithful replication.

- **`metrics.py`** — logging and checkpointing: everything that gets saved to disk. The primary output is `metrics.npz` — a time series of reward weight trajectories, population sizes, and ecological metrics that can be loaded and analyzed post-hoc.

The simulation loop runs in this order each step: observe → sample actions → step physics → detect eating → compute rewards → update energy → process births/deaths → regenerate food → (conditionally) PPO update → log.

---

## The architectural comparison as a contribution

One of the most interesting design decisions in this project is treating what started as an engineering shortcut as a scientific variable.

K&D's faithful setup runs independent PPO for each agent — 150 agents × individual learning = expensive. Our computational simplification uses a **shared policy** conditioned on the reward genome, reducing PPO updates from O(N agents) to O(1) per species. This is faster, but it changes the dynamics: under shared policy, two agents with identical genomes behave identically. There's no idiosyncratic learning.

Rather than treating this as just an approximation to validate and move past, we formalized it as a scientific comparison with seven testable hypotheses (H1–H7) covering reward convergence speed, behavioral diversity, reward weight branching patterns, population dynamics, Baldwin effect strength, and implicit cultural transmission. The shared policy actually creates something interesting: a form of population-level knowledge transfer, where newborns immediately inherit behavioral competence from a policy trained on the entire species' experience — a computational analog of weak cultural transmission.

We also made continuous vs. generational lifecycle a scientific variable. K&D's continuous birth-death produces Lotka-Volterra oscillations and an implicit fitness function. Generational batching requires an explicit fitness metric and produces flat population sizes. These are not equivalent, and the differences are scientifically interesting.

This is a general pattern worth noting: when you're resource-constrained and must simplify, the simplifications themselves become variables you can study. The constraint becomes a contribution.

---

## The experimental plan

The project runs in phases, each with a hard gate before proceeding:

**Phase 0 (infrastructure):** Build the eight modules, pass unit tests. No science yet.

**Phase 1a (K&D replication):** Run the faithful baseline and confirm fear and social affiliation emerge in at least 3 of 5 seeds. This is the proof of concept. Nothing else matters until this works.

**Phase 1b (architectural comparison):** Compare shared vs. independent policy. Tests H1–H7. Determines the default mode for extension experiments.

**Phase 2 (one-axis ablations):** Test each extension axis independently, 1 seed each for breadth scan. Looking for qualitative signal — does capacity utilization register?

**Phase 3 (key combinations):** The 2×2 of temporal × social, plus LSTM interactions. Invest depth where signal appeared in Phase 2.

**Phase 4 (depth + report):** 5 seeds on the most interesting conditions. Write the report.

The seed strategy throughout is exploration-first: scan at 1 seed, invest depth only where signal appears. K&D used 5–6 seeds for their final claims; we target the same.

---

## What success looks like

The minimum viable scientific result is Phase 1a: fear and social affiliation emerge from our replication. That, plus the architectural comparison from Phase 1b, is already a publishable observation about the robustness and mechanisms of reward evolution.

A strong result would additionally show one of: (a) evolution exploits nonlinear reward capacity in ways that produce qualitatively different evolved structures, (b) social observation shifts what evolution discovers about conspecific reward, (c) temporal reward context produces something resembling prediction-error structure in the evolved reward function — which would be computational evidence for Doya's theory about why the dopaminergic system works the way it does.

Either direction on any of these axes is interesting. The project is structured so that "evolution doesn't use expanded capacity" is as publishable as "evolution does" — it just tells a different story about the robustness of simple reward structures.

---

## Document map

For a human reading this project:

- **This document** — the conceptual map; start here
- **`docs/background.md`** — deeper conceptual grounding on the two-loop architecture, K&D's environment, coevolutionary dynamics
- **`docs/full-extension-design-doc.md`** — detailed scientific rationale for each extension axis and what we'd expect to find
- **`docs/experimental-plan.md`** — the full phased plan with hypotheses, seed strategies, compute budgets, and decision rules
- **`papers/kanagawa-doya-2025-v2.pdf`** — the paper we're replicating; read Sections 3–4 and Appendix A

For technical implementation:

- **`docs/technical-spec-kd-replication.md`** — every numerical parameter from K&D's tables, the exact reward equation, Phase 1a success criteria
- **`docs/interfaces.md`** — module contracts, function signatures, data structures, observation vector layout
- **`docs/emevo-diff.md`** — what we change from K&D's open-source code and why
- **`docs/development-roadmap.md`** — build order, full test specifications, engineering improvement backlog
- **`AGENTS.md`** — instructions for coding agents (gate sequence, key numbers, simulation loop order)

---

## A note on the longer vision

The course project is the first layer of something larger. This semester: evolved reward functions. The reward function is the evolved component, PPO is the evaluation mechanism.

The natural next step is to replace the static reward function with an evolved plasticity rule — a mechanism that determines how the agent's own neural connections change during its lifetime. Instead of inheriting "what to find rewarding," the agent inherits "how to learn." This maps onto Doya's neuromodulation framework more directly: neuromodulatory systems don't just signal reward, they gate synaptic plasticity. The agent would arrive with random weights and bootstrap competence through evolved Hebbian learning rules, with no external reward signal at all.

Beyond that: add social observation and trajectory-based learning, and you have the substrate for teaching and cultural transmission to emerge from first principles — not because you engineered it, but because evolved plasticity rules operating on behavioral observation of conspecifics produce it spontaneously.

Each step builds on the previous one. The course project validates the foundations.
