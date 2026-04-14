# Background

This document covers the concepts you need to understand the project if you're coming from an RL background. It's not a survey of the field — it's the minimum scaffolding to make our experimental plan and extension design doc legible.

---

## The two-loop architecture

Most RL systems have one optimization loop: a learning algorithm (PPO, DQN, etc.) trains a policy to maximize a reward function. The reward function is fixed — designed by the researcher before training starts.

Our system has two nested loops:

**Inner loop (PPO, within a single agent's lifetime):** Each agent is born with a reward function and a randomly initialized policy network. PPO trains the policy to maximize the agent's inherited reward. This is standard RL — nothing unusual here except that the reward function varies across agents.

**Outer loop (evolution, across generations):** The reward function parameters are the *genome* — the heritable component. When an agent reproduces, its offspring inherits the reward weights with small random mutations. Natural selection acts on these reward weights: agents whose reward functions lead to behavior that accumulates energy (eating food, avoiding predators) survive longer, reproduce more, and pass their reward weights to the next generation.

The key unintuitive point: **the policy is not inherited**. Every newborn starts from scratch with random weights and must learn behavior from zero via PPO. What passes from parent to child is *what the agent finds rewarding*, not *what the agent has learned to do*. Evolution shapes motivation; RL shapes competence.

This is directly inspired by biology: animals inherit their dopaminergic reward circuitry (what feels good, what feels bad) through genetics, but they must learn behavioral skills during their lifetime. A gazelle is born with an innate fear response to predator stimuli (inherited), but it must learn the specific motor skills to flee effectively (individual learning).

---

## K&D's environment and birth-death dynamics

The simulation is a continuous 2D world (960×960 units) with rigid-body physics. Agents are circular entities with proximity sensors — not a grid world. There are three types of entities: prey (blue), predators (red), and food (green).

**Sensors.** Each agent has 32 proximity sensors in a 120-degree forward arc, plus 18 tactile sensors around its body. Sensors detect the type and distance of the nearest object (food, predator, conspecific, wall).

**Actions.** Agents apply force to two rear points on their body, producing movement. The mapping from sensor inputs to force outputs is the learned policy.

**Energy and metabolism.** Each agent has an energy level that increases from eating (+1 per food item for prey; +6–10 per prey caught for predators) and decreases from basal metabolism and motor activity. Energy is the bridge between behavior and evolutionary fitness — there is no separate "fitness score."

**Death.** Agents die when energy drops to zero, or stochastically via a hazard function h(t,e) that increases with age and decreases with energy. This produces natural lifespans without a hard cutoff.

**Reproduction.** Agents reproduce asexually with probability b(e) that increases with energy. Offspring inherit the parent's reward weights with Student's t(df=2, scale=0.4)-distributed mutations (allowing occasional large jumps) and spawn nearby. The parent loses a fraction of its energy. Newborns get a fresh, randomly initialized policy network.

**No explicit fitness function.** This is important: nobody defines "fitness = total energy" or "fitness = number of offspring." Reproductive success *emerges* from the energy dynamics. An agent that evolves reward weights leading it to eat efficiently and avoid predators will naturally accumulate energy, reproduce more, and spread those reward weights. This is what the embodied evolution literature calls "implicit fitness" — the environment itself is the fitness function.

**Population dynamics.** The system produces Lotka-Volterra-like oscillations: when predators are scarce, prey populations grow; abundant prey supports more predators; predator population grows; prey population declines; predators starve; cycle repeats. K&D's Figure 6 shows these oscillations clearly. The oscillations create temporally varying selection pressure — prey must evolve reward functions robust enough to survive the peaks of predation, not just the average.

---

## What K&D discovered

The headline results from their predator-prey experiments:

**Fear emerges.** Prey evolve negative w_pred (reward weight for predator proximity), meaning they experience negative reward when predators are nearby. This drives PPO to learn avoidance behavior. Nobody programmed fear — evolution discovered that a negative reward signal for predator proximity leads to survival-promoting behavior.

**Social affiliation emerges.** Prey evolve positive w_prey (reward weight for conspecific proximity), meaning they experience positive reward when near other prey. This drives grouping/flocking behavior. Again, not programmed — evolution discovered that social reward leads to diluted predation risk.

**Reward weight branching.** Within a single population, different lineages evolve different reward strategies. Some prey develop strong fear with weak social affiliation; others develop strong social affiliation with weak fear. These are alternative survival strategies coexisting in the same population — analogous to biological polymorphism.

**Environmental sensitivity.** When predator mouths are larger (more lethal predators), prey evolve stronger fear. When food is scarcer, prey evolve weaker social affiliation (solitary foraging becomes more important than group safety). The evolved reward structures are adaptive responses to ecological conditions.

---

## Coevolutionary dynamics: what to expect and what can go wrong

Predator-prey is a coevolutionary system: each species creates the selection pressure for the other. Predators that catch prey survive; prey that avoid predators survive. As one side adapts, the other must counter-adapt. This can produce several dynamics:

**Arms races.** Each species drives increasing sophistication in the other. Prey evolve faster evasion; predators evolve better pursuit; prey evolve group defense; predators evolve coordinated hunting. This is the productive dynamic we hope to see — sustained complexity growth driven by competitive pressure. Bansal et al. (2018) showed this can produce remarkably complex behaviors (wrestling, grappling) from simple environments.

**Red Queen cycling.** Populations chase each other in circles without net progress. Predators get good at catching slow prey → prey evolve speed → predators can't catch fast prey → predators evolve for slow-but-sneaky prey → cycle repeats. This looks like oscillation in strategy space without cumulative improvement.

**Mediocre stable states.** Both populations converge on low-sophistication strategies that are best-responses to each other. Nobody can improve unilaterally, even though both could theoretically be much more capable. This is the coevolutionary analog of a Nash equilibrium at a suboptimal level.

**Disengagement.** One population becomes so dominant that there's no selection gradient for the other. If predators always catch prey instantly, there's no variation in prey fitness to select on — the arms race stalls.

These pathologies are documented in the coevolution literature (Rosin & Belew 1997). Recognizing them in our simulation output is important for interpreting results — a flat reward trajectory might mean "evolution converged" or it might mean "the system is stuck."

---

## The capacity utilization question

K&D's reward function is a linear function of instantaneous, position-only observations. This is the simplest possible reward representation. Our project asks: what happens when we give evolution access to richer representations?

The framing is deliberately agnostic about direction. We're not asking "does a richer reward function improve performance?" (that would be an engineering question). We're asking "does evolution *exploit* a richer reward function?" — and treating both answers as informative.

If evolution does exploit richer capacity (e.g., the MLP reward genome develops nonlinear, context-dependent fear), that tells us simple linear reward is leaving adaptive value on the table. If evolution doesn't (e.g., the MLP converges to something approximately linear), that tells us the simple structure is sufficient — which would explain why many biological organisms rely on relatively simple neuromodulatory systems despite having neural capacity for more complex ones.

Each axis has a specific metric for capacity utilization so the answer is quantitative, not just qualitative.

---

## The Doya connection (why temporal reward matters)

Kenji Doya (2002) proposed a mapping between biological neuromodulators and RL meta-parameters: dopamine encodes temporal difference error (the difference between expected and actual reward), serotonin modulates the discount factor, noradrenaline controls exploration, and acetylcholine regulates learning rate.

The critical detail for our project: dopamine doesn't encode reward from the current stimulus. It encodes *prediction error* — a fundamentally temporal signal that compares what happened to what was expected based on recent history. A monkey trained that a light predicts juice shows dopamine firing at the light (prediction of reward), not the juice (expected reward), and a dopamine *dip* when the light appears but juice doesn't follow (worse than predicted).

K&D's instantaneous reward function can't represent this. `r(t) = w · stimulus(t)` responds only to the current state. Our Axis 3 (temporal context window) gives the reward function access to recent history: `r(t) = MLP(obs(t), obs(t-1), ..., obs(t-k))`. This means evolution *can* discover prediction-error-like reward structures — reward that depends on whether things are getting better or worse, not just how they are right now.

Doya's framework predicts that evolved reward functions with temporal context *should* converge on prediction-error-like structures, because prediction error is the reward signal that produces the most effective learning. If we observe this convergence, it's computational evidence for Doya's theory about why dopamine works the way it does. If we don't, it suggests that prediction-error reward, while theoretically optimal, may not be easily discoverable by evolution — which is also informative.

---

## Shared vs. independent policies (why we test both)

In K&D's setup, every agent has its own policy network trained independently via PPO. With 150 agents, that's 150 separate PPO training runs per simulation step. This is faithful to biology (each organism learns individually) but computationally expensive.

Our speedup: train a *single shared policy* per species, conditioned on the agent's reward weights as input. The shared policy learns "given these reward weights, how should I behave?" All agents in a species share this network; behavioral variation comes from having different reward genomes, not from idiosyncratic learning.

This isn't just an engineering choice — it changes the dynamics. Under independent policies, two agents with identical reward weights can develop different behavioral strategies through the randomness of their individual training. Under shared policy, same reward weights → same behavior. This eliminates one source of behavioral diversity and may affect what reward strategies evolution discovers.

We formalize this as hypotheses and test both modes in Phase 1 of the experimental plan.
