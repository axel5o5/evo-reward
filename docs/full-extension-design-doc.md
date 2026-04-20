# The Full Extension Space
### From reactive to anticipatory reward — and why each step is a deeper engagement with RL science

---

## The baseline we're extending

Kanagawa & Doya (2025) have agents whose reward function is:

```
r(t) = w_eat · food(t) + w_pred · predator(t) + w_prey · conspecific(t) + w_act · action(t)
```

Each stimulus term — food(t), predator(t), etc. — is a scalar describing how much of that stimulus is currently in the agent's visual field. The weight vector `w` is the genome. PPO trains a feedforward policy network against this reward for the agent's entire lifetime. When the agent reproduces, the child inherits `w` with Gaussian mutation and gets a fresh, randomly initialized policy network.

This is **reactive reward over instantaneous stimuli**. The agent's reward at time t depends only on what it sees at time t. No memory. No context. No anticipation. The policy network can learn temporal strategies (PPO with a feedforward network can still learn "if I see a predator, run for several timesteps"), but the *reward signal itself* is memoryless.

Let's lay out every way we might extend this, what each one means biologically and for RL science, and which combinations make sense for the project.

---

## Extension 1: State-dependent reward (MLP genome)

### What changes mechanically

Replace the dot product `w · stimulus(t)` with a small MLP:

```
r(t) = MLP_genome(food(t), predator(t), conspecific(t), energy(t), ...)
```

The genome now encodes MLP weights (~100-200 parameters) instead of 4-6 scalars. The MLP takes the same instantaneous stimulus inputs but can compute nonlinear, context-dependent functions of them.

### What this means for the agent

Fear is no longer a fixed scalar. The MLP can output:
- Strong negative reward for predator-close AND energy-low (I'm vulnerable and there's danger — maximum fear)
- Weak negative reward for predator-close AND conspecific-high (I'm in a group, the predator is less threatening)
- Positive reward for food AND predator-distant (safe to forage)
- Near-zero reward for food AND predator-close (not worth the risk right now)

These are *conditional* reward structures. The agent still has no memory — it's reacting to instantaneous state — but it's reacting in a richer, state-dependent way.

### Biological analog

This is closer to how actual neuromodulatory systems work. Dopamine release isn't a fixed function of stimulus type — it depends on context. Seeing food produces different dopamine responses depending on whether you're hungry or sated (energy level), whether you're in a safe or dangerous environment (predator presence), whether you're alone or in a group (social context). The Doya mapping says neuromodulators implement RL meta-parameters, but those meta-parameters are *state-dependent*, not fixed. A flat weight vector implements the crudest version of this; an MLP implements a richer version.

### Prior work comparison

Wang, Hughes et al. (2019, DeepMind) already used MLP reward networks — but in cooperative social dilemmas, not competitive predator-prey. Their evolved reward MLPs learned to encode social preferences (altruism, cooperation) that linear weights couldn't represent. We're asking: in a competitive ecology, does the same enrichment happen? Does fear become conditional? Does social reward become context-dependent?

### What we'd measure

Visualize the MLP's output as heatmaps over state variables. Plot `r(predator_distance, energy_level)` — does it show the conditional structure? Compare the evolved MLP reward landscape across prey lineages — do different lineages evolve different contextual strategies? Compare survival rates and behavioral diversity between linear-reward and MLP-reward conditions.

---

## Extension 2: Social observation channel

### What changes mechanically

Expand the observation vector that the policy network receives:

```
# Baseline (Kanagawa & Doya):
obs = [food_positions, predator_positions, conspecific_positions, own_state]

# Extension:
obs = [food_positions, predator_positions, conspecific_positions, own_state,
       conspecific_headings, conspecific_velocities]
```

The reward genome is unchanged — same flat vector or same MLP operating on the same stimulus channels. What changes is the information available to the *policy network* during PPO training.

### What this means for the agent

The agent can now distinguish between:
- A conspecific that's fleeing (heading away from predator at high velocity) — danger signal
- A conspecific that's approaching food (heading toward a food cluster at moderate velocity) — foraging signal  
- A conspecific that's stationary (not moving) — possibly dead, possibly safe
- A group of conspecifics moving together in the same direction — coordinated behavior worth joining

None of this is available in position-only observation. Position tells you *where* your neighbors are. Behavioral observation tells you *what they're doing* and *what they might know*.

### Biological analog

This is the sensory prerequisite for social learning. In nature, animals don't just detect conspecific presence — they read behavioral cues. A bird sees its flockmate's flight direction and adjusts accordingly. A fish in a school reads the velocity vectors of neighbors to maintain formation. A meerkat sees a sentinel's alert posture and takes cover. The information channel from "where are they" to "what are they doing" is the transition from aggregation to coordination.

### Connection to Ndousse et al. (2021) and Bhoopchand et al. (2023)

Ndousse et al. showed that vanilla model-free RL agents do NOT spontaneously develop social learning — they need the right inductive bias. Their agents could observe expert behavior but didn't learn from it without an auxiliary prediction loss forcing them to model what they'd see next. Our question is: does *evolved reward* provide a natural inductive bias? If evolution produces `w_prey > 0` for prey (the conspecific social reward — which Kanagawa & Doya showed it does), and agents can observe conspecific behavior, does the reward for proximity combined with behavioral observation produce coordination that neither element achieves alone?

Bhoopchand et al. (2023) showed that with the right architecture, agents can solve the correspondence problem — translating observation of another's behavior into motor reproduction — in rich 3D environments with no pre-collected demonstrations. They used fixed, hand-designed rewards. We're asking: do agents evolve the reward structures that make behavioral observation useful, or does evolution only discover proximity-seeking?

### What we'd measure

Coordination metrics: velocity alignment (do conspecifics move in the same direction?), group cohesion (do they maintain stable formations?), coordinated evasion (when a predator approaches a group, do they scatter in organized patterns or random directions?). Ablation: at test time, zero out the behavioral observation channels. If performance drops, the agent was using them. Compare these metrics between position-only and behavioral-observation conditions.

### The key novelty claim

Nobody has combined evolved reward functions with behavioral observation of conspecifics. The evolved-reward literature (Singh et al. 2009, Wang et al. 2019, Kanagawa & Doya 2025) gives agents positions of others. The social learning literature (Ndousse et al. 2021, Bhoopchand et al. 2023) gives agents behavioral observations but with fixed rewards. The intersection is empty. We're filling it.

---

## Extension 3: Temporal context window for the reward function

### What changes mechanically

Instead of the reward function seeing only the current timestep, it sees a short window of recent observations:

```
# Baseline:
r(t) = f_genome(obs(t))

# Temporal extension:
context(t) = [obs(t), obs(t-1), obs(t-2), ..., obs(t-k)]
r(t) = f_genome(context(t))
```

Where k is a small window, say 5-15 timesteps. The reward genome (MLP or flat weights) now takes a flattened vector of the last k observations as input. If each observation is 20-dimensional and k=10, the reward MLP input is 200-dimensional. The genome encodes how to map recent history to a scalar reward.

### What this means for the agent

The reward function can now encode temporal patterns:
- "I've been seeing a predator consistently for the last 10 steps" → high sustained fear (not just a momentary blip)
- "A predator was visible 10 steps ago but not now" → lingering caution (it went behind an obstacle)
- "Food was to my left 5 steps ago and is now to my right" → I'm circling it (maybe penalize inefficiency)
- "Conspecifics were scattered 10 steps ago and are now clustered" → grouping is happening (reward joining)
- "Predator was far 10 steps ago and is now close" → it's approaching (alarm signal stronger than static proximity)

This is the transition from reactive to **anticipatory** reward. The agent doesn't just respond to what's here now — it responds to what has been happening. It can fear a trajectory (predator-getting-closer) rather than just a state (predator-is-close).

### Why this is different from the LSTM approach and why that matters

An LSTM policy can learn temporal strategies from instantaneous reward — PPO with a recurrent network can learn "if I saw a predator recently, keep fleeing even though it's gone." The temporal information lives in the policy's hidden state. But the *reward signal* is still instantaneous. The policy uses memory to interpret a memoryless reward.

With a temporal context window on the reward function, the reward signal itself carries temporal information. The *signal the agent learns from* is different, not just the *way it processes it*. This matters because evolution is shaping the reward function, not the policy. If the reward function can't represent "predator-approaching is worse than predator-static," then evolution can't discover that as a useful reward structure, no matter how good the policy network is.

The practical advantage over coupling the reward to an LSTM hidden state (which we discussed earlier): no chicken-and-egg problem. The context window is raw observations — they're meaningful from birth. The agent doesn't need to learn good representations before the reward function becomes useful. On the first timestep of life, `context(t) = [obs(t), zeros, zeros, ...]`, and the reward function produces something based on the current observation (defaulting to the reactive baseline). As the agent lives and the window fills up, the reward function gains access to temporal patterns. This is a graceful warmup, not a bootstrap.

### Biological analog: the key Doya connection

This is where Doya's framework becomes deeply relevant rather than just theoretically supportive.

Doya (2002) argues that dopamine encodes **temporal difference error** — the difference between expected and actual reward. This is not a response to a stimulus. It's a response to a *temporal pattern*: "what happened was different from what I predicted based on recent experience." The classic experiment: a monkey learns that a light predicts juice. Initially, dopamine fires when juice arrives. After learning, dopamine fires when the light appears (prediction of juice) and *dips below baseline* if the light appears but no juice follows (worse than predicted). The signal is fundamentally temporal — it requires comparing the present to the recent past.

A reward function operating on a context window can represent exactly this structure. If the genome evolves a reward function where `r(t)` is high when `obs(t)` differs from the pattern of `obs(t-1)...obs(t-k)` in specific ways, it has evolved a prediction-error reward. Not because we told it to — because the temporal context window gives it the raw material to discover prediction-error-like reward structures, and natural selection favors agents whose reward drives effective learning.

This is deep: **the Doya mapping predicts that evolved reward functions with temporal context should converge on prediction-error-like structures** — because prediction error is the reward signal that produces the most effective learning. If we observe this convergence in our simulation, we've provided computational evidence for Doya's theory about why the dopaminergic system works the way it does.

### Connection to the Frémaux & Gerstner (2016) three-factor framework

The three-factor learning rule says: Δw = f(pre, post, M), where M is the neuromodulatory third factor. The eligibility trace (pre × post) flags which synapses were recently active. The modulatory signal M determines whether those traces get written into lasting weight changes.

In the biological system, M operates on a different timescale than the eligibility trace — the trace decays in 200ms-2s, while M reflects reward that arrives seconds later. The *temporal gap* between the activity (eligibility trace) and the evaluation (modulatory signal) is the whole point of the three-factor rule. It's what makes it possible to learn from delayed reward.

A reward function with a temporal context window implements a primitive version of this temporal gap. The reward at time t depends on what happened at times t-1 through t-k. The agent's learning algorithm (PPO) uses this temporally-informed reward to update the policy, bridging the gap between past actions and current evaluation. It's not the full three-factor rule (that requires per-synapse modulatory gating, which is the Phase 2 vision), but it's the same principle operating at the level of the reward signal rather than individual synapses.

### What we'd measure

Compare evolved reward functions between the instantaneous condition (baseline) and temporal-context condition. Specifically: does the temporal reward function develop prediction-error-like structure? Visualize by holding the context window constant except for one variable and seeing how r(t) changes — does it respond to *changes* rather than *levels*? Measure whether anticipatory avoidance emerges — does the agent avoid areas where predators *were* even when no predator is currently visible? Does this improve survival in environments with spatial structure (persistent danger zones, predator patrol routes)?

---

## How the extensions compose: the refined experimental design

### The two strongest axes

Given the analysis above, I think the two most interesting axes for the course project are:

**Axis 1: Reward temporal depth**
- Level A: Instantaneous reward — `r(t) = w · stimulus(t)` (Kanagawa & Doya baseline)
- Level B: Temporal context reward — `r(t) = MLP_genome(obs(t), obs(t-1), ..., obs(t-k))`

**Axis 2: Social information richness**
- Level X: Position-only observation of conspecifics
- Level Y: Position + behavioral state (heading, velocity) of conspecifics

This gives a 2×2:

| | Position-only (X) | + Behavioral obs. (Y) |
|---|---|---|
| **Instantaneous reward (A)** | AX: K&D baseline | AY: Can linear reward exploit behavioral info? |
| **Temporal context reward (B)** | BX: Does temporal reward → anticipatory fear? | BY: Full system — temporal + social |

### Why this 2×2 is sharper than the previous one

The previous version had linear vs. MLP as one axis. That's about function approximation capacity — an engineering question. The new version has instantaneous vs. temporal as the axis. That's about **what kind of reward structure evolution can discover** — a scientific question grounded in Doya's theory. The MLP is implicit in the temporal version (you need an MLP to map a 200-dimensional context window to a scalar), so you get the nonlinear capacity for free.

### What each cell tests

**AX (baseline replication):** Reproduce Kanagawa & Doya. Validate that fear (negative w_pred) and social reward (positive w_prey) emerge. This is your control condition and proof that your implementation works.

**AY (social observation with reactive reward):** Same flat reward weights, but the policy can see what conspecifics are doing. The reward doesn't change — evolution still operates on the same 4-6 scalars. But the policy has richer input. Question: does the evolved value of w_prey change when behavioral observation is available? If w_prey evolves *more* positive (stronger social reward), it means the *usefulness* of social proximity increased because agents can now coordinate, which makes evolution favor social reward more. The reward function didn't get more complex — but its fitness landscape shifted because the policy can exploit the behavioral channel.

**BX (temporal reward without social observation):** The reward function takes a context window, but agents only see positions of conspecifics, not behavior. Question: does anticipatory fear emerge? Does the temporal reward function learn to produce negative reward for "predator was getting closer over the last 10 steps" rather than just "predator is close now"? Does this improve survival in environments with ambush dynamics (predators that stalk slowly then sprint)?

**BY (full system):** Temporal context reward + behavioral observation. The richest condition. The reward function can represent temporal patterns in what conspecifics are doing — "my neighbors just started fleeing" generates different reward than "my neighbors are stationary." Question: does this combination produce qualitatively new behaviors that neither extension alone produces? Specifically, does it enable coordinated temporal strategies — not just grouping (AY) and not just anticipation (BX), but anticipatory coordination (a group that detects a forming threat and preemptively repositions together)?

### The overarching question

Across all four cells, the meta-question is: **what reward structures does evolution discover as a function of what the agent is capable of representing?** A reactive agent with position-only observation evolves reactive, stimulus-level reward (fear as a flat weight). A temporal agent with behavioral observation has the *capacity* to evolve much richer reward structures — but does it? Does evolution actually use the expanded representational capacity, or does it converge on the same simple structures because they're sufficient?

If evolution *does* discover richer structures (conditional fear, prediction-error-like reward, social-coordination reward), that's evidence that these structures are adaptively useful and that simple organisms might be leaving fitness on the table. If evolution *doesn't*, that's evidence that simple reactive reward is a stronger attractor — which would explain why many real organisms rely on relatively simple neuromodulatory systems despite having the neural capacity for more complex ones.

Either result is interesting. Either result is publishable. Either result tells us something about the relationship between agent capacity and evolved reward structure, which is the core question at the intersection of RL and evolutionary biology that Doya's framework points toward.

---

## The connection to the full research vision

### How each extension maps to the longer roadmap

**This semester (course project):** Evolved reward functions in RL agents. The reward function is the evolved component. PPO is the learning algorithm. The three-factor rule is not implemented — M(t) is a reward signal, not a plasticity gate. But the temporal context window version takes a meaningful step toward the Doya mapping by giving the reward function access to temporal patterns, the substrate from which prediction-error-like reward structures can emerge.

**Phase 2 (post-course):** Reintroduce within-lifetime plasticity. M(t) becomes a plasticity gate (Backpropamine architecture). The reward function is no longer the evolved component — instead, evolution shapes the plasticity *rules* that determine how the network modifies itself during a lifetime. The agent arrives with random weights and bootstraps competence through evolved Hebbian learning gated by M(t). If the course project shows that temporal reward structure improves fitness, Phase 2 asks: can evolved plasticity rules discover the same temporal structures from first principles?

**Phase 3 (the full vision):** Add trajectory-based social learning. The social observation channel from Extension 2 becomes the input to the evolved plasticity rules from Phase 2. Agents observe conspecific behavior and their evolved Hebbian plasticity extracts useful patterns from it. Teaching, trust, and cultural transmission emerge from the interaction of evolved plasticity and social observation — not because we engineered them, but because the three-factor learning rule operating on behavioral observation is sufficient to produce them.

The course project validates the components: temporal reward structure matters (Extension 3), social observation changes what evolves (Extension 2). Phases 2 and 3 replace the external reward function with internal plasticity, showing that the same phenomena can arise from a more biologically plausible mechanism.

---

## Summary: what to propose

### The pitch in one paragraph

We replicate and extend Kanagawa & Doya's (2024, 2025) evolutionary simulation of RL agents with evolved reward functions. In their framework, PPO-trained agents in a predator-prey world inherit reward function parameters from parents via birth-death dynamics, and natural selection discovers reward structures like fear (negative reward for predator stimuli) and social affiliation (positive reward for conspecific proximity). We extend this in two directions: (1) temporal context windows on the reward function, allowing evolution to discover anticipatory reward structures — not just "fear of seeing predators" but "fear of predators approaching," connecting to Doya's (2002) theory that neuromodulatory reward signals encode prediction error; and (2) social behavioral observation, allowing agents to perceive conspecific heading and velocity, not just position, connecting to the social learning literature (Ndousse et al. 2021, Bhoopchand et al. 2023) and asking whether evolved reward functions adapt to exploit richer social information channels. We study these extensions in a 2×2 factorial design and measure whether evolution discovers qualitatively different reward structures as a function of agent temporal and social capacity.

### Papers we cite and how we use them

| Paper | How we use it |
|-------|--------------|
| Doya (2002) | Theoretical foundation: neuromodulators as RL meta-parameters, prediction error as reward |
| Kanagawa & Doya (2024, 2025) | Direct baseline we replicate and extend |
| Singh, Lewis & Barto (2009) | Optimal Rewards Framework: the formalization of evolving reward functions |
| Wang, Hughes et al. (2019) | Prior work on evolved reward networks — cooperative, not competitive |
| Du et al. (2019, LIIR) | Learned intrinsic reward in MARL — gradient-based, not evolutionary |
| Ndousse et al. (2021) | Social learning requires inductive bias — our Extension 2 tests whether evolved reward provides it |
| Bhoopchand et al. (2023) | Cultural transmission through behavioral observation — Extension 2's inspiration |
| Frémaux & Gerstner (2016) | Three-factor rules — theoretical grounding for why temporal reward matters |
| Bansal et al. (2018) | Emergent complexity from competitive RL — empirical precedent |
| Baker et al. (2020) | Autocurricula and behavioral phase analysis — our analysis methodology |
| Schulman et al. (2017) | PPO — our inner-loop algorithm |