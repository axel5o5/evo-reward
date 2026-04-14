# Experimental Plan: Evolved Reward Structures in Predator-Prey RL

*Working document — March 2026*

---

## Architecture modes

The codebase supports four architecture modes via two independent config flags. This flexibility is critical — we don't know in advance which simplifications preserve K&D's core dynamics, so the system must support switching between modes without code changes.

**Policy mode** (`policy_mode`):

- **`independent`** — each agent has its own policy network, trained via its own PPO updates against its own reward signal. Faithful to K&D. Cost: O(N) PPO updates per rollout cycle.
- **`shared`** — one policy network per species, conditioned on the agent's reward genome as additional input. All agents share the policy; evolution touches only the reward weights/MLP. Cost: O(1) PPO updates. Biggest computational speedup. Main sacrifice: agents with identical reward weights produce identical behavior (no idiosyncratic learning).

**Lifecycle mode** (`lifecycle_mode`):

- **`continuous`** — overlapping generations with continuous birth-death driven by energy accumulation, hazard functions, and birth probability. Faithful to K&D. No designer-imposed fitness function — reproductive success emerges from behavior. Richer ecological dynamics (age structure, overlapping generations).
- **`generational`** — discrete generations. Initialize population → run all agents K episodes → measure fitness (total energy) → select + mutate → next generation. Simpler to implement and parallelize (`jax.vmap` across uniform-length episodes). Sacrifice: requires an explicit fitness function; removes overlapping-generation effects.

**The four combinations and when to use each:**

| | Independent policy | Shared policy |
|---|---|---|
| **Continuous birth-death** | K&D faithful (Phase 1a) | Recommended default for extensions (Phase 1b+) |
| **Generational batching** | Unusual — skip unless needed | Fastest mode (fallback if compute-constrained) |

**Recommended default for extension experiments:** shared policy + continuous birth-death. This preserves the ecological dynamics that give the project its biological grounding (no designer-imposed fitness function, overlapping generations, energy-driven reproduction) while getting the major computational speedup from amortized PPO. Generational batching is available as a further speedup if needed but is the bigger conceptual departure from K&D.

**Population size:** ~80 per species (tunable), vs. K&D's ~130–150. Can scale up as a robustness check.

**Reward genome options (always both available in the codebase):**

- **Linear:** `r(t) = w · stimulus(t)` — 4–6 scalar weights, directly comparable to K&D.
- **MLP:** `r(t) = MLP_θ(stimulus(t))` — small network (~100–200 parameters), weights are the genome.

**Policy options (always both available):**

- **Feedforward MLP** — baseline.
- **LSTM** — for policy temporal depth axis.

---

## Architectural choices as a scientific contribution

The choice between shared/independent policy and continuous/generational lifecycle is not merely an engineering trade-off. Each combination embodies different assumptions about how evolution, learning, and ecological dynamics interact. Rather than treating these as simplifications to be validated and moved past, we frame them as a **controlled study of how evolutionary mechanism affects evolved reward structure** — a contribution in its own right.

### What shared policy changes and why it matters

In K&D's independent-policy setup, each agent trains its own policy network against its own reward signal. Two agents with different reward weights develop different behavioral strategies, and two agents with *identical* reward weights can still diverge through the randomness of their individual training (different experiences, different gradient updates). The policy is a product of both the genome (reward weights) and individual lifetime learning.

With a shared policy conditioned on reward weights, the single network must generalize across the entire population distribution of reward genomes. It learns the *average* behavioral strategy that works well for the current distribution, rather than specializing for any particular genome. Behavioral variation within a species comes exclusively from reward weight differences, not from idiosyncratic learning.

This connects to a key finding from Duarte, Scholtens & Weissing (2012): division of labor evolves differently depending on neural architecture. Recurrent/independent networks enable *experience-dependent specialization* (genetically similar agents develop different roles through learning), while feedforward/shared networks only support *genetically determined specialization* (behavioral variation must come from the genome). Shared policy forces all behavioral diversity to be genetically encoded, which is a cleaner experimental setup for studying evolved reward but may suppress emergent diversity that K&D's system produces.

There is also a subtle implicit knowledge transfer under shared policy. In K&D, newborns start with randomly initialized policy networks — they learn from scratch. Under shared policy, a newborn immediately benefits from a policy trained on the entire population's experience. This is analogous to a weak form of cultural transmission: the "knowledge" of how to behave given a reward function is shared across the population and persists across generations through the policy network, even though no individual agent's learned weights are inherited. This should accelerate early-life competence and reduce the selection pressure for reward functions that enable fast individual learning (weakening the Baldwin effect pathway).

LIIR (Du et al. 2019), already in our reference set, uses shared policies in cooperative MARL and notes that sharing "does not imply agents act the same" because each agent has its own partial observation. Our setting differs: agents within a species have similar observations but *different reward functions*. The heterogeneity comes from the genome rather than the observation, making sharing potentially more constraining. Christianos et al. (2021), "Scaling Multi-Agent Reinforcement Learning with Selective Parameter Sharing," found that sharing helps in homogeneous teams but hurts in heterogeneous settings — our agents are genetically heterogeneous, so this is a relevant concern.

### What generational batching changes and why it matters

K&D's continuous birth-death model has no explicit fitness function. Reproductive success is *emergent*: energy accumulation → birth probability b(e), energy depletion → death via hazard h(t,e). The environment is the fitness function. This is what the embodied evolution literature (Bredeche & Montanier 2012, mEDEA) calls "implicit fitness" — a core feature distinguishing ALife-style evolution from standard evolutionary algorithms.

Generational batching reintroduces a *designer-imposed* fitness function: run agents for K episodes, measure a scalar (total energy, survival time), rank, select. This changes several dynamics:

**Population dynamics.** K&D's Figure 6 shows Lotka-Volterra-like oscillations — prey and predator populations fluctuate with coupled dynamics. These oscillations are not cosmetic: they create alternating periods of intense and relaxed predation pressure, and the reward weights that evolve reflect that temporal structure. With fixed population sizes per generation, these oscillations disappear entirely.

**Selection pressure uniformity.** In continuous birth-death, an agent that gets lucky early can reproduce quickly and have offspring already competing while other agents are still in their first life. There's a rich age structure. In generational batching, everyone gets the same evaluation period — more "fair" but less ecologically realistic.

**The Baldwin effect.** In continuous evolution, an agent whose reward function enables faster policy learning survives longer and reproduces more *within its lifetime*. This creates direct selection for reward functions that make learning easy — a clear Baldwin effect pathway. In generational batching, all agents train for the same duration regardless of reward function quality. The Baldwin effect operates only through the final fitness ranking, not through differential survival during training. This is a weaker channel.

**Fitness function sensitivity.** With generational batching, you must choose an explicit fitness metric. "Total energy accumulated" approximates K&D's implicit fitness but isn't identical. You could test alternative fitness functions (survival time, number of food items eaten, time spent near predators inverted) and ask whether evolved reward structures change. This would be a result about the sensitivity of reward evolution to the meta-objective — a question K&D's framework cannot ask because there is no explicit meta-objective to vary.

The closest bridge concept is PBT (Jaderberg et al. 2017), which operates as quasi-continuous evolution applied to deep RL — maintaining a population of models with continuous exploitation (copy weights from top performers) and exploration (perturb hyperparameters). PBT is intermediate between generational and continuous: it has no discrete "generations" but does have explicit fitness-based selection events. K&D's continuous birth-death is more ecologically grounded; generational batching is more like a standard genetic algorithm.

### Hypotheses: predicted effects of each architectural choice

**H1 (Reward convergence speed).** Shared policy produces faster reward weight convergence than independent policy, because the amortized policy provides a more stable (less noisy) mapping from reward genome to fitness. Generational batching produces faster convergence than continuous birth-death, because the selection signal is cleaner (no noise from variable lifetimes and overlapping generations).

**H2 (Intra-species behavioral diversity).** Independent policy produces greater behavioral diversity within a species than shared policy, even among agents with similar reward weights, because individual learning trajectories diverge. This should be measurable via behavioral clustering metrics (strategy entropy, number of distinct behavioral modes in the population at any given time).

**H3 (Reward weight branching).** K&D's Figure 8 shows branching of reward weights within a population — some lineages evolve strong fear, others evolve strong social affiliation. Under shared policy, we predict *less* branching, because the shared policy acts as a smoothing function over the reward-fitness landscape. Extreme reward genomes are less viable when the policy is optimized for the population mean rather than the individual.

**H4 (Population dynamics and ecological structure).** Continuous birth-death produces Lotka-Volterra-like oscillations in population sizes. Generational batching produces flat (fixed) population sizes. The oscillatory dynamics create temporally varying selection pressure that may drive different reward equilibria — specifically, fear may need to be *stronger* under continuous dynamics because predation pressure fluctuates and agents must survive the peaks.

**H5 (Qualitative robustness of core result).** Despite the above quantitative differences, we predict that fear (negative w_pred) and social affiliation (positive w_con) emerge under *all four* architecture combinations. These are robust consequences of predator-prey competitive dynamics, not artifacts of K&D's specific lifecycle model. If this hypothesis holds, it's a meaningful robustness result. If it fails for a specific combination, identifying *which* architectural feature is necessary for the phenomenon is itself a contribution.

**H6 (Baldwin effect strength).** Under independent policy + continuous birth-death (K&D faithful), reward functions that enable fast learning should be selectively favored because fast learners survive longer and reproduce more. Under shared policy, this pressure is weakened because all agents benefit from the shared policy regardless of their individual reward function's "learnability." Measurable via: correlation between reward genome complexity and agent lifetime (should be positive under independent/continuous, weaker under shared).

**H7 (Implicit cultural transmission via shared policy).** Shared policy acts as a form of population-level knowledge transfer — newborns immediately inherit behavioral competence from the shared policy trained on the full population. This should reduce the "infant mortality" rate (fraction of newborns that die before learning useful behavior) relative to independent policy, where each newborn starts from a randomly initialized network. Measurable via: mean age at first reproduction, survival rate in the first N timesteps of life.

### Measurements specific to architectural comparison

Beyond the standard metrics (reward weight trajectories, KDE plots, capacity utilization), Phase 1 should specifically track:

- **Reward weight distribution width** (standard deviation of each reward weight across the population) over evolutionary time — narrower under shared policy (H3).
- **Behavioral diversity index** — Shannon entropy over a discretized behavioral space (speed × heading × nearest-neighbor-distance bins) — lower under shared policy (H2).
- **Population size time series** — oscillatory under continuous, flat under generational (H4).
- **Convergence time** — number of generations until reward weights stabilize (within some threshold of final values) — faster under shared and/or generational (H1).
- **Newborn survival rate** — fraction of agents surviving past a fixed age threshold — higher under shared policy (H7).
- **Reward-learnability correlation** — Spearman correlation between reward genome fitness and speed of policy learning (measured by learning curve slope in early training steps) — positive under independent/continuous, attenuated under shared (H6).

### Additional references to pursue

- Christianos, Schäfer & Albrecht (2021). "Scaling Multi-Agent Reinforcement Learning with Selective Parameter Sharing." — When does sharing help vs. hurt in heterogeneous MARL?
- Hornby (2006). "ALPS: the Age-Layered Population Structure for Reducing Premature Convergence." — Age-layered selection as a middle ground between generational and continuous.
- Lehman et al. (2020). "The Surprising Creativity of Digital Evolution," *Nature*. — How different evolutionary frameworks produce qualitatively different outcomes.
- Sasaki & Tokoro (1999). "Comparison of Lamarckian and Darwinian evolution," *Artificial Life* 5(3). — Partial Lamarckian inheritance rates; shared policy as implicit zero-Lamarckian with population-level knowledge transfer.
- Terry et al. (2021). PettingZoo documentation on shared vs. independent policies in cooperative-competitive environments.
- Bredeche & Montanier (2012). mEDEA — implicit fitness from embodied evolution, the closest analog to K&D's continuous birth-death in the robotics literature.

---

## Software architecture principles

Build once, configure per-experiment. Every run is defined by a config dict, not by different code paths.

```
config = {
    # Architecture mode
    "policy_mode": "independent" | "shared",
    "lifecycle_mode": "continuous" | "generational",

    # Reward genome
    "reward_type": "linear" | "mlp",
    "reward_context_window": 1 | 10,       # 1 = instantaneous, >1 = temporal

    # Observation
    "social_obs": "position_only" | "position_heading_velocity",

    # Policy
    "policy_type": "mlp" | "lstm",

    # Evolution
    "coevolution_mode": "concurrent" | "alternating",
    "population_size": 80,
    "num_generations": 300,
    "mutation_scale": 0.4,

    # Environment
    "world_size": 960,
    "food_regen_rate": 0.5,
    ...
}
```

### Codebase structure

```
evo-reward/
├── configs/                 # One YAML per experiment condition
│   ├── baseline_faithful.yaml    # K&D faithful: independent + continuous
│   ├── baseline_simplified.yaml  # shared + continuous (default for extensions)
│   ├── axis1_mlp.yaml
│   ├── axis2_social.yaml
│   └── ...
├── src/
│   ├── environment.py       # JAX 2D world, physics, food, collisions
│   ├── agents.py            # Observation construction, reward computation
│   ├── policy.py            # Policy networks (MLP + LSTM, shared + independent)
│   ├── reward.py            # Reward genome (linear + MLP variants)
│   ├── evolution.py         # Selection, mutation (supports both lifecycle modes)
│   ├── lifecycle.py         # Continuous birth-death vs. generational batching
│   ├── ppo.py               # PPO inner loop (handles both FF and LSTM)
│   └── metrics.py           # All measurement functions (see below)
├── analysis/
│   ├── dashboards.py        # Automated per-run visualization
│   ├── comparison.py        # Cross-condition comparison plots
│   └── capacity_util.py     # Capacity utilization metrics
├── scripts/
│   ├── run_experiment.py    # Single run from config
│   ├── run_sweep.py         # Batch launcher (multiple seeds/conditions)
│   └── analyze_results.py   # Post-hoc analysis from saved checkpoints
└── results/                 # Auto-organized: results/{condition}/{seed}/
    └── baseline_faithful/
        ├── seed_0/
        │   ├── checkpoints/     # Per-generation snapshots
        │   ├── metrics.npz      # Time-series of all scalar metrics
        │   └── config.yaml      # Exact config used (reproducibility)
        └── seed_1/
```

### Measurement infrastructure (build in Phase 0, use everywhere)

Every generation, log automatically:

**Ecological metrics** — population fitness distribution, survival rates, mean energy at end of episode, capture/escape rates.

**Reward genome metrics** — mean/std of evolved reward weights (or MLP params), reward weight trajectories over generations, KDE of population reward weight distributions (reproducing K&D's Figure 7/12 style plots).

**Behavioral metrics** — mean speed, group cohesion (mean distance to nearest conspecific), velocity alignment (cosine similarity of heading vectors among nearby conspecifics), spatial dispersion.

**Capacity utilization metrics** (the answer to "what is your metric of success"):

- *Reward nonlinearity utilization:* fit best-linear-approximation to MLP reward output across sampled states; report residual norm. High residual = evolution discovered useful nonlinearity.
- *Social observation utilization:* mutual information between conspecific heading/velocity inputs and agent actions. Zero MI = agent ignores the social channel.
- *Temporal reward utilization:* autocorrelation of reward signal across the context window; compare to instantaneous-only control. Also: sensitivity ratio (∂r/∂obs(t) vs ∂r/∂obs(t-k)) — does the reward function weight recent vs. old observations differently?
- *LSTM memory utilization:* hidden-state information content (entropy of h_t); ablation test — zero h_t mid-episode, measure performance drop.

**Save checkpoints** every N generations (e.g., every 25) so you can reconstruct the evolutionary trajectory, run dominance tournaments between timepoints, and generate visualizations post-hoc without re-running.

### Visualization toolkit (build early, use constantly)

Standardized plot functions that work on any run's `metrics.npz`:

1. **Reward weight trajectory plot** — evolved reward weights over generations (K&D Figure 7 style). One line per seed, shaded confidence band.
2. **Reward weight KDE** — population distribution of reward weights at a given generation (K&D Figure 12 style).
3. **Population dynamics** — population size / mean fitness over generations.
4. **Behavioral phase diagram** — 2D scatter of behavioral metrics (e.g., cohesion vs. speed) colored by generation, showing evolutionary trajectory through behavioral space.
5. **Capacity utilization bar chart** — per-axis utilization score, grouped by condition. This is the key figure for the professor.
6. **Reward heatmap** (MLP conditions only) — MLP output plotted as heatmap over 2 state variables (e.g., predator distance × energy).

---

## Phase 0: Infrastructure (Week 1)

**Goal:** Working environment + PPO + evolution loop supporting all architecture modes. No experiments yet — just validated plumbing.

**Build (in this order):**

1. JAX 2D environment: continuous world, food spawning, proximity sensors, collision detection. Borrow heavily from K&D's open-source code (github.com/oist/emevo). The environment is the same regardless of architecture mode.
2. Continuous birth-death lifecycle: energy accumulation, hazard/birth functions, asexual reproduction with Student's t(df=2, scale=0.4) mutation on reward genomes. This is the K&D-faithful lifecycle and should be built first.
3. Independent per-agent policy networks + PPO. Each agent gets its own MLP policy, trained against its own reward signal. This is the K&D-faithful policy mode.
4. Shared policy mode: single policy network per species conditioned on reward genome. This is a separate code path behind the `policy_mode` config flag — the environment and lifecycle code are unchanged.
5. Generational batching lifecycle: discrete generation loop (evaluate → select → mutate → repeat). Behind the `lifecycle_mode` config flag.
6. Metrics infrastructure + checkpoint saving. These must work identically across all modes.
7. One-command run script: `python run_experiment.py --config configs/baseline_faithful.yaml --seed 0`.

**Validation gate:** Run a minimal faithful-mode test (20 agents, 50 generations, independent policy, continuous birth-death) and confirm:

- Agents learn to move toward food (PPO is working).
- Reward weights drift from initialization (evolution is working).
- Birth-death dynamics produce population fluctuations.
- Metrics are being logged and checkpoints are being saved.
- One run completes in a reasonable time on your available GPU.

---

## Phase 1: Baseline replication + simplification validation (Weeks 2–3)

### Phase 1a: Faithful K&D replication

**Goal:** Reproduce K&D's core result using their architecture: independent policies, continuous birth-death. This is your ground truth.

**Run:** `policy_mode: independent`, `lifecycle_mode: continuous`, linear reward, position-only observation, feedforward policy. 1–2 seeds, 300 generations.

**Success criteria:**

- Prey evolve negative w_pred (fear) and positive w_con (social affiliation). This is the headline result from K&D.
- Predators evolve positive w_prey (attraction to prey). Predator reward structure should be simpler than prey.
- Population dynamics show oscillatory predator-prey cycles (Lotka-Volterra-like).
- Reward weight trajectories converge within ~100–200 generations.

**If baseline fails:** Debug cycle. Most likely issues: PPO hyperparameters (learning rate, number of epochs, rollout length), mutation scale too high/low, population too small for evolutionary signal. Refer to K&D's Table 4 for PPO params and Table 3 for evolutionary params. Compare directly against their open-source code output.

**Deliverable:** Reward weight trajectory plots + KDE plots that visually match K&D's qualitative pattern. This is your proof that the environment and evolution code are correct before any simplifications are introduced.

**Compute cost:** ~10–24 hours (1–2 runs at K&D's ~10–12 hours per run).

### Phase 1b: Simplification validation and architectural comparison

**Goal:** Test how shared policy affects K&D's core dynamics. This is not just validation — it's a controlled experiment testing H1–H3 and H5–H7 (see "Architectural choices as a scientific contribution" above).

**Run:** Same as Phase 1a but with `policy_mode: shared`. 1–2 seeds.

**Compare to Phase 1a (qualitative — testing H5):**

- Does fear (negative w_pred) still emerge? Does social affiliation (positive w_con) still emerge?
- Are the reward weight trajectories qualitatively similar?
- Are population dynamics qualitatively similar?
- How much faster is the run? (Document the speedup factor.)

**Compare to Phase 1a (quantitative — testing H1–H3, H6–H7):**

- Reward weight distribution width over time: narrower under shared? (H3)
- Convergence speed: faster under shared? (H1)
- Behavioral diversity index: lower under shared? (H2)
- Newborn survival rate: higher under shared? (H7)
- Reward-learnability correlation: weaker under shared? (H6)

**If the qualitative result holds (H5 confirmed):** Shared policy + continuous birth-death becomes the default mode for all extension experiments (Phase 2+). Document the quantitative differences — these are results, not just validation artifacts. The finding "fear and social affiliation are robust to policy sharing; behavioral diversity is reduced but reward structure is preserved" is a publishable-quality observation about the role of individual learning in evolved reward systems.

**If the qualitative result breaks (H5 falsified for shared policy):** This is also an interesting finding — it means individual policy learning is *necessary* for the emergence of evolved reward structure, not just a source of additional diversity. Options in priority order:
1. Check whether the policy conditioning architecture is the issue (maybe the policy needs more capacity to distinguish reward genomes — try a larger hidden layer or a hypernetwork-style conditioning).
2. Fall back to independent policies + continuous birth-death (K&D faithful) for all experiments. This is slower but known to work.
3. Try shared policy + generational batching as a last resort — generational batching changes the dynamics more, but the combination might be compensating in unexpected ways.

**Optional but recommended (if compute permits):** Run all four architecture combinations (1 seed each) to get the full 2×2 comparison. This directly tests H4 (population dynamics under generational) and provides the most complete picture of how architectural choices interact. Total additional cost: 2 more runs (~14–26 hours).

| Run | Policy | Lifecycle | Tests |
|-----|--------|-----------|-------|
| 1a  | Independent | Continuous | Ground truth (K&D faithful) |
| 1b  | Shared | Continuous | H1–H3, H5–H7 |
| 1c (optional) | Independent | Generational | H4, H5 (lifecycle effect isolated) |
| 1d (optional) | Shared | Generational | H4, H5 (both simplifications combined) |

**Deliverable:** A table or figure comparing the four (or two) architecture modes across the hypothesis-specific metrics. This becomes a standalone result in the Methods section of the final report, framed as: "We systematically evaluated how evolutionary mechanism affects evolved reward structure before proceeding to extension experiments."

---

## Phase 2: One-axis-at-a-time ablations (Weeks 3–5)

**Goal:** Test each extension axis independently against the baseline. This is the one-hot design — 4 conditions, each changing exactly one thing.

**Runs (1 seed each for exploration, 4 runs total):**

| Condition | Reward | Observation | Policy | What's different |
|-----------|--------|-------------|--------|-----------------|
| Baseline  | Linear | Position    | FF MLP | (control — already done in Phase 1b) |
| +MLP reward | MLP  | Position    | FF MLP | Axis 1 only     |
| +Social obs | Linear | Pos+Head+Vel | FF MLP | Axis 2 only   |
| +Temporal reward | MLP (context window) | Position | FF MLP | Axis 3 only |
| +LSTM policy | Linear | Position | LSTM | Axis 4 only     |

Note: Axis 3 (temporal reward) necessarily uses an MLP to process the context window, so it implicitly includes Axis 1's nonlinearity. This is fine — it's a deliberate design feature noted in the full extension design doc. If you want to disentangle, you could add a "temporal linear" condition where the context window is processed by a linear function of the flattened vector, but that's lower priority.

**At 1 seed, you're looking for qualitative signal only:**

1. Does the K&D result (fear + social affiliation) still emerge? It should — these extensions add capacity, they don't remove anything.
2. Does the capacity utilization metric for the manipulated axis show nonzero signal? This is the key question — even from 1 seed, a clearly nonzero capacity utilization score (or a clearly zero one) is informative.
3. Do ecological or behavioral metrics visibly change? (E.g., +Social obs should show different group cohesion patterns.)

**After the 1-seed sweep, invest depth in what's interesting:** Add 2–4 more seeds to conditions where the single run showed a clear positive signal or a surprising result. Skip additional seeds for conditions that look flat. This is how you avoid wasting compute on uninteresting cells.

**Priority order if time is tight:** Run Axis 2 (social obs) and Axis 3 (temporal reward) first. These are the most scientifically interesting and have the strongest theoretical motivation (Doya's prediction-error connection for Axis 3, social learning literature gap for Axis 2). Axis 1 (MLP alone) is somewhat subsumed by Axis 3. Axis 4 (LSTM) is important but interacts most meaningfully with the others, so it's more valuable in combination.

---

## Phase 3: Key combinations (Weeks 5–7)

**Goal:** Test the interaction effects. The 2×2 from the design doc is the core, plus a few targeted combinations.

**Seed strategy continues: 1 seed per new condition for exploration, then depth where warranted.**

**Priority 1 — the core 2×2 (temporal × social):**

| | Position only (X) | Social obs (Y) |
|---|---|---|
| **Instantaneous reward (A)** | AX: Baseline (already done) | AY: Phase 2 social (already done) |
| **Temporal reward (B)** | BX: Phase 2 temporal (already done) | BY: **New — full system** |

You only need to run BY (temporal reward + social observation) since the other three cells are already done. 1 seed initially.

**Priority 2 — LSTM interactions (pick 2):**

- Temporal reward + LSTM (Axes 3+4): The theoretically motivated pairing. Temporal reward provides anticipatory signal; LSTM provides memory to act on it. Hypothesis: this combination should show the highest capacity utilization for both axes.
- Social obs + LSTM (Axes 2+4): LSTM can track conspecific behavior over time even with instantaneous reward. Does this produce different social strategies than feedforward + social obs?

1 seed each.

**Priority 3 — Iterative best response comparison (if time permits):**

Pick 2 conditions from above (suggest: baseline + BY full system). Re-run each under alternating coevolution: freeze prey reward weights for 10 generations while predator evolves, then switch. 1 seed each, 2 runs.

Compare: convergence speed, final reward weight distributions, behavioral diversity, population dynamics stability. Frame as concurrent vs. punctuated coevolution.

**End-of-Phase 3 depth investment:** By now you've explored ~8–10 conditions at 1 seed each. Identify the 3–5 most interesting and add 3–5 more seeds to each for the final report. This is where you shift from exploration to confirmation.

---

## Phase 4: Full factorial + report (Weeks 7–9)

**Goal:** Fill in remaining cells of the 2×2×2×2, collect robust seed counts on key conditions, write the report.

By this point you have: baseline, 4 one-hot conditions, 3+ combinations = ~8–10 conditions explored at 1 seed each. Two things happen in parallel during this phase:

**Depth:** Run additional seeds (targeting 5 total) on the 3–5 most interesting conditions identified in Phases 2–3. These are the conditions you'll make quantitative claims about in the report.

**Breadth (if compute permits):** Fill in remaining factorial cells at 1 seed each, prioritizing any cells where Phase 2–3 results suggest interesting interactions. The 4-way combination (MLP reward + social obs + temporal context + LSTM) is the "kitchen sink" — run it even if you skip some 3-way cells, because it tells you whether the full capacity stack produces qualitatively different emergent behavior than any subset.

**Report structure:**

1. Introduction: K&D baseline, the capacity question, Doya framing.
2. Methods: environment, architecture modes (faithful vs. simplified, with validation), four axes, metrics.
3. Results by phase: baseline replication → simplification validation → one-axis ablations → combinations. Lead with capacity utilization chart.
4. Iterative best response comparison (if run).
5. Discussion: what evolution discovers vs. what it ignores; biological interpretation; limitations and future work (relaxing simplifications, larger populations).

---

## Phase 5: Deepening the architectural comparison + relaxing simplifications (stretch / post-semester)

### Extending the Phase 1 architectural study

If the Phase 1 comparison produced interesting results (e.g., H5 confirmed but H3 showed significant diversity reduction), invest additional seeds in the architectural comparison and consider extending it:

**Full 2×2 with depth:** If Phase 1 only ran 2 of the 4 architecture combinations, complete the grid with 3–5 seeds per cell. Run the most interesting extension condition (from Phase 2–3) under all four modes. This directly tests whether the architectural effects interact with the capacity extensions — e.g., does shared policy suppress the benefit of social observation (because behavioral diversity needed for social information use is reduced)?

**Alternative fitness functions under generational batching:** Run the generational mode with different explicit fitness metrics (total energy, survival time, mean energy, number of food items) and compare evolved reward structures. This tests whether reward evolution is sensitive to the meta-objective — a question that continuous birth-death cannot ask because there is no explicit meta-objective.

### Relaxing simplifications on extension experiments

**Relaxation 1 — Independent policies on key extensions:** Switch `policy_mode: independent` on the 2 most interesting extension conditions. Compare: does individual policy learning change the evolved reward structures? Specifically, does it increase capacity utilization (because individual policies can specialize for extreme reward genomes that the shared policy couldn't serve)?

**Relaxation 2 — Generational batching comparison:** Run key conditions under generational batching and compare population dynamics, convergence speed, and reward equilibria to continuous birth-death.

**Relaxation 3 — Larger population:** Scale to 150+ agents per species. Re-run conditions where Phase 2–3 showed marginal effects — maybe the signal was there but genetic drift in the small population washed it out.

**Relaxation 4 — Iterative best response as full axis:** If the Phase 3 IBR comparison showed interesting differences, promote it to a full axis and run more cells.

Each relaxation is a focused comparison (2–3 conditions, 3 seeds each), not a full re-run of the factorial.

---

## Seed strategy: exploration first, depth second

Different stages of the project need different levels of statistical confidence. The key insight: scanning the experimental space at 1 seed each is far more valuable early on than deeply confirming a single condition.

**1 seed (exploration):** Is there a qualitative signal at all? Does fear emerge? Is capacity utilization nonzero? One run tells you whether a condition is worth investigating further. This is the right depth for Phase 2 (one-axis ablations) and Phase 3 (combinations) initial sweeps.

**3–5 seeds (proof of concept):** The result showed up in 1 seed; now confirm it's not a lucky initialization. Three seeds showing the same qualitative pattern is sufficient for a course report. Five seeds with confidence bands is strong.

**5–10 seeds (robust results for final report):** For conditions you're making quantitative claims about — this is where you invest after you've identified the interesting cells. K&D used 5–6 seeds for most conditions and 10 for their baseline.

**100+ Monte Carlo:** Not needed for this kind of simulation. Each evolutionary run is already deeply stochastic (random initialization, random mutation, random encounters). 5–10 seeds is standard in this literature.

**The workflow:** Sweep at 1 seed → identify 3–5 interesting conditions → invest 5+ seeds in those → write the report around the conditions with depth. This avoids the trap of running 5 seeds on a condition that turns out to be uninteresting.

---

## Compute budget estimate

**Faithful mode (independent policy + continuous birth-death):** ~10–12 hours per run on A100 (per K&D).

**Simplified mode (shared policy + continuous birth-death):** ~2–4 hours per run (estimated 3–5× speedup from shared policy).

**Fastest mode (shared policy + generational batching):** ~1–2 hours per run (estimated 5–10× speedup total).

| Phase | Mode | Conditions | Seeds each | Runs | Estimated hours |
|-------|------|-----------|------------|------|-----------------|
| 0: Infrastructure | — | debug only | — | — | ~4 (dev time) |
| 1a: Faithful replication | Faithful | 1 | 1–2 | 1–2 | ~12–24 |
| 1b: Simplification validation | Simplified | 1 | 1–2 | 1–2 | ~4–8 |
| 2: One-axis sweep | Simplified | 4 | 1 | 4 | ~8–16 |
| 2 (depth follow-up) | Simplified | 2–3 (interesting ones) | +3–4 | 6–12 | ~12–48 |
| 3: Combinations sweep | Simplified | 3–5 | 1 | 3–5 | ~6–20 |
| 3 (depth follow-up) | Simplified | 2–3 (interesting ones) | +3–4 | 6–12 | ~12–48 |
| 4: Remaining factorial + report | Simplified | up to 8 | 1 | up to 8 | ~16–32 |
| IBR comparison | Simplified | 2 | 1–3 | 2–6 | ~4–24 |
| **Total** | | | | **~31–51** | **~78–224** |

The wide range reflects the exploration-first strategy: compute is front-loaded on exploration (cheap, 1 seed) and selectively invested in depth (expensive, 5+ seeds) only where the signal warrants it. The critical path (Phase 0 through Phase 2 initial sweep) requires ~28–52 hours of compute — feasible within the first 3–4 weeks on a single GPU, or faster with Columbia's cluster.

---

## Decision points and off-ramps

**After Phase 1a:** If faithful baseline fails to reproduce fear/social affiliation, stop and debug before proceeding. This is a code bug, not a design issue — K&D's result is published and their code is open-source. Do not move to Phase 1b until 1a works.

**After Phase 1b:** Evaluate against the architectural hypotheses (H1–H7). If H5 is falsified (shared policy breaks the core result), fall back to independent policies for all extension experiments — but write up the H5 falsification as a finding. If H5 is confirmed, proceed with shared policy as default and document the quantitative effects (H1–H3, H6–H7) as results. Either way, Phase 1 produces a contribution.

**After Phase 2 (1-seed sweep):** If no axis shows any qualitative signal, that's potentially a result — it means K&D's setup is already sufficient and evolution doesn't exploit expanded capacity. But before concluding this, verify with: (a) longer evolutionary runs (maybe 300 generations isn't enough for the richer genomes), (b) larger populations (maybe there's too much genetic drift), (c) independent policies (maybe the shared policy is masking the effect — this connects directly to H3, where shared policy's smoothing of the fitness landscape might suppress exploration of novel reward structures). Only declare a negative result after ruling these out.

**After Phase 3 (1-seed sweep):** You've explored ~8–10 conditions. Identify the 3–5 most interesting for depth investment. You have enough for a strong course report regardless of what happens in Phase 4. The architectural comparison from Phase 1 + the capacity extension results from Phase 2–3 are two distinct contributions.

**Throughout:** If any single condition produces a surprising or striking result (e.g., temporal reward develops clear prediction-error structure, or social observation causes a dramatic behavioral phase transition), it may be worth investing extra seeds in that condition at the expense of breadth. Depth on a surprising finding beats breadth across unsurprising ones.