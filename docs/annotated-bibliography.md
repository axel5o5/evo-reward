# Annotated Bibliography
### Evolved Intrinsic Motivation, Neuromodulation, and Multi-Agent Coevolution

---

## Core Doya Papers

**Doya, K. (2002). Metalearning and neuromodulation. *Neural Networks*, 15(4–6), 495–506.**

The foundational theoretical paper for this entire research direction. Doya proposes that the brain's neuromodulatory systems (dopamine, serotonin, acetylcholine, norepinephrine) implement the meta-parameters of reinforcement learning: dopamine signals TD error (reward prediction), serotonin controls the time horizon of reward integration, acetylcholine gates learning rate and signal-to-noise ratio, and norepinephrine modulates the exploration-exploitation tradeoff. The key contribution is formal: each neuromodulator maps onto a specific algorithmic role in RL, grounding biologically plausible learning rules in computational terms. This paper is the theoretical anchor for the M(t) concept across both the course project (M(t) as learned intrinsic reward) and the full project (M(t) as evolved plasticity gate). If you cite one paper in this entire bibliography, it's this one.

---

**Kanagawa, Y., & Doya, K. (2024). Evolution of rewards for food and motor action by simulating birth and death. In *Proceedings of the 2024 Artificial Life Conference (ALIFE 2024)*. MIT Press.**

The direct precursor to the 2025 paper. Introduces the core simulation model: agents maintain energy levels and can die or reproduce based on energy. The genome encodes reward functions rather than behavioral policies directly — agents evolve *what to care about*, then learn behaviors via RL given those rewards. Demonstrates that food-seeking rewards and basic motor action rewards can emerge from birth-death selection pressure without being hand-designed. Establishes the distributed evolutionary simulation framework that the 2025 paper extends to prey-predator settings. The architecture — evolving reward functions, RL inner loop — is a cleaner precedent for your course project than FTW's PBT framing.

---

**Kanagawa, Y., & Doya, K. (2025). Evolution of fear and social rewards in prey-predator relationship. *arXiv:2507.09992*. (Under review.)**

The most directly relevant paper to your course project. Extends the 2024 framework to a two-species competitive setting: prey and predator RL agents co-evolve their reward functions, including visual rewards for observing conspecifics and opponents. Fear — operationalized as negative visual reward for predators — reliably emerged in prey under sufficient predatory pressure. Notably, the paper also found cases where prey evolved *positive* rewards for both predators and conspecifics, interpreted as a social grouping reward that improves collective defense. Stronger predator hunting capability promoted stronger fear evolution; larger prey groups reduced it. This paper is essentially a proof-of-concept for your research question — it demonstrates that competing populations can simultaneously evolve reward functions in response to each other, producing qualitatively distinct emergent behaviors under different environmental conditions. The main difference from your course project: their outer loop is evolutionary (birth-death selection), yours would be gradient-based (PPO with jointly trained M(t)).

---

## Intrinsic Motivation and Learned Reward

**Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-driven exploration by self-supervised prediction. In *Proceedings of the 34th International Conference on Machine Learning (ICML 2017)*.**

Introduces the Intrinsic Curiosity Module (ICM): an auxiliary network that generates intrinsic reward proportional to prediction error in a learned feature space. The agent is rewarded for transitioning to states it cannot yet predict well, driving exploration in sparse-reward environments without hand-designed curiosity signals. Validated on VizDoom and Mario. This paper is the canonical reference for learned intrinsic reward and is your primary fixed-baseline comparison point. Importantly, curiosity here is fixed architecture, fixed type — agents don't discover *what kind* of thing to be curious about. Your project's claim is that adaptive M(t) can discover better intrinsic signals than fixed curiosity.

---

**Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2019). Exploration by random network distillation. In *Proceedings of the 7th International Conference on Learning Representations (ICLR 2019)*.**

Random Network Distillation (RND) generates intrinsic reward by measuring prediction error between a fixed random target network and a trained predictor network. Simpler than ICM, avoids the noisy TV problem (a known failure mode where ICM agents prefer unpredictable but uninformative stimuli). Achieves strong results on Montezuma's Revenge. RND is the recommended fixed intrinsic reward baseline for your course project because it's cleaner than ICM for competitive settings — there's no risk that the baseline's curiosity signal is inadvertently adaptive in ways that confound your comparison.

---

**Du, Y., Han, L., Fang, M., Liu, J., Jiang, T., & Tao, D. (2019). LIIR: Learning individual intrinsic reward in multi-agent reinforcement learning. In *Advances in Neural Information Processing Systems (NeurIPS 2019)*.**

Extends learned intrinsic reward to cooperative multi-agent RL. Each agent learns its own individual intrinsic reward function via a bi-level optimization: inner loop optimizes policy given current intrinsic reward; outer loop updates the intrinsic reward network to improve collective performance. Shows that individualized intrinsic rewards, even in cooperative settings, outperform shared or fixed intrinsic baselines. This is the closest multi-agent precedent to your project — the key difference is that LIIR is cooperative (shared collective objective) while your setting is competitive (opposing objectives for predators and prey). The gap your project addresses: no existing work studies jointly learned intrinsic reward in competitive coevolution.

---

## Multi-Agent RL and Emergent Complexity

**Bansal, T., Pachocki, J., Sidor, S., Sutskever, I., & Mordatch, I. (2018). Emergent complexity via multi-agent competition. In *Proceedings of the 6th International Conference on Learning Representations (ICLR 2018)*.**

Demonstrates that purely competitive self-play in continuous 3D MuJoCo environments (sumo, wrestling, running) produces behavioral complexity far exceeding what the task reward specifies. Agents develop sophisticated grappling, footwork, and counter-strategy behaviors with no explicit programming. Key finding: the complexity of emergent behavior scales with the difficulty of the opponent — agents trained against adaptive opponents develop richer repertoires than those trained against fixed or random opponents. Primary empirical precedent for the claim that competitive pressure alone generates interesting emergent behavior in RL.

---

**Baker, B., Kanitscheider, I., Marber, T., Wu, Y., Powell, G., McGrew, B., & Mordatch, I. (2020). Emergent tool use from multi-agent autocurricula. In *Proceedings of the 8th International Conference on Learning Representations (ICLR 2020)*.**

The hide-and-seek paper. Six distinct phases of emergent strategy use appeared sequentially over training — individual play, basic tool use, ramp climbing, ramp blocking, box surfing, box surfing counter — all from a reward that only specified whether agents could see/be seen. Established the concept of *autocurriculum*: competitive pressure between populations automatically generates an increasingly difficult sequence of challenges without any curriculum design. Also introduced the behavioral phases analysis methodology — identifying qualitative strategy transitions rather than only measuring scalar performance — which is exactly the kind of emergent behavior analysis your project should aim to produce.

---

**Jaderberg, M., et al. (2019). Human-level performance in 3D multiplayer games with population-based reinforcement learning. *Science*, 364(6443), 859–865.**

FTW. The most important conceptual precedent for your course project. Agents learned Quake III Capture the Flag at human level by training with an internal reward function separate from the game score. The inner loop was PPO; the outer loop used Population-Based Training (PBT) to adapt internal reward parameters across a population of agents. Key finding: learned internal rewards produced qualitatively different and more effective behaviors than fixed game-score reward alone. PBT works by maintaining a population of agents, periodically copying hyperparameters and weights from better-performing agents to worse-performing ones with random perturbations — evolutionary in structure (selection + mutation) but operating on reward parameters rather than neural architectures. The reason this paper is cited carefully in an RL course context: the outer loop (PBT) is evolutionary computation, even if the inner loop is RL. Acknowledge this distinction; don't elide it.

---

## Neuroevolution and Differentiable Plasticity

**Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. *Evolutionary Computation*, 10(2), 99–127.**

Introduces NEAT. Evolves both network weights and topology through three key mechanisms: historical markings (tracking gene lineage to enable meaningful crossover between networks of different structure), speciation (protecting structural innovations by competing only within niches), and minimal starting structure (growing complexity incrementally). Relevant as the candidate outer loop for the full project and as a comparison baseline for the spider web project. The key distinction for your course: NEAT optimizes via selection across generations, not gradient descent — it's evolutionary computation, not RL proper. This is the distinction the professor flagged.

---

**Miconi, T., Clune, J., & Stanley, K. O. (2018). Differentiable plasticity: Training plastic neural networks with backpropagation. In *Proceedings of the 35th International Conference on Machine Learning (ICML 2018)*.**

Backpropamine. Each synapse has a fixed weight component and a plastic (Hebbian) component, and a modulatory signal M(t) gates how much the plastic component updates based on recent activity. Both the fixed weights and the plasticity rules are trained via gradient descent — not evolved. This is the closest computational implementation to the full project's architecture: M(t) as a per-synapse plasticity gate operating during a lifetime. The key difference from the full project: Backpropamine trains plasticity rules via gradient on a fixed task; the full project intends to *evolve* these rules via NEAT so they generalize to novel environments within a lifetime. Directly relevant to Phase 2 of the roadmap.

---

## Hybrid Evolutionary-RL

**Khadka, S., & Tumer, K. (2018). Evolution-guided policy gradient in reinforcement learning. In *Advances in Neural Information Processing Systems (NeurIPS 2018)*.**

ERL. Maintains a population of actors trained via an evolutionary algorithm alongside a single actor trained via TD3 (a policy gradient method). The RL actor periodically injects its learned policy into the evolutionary population; the evolutionary population provides diverse behavioral rollouts to the RL actor's replay buffer. Outperforms both pure RL and pure evolutionary methods on continuous control benchmarks. The most directly relevant hybrid framing for incorporating evolution into an RL project: RL is the primary learning mechanism, evolution is an auxiliary optimization layer that improves exploration diversity. The borderline case for "defensibly RL" — RL is clearly the core contribution, EA is auxiliary.

---

## RL Algorithms

**Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.**

PPO. Improves on TRPO by replacing the trust region constraint with a clipped surrogate objective, making it simpler to implement while retaining stability. Widely adopted as the default RL algorithm for continuous control and multi-agent settings. In your project architecture: PPO is the inner loop for both predator and prey policies, with the M(t) network trained jointly alongside the policy. Short paper, worth reading in full before implementation.

---

**Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. In *Advances in Neural Information Processing Systems (NeurIPS 2017)*.**

MADDPG. Introduces centralized training with decentralized execution: during training, each agent's critic has access to all agents' observations and actions, enabling stable gradient estimates despite non-stationarity; at test time, agents act only on local observations. Addresses the non-stationarity problem in competitive MARL directly. Relevant as a baseline algorithm and as the source of the centralized critic technique, which your project may need if joint M(t) training proves unstable under naive decentralized PPO.

---

*Last updated: March 2026*