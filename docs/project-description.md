**Evolved Reward Structures in Predator-Prey RL: How Agent Capacity
Shapes What Evolution Discovers**

*Axel --- Columbia University, Advanced Reinforcement Learning*

**Problem and Motivation**

Kanagawa & Doya (2024, 2025) showed that when RL agents in a
predator-prey world inherit reward function parameters through
birth-death dynamics, natural selection discovers interpretable reward
structures: prey evolve *fear* (negative reward for predator stimuli)
and *social affiliation* (positive reward for conspecific proximity).
Their agents use PPO as the inner loop with evolved linear reward
weights as the heritable genome. However, their reward is a fixed linear
function of instantaneous stimuli, and observations include only
positions of others. We ask: **what reward structures does evolution
discover when agents have richer representational capacity?**

**Research Design: Four Axes of Extension**

**Axis 1 --- Reward nonlinearity.** Replace the linear r(t) = w ·
stimulus(t) with a small MLP whose weights are the genome, enabling
context-dependent fear (e.g., stronger aversion when energy is low,
weaker when in a group). Does evolution exploit nonlinear capacity or
converge on the same linear structures?

**Axis 2 --- Social behavioral observation.** Expand the policy's
observation to include conspecific heading and velocity, not just
position. The reward genome is unchanged, but the policy can now
distinguish a fleeing neighbor from a stationary one. Do evolved reward
weights shift when the policy can exploit behavioral cues?

**Axis 3 --- Reward temporal depth.** Give the reward function a context
window: r(t) = MLP(obs(t), ..., obs(t−k)), enabling *anticipatory*
reward---responding to a predator approaching rather than merely
present. Doya (2002) predicts temporal reward should converge on
prediction-error-like structures; we test this directly.

**Axis 4 --- Policy temporal depth.** Replace the feedforward policy
with an LSTM. The interaction with Axis 3 is key: temporal reward
without memory provides a richer signal the agent cannot act on; memory
without temporal reward lets the agent use past experience but does not
reward temporal patterns.

**Experimental Plan and Metrics**

We replicate Kanagawa & Doya as our control, then test each axis
independently and in key combinations. The environment is a continuous
2D predator-prey world with birth-death evolutionary dynamics and PPO as
the within-lifetime learner. Key metrics: evolved reward weight
trajectories, survival/capture rates, behavioral diversity (strategy
clustering), reward function heatmaps (MLP output across state
variables), coordination metrics (velocity alignment, group cohesion),
and---for the temporal condition---whether reward functions develop
prediction-error-like structure (sensitivity to changes vs. levels).

**Novelty and Related Work**

The evolved-reward literature (Singh et al. 2009; Wang et al. 2019;
Kanagawa & Doya 2025) gives agents only linear reward over
instantaneous, position-only input. The social learning literature
(Ndousse et al. 2021; Bhoopchand et al. 2023) provides behavioral
observation but with fixed rewards. No prior work asks how the
*capacity* of the reward function and the *richness* of observation
jointly shape what evolution discovers---a question grounded in Doya's
(2002) neuromodulation framework.

**Plan Going Forward**

Weeks 1--2: Replicate baseline; validate fear and social reward
emergence. Weeks 3--5: Implement and run each axis. Weeks 6--7: Key
combinations (temporal reward × LSTM, temporal reward × social
observation). Weeks 8--9: Analysis, visualization, report. Codebase: JAX
for GPU-accelerated population evaluation and PPO. Seeking teammates
with skills in multi-agent RL or evolutionary computation.