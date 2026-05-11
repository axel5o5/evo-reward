"""
reward.py
---------
Reward genome initialization and the K&D linear reward equation.

Genome order: [w_eat, w_act, w_prey, w_pred] -- canonical everywhere.

Reward equation:
  r = w_eat * n_eaten
    + 0.01 * w_act * (||f|| / F)
    + 0.1  * w_prey * max_s_prey
    + 0.1  * w_pred * max_s_pred

The 0.01 and 0.1 are fixed coefficients, not genome parameters.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


def init_genome(rng_key, config: dict) -> jnp.ndarray:
    """
    Initialize reward genome: shape (4,), float32.
    Each weight ~ N(0, config["reward_weights_init_std"]).
    Order: [w_eat, w_act, w_prey, w_pred].
    """
    std = config["reward_weights_init_std"]
    return jax.random.normal(rng_key, shape=(4,)) * std


def compute_linear_reward(
    genome: jnp.ndarray,
    n_eaten: int,
    motor_norm: float,
    max_s_prey: float,
    max_s_pred: float,
) -> float:
    """
    Pure function implementing K&D reward equation.

    r = genome[0]*n_eaten + 0.01*genome[1]*motor_norm
      + 0.1*genome[2]*max_s_prey + 0.1*genome[3]*max_s_pred
    """
    r = (genome[0] * n_eaten
         + 0.01 * genome[1] * motor_norm
         + 0.1 * genome[2] * max_s_prey
         + 0.1 * genome[3] * max_s_pred)
    return r


# ─── MLP reward genome (Axis 1) ──────────────────────────────────────────────

class RewardMLP(nn.Module):
    """Small MLP reward network: 4 → hidden → hidden → 1.

    The genome (Flax param PyTree) encodes this network's weights.
    Architecture: input(4) → Dense(h, tanh) → Dense(h, tanh) → Dense(1, linear).
    With h=8: 121 total parameters (40 + 72 + 9).
    """
    hidden_size: int = 8

    @nn.compact
    def __call__(self, stimuli):
        x = nn.Dense(self.hidden_size)(stimuli)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)


def init_mlp_genome(rng_key, config):
    """Initialize MLP reward genome as a Flax parameter PyTree.

    Uses Flax default init (lecun_normal kernel, zeros bias) which is
    appropriate for tanh networks.
    """
    hidden = config["mlp_hidden_size"]
    net = RewardMLP(hidden_size=hidden)
    params = net.init(rng_key, jnp.zeros(4))
    return params


def compute_mlp_reward(genome, stimuli):
    """Forward pass through the MLP reward network.

    Args:
        genome: Flax parameter PyTree from init_mlp_genome.
        stimuli: shape (4,) — [n_eaten, motor_norm, max_s_prey, max_s_pred].
                 Raw values, no fixed coefficients applied.
    Returns:
        Scalar float32 reward.
    """
    hidden_size = genome['params']['Dense_0']['kernel'].shape[1]
    net = RewardMLP(hidden_size=hidden_size)
    return net.apply(genome, stimuli)


# ─── Residual reward genome (Axis 1 v4 — converged design) ─────────────────
#
# Combines the K&D linear baseline with a small MLP perturbation. The MLP is
# zero-initialized so at t=0 the network output is exactly the K&D linear
# reward. Evolution mutates both parts; the MLP can grow to express
# nonlinear reward structure if it improves fitness, or stay near zero if
# linear is sufficient.
#
# Why this design (vs the original axis-1 MLP that replaces the linear
# reward entirely): random-init 121-param MLPs produce essentially random
# rewards initially → predator behavior is random → predators starve before
# anything useful evolves. With residual init at zero, the system starts in
# the proven-stable K&D-faithful regime and can only deviate through
# mutation pressure that improves fitness.
#
# Architecture: input(4) → Dense(h, tanh) → Dense(1, linear), single hidden
# layer. With h=4 (default): 4*4+4 + 4*1+1 = 25 params. Compare to original
# RewardMLP (h=8, two hidden) at 121 params.
#
# See findings.md §15.

class ResidualRewardMLP(nn.Module):
    """Single-hidden-layer MLP for the residual reward perturbation.

    input(4) → Dense(h, tanh) → Dense(1, linear). Zero-initialized so the
    network outputs 0 at t=0 (linear baseline runs unchanged).
    """
    hidden_size: int = 4

    @nn.compact
    def __call__(self, stimuli):
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.zeros)(stimuli)
        x = nn.tanh(x)
        x = nn.Dense(1, kernel_init=nn.initializers.zeros)(x)
        return jnp.squeeze(x, axis=-1)


def init_residual_genome(rng_key, config):
    """Initialize residual MLP genome (zero-init).

    Returns a Flax PyTree with all weights and biases at zero. Stimuli flow
    through the network as 0 → tanh(0)=0 → 0, so the residual contributes
    zero to the reward at birth. Evolution adds nonlinear structure via
    mutation if it helps.
    """
    hidden = config.get("residual_hidden_size", 4)
    net = ResidualRewardMLP(hidden_size=hidden)
    params = net.init(rng_key, jnp.zeros(4))
    return params


def compute_residual_reward(linear_genome, residual_genome, stimuli):
    """Combined linear + residual MLP reward.

    Linear part uses the K&D fixed coefficients (1.0, 0.01, 0.1, 0.1).
    Residual part adds the MLP output on top — zero at birth, evolves freely.

    Args:
        linear_genome: shape (4,) float32 — [w_eat, w_act, w_prey, w_pred].
        residual_genome: Flax PyTree from init_residual_genome.
        stimuli: shape (4,) float32 — [n_eaten, motor_norm, max_s_prey, max_s_pred].

    Returns:
        Scalar float32 reward.
    """
    coefs = jnp.array([1.0, 0.01, 0.1, 0.1])
    linear_part = jnp.sum(linear_genome * stimuli * coefs)
    hidden_size = residual_genome['params']['Dense_0']['kernel'].shape[1]
    net = ResidualRewardMLP(hidden_size=hidden_size)
    residual_part = net.apply(residual_genome, stimuli)
    return linear_part + residual_part


# ─── Polynomial residual reward genome (Axis 1 v11) ─────────────────────────
#
# Replaces the small MLP residual ("linear_plus_mlp_residual") with explicit
# polynomial features: K&D linear + quadratic + pairwise interactions over
# the 4 stimuli. 10 evolvable params per agent (4 quadratic + 6 interaction)
# vs 25 for the MLP residual. Interpretable (each weight has a meaning) and
# better evolutionary search structure than a black-box MLP — every direction
# in genome space directly perturbs a meaningful reward feature.
#
# Layout (poly_genome shape (10,)):
#   indices 0..3 → w_sq_i for x_i^2, i in [0, 1, 2, 3]
#   indices 4..9 → w_xy_ij for x_i * x_j, pairs in canonical order
#                  (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
#
# Reward = K&D linear (with fixed coefs [1.0, 0.01, 0.1, 0.1]) + sum(w_sq * x^2)
#        + sum(w_xy * x_i * x_j over the 6 ordered pairs).
#
# Zero-init at birth → residual = 0 → starts at exact K&D linear baseline,
# evolution adds nonlinear structure if useful.

# Canonical pair indices for the 6 pairwise interaction terms. Kept as a
# module-level constant so compute_poly_reward and tests share the order.
_POLY_PAIRS = jnp.array(
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=jnp.int32
)


def init_poly_genome(rng_key, config: dict) -> jnp.ndarray:
    """Initialize polynomial residual genome: shape (10,), all zeros.

    Zero-init so the residual contributes 0 to the reward at birth (system
    starts at exact K&D linear baseline). rng_key is accepted for API
    symmetry with init_residual_genome but unused.
    """
    del rng_key, config  # unused — zero init has no randomness
    return jnp.zeros((10,), dtype=jnp.float32)


def compute_poly_reward(
    linear_genome: jnp.ndarray,
    poly_genome: jnp.ndarray,
    stimuli: jnp.ndarray,
) -> jnp.ndarray:
    """Combined K&D linear + polynomial (quadratic + interaction) reward.

    Args:
        linear_genome: shape (4,) float32 — [w_eat, w_act, w_prey, w_pred].
        poly_genome:   shape (10,) float32 — first 4 are quadratic weights,
                       last 6 are pairwise interaction weights in pair order
                       (0,1),(0,2),(0,3),(1,2),(1,3),(2,3).
        stimuli:       shape (4,) float32 — [n_eaten, motor_norm, max_s_prey,
                       max_s_pred].

    Returns:
        Scalar float32 reward.
    """
    coefs = jnp.array([1.0, 0.01, 0.1, 0.1])
    linear_part = jnp.sum(linear_genome * stimuli * coefs)

    w_sq = poly_genome[:4]                       # (4,)
    sq_part = jnp.sum(w_sq * stimuli * stimuli)

    w_xy = poly_genome[4:]                       # (6,)
    # Gather x_i and x_j for each canonical pair, then dot with w_xy.
    xi = stimuli[_POLY_PAIRS[:, 0]]              # (6,)
    xj = stimuli[_POLY_PAIRS[:, 1]]              # (6,)
    interaction_part = jnp.sum(w_xy * xi * xj)

    return linear_part + sq_part + interaction_part


# ─── Temporal reward genome (Axis 3) ────────────────────────────────────────

class TemporalRewardMLP(nn.Module):
    """MLP reward network over a rolling window of stimuli.

    input(k*4) → Dense(h, tanh) → Dense(h, tanh) → Dense(1, linear).
    With k=10, h=16: 945 total parameters (656 + 272 + 17).
    """
    hidden_size: int = 16

    @nn.compact
    def __call__(self, flat_window):
        x = nn.Dense(self.hidden_size)(flat_window)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)


def init_temporal_genome(rng_key, config):
    """Initialize temporal reward genome as a Flax parameter PyTree.

    Architecture: input(k*4) → Dense(h, tanh) → Dense(h, tanh) → Dense(1).
    """
    k = config["reward_context_window"]
    hidden = config["temporal_hidden_size"]
    net = TemporalRewardMLP(hidden_size=hidden)
    params = net.init(rng_key, jnp.zeros(k * 4))
    return params


def compute_temporal_reward(genome, obs_window):
    """Forward pass through temporal reward MLP.

    Args:
        genome: Flax parameter PyTree from init_temporal_genome.
        obs_window: shape (k, 4) — rolling window of stimulus vectors.
                    Each row is [n_eaten, motor_norm, max_s_prey, max_s_pred].
    Returns:
        Scalar float32 reward.
    """
    hidden_size = genome['params']['Dense_0']['kernel'].shape[1]
    net = TemporalRewardMLP(hidden_size=hidden_size)
    flat = obs_window.reshape(-1)
    return net.apply(genome, flat)
