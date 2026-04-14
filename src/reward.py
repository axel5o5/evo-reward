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


def init_temporal_genome(rng_key, config):
    raise NotImplementedError("Phase 2")


def compute_temporal_reward(genome, obs_window):
    raise NotImplementedError("Phase 2")
