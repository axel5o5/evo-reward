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


# Extension stubs -- raise NotImplementedError in Phase 0
def init_mlp_genome(rng_key, config):
    raise NotImplementedError("Phase 2")


def compute_mlp_reward(genome, stimuli):
    raise NotImplementedError("Phase 2")


def init_temporal_genome(rng_key, config):
    raise NotImplementedError("Phase 2")


def compute_temporal_reward(genome, obs_window):
    raise NotImplementedError("Phase 2")
