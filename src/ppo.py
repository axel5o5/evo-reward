"""
ppo.py
------
GAE computation and PPO update logic.

GAE parameters: gamma = 0.999, lambda = 0.95.
PPO parameters: clip_epsilon = 0.2, entropy_coef = 0.001,
                lr = 3e-4, adam_eps = 1e-7, epochs = 10.
"""

import jax.numpy as jnp


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: float,
    config: dict,
) -> tuple:
    """
    Generalized Advantage Estimation.

    last_value = V(s_{N+1}); use 0.0 if agent died at end of rollout.
    Returns: (advantages (N,), returns (N,))
    """
    gamma = config["gamma"]
    lam = config["gae_lambda"]
    N = len(rewards)

    advantages = jnp.zeros(N)
    gae = 0.0

    for t in reversed(range(N)):
        if t == N - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        # If done, next_value is zeroed out (no bootstrap beyond terminal)
        next_non_terminal = 1.0 - dones[t].astype(jnp.float32)
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages = advantages.at[t].set(gae)

    returns = advantages + values
    return advantages, returns
