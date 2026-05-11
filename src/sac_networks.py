"""
sac_networks.py
---------------
Networks specific to SAC (Soft Actor-Critic) — Q-critic and stochastic
actor. Kept separate from src/policy.py so the PPO path is untouched and
the SAC code can evolve without breaking it.

SACActorNetwork outputs (mean, log_std) — same shape as PolicyNetwork's
policy head, but log_std is **state-dependent** (a second head) rather
than the state-independent learned scalar PPO uses. The state-dependent
variant is what the SAC papers use; it lets the policy be sharper in
some states and broader in others, which matters more for SAC's
entropy-regularized objective than for PPO's clipped surrogate.

QNetwork takes (obs, action) → scalar. Twin critics: call init_q_network
twice with different RNG keys to get Q1 and Q2.

Action squashing: actions are mapped to [-20, 80] via sigmoid_scale, the
same mapping PPO uses. The reparameterized log-prob includes the
change-of-variables correction:
    a = 100 · σ(raw) - 20
    da/draw = 100 · σ(raw) · (1 - σ(raw))
    log|da/draw| = log(100) - softplus(-raw) - softplus(raw)
log π(a|s) = log N(raw | μ(s), σ(s)) - log|da/draw|
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


# Standard SAC clipping range for log_std. Outside this band the policy
# collapses to nearly-deterministic (-5) or pathologically broad (2).
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class SACActorNetwork(nn.Module):
    """Stochastic actor: obs → (action_mean, log_std). Both heads share
    the trunk; log_std is state-dependent (unlike PPO's scalar log_std).
    """
    hidden_size: int = 64
    action_dim: int = 2

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_size)(obs)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.tanh(x)
        action_mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return action_mean, log_std


class QNetwork(nn.Module):
    """Q(s, a) → scalar. Concatenates obs and action then trunks through
    two tanh hidden layers."""
    hidden_size: int = 64

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.tanh(x)
        q = nn.Dense(1)(x)
        return jnp.squeeze(q, axis=-1)


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------

def init_sac_actor(rng_key, config: dict) -> tuple:
    """Initialize one SAC actor + its Adam optimizer state.

    Returns: (params, opt_state)
    """
    hidden_size = config["policy_hidden_size"]
    obs_dim = config["obs_dim"]
    net = SACActorNetwork(hidden_size=hidden_size, action_dim=2)
    dummy_obs = jnp.zeros(obs_dim)
    params = net.init(rng_key, dummy_obs)

    lr = float(config.get("sac_actor_lr", config["lr"]))
    optimizer = optax.adam(learning_rate=lr, eps=config["adam_eps"])
    opt_state = optimizer.init(params)
    return params, opt_state


def init_q_network(rng_key, config: dict) -> tuple:
    """Initialize one Q-network + its Adam optimizer state.

    For twin critics, call this twice with different rng keys.
    """
    hidden_size = config["policy_hidden_size"]
    obs_dim = config["obs_dim"]
    net = QNetwork(hidden_size=hidden_size)
    dummy_obs = jnp.zeros(obs_dim)
    dummy_action = jnp.zeros(2)
    params = net.init(rng_key, dummy_obs, dummy_action)

    lr = float(config.get("sac_critic_lr", config["lr"]))
    optimizer = optax.adam(learning_rate=lr, eps=config["adam_eps"])
    opt_state = optimizer.init(params)
    return params, opt_state


def init_alpha(config: dict) -> tuple:
    """Initialize the temperature parameter α (stored as log_alpha for
    unconstrained optimization).

    Returns: (log_alpha, opt_state)
    """
    log_alpha = jnp.array(float(config.get("sac_initial_log_alpha", 0.0)),
                          dtype=jnp.float32)
    lr = float(config.get("sac_alpha_lr", config["lr"]))
    optimizer = optax.adam(learning_rate=lr, eps=config["adam_eps"])
    opt_state = optimizer.init(log_alpha)
    return log_alpha, opt_state


# ---------------------------------------------------------------------------
# Reparameterized squashed sampling
# ---------------------------------------------------------------------------

def reparam_sample(action_mean, log_std, rng_key):
    """Reparameterized sample with sigmoid squashing.

    Inputs:
      action_mean: (action_dim,) — actor mean
      log_std:     (action_dim,) — actor log-std (already clipped)
      rng_key:     PRNGKey

    Returns:
      action:   (action_dim,) — sigmoid-scaled to [-20, 80]
      log_prob: scalar — log π(a|s) with change-of-variables correction
      raw:      (action_dim,) — pre-squash sample (for debugging / tests)

    Designed so gradients flow through action_mean and log_std (the
    sample noise eps is detached). Mirrors the standard SAC trick.
    """
    std = jnp.exp(log_std)
    eps = jax.random.normal(rng_key, shape=action_mean.shape)
    raw = action_mean + std * eps  # reparam

    # Gaussian log-prob in raw space.
    log_prob_raw = jnp.sum(
        -0.5 * jnp.log(2 * jnp.pi) - log_std
        - 0.5 * ((raw - action_mean) / std) ** 2,
        axis=-1,
    )

    # a = 100·σ(raw) - 20 → log|da/draw| = log(100) - softplus(-raw) - softplus(raw)
    # (per dim; sum over action_dim).
    log_det_jacobian = jnp.sum(
        jnp.log(100.0) - jax.nn.softplus(-raw) - jax.nn.softplus(raw),
        axis=-1,
    )
    log_prob = log_prob_raw - log_det_jacobian

    action = 100.0 * jax.nn.sigmoid(raw) - 20.0
    return action, log_prob, raw
