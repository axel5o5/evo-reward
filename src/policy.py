"""
policy.py
---------
MLP policy network with shared trunk, separate policy and value heads.

Architecture (confirmed against emevo cf_predator.py):
  Input (obs_dim=205) -> Dense(64, tanh) -> Dense(64, tanh)
    -> policy head: Dense(2) action means + learned log_std (2,)
    -> value head:  Dense(1)

Action mapping (sigmoid_scale):
  action = 100 * sigmoid(raw) - 20   maps to [-20, 80]
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


class PolicyNetwork(nn.Module):
    """Shared-trunk MLP with separate policy and value heads.

    2 hidden layers (64 units, tanh), matching emevo ppo_normal.py.
    """
    hidden_size: int = 64
    action_dim: int = 2

    @nn.compact
    def __call__(self, obs):
        # Shared trunk: 2 hidden layers
        x = nn.Dense(self.hidden_size)(obs)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.tanh(x)

        # Policy head: action means
        action_mean = nn.Dense(self.action_dim)(x)

        # Value head: scalar
        value = nn.Dense(1)(x)
        value = jnp.squeeze(value, axis=-1)

        # Learned log_std (state-independent)
        log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.action_dim,),
        )

        return action_mean, log_std, value


def sigmoid_scale(raw_action):
    """Map raw network output to [-20, 80] via sigmoid.

    action = 100 * sigmoid(raw) - 20
    Confirmed in emevo source (Session 2).
    """
    return 100.0 * jax.nn.sigmoid(raw_action) - 20.0


def init_policy(rng_key, config: dict) -> tuple:
    """Initialize fresh policy network params and Adam optimizer state.

    Returns: (params, opt_state)
    """
    net = PolicyNetwork(
        hidden_size=config["policy_hidden_size"],
        action_dim=2,
    )
    dummy_obs = jnp.zeros(config["obs_dim"])
    params = net.init(rng_key, dummy_obs)

    optimizer = optax.adam(
        learning_rate=config["lr"],
        eps=config["adam_eps"],
    )
    opt_state = optimizer.init(params)

    return params, opt_state


def policy_forward(params, obs, config: dict) -> tuple:
    """Deterministic forward pass (no sampling).

    Returns: (action_mean (2,), log_std (2,), value float)
    """
    net = PolicyNetwork(
        hidden_size=config["policy_hidden_size"],
        action_dim=2,
    )
    action_mean, log_std, value = net.apply(params, obs)
    return action_mean, log_std, value


def sample_action(params, obs, rng_key, config: dict) -> tuple:
    """Forward pass + stochastic action sample.

    Returns: (action (2,), log_prob float, value float)
    action is sigmoid-scaled to [-20, 80].
    """
    action_mean, log_std, value = policy_forward(params, obs, config)
    std = jnp.exp(log_std)

    # Sample from Normal(mean, std)
    noise = jax.random.normal(rng_key, shape=action_mean.shape)
    raw_action = action_mean + std * noise

    # Log probability under the Gaussian (before sigmoid transform)
    log_prob = -0.5 * jnp.sum(
        jnp.log(2 * jnp.pi) + 2 * log_std + ((raw_action - action_mean) / std) ** 2
    )

    # Apply sigmoid scaling to get motor forces
    action = sigmoid_scale(raw_action)

    return action, float(log_prob), float(value)


# ─── LSTM policy (Axis 4) ────────────────────────────────────────────────────

class LSTMPolicyNetwork(nn.Module):
    """LSTM policy with dense layer and separate policy/value heads.

    Architecture: obs(obs_dim) → LSTM(lstm_hidden_size) → Dense(hidden_size, tanh)
        → policy head: Dense(action_dim) + learned log_std
        → value head: Dense(1)

    The LSTM cell takes (carry, obs) where carry = (c, h).
    """
    lstm_hidden_size: int = 64
    hidden_size: int = 64
    action_dim: int = 2

    @nn.compact
    def __call__(self, carry, obs):
        lstm_cell = nn.OptimizedLSTMCell(features=self.lstm_hidden_size)
        new_carry, lstm_out = lstm_cell(carry, obs)

        x = nn.Dense(self.hidden_size)(lstm_out)
        x = nn.tanh(x)

        action_mean = nn.Dense(self.action_dim)(x)

        value = nn.Dense(1)(x)
        value = jnp.squeeze(value, axis=-1)

        log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.action_dim,),
        )

        return new_carry, action_mean, log_std, value


def init_lstm_policy(rng_key, config: dict) -> tuple:
    """Initialize LSTM policy params, optimizer state, and zero hidden state.

    Returns: (params, opt_state, initial_hidden_state)
        initial_hidden_state: shape (2, lstm_hidden_size) — packed (c, h), zeros.
    """
    lstm_hidden_size = config["lstm_hidden_size"]
    net = LSTMPolicyNetwork(
        lstm_hidden_size=lstm_hidden_size,
        hidden_size=config["policy_hidden_size"],
        action_dim=2,
    )
    carry_init = (jnp.zeros(lstm_hidden_size), jnp.zeros(lstm_hidden_size))
    dummy_obs = jnp.zeros(config["obs_dim"])
    params = net.init(rng_key, carry_init, dummy_obs)

    optimizer = optax.adam(
        learning_rate=config["lr"],
        eps=config["adam_eps"],
    )
    opt_state = optimizer.init(params)

    initial_hidden = jnp.zeros((2, lstm_hidden_size))
    return params, opt_state, initial_hidden


def policy_forward_lstm(params, obs, hidden_state, config: dict) -> tuple:
    """Deterministic LSTM forward pass (no sampling).

    Args:
        params: Flax parameter PyTree.
        obs: shape (obs_dim,).
        hidden_state: shape (2, lstm_hidden_size) — packed (c, h).
        config: dict with lstm_hidden_size, policy_hidden_size.

    Returns: (action_mean (2,), log_std (2,), value float, new_hidden (2, H))
    """
    net = LSTMPolicyNetwork(
        lstm_hidden_size=config["lstm_hidden_size"],
        hidden_size=config["policy_hidden_size"],
        action_dim=2,
    )
    carry = (hidden_state[0], hidden_state[1])
    new_carry, action_mean, log_std, value = net.apply(params, carry, obs)
    new_hidden = jnp.stack([new_carry[0], new_carry[1]], axis=0)
    return action_mean, log_std, value, new_hidden


def sample_action_lstm(params, obs, hidden_state, rng_key, config: dict) -> tuple:
    """LSTM forward pass + stochastic action sample.

    Args:
        params: Flax parameter PyTree.
        obs: shape (obs_dim,).
        hidden_state: shape (2, lstm_hidden_size) — packed (c, h).
        rng_key: JAX PRNGKey.
        config: dict with lstm_hidden_size, policy_hidden_size.

    Returns: (action (2,), log_prob float, value float, new_hidden (2, H))
    """
    action_mean, log_std, value, new_hidden = policy_forward_lstm(
        params, obs, hidden_state, config,
    )
    std = jnp.exp(log_std)

    noise = jax.random.normal(rng_key, shape=action_mean.shape)
    raw_action = action_mean + std * noise

    log_prob = -0.5 * jnp.sum(
        jnp.log(2 * jnp.pi) + 2 * log_std + ((raw_action - action_mean) / std) ** 2
    )

    action = sigmoid_scale(raw_action)

    return action, log_prob, value, new_hidden
