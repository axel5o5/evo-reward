"""
ppo.py
------
GAE computation and PPO update logic.

GAE parameters: gamma = 0.999, lambda = 0.95.
PPO parameters: clip_epsilon = 0.2, entropy_coef = 0.001,
                vf_coef = 0.5, lr = 3e-4, adam_eps = 1e-7, epochs = 10.

Performance note: _get_jit_ppo_step() returns a @jax.jit-compiled per-minibatch
update. It is compiled once per unique (hidden_size, obs_dim, clip_eps, ...) tuple
and cached. Without this, jax.value_and_grad re-traces on every call (10 epochs ×
4 minibatches × N_agents = thousands of traces), which is extremely slow.
"""

import jax
import jax.numpy as jnp
import optax

from src.policy import PolicyNetwork


# Cache for JIT-compiled PPO step functions
_ppo_step_cache = {}


def _get_jit_ppo_step(config):
    """Get or create a JIT-compiled per-minibatch PPO gradient step.

    Keyed by hyperparameters so the function is compiled exactly once per
    unique configuration (typically just once per run).
    """
    cache_key = (
        config["policy_hidden_size"],
        config["obs_dim"],
        float(config["clip_epsilon"]),
        float(config["entropy_coef"]),
        float(config.get("vf_coef", 0.5)),
        float(config["lr"]),
        float(config["adam_eps"]),
    )
    if cache_key in _ppo_step_cache:
        return _ppo_step_cache[cache_key]

    net = PolicyNetwork(hidden_size=config["policy_hidden_size"], action_dim=2)
    optimizer = optax.adam(learning_rate=config["lr"], eps=config["adam_eps"])
    clip_eps = float(config["clip_epsilon"])
    entropy_coef = float(config["entropy_coef"])
    vf_coef = float(config.get("vf_coef", 0.5))

    @jax.jit
    def ppo_step(params, opt_state, mb_obs, mb_actions, mb_old_log_probs,
                 mb_advantages, mb_returns):
        """One minibatch gradient update. Returns (params, opt_state, aux)."""
        def loss_fn(p):
            action_means, log_stds, vals = jax.vmap(
                lambda o: net.apply(p, o)
            )(mb_obs)

            stds = jnp.exp(log_stds)

            # Invert sigmoid_scale: sigmoid(raw) = (action+20)/100
            # → raw = log((action+20)/(80−action))
            clamped = jnp.clip(mb_actions, -19.99, 79.99)
            raw_actions = jnp.log((clamped + 20.0) / (80.0 - clamped))

            new_log_probs = -0.5 * jnp.sum(
                jnp.log(2 * jnp.pi) + 2 * log_stds
                + ((raw_actions - action_means) / stds) ** 2,
                axis=-1,
            )

            ratio = jnp.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

            value_loss = 0.5 * jnp.mean((vals - mb_returns) ** 2)

            # Gaussian entropy (per action dim, then sum)
            entropy = jnp.mean(0.5 * jnp.sum(1.0 + jnp.log(2 * jnp.pi) + 2 * log_stds, axis=-1))

            approx_kl = jnp.mean(mb_old_log_probs - new_log_probs)

            total_loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy
            return total_loss, (policy_loss, value_loss, entropy, approx_kl)

        (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, aux

    _ppo_step_cache[cache_key] = ppo_step
    return ppo_step


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: float,
    config: dict,
) -> tuple:
    """
    Generalized Advantage Estimation via jax.lax.scan (JIT-compilable).

    last_value = V(s_{N+1}); use 0.0 if agent died at end of rollout.
    Returns: (advantages (N,), returns (N,))
    """
    gamma = config["gamma"]
    lam = config["gae_lambda"]

    # next_values[t] = V(s_{t+1}), with last_value at the boundary
    next_values = jnp.concatenate([values[1:], jnp.array([last_value])])
    next_non_terminals = 1.0 - dones.astype(jnp.float32)
    deltas = rewards + gamma * next_values * next_non_terminals - values

    # Backward scan: gae_t = delta_t + gamma*lam*(1-done_t)*gae_{t+1}
    def scan_fn(gae_next, inputs):
        delta_t, nnt_t = inputs
        gae_t = delta_t + gamma * lam * nnt_t * gae_next
        return gae_t, gae_t

    _, reversed_advantages = jax.lax.scan(
        scan_fn,
        jnp.float32(0.0),
        (jnp.flip(deltas), jnp.flip(next_non_terminals)),
    )
    advantages = jnp.flip(reversed_advantages)
    returns = advantages + values
    return advantages, returns


def ppo_update(
    params,
    opt_state,
    rollout: dict,
    config: dict,
) -> tuple:
    """
    Run PPO update on a completed rollout buffer.

    rollout keys: observations (N, obs_dim), actions (N, 2), log_probs (N,),
                  rewards (N,), values (N,), dones (N,).

    Returns: (new_params, new_opt_state, info_dict)
    info_dict keys: "policy_loss", "value_loss", "entropy", "approx_kl"
    """
    # Accept numpy or JAX arrays in the rollout (numpy is faster to write to)
    observations = jnp.asarray(rollout["observations"])
    actions = jnp.asarray(rollout["actions"])
    old_log_probs = jnp.asarray(rollout["log_probs"])
    rewards = jnp.asarray(rollout["rewards"])
    values = jnp.asarray(rollout["values"])
    dones = jnp.asarray(rollout["dones"])

    N = observations.shape[0]

    last_value = 0.0
    advantages, returns = compute_gae(rewards, values, dones, last_value, config)
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    n_epochs = config["ppo_epochs"]
    minibatch_size = config["minibatch_size"]

    ppo_step = _get_jit_ppo_step(config)

    rng = jax.random.PRNGKey(0)

    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0
    last_approx_kl = 0.0

    for epoch in range(n_epochs):
        rng, shuffle_rng = jax.random.split(rng)
        indices = jax.random.permutation(shuffle_rng, N)

        for start in range(0, N, minibatch_size):
            end = min(start + minibatch_size, N)
            mb_idx = indices[start:end]

            params, opt_state, aux = ppo_step(
                params, opt_state,
                observations[mb_idx],
                actions[mb_idx],
                old_log_probs[mb_idx],
                advantages[mb_idx],
                returns[mb_idx],
            )

            last_policy_loss = float(aux[0])
            last_value_loss = float(aux[1])
            last_entropy = float(aux[2])
            last_approx_kl = float(aux[3])

    info = {
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "entropy": last_entropy,
        "approx_kl": last_approx_kl,
    }
    return params, opt_state, info
