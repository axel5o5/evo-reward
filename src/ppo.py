"""
ppo.py
------
GAE computation and PPO update logic.

GAE parameters: gamma = 0.999, lambda = 0.95.
PPO parameters: clip_epsilon = 0.2, entropy_coef = 0.001,
                vf_coef = 0.5, lr = 3e-4, adam_eps = 1e-7, epochs = 10.
"""

import jax
import jax.numpy as jnp
import optax

from src.policy import PolicyNetwork


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
    # Unpack rollout
    observations = rollout["observations"]
    actions = rollout["actions"]
    old_log_probs = rollout["log_probs"]
    rewards = rollout["rewards"]
    values = rollout["values"]
    dones = rollout["dones"]

    N = observations.shape[0]

    # Compute GAE targets
    last_value = 0.0  # conservative: assume episode ends
    advantages, returns = compute_gae(rewards, values, dones, last_value, config)
    # Normalize advantages
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    # Build network for forward passes
    net = PolicyNetwork(
        hidden_size=config["policy_hidden_size"],
        action_dim=2,
    )

    # Optimizer
    optimizer = optax.adam(
        learning_rate=config["lr"],
        eps=config["adam_eps"],
    )

    clip_eps = config["clip_epsilon"]
    entropy_coef = config["entropy_coef"]
    vf_coef = config.get("vf_coef", 0.5)
    n_epochs = config["ppo_epochs"]
    minibatch_size = config["minibatch_size"]

    rng = jax.random.PRNGKey(0)

    # Track metrics from last epoch for reporting
    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0
    last_approx_kl = 0.0

    for epoch in range(n_epochs):
        # Shuffle indices
        rng, shuffle_rng = jax.random.split(rng)
        indices = jax.random.permutation(shuffle_rng, N)

        for start in range(0, N, minibatch_size):
            end = min(start + minibatch_size, N)
            mb_idx = indices[start:end]

            mb_obs = observations[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            def loss_fn(p):
                # Vectorized forward pass over minibatch
                action_means, log_stds, vals = jax.vmap(
                    lambda o: net.apply(p, o)
                )(mb_obs)

                stds = jnp.exp(log_stds)

                # Log probability of taken actions under current policy
                # actions were stored as sigmoid-scaled; we need raw actions
                # to compute log probs. But since we stored log_probs at
                # collection time using the raw action, we need to recompute
                # from the Gaussian on raw action space.
                #
                # The actions in the rollout are sigmoid-scaled motor forces.
                # Invert sigmoid_scale: raw = logit((action + 20) / 100)
                raw_actions = jnp.log(
                    (mb_actions + 20.0) / (100.0 - mb_actions - 20.0 + 100.0)
                )
                # Simplify: sigmoid(raw) = (action + 20) / 100
                # raw = log((action + 20) / (80 - action))
                clamped = jnp.clip(mb_actions, -19.99, 79.99)
                raw_actions = jnp.log((clamped + 20.0) / (80.0 - clamped))

                # Gaussian log prob
                new_log_probs = -0.5 * jnp.sum(
                    jnp.log(2 * jnp.pi) + 2 * log_stds + ((raw_actions - action_means) / stds) ** 2,
                    axis=-1,
                )

                # Policy loss (clipped surrogate)
                ratio = jnp.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
                policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

                # Value loss
                value_loss = 0.5 * jnp.mean((vals - mb_returns) ** 2)

                # Entropy bonus (Gaussian entropy = 0.5 * ln(2*pi*e*sigma^2) per dim)
                entropy = 0.5 * jnp.sum(1.0 + jnp.log(2 * jnp.pi) + 2 * log_stds[0])

                # Approx KL for diagnostics
                approx_kl = jnp.mean(mb_old_log_probs - new_log_probs)

                total_loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy
                return total_loss, (policy_loss, value_loss, entropy, approx_kl)

            (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

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
