"""
jax_sac.py
----------
Per-agent SAC (Soft Actor-Critic) update operating on SimState replay-
buffer slices. Mirrors the structure of src/jax_ppo.py: vmap over agent
slots with lax.cond to selectively update only agents whose replay
buffers have warmed up (replay_size >= min_size).

One key difference from PPO: SAC updates **every env step** (not every
rollout_steps env steps). Cost is amortized by minibatch size — each
update does one Adam step on each of {actor, Q1, Q2, alpha}, plus a
Polyak soft-update of the two target critics.

What this module does NOT do (yet):
  - It does not allocate or read the replay buffer fields on SimState
    — that's the job of the integration step (chunk 2). The
    `build_sac_update_fn` returned here accepts the replay arrays as
    explicit args so it can be wired in cleanly later.
  - It does not branch on `learner_type` in the simulator loop.

The SAC math implemented here:

  Actor loss:
      L_actor = E_{a ~ π}[ α log π(a|s) - min(Q1(s,a), Q2(s,a)) ]
    (with reparameterization through the squashed Gaussian)

  Critic loss (per critic):
      y = r + γ (1-d) [ min(Q1'(s', a'), Q2'(s', a')) - α log π(a'|s') ]
      L_critic = E[ (Q(s, a) - y)² ]
    (a' is sampled fresh from the current actor at s'; targets use the
    target networks Q1', Q2' for the Q-values inside the bracket)

  Alpha loss (autotune temperature):
      L_α = -E[ log α · (log π(a|s) + target_entropy) ]
    target_entropy defaults to -action_dim = -2.
"""

import functools

import jax
import jax.numpy as jnp
import optax

from src.sac_networks import (
    SACActorNetwork, QNetwork, reparam_sample,
    LOG_STD_MIN, LOG_STD_MAX,
)


def build_sac_update_fn(config):
    """Build the JIT-compiled per-step SAC update for all agent slots.

    Returns: maybe_sac_update_all(...) — a function that takes all the
    per-agent SAC state plus the replay buffer arrays and returns the
    updated state. Designed for one call per env-step (after the step's
    transition has been written to the replay buffer).

    Conditional update: skips slots where `is_active=False` OR where
    `replay_size < min_size` (warmup period). Inactive / unwarmed slots
    return their inputs unchanged.
    """
    hidden_size = config["policy_hidden_size"]
    obs_dim = config["obs_dim"]
    action_dim = 2

    # Optimizer LRs default to the global PPO lr if not separately set.
    actor_lr = float(config.get("sac_actor_lr", config["lr"]))
    critic_lr = float(config.get("sac_critic_lr", config["lr"]))
    alpha_lr = float(config.get("sac_alpha_lr", config["lr"]))
    adam_eps = float(config["adam_eps"])

    gamma = float(config["gamma"])
    tau = float(config.get("sac_target_tau", 0.005))
    minibatch_size = int(config.get("sac_minibatch_size", 64))
    min_replay_size = int(config.get("sac_replay_min_size", 256))
    # Default to -action_dim, the SAC paper convention for continuous control.
    target_entropy = float(config.get("sac_target_entropy", -float(action_dim)))

    # Age-keyed LR + per-species multiplier — shared with PPO so existing
    # configs control both algorithms the same way.
    lr_sched_enable = bool(config.get("lr_schedule_enable", False))
    lr_initial = float(config.get("lr_schedule_initial", actor_lr))
    lr_final = float(config.get("lr_schedule_final", actor_lr))
    lr_decay_steps = float(config.get("lr_schedule_decay_steps", 1.0))
    lr_prey_mult = float(config.get("lr_prey_multiplier", 1.0))
    lr_pred_mult = float(config.get("lr_pred_multiplier", 1.0))

    def _lr_scale(age, species):
        species_mult = jnp.where(species == 0, lr_prey_mult, lr_pred_mult)
        if not lr_sched_enable:
            return species_mult.astype(jnp.float32)
        a = age.astype(jnp.float32)
        frac = jnp.minimum(a / lr_decay_steps, 1.0)
        lr_at_age = lr_initial + (lr_final - lr_initial) * frac
        # Scale relative to actor_lr — Adam's base LR is set to actor_lr.
        return species_mult * (lr_at_age / actor_lr)

    actor_net = SACActorNetwork(hidden_size=hidden_size, action_dim=action_dim)
    q_net = QNetwork(hidden_size=hidden_size)

    actor_opt = optax.adam(learning_rate=actor_lr, eps=adam_eps)
    critic_opt = optax.adam(learning_rate=critic_lr, eps=adam_eps)
    alpha_opt = optax.adam(learning_rate=alpha_lr, eps=adam_eps)

    # -----------------------------------------------------------------------
    # Loss functions — all return a scalar suitable for jax.grad.
    # -----------------------------------------------------------------------

    def critic_loss_fn(q_params, q_target_other_params,
                       q1_target_params, q2_target_params,
                       actor_params, log_alpha,
                       mb_obs, mb_action, mb_reward, mb_next_obs, mb_done,
                       rng):
        """Loss for ONE critic. q_params is the online critic we're
        optimizing; q_target_other_params is the *other* online critic's
        params (unused in this minimal formulation; kept for ease of a
        twin-target extension if we add CDQ-style joint loss later)."""
        del q_target_other_params  # not used in this variant

        # Bootstrap target — gradient stops at y (target nets are
        # constants in this loss; the actor params used here are also
        # constants for this critic's update).
        def compute_target():
            next_means, next_log_stds = jax.vmap(
                lambda o: actor_net.apply(actor_params, o)
            )(mb_next_obs)
            # Per-sample reparam needs per-sample rngs.
            keys = jax.random.split(rng, mb_next_obs.shape[0])
            next_actions, next_log_probs, _ = jax.vmap(reparam_sample)(
                next_means, next_log_stds, keys,
            )
            q1_next = jax.vmap(
                lambda o, a: q_net.apply(q1_target_params, o, a)
            )(mb_next_obs, next_actions)
            q2_next = jax.vmap(
                lambda o, a: q_net.apply(q2_target_params, o, a)
            )(mb_next_obs, next_actions)
            q_min_next = jnp.minimum(q1_next, q2_next)
            alpha = jnp.exp(log_alpha)
            soft_q_next = q_min_next - alpha * next_log_probs
            y = mb_reward + gamma * (1.0 - mb_done.astype(jnp.float32)) * soft_q_next
            return jax.lax.stop_gradient(y)

        y = compute_target()
        q_pred = jax.vmap(
            lambda o, a: q_net.apply(q_params, o, a)
        )(mb_obs, mb_action)
        return jnp.mean((q_pred - y) ** 2)

    def actor_loss_fn(actor_params,
                      q1_params, q2_params, log_alpha,
                      mb_obs, rng):
        """Actor loss: maximize Q - α log π. Gradients flow into
        actor_params through the reparameterized action a = f(s; θ, ε)
        and into the Q-networks (frozen here)."""
        means, log_stds = jax.vmap(
            lambda o: actor_net.apply(actor_params, o)
        )(mb_obs)
        keys = jax.random.split(rng, mb_obs.shape[0])
        actions, log_probs, _ = jax.vmap(reparam_sample)(means, log_stds, keys)

        q1 = jax.vmap(lambda o, a: q_net.apply(q1_params, o, a))(mb_obs, actions)
        q2 = jax.vmap(lambda o, a: q_net.apply(q2_params, o, a))(mb_obs, actions)
        q_min = jnp.minimum(q1, q2)

        alpha = jnp.exp(log_alpha)
        # Loss: minimize α log π - Q. Equivalently maximize Q - α log π.
        return jnp.mean(alpha * log_probs - q_min), log_probs

    def alpha_loss_fn(log_alpha, log_probs_detached):
        """Temperature autotune: drive E[log π] toward target_entropy.

        Standard derivation: ∂L_α/∂log_α = -mean(log π + target_entropy).
        Detaching log_probs is essential — α and π are optimized
        independently.
        """
        # log_alpha is a scalar; broadcast over the minibatch.
        return -jnp.mean(log_alpha * (log_probs_detached + target_entropy))

    def polyak(target_params, online_params):
        return jax.tree_util.tree_map(
            lambda t, o: tau * o + (1.0 - tau) * t,
            target_params, online_params,
        )

    # -----------------------------------------------------------------------
    # Per-agent step
    # -----------------------------------------------------------------------

    def single_agent_sac(
        actor_params, q1_params, q2_params,
        q1_target_params, q2_target_params, log_alpha,
        actor_opt_state, q1_opt_state, q2_opt_state, alpha_opt_state,
        replay_obs, replay_action, replay_reward, replay_next_obs, replay_done,
        replay_size, is_active, rng, age, species,
    ):
        should_update = is_active & (replay_size >= min_replay_size)
        lr_scale = _lr_scale(age, species)

        def do_update(_):
            # Split RNG: minibatch sample, target sample, actor sample.
            rng_mb, rng_critic, rng_actor = jax.random.split(rng, 3)

            # Uniform-with-replacement index sample over [0, replay_size).
            # Repeats are fine for SAC; the buffer is large.
            idx = jax.random.randint(
                rng_mb, (minibatch_size,), 0, jnp.maximum(replay_size, 1),
            )
            mb_obs = replay_obs[idx]
            mb_action = replay_action[idx]
            mb_reward = replay_reward[idx]
            mb_next_obs = replay_next_obs[idx]
            mb_done = replay_done[idx]

            # ---- Critic updates (both critics in parallel) ----
            grad_c = jax.grad(critic_loss_fn)
            grads_q1 = grad_c(
                q1_params, q2_params,
                q1_target_params, q2_target_params,
                actor_params, log_alpha,
                mb_obs, mb_action, mb_reward, mb_next_obs, mb_done,
                rng_critic,
            )
            grads_q2 = grad_c(
                q2_params, q1_params,
                q1_target_params, q2_target_params,
                actor_params, log_alpha,
                mb_obs, mb_action, mb_reward, mb_next_obs, mb_done,
                rng_critic,
            )
            updates_q1, new_q1_opt = critic_opt.update(grads_q1, q1_opt_state, q1_params)
            updates_q2, new_q2_opt = critic_opt.update(grads_q2, q2_opt_state, q2_params)
            updates_q1 = jax.tree_util.tree_map(lambda u: u * lr_scale, updates_q1)
            updates_q2 = jax.tree_util.tree_map(lambda u: u * lr_scale, updates_q2)
            new_q1 = optax.apply_updates(q1_params, updates_q1)
            new_q2 = optax.apply_updates(q2_params, updates_q2)

            # ---- Actor update (uses fresh critics) ----
            (a_loss, log_probs), grads_actor = jax.value_and_grad(
                actor_loss_fn, has_aux=True,
            )(actor_params, new_q1, new_q2, log_alpha, mb_obs, rng_actor)
            updates_actor, new_actor_opt = actor_opt.update(
                grads_actor, actor_opt_state, actor_params,
            )
            updates_actor = jax.tree_util.tree_map(lambda u: u * lr_scale, updates_actor)
            new_actor = optax.apply_updates(actor_params, updates_actor)

            # ---- Alpha update (uses detached log_probs from actor step) ----
            log_probs_detached = jax.lax.stop_gradient(log_probs)
            grads_alpha = jax.grad(alpha_loss_fn)(log_alpha, log_probs_detached)
            updates_alpha, new_alpha_opt = alpha_opt.update(
                grads_alpha, alpha_opt_state, log_alpha,
            )
            new_log_alpha = optax.apply_updates(log_alpha, updates_alpha)

            # ---- Polyak soft-update of target critics ----
            new_q1_target = polyak(q1_target_params, new_q1)
            new_q2_target = polyak(q2_target_params, new_q2)

            return (
                new_actor, new_q1, new_q2,
                new_q1_target, new_q2_target, new_log_alpha,
                new_actor_opt, new_q1_opt, new_q2_opt, new_alpha_opt,
            )

        def no_update(_):
            return (
                actor_params, q1_params, q2_params,
                q1_target_params, q2_target_params, log_alpha,
                actor_opt_state, q1_opt_state, q2_opt_state, alpha_opt_state,
            )

        return jax.lax.cond(should_update, do_update, no_update, None)

    batched_sac = jax.vmap(single_agent_sac)

    # Donate the params + opt states (their outputs replace them).
    # Replay buffers and other read-only tensors are NOT donated.
    @functools.partial(
        jax.jit,
        donate_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    )
    def maybe_sac_update_all(
        actor_params, q1_params, q2_params,
        q1_target_params, q2_target_params, log_alpha,
        actor_opt_state, q1_opt_state, q2_opt_state, alpha_opt_state,
        replay_obs, replay_action, replay_reward, replay_next_obs, replay_done,
        replay_size, is_active, rng_keys, ages, species,
    ):
        return batched_sac(
            actor_params, q1_params, q2_params,
            q1_target_params, q2_target_params, log_alpha,
            actor_opt_state, q1_opt_state, q2_opt_state, alpha_opt_state,
            replay_obs, replay_action, replay_reward, replay_next_obs, replay_done,
            replay_size, is_active, rng_keys, ages, species,
        )

    return maybe_sac_update_all
