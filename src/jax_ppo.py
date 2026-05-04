"""
jax_ppo.py
----------
JIT-compiled per-agent PPO update operating on SimState arrays.

Uses vmap over all agent slots with lax.cond to selectively update
agents whose rollout buffers are full (ptr >= rollout_steps).

Inner PPO loop uses lax.fori_loop for epochs and minibatches,
making the entire update JIT-compilable.
"""

import functools

import jax
import jax.numpy as jnp
import optax

from src.policy import PolicyNetwork


def _compute_gae_jax(rewards, values, dones, last_value, gamma, gae_lambda):
    """GAE via jax.lax.scan (backward). Pure JAX."""
    next_values = jnp.concatenate([values[1:], jnp.array([last_value])])
    next_non_terminals = 1.0 - dones.astype(jnp.float32)
    deltas = rewards + gamma * next_values * next_non_terminals - values

    def scan_fn(gae_next, inputs):
        delta_t, nnt_t = inputs
        gae_t = delta_t + gamma * gae_lambda * nnt_t * gae_next
        return gae_t, gae_t

    _, reversed_adv = jax.lax.scan(
        scan_fn,
        jnp.float32(0.0),
        (jnp.flip(deltas), jnp.flip(next_non_terminals)),
    )
    advantages = jnp.flip(reversed_adv)
    returns = advantages + values
    return advantages, returns


def build_ppo_update_fn(config):
    """Build a JIT-compiled function that does per-agent conditional PPO.

    Returns a function: (policy_params, opt_states, rollout_*, rollout_ptrs, rng)
        → (new_params, new_opt_states, new_ptrs)

    This function is vmapped over all agents — each agent independently
    checks whether its buffer is full and runs PPO if so.
    """
    hidden_size = config["policy_hidden_size"]
    obs_dim = config["obs_dim"]
    lr = config["lr"]
    adam_eps = config["adam_eps"]
    clip_eps = config["clip_epsilon"]
    entropy_coef = config["entropy_coef"]
    vf_coef = config.get("vf_coef", 0.5)
    gamma = config["gamma"]
    gae_lambda = config["gae_lambda"]
    rollout_steps = config["rollout_steps"]
    ppo_epochs = config["ppo_epochs"]
    minibatch_size = config["minibatch_size"]
    n_minibatches = rollout_steps // minibatch_size

    # v10 age-keyed LR: per-agent linear ramp from `lr_initial` down to
    # `lr_final` over `lr_decay_steps`, then flat. Implemented by scaling
    # the optimizer's update tree per-agent before apply_updates. Adam's
    # base LR stays at config["lr"]; the schedule multiplies the resulting
    # update by lr_at(age) / lr. Disabled when lr_schedule_enable is false
    # (or absent) — in that case scale_at(age) ≡ 1.0 for any age.
    lr_sched_enable = bool(config.get("lr_schedule_enable", False))
    lr_initial = float(config.get("lr_schedule_initial", lr))
    lr_final = float(config.get("lr_schedule_final", lr))
    lr_decay_steps = float(config.get("lr_schedule_decay_steps", 1.0))

    # Per-species LR multiplier. Default 1.0 → identical to single-LR behavior.
    # Stacks multiplicatively on top of the age schedule. Useful for ablations
    # where one species needs more / less plasticity than the other (e.g.,
    # predators need to learn faster because their lifespans are short).
    lr_prey_mult = float(config.get("lr_prey_multiplier", 1.0))
    lr_pred_mult = float(config.get("lr_pred_multiplier", 1.0))

    def _lr_scale(age, species):
        species_mult = jnp.where(species == 0, lr_prey_mult, lr_pred_mult)
        if not lr_sched_enable:
            return species_mult.astype(jnp.float32)
        a = age.astype(jnp.float32)
        frac = jnp.minimum(a / lr_decay_steps, 1.0)
        lr_at_age = lr_initial + (lr_final - lr_initial) * frac
        return species_mult * (lr_at_age / lr)

    net = PolicyNetwork(hidden_size=hidden_size, action_dim=2)
    optimizer = optax.adam(learning_rate=lr, eps=adam_eps)

    def ppo_loss_and_grad(params, mb_obs, mb_actions, mb_old_log_probs,
                          mb_advantages, mb_returns):
        """Compute PPO loss + gradient for one minibatch."""
        def loss_fn(p):
            action_means, log_stds, vals = jax.vmap(
                lambda o: net.apply(p, o)
            )(mb_obs)

            stds = jnp.exp(log_stds)
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
            entropy = jnp.mean(0.5 * jnp.sum(1.0 + jnp.log(2 * jnp.pi) + 2 * log_stds, axis=-1))

            total_loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy
            return total_loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        return loss, grads

    def single_agent_ppo(params, opt_state, obs, actions, log_probs,
                         rewards, values, dones, ptr, is_active, rng_key,
                         age, species):
        """PPO update for one agent. Called via vmap over all slots.

        If ptr < rollout_steps or agent is inactive, returns inputs unchanged.
        Otherwise, runs full PPO and resets ptr to 0.
        """
        should_update = is_active & (ptr >= rollout_steps)
        lr_scale = _lr_scale(age, species)

        def do_update(_):
            # GAE
            advantages, returns = _compute_gae_jax(
                rewards, values, dones, 0.0, gamma, gae_lambda
            )
            advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

            # PPO training loop: lax.fori_loop over epochs
            def epoch_body(epoch_idx, carry):
                p, os, rng = carry
                rng, shuffle_key = jax.random.split(rng)
                indices = jax.random.permutation(shuffle_key, rollout_steps)

                # lax.fori_loop over minibatches
                def mb_body(mb_idx, inner_carry):
                    p2, os2 = inner_carry
                    start = mb_idx * minibatch_size
                    mb_idx_arr = jax.lax.dynamic_slice(indices, (start,), (minibatch_size,))

                    mb_obs = obs[mb_idx_arr]
                    mb_actions = actions[mb_idx_arr]
                    mb_old_lp = log_probs[mb_idx_arr]
                    mb_adv = advantages[mb_idx_arr]
                    mb_ret = returns[mb_idx_arr]

                    _, grads = ppo_loss_and_grad(p2, mb_obs, mb_actions, mb_old_lp, mb_adv, mb_ret)
                    updates, new_os = optimizer.update(grads, os2, p2)
                    # v10 age-keyed LR: scale the per-step update by the
                    # age-dependent factor. lr_scale ≡ 1.0 when the schedule
                    # is disabled, so this is a no-op for L3 runs.
                    updates = jax.tree_util.tree_map(lambda u: u * lr_scale, updates)
                    new_p = optax.apply_updates(p2, updates)
                    return new_p, new_os

                p, os = jax.lax.fori_loop(0, n_minibatches, mb_body, (p, os))
                return p, os, rng

            new_params, new_opt, _ = jax.lax.fori_loop(
                0, ppo_epochs, epoch_body, (params, opt_state, rng_key)
            )
            return new_params, new_opt, jnp.int32(0)

        def no_update(_):
            return params, opt_state, ptr

        new_params, new_opt, new_ptr = jax.lax.cond(
            should_update, do_update, no_update, None
        )
        return new_params, new_opt, new_ptr

    # vmap over all agent slots
    batched_ppo = jax.vmap(single_agent_ppo)

    # donate_argnums=(0, 1, 8): policy_params, opt_states, and rollout_ptrs
    # each have an output of identical shape/dtype that semantically replaces
    # them at the call site (state.replace in _maybe_fire_ppo). Donation lets
    # XLA reuse the input buffers in-place instead of allocating fresh ones,
    # cutting peak HBM during the PPO update by ~policy+opt-state size. The
    # rollout_* buffers (idx 2-7) are NOT donated — they're carried forward in
    # the new SimState and read by the next sim_step.
    @functools.partial(jax.jit, donate_argnums=(0, 1, 8))
    def maybe_ppo_update_all(policy_params, opt_states, rollout_obs, rollout_actions,
                             rollout_log_probs, rollout_rewards, rollout_values,
                             rollout_dones, rollout_ptrs, is_active, rng_keys,
                             ages, species):
        """Run conditional PPO update on all agent slots.

        Returns: (new_params, new_opt_states, new_ptrs)
        """
        return batched_ppo(
            policy_params, opt_states,
            rollout_obs, rollout_actions, rollout_log_probs,
            rollout_rewards, rollout_values, rollout_dones,
            rollout_ptrs, is_active, rng_keys, ages, species,
        )

    return maybe_ppo_update_all


# ---------------------------------------------------------------------------
# LSTM PPO with truncated BPTT (Axis 4)
# ---------------------------------------------------------------------------

def build_ppo_update_fn_lstm(config):
    """Build a JIT-compiled PPO update function for LSTM policies.

    Uses truncated BPTT: the rollout (1024 steps) is split into chunks
    of lstm_chunk_length (default 128). Each chunk is processed sequentially,
    with hidden state carried forward but gradients stopped at chunk boundaries.

    Returns a function with the same interface as build_ppo_update_fn but with
    an additional init_hidden argument.
    """
    from src.policy import LSTMPolicyNetwork

    lstm_hidden_size = config["lstm_hidden_size"]
    hidden_size = config["policy_hidden_size"]
    obs_dim = config["obs_dim"]
    lr = config["lr"]
    adam_eps = config["adam_eps"]
    clip_eps = config["clip_epsilon"]
    entropy_coef = config["entropy_coef"]
    vf_coef = config.get("vf_coef", 0.5)
    gamma = config["gamma"]
    gae_lambda = config["gae_lambda"]
    rollout_steps = config["rollout_steps"]
    ppo_epochs = config["ppo_epochs"]
    chunk_length = config.get("lstm_chunk_length", 128)
    n_chunks = rollout_steps // chunk_length

    # v10 age-keyed LR + per-species multiplier (mirror of the MLP path).
    lr_sched_enable = bool(config.get("lr_schedule_enable", False))
    lr_initial = float(config.get("lr_schedule_initial", lr))
    lr_final = float(config.get("lr_schedule_final", lr))
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
        return species_mult * (lr_at_age / lr)

    net = LSTMPolicyNetwork(
        lstm_hidden_size=lstm_hidden_size,
        hidden_size=hidden_size,
        action_dim=2,
    )
    optimizer = optax.adam(learning_rate=lr, eps=adam_eps)

    def _chunk_loss_and_final_h(params, chunk_obs, chunk_actions, chunk_old_lp,
                                chunk_adv, chunk_ret, init_carry):
        """Compute PPO loss over one chunk and return final hidden state."""
        def loss_fn(p):
            def scan_step(carry, obs):
                new_carry, mean, log_std, value = net.apply(p, carry, obs)
                return new_carry, (mean, log_std, value)

            final_carry, (all_means, all_log_stds, all_values) = jax.lax.scan(
                scan_step, init_carry, chunk_obs,
            )

            stds = jnp.exp(all_log_stds)
            clamped = jnp.clip(chunk_actions, -19.99, 79.99)
            raw_actions = jnp.log((clamped + 20.0) / (80.0 - clamped))

            new_log_probs = -0.5 * jnp.sum(
                jnp.log(2 * jnp.pi) + 2 * all_log_stds
                + ((raw_actions - all_means) / stds) ** 2,
                axis=-1,
            )

            ratio = jnp.exp(new_log_probs - chunk_old_lp)
            surr1 = ratio * chunk_adv
            surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * chunk_adv
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

            value_loss = 0.5 * jnp.mean((all_values - chunk_ret) ** 2)
            entropy = jnp.mean(
                0.5 * jnp.sum(1.0 + jnp.log(2 * jnp.pi) + 2 * all_log_stds, axis=-1)
            )

            total_loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy
            return total_loss, final_carry

        (loss, final_carry), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return loss, grads, final_carry

    def single_agent_ppo_lstm(params, opt_state, obs, actions, log_probs,
                              rewards, values, dones, ptr, is_active,
                              rng_key, init_hidden_packed, age, species):
        """PPO update for one LSTM agent. Called via vmap."""
        should_update = is_active & (ptr >= rollout_steps)
        lr_scale = _lr_scale(age, species)

        def do_update(_):
            # GAE (same as MLP — computed from stored values)
            advantages, returns = _compute_gae_jax(
                rewards, values, dones, 0.0, gamma, gae_lambda
            )
            advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

            # Reshape into chunks: (n_chunks, chunk_length, ...)
            obs_chunks = obs.reshape(n_chunks, chunk_length, -1)
            act_chunks = actions.reshape(n_chunks, chunk_length, -1)
            old_lp_chunks = log_probs.reshape(n_chunks, chunk_length)
            adv_chunks = advantages.reshape(n_chunks, chunk_length)
            ret_chunks = returns.reshape(n_chunks, chunk_length)

            init_carry = (init_hidden_packed[0], init_hidden_packed[1])

            def epoch_body(epoch_idx, carry):
                p, os, rng = carry

                def chunk_body(chunk_idx, inner_carry):
                    p2, os2, h = inner_carry

                    c_obs = jax.lax.dynamic_slice(
                        obs_chunks, (chunk_idx, 0, 0),
                        (1, chunk_length, obs_dim),
                    ).squeeze(0)
                    c_act = jax.lax.dynamic_slice(
                        act_chunks, (chunk_idx, 0, 0),
                        (1, chunk_length, 2),
                    ).squeeze(0)
                    c_old_lp = jax.lax.dynamic_slice(
                        old_lp_chunks, (chunk_idx, 0),
                        (1, chunk_length),
                    ).squeeze(0)
                    c_adv = jax.lax.dynamic_slice(
                        adv_chunks, (chunk_idx, 0),
                        (1, chunk_length),
                    ).squeeze(0)
                    c_ret = jax.lax.dynamic_slice(
                        ret_chunks, (chunk_idx, 0),
                        (1, chunk_length),
                    ).squeeze(0)

                    h_detached = jax.lax.stop_gradient(h)
                    _, grads, final_h = _chunk_loss_and_final_h(
                        p2, c_obs, c_act, c_old_lp, c_adv, c_ret, h_detached
                    )

                    updates, new_os = optimizer.update(grads, os2, p2)
                    updates = jax.tree_util.tree_map(lambda u: u * lr_scale, updates)
                    new_p = optax.apply_updates(p2, updates)

                    return new_p, new_os, final_h

                p, os, _ = jax.lax.fori_loop(
                    0, n_chunks, chunk_body, (p, os, init_carry)
                )
                return p, os, rng

            new_params, new_opt, _ = jax.lax.fori_loop(
                0, ppo_epochs, epoch_body, (params, opt_state, rng_key)
            )
            return new_params, new_opt, jnp.int32(0)

        def no_update(_):
            return params, opt_state, ptr

        new_params, new_opt, new_ptr = jax.lax.cond(
            should_update, do_update, no_update, None,
        )
        return new_params, new_opt, new_ptr

    batched_ppo_lstm = jax.vmap(single_agent_ppo_lstm)

    # Same donation rationale as the MLP variant: indices 0, 1, 8 each have
    # an output replacing them at the call site. init_hiddens (idx 11) is
    # NOT donated — it's read elsewhere on agent reset.
    @functools.partial(jax.jit, donate_argnums=(0, 1, 8))
    def maybe_ppo_update_all_lstm(policy_params, opt_states, rollout_obs, rollout_actions,
                                  rollout_log_probs, rollout_rewards, rollout_values,
                                  rollout_dones, rollout_ptrs, is_active, rng_keys,
                                  init_hiddens, ages, species):
        """Run conditional LSTM PPO update on all agent slots.

        Returns: (new_params, new_opt_states, new_ptrs)
        """
        return batched_ppo_lstm(
            policy_params, opt_states,
            rollout_obs, rollout_actions, rollout_log_probs,
            rollout_rewards, rollout_values, rollout_dones,
            rollout_ptrs, is_active, rng_keys,
            init_hiddens, ages, species,
        )

    return maybe_ppo_update_all_lstm
