"""
sac_runtime.py
--------------
Runtime hooks for using SacState during a simulation step. Three
per-env-step primitives, each JIT-compiled:

  1. sample_actions(sac_state, all_obs, rng) -> actions
        Forward each agent's actor and reparam-sample. Returns
        (max_agents, 2) action vectors in the [-20, 80] sigmoid-scaled
        range. Inactive agents get computed actions too (cheaper than
        masking inside vmap); the physics step is responsible for
        ignoring them.

  2. write_transitions(sac_state, obs, actions, rewards, next_obs,
                       dones, is_active) -> sac_state
        Append one (s, a, r, s', d) tuple per active agent to the ring
        at slot `replay_ptr[i] % capacity`. Bumps ptr (mod capacity) and
        increments size (capped at capacity). Inactive slots are
        unchanged.

  3. step_update(sac_state, is_active, ages, species, rng) -> sac_state
        Wraps build_sac_update_fn from src/jax_sac.py. Runs one SAC
        gradient step per agent — actor + Q1 + Q2 + alpha + Polyak
        target sync. Slots with replay_size < min_replay_size or
        is_active=False are skipped.

Use:
    runtime = build_sac_runtime(config)
    actions = runtime["sample_actions"](sac_state, obs, rng)
    sac_state = runtime["write_transitions"](
        sac_state, obs, actions, rewards, next_obs, dones, is_active,
    )
    sac_state = runtime["step_update"](sac_state, is_active, ages, species, rng)

The runtime dict is built once and reused across steps; each entry is
its own JIT compilation.
"""

import functools

import jax
import jax.numpy as jnp

from src.sac_networks import SACActorNetwork, reparam_sample
from src.jax_sac import build_sac_update_fn


def build_sac_runtime(config: dict) -> dict:
    """Build the three JIT-compiled runtime functions. Returns a dict of
    {sample_actions, write_transitions, step_update}."""
    hidden_size = int(config["policy_hidden_size"])
    action_dim = 2
    capacity = int(config.get("sac_replay_capacity", 4096))

    actor_net = SACActorNetwork(hidden_size=hidden_size, action_dim=action_dim)
    sac_update_fn = build_sac_update_fn(config)

    # -----------------------------------------------------------------------
    # 1. Action sampling
    # -----------------------------------------------------------------------

    def _sample_one(actor_params_slot, obs_slot, rng_slot):
        mean, log_std = actor_net.apply(actor_params_slot, obs_slot)
        action, _log_prob, _raw = reparam_sample(mean, log_std, rng_slot)
        return action

    @jax.jit
    def sample_actions(sac_state, all_obs, rng):
        """Returns (max_agents, 2) action array — one per agent."""
        n_agents = all_obs.shape[0]
        rng_keys = jax.random.split(rng, n_agents)
        return jax.vmap(_sample_one)(sac_state.actor_params, all_obs, rng_keys)

    # -----------------------------------------------------------------------
    # 2. Replay-buffer write
    # -----------------------------------------------------------------------
    #
    # Per-slot write: each agent writes to its own row at its own ptr
    # column. Inactive slots leave their row + ptr + size untouched.
    # vmap over (row, ptr, transition_value, is_active) gives per-agent
    # writes without a Python loop.

    def _write_row(row, ptr, value, active):
        return jax.lax.cond(
            active,
            lambda _: row.at[ptr].set(value),
            lambda _: row,
            None,
        )

    @jax.jit
    def write_transitions(sac_state, obs, actions, rewards, next_obs, dones, is_active):
        """Append (s, a, r, s', d) for each active agent. Ring-buffer
        write: new_ptr = (ptr + 1) mod capacity; size = min(size+1, cap)."""
        # Each per-field call vmaps over the agent axis. The .at[ptr].set
        # then writes to that agent's ring at that agent's ptr.
        new_replay_obs = jax.vmap(_write_row)(
            sac_state.replay_obs, sac_state.replay_ptr, obs, is_active,
        )
        new_replay_action = jax.vmap(_write_row)(
            sac_state.replay_action, sac_state.replay_ptr, actions, is_active,
        )
        new_replay_reward = jax.vmap(_write_row)(
            sac_state.replay_reward, sac_state.replay_ptr, rewards, is_active,
        )
        new_replay_next_obs = jax.vmap(_write_row)(
            sac_state.replay_next_obs, sac_state.replay_ptr, next_obs, is_active,
        )
        new_replay_done = jax.vmap(_write_row)(
            sac_state.replay_done, sac_state.replay_ptr, dones, is_active,
        )

        # Ptr / size bookkeeping: only bump for active slots.
        new_ptr = jnp.where(
            is_active,
            (sac_state.replay_ptr + 1) % capacity,
            sac_state.replay_ptr,
        )
        new_size = jnp.where(
            is_active,
            jnp.minimum(sac_state.replay_size + 1, capacity),
            sac_state.replay_size,
        )

        return sac_state.replace(
            replay_obs=new_replay_obs,
            replay_action=new_replay_action,
            replay_reward=new_replay_reward,
            replay_next_obs=new_replay_next_obs,
            replay_done=new_replay_done,
            replay_ptr=new_ptr,
            replay_size=new_size,
        )

    # -----------------------------------------------------------------------
    # 3. Step update
    # -----------------------------------------------------------------------

    def step_update(sac_state, is_active, ages, species, rng):
        """Run one per-agent SAC gradient step. Wraps the JIT-compiled
        update fn from src/jax_sac.py and threads its outputs back into
        a fresh SacState. Not @jax.jit at the wrapper level because the
        inner fn already is (and uses donate_argnums on its params/opt
        inputs — donation can't be nested cleanly with a wrapper-level
        jit).
        """
        n_agents = is_active.shape[0]
        rng_keys = jax.random.split(rng, n_agents)
        (new_actor, new_q1, new_q2,
         new_q1_t, new_q2_t, new_log_alpha,
         new_actor_opt, new_q1_opt, new_q2_opt, new_alpha_opt) = sac_update_fn(
            sac_state.actor_params, sac_state.q1_params, sac_state.q2_params,
            sac_state.q1_target_params, sac_state.q2_target_params, sac_state.log_alpha,
            sac_state.actor_opt_state, sac_state.q1_opt_state,
            sac_state.q2_opt_state, sac_state.alpha_opt_state,
            sac_state.replay_obs, sac_state.replay_action, sac_state.replay_reward,
            sac_state.replay_next_obs, sac_state.replay_done,
            sac_state.replay_size, is_active, rng_keys, ages, species,
        )
        return sac_state.replace(
            actor_params=new_actor,
            q1_params=new_q1, q2_params=new_q2,
            q1_target_params=new_q1_t, q2_target_params=new_q2_t,
            log_alpha=new_log_alpha,
            actor_opt_state=new_actor_opt,
            q1_opt_state=new_q1_opt, q2_opt_state=new_q2_opt,
            alpha_opt_state=new_alpha_opt,
        )

    return {
        "sample_actions": sample_actions,
        "write_transitions": write_transitions,
        "step_update": step_update,
    }
