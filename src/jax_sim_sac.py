"""
jax_sim_sac.py
--------------
SAC-aware simulation step. Parallel to src/jax_sim.py's sim_step_core
but with three changes:

  - Action sampling uses the SAC actor (state-dependent Gaussian, reparam
    sample, sigmoid squash to [-20, 80]) instead of the PPO policy net.
  - Rollout-buffer writes are replaced with **replay-buffer writes**.
    Each active agent appends one (s, a, r, s', done) tuple to its
    per-agent ring at slot `replay_ptr[i] % capacity`.
  - Per-step gradient bookkeeping: the SAC gradient step is NOT done
    inline here. The caller invokes it via the sac_runtime fn after
    each sim_step. This keeps donate_argnums in build_sac_update_fn
    clean and lets the orchestrator decide when to update.

Returns the updated (sim_state, sac_state) plus the immediate-step
sampled actions (for any caller-side logging — physics already saw them).
The replay write is performed INSIDE this function, so by the time the
caller gets sac_state back, the latest (s, a, r, s', d) is already
queued.

Birth handling:
  - When a slot is reused by birth (a newborn takes over a dead agent's
    slot), this step fully resets the slot — both the replay buffer
    AND all networks (actor / Q1 / Q2 / target Q1 / target Q2) plus
    their optimizer states and log_alpha. Matches PPO's "fresh policy
    at birth" semantics.
  - Implementation: at every step we vmap-init fresh networks for ALL
    slots (most are discarded), then jnp.where-select between
    old-network and fresh-network based on birth_mask. The per-step
    init cost is small (each network is ~13k params, zero-state Adam
    init) and amortizes across the rest of the step's work.

Reused unchanged from jax_sim.py:
  - Physics step (force application + n-substep velocity solver).
  - Eating / catch logic (check_eating_jax, remove_eaten_food_jax).
  - Reward computation (linear or MLP variants).
  - Energy update (with optional DDM scaling).
  - Birth/death processing (process_births_and_deaths_jax).
  - Food regeneration.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import phyjax2d as pj

from src.sac_networks import SACActorNetwork, reparam_sample
from src.sac_state import _init_single_agent_nets
from src.observations import _build_obs_fn
from src.jax_food import check_eating_jax, remove_eaten_food_jax, regenerate_food_jax
from src.jax_lifecycle import update_energies_jax, process_births_and_deaths_jax
from src.environment import N_PHYSICS_ITER, CHANNEL_PREY, CHANNEL_PREDATOR
import jax.tree_util as jtu


def build_sim_step_sac(config, space):
    """Build the JIT-compiled SAC sim step.

    Returns a single function `sim_step_core_sac(sim_state, sac_state)
    -> (sim_state, sac_state)` that does steps 1-9 of one env-step plus
    a replay-buffer write. The orchestrator is responsible for calling
    the SAC gradient step (sac_runtime["step_update"]) after this.

    Mirrors build_sim_step's option closure (sensor_agg, reward timing,
    reward_type dispatch) so a YAML config can be reused across PPO and
    SAC by just flipping `learner_type`.
    """
    max_agents = config["prey_cap"] + config["predator_cap"]
    food_max = config["food_max"]
    obs_dim = config["obs_dim"]
    n_sensors = config["n_proximity_sensors"]
    n_channels = config.get("n_proximity_channels", 4)
    F_max = config["max_motor_norm"]
    replay_capacity = int(config.get("sac_replay_capacity", 4096))
    rollout_steps = int(config["rollout_steps"])

    # Closure-captured config dict — used by the vmap'd birth-init call.
    # Putting `_init_single_agent_nets` inside vmap means each slot
    # independently runs the actor/critic/alpha init. The output is a
    # stacked pytree {actor_params: (n, ...), ...} that we jnp.where
    # against the existing sac_state values via birth_mask.
    _config_ref = config

    def _fresh_one_agent(rng):
        return _init_single_agent_nets(rng, _config_ref)

    obs_fn = _build_obs_fn(config, max_agents, food_max)

    _agg_type = config.get("sensor_agg_type", "mean")
    _sensor_agg = jnp.mean if _agg_type == "mean" else jnp.max

    _reward_obs_timing = config.get("reward_obs_timing", "post_step")
    _use_post_obs_for_reward = _reward_obs_timing == "post_step"

    actor_net = SACActorNetwork(
        hidden_size=config["policy_hidden_size"],
        action_dim=2,
    )

    # Reward dispatch — copied from jax_sim.build_sim_step so this file
    # stays self-contained and SAC-only edits don't drift back into PPO.
    reward_type = config.get("reward_type", "linear")
    if reward_type == "linear":
        _linear_coefs = jnp.array([1.0, 0.01, 0.1, 0.1])

        def _compute_rewards(sim_state, stimuli):
            r = jnp.sum(sim_state.reward_weights * stimuli * _linear_coefs, axis=1)
            return r, sim_state
    elif reward_type == "mlp":
        from src.reward import compute_mlp_reward

        def _compute_rewards(sim_state, stimuli):
            r = jax.vmap(compute_mlp_reward)(sim_state.reward_mlp_params, stimuli)
            return r, sim_state
    elif reward_type == "linear_plus_mlp_residual":
        from src.reward import compute_residual_reward

        def _compute_rewards(sim_state, stimuli):
            r = jax.vmap(compute_residual_reward)(
                sim_state.reward_weights,
                sim_state.reward_mlp_params,
                stimuli,
            )
            return r, sim_state
    elif reward_type == "temporal":
        from src.reward import compute_temporal_reward

        def _compute_rewards(sim_state, stimuli):
            old = sim_state.obs_buffer
            new_buffer = jnp.concatenate(
                [old[:, 1:], stimuli[:, None, :]], axis=1,
            )
            r = jax.vmap(compute_temporal_reward)(
                sim_state.reward_temporal_params, new_buffer,
            )
            return r, sim_state.replace(obs_buffer=new_buffer)
    else:
        raise ValueError(f"reward_type {reward_type!r} not recognized")

    n_physics_iter = int(config.get("n_physics_iter", N_PHYSICS_ITER))

    @jax.jit
    def physics_step(stated, solver, act_p1, act_p2, f1, f2):
        circle = stated.get("circle")
        circle = circle.apply_force_local(act_p1, f1)
        circle = circle.apply_force_local(act_p2, f2)
        stated = stated.replace(circle=circle)

        def body(carry, _):
            st, sol = carry
            st, sol, contact = pj.step(space, st, sol)
            return (st, sol), contact.penetration >= 0.0

        (stated, solver), nstep_contacts = jax.lax.scan(
            body, (stated, solver), None, length=n_physics_iter
        )
        contacts = jnp.max(nstep_contacts, axis=0)
        return stated, solver, contacts

    # -----------------------------------------------------------------------
    # SAC action sample (one agent's slice — vmapped over slots below)
    # -----------------------------------------------------------------------

    def _sample_one(actor_params_slot, obs_slot, rng_slot):
        mean, log_std = actor_net.apply(actor_params_slot, obs_slot)
        action, _, _ = reparam_sample(mean, log_std, rng_slot)
        return action

    # -----------------------------------------------------------------------
    # Per-slot ring write — vmapped over slots below
    # -----------------------------------------------------------------------

    def _write_row(row, ptr, value, active):
        return jax.lax.cond(
            active,
            lambda _: row.at[ptr].set(value),
            lambda _: row,
            None,
        )

    # -----------------------------------------------------------------------
    # The step
    # -----------------------------------------------------------------------

    @jax.jit
    def sim_step_core_sac(sim_state, sac_state):
        # Snapshot pre-step quantities for the replay write + birth detection.
        pre_is_active = sim_state.is_active
        pre_agent_ids = sim_state.agent_ids

        # === 1. Observations (pre-step, used as `s` in the transition) ===
        circle = sim_state.phyjax_stated.get("circle")
        obs_state = {
            "positions": circle.p.xy,
            "angles": circle.p.angle,
            "velocities_xy": circle.v.xy,
            "velocities_ang": circle.v.angle,
            "is_active": pre_is_active,
            "species": sim_state.species,
            "radii": sim_state.radii,
            "energies": sim_state.energies,
            "food_positions": sim_state.food_positions,
            "food_active": sim_state.food_active,
            "max_agents": max_agents,
        }
        all_obs = obs_fn(obs_state)

        # === 2. SAC action sample (reparameterized; uses sac_state.actor_params) ===
        rng, sample_key = jax.random.split(sim_state.rng_key)
        all_rngs = jax.random.split(sample_key, max_agents)
        all_actions = jax.vmap(_sample_one)(
            sac_state.actor_params, all_obs, all_rngs,
        )
        all_actions = jnp.where(pre_is_active[:, None], all_actions, 0.0)

        # === 3. Physics step ===
        action_arr = all_actions * sim_state.act_ratio
        f1_raw = action_arr[:, 0:1]
        f2_raw = action_arr[:, 1:2]
        f1 = jnp.concatenate([jnp.zeros_like(f1_raw), f1_raw], axis=1)
        f2 = jnp.concatenate([jnp.zeros_like(f2_raw), f2_raw], axis=1)
        new_stated, new_solver, contacts_flat = physics_step(
            sim_state.phyjax_stated, sim_state.phyjax_solver,
            sim_state.act_p1, sim_state.act_p2, f1, f2,
        )
        contact_mat = space.get_contact_mat("circle", "circle", contacts_flat)
        sim_state = sim_state.replace(
            phyjax_stated=new_stated, phyjax_solver=new_solver,
        )

        # === 4. Check eating ===
        (prey_n_eaten, pred_caught_energy, pred_n_catches, food_eaten_mask,
         new_predator_eat_timer, prey_caught_mask) = check_eating_jax(
             sim_state, config, contact_mat
         )
        sim_state = remove_eaten_food_jax(sim_state, food_eaten_mask)
        sim_state = sim_state.replace(predator_eat_timer=new_predator_eat_timer)

        # Deactivate caught prey (D20 logic, same as PPO).
        circle = sim_state.phyjax_stated.get("circle")
        new_phys_active = circle.is_active & ~prey_caught_mask
        new_vel_xy = jnp.where(prey_caught_mask[:, None], 0.0, circle.v.xy)
        new_vel_ang = jnp.where(prey_caught_mask, 0.0, circle.v.angle)
        circle = circle.replace(
            v=pj.Velocity(angle=new_vel_ang, xy=new_vel_xy),
            is_active=new_phys_active,
        )
        catches_this_step = jnp.sum(prey_caught_mask.astype(jnp.int32))
        feedings_this_step = jnp.sum(food_eaten_mask.astype(jnp.int32))
        sim_state = sim_state.replace(
            is_active=sim_state.is_active & ~prey_caught_mask,
            phyjax_stated=sim_state.phyjax_stated.replace(circle=circle),
            cum_catches=sim_state.cum_catches + catches_this_step,
            cum_deaths=sim_state.cum_deaths + catches_this_step,
            cum_feedings=sim_state.cum_feedings + feedings_this_step,
        )

        # === 5. Compute rewards ===
        if _use_post_obs_for_reward:
            circle_post = sim_state.phyjax_stated.get("circle")
            obs_state_post = {
                "positions": circle_post.p.xy,
                "angles": circle_post.p.angle,
                "velocities_xy": circle_post.v.xy,
                "velocities_ang": circle_post.v.angle,
                "is_active": sim_state.is_active,
                "species": sim_state.species,
                "radii": sim_state.radii,
                "energies": sim_state.energies,
                "food_positions": sim_state.food_positions,
                "food_active": sim_state.food_active,
                "max_agents": max_agents,
            }
            reward_obs = obs_fn(obs_state_post)
        else:
            reward_obs = all_obs

        prox_all = reward_obs[:, :n_sensors * n_channels].reshape(
            max_agents, n_sensors, n_channels,
        )
        s_prey = _sensor_agg(jnp.clip(prox_all[:, :, CHANNEL_PREY], 0.0), axis=1)
        s_pred = _sensor_agg(jnp.clip(prox_all[:, :, CHANNEL_PREDATOR], 0.0), axis=1)
        motor_norms = jnp.linalg.norm(all_actions, axis=1) / F_max
        n_eaten_reward = jnp.where(
            sim_state.species == 0,
            prey_n_eaten.astype(jnp.float32),
            pred_n_catches.astype(jnp.float32),
        )
        stimuli = jnp.stack([n_eaten_reward, motor_norms, s_prey, s_pred], axis=1)
        all_rewards, sim_state = _compute_rewards(sim_state, stimuli)
        all_rewards = jnp.where(sim_state.is_active, all_rewards, 0.0)

        # === 6. Update energies ===
        pred_count = jnp.sum(sim_state.is_active & (sim_state.species == 1))
        prey_count = jnp.sum(sim_state.is_active & (sim_state.species == 0))
        sim_state = update_energies_jax(
            sim_state, prey_n_eaten, pred_caught_energy, pred_n_catches,
            all_actions, config, pred_count=pred_count, prey_count=prey_count,
        )

        # === 7. Births and deaths ===
        # `rollout_ptrs_for_done` is required by process_births_and_deaths_jax
        # to mark the dead agent's last PPO rollout slot as terminal. For SAC
        # this is irrelevant (we never read the rollout buffer), but the
        # function still wants the array. Pass zeros — the writes go into a
        # buffer we ignore.
        zero_ptrs = jnp.zeros(max_agents, dtype=jnp.int32)
        sim_state = process_births_and_deaths_jax(
            sim_state, config, rollout_ptrs_for_done=zero_ptrs,
        )

        # === 8. Food regen ===
        sim_state = regenerate_food_jax(sim_state, config)

        # === 9. Compute next_obs (s') for the replay write ===
        circle_final = sim_state.phyjax_stated.get("circle")
        obs_state_final = {
            "positions": circle_final.p.xy,
            "angles": circle_final.p.angle,
            "velocities_xy": circle_final.v.xy,
            "velocities_ang": circle_final.v.angle,
            "is_active": sim_state.is_active,
            "species": sim_state.species,
            "radii": sim_state.radii,
            "energies": sim_state.energies,
            "food_positions": sim_state.food_positions,
            "food_active": sim_state.food_active,
            "max_agents": max_agents,
        }
        all_next_obs = obs_fn(obs_state_final)

        # === 10. Done detection ===
        # A transition's `done` flag is True iff the agent that took the
        # action (pre-step occupant of the slot) is gone post-step. That
        # covers: caught prey (deactivated), hazard/starvation deaths
        # (handled in process_births_and_deaths_jax), and the rare case
        # where the slot was both killed AND immediately repopulated by a
        # birth (then post_agent_ids != pre_agent_ids even though
        # is_active stays True).
        post_is_active = sim_state.is_active
        post_agent_ids = sim_state.agent_ids
        slot_died = pre_is_active & (~post_is_active)
        slot_replaced = pre_is_active & post_is_active & (pre_agent_ids != post_agent_ids)
        done_mask = (slot_died | slot_replaced).astype(jnp.float32)

        # === 11. Replay write — per active pre-step slot, append (s, a, r, s', d) ===
        # Inactive pre-step slots wrote nothing this turn.
        new_replay_obs = jax.vmap(_write_row)(
            sac_state.replay_obs, sac_state.replay_ptr, all_obs, pre_is_active,
        )
        new_replay_action = jax.vmap(_write_row)(
            sac_state.replay_action, sac_state.replay_ptr, all_actions, pre_is_active,
        )
        new_replay_reward = jax.vmap(_write_row)(
            sac_state.replay_reward, sac_state.replay_ptr, all_rewards, pre_is_active,
        )
        new_replay_next_obs = jax.vmap(_write_row)(
            sac_state.replay_next_obs, sac_state.replay_ptr, all_next_obs, pre_is_active,
        )
        new_replay_done = jax.vmap(_write_row)(
            sac_state.replay_done, sac_state.replay_ptr, done_mask, pre_is_active,
        )
        new_ptr = jnp.where(
            pre_is_active,
            (sac_state.replay_ptr + 1) % replay_capacity,
            sac_state.replay_ptr,
        )
        new_size = jnp.where(
            pre_is_active,
            jnp.minimum(sac_state.replay_size + 1, replay_capacity),
            sac_state.replay_size,
        )

        # === 12. Birth-slot full reset ===
        # When a slot is reused by birth (post_active=True AND agent_id
        # changed, OR was inactive and is now active), reset BOTH:
        #   (a) the replay buffer for that slot — newborn shouldn't train
        #       on the dead predecessor's transitions.
        #   (b) the actor / Q1 / Q2 / target Q1 / target Q2 networks +
        #       optimizer states + log_alpha — matches PPO's "fresh
        #       policy at birth" semantics.
        birth_mask = (~pre_is_active & post_is_active) | slot_replaced

        def _zero_if_birth(arr):
            m = birth_mask
            for _ in range(arr.ndim - 1):
                m = m[..., None]
            return jnp.where(m, jnp.zeros_like(arr), arr)

        new_replay_obs = _zero_if_birth(new_replay_obs)
        new_replay_action = _zero_if_birth(new_replay_action)
        new_replay_reward = _zero_if_birth(new_replay_reward)
        new_replay_next_obs = _zero_if_birth(new_replay_next_obs)
        new_replay_done = _zero_if_birth(new_replay_done)
        new_ptr = jnp.where(birth_mask, 0, new_ptr)
        new_size = jnp.where(birth_mask, 0, new_size)

        # (b) Network reset. Vmap-init fresh per-slot networks, then select
        # per-leaf between old (stacked) and fresh (stacked) via birth_mask.
        # Most slots' fresh init is discarded — but the init is cheap (a
        # few small matmul-shaped random draws + zero opt-state) and XLA
        # may fuse parts of it away when birth_mask is all-False.
        rng, birth_rng = jax.random.split(rng)
        birth_keys = jax.random.split(birth_rng, max_agents)
        fresh = jax.vmap(_fresh_one_agent)(birth_keys)

        def _select_per_slot(old_stacked, fresh_stacked):
            m = birth_mask
            for _ in range(old_stacked.ndim - 1):
                m = m[..., None]
            return jnp.where(m, fresh_stacked, old_stacked)

        new_actor = jtu.tree_map(_select_per_slot, sac_state.actor_params, fresh["actor_params"])
        new_q1 = jtu.tree_map(_select_per_slot, sac_state.q1_params, fresh["q1_params"])
        new_q2 = jtu.tree_map(_select_per_slot, sac_state.q2_params, fresh["q2_params"])
        new_q1_t = jtu.tree_map(_select_per_slot, sac_state.q1_target_params, fresh["q1_target_params"])
        new_q2_t = jtu.tree_map(_select_per_slot, sac_state.q2_target_params, fresh["q2_target_params"])
        new_log_alpha = jnp.where(birth_mask, fresh["log_alpha"], sac_state.log_alpha)
        new_actor_opt = jtu.tree_map(_select_per_slot, sac_state.actor_opt_state, fresh["actor_opt_state"])
        new_q1_opt = jtu.tree_map(_select_per_slot, sac_state.q1_opt_state, fresh["q1_opt_state"])
        new_q2_opt = jtu.tree_map(_select_per_slot, sac_state.q2_opt_state, fresh["q2_opt_state"])
        new_alpha_opt = jtu.tree_map(_select_per_slot, sac_state.alpha_opt_state, fresh["alpha_opt_state"])

        sac_state = sac_state.replace(
            actor_params=new_actor,
            q1_params=new_q1, q2_params=new_q2,
            q1_target_params=new_q1_t, q2_target_params=new_q2_t,
            log_alpha=new_log_alpha,
            actor_opt_state=new_actor_opt,
            q1_opt_state=new_q1_opt, q2_opt_state=new_q2_opt,
            alpha_opt_state=new_alpha_opt,
            replay_obs=new_replay_obs,
            replay_action=new_replay_action,
            replay_reward=new_replay_reward,
            replay_next_obs=new_replay_next_obs,
            replay_done=new_replay_done,
            replay_ptr=new_ptr,
            replay_size=new_size,
        )

        # === 13. Replay-recorder compatibility: write the action that
        # drove this step into sim_state.rollout_actions, mod-indexed by
        # rollout_ptrs. The replay recorder reads
        # rollout_actions[arange, (ptrs-1) % rollout_size] each frame to
        # surface "the action this agent just took." SAC never reads this
        # buffer for learning; it's a passive log.
        ptrs = sim_state.rollout_ptrs
        safe_ptrs = jnp.clip(ptrs % rollout_steps, 0, rollout_steps - 1)
        agent_idx = jnp.arange(max_agents)
        new_rollout_actions = sim_state.rollout_actions.at[agent_idx, safe_ptrs].set(all_actions)
        new_rollout_ptrs = jnp.where(
            pre_is_active, (ptrs + 1) % rollout_steps, ptrs,
        )
        # On birth or catch, reset that slot's rollout ptr to 0 (matches
        # PPO's reset-on-life-event behavior so the recorder reads a
        # clean window for the newborn).
        new_rollout_ptrs = jnp.where(birth_mask, 0, new_rollout_ptrs)
        sim_state = sim_state.replace(
            rollout_actions=new_rollout_actions,
            rollout_ptrs=new_rollout_ptrs,
            rng_key=rng,
            step=sim_state.step + 1,
        )
        return sim_state, sac_state

    return sim_step_core_sac
