"""
jax_sim.py
----------
Unified JIT-compiled simulation step function.

sim_step(sim_state, space) → sim_state

One call = one complete simulation step matching AGENTS.md order:
  1. Observations  2. Policy sample  3. Rollout write
  4. Physics       5. Eating         6. Rewards
  7. Energy        8. Birth/death    9. Food regen
  10. PPO update

All pure JAX. Same code runs on CPU and GPU.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import phyjax2d as pj

from src.policy import PolicyNetwork
from src.observations import _build_obs_fn
from src.jax_food import check_eating_jax, remove_eaten_food_jax, regenerate_food_jax
from src.jax_lifecycle import update_energies_jax, process_births_and_deaths_jax
from src.jax_ppo import build_ppo_update_fn
from src.environment import N_PHYSICS_ITER, CHANNEL_PREY, CHANNEL_PREDATOR


def build_sim_step(config, space):
    """Build the JIT-compiled simulation step function.

    Returns two functions:
        sim_step_core(sim_state) → sim_state  — steps 1-9, JIT-compiled
        ppo_update_fn — for step 10, called from Python for ready agents only

    PPO is separated because vmap+lax.cond over all 500 slots is too expensive
    on CPU (traces both branches for all slots). Instead, the runner checks
    which agents have full buffers and calls PPO only for those.
    On GPU, you can optionally inline PPO into sim_step for full JIT.
    """
    max_agents = config["prey_cap"] + config["predator_cap"]
    food_max = config["food_max"]
    obs_dim = config["obs_dim"]
    rollout_steps = config["rollout_steps"]
    n_sensors = config["n_proximity_sensors"]
    n_channels = config.get("n_proximity_channels", 4)
    F_max = config["max_motor_norm"]

    # Pre-build the JIT-compiled observation function
    obs_fn = _build_obs_fn(config, max_agents, food_max)

    # D29: sensor aggregation for the prey/pred reward stimuli. emevo's
    # cf_predator.py defaults to `sensor_agg_type="mean"` (averages over the
    # 32 proximity bins); our old code was hardcoded to `max`, giving a
    # stronger fear/chase signal than the paper actually used. Config-gated
    # with "mean" as the paper-faithful default.
    _agg_type = config.get("sensor_agg_type", "mean")
    assert _agg_type in ("mean", "max"), f"sensor_agg_type must be mean|max, got {_agg_type!r}"
    _sensor_agg = jnp.mean if _agg_type == "mean" else jnp.max

    # D30 ablation control: reward proximity stimulus can be computed from
    # pre-step observations (`all_obs`) or post-step observations (`all_obs_post`).
    # emevo computes reward from obs_t1.sensor (post-step). Keep that as default,
    # but expose a toggle so we can run one-variable ablations cleanly.
    _reward_obs_timing = config.get("reward_obs_timing", "post_step")
    assert _reward_obs_timing in ("pre_step", "post_step"), (
        f"reward_obs_timing must be pre_step|post_step, got {_reward_obs_timing!r}"
    )
    _use_post_obs_for_reward = _reward_obs_timing == "post_step"

    # Pre-build the policy network for action sampling
    net = PolicyNetwork(hidden_size=config["policy_hidden_size"], action_dim=2)

    # Pre-build the PPO update function (used outside JIT by the runner)
    ppo_update_fn = build_ppo_update_fn(config)

    # Reward dispatch (Axis 1). Linear keeps the K&D formula with
    # [1.0, 0.01, 0.1, 0.1] coefs; MLP eats RAW stimuli and reads its
    # per-agent params from sim_state.reward_mlp_params. Selected at build
    # time so the JIT body branches on a Python-static flag — no lax.cond
    # needed because each run uses a single reward_type.
    reward_type = config.get("reward_type", "linear")
    if reward_type == "linear":
        _linear_coefs = jnp.array([1.0, 0.01, 0.1, 0.1])

        def _compute_rewards(sim_state, stimuli):
            return jnp.sum(sim_state.reward_weights * stimuli * _linear_coefs, axis=1)
    elif reward_type == "mlp":
        from src.reward import compute_mlp_reward

        def _compute_rewards(sim_state, stimuli):
            return jax.vmap(compute_mlp_reward)(sim_state.reward_mlp_params, stimuli)
    else:
        raise ValueError(
            f"reward_type {reward_type!r} not yet wired through jax_sim "
            "(linear/mlp supported in Phase A; temporal lands in Phase B)"
        )

    # Physics stepper
    # D19 fix: also return a per-substep "did-touch" bool array, reduced via
    # max across substeps. emevo's circle_foraging.py::nstep uses the same
    # pattern: `contact.penetration >= 0.0` per substep, then jnp.max(axis=0).
    # This lets check_eating_jax detect contacts that happened DURING a
    # substep but were separated by the velocity solver before the step end
    # — exactly the catches our old post-step distance check was missing.
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
            body, (stated, solver), None, length=N_PHYSICS_ITER
        )
        contacts = jnp.max(nstep_contacts, axis=0)  # (n_pair,) bool
        return stated, solver, contacts

    @jax.jit
    def sim_step_core(sim_state):
        """Steps 1-9 of the simulation. Pure JAX, JIT-compiled.

        PPO (step 10) is handled separately by the runner.
        """

        # === 1. Observations (vectorized) ===
        circle = sim_state.phyjax_stated.get("circle")
        obs_state = {
            "positions": circle.p.xy,
            "angles": circle.p.angle,
            "velocities_xy": circle.v.xy,
            "velocities_ang": circle.v.angle,
            "is_active": sim_state.is_active,
            "species": sim_state.species,
            "radii": sim_state.radii,
            "energies": sim_state.energies,
            "food_positions": sim_state.food_positions,
            "food_active": sim_state.food_active,
            "max_agents": max_agents,
        }
        all_obs = obs_fn(obs_state)  # (max_agents, obs_dim)

        # === 2. Policy sample (vmap over all agents) ===
        rng, sample_key = jax.random.split(sim_state.rng_key)
        all_rngs = jax.random.split(sample_key, max_agents)

        def sample_one(params, obs, rng_k):
            mean, log_std, value = net.apply(params, obs)
            std = jnp.exp(log_std)
            noise = jax.random.normal(rng_k, shape=mean.shape)
            raw_action = mean + std * noise
            log_prob = -0.5 * jnp.sum(
                jnp.log(2 * jnp.pi) + 2 * log_std + ((raw_action - mean) / std) ** 2
            )
            action = 100.0 * jax.nn.sigmoid(raw_action) - 20.0
            return action, log_prob, value

        all_actions, all_log_probs, all_values = jax.vmap(sample_one)(
            sim_state.policy_params, all_obs, all_rngs,
        )
        all_actions = jnp.where(sim_state.is_active[:, None], all_actions, 0.0)

        # === 3. Write to rollout buffers ===
        ptrs = sim_state.rollout_ptrs
        safe_ptrs = jnp.clip(ptrs, 0, rollout_steps - 1)
        agent_idx = jnp.arange(max_agents)

        new_rollout_obs = sim_state.rollout_obs.at[agent_idx, safe_ptrs].set(all_obs)
        new_rollout_actions = sim_state.rollout_actions.at[agent_idx, safe_ptrs].set(all_actions)
        new_rollout_log_probs = sim_state.rollout_log_probs.at[agent_idx, safe_ptrs].set(all_log_probs)
        new_rollout_values = sim_state.rollout_values.at[agent_idx, safe_ptrs].set(all_values)

        # === 4. Physics step ===
        action_arr = all_actions * sim_state.act_ratio
        f1_raw = action_arr[:, 0:1]
        f2_raw = action_arr[:, 1:2]
        f1 = jnp.concatenate([jnp.zeros_like(f1_raw), f1_raw], axis=1)
        f2 = jnp.concatenate([jnp.zeros_like(f2_raw), f2_raw], axis=1)

        new_stated, new_solver, contacts_flat = physics_step(
            sim_state.phyjax_stated, sim_state.phyjax_solver,
            sim_state.act_p1, sim_state.act_p2, f1, f2,
        )

        # Build (A, A) symmetric "did-touch" matrix from per-pair contact flags.
        # emevo's nstep helper in gecco2026 circle_foraging.py does the same:
        # per-substep penetration>=0 → max-reduce → get_contact_mat.
        contact_mat = space.get_contact_mat("circle", "circle", contacts_flat)

        sim_state = sim_state.replace(
            phyjax_stated=new_stated, phyjax_solver=new_solver,
            rollout_obs=new_rollout_obs, rollout_actions=new_rollout_actions,
            rollout_log_probs=new_rollout_log_probs, rollout_values=new_rollout_values,
        )

        # === 5. Check eating ===
        (prey_n_eaten, pred_caught_energy, pred_n_catches, food_eaten_mask,
         new_predator_eat_timer, prey_caught_mask) = check_eating_jax(
             sim_state, config, contact_mat
         )
        sim_state = remove_eaten_food_jax(sim_state, food_eaten_mask)
        sim_state = sim_state.replace(predator_eat_timer=new_predator_eat_timer)

        # D20: deactivate caught prey same-step. Without this the predator
        # harvests +eta*E but the prey keeps existing and the cooldown lets
        # the same pair pairwise-feed the predator forever — see docs/emevo-diff.md.
        # IMPORTANT: we do NOT zero caught prey energies here, because
        # update_energies_jax (step 7 below) reads `caught_energies =
        # sim_state.energies[safe_slots]` to compute `pred_gain = eta * E`.
        # Zeroing pre-transfer would make predators starve despite catching.
        # Inactive slots are skipped elsewhere via is_active masks, so
        # leaving the stale energy there is harmless; a future birth at
        # that slot overwrites it via spawn_offspring_jax.
        circle = sim_state.phyjax_stated.get("circle")
        new_phys_active = circle.is_active & ~prey_caught_mask
        new_vel_xy = jnp.where(prey_caught_mask[:, None], 0.0, circle.v.xy)
        new_vel_ang = jnp.where(prey_caught_mask, 0.0, circle.v.angle)
        circle = circle.replace(
            v=pj.Velocity(angle=new_vel_ang, xy=new_vel_xy),
            is_active=new_phys_active,
        )
        # D21: cumulative event counters — read by the runner each log
        # interval to populate progress.json `events` block. Catches count
        # here; post-catch deaths also bump cum_deaths so the total death
        # counter lines up.
        catches_this_step = jnp.sum(prey_caught_mask.astype(jnp.int32))
        feedings_this_step = jnp.sum(food_eaten_mask.astype(jnp.int32))
        sim_state = sim_state.replace(
            is_active=sim_state.is_active & ~prey_caught_mask,
            phyjax_stated=sim_state.phyjax_stated.replace(circle=circle),
            # Reset rollout ptr to 0 so the freed slot's buffer can fill cleanly
            # when a future birth reuses it.
            rollout_ptrs=jnp.where(prey_caught_mask, 0, sim_state.rollout_ptrs),
            cum_catches=sim_state.cum_catches + catches_this_step,
            cum_deaths=sim_state.cum_deaths + catches_this_step,
            cum_feedings=sim_state.cum_feedings + feedings_this_step,
        )

        # === 6. Compute rewards (vectorized) ===
        # D30 default: reward uses POST-physics, POST-catch observations
        # (emevo's `obs_t1.sensor` semantics). For controlled ablations, we can
        # optionally force PRE-step obs via reward_obs_timing="pre_step".
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

        # D29: aggregate proximity stimuli via mean (emevo default) or max
        # per sensor_agg_type config key. Captured in closure at build time.
        prox_all = reward_obs[:, :n_sensors * n_channels].reshape(max_agents, n_sensors, n_channels)
        s_prey = _sensor_agg(jnp.clip(prox_all[:, :, CHANNEL_PREY], 0.0), axis=1)
        s_pred = _sensor_agg(jnp.clip(prox_all[:, :, CHANNEL_PREDATOR], 0.0), axis=1)
        motor_norms = jnp.linalg.norm(all_actions, axis=1) / F_max

        n_eaten_reward = jnp.where(
            sim_state.species == 0,
            prey_n_eaten.astype(jnp.float32),
            pred_n_catches.astype(jnp.float32),
        )

        stimuli = jnp.stack([n_eaten_reward, motor_norms, s_prey, s_pred], axis=1)
        all_rewards = _compute_rewards(sim_state, stimuli)
        all_rewards = jnp.where(sim_state.is_active, all_rewards, 0.0)

        new_rollout_rewards = sim_state.rollout_rewards.at[agent_idx, safe_ptrs].set(all_rewards)
        # D23: mark caught prey's last rollout slot as terminal. Without this
        # the GAE bootstrap over a dead agent's trajectory would treat its
        # transitions as non-terminal. Only affects correctness if PPO ever
        # consumes a dead-agent rollout (guarded by is_active gate today),
        # but semantically right. Hazard-death done flags are written further
        # down after process_births_and_deaths_jax.
        new_rollout_dones = sim_state.rollout_dones.at[agent_idx, safe_ptrs].set(prey_caught_mask)
        new_ptrs = jnp.where(sim_state.is_active, ptrs + 1, ptrs)

        sim_state = sim_state.replace(
            rollout_rewards=new_rollout_rewards, rollout_dones=new_rollout_dones,
            rollout_ptrs=new_ptrs,
        )

        # === 7. Update energies ===
        sim_state = update_energies_jax(
            sim_state, prey_n_eaten, pred_caught_energy, pred_n_catches,
            all_actions, config,
        )

        # === 8. Process births and deaths ===
        # Pass the pre-step safe_ptrs so hazard-dead agents get their last
        # rollout slot flagged as terminal too (D23).
        sim_state = process_births_and_deaths_jax(
            sim_state, config, rollout_ptrs_for_done=safe_ptrs,
        )

        # === 9. Regenerate food ===
        sim_state = regenerate_food_jax(sim_state, config)

        sim_state = sim_state.replace(rng_key=rng, step=sim_state.step + 1)
        return sim_state

    return sim_step_core, ppo_update_fn
