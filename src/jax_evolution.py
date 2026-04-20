"""
jax_evolution.py
----------------
Pure-JAX mutation and offspring spawning. Replaces scipy.stats.t with
a JAX-native Student's t sampler.

Student's t(df=2, scale=0.4):
  t = Normal(0,1) / sqrt(Chi2(2) / 2)
  where Chi2(2) = -2 * log(Uniform(0,1))

This produces the correct heavy-tailed distribution matching the K&D 2025
paper's mutation specification.
"""

import jax
import jax.numpy as jnp

from src.policy import init_policy


def sample_students_t(rng_key, shape, df=2.0, scale=0.4):
    """Sample from Student's t distribution using JAX primitives.

    t(df) = Normal(0,1) / sqrt(Chi2(df) / df)

    For df=2: Chi2(2) = -2 * log(Uniform), so:
    t(2) = Normal(0,1) / sqrt(-log(Uniform))
    """
    k1, k2 = jax.random.split(rng_key)
    normal = jax.random.normal(k1, shape=shape)
    uniform = jax.random.uniform(k2, shape=shape, minval=1e-7, maxval=1.0)

    # Chi2(df) via sum of squared normals for general df,
    # but for df=2: Chi2(2) = -2*log(U) (exponential distribution)
    chi2 = -2.0 * jnp.log(uniform)
    t_sample = normal / jnp.sqrt(chi2 / df)

    return t_sample * scale


def mutate_genome_jax(parent_genome, rng_key, config):
    """Mutate reward weights using Student's t(df=2, scale=0.4), clipped to ±100.

    Pure JAX — no scipy dependency. JIT-compatible.
    """
    df = config.get("mutation_df", 2.0)
    scale = config.get("mutation_scale", 0.4)
    clip_val = config.get("weight_clip", 100.0)

    delta = sample_students_t(rng_key, parent_genome.shape, df=df, scale=scale)
    child = jnp.clip(parent_genome + delta, -clip_val, clip_val)
    return child


def spawn_offspring_jax(sim_state, parent_slot, new_slot, rng_key, config):
    """Spawn one offspring into new_slot from parent_slot. Operates on SimState arrays.

    Returns updated SimState with the new agent activated at new_slot.
    Caller is responsible for updating parent energy (energy share).
    """
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)

    world_size = config["world_size"]
    spawn_spread = config["spawn_spread"]
    prey_radius = config["prey_radius"]
    pred_radius = config["predator_radius"]
    energy_share_ratio = config["energy_share_ratio"]

    parent_species = sim_state.species[parent_slot]
    parent_energy = sim_state.energies[parent_slot]
    parent_genome = sim_state.reward_weights[parent_slot]
    parent_pos = sim_state.phyjax_stated.get("circle").p.xy[parent_slot]
    parent_id = sim_state.agent_ids[parent_slot]

    # Child position: N(parent_pos, spawn_spread), clamped to world
    child_radius = jnp.where(parent_species == 0, prey_radius, pred_radius)
    margin = child_radius * 2
    child_pos = parent_pos + jax.random.normal(k1, (2,)) * spawn_spread
    child_pos = jnp.clip(child_pos, margin, world_size - margin)

    # Child angle: uniform
    child_angle = jax.random.uniform(k2, minval=-jnp.pi, maxval=jnp.pi)

    # Child genome: mutated from parent
    child_genome = mutate_genome_jax(parent_genome, k3, config)

    # Child energy: parent shares
    child_energy = parent_energy * energy_share_ratio

    # Child policy: fresh initialization
    child_params, child_opt = init_policy(k4, config)

    # New agent ID
    child_id = sim_state.next_agent_id

    # --- Write into SimState arrays at new_slot ---
    import phyjax2d as pj

    # Update physics circle state
    circle = sim_state.phyjax_stated.get("circle")
    new_xy = circle.p.xy.at[new_slot].set(child_pos)
    new_angle = circle.p.angle.at[new_slot].set(child_angle)
    new_vel_xy = circle.v.xy.at[new_slot].set(jnp.zeros(2))
    new_vel_ang = circle.v.angle.at[new_slot].set(0.0)
    new_is_active_phys = circle.is_active.at[new_slot].set(True)

    circle = circle.replace(
        p=pj.Position(angle=new_angle, xy=new_xy),
        v=pj.Velocity(angle=new_vel_ang, xy=new_vel_xy),
        is_active=new_is_active_phys,
    )
    new_stated = sim_state.phyjax_stated.replace(circle=circle)

    # Update SoA arrays
    import jax.tree_util as jtu

    new_params = jtu.tree_map(
        lambda stack, single: stack.at[new_slot].set(single),
        sim_state.policy_params, child_params,
    )
    new_opt = jtu.tree_map(
        lambda stack, single: stack.at[new_slot].set(single),
        sim_state.policy_opt_states, child_opt,
    )

    # Compute act_ratio for child
    pred_ratio = (pred_radius ** 2) / (prey_radius ** 2)
    child_act_ratio = jnp.where(parent_species == 1, pred_ratio, 1.0)

    return sim_state.replace(
        is_active=sim_state.is_active.at[new_slot].set(True),
        species=sim_state.species.at[new_slot].set(parent_species),
        agent_ids=sim_state.agent_ids.at[new_slot].set(child_id),
        parent_ids=sim_state.parent_ids.at[new_slot].set(parent_id),
        ages=sim_state.ages.at[new_slot].set(0),
        energies=sim_state.energies.at[new_slot].set(child_energy),
        reward_weights=sim_state.reward_weights.at[new_slot].set(child_genome),
        policy_params=new_params,
        policy_opt_states=new_opt,
        rollout_obs=sim_state.rollout_obs.at[new_slot].set(0.0),
        rollout_actions=sim_state.rollout_actions.at[new_slot].set(0.0),
        rollout_log_probs=sim_state.rollout_log_probs.at[new_slot].set(0.0),
        rollout_rewards=sim_state.rollout_rewards.at[new_slot].set(0.0),
        rollout_values=sim_state.rollout_values.at[new_slot].set(0.0),
        rollout_dones=sim_state.rollout_dones.at[new_slot].set(False),
        rollout_ptrs=sim_state.rollout_ptrs.at[new_slot].set(0),
        radii=sim_state.radii.at[new_slot].set(child_radius),
        act_ratio=sim_state.act_ratio.at[new_slot].set(jnp.array([child_act_ratio])),
        obs_buffer=sim_state.obs_buffer.at[new_slot].set(0.0),
        lstm_hidden=sim_state.lstm_hidden.at[new_slot].set(0.0),
        rollout_init_hidden=sim_state.rollout_init_hidden.at[new_slot].set(0.0),
        phyjax_stated=new_stated,
        next_agent_id=sim_state.next_agent_id + 1,
    )
