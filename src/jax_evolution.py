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
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree

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


def mutate_pytree_genome_jax(parent_params, rng_key, scale, clip_val, df=2.0):
    """JAX-native Student's-t mutation for any flattenable PyTree genome.

    Used by both Axis 1 (MLP, scale = mlp_mutation_scale) and Axis 3
    (temporal, scale = temporal_mutation_scale). Stays in JIT so the spawn
    path avoids a host sync per birth.
    """
    flat, unflatten_fn = ravel_pytree(parent_params)
    delta = sample_students_t(rng_key, flat.shape, df=df, scale=scale)
    child_flat = jnp.clip(flat + delta, -clip_val, clip_val)
    return unflatten_fn(child_flat)


def mutate_mlp_genome_jax(parent_params, rng_key, config):
    """Mutate an MLP reward genome (Flax PyTree). See mutate_pytree_genome_jax."""
    return mutate_pytree_genome_jax(
        parent_params, rng_key,
        scale=config["mlp_mutation_scale"],
        clip_val=config["mlp_weight_clip"],
        df=config.get("mutation_df", 2.0),
    )


def mutate_residual_genome_jax(parent_params, rng_key, config):
    """Mutate a residual MLP reward genome (axis 1 v4).

    Residual MLPs are zero-initialized and grow via mutation if useful.
    Uses a smaller mutation scale than the full-MLP variant since residual
    perturbations should accumulate gradually on top of the linear baseline.
    """
    return mutate_pytree_genome_jax(
        parent_params, rng_key,
        scale=config.get("residual_mutation_scale", 0.03),
        clip_val=config.get("residual_weight_clip", 5.0),
        df=config.get("mutation_df", 2.0),
    )


def mutate_temporal_genome_jax(parent_params, rng_key, config):
    """Mutate a temporal reward genome (Flax PyTree, k*4 → h → h → 1)."""
    return mutate_pytree_genome_jax(
        parent_params, rng_key,
        scale=config["temporal_mutation_scale"],
        clip_val=config["temporal_weight_clip"],
        df=config.get("mutation_df", 2.0),
    )


def mutate_poly_genome_jax(parent_poly, rng_key, config):
    """Mutate a polynomial residual genome (shape (10,) ndarray).

    Mirrors mutate_residual_genome_jax but operates on a flat (10,) array
    instead of a Flax PyTree — no ravel_pytree needed. Uses Student's t
    (df=2) with `poly_mutation_scale` (default 0.05) and clips to
    ±`poly_weight_clip` (default 5.0).
    """
    scale = config.get("poly_mutation_scale", 0.05)
    clip_val = config.get("poly_weight_clip", 5.0)
    df = config.get("mutation_df", 2.0)
    delta = sample_students_t(rng_key, parent_poly.shape, df=df, scale=scale)
    return jnp.clip(parent_poly + delta, -clip_val, clip_val)


def spawn_offspring_jax(sim_state, parent_slot, new_slot, rng_key, config, energy_bonus=0.0):
    """Spawn one offspring into new_slot from parent_slot. Operates on SimState arrays.

    Returns updated SimState with the new agent activated at new_slot.
    Caller is responsible for updating parent energy (energy share + bonus).

    energy_bonus: scalar (jnp or Python float). Additive birth-time energy
    bonus applied to the child on top of the K&D `parent_E * energy_share_ratio`
    transfer. The caller computes bonus from scaffold engagement (see
    process_births_and_deaths_jax). Default 0.0 → paper-faithful K&D mechanics.
    """
    k1, k2, k3, k4, k5 = jax.random.split(rng_key, 5)

    from src.environment import world_bounds
    world_x, world_y = world_bounds(config)
    spawn_spread = config["spawn_spread"]
    prey_radius = config["prey_radius"]
    pred_radius = config["predator_radius"]
    energy_share_ratio = config["energy_share_ratio"]
    reward_type = config.get("reward_type", "linear")

    parent_species = sim_state.species[parent_slot]
    parent_energy = sim_state.energies[parent_slot]
    parent_genome = sim_state.reward_weights[parent_slot]
    parent_pos = sim_state.phyjax_stated.get("circle").p.xy[parent_slot]
    parent_id = sim_state.agent_ids[parent_slot]

    # Child position: N(parent_pos, spawn_spread), clamped to world
    child_radius = jnp.where(parent_species == 0, prey_radius, pred_radius)
    margin = child_radius * 2
    child_pos = parent_pos + jax.random.normal(k1, (2,)) * spawn_spread
    child_pos = jnp.clip(
        child_pos,
        jnp.array([margin, margin]),
        jnp.array([world_x - margin, world_y - margin]),
    )

    # Child angle: uniform
    child_angle = jax.random.uniform(k2, minval=-jnp.pi, maxval=jnp.pi)

    # Child genome: linear weights mutate for "linear" and the residual
    # variant ("linear_plus_mlp_residual"), since both actually read the
    # 4-vector. For "mlp" / "temporal" the linear field is unused by the
    # reward dispatch, so leaving it equal to the parent's value keeps
    # logged linear stats quiet instead of showing a drifting "ghost
    # evolution" the simulator is ignoring.
    if reward_type in ("linear", "linear_plus_mlp_residual", "linear_plus_poly"):
        child_genome = mutate_genome_jax(parent_genome, k3, config)
    else:
        child_genome = parent_genome

    # Child energy: parent shares (K&D) + scaffold-aware additive bonus.
    # Bonus is computed by caller based on scaffold engagement; defaults to
    # 0.0 → paper-faithful K&D 0.4 transfer.
    child_energy = parent_energy * energy_share_ratio + energy_bonus

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

    # MLP / temporal / poly genome mutation + scatter into the per-agent
    # stacked params. Branch on the static config flag so the entire path
    # stays in JIT. Linear runs leave both empty dicts as no-ops; non-poly
    # runs leave reward_poly_params (a zero-filled array) untouched.
    new_reward_mlp_params = sim_state.reward_mlp_params
    new_reward_temporal_params = sim_state.reward_temporal_params
    new_reward_poly_params = sim_state.reward_poly_params
    if reward_type == "mlp":
        parent_mlp = jtu.tree_map(
            lambda leaf: leaf[parent_slot], sim_state.reward_mlp_params
        )
        child_mlp = mutate_mlp_genome_jax(parent_mlp, k5, config)
        new_reward_mlp_params = jtu.tree_map(
            lambda stack, single: stack.at[new_slot].set(single),
            sim_state.reward_mlp_params, child_mlp,
        )
    elif reward_type == "linear_plus_mlp_residual":
        parent_residual = jtu.tree_map(
            lambda leaf: leaf[parent_slot], sim_state.reward_mlp_params
        )
        child_residual = mutate_residual_genome_jax(parent_residual, k5, config)
        new_reward_mlp_params = jtu.tree_map(
            lambda stack, single: stack.at[new_slot].set(single),
            sim_state.reward_mlp_params, child_residual,
        )
    elif reward_type == "temporal":
        parent_temporal = jtu.tree_map(
            lambda leaf: leaf[parent_slot], sim_state.reward_temporal_params
        )
        child_temporal = mutate_temporal_genome_jax(parent_temporal, k5, config)
        new_reward_temporal_params = jtu.tree_map(
            lambda stack, single: stack.at[new_slot].set(single),
            sim_state.reward_temporal_params, child_temporal,
        )
    elif reward_type == "linear_plus_poly":
        parent_poly = sim_state.reward_poly_params[parent_slot]
        child_poly = mutate_poly_genome_jax(parent_poly, k5, config)
        new_reward_poly_params = sim_state.reward_poly_params.at[new_slot].set(child_poly)

    return sim_state.replace(
        is_active=sim_state.is_active.at[new_slot].set(True),
        species=sim_state.species.at[new_slot].set(parent_species),
        agent_ids=sim_state.agent_ids.at[new_slot].set(child_id),
        parent_ids=sim_state.parent_ids.at[new_slot].set(parent_id),
        ages=sim_state.ages.at[new_slot].set(0),
        energies=sim_state.energies.at[new_slot].set(child_energy),
        reward_weights=sim_state.reward_weights.at[new_slot].set(child_genome),
        reward_mlp_params=new_reward_mlp_params,
        reward_temporal_params=new_reward_temporal_params,
        reward_poly_params=new_reward_poly_params,
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
        # Reset the predator-catch cooldown so a newborn predator doesn't
        # inherit its dead predecessor's stale countdown. Without this,
        # a slot that went dormant with eat_timer=9 would leave the
        # newborn unable to catch for 9 steps after birth. For prey
        # slots the timer is never read, so the write is a no-op —
        # harmless to set uniformly.
        predator_eat_timer=sim_state.predator_eat_timer.at[new_slot].set(0),
        phyjax_stated=new_stated,
        next_agent_id=sim_state.next_agent_id + 1,
    )
