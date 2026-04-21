"""
jax_lifecycle.py
----------------
JIT-compiled energy updates, death/birth processing operating on SimState.

All functions are pure JAX — no Python loops or mutable objects.
Species-conditional parameters use jnp.where for branchless computation.
"""

import jax
import jax.numpy as jnp

from src.jax_evolution import spawn_offspring_jax


# ---------------------------------------------------------------------------
# Energy updates
# ---------------------------------------------------------------------------

def update_energies_jax(sim_state, prey_n_eaten, pred_catch_slots, pred_n_catches,
                        all_actions, config):
    """Vectorized energy update for all agents. Pure JAX.

    Args:
        sim_state: SimState
        prey_n_eaten: (max_agents,) int32 — food eaten per prey
        pred_catch_slots: (max_agents, max_catches) int32 — caught prey slot indices
        pred_n_catches: (max_agents,) int32 — catches per predator
        all_actions: (max_agents, 2) — actions taken (sigmoid-scaled)
        config: dict
    """
    e_food = config["prey_e_food"]
    c_b = config["prey_c_b"]
    c_a = config["prey_c_a"]
    d_b = config["predator_d_b"]
    d_a = config["predator_d_a"]
    eta = config["predator_eta"]
    cap = config.get("energy_capacity", 1000.0)

    action_norms = jnp.linalg.norm(all_actions, axis=1)  # (max_agents,)

    # Prey food gain
    prey_gain = prey_n_eaten.astype(jnp.float32) * e_food

    # Predator food gain: sum of eta * caught_prey_energy
    # pred_catch_slots has -1 for empty; gather energies safely
    safe_slots = jnp.clip(pred_catch_slots, 0, sim_state.energies.shape[0] - 1)
    caught_energies = sim_state.energies[safe_slots]  # (max_agents, max_catches)
    caught_valid = pred_catch_slots >= 0               # (max_agents, max_catches) bool
    pred_gain = eta * jnp.sum(caught_energies * caught_valid, axis=1)  # (max_agents,)

    # Species-conditional update
    is_prey = sim_state.species == 0
    food_gain = jnp.where(is_prey, prey_gain, pred_gain)
    cost_a = jnp.where(is_prey, c_a, d_a)
    cost_b = jnp.where(is_prey, c_b, d_b)

    delta = food_gain - cost_a * action_norms - cost_b
    new_energies = jnp.minimum(sim_state.energies + delta, cap)

    # Only update active agents
    new_energies = jnp.where(sim_state.is_active, new_energies, sim_state.energies)
    new_ages = jnp.where(sim_state.is_active, sim_state.ages + 1, sim_state.ages)

    return sim_state.replace(energies=new_energies, ages=new_ages)


# ---------------------------------------------------------------------------
# Death and birth processing
# ---------------------------------------------------------------------------

def _batch_hazard_prob_jax(ages, energies, species, config):
    """Vectorized hazard probability. Pure JAX."""
    kappa_h = config["kappa_h"]
    alpha_e = config["alpha_e"]
    beta_h = config["beta_h"]

    alpha_t = jnp.where(species == 0, config["alpha_t_prey"], config["alpha_t_pred"])
    beta_t = jnp.where(species == 0, config["beta_t_prey"], config["beta_t_pred"])

    energy_term = 1.0 - 1.0 / (1.0 + alpha_e * jnp.exp(jnp.clip(-beta_h * energies, -700, 700)))
    age_term = alpha_t * jnp.exp(jnp.clip(beta_t * ages.astype(jnp.float32), -700, 700))
    h = kappa_h * energy_term * age_term
    return jnp.clip(h, 0.0, 1.0)


def _batch_birth_prob_jax(energies, species, config):
    """Vectorized birth probability. Pure JAX."""
    kappa_b = config["kappa_b"]
    beta_b = config["beta_b"]
    zeta = jnp.where(species == 0, config["zeta_b_prey"], config["zeta_b_pred"])

    exponent = jnp.clip(zeta - beta_b * energies, -700, 700)
    return kappa_b / (1.0 + jnp.exp(exponent))


def process_births_and_deaths_jax(sim_state, config):
    """Process deaths and births for all agents. Pure JAX, JIT-compatible.

    Deaths: energy < 0 OR random < hazard_prob.
    Births: random < birth_prob AND under population cap.
    Uses lax.scan for births (fixed max_births_per_step iterations).

    Returns updated SimState with deaths deactivated and newborns activated.
    """
    max_births_per_step = 20  # fixed scan length
    max_agents = sim_state.is_active.shape[0]
    energy_share_ratio = config["energy_share_ratio"]
    prey_cap = config.get("prey_cap", 450)
    predator_cap = config.get("predator_cap", 50)

    # --- Death processing ---
    rng, death_key = jax.random.split(sim_state.rng_key)
    death_randoms = jax.random.uniform(death_key, shape=(max_agents,))

    h_all = _batch_hazard_prob_jax(sim_state.ages, sim_state.energies, sim_state.species, config)
    dead_mask = sim_state.is_active & ((sim_state.energies < 0) | (death_randoms < h_all))

    # Deactivate dead agents
    new_is_active = sim_state.is_active & ~dead_mask

    # Zero dead agents' rollout pointers and velocities
    new_rollout_ptrs = jnp.where(dead_mask, 0, sim_state.rollout_ptrs)

    # Update physics is_active
    circle = sim_state.phyjax_stated.get("circle")
    import phyjax2d as pj
    new_phys_active = circle.is_active & ~dead_mask
    new_vel_xy = jnp.where(dead_mask[:, None], 0.0, circle.v.xy)
    new_vel_ang = jnp.where(dead_mask, 0.0, circle.v.angle)
    circle = circle.replace(
        v=pj.Velocity(angle=new_vel_ang, xy=new_vel_xy),
        is_active=new_phys_active,
    )
    new_stated = sim_state.phyjax_stated.replace(circle=circle)

    # D21: bump cumulative death counter for hazard/starvation deaths here.
    # (D20 catch-deaths are counted separately in sim_step_core.)
    deaths_this_step = jnp.sum(dead_mask.astype(jnp.int32))
    sim_state = sim_state.replace(
        is_active=new_is_active,
        rollout_ptrs=new_rollout_ptrs,
        phyjax_stated=new_stated,
        cum_deaths=sim_state.cum_deaths + deaths_this_step,
    )

    # --- Birth processing ---
    rng, birth_key = jax.random.split(rng)
    birth_randoms = jax.random.uniform(birth_key, shape=(max_agents,))

    b_all = _batch_birth_prob_jax(sim_state.energies, sim_state.species, config)
    wants_birth = sim_state.is_active & (birth_randoms < b_all)

    # Population caps
    prey_count = jnp.sum(new_is_active & (sim_state.species == 0))
    pred_count = jnp.sum(new_is_active & (sim_state.species == 1))

    prey_under_cap = prey_count < prey_cap
    pred_under_cap = pred_count < predator_cap

    # Filter by cap: prey parents only if under prey cap, etc.
    can_birth = wants_birth & jnp.where(
        sim_state.species == 0, prey_under_cap, pred_under_cap
    )

    # Collect parent slots that will birth (take first max_births_per_step).
    # We then pad out to max_births_per_step with the max_agents sentinel so
    # the lax.scan below always gets a fixed-length input even for tiny
    # configs where max_agents < max_births_per_step (e.g. unit tests).
    birth_indices = jnp.where(can_birth, jnp.arange(max_agents), max_agents)
    sorted_birth = jnp.sort(birth_indices)  # valid births first, max_agents padding at end
    pad_tail = jnp.full(max_births_per_step, max_agents, dtype=jnp.int32)
    sorted_birth = jnp.concatenate([sorted_birth, pad_tail])
    parent_slots = sorted_birth[:max_births_per_step]

    # lax.scan over potential births
    rng, spawn_rng = jax.random.split(rng)
    spawn_keys = jax.random.split(spawn_rng, max_births_per_step)

    def do_one_birth(carry, inputs):
        state, prey_ct, pred_ct = carry
        parent_slot, spawn_key = inputs

        is_valid = parent_slot < max_agents
        parent_species = state.species[parent_slot]

        # Check cap at this point (caps may have been reached by earlier births in scan)
        still_under_cap = jnp.where(
            parent_species == 0,
            prey_ct < prey_cap,
            pred_ct < predator_cap,
        )
        should_spawn = is_valid & still_under_cap & state.is_active[parent_slot]

        # Find first free slot WITHIN the parent's species range (D19 fix).
        # Prey slots are [0, prey_cap); predator slots are [prey_cap, max_agents).
        # Physics body radii are bound to slot index at builder time, so an
        # offspring must be spawned in a slot whose physics body matches its
        # species — otherwise it ends up in a wrong-sized body.
        slot_idx = jnp.arange(max_agents)
        in_species_range = jnp.where(
            parent_species == 0,
            slot_idx < prey_cap,
            slot_idx >= prey_cap,
        )
        inactive_slots = jnp.where(
            ~state.is_active & in_species_range,
            slot_idx,
            max_agents,
        )
        first_free = jnp.min(inactive_slots)
        has_slot = first_free < max_agents

        do_spawn = should_spawn & has_slot

        # Spawn offspring (always executes due to JIT, but result discarded if !do_spawn)
        new_state = spawn_offspring_jax(state, parent_slot, first_free, spawn_key, config)

        # Parent energy reduction
        parent_energy = state.energies[parent_slot]
        new_parent_energy = parent_energy - parent_energy * energy_share_ratio
        new_state = new_state.replace(
            energies=new_state.energies.at[parent_slot].set(
                jnp.where(do_spawn, new_parent_energy, state.energies[parent_slot])
            )
        )

        # Select between spawned and original state
        state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(do_spawn, new, old) if hasattr(new, 'shape') else new,
            new_state, state,
        )

        # Update counters
        prey_ct = prey_ct + jnp.where(do_spawn & (parent_species == 0), 1, 0)
        pred_ct = pred_ct + jnp.where(do_spawn & (parent_species == 1), 1, 0)

        return (state, prey_ct, pred_ct), None

    (sim_state, _, _), _ = jax.lax.scan(
        do_one_birth,
        (sim_state, prey_count, pred_count),
        (parent_slots, spawn_keys),
    )

    sim_state = sim_state.replace(rng_key=rng)
    return sim_state
