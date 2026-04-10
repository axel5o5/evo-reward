"""
jax_food.py
-----------
JIT-compiled eating detection and food management operating on SimState arrays.

All functions are pure JAX — no Python loops, dicts, or sets.
Fixed-size arrays with boolean masks throughout.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp


def _wrap_angle(a):
    """Wrap angle difference to [-pi, pi]."""
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def check_eating_jax(sim_state, config):
    """Detect eating events for all agents. Pure JAX, JIT-compatible.

    Returns:
        prey_n_eaten: (max_agents,) int32 — food count eaten by each agent (0 for non-prey/inactive)
        pred_catch_slots: (max_agents, max_catches) int32 — slot indices of caught prey (-1 = empty)
        pred_n_catches: (max_agents,) int32 — number of prey caught per predator
        food_eaten_mask: (food_max,) bool — which food items were eaten
    """
    max_catches = 5  # max prey a single predator can catch per step

    # Extract state
    circle = sim_state.phyjax_stated.get("circle")
    positions = circle.p.xy
    angles = circle.p.angle
    is_active = sim_state.is_active
    species = sim_state.species
    energies = sim_state.energies
    food_pos = sim_state.food_positions
    food_active = sim_state.food_active

    max_agents = positions.shape[0]
    food_max = food_pos.shape[0]

    prey_radius = config["prey_radius"]
    fov_half = math.radians(config["proximity_fov_deg"]) / 2.0
    mouth_range_min = config.get("predator_mouth_range_min", 40.0)
    mouth_range_max = config.get("predator_mouth_range_max", 80.0)
    mouth_half_rad = math.radians(config.get("predator_mouth_deg", 60.0)) / 2.0

    # ---- Prey eating food ----
    # Pairwise distances: (max_agents, food_max)
    diffs_food = food_pos[None, :, :] - positions[:, None, :]    # (A, F, 2)
    dists_food = jnp.linalg.norm(diffs_food, axis=-1)            # (A, F)

    # Contact: dist <= prey_radius
    contact = dists_food <= prey_radius

    # FOV check
    angles_to_food = jnp.arctan2(diffs_food[:, :, 1], diffs_food[:, :, 0])
    angle_diffs_food = jnp.abs(_wrap_angle(angles_to_food - angles[:, None]))
    in_fov = angle_diffs_food <= fov_half

    # Valid: active prey + active food + contact + in FOV
    prey_mask = (species == 0) & is_active                        # (A,)
    valid_food = contact & in_fov & prey_mask[:, None] & food_active[None, :]  # (A, F)

    # Deduplication: each food eaten by nearest valid prey
    valid_food_dists = jnp.where(valid_food, dists_food, jnp.inf)
    nearest_prey = jnp.argmin(valid_food_dists, axis=0)           # (F,)
    food_has_eater = jnp.min(valid_food_dists, axis=0) < jnp.inf  # (F,)
    food_eaten_mask = food_has_eater & food_active                 # (F,)

    # Count food eaten per prey agent
    # Use scatter-add: for each eaten food, add 1 to nearest_prey's count
    prey_n_eaten = jnp.zeros(max_agents, dtype=jnp.int32)
    clamped_nearest = jnp.clip(nearest_prey, 0, max_agents - 1)
    prey_n_eaten = prey_n_eaten.at[clamped_nearest].add(
        food_eaten_mask.astype(jnp.int32)
    )
    # Zero out non-prey agents
    prey_n_eaten = jnp.where(prey_mask, prey_n_eaten, 0)

    # ---- Predator catching prey ----
    # Pairwise distances: (max_agents, max_agents) — pred×prey
    diffs_agents = positions[None, :, :] - positions[:, None, :]  # (A, A, 2)
    dists_agents = jnp.linalg.norm(diffs_agents, axis=-1)         # (A, A)

    # Mouth range check
    in_range = (dists_agents >= mouth_range_min) & (dists_agents <= mouth_range_max)

    # Mouth angle check
    angles_to_agents = jnp.arctan2(diffs_agents[:, :, 1], diffs_agents[:, :, 0])
    angle_diffs_agents = jnp.abs(_wrap_angle(angles_to_agents - angles[:, None]))
    in_mouth = angle_diffs_agents <= mouth_half_rad

    # Valid: active predator × active prey + in range + in mouth
    pred_mask = (species == 1) & is_active                        # (A,)
    prey_target_mask = (species == 0) & is_active                 # (A,)
    valid_catch = in_range & in_mouth & pred_mask[:, None] & prey_target_mask[None, :]  # (A, A)

    # Deduplication: each prey caught by nearest valid predator
    valid_catch_dists = jnp.where(valid_catch, dists_agents, jnp.inf)
    nearest_pred = jnp.argmin(valid_catch_dists, axis=0)          # (A,) — which pred catches each prey
    prey_is_caught = jnp.min(valid_catch_dists, axis=0) < jnp.inf  # (A,)
    prey_caught_mask = prey_is_caught & prey_target_mask          # (A,)

    # Build per-predator catch arrays: (max_agents, max_catches) slot indices
    # For each predator, collect up to max_catches caught prey slots
    pred_catch_slots = jnp.full((max_agents, max_catches), -1, dtype=jnp.int32)
    pred_n_catches = jnp.zeros(max_agents, dtype=jnp.int32)

    # For each caught prey, scatter into the catching predator's catch list
    # This is tricky to do purely in JAX without loops. Use lax.scan over caught prey.
    caught_prey_indices = jnp.where(prey_caught_mask, jnp.arange(max_agents), -1)

    def add_catch(carry, prey_slot):
        catch_slots, n_catches = carry
        is_valid = prey_slot >= 0
        pred_slot = jnp.where(is_valid, nearest_pred[prey_slot], 0)
        catch_idx = jnp.where(is_valid, n_catches[pred_slot], 0)
        can_add = is_valid & (catch_idx < max_catches)

        catch_slots = jnp.where(
            can_add,
            catch_slots.at[pred_slot, catch_idx].set(prey_slot),
            catch_slots,
        )
        n_catches = jnp.where(
            can_add,
            n_catches.at[pred_slot].add(1),
            n_catches,
        )
        return (catch_slots, n_catches), None

    (pred_catch_slots, pred_n_catches), _ = jax.lax.scan(
        add_catch,
        (pred_catch_slots, pred_n_catches),
        caught_prey_indices,
    )

    # Zero out non-predator agents
    pred_n_catches = jnp.where(pred_mask, pred_n_catches, 0)

    return prey_n_eaten, pred_catch_slots, pred_n_catches, food_eaten_mask


def remove_eaten_food_jax(sim_state, food_eaten_mask):
    """Remove eaten food by deactivating slots. Pure JAX."""
    new_food_active = sim_state.food_active & ~food_eaten_mask
    n_eaten = jnp.sum(food_eaten_mask)
    new_internal = jnp.maximum(sim_state.food_internal - n_eaten, 0.0)
    return sim_state.replace(
        food_active=new_food_active,
        food_internal=new_internal,
    )


def regenerate_food_jax(sim_state, config):
    """Regenerate food up to max capacity. Pure JAX, JIT-compatible.

    Uses lax.scan over max_regen_per_step iterations, each activating one food slot.
    """
    growth_rate = config["food_growth_rate"]
    food_max = config["food_max"]
    max_regen = config.get("food_max_regen_per_step", 10)
    world_size = float(config["world_size"])

    new_internal = jnp.minimum(sim_state.food_internal + growth_rate, jnp.float32(food_max))
    target_count = jnp.floor(new_internal).astype(jnp.int32)
    current_count = jnp.sum(sim_state.food_active).astype(jnp.int32)
    n_to_spawn = jnp.minimum(jnp.maximum(target_count - current_count, 0), max_regen)

    def spawn_one_food(carry, i):
        food_pos, food_active, rng, n_spawned = carry
        should_spawn = i < n_to_spawn

        rng, pos_key = jax.random.split(rng)
        new_pos = jax.random.uniform(pos_key, (2,), minval=0.0, maxval=world_size)

        # Find first inactive food slot
        inactive_indices = jnp.where(~food_active, jnp.arange(food_max), food_max)
        first_inactive = jnp.min(inactive_indices)
        has_slot = first_inactive < food_max

        do_spawn = should_spawn & has_slot
        food_pos = jnp.where(do_spawn, food_pos.at[first_inactive].set(new_pos), food_pos)
        food_active = jnp.where(do_spawn, food_active.at[first_inactive].set(True), food_active)
        n_spawned = n_spawned + do_spawn.astype(jnp.int32)

        return (food_pos, food_active, rng, n_spawned), None

    rng, food_rng = jax.random.split(sim_state.rng_key)
    (new_food_pos, new_food_active, _, _), _ = jax.lax.scan(
        spawn_one_food,
        (sim_state.food_positions, sim_state.food_active, food_rng, jnp.int32(0)),
        jnp.arange(max_regen),
    )

    return sim_state.replace(
        food_positions=new_food_pos,
        food_active=new_food_active,
        food_internal=new_internal,
        rng_key=rng,
    )
