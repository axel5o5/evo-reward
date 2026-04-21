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


def check_eating_jax(sim_state, config, contact_mat):
    """Detect eating events for all agents. Pure JAX, JIT-compatible.

    Predator catching semantics (matches emevo's circle_foraging_with_predator
    per gecco2026 branch; see docs/emevo-diff.md D18/D19):
      * **Contact**: predator and prey touched *at any physics substep*, per
        `contact_mat` (an (A, A) bool produced upstream from phyjax2d's
        per-substep `contact.penetration >= 0`, max-reduced across substeps,
        then expanded via `space.get_contact_mat`). The previous post-step
        distance check missed mid-step contacts that the velocity solver
        separated before the step ended — D19 fix.
      * **Mouth**: the prey must fall into one of the tactile bins listed in
        config['predator_mouth_tactile_bins'] (default [0, 1, 17] = 60°
        front arc, same bin layout as our tactile sensors in observations.py).
      * **Cooldown**: each predator has its own eat-timer that decrements
        per step and resets to config['predator_eat_interval'] on a catch.
        A predator can only catch when timer <= 0.

    Args:
        sim_state: SimState
        config: dict
        contact_mat: (A, A) bool — per-pair "did touch during this physics
            step" matrix, from space.get_contact_mat("circle", "circle", ...).

    Returns:
        prey_n_eaten: (max_agents,) int32 — food count eaten by each agent
        pred_catch_slots: (max_agents, max_catches) int32 — caught prey slot indices (-1 = empty)
        pred_n_catches: (max_agents,) int32 — number of prey caught per predator
        food_eaten_mask: (food_max,) bool — which food items were eaten
        new_predator_eat_timer: (max_agents,) int32 — updated cooldown state
        prey_caught_mask: (max_agents,) bool — prey slots that were caught this step
            (caller must deactivate these; see D20 in docs/emevo-diff.md). Without
            this the predator gets the energy bonus but the prey keeps existing
            and can be re-caught every cooldown, which lets predators eat for
            free forever.
    """
    max_catches = 5  # max prey a single predator can catch per step (per eat event)

    # Extract state
    circle = sim_state.phyjax_stated.get("circle")
    positions = circle.p.xy
    angles = circle.p.angle
    is_active = sim_state.is_active
    species = sim_state.species
    energies = sim_state.energies
    food_pos = sim_state.food_positions
    food_active = sim_state.food_active
    predator_eat_timer = sim_state.predator_eat_timer             # (A,) int32

    max_agents = positions.shape[0]
    food_max = food_pos.shape[0]

    prey_radius = config["prey_radius"]
    # Food has its own physics radius (4.0 in emevo defaults). Contact with
    # prey happens when center-to-center distance <= prey_r + food_r,
    # matching phyjax2d's circle-circle contact formula (D22 fix).
    # Fallback to 0.0 preserves legacy behavior for configs that pre-date
    # the addition of food_radius.
    food_radius = float(config.get("food_radius", 0.0))
    food_contact_dist = prey_radius + food_radius
    fov_half = math.radians(config["proximity_fov_deg"]) / 2.0

    # Tactile-bin mouth geometry — see emevo predator_mouth_range = [0, 1, 17]
    n_tactile_bins = config["n_tactile_sensors"]
    tactile_spacing_rad = math.radians(config["tactile_spacing_deg"])
    mouth_bin_indices = tuple(config.get("predator_mouth_tactile_bins", [0, 1, 17]))
    eat_interval = config.get("predator_eat_interval", 10)

    # ---- Prey eating food ----
    # Pairwise distances: (max_agents, food_max)
    diffs_food = food_pos[None, :, :] - positions[:, None, :]    # (A, F, 2)
    dists_food = jnp.linalg.norm(diffs_food, axis=-1)            # (A, F)

    # Contact: dist <= prey_radius + food_radius (circle-circle physics contact).
    # Pre-D22 we used only prey_radius (effective contact area ≈ 51% of emevo's).
    contact = dists_food <= food_contact_dist

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

    # ---- Predator catching prey (emevo-faithful: contact + tactile-bin + cooldown) ----
    # Pairwise distances: (max_agents, max_agents) — pred×prey.
    # Still needed for tactile-bin angle computation and for nearest-pred
    # deduplication; just no longer the source of truth for "did touch."
    diffs_agents = positions[None, :, :] - positions[:, None, :]  # (A, A, 2)
    dists_agents = jnp.linalg.norm(diffs_agents, axis=-1)         # (A, A)

    # Contact comes from phyjax2d's per-substep penetration check (D19).
    in_contact = contact_mat

    # Tactile-bin assignment — nearest bin to the angle-from-heading of each prey
    angles_to_agents = jnp.arctan2(diffs_agents[:, :, 1], diffs_agents[:, :, 0])
    # Bin centers: 0°, 20°, 40°, ... around the agent
    bin_centers = jnp.arange(n_tactile_bins) * tactile_spacing_rad
    bin_half_width = tactile_spacing_rad / 2.0
    # Angle of prey relative to predator's heading (A pred, A prey, B bins)
    angle_rel = _wrap_angle(angles_to_agents[:, :, None] - angles[:, None, None]
                            - bin_centers[None, None, :])
    nearest_bin = jnp.argmin(jnp.abs(angle_rel), axis=-1)         # (A, A)
    # In-bin only if prey is within ±half_width of nearest bin center
    in_bin = jnp.take_along_axis(
        jnp.abs(angle_rel), nearest_bin[..., None], axis=-1
    )[..., 0] <= bin_half_width                                    # (A, A)

    # In mouth: nearest bin is one of the mouth bins
    is_mouth_bin = jnp.zeros_like(nearest_bin, dtype=bool)
    for b in mouth_bin_indices:
        is_mouth_bin = is_mouth_bin | (nearest_bin == b)
    in_mouth = in_bin & is_mouth_bin                               # (A, A)

    # Cooldown: predator can only catch when timer <= 0
    can_eat = predator_eat_timer <= 0                              # (A,)

    # Valid: active predator × active prey + in contact + in mouth + can_eat
    pred_mask = (species == 1) & is_active                        # (A,)
    prey_target_mask = (species == 0) & is_active                 # (A,)
    valid_catch = (in_contact & in_mouth
                   & pred_mask[:, None] & prey_target_mask[None, :]
                   & can_eat[:, None])                            # (A, A)

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

    # Cooldown update (mirrors emevo's circle_foraging_with_predator step):
    #   timer = eat_interval if caught_this_step else max(timer - 1, floor)
    # Countdown is clipped at -1 so it doesn't underflow for agents that never
    # catch (semantics-preserving: "<=0 means ready"). For non-predators the
    # value is irrelevant; we still decrement so the array layout stays uniform.
    caught_this_step = pred_n_catches > 0
    decremented = jnp.maximum(predator_eat_timer - 1, -1)
    new_predator_eat_timer = jnp.where(
        caught_this_step, jnp.int32(eat_interval), decremented
    )

    return (prey_n_eaten, pred_catch_slots, pred_n_catches, food_eaten_mask,
            new_predator_eat_timer, prey_caught_mask)


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
