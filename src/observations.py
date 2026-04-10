"""
observations.py
---------------
Vectorized observation computation for all agents simultaneously using JAX.

Replaces the per-agent Python loop (get_observation → compute_proximity_sensors →
compute_tactile_sensors) with a single JIT-compiled batched call. All operations
are pure JAX, so the same code runs on CPU or GPU.

Key technique: jax.vmap over the observer axis, with scatter-min
(jnp.ndarray.at[bin_idx].min(dist)) for binning closest objects per sensor.

Observation layout (205 dims):
  0-127:   proximity sensors (32, 4) flattened row-major
  128-199: tactile sensors (4, 18) flattened row-major
  200-201: velocity (vx, vy)
  202:     angle
  203:     angular velocity
  204:     energy
"""

import math
from functools import partial

import jax
import jax.numpy as jnp


FOOD_RADIUS = 4.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _wrap(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


# ---------------------------------------------------------------------------
# Proximity sensors — agent channels (prey / predator)
# ---------------------------------------------------------------------------

def _single_proximity_agents(obs_pos, obs_angle, obs_radius, obs_idx,
                             all_pos, all_active, all_species, all_radii,
                             half_fov, bin_width, max_range, n_sensors):
    """Proximity sensor readings for one observer against all agent targets.

    Returns (closest_prey, closest_pred) each shape (n_sensors,) — edge distances.
    Sensors with no detection have value inf.
    """
    A = all_pos.shape[0]

    delta = all_pos - obs_pos                                       # (A, 2)
    dist = jnp.linalg.norm(delta, axis=1)                          # (A,)
    edge_dist = jnp.maximum(dist - obs_radius - all_radii, 0.0)    # (A,)
    angle_to = jnp.arctan2(delta[:, 1], delta[:, 0])               # (A,)

    rel_angle = _wrap(angle_to - obs_angle)                         # (A,)
    bin_idx = jnp.floor((rel_angle + half_fov) / bin_width).astype(jnp.int32)

    self_mask = jnp.arange(A) != obs_idx
    in_fov = (bin_idx >= 0) & (bin_idx < n_sensors)
    in_range = edge_dist <= max_range
    valid = all_active & self_mask & in_fov & in_range

    clamped_bin = jnp.clip(bin_idx, 0, n_sensors - 1)

    prey_dist = jnp.where(valid & (all_species == 0), edge_dist, jnp.inf)
    pred_dist = jnp.where(valid & (all_species == 1), edge_dist, jnp.inf)

    closest_prey = jnp.full(n_sensors, jnp.inf).at[clamped_bin].min(prey_dist)
    closest_pred = jnp.full(n_sensors, jnp.inf).at[clamped_bin].min(pred_dist)

    return closest_prey, closest_pred


# ---------------------------------------------------------------------------
# Proximity sensors — food channel
# ---------------------------------------------------------------------------

def _single_proximity_food(obs_pos, obs_angle, obs_radius,
                           food_pos, food_active,
                           half_fov, bin_width, max_range, n_sensors):
    """Proximity sensor readings for one observer against all food targets.

    Returns closest_food shape (n_sensors,) — edge distances.
    """
    delta = food_pos - obs_pos                                      # (F, 2)
    dist = jnp.linalg.norm(delta, axis=1)                          # (F,)
    # Reference only subtracts observer radius, not food radius
    edge_dist = jnp.maximum(dist - obs_radius, 0.0)                # (F,)
    angle_to = jnp.arctan2(delta[:, 1], delta[:, 0])               # (F,)

    rel_angle = _wrap(angle_to - obs_angle)
    bin_idx = jnp.floor((rel_angle + half_fov) / bin_width).astype(jnp.int32)

    in_fov = (bin_idx >= 0) & (bin_idx < n_sensors)
    in_range = edge_dist <= max_range
    valid = food_active & in_fov & in_range

    clamped_bin = jnp.clip(bin_idx, 0, n_sensors - 1)
    masked = jnp.where(valid, edge_dist, jnp.inf)

    closest_food = jnp.full(n_sensors, jnp.inf).at[clamped_bin].min(masked)
    return closest_food


# ---------------------------------------------------------------------------
# Proximity sensors — wall channel (fully vectorized, no vmap)
# ---------------------------------------------------------------------------

def _compute_wall_distances(positions, angles, radii,
                            world_size, half_fov, bin_width, max_range, n_sensors):
    """Wall proximity for all agents × all sensors.

    Returns (max_agents, n_sensors) — edge distances to closest wall per sensor.
    """
    k = jnp.arange(n_sensors)
    offsets = -half_fov + (k + 0.5) * bin_width                    # (S,)
    sensor_dirs = angles[:, None] + offsets[None, :]                # (A, S)

    cos_s = jnp.cos(sensor_dirs)
    sin_s = jnp.sin(sensor_dirs)

    px = positions[:, 0:1]   # (A, 1)
    py = positions[:, 1:2]   # (A, 1)

    INF = jnp.float32(1e9)
    eps = 1e-9

    t_right  = jnp.where(cos_s > eps,  (world_size - px) / cos_s, INF)
    t_left   = jnp.where(cos_s < -eps, -px / cos_s,               INF)
    t_top    = jnp.where(sin_s > eps,  (world_size - py) / sin_s, INF)
    t_bottom = jnp.where(sin_s < -eps, -py / sin_s,               INF)

    # Minimum positive distance to any wall
    wall_dist = jnp.minimum(
        jnp.minimum(t_right, t_left),
        jnp.minimum(t_top, t_bottom),
    )
    wall_edge = jnp.maximum(wall_dist - radii[:, None], 0.0)       # (A, S)
    return wall_edge


# ---------------------------------------------------------------------------
# Winner-take-all encoding
# ---------------------------------------------------------------------------

def _winner_take_all(closest_prey, closest_pred, closest_food, closest_wall,
                     max_range, n_sensors):
    """Convert per-channel closest distances to (A, S, 4) proximity readings.

    Winner channel gets 1 - dist/max_range, others get -1.0.
    Sensors with no detection (all channels > max_range) get all -1.0.
    """
    # (A, S, 4)
    all_dists = jnp.stack([closest_prey, closest_pred, closest_food, closest_wall], axis=-1)

    winner = jnp.argmin(all_dists, axis=-1)         # (A, S)
    min_dist = jnp.min(all_dists, axis=-1)           # (A, S)
    detected = min_dist <= max_range                  # (A, S)

    reading = jnp.clip(1.0 - min_dist / max_range, 0.0, 1.0)

    winner_onehot = jax.nn.one_hot(winner, 4)        # (A, S, 4)

    result = jnp.where(
        detected[:, :, None] & (winner_onehot > 0.5),
        reading[:, :, None],
        -1.0,
    )
    return result  # (A, S, 4)


# ---------------------------------------------------------------------------
# Tactile sensors
# ---------------------------------------------------------------------------

def _single_tactile(obs_pos, obs_radius, obs_species, obs_idx,
                    all_pos, all_active, all_species, all_radii,
                    food_pos, food_active,
                    world_size, bin_centers, bin_half_width, n_bins):
    """Tactile sensor readings for one observer. Returns (4, n_bins)."""
    A = all_pos.shape[0]

    # --- Agent contacts ---
    delta = all_pos - obs_pos                                       # (A, 2)
    dist = jnp.linalg.norm(delta, axis=1)                          # (A,)
    contact_thresh = obs_radius + all_radii
    contact = (dist <= contact_thresh) & all_active & (jnp.arange(A) != obs_idx)

    angle_to = jnp.arctan2(delta[:, 1], delta[:, 0])               # (A,)
    # Bin assignment: find nearest bin
    adiff = _wrap(angle_to[:, None] - bin_centers[None, :])         # (A, B)
    nearest_bin = jnp.argmin(jnp.abs(adiff), axis=1)               # (A,)
    in_bin = jnp.abs(adiff[jnp.arange(A), nearest_bin]) <= bin_half_width

    valid_contact = contact & in_bin
    clamped_nearest = jnp.clip(nearest_bin, 0, n_bins - 1)

    # Channel 0: conspecific (same species)
    con = valid_contact & (all_species == obs_species)
    con_bins = jnp.zeros(n_bins).at[clamped_nearest].max(con.astype(jnp.float32))

    # Channel 1: other species
    other = valid_contact & (all_species != obs_species)
    other_bins = jnp.zeros(n_bins).at[clamped_nearest].max(other.astype(jnp.float32))

    # --- Food contacts ---
    food_delta = food_pos - obs_pos                                 # (F, 2)
    food_dist = jnp.linalg.norm(food_delta, axis=1)                # (F,)
    # Reference uses agent_radius only (no FOOD_RADIUS) for tactile contact
    food_contact = (food_dist <= obs_radius) & food_active

    food_angle = jnp.arctan2(food_delta[:, 1], food_delta[:, 0])
    food_adiff = _wrap(food_angle[:, None] - bin_centers[None, :])  # (F, B)
    food_nearest = jnp.argmin(jnp.abs(food_adiff), axis=1)         # (F,)
    food_in_bin = jnp.abs(food_adiff[jnp.arange(food_pos.shape[0]), food_nearest]) <= bin_half_width
    food_valid = food_contact & food_in_bin
    food_clamped = jnp.clip(food_nearest, 0, n_bins - 1)
    food_bins = jnp.zeros(n_bins).at[food_clamped].max(food_valid.astype(jnp.float32))

    # --- Wall contacts ---
    wall_bins = jnp.zeros(n_bins)
    wall_contacts = [
        (obs_pos[0] <= obs_radius, jnp.pi),            # left wall, direction = π
        (obs_pos[0] >= world_size - obs_radius, 0.0),   # right wall, direction = 0
        (obs_pos[1] <= obs_radius, -jnp.pi / 2),        # bottom wall, direction = -π/2
        (obs_pos[1] >= world_size - obs_radius, jnp.pi / 2),  # top wall, direction = π/2
    ]
    for is_contact, wall_dir in wall_contacts:
        w_adiff = _wrap(wall_dir - bin_centers)                     # (B,)
        w_nearest = jnp.argmin(jnp.abs(w_adiff))
        w_in_bin = jnp.abs(w_adiff[w_nearest]) <= bin_half_width
        wall_bins = jnp.where(
            is_contact & w_in_bin,
            wall_bins.at[w_nearest].set(1.0),
            wall_bins,
        )

    return jnp.stack([con_bins, other_bins, food_bins, wall_bins])  # (4, n_bins)


# ---------------------------------------------------------------------------
# Top-level: compute all observations
# ---------------------------------------------------------------------------

_obs_fn_cache = {}


def compute_all_observations(obs_state: dict, config: dict) -> jnp.ndarray:
    """Compute observations for ALL agents simultaneously.

    Args:
        obs_state: dict from extract_obs_state() with fixed-shape JAX arrays
        config: experiment config dict

    Returns:
        (max_agents, obs_dim) float32 — rows for inactive agents are zeros.
    """
    max_agents = obs_state["max_agents"]
    food_max = config["food_max"]
    n_sensors = config["n_proximity_sensors"]
    n_tactile_bins = config["n_tactile_sensors"]

    cache_key = (max_agents, food_max, n_sensors, n_tactile_bins)
    if cache_key not in _obs_fn_cache:
        _obs_fn_cache[cache_key] = _build_obs_fn(config, max_agents, food_max)

    return _obs_fn_cache[cache_key](obs_state)


def _build_obs_fn(config, max_agents, food_max):
    """Build and JIT-compile the observation function for given shapes."""
    n_sensors = config["n_proximity_sensors"]
    n_tactile_bins = config["n_tactile_sensors"]
    fov_rad = math.radians(config["proximity_fov_deg"])
    half_fov = fov_rad / 2
    bin_width = fov_rad / n_sensors
    max_range = float(config["proximity_max_range"])
    world_size = float(config["world_size"])
    tactile_spacing_rad = math.radians(config["tactile_spacing_deg"])
    tactile_half_width = tactile_spacing_rad / 2 + 1e-9
    tactile_bin_centers = jnp.arange(n_tactile_bins) * tactile_spacing_rad

    # Build vmapped proximity functions
    _vmap_prox_agents = jax.vmap(
        lambda pos, ang, rad, idx, ap, aa, asp, ar: _single_proximity_agents(
            pos, ang, rad, idx, ap, aa, asp, ar,
            half_fov, bin_width, max_range, n_sensors
        ),
        in_axes=(0, 0, 0, 0, None, None, None, None),
    )

    _vmap_prox_food = jax.vmap(
        lambda pos, ang, rad, fp, fa: _single_proximity_food(
            pos, ang, rad, fp, fa,
            half_fov, bin_width, max_range, n_sensors
        ),
        in_axes=(0, 0, 0, None, None),
    )

    _vmap_tactile = jax.vmap(
        lambda pos, rad, sp, idx, ap, aa, asp, ar, fp, fa: _single_tactile(
            pos, rad, sp, idx, ap, aa, asp, ar, fp, fa,
            world_size, tactile_bin_centers, tactile_half_width, n_tactile_bins
        ),
        in_axes=(0, 0, 0, 0, None, None, None, None, None, None),
    )

    @jax.jit
    def _compute(obs_state):
        positions = obs_state["positions"]
        angles = obs_state["angles"]
        velocities_xy = obs_state["velocities_xy"]
        velocities_ang = obs_state["velocities_ang"]
        is_active = obs_state["is_active"]
        species = obs_state["species"]
        radii = obs_state["radii"]
        energies = obs_state["energies"]
        food_positions = obs_state["food_positions"]
        food_active = obs_state["food_active"]

        obs_indices = jnp.arange(max_agents)

        # 1. Proximity — agent channels (prey, predator)
        closest_prey, closest_pred = _vmap_prox_agents(
            positions, angles, radii, obs_indices,
            positions, is_active, species, radii,
        )  # each (A, S)

        # 2. Proximity — food channel
        closest_food = _vmap_prox_food(
            positions, angles, radii,
            food_positions, food_active,
        )  # (A, S)

        # 3. Proximity — wall channel
        closest_wall = _compute_wall_distances(
            positions, angles, radii,
            world_size, half_fov, bin_width, max_range, n_sensors,
        )  # (A, S)

        # 4. Winner-take-all encoding
        proximity = _winner_take_all(
            closest_prey, closest_pred, closest_food, closest_wall,
            max_range, n_sensors,
        )  # (A, S, 4)

        # 5. Tactile sensors
        tactile = _vmap_tactile(
            positions, radii, species, obs_indices,
            positions, is_active, species, radii,
            food_positions, food_active,
        )  # (A, 4, B)

        # 6. Assemble observation vector (A, 205)
        prox_flat = proximity.reshape(max_agents, -1)       # (A, 128)
        tact_flat = tactile.reshape(max_agents, -1)         # (A, 72)

        obs = jnp.concatenate([
            prox_flat,                                       # 0-127
            tact_flat,                                       # 128-199
            velocities_xy,                                   # 200-201
            angles[:, None],                                 # 202
            velocities_ang[:, None],                         # 203
            energies[:, None],                               # 204
        ], axis=1)

        # Mask inactive agents to zero
        obs = jnp.where(is_active[:, None], obs, 0.0)
        return obs

    return _compute
