"""
observations.py
---------------
Vectorized observation computation for all agents simultaneously using JAX.

Replaces the per-agent Python loop (get_observation → compute_proximity_sensors →
compute_tactile_sensors) with a single JIT-compiled batched call. All operations
are pure JAX, so the same code runs on CPU or GPU.

Key technique: jax.vmap over the observer axis, with scatter-min
(jnp.ndarray.at[bin_idx].min(dist)) for binning closest objects per sensor.

Observation layout (baseline, 205 dims):
  0-127:   proximity sensors (32, 4) flattened row-major
  128-199: tactile sensors (4, 18) flattened row-major
  200-201: velocity (vx, vy)
  202:     angle
  203:     angular velocity
  204:     energy

Extension: social observation (social_obs = "position_heading_velocity", 215 dims):
  205-214: social obs (5, 2) -- [heading, speed] x 5 closest conspecifics
           Sorted by Euclidean distance (closest first), zero-padded.
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
                            world_x, world_y, half_fov, bin_width, max_range, n_sensors):
    """Wall proximity for all agents × all sensors.

    Returns (max_agents, n_sensors) — edge distances to closest wall per sensor.
    D25: world bounds are (x, y) separately to support rectangular worlds.
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

    t_right  = jnp.where(cos_s > eps,  (world_x - px) / cos_s, INF)
    t_left   = jnp.where(cos_s < -eps, -px / cos_s,            INF)
    t_top    = jnp.where(sin_s > eps,  (world_y - py) / sin_s, INF)
    t_bottom = jnp.where(sin_s < -eps, -py / sin_s,            INF)

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

def _bin_in_agent_frame(angle_world, obs_angle, n_bins):
    """Classify a world-frame angle into one of `n_bins` tactile bins in the
    observer's local frame.

    D26: phyjax2d/emevo convention is that agent heading=0 means forward is
    WORLD +y (not +x). Bins are 0-indexed from "directly forward" going
    counter-clockwise, each covering 2π/n_bins of arc. Pre-D26 we omitted
    both the heading subtraction AND the π/2 offset, so tactile bins were
    world-frame — effectively random from the agent's perspective and
    preventing useful policy learning on tactile inputs.
    """
    TWO_PI = 2.0 * jnp.pi
    # JAX's `%` returns non-negative when the divisor is positive, so we
    # don't need a `+ 3*TWO_PI` offset — and adding one introduces fp
    # rounding that can snap "directly in front" (rel=0) to ~TWO_PI,
    # landing in bin n-1 instead of bin 0.
    rel = (angle_world - obs_angle - jnp.pi / 2.0) % TWO_PI
    bin_width = TWO_PI / n_bins
    idx = jnp.clip((rel / bin_width).astype(jnp.int32), 0, n_bins - 1)
    return idx


def _single_tactile(obs_pos, obs_angle, obs_radius, obs_species, obs_idx,
                    all_pos, all_active, all_species, all_radii,
                    food_pos, food_active,
                    world_x, world_y, n_bins):
    """Tactile sensor readings for one observer. Returns (4, n_bins).

    D26: signature now takes `obs_angle` (the observer's heading) so that
    bin assignments are rotated into the agent's local frame. Bin-centers
    / bin_half_width are gone; we use emevo's boundary-based assignment
    via `_bin_in_agent_frame`.
    """
    A = all_pos.shape[0]
    F = food_pos.shape[0]

    # --- Agent contacts ---
    delta = all_pos - obs_pos                                       # (A, 2)
    dist = jnp.linalg.norm(delta, axis=1)                          # (A,)
    contact_thresh = obs_radius + all_radii
    contact = (dist <= contact_thresh) & all_active & (jnp.arange(A) != obs_idx)

    angle_to = jnp.arctan2(delta[:, 1], delta[:, 0])               # (A,)
    nearest_bin = _bin_in_agent_frame(angle_to, obs_angle, n_bins) # (A,)

    # Channel 0: conspecific (same species)
    con = contact & (all_species == obs_species)
    con_bins = jnp.zeros(n_bins).at[nearest_bin].max(con.astype(jnp.float32))

    # Channel 1: other species
    other = contact & (all_species != obs_species)
    other_bins = jnp.zeros(n_bins).at[nearest_bin].max(other.astype(jnp.float32))

    # --- Food contacts ---
    food_delta = food_pos - obs_pos                                 # (F, 2)
    food_dist = jnp.linalg.norm(food_delta, axis=1)                # (F,)
    food_contact = (food_dist <= obs_radius) & food_active

    food_angle = jnp.arctan2(food_delta[:, 1], food_delta[:, 0])
    food_bin = _bin_in_agent_frame(food_angle, obs_angle, n_bins)  # (F,)
    food_bins = jnp.zeros(n_bins).at[food_bin].max(food_contact.astype(jnp.float32))

    # --- Wall contacts ---
    # Wall direction in world frame → rotate into agent frame via the
    # same helper. Each wall the agent is against lights up exactly one bin.
    wall_bins = jnp.zeros(n_bins)
    wall_contacts = [
        (obs_pos[0] <= obs_radius,                jnp.pi),          # left wall
        (obs_pos[0] >= world_x - obs_radius,      0.0),             # right wall
        (obs_pos[1] <= obs_radius,                -jnp.pi / 2),     # bottom wall
        (obs_pos[1] >= world_y - obs_radius,      jnp.pi / 2),      # top wall
    ]
    for is_contact, wall_dir in wall_contacts:
        w_bin = _bin_in_agent_frame(jnp.asarray(wall_dir), obs_angle, n_bins)
        wall_bins = jnp.where(
            is_contact,
            wall_bins.at[w_bin].set(1.0),
            wall_bins,
        )

    return jnp.stack([con_bins, other_bins, food_bins, wall_bins])  # (4, n_bins)


# ---------------------------------------------------------------------------
# Social observation — heading + speed of N closest conspecifics
# ---------------------------------------------------------------------------

def _single_social_obs(obs_pos, obs_species, obs_idx,
                       all_pos, all_active, all_species,
                       all_angles, all_velocities_xy,
                       max_range, n_neighbors):
    """Social observation for one observer. Returns (2 * n_neighbors,).

    For each of the n_neighbors closest conspecifics (same species, active,
    within max_range), returns [heading, speed] interleaved. Zero-padded
    if fewer than n_neighbors are visible.
    """
    A = all_pos.shape[0]

    # Euclidean distance from observer to all agents
    delta = all_pos - obs_pos                                          # (A, 2)
    dist = jnp.linalg.norm(delta, axis=1)                             # (A,)

    # Validity mask: same species, active, not self, within range
    same_species = (all_species == obs_species)
    not_self = (jnp.arange(A) != obs_idx)
    in_range = (dist <= max_range)
    valid = all_active & same_species & not_self & in_range

    # Mask invalid agents to inf distance for sorting
    masked_dist = jnp.where(valid, dist, jnp.inf)                     # (A,)

    # Get indices of N closest (ascending distance)
    sorted_indices = jnp.argsort(masked_dist)                          # (A,)
    top_n_indices = sorted_indices[:n_neighbors]                       # (N,)

    # Extract heading and speed for top N
    headings = all_angles[top_n_indices]                               # (N,)
    speeds = jnp.linalg.norm(all_velocities_xy[top_n_indices], axis=1) # (N,)

    # Zero-pad entries where the neighbor was actually invalid (dist == inf)
    top_n_dists = masked_dist[top_n_indices]                           # (N,)
    is_real = jnp.isfinite(top_n_dists)                                # (N,)
    headings = jnp.where(is_real, headings, 0.0)
    speeds = jnp.where(is_real, speeds, 0.0)

    # Interleave: [h1, s1, h2, s2, ..., hN, sN]
    social = jnp.stack([headings, speeds], axis=1).reshape(-1)         # (2*N,)
    return social


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
    social_obs = config.get("social_obs", "position_only")
    n_social = config.get("n_social_neighbors", 0) if social_obs != "position_only" else 0

    cache_key = (max_agents, food_max, n_sensors, n_tactile_bins, social_obs, n_social)
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
    from src.environment import world_bounds
    world_x, world_y = world_bounds(config)
    tactile_spacing_rad = math.radians(config["tactile_spacing_deg"])
    tactile_half_width = tactile_spacing_rad / 2 + 1e-9
    tactile_bin_centers = jnp.arange(n_tactile_bins) * tactile_spacing_rad

    # Social observation config (compile-time branch)
    social_obs = config.get("social_obs", "position_only")
    include_social = (social_obs == "position_heading_velocity")
    n_neighbors = config.get("n_social_neighbors", 5) if include_social else 0

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
        lambda pos, ang, rad, sp, idx, ap, aa, asp, ar, fp, fa: _single_tactile(
            pos, ang, rad, sp, idx, ap, aa, asp, ar, fp, fa,
            world_x, world_y, n_tactile_bins,
        ),
        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None),
    )

    # Build vmapped social observation function (only if social_obs active)
    if include_social:
        _vmap_social = jax.vmap(
            lambda pos, sp, idx, ap, aa, asp, ang, vel: _single_social_obs(
                pos, sp, idx, ap, aa, asp, ang, vel,
                max_range, n_neighbors
            ),
            in_axes=(0, 0, 0, None, None, None, None, None),
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
            world_x, world_y, half_fov, bin_width, max_range, n_sensors,
        )  # (A, S)

        # 4. Winner-take-all encoding
        proximity = _winner_take_all(
            closest_prey, closest_pred, closest_food, closest_wall,
            max_range, n_sensors,
        )  # (A, S, 4)

        # 5. Tactile sensors — D26: pass each observer's `angle` so bin
        # classification happens in the agent's local frame.
        tactile = _vmap_tactile(
            positions, angles, radii, species, obs_indices,
            positions, is_active, species, radii,
            food_positions, food_active,
        )  # (A, 4, B)

        # 6. Assemble observation vector
        prox_flat = proximity.reshape(max_agents, -1)       # (A, 128)
        tact_flat = tactile.reshape(max_agents, -1)         # (A, 72)

        parts = [
            prox_flat,                                       # 0-127
            tact_flat,                                       # 128-199
            velocities_xy,                                   # 200-201
            angles[:, None],                                 # 202
            velocities_ang[:, None],                         # 203
            energies[:, None],                               # 204
        ]

        # 6.5 Social observation (Axis 2)
        if include_social:
            social = _vmap_social(
                positions, species, obs_indices,
                positions, is_active, species, angles, velocities_xy,
            )  # (A, 2*N)
            parts.append(social)

        obs = jnp.concatenate(parts, axis=1)

        # Mask inactive agents to zero
        obs = jnp.where(is_active[:, None], obs, 0.0)
        return obs

    return _compute
