"""
agents.py
---------
Observation vector construction, stimulus extraction, reward computation.

Observation vector layout (baseline, obs_dim = 205):
  Index 0-127:   proximity sensors  (32, 4)  -- [prey, predator, food, wall]
  Index 128-199: tactile collision   (4, 18)  -- [conspecific, other, food, wall]
  Index 200-201: velocity            (2,)     -- (vx, vy)
  Index 202:     angle               (1,)
  Index 203:     angular velocity    (1,)
  Index 204:     energy              (1,)

Extension: social observation (social_obs = "position_heading_velocity"):
  Index 205-214: social obs (5, 2)  -- [heading, speed] x 5 closest conspecifics
                 Sorted by Euclidean distance (closest first).
                 Zero-padded if fewer than n_social_neighbors conspecifics visible.
                 obs_dim = 215 when active.

Stimulus extraction for reward computation:
  max_s_prey = MAX over 32 sensors of channel 0 (prey), clipped >= 0
  max_s_pred = MAX over 32 sensors of channel 1 (predator), clipped >= 0

  This is the max over the 32 predator-channel sensor readings -- the closest
  predator in the most favorable direction. Not the closest predator overall.
  The per-type channel separation feeds directly into this.
"""

import jax.numpy as jnp

from src.environment import (
    get_sensor_readings,
    CHANNEL_PREY,
    CHANNEL_PREDATOR,
)
from src.reward import compute_linear_reward


def _compute_social_obs_single(agent, agents_list, n_neighbors, max_range):
    """Compute social observation block for one agent.

    Returns shape (2 * n_neighbors,) float32: [heading_1, speed_1, ..., heading_N, speed_N].
    Sorted by Euclidean distance (closest first). Zero-padded if fewer than
    n_neighbors conspecifics are within max_range.
    """
    observer_pos = agent.position
    observer_species = agent.species
    observer_id = agent.agent_id

    # Collect conspecifics within range
    neighbors = []
    for other in agents_list:
        if other.agent_id == observer_id:
            continue
        if other.species != observer_species:
            continue
        other_pos = other.position if other.position is not None else jnp.zeros(2)
        dist = float(jnp.linalg.norm(other_pos - observer_pos))
        if dist <= max_range:
            heading = float(other.angle)
            other_vel = other.velocity if other.velocity is not None else jnp.zeros(2)
            speed = float(jnp.linalg.norm(other_vel))
            neighbors.append((dist, heading, speed))

    # Sort by distance ascending (closest first)
    neighbors.sort(key=lambda x: x[0])

    # Build interleaved block: [h1, s1, h2, s2, ...]
    social = []
    for i in range(n_neighbors):
        if i < len(neighbors):
            _, heading, speed = neighbors[i]
            social.extend([heading, speed])
        else:
            social.extend([0.0, 0.0])

    return jnp.array(social, dtype=jnp.float32)


def get_observation(world, agent_id: int, config: dict) -> jnp.ndarray:
    """
    Build observation vector for one agent.
    Returns: shape (config["obs_dim"],), float32.
    Layout pinned in docs/interfaces.md.
    config["social_obs"] controls whether social channels are appended.
    """
    agent = None
    for a in world.agents:
        if a.agent_id == agent_id:
            agent = a
            break
    if agent is None:
        raise ValueError(f"Agent {agent_id} not found")

    sensors = get_sensor_readings(world, agent_id, config)
    proximity = sensors["proximity"]   # (32, 4)
    tactile = sensors["tactile"]       # (4, 18)

    proximity_flat = proximity.reshape(-1)  # (128,)
    tactile_flat = tactile.reshape(-1)      # (72,)

    velocity = agent.velocity if agent.velocity is not None else jnp.zeros(2)
    angle = jnp.array([agent.angle])
    ang_vel = jnp.array([agent.ang_vel])
    energy = jnp.array([agent.energy])

    obs = jnp.concatenate([
        proximity_flat,    # 0-127
        tactile_flat,      # 128-199
        velocity,          # 200-201
        angle,             # 202
        ang_vel,           # 203
        energy,            # 204
    ])

    # Axis 2: social observation — heading + speed of N closest conspecifics
    social_mode = config.get("social_obs", "position_only")
    if social_mode == "position_heading_velocity":
        n_neighbors = config.get("n_social_neighbors", 5)
        max_range = float(config["proximity_max_range"])
        social_block = _compute_social_obs_single(
            agent, world.agents, n_neighbors, max_range
        )
        obs = jnp.concatenate([obs, social_block])

    return obs


def get_stimulus_scalars(world, agent_id: int, config: dict) -> dict:
    """
    Extract reward-relevant sensor scalars from world state.

    Returns dict with keys: n_eaten, motor_norm, max_s_prey, max_s_pred.
    n_eaten and motor_norm default to 0 -- caller must override from
    check_eating() result and action norm.

    max_s_prey: MAX over 32 sensors of channel 0 (prey), clipped >= 0.
    max_s_pred: MAX over 32 sensors of channel 1 (predator), clipped >= 0.
    """
    sensors = get_sensor_readings(world, agent_id, config)
    proximity = sensors["proximity"]  # (32, 4)

    prey_readings = proximity[:, CHANNEL_PREY]       # (32,)
    pred_readings = proximity[:, CHANNEL_PREDATOR]   # (32,)

    # Clip >= 0 before aggregation (winner-take-all gives -1 for non-detected)
    max_s_prey = float(jnp.max(jnp.clip(prey_readings, 0.0)))
    max_s_pred = float(jnp.max(jnp.clip(pred_readings, 0.0)))

    return {
        "n_eaten": 0,
        "motor_norm": 0.0,
        "max_s_prey": max_s_prey,
        "max_s_pred": max_s_pred,
    }


def compute_reward(
    genome: jnp.ndarray,
    stimuli: dict,
) -> float:
    """
    Apply K&D reward equation:
      r = w_eat * n_eaten
        + 0.01 * w_act * motor_norm
        + 0.1  * w_prey * max_s_prey
        + 0.1  * w_pred * max_s_pred

    genome order: [w_eat, w_act, w_prey, w_pred]
    """
    return compute_linear_reward(
        genome,
        n_eaten=stimuli["n_eaten"],
        motor_norm=stimuli["motor_norm"],
        max_s_prey=stimuli["max_s_prey"],
        max_s_pred=stimuli["max_s_pred"],
    )
