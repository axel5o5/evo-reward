"""
agents.py
---------
Observation vector construction, stimulus extraction, reward computation.

Observation vector layout (obs_dim = 205):
  Index 0-127:   proximity sensors  (32, 4)  -- [prey, predator, food, wall]
  Index 128-199: tactile collision   (4, 18)  -- [conspecific, other, food, wall]
  Index 200-201: velocity            (2,)     -- (vx, vy)
  Index 202:     angle               (1,)
  Index 203:     angular velocity    (1,)
  Index 204:     energy              (1,)

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


def get_observation(world, agent_id: int, config: dict) -> jnp.ndarray:
    """
    Build full 205-dim observation vector for one agent.
    Returns: shape (config["obs_dim"],), float32.
    Layout pinned in docs/interfaces.md.
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
