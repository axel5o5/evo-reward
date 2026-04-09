"""
environment.py
--------------
2D world, physics, 4-channel proximity sensors (32 sensors x 4 object types),
72-bin tactile sensors, food dynamics (max=600), eating/capture detection.

Matches the confirmed 205-dim observation layout from emevo gecco2026 branch:
  Index 0-127:   proximity sensors  (32, 4)  -- [prey, predator, food, wall]
  Index 128-199: tactile collision   (4, 18)  -- [conspecific, other, food, wall]
  Index 200-201: velocity            (2,)     -- (vx, vy)
  Index 202:     angle               (1,)
  Index 203:     angular velocity    (1,)
  Index 204:     energy              (1,)

Physics via phyjax2d where available; pure-Python fallback for testing.

emevo source references:
  - circle_foraging.py:740-748 (obs_space definition)
  - circle_foraging_with_predator.py:84-111 (_observe_closest)
  - circle_foraging_with_predator.py:466-473 (observation construction)
  - circle_foraging_with_predator.py:401 (action clip)
  - cf_predator.py:122 (sigmoid_scale action mapping)
  - config/env/20251122-predator-square.toml (all parameter values)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Complete runtime state of one agent."""
    # Identity
    agent_id: int = 0
    species: int = 0         # 0 = prey, 1 = predator
    parent_id: int = -1

    # Physics
    position: object = None   # jnp.ndarray shape (2,)
    velocity: object = None   # jnp.ndarray shape (2,)
    angle: float = 0.0        # radians
    ang_vel: float = 0.0      # radians/step

    # Lifecycle
    age: int = 0
    energy: float = 100.0

    # Genome — THE ONLY HERITABLE COMPONENT
    # FIXED ORDER: [w_eat, w_act, w_prey, w_pred]
    reward_weights: object = None  # jnp.ndarray shape (4,)

    # Policy (NOT inherited, reset fresh at every birth)
    policy_params: object = None
    policy_opt_state: object = None

    # PPO rollout buffer
    rollout: object = None


@dataclass
class WorldState:
    """Complete simulation state at one timestep."""
    step: int = 0
    agents: list = field(default_factory=list)
    food_internal: float = 0.0            # real-valued food counter
    food_positions: object = None          # jnp.ndarray shape (n_food, 2)
    rng_key: object = None                 # jax.random.PRNGKey


# ---------------------------------------------------------------------------
# Sensor constants
# ---------------------------------------------------------------------------

# Channel indices for proximity sensors (per sensor)
CHANNEL_PREY = 0
CHANNEL_PREDATOR = 1
CHANNEL_FOOD = 2
CHANNEL_WALL = 3
N_CHANNELS = 4

# Channel indices for tactile sensors
TACTILE_CONSPECIFIC = 0
TACTILE_OTHER_SPECIES = 1
TACTILE_FOOD = 2
TACTILE_WALL = 3
N_TACTILE_CHANNELS = 4


# ---------------------------------------------------------------------------
# Action mapping: sigmoid scaling (D10 in emevo-diff.md)
# ---------------------------------------------------------------------------

def sigmoid_scale(x, low, high):
    """
    Map unbounded network output to [low, high] via sigmoid.
    sigmoid_scale(x) = (high - low) * sigmoid(x) + low

    emevo source: cf_predator.py:122, env.act_space.sigmoid_scale(actions)
    For action_clip_low=-20, action_clip_high=80:
      sigmoid_scale(x) = 100 * sigmoid(x) - 20
    """
    return (high - low) * jax.nn.sigmoid(x) + low


def map_actions(raw_actions, config):
    """
    Map raw network outputs to motor forces via sigmoid scaling.
    raw_actions: shape (2,) unbounded
    Returns: shape (2,) in [action_clip_low, action_clip_high]
    """
    low = config.get("action_clip_low", -20.0)
    high = config.get("action_clip_high", 80.0)
    mapping = config.get("action_mapping", "sigmoid")

    if mapping == "sigmoid":
        return sigmoid_scale(raw_actions, low, high)
    else:
        # Hard clip fallback
        return jnp.clip(raw_actions, low, high)


# ---------------------------------------------------------------------------
# World initialization
# ---------------------------------------------------------------------------

def init_world(config: dict, rng_key) -> WorldState:
    """
    Initialize world:
    - config["prey_initial"] prey at random positions
    - config["predator_initial"] predators at random positions
    - food_initial food items at random positions
    - food_internal = food_initial
    All agents: reward_weights ~ N(0, config["reward_weights_init_std"])
    """
    world_size = config["world_size"]
    prey_initial = config.get("prey_initial", 150)
    pred_initial = config.get("predator_initial", 10)
    food_initial = config.get("food_initial", 40)
    prey_e_init = config.get("prey_e_initial", 100.0)
    pred_e_init = config.get("predator_e_initial", 100.0)
    rw_std = config.get("reward_weights_init_std", 0.1)

    agents = []
    agent_id = 0

    # Create prey
    for i in range(prey_initial):
        rng_key, pos_key, angle_key, rw_key = jax.random.split(rng_key, 4)
        pos = jax.random.uniform(pos_key, shape=(2,), minval=0.0, maxval=float(world_size))
        angle = float(jax.random.uniform(angle_key, minval=-math.pi, maxval=math.pi))
        rw = jax.random.normal(rw_key, shape=(4,)) * rw_std

        agents.append(AgentState(
            agent_id=agent_id,
            species=0,
            parent_id=-1,
            position=pos,
            velocity=jnp.zeros(2),
            angle=angle,
            ang_vel=0.0,
            age=0,
            energy=prey_e_init,
            reward_weights=rw,
        ))
        agent_id += 1

    # Create predators
    for i in range(pred_initial):
        rng_key, pos_key, angle_key, rw_key = jax.random.split(rng_key, 4)
        pos = jax.random.uniform(pos_key, shape=(2,), minval=0.0, maxval=float(world_size))
        angle = float(jax.random.uniform(angle_key, minval=-math.pi, maxval=math.pi))
        rw = jax.random.normal(rw_key, shape=(4,)) * rw_std

        agents.append(AgentState(
            agent_id=agent_id,
            species=1,
            parent_id=-1,
            position=pos,
            velocity=jnp.zeros(2),
            angle=angle,
            ang_vel=0.0,
            age=0,
            energy=pred_e_init,
            reward_weights=rw,
        ))
        agent_id += 1

    # Create food
    rng_key, food_key = jax.random.split(rng_key)
    food_positions = jax.random.uniform(
        food_key, shape=(food_initial, 2),
        minval=0.0, maxval=float(world_size),
    )

    return WorldState(
        step=0,
        agents=agents,
        food_internal=float(food_initial),
        food_positions=food_positions,
        rng_key=rng_key,
    )


# ---------------------------------------------------------------------------
# Proximity sensors: 32 sensors x 4 channels (per-type, winner-take-all)
# ---------------------------------------------------------------------------

def _angle_diff(a, b):
    """Signed angle difference, result in [-pi, pi]."""
    d = a - b
    return (d + math.pi) % (2 * math.pi) - math.pi


def compute_proximity_sensors(
    agent: AgentState,
    world: WorldState,
    config: dict,
) -> jnp.ndarray:
    """
    Compute 32-sensor x 4-channel proximity readings.

    Each sensor covers a slice of the 120-degree forward FOV.
    For each sensor, raycast to find the closest object of each type.
    Winner-take-all: only the closest type gets a positive value (inverse
    distance scaled to [0,1]); all other channels = -1.0.

    Returns: shape (32, 4), flattened to (128,) by caller.

    emevo source: circle_foraging_with_predator.py:84-111 _observe_closest()
    """
    n_sensors = config["n_proximity_sensors"]
    fov_rad = math.radians(config["proximity_fov_deg"])
    max_range = config["proximity_max_range"]
    agent_heading = agent.angle

    # Sensor angles: evenly distributed across FOV, centered on heading
    half_fov = fov_rad / 2.0
    sensor_angles = np.linspace(
        agent_heading - half_fov,
        agent_heading + half_fov,
        n_sensors,
    )

    # Half-angle of each sensor's bin
    sensor_half_width = (fov_rad / n_sensors) / 2.0

    agent_pos = np.array(agent.position)
    agent_radius = config["prey_radius"] if agent.species == 0 else config["predator_radius"]

    # Initialize: all channels = -1 (nothing detected)
    readings = np.full((n_sensors, N_CHANNELS), -1.0, dtype=np.float32)

    # For each sensor, track closest distance per channel
    closest_dist = np.full((n_sensors, N_CHANNELS), float('inf'))

    # --- Scan other agents ---
    for other in world.agents:
        if other.agent_id == agent.agent_id:
            continue

        other_pos = np.array(other.position)
        diff = other_pos - agent_pos
        dist = np.linalg.norm(diff)
        other_radius = config["prey_radius"] if other.species == 0 else config["predator_radius"]
        # Distance edge-to-edge
        edge_dist = dist - agent_radius - other_radius

        if edge_dist > max_range or edge_dist < 0:
            if edge_dist < 0:
                edge_dist = 0.0  # contact
            else:
                continue

        # Angle to other agent
        angle_to = math.atan2(diff[1], diff[0])

        # Determine channel: prey=0, predator=1
        if other.species == 0:
            channel = CHANNEL_PREY
        else:
            channel = CHANNEL_PREDATOR

        # Check which sensor bin this falls into
        for s in range(n_sensors):
            adiff = _angle_diff(angle_to, sensor_angles[s])
            if abs(adiff) <= sensor_half_width:
                if edge_dist < closest_dist[s, channel]:
                    closest_dist[s, channel] = edge_dist
                break  # Each object falls in at most one sensor bin

    # --- Scan food items ---
    if world.food_positions is not None and len(world.food_positions) > 0:
        food_pos_np = np.array(world.food_positions)
        for fi in range(len(food_pos_np)):
            diff = food_pos_np[fi] - agent_pos
            dist = np.linalg.norm(diff)
            # Food is a point (radius ~0)
            edge_dist = dist - agent_radius
            if edge_dist > max_range:
                continue
            if edge_dist < 0:
                edge_dist = 0.0

            angle_to = math.atan2(diff[1], diff[0])
            for s in range(n_sensors):
                adiff = _angle_diff(angle_to, sensor_angles[s])
                if abs(adiff) <= sensor_half_width:
                    if edge_dist < closest_dist[s, CHANNEL_FOOD]:
                        closest_dist[s, CHANNEL_FOOD] = edge_dist
                    break

    # --- Scan walls ---
    world_size = config["world_size"]
    # For each sensor direction, compute distance to nearest wall
    for s in range(n_sensors):
        sa = sensor_angles[s]
        dx = math.cos(sa)
        dy = math.sin(sa)

        # Ray from agent_pos in direction (dx, dy), find intersection with walls
        # Walls at x=0, x=world_size, y=0, y=world_size
        min_wall_dist = float('inf')

        if dx > 1e-9:
            t = (world_size - agent_pos[0]) / dx
            if t > 0:
                min_wall_dist = min(min_wall_dist, t)
        elif dx < -1e-9:
            t = -agent_pos[0] / dx
            if t > 0:
                min_wall_dist = min(min_wall_dist, t)

        if dy > 1e-9:
            t = (world_size - agent_pos[1]) / dy
            if t > 0:
                min_wall_dist = min(min_wall_dist, t)
        elif dy < -1e-9:
            t = -agent_pos[1] / dy
            if t > 0:
                min_wall_dist = min(min_wall_dist, t)

        wall_edge_dist = min_wall_dist - agent_radius
        if wall_edge_dist < 0:
            wall_edge_dist = 0.0
        if wall_edge_dist <= max_range:
            closest_dist[s, CHANNEL_WALL] = wall_edge_dist

    # --- Convert distances to inverse-distance readings + winner-take-all ---
    for s in range(n_sensors):
        # Find the channel with the closest object
        min_channel = -1
        min_dist = float('inf')
        for c in range(N_CHANNELS):
            if closest_dist[s, c] < min_dist:
                min_dist = closest_dist[s, c]
                min_channel = c

        if min_channel >= 0 and min_dist <= max_range:
            # Inverse distance: 1.0 at contact, 0.0 at max_range
            inv_dist = 1.0 - (min_dist / max_range)
            inv_dist = max(0.0, min(1.0, inv_dist))
            readings[s, min_channel] = inv_dist
            # Other channels remain -1.0 (winner-take-all)

    return jnp.array(readings)


# ---------------------------------------------------------------------------
# Tactile sensors: 4 channels x 18 bins
# ---------------------------------------------------------------------------

def compute_tactile_sensors(
    agent: AgentState,
    world: WorldState,
    config: dict,
) -> jnp.ndarray:
    """
    Compute 4-channel x 18-bin tactile (collision) readings.

    18 bins at 20-degree spacing around the full 360 degrees.
    4 channels: [conspecific, other_species, food, wall].
    Binary: 1 if contact in that bin for that type, else 0.

    For prey: conspecific = other prey, other_species = predator.
    For predator: conspecific = other predator, other_species = prey.

    Returns: shape (4, 18), flattened to (72,) by caller.
    """
    n_bins = config["n_tactile_sensors"]
    bin_spacing_deg = config.get("tactile_spacing_deg", 20.0)

    agent_pos = np.array(agent.position)
    agent_radius = config["prey_radius"] if agent.species == 0 else config["predator_radius"]

    # Contact threshold: sum of radii (touching)
    readings = np.zeros((N_TACTILE_CHANNELS, n_bins), dtype=np.float32)

    # Bin angles: evenly spaced around full 360
    bin_angles = np.array([i * math.radians(bin_spacing_deg) for i in range(n_bins)])
    bin_half_width = math.radians(bin_spacing_deg) / 2.0

    # --- Check agent contacts ---
    for other in world.agents:
        if other.agent_id == agent.agent_id:
            continue

        other_pos = np.array(other.position)
        diff = other_pos - agent_pos
        dist = np.linalg.norm(diff)
        other_radius = config["prey_radius"] if other.species == 0 else config["predator_radius"]

        # Contact: overlapping or touching
        if dist > agent_radius + other_radius:
            continue

        angle_to = math.atan2(diff[1], diff[0])
        # Relative to agent heading
        rel_angle = _angle_diff(angle_to, 0.0)  # absolute angle
        # Normalize to [0, 2pi)
        rel_angle = rel_angle % (2 * math.pi)

        # Determine channel
        if agent.species == 0:  # prey
            if other.species == 0:
                channel = TACTILE_CONSPECIFIC
            else:
                channel = TACTILE_OTHER_SPECIES
        else:  # predator
            if other.species == 1:
                channel = TACTILE_CONSPECIFIC
            else:
                channel = TACTILE_OTHER_SPECIES

        # Find bin
        for b in range(n_bins):
            adiff = _angle_diff(rel_angle, bin_angles[b])
            if abs(adiff) <= bin_half_width:
                readings[channel, b] = 1.0
                break

    # --- Check food contacts ---
    if world.food_positions is not None and len(world.food_positions) > 0:
        food_pos_np = np.array(world.food_positions)
        for fi in range(len(food_pos_np)):
            diff = food_pos_np[fi] - agent_pos
            dist = np.linalg.norm(diff)
            if dist > agent_radius:
                continue

            angle_to = math.atan2(diff[1], diff[0])
            rel_angle = angle_to % (2 * math.pi)

            for b in range(n_bins):
                adiff = _angle_diff(rel_angle, bin_angles[b])
                if abs(adiff) <= bin_half_width:
                    readings[TACTILE_FOOD, b] = 1.0
                    break

    # --- Check wall contacts ---
    world_size = config["world_size"]
    # Left wall (x=0)
    if agent_pos[0] <= agent_radius:
        wall_angle = math.pi  # wall is to the left
        rel = wall_angle % (2 * math.pi)
        for b in range(n_bins):
            if abs(_angle_diff(rel, bin_angles[b])) <= bin_half_width:
                readings[TACTILE_WALL, b] = 1.0
                break
    # Right wall (x=world_size)
    if agent_pos[0] >= world_size - agent_radius:
        wall_angle = 0.0
        rel = wall_angle % (2 * math.pi)
        for b in range(n_bins):
            if abs(_angle_diff(rel, bin_angles[b])) <= bin_half_width:
                readings[TACTILE_WALL, b] = 1.0
                break
    # Bottom wall (y=0)
    if agent_pos[1] <= agent_radius:
        wall_angle = 3 * math.pi / 2  # wall is below
        rel = wall_angle % (2 * math.pi)
        for b in range(n_bins):
            if abs(_angle_diff(rel, bin_angles[b])) <= bin_half_width:
                readings[TACTILE_WALL, b] = 1.0
                break
    # Top wall (y=world_size)
    if agent_pos[1] >= world_size - agent_radius:
        wall_angle = math.pi / 2
        rel = wall_angle % (2 * math.pi)
        for b in range(n_bins):
            if abs(_angle_diff(rel, bin_angles[b])) <= bin_half_width:
                readings[TACTILE_WALL, b] = 1.0
                break

    return jnp.array(readings)


# ---------------------------------------------------------------------------
# Full sensor readings (for agents.py)
# ---------------------------------------------------------------------------

def get_sensor_readings(world: WorldState, agent_id: int, config: dict) -> dict:
    """
    Returns raw sensor readings for one agent:
    {
        "proximity": jnp.ndarray shape (32, 4),
        "tactile":   jnp.ndarray shape (4, 18),
    }
    """
    agent = None
    for a in world.agents:
        if a.agent_id == agent_id:
            agent = a
            break
    if agent is None:
        raise ValueError(f"Agent {agent_id} not found")

    proximity = compute_proximity_sensors(agent, world, config)
    tactile = compute_tactile_sensors(agent, world, config)

    return {
        "proximity": proximity,  # (32, 4)
        "tactile": tactile,      # (4, 18)
    }


# ---------------------------------------------------------------------------
# Eating / capture detection
# ---------------------------------------------------------------------------

def check_eating(world: WorldState, config: dict) -> dict:
    """
    Check all eating/catching events this step.

    Prey: eats food if within contact distance (food within agent radius).
    Predator: catches prey if within mouth range and angle.

    Returns: {
        agent_id: n_food_eaten (int) for prey,
        agent_id: [(prey_id, prey_energy)] for predators,
    }

    Does NOT modify world state.
    """
    prey_radius = config["prey_radius"]
    pred_radius = config["predator_radius"]
    mouth_deg = config.get("predator_mouth_deg", 60.0)
    mouth_range_min = config.get("predator_mouth_range_min", 40.0)
    mouth_range_max = config.get("predator_mouth_range_max", 80.0)
    mouth_half_rad = math.radians(mouth_deg) / 2.0

    eating_events = {}

    # --- Prey eating food ---
    food_eaten_indices = set()
    prey_agents = [a for a in world.agents if a.species == 0]
    pred_agents = [a for a in world.agents if a.species == 1]

    if world.food_positions is not None and len(world.food_positions) > 0:
        food_pos_np = np.array(world.food_positions)
        for agent in prey_agents:
            agent_pos = np.array(agent.position)
            n_eaten = 0
            for fi in range(len(food_pos_np)):
                if fi in food_eaten_indices:
                    continue
                diff = food_pos_np[fi] - agent_pos
                dist = np.linalg.norm(diff)
                if dist > prey_radius:
                    continue

                # Check if food is in forward arc (120 degree FOV)
                angle_to_food = math.atan2(diff[1], diff[0])
                angle_diff = abs(_angle_diff(angle_to_food, agent.angle))
                fov_half = math.radians(config["proximity_fov_deg"]) / 2.0
                if angle_diff <= fov_half:
                    n_eaten += 1
                    food_eaten_indices.add(fi)

            if n_eaten > 0:
                eating_events[agent.agent_id] = n_eaten

    # --- Predator catching prey ---
    prey_caught = set()
    for predator in pred_agents:
        pred_pos = np.array(predator.position)
        catches = []

        for prey in prey_agents:
            if prey.agent_id in prey_caught:
                continue
            prey_pos = np.array(prey.position)
            diff = prey_pos - pred_pos
            dist = np.linalg.norm(diff)

            # Check mouth range
            if dist < mouth_range_min or dist > mouth_range_max:
                continue

            # Check mouth angle
            angle_to_prey = math.atan2(diff[1], diff[0])
            angle_diff = abs(_angle_diff(angle_to_prey, predator.angle))
            if angle_diff <= mouth_half_rad:
                catches.append((prey.agent_id, prey.energy))
                prey_caught.add(prey.agent_id)

        if catches:
            eating_events[predator.agent_id] = catches

    return eating_events, food_eaten_indices


def remove_eaten_food(world: WorldState, food_eaten_indices: set) -> WorldState:
    """Remove food items that were eaten this step."""
    if not food_eaten_indices or world.food_positions is None:
        return world

    mask = np.ones(len(world.food_positions), dtype=bool)
    for idx in food_eaten_indices:
        mask[idx] = False

    if mask.sum() > 0:
        world.food_positions = jnp.array(np.array(world.food_positions)[mask])
    else:
        world.food_positions = jnp.zeros((0, 2))

    # Decrease internal counter
    world.food_internal -= len(food_eaten_indices)
    world.food_internal = max(0.0, world.food_internal)

    return world


# ---------------------------------------------------------------------------
# Physics step (placeholder — full phyjax2d integration later)
# ---------------------------------------------------------------------------

def step_physics(world: WorldState, actions: dict, config: dict) -> WorldState:
    """
    Apply one physics step. Placeholder implementation using simple
    differential-drive kinematics until phyjax2d is integrated.

    actions: {agent_id: jnp.ndarray shape (2,)} — [f_left, f_right]
    Motor forces mapped via sigmoid_scale before reaching here.

    Returns updated WorldState with new positions/velocities/angles.
    """
    world_size = config["world_size"]
    max_velocity = 10.0  # MAX_VELOCITY from emevo

    for agent in world.agents:
        aid = agent.agent_id
        action = actions.get(aid, jnp.zeros(2))

        # Map raw actions through sigmoid
        mapped_action = map_actions(action, config)
        f_left = float(mapped_action[0])
        f_right = float(mapped_action[1])

        # Simple differential drive model
        # Translation force = (f_left + f_right) / 2
        # Rotation torque = (f_right - f_left) / (2 * radius)
        radius = config["prey_radius"] if agent.species == 0 else config["predator_radius"]

        force = (f_left + f_right) / 2.0
        torque = (f_right - f_left) / (2.0 * radius)

        # Update angular velocity and angle
        ang_vel = agent.ang_vel + torque * 0.01  # simplified
        ang_vel = max(-math.pi / 10, min(math.pi / 10, ang_vel))
        angle = agent.angle + ang_vel

        # Update velocity
        vx = float(agent.velocity[0]) + force * math.cos(angle) * 0.01
        vy = float(agent.velocity[1]) + force * math.sin(angle) * 0.01

        # Clamp velocity
        speed = math.sqrt(vx**2 + vy**2)
        if speed > max_velocity:
            scale = max_velocity / speed
            vx *= scale
            vy *= scale

        # Update position
        px = float(agent.position[0]) + vx
        py = float(agent.position[1]) + vy

        # Clamp to world boundaries
        px = max(0.0, min(float(world_size), px))
        py = max(0.0, min(float(world_size), py))

        # Apply updates
        agent.position = jnp.array([px, py])
        agent.velocity = jnp.array([vx, vy])
        agent.angle = angle
        agent.ang_vel = ang_vel

    world.step += 1
    return world
