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

Physics via phyjax2d — differential drive matching emevo gecco2026 branch.

emevo source references:
  - circle_foraging.py:740-748 (obs_space definition)
  - circle_foraging.py: act_p1/act_p2 force points, Vec2d(0, r).rotated(±0.75π)
  - circle_foraging_with_predator.py:84-111 (_observe_closest)
  - circle_foraging_with_predator.py:466-473 (observation construction)
  - circle_foraging_with_predator.py:401 (action clip)
  - cf_predator.py:122 (sigmoid_scale action mapping)
  - config/env/20251122-predator-square.toml (all parameter values)

phyjax2d physics parameters (from emevo config):
  dt=0.1, linear_damping=0.8, angular_damping=0.6
  n_velocity_iter=6, n_position_iter=2, n_physics_iter=5
  max_velocity=10.0, max_angular_velocity=π
  density=0.1, friction=0.2, elasticity=0.4
  Force points: Vec2d(0, agent_radius).rotated(±0.75π)
  Predator act_ratio: (predator_radius² / prey_radius²)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import phyjax2d as pj


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
    # phyjax2d physics state (set by init_world)
    physics: object = None                 # dict with space, stated, solver, slot maps


# ---------------------------------------------------------------------------
# phyjax2d constants (from emevo gecco2026 config)
# ---------------------------------------------------------------------------

N_PHYSICS_ITER = 5        # sub-steps per simulation step
PHYSICS_DT = 0.1
LINEAR_DAMPING = 0.8
ANGULAR_DAMPING = 0.6
N_VELOCITY_ITER = 6
N_POSITION_ITER = 2
MAX_VELOCITY = 10.0
MAX_ANGULAR_VELOCITY = float(math.pi)
AGENT_DENSITY = 0.1
AGENT_FRICTION = 0.2
AGENT_ELASTICITY = 0.4
FOOD_RADIUS = 4.0         # emevo config: food_radius = 4.0


# ---------------------------------------------------------------------------
# Sensor constants
# ---------------------------------------------------------------------------

# Channel indices for proximity sensors (per sensor)
CHANNEL_PREY = 0
CHANNEL_PREDATOR = 1
CHANNEL_FOOD = 2
CHANNEL_WALL = 3


def world_bounds(config: dict) -> tuple[float, float]:
    """Return (world_size_x, world_size_y) for the simulation domain.

    Supports three config shapes (tried in order):
      * explicit rect: `world_size_x` + `world_size_y`
      * explicit tuple: `world_size` as a 2-item list/tuple [x, y]
      * scalar (square, legacy): `world_size` as a single number

    D25: introduced to support rectangular worlds (e.g. emevo's 1200×600
    predator TOML) without rewriting every call site. When paper-text
    config is used (square 960×960), all three paths return (960, 960).
    """
    if "world_size_x" in config and "world_size_y" in config:
        return float(config["world_size_x"]), float(config["world_size_y"])
    raw = config["world_size"]
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    s = float(raw)
    return s, s
N_CHANNELS = 4

# Channel indices for tactile sensors
TACTILE_CONSPECIFIC = 0
TACTILE_OTHER_SPECIES = 1
TACTILE_FOOD = 2
TACTILE_WALL = 3
N_TACTILE_CHANNELS = 4


# ---------------------------------------------------------------------------
# phyjax2d world building
# ---------------------------------------------------------------------------

def _build_physics(config: dict, n_agent_slots=None):
    """Build the phyjax2d Space with circle slots for agents and wall segments.

    Food is managed in Python (not in phyjax2d) since eating detection
    is distance-based and food doesn't need physics collisions.

    n_agent_slots: number of agent circle slots (default: prey_cap + predator_cap)

    Returns: (space, max_agents)
    """
    max_agents = n_agent_slots or (config.get("prey_cap", 450) + config.get("predator_cap", 50))
    world_x, world_y = world_bounds(config)
    prey_radius = config["prey_radius"]
    predator_radius = config["predator_radius"]

    builder = pj.SpaceBuilder(
        gravity=(0.0, 0.0),
        dt=PHYSICS_DT,
        linear_damping=LINEAR_DAMPING,
        angular_damping=ANGULAR_DAMPING,
        n_velocity_iter=N_VELOCITY_ITER,
        n_position_iter=N_POSITION_ITER,
        max_velocity=MAX_VELOCITY,
        max_angular_velocity=MAX_ANGULAR_VELOCITY,
    )

    # Wall segments. `make_square_segments(xmin, xmax, ymin, ymax)` actually
    # produces a rectangle — the name is legacy. Emevo calls it the same way
    # for its 1200×600 TOML.
    segments = pj.make_square_segments(0.0, world_x, 0.0, world_y)
    for a, b in segments:
        builder.add_segment(p1=a, p2=b, elasticity=AGENT_ELASTICITY, friction=AGENT_FRICTION)

    # Agent circles — prey slots first, then predator slots
    prey_cap = config.get("prey_cap", 450)
    pred_cap = config.get("predator_cap", 50)
    total_cap = prey_cap + pred_cap
    prey_slots = int(max_agents * prey_cap / total_cap)
    pred_slots = max_agents - prey_slots

    for _ in range(prey_slots):
        builder.add_circle(
            radius=prey_radius, density=AGENT_DENSITY,
            friction=AGENT_FRICTION, elasticity=AGENT_ELASTICITY,
        )
    for _ in range(pred_slots):
        builder.add_circle(
            radius=predator_radius, density=AGENT_DENSITY,
            friction=AGENT_FRICTION, elasticity=AGENT_ELASTICITY,
        )

    space = builder.build()
    return space, max_agents


def _init_physics_state(space, agents, max_agents, config):
    """Initialize phyjax2d state from agent list.

    Returns: physics dict with space, stated, solver, slot maps, force points.
    """
    prey_radius = config["prey_radius"]
    pred_radius = config["predator_radius"]

    stated = space.zeros_state()
    solver = space.init_solver()

    # --- Agent positions ---
    circle_state = stated.get("circle")
    positions = jnp.zeros((max_agents, 2))
    angles = jnp.zeros(max_agents)
    velocities_xy = jnp.zeros((max_agents, 2))
    velocities_ang = jnp.zeros(max_agents)
    is_active = jnp.zeros(max_agents, dtype=bool)

    agent_id_to_slot = {}
    slot_to_agent_id = {}
    next_slot = 0

    for agent in agents:
        slot = next_slot
        next_slot += 1
        agent_id_to_slot[agent.agent_id] = slot
        slot_to_agent_id[slot] = agent.agent_id
        positions = positions.at[slot].set(agent.position)
        angles = angles.at[slot].set(agent.angle)
        if agent.velocity is not None:
            velocities_xy = velocities_xy.at[slot].set(agent.velocity)
        velocities_ang = velocities_ang.at[slot].set(agent.ang_vel)
        is_active = is_active.at[slot].set(True)

    circle_state = circle_state.replace(
        p=pj.Position(angle=angles, xy=positions),
        v=pj.Velocity(angle=velocities_ang, xy=velocities_xy),
        is_active=is_active,
    )
    stated = stated.replace(circle=circle_state)

    # --- Force application points (emevo: Vec2d(0, agent_radius).rotated(±0.75π)) ---
    # All agents use prey_radius for force points (matching emevo)
    act_p1_vec = pj.Vec2d(0, prey_radius).rotated(math.pi * 0.75)
    act_p2_vec = pj.Vec2d(0, prey_radius).rotated(-math.pi * 0.75)
    act_p1 = jnp.tile(jnp.array(act_p1_vec), (max_agents, 1))
    act_p2 = jnp.tile(jnp.array(act_p2_vec), (max_agents, 1))

    # --- Act ratio: 1.0 for prey, (pred_r²/prey_r²) for predators ---
    # Predator slots start after prey slots in the circle array
    act_ratio = jnp.ones((max_agents, 1))
    pred_ratio = (pred_radius ** 2) / (prey_radius ** 2)
    # --- Per-slot species, radii, energies (for vectorized observation computation) ---
    species_arr = jnp.zeros(max_agents, dtype=jnp.int32)
    radii_arr = jnp.zeros(max_agents)

    # Apply pred_ratio to slots occupied by predators
    for agent in agents:
        slot = agent_id_to_slot[agent.agent_id]
        species_arr = species_arr.at[slot].set(agent.species)
        radii_arr = radii_arr.at[slot].set(pred_radius if agent.species == 1 else prey_radius)
        if agent.species == 1:
            act_ratio = act_ratio.at[slot].set(pred_ratio)

    # Track free slots
    free_slots = list(range(next_slot, max_agents))

    return {
        "space": space,
        "stated": stated,
        "solver": solver,
        "agent_id_to_slot": agent_id_to_slot,
        "slot_to_agent_id": slot_to_agent_id,
        "free_slots": free_slots,
        "act_p1": act_p1,
        "act_p2": act_p2,
        "act_ratio": act_ratio,
        "max_agents": max_agents,
        "species": species_arr,
        "radii": radii_arr,
    }


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

    # --- Build phyjax2d physics ---
    # Use full capacity for agent slots (population can grow to caps)
    space, max_agents = _build_physics(config)
    physics = _init_physics_state(space, agents, max_agents, config)

    return WorldState(
        step=0,
        agents=agents,
        food_internal=float(food_initial),
        food_positions=food_positions,
        rng_key=rng_key,
        physics=physics,
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
    Compute 32-sensor x 4-channel proximity readings (vectorized).

    Each sensor covers a slice of the 120-degree forward FOV.
    For each sensor, find the closest object of each type.
    Winner-take-all: only the closest type gets a positive value (inverse
    distance scaled to [0,1]); all other channels = -1.0.

    Returns: shape (32, 4).

    emevo source: circle_foraging_with_predator.py:84-111 _observe_closest()
    """
    n_sensors = config["n_proximity_sensors"]
    fov_rad = math.radians(config["proximity_fov_deg"])
    max_range = config["proximity_max_range"]
    agent_heading = agent.angle

    half_fov = fov_rad / 2.0
    bin_width = fov_rad / n_sensors
    # D31: proximity sensors follow phyjax2d's heading convention where
    # heading=0 points along world +y (forward). Add +pi/2 so the FOV is
    # centered on forward, matching emevo's _get_sensors local +y rays.
    sensor_centers = np.array([
        agent_heading + math.pi / 2.0 - half_fov + (i + 0.5) * bin_width
        for i in range(n_sensors)
    ])
    sensor_half_width = bin_width / 2.0 + 1e-9

    agent_pos = np.array(agent.position, dtype=np.float64)
    agent_radius = config["prey_radius"] if agent.species == 0 else config["predator_radius"]

    closest_dist = np.full((n_sensors, N_CHANNELS), np.inf)

    # --- Vectorized scan of other agents ---
    others = [a for a in world.agents if a.agent_id != agent.agent_id]
    if others:
        other_pos = np.array([np.array(a.position) for a in others], dtype=np.float64)
        other_species = np.array([a.species for a in others])
        other_radii = np.where(other_species == 0, config["prey_radius"], config["predator_radius"])

        diffs = other_pos - agent_pos[np.newaxis, :]
        dists = np.linalg.norm(diffs, axis=1)
        edge_dists = dists - agent_radius - other_radii
        edge_dists = np.maximum(edge_dists, 0.0)

        angles_to = np.arctan2(diffs[:, 1], diffs[:, 0])
        channels = np.where(other_species == 0, CHANNEL_PREY, CHANNEL_PREDATOR)

        # Filter by range
        in_range = edge_dists <= max_range
        for idx in np.where(in_range)[0]:
            angle = angles_to[idx]
            ed = edge_dists[idx]
            ch = channels[idx]
            # Compute angle diff to each sensor and find the bin
            adiffs = (angle - sensor_centers + math.pi) % (2 * math.pi) - math.pi
            bin_idx = np.argmin(np.abs(adiffs))
            if abs(adiffs[bin_idx]) <= sensor_half_width:
                if ed < closest_dist[bin_idx, ch]:
                    closest_dist[bin_idx, ch] = ed

    # --- Vectorized scan of food ---
    if world.food_positions is not None and len(world.food_positions) > 0:
        food_pos = np.array(world.food_positions, dtype=np.float64)
        diffs = food_pos - agent_pos[np.newaxis, :]
        dists = np.linalg.norm(diffs, axis=1)
        edge_dists = np.maximum(dists - agent_radius, 0.0)

        in_range = edge_dists <= max_range
        if np.any(in_range):
            angles_to = np.arctan2(diffs[in_range, 1], diffs[in_range, 0])
            eds = edge_dists[in_range]
            for i in range(len(angles_to)):
                angle = angles_to[i]
                ed = eds[i]
                adiffs = (angle - sensor_centers + math.pi) % (2 * math.pi) - math.pi
                bin_idx = np.argmin(np.abs(adiffs))
                if abs(adiffs[bin_idx]) <= sensor_half_width:
                    if ed < closest_dist[bin_idx, CHANNEL_FOOD]:
                        closest_dist[bin_idx, CHANNEL_FOOD] = ed

    # --- Vectorized wall scan ---
    world_size = config["world_size"]
    cos_sa = np.cos(sensor_centers)
    sin_sa = np.sin(sensor_centers)

    wall_dists = np.full(n_sensors, np.inf)
    # Right wall (x=world_size)
    mask = cos_sa > 1e-9
    wall_dists[mask] = np.minimum(wall_dists[mask], (world_size - agent_pos[0]) / cos_sa[mask])
    # Left wall (x=0)
    mask = cos_sa < -1e-9
    wall_dists[mask] = np.minimum(wall_dists[mask], -agent_pos[0] / cos_sa[mask])
    # Top wall (y=world_size)
    mask = sin_sa > 1e-9
    wall_dists[mask] = np.minimum(wall_dists[mask], (world_size - agent_pos[1]) / sin_sa[mask])
    # Bottom wall (y=0)
    mask = sin_sa < -1e-9
    wall_dists[mask] = np.minimum(wall_dists[mask], -agent_pos[1] / sin_sa[mask])

    wall_edge = np.maximum(wall_dists - agent_radius, 0.0)
    in_range = wall_edge <= max_range
    closest_dist[in_range, CHANNEL_WALL] = np.minimum(
        closest_dist[in_range, CHANNEL_WALL], wall_edge[in_range]
    )

    # --- Winner-take-all: convert to readings ---
    readings = np.full((n_sensors, N_CHANNELS), -1.0, dtype=np.float32)
    min_channels = np.argmin(closest_dist, axis=1)
    min_dists = closest_dist[np.arange(n_sensors), min_channels]
    detected = min_dists <= max_range
    inv_dist = np.clip(1.0 - min_dists[detected] / max_range, 0.0, 1.0)
    readings[np.where(detected)[0], min_channels[detected]] = inv_dist

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
        # D26: rotate into agent's local frame and shift by -π/2 because
        # phyjax2d convention is heading=0 → forward=+y (not +x).
        rel_angle = (angle_to - agent.angle - math.pi / 2.0) % (2 * math.pi)

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

        # Find bin via boundary rule: bin k covers [k·Δ, (k+1)·Δ).
        b = min(int(rel_angle / math.radians(bin_spacing_deg)), n_bins - 1)
        readings[channel, b] = 1.0

    # --- Check food contacts ---
    if world.food_positions is not None and len(world.food_positions) > 0:
        food_pos_np = np.array(world.food_positions)
        for fi in range(len(food_pos_np)):
            diff = food_pos_np[fi] - agent_pos
            dist = np.linalg.norm(diff)
            if dist > agent_radius:
                continue

            angle_to = math.atan2(diff[1], diff[0])
            rel_angle = (angle_to - agent.angle - math.pi / 2.0) % (2 * math.pi)
            b = min(int(rel_angle / math.radians(bin_spacing_deg)), n_bins - 1)
            readings[TACTILE_FOOD, b] = 1.0

    # --- Check wall contacts (D26: rotate into agent's local frame) ---
    world_x, world_y = world_bounds(config)
    bin_spacing_rad = math.radians(bin_spacing_deg)

    def _set_wall_bin(wall_angle_world: float) -> None:
        rel = (wall_angle_world - agent.angle - math.pi / 2.0) % (2 * math.pi)
        b = min(int(rel / bin_spacing_rad), n_bins - 1)
        readings[TACTILE_WALL, b] = 1.0

    if agent_pos[0] <= agent_radius:                  # left wall
        _set_wall_bin(math.pi)
    if agent_pos[0] >= world_x - agent_radius:        # right wall
        _set_wall_bin(0.0)
    if agent_pos[1] <= agent_radius:                  # bottom wall
        _set_wall_bin(-math.pi / 2)
    if agent_pos[1] >= world_y - agent_radius:        # top wall
        _set_wall_bin(math.pi / 2)

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
    Check all eating/catching events this step (vectorized).

    Prey: eats food if within contact distance and forward FOV.
    Predator: catches prey if within mouth range and angle.

    Uses NumPy broadcasting instead of nested Python loops.
    At 150 agents + 600 food: ~2ms vs ~200ms for the loop version.

    Returns: (eating_events, food_eaten_indices)
        eating_events: {agent_id: n_food_eaten (int) for prey,
                        agent_id: [(prey_id, prey_energy)] for predators}
        food_eaten_indices: set of food indices that were eaten
    """
    prey_radius = config["prey_radius"]
    mouth_deg = config.get("predator_mouth_deg", 60.0)
    mouth_range_min = config.get("predator_mouth_range_min", 40.0)
    mouth_range_max = config.get("predator_mouth_range_max", 80.0)
    mouth_half_rad = math.radians(mouth_deg) / 2.0
    fov_half = math.radians(config["proximity_fov_deg"]) / 2.0

    eating_events = {}
    food_eaten_indices = set()

    prey_agents = [a for a in world.agents if a.species == 0]
    pred_agents = [a for a in world.agents if a.species == 1]

    # --- Prey eating food (vectorized) ---
    if prey_agents and world.food_positions is not None and len(world.food_positions) > 0:
        food_pos = np.array(world.food_positions)
        n_food = len(food_pos)
        n_prey = len(prey_agents)

        prey_pos = np.array([np.array(a.position) for a in prey_agents])    # (n_prey, 2)
        prey_ang = np.array([a.angle for a in prey_agents])                  # (n_prey,)

        # Pairwise: (n_prey, n_food, 2) → distances (n_prey, n_food)
        diffs = food_pos[None, :, :] - prey_pos[:, None, :]
        dists = np.linalg.norm(diffs, axis=-1)

        # Contact check: dist <= prey_radius
        contact = dists <= prey_radius

        # FOV check: angle to food within ±fov_half of heading
        angles_to = np.arctan2(diffs[:, :, 1], diffs[:, :, 0])
        angle_diffs = np.abs((angles_to - prey_ang[:, None] + np.pi) % (2 * np.pi) - np.pi)
        in_fov = angle_diffs <= fov_half

        valid = contact & in_fov  # (n_prey, n_food)

        # Deduplication: each food eaten by nearest valid prey
        valid_dists = np.where(valid, dists, np.inf)
        nearest_prey = np.argmin(valid_dists, axis=0)         # (n_food,)
        food_has_eater = np.min(valid_dists, axis=0) < np.inf  # (n_food,)

        eaten_mask = food_has_eater
        if np.any(eaten_mask):
            eaten_indices = np.where(eaten_mask)[0]
            prey_eating = nearest_prey[eaten_mask]
            prey_eat_count = np.bincount(prey_eating, minlength=n_prey)

            for pi in range(n_prey):
                if prey_eat_count[pi] > 0:
                    eating_events[prey_agents[pi].agent_id] = int(prey_eat_count[pi])
            food_eaten_indices = set(eaten_indices.tolist())

    # --- Predator catching prey (vectorized) ---
    if pred_agents and prey_agents:
        n_pred = len(pred_agents)
        n_prey = len(prey_agents)

        pred_pos = np.array([np.array(a.position) for a in pred_agents])
        pred_ang = np.array([a.angle for a in pred_agents])
        prey_pos = np.array([np.array(a.position) for a in prey_agents])

        diffs = prey_pos[None, :, :] - pred_pos[:, None, :]     # (n_pred, n_prey, 2)
        dists = np.linalg.norm(diffs, axis=-1)                   # (n_pred, n_prey)

        # Mouth range check
        in_range = (dists >= mouth_range_min) & (dists <= mouth_range_max)

        # Mouth angle check
        angles_to = np.arctan2(diffs[:, :, 1], diffs[:, :, 0])
        angle_diffs = np.abs((angles_to - pred_ang[:, None] + np.pi) % (2 * np.pi) - np.pi)
        in_mouth = angle_diffs <= mouth_half_rad

        valid = in_range & in_mouth  # (n_pred, n_prey)

        # Deduplication: each prey caught by nearest valid predator
        valid_dists = np.where(valid, dists, np.inf)
        nearest_pred = np.argmin(valid_dists, axis=0)              # (n_prey,)
        prey_has_catcher = np.min(valid_dists, axis=0) < np.inf    # (n_prey,)

        if np.any(prey_has_catcher):
            caught_indices = np.where(prey_has_catcher)[0]
            for ci in caught_indices:
                pi = int(nearest_pred[ci])
                prey_a = prey_agents[int(ci)]
                pred_a = pred_agents[pi]
                if pred_a.agent_id not in eating_events:
                    eating_events[pred_a.agent_id] = []
                eating_events[pred_a.agent_id].append(
                    (prey_a.agent_id, prey_a.energy)
                )

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
# Physics step via phyjax2d (differential drive matching emevo)
# ---------------------------------------------------------------------------

def _make_jit_stepper(space):
    """Create a JIT-compiled physics stepper for the given space.

    Returns a function: (stated, solver, act_p1, act_p2, f1, f2) -> (stated, solver)
    that applies forces and runs N_PHYSICS_ITER sub-steps.
    """
    @jax.jit
    def _jit_step(stated, solver, act_p1, act_p2, f1, f2):
        circle = stated.get("circle")
        circle = circle.apply_force_local(act_p1, f1)
        circle = circle.apply_force_local(act_p2, f2)
        stated = stated.replace(circle=circle)

        def body(carry, _):
            st, sol = carry
            st, sol, _contact = pj.step(space, st, sol)
            return (st, sol), None
        (stated, solver), _ = jax.lax.scan(body, (stated, solver), None, length=N_PHYSICS_ITER)
        return stated, solver

    return _jit_step


def step_physics(world: WorldState, actions: dict, config: dict) -> WorldState:
    """
    Apply one simulation step of physics via phyjax2d (JIT-compiled).

    Differential drive: two rear force points at Vec2d(0, agent_radius).rotated(±0.75π).
    Forces are in the local y-direction: f = [0, action_value * act_ratio].
    5 phyjax2d sub-steps per simulation step (n_physics_iter=5).

    actions: {agent_id: jnp.ndarray shape (2,)} — already sigmoid-scaled motor forces.

    Returns updated WorldState with new positions/velocities/angles.
    """
    physics = world.physics
    stated = physics["stated"]
    space = physics["space"]
    solver = physics["solver"]
    agent_id_to_slot = physics["agent_id_to_slot"]
    act_p1 = physics["act_p1"]
    act_p2 = physics["act_p2"]
    act_ratio = physics["act_ratio"]
    max_agents = physics["max_agents"]

    # Build action array: shape (max_agents, 2), zeros for inactive
    action_arr = jnp.zeros((max_agents, 2))
    for agent in world.agents:
        aid = agent.agent_id
        slot = agent_id_to_slot.get(aid)
        if slot is None:
            continue
        action = actions.get(aid, jnp.zeros(2))
        action_arr = action_arr.at[slot].set(action)

    # Construct forces in local frame: f = [0, action_value * act_ratio]
    f1_raw = action_arr[:, 0:1] * act_ratio
    f2_raw = action_arr[:, 1:2] * act_ratio
    f1 = jnp.concatenate([jnp.zeros_like(f1_raw), f1_raw], axis=1)
    f2 = jnp.concatenate([jnp.zeros_like(f2_raw), f2_raw], axis=1)

    # Get or create JIT-compiled stepper
    if "_jit_step" not in physics:
        physics["_jit_step"] = _make_jit_stepper(space)
    jit_step = physics["_jit_step"]

    stated, solver = jit_step(stated, solver, act_p1, act_p2, f1, f2)

    # Read back positions, velocities, angles to AgentState objects
    circle_state = stated.get("circle")
    for agent in world.agents:
        slot = agent_id_to_slot.get(agent.agent_id)
        if slot is None:
            continue
        agent.position = circle_state.p.xy[slot]
        agent.velocity = circle_state.v.xy[slot]
        agent.angle = float(circle_state.p.angle[slot])
        agent.ang_vel = float(circle_state.v.angle[slot])

    # Store updated physics state
    physics["stated"] = stated
    physics["solver"] = solver

    world.step += 1
    return world


# ---------------------------------------------------------------------------
# Physics state synchronization (after births/deaths)
# ---------------------------------------------------------------------------

def sync_physics_after_population_change(world: WorldState, dead_ids: list, newborns: list, config: dict):
    """
    Update phyjax2d state after agents die or are born.

    dead_ids: list of agent_ids that died
    newborns: list of AgentState objects that were just born
    """
    physics = world.physics
    if physics is None:
        return

    stated = physics["stated"]
    agent_id_to_slot = physics["agent_id_to_slot"]
    slot_to_agent_id = physics["slot_to_agent_id"]
    free_slots = physics["free_slots"]
    max_agents = physics["max_agents"]

    circle_state = stated.get("circle")
    positions = circle_state.p.xy
    angles = circle_state.p.angle
    velocities_xy = circle_state.v.xy
    velocities_ang = circle_state.v.angle
    is_active = circle_state.is_active

    # Deactivate dead agents
    for aid in dead_ids:
        slot = agent_id_to_slot.pop(aid, None)
        if slot is not None:
            slot_to_agent_id.pop(slot, None)
            is_active = is_active.at[slot].set(False)
            velocities_xy = velocities_xy.at[slot].set(jnp.zeros(2))
            velocities_ang = velocities_ang.at[slot].set(0.0)
            free_slots.append(slot)

    # Activate newborns
    prey_radius = config["prey_radius"]
    pred_radius = config["predator_radius"]
    pred_ratio = (pred_radius ** 2) / (prey_radius ** 2)
    act_ratio = physics["act_ratio"]
    species_arr = physics["species"]
    radii_arr = physics["radii"]
    for agent in newborns:
        if not free_slots:
            break
        slot = free_slots.pop(0)
        agent_id_to_slot[agent.agent_id] = slot
        slot_to_agent_id[slot] = agent.agent_id
        positions = positions.at[slot].set(agent.position)
        angles = angles.at[slot].set(agent.angle)
        velocities_xy = velocities_xy.at[slot].set(jnp.zeros(2))
        velocities_ang = velocities_ang.at[slot].set(0.0)
        is_active = is_active.at[slot].set(True)
        species_arr = species_arr.at[slot].set(agent.species)
        radii_arr = radii_arr.at[slot].set(pred_radius if agent.species == 1 else prey_radius)
        if agent.species == 1:
            act_ratio = act_ratio.at[slot].set(pred_ratio)
        else:
            act_ratio = act_ratio.at[slot].set(1.0)
    physics["act_ratio"] = act_ratio
    physics["species"] = species_arr
    physics["radii"] = radii_arr

    circle_state = circle_state.replace(
        p=pj.Position(angle=angles, xy=positions),
        v=pj.Velocity(angle=velocities_ang, xy=velocities_xy),
        is_active=is_active,
    )
    stated = stated.replace(circle=circle_state)
    physics["stated"] = stated


def sync_physics_food(world: WorldState, config: dict):
    """No-op: food is managed in Python, not in phyjax2d."""
    pass


def extract_obs_state(world: WorldState, config: dict) -> dict:
    """Extract fixed-shape SoA arrays for vectorized observation computation.

    Returns a dict of JAX arrays with static shapes (max_agents, food_max)
    suitable for passing to JIT-compiled observation functions.
    """
    physics = world.physics
    max_agents = physics["max_agents"]
    food_max = config["food_max"]

    circle_state = physics["stated"].get("circle")

    # Build energies array from agent list (batch set, not per-agent .at[].set())
    slots_list = []
    energy_vals = []
    for agent in world.agents:
        s = physics["agent_id_to_slot"].get(agent.agent_id)
        if s is not None:
            slots_list.append(s)
            energy_vals.append(agent.energy)
    energies = jnp.zeros(max_agents)
    if slots_list:
        energies = energies.at[jnp.array(slots_list)].set(jnp.array(energy_vals))

    # Pad food positions to fixed size
    if world.food_positions is not None and len(world.food_positions) > 0:
        n_food = len(world.food_positions)
        food_pos = jnp.array(world.food_positions[:food_max])
        pad = food_max - food_pos.shape[0]
        if pad > 0:
            food_pos = jnp.concatenate([food_pos, jnp.zeros((pad, 2))], axis=0)
        food_active = jnp.zeros(food_max, dtype=bool).at[:n_food].set(True)
    else:
        food_pos = jnp.zeros((food_max, 2))
        food_active = jnp.zeros(food_max, dtype=bool)

    return {
        "positions": circle_state.p.xy,            # (max_agents, 2)
        "angles": circle_state.p.angle,             # (max_agents,)
        "velocities_xy": circle_state.v.xy,         # (max_agents, 2)
        "velocities_ang": circle_state.v.angle,     # (max_agents,)
        "is_active": circle_state.is_active,        # (max_agents,) bool
        "species": physics["species"],              # (max_agents,) int32
        "radii": physics["radii"],                  # (max_agents,)
        "energies": energies,                       # (max_agents,)
        "food_positions": food_pos,                 # (food_max, 2)
        "food_active": food_active,                 # (food_max,) bool
        "max_agents": max_agents,
    }
