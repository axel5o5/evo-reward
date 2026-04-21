"""
jax_state.py
------------
SimState: the single JAX pytree that holds all simulation state.

Replaces WorldState + list[AgentState] + physics dict with fixed-size
JAX arrays indexed by slot. Everything is JIT-compatible — no Python
dicts, no variable-length lists, no mutable objects.

Population dynamics use is_active masks on fixed-capacity arrays:
  max_agents = prey_cap + predator_cap (default 500)
  food_max = 600
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax import struct

from src.policy import PolicyNetwork, init_policy


@struct.dataclass
class SimState:
    """Complete simulation state as a JAX pytree.

    All arrays have static shapes determined by (max_agents, food_max, rollout_steps, obs_dim).
    Active/inactive elements are tracked via boolean masks.
    """
    # --- Agent identity & lifecycle (max_agents,) ---
    is_active: jnp.ndarray          # bool
    species: jnp.ndarray            # int32: 0=prey, 1=predator
    agent_ids: jnp.ndarray          # int32: unique monotonic ID per agent
    parent_ids: jnp.ndarray         # int32: parent's agent_id (-1 for initial)
    ages: jnp.ndarray               # int32: steps alive
    energies: jnp.ndarray           # float32
    reward_weights: jnp.ndarray     # (max_agents, 4) float32: [w_eat, w_act, w_prey, w_pred]

    # --- Policy params & optimizer (stacked pytrees, each leaf (max_agents, ...)) ---
    policy_params: dict
    policy_opt_states: tuple        # optax state is a tuple of pytrees

    # --- Rollout buffers (max_agents, rollout_steps, ...) ---
    rollout_obs: jnp.ndarray        # (max_agents, rollout_steps, obs_dim)
    rollout_actions: jnp.ndarray    # (max_agents, rollout_steps, 2)
    rollout_log_probs: jnp.ndarray  # (max_agents, rollout_steps)
    rollout_rewards: jnp.ndarray    # (max_agents, rollout_steps)
    rollout_values: jnp.ndarray     # (max_agents, rollout_steps)
    rollout_dones: jnp.ndarray      # (max_agents, rollout_steps) bool
    rollout_ptrs: jnp.ndarray       # (max_agents,) int32

    # --- Food (food_max, ...) ---
    food_positions: jnp.ndarray     # (food_max, 2)
    food_active: jnp.ndarray        # (food_max,) bool
    food_internal: jnp.ndarray      # scalar float32

    # --- Physics engine (phyjax2d pytrees) ---
    phyjax_stated: object           # phyjax2d StateDict
    phyjax_solver: object           # phyjax2d VelocitySolver

    # --- Per-slot physics metadata (max_agents, ...) ---
    act_p1: jnp.ndarray             # (max_agents, 2) force application point 1
    act_p2: jnp.ndarray             # (max_agents, 2) force application point 2
    act_ratio: jnp.ndarray          # (max_agents, 1) force scaling
    radii: jnp.ndarray              # (max_agents,) body radius

    # --- Temporal reward buffer (Axis 3) ---
    obs_buffer: jnp.ndarray         # (max_agents, k, 4) float32: rolling stimuli window

    # --- LSTM hidden state (Axis 4) ---
    lstm_hidden: jnp.ndarray         # (max_agents, 2, lstm_hidden_size) packed (c, h)
    rollout_init_hidden: jnp.ndarray # (max_agents, 2, lstm_hidden_size) h_0 for PPO replay

    # --- Predator eating cooldown (K&D eat_interval, see emevo-diff.md D18) ---
    predator_eat_timer: jnp.ndarray # (max_agents,) int32: counts down; predator can catch
                                    # only when <= 0. Resets to predator_eat_interval on catch.
                                    # Applies only to predator slots; prey values unused.

    # --- RNG & bookkeeping ---
    rng_key: jnp.ndarray            # PRNGKey
    step: jnp.ndarray               # scalar int32
    next_agent_id: jnp.ndarray      # scalar int32, monotonic counter


# ---------------------------------------------------------------------------
# Initialization (pure JAX, no old WorldState dependency)
# ---------------------------------------------------------------------------

def init_simstate(config: dict, rng_key) -> SimState:
    """Create initial SimState from config. Pure JAX — no Python objects."""
    from src.environment import _build_physics, _init_physics_state, AgentState
    import math
    import numpy as np

    max_agents = config["prey_cap"] + config["predator_cap"]
    food_max = config["food_max"]
    rollout_steps = config["rollout_steps"]
    obs_dim = config["obs_dim"]
    prey_radius = config["prey_radius"]
    pred_radius = config["predator_radius"]
    world_size = config["world_size"]

    # --- Initialize agent arrays ---
    n_prey = config["prey_initial"]
    n_pred = config["predator_initial"]
    n_initial = n_prey + n_pred

    rng_key, pos_key, angle_key, genome_key, policy_key = jax.random.split(rng_key, 5)

    # Positions: random within world bounds (with margin)
    margin = max(prey_radius, pred_radius) * 2
    positions = jax.random.uniform(pos_key, (n_initial, 2),
                                   minval=margin, maxval=world_size - margin)
    angles = jax.random.uniform(angle_key, (n_initial,),
                                minval=-jnp.pi, maxval=jnp.pi)

    # Species: first n_prey are prey, rest are predators
    species_init = jnp.concatenate([
        jnp.zeros(n_prey, dtype=jnp.int32),
        jnp.ones(n_pred, dtype=jnp.int32),
    ])

    radii_init = jnp.where(species_init == 0, prey_radius, pred_radius)

    # Genome: N(0, init_std)
    genome_keys = jax.random.split(genome_key, n_initial)
    reward_weights_init = jax.random.normal(genome_keys[0], (n_initial, 4)) * config["reward_weights_init_std"]
    # Re-sample properly per agent
    reward_weights_init = jax.vmap(
        lambda k: jax.random.normal(k, (4,)) * config["reward_weights_init_std"]
    )(genome_keys)

    # Energy
    energies_init = jnp.full(n_initial, config.get("initial_energy", 100.0))

    # --- Pad to max_agents ---
    def pad(arr, target_len, fill=0):
        pad_width = target_len - arr.shape[0]
        if pad_width <= 0:
            return arr[:target_len]
        if arr.ndim == 1:
            return jnp.concatenate([arr, jnp.full(pad_width, fill, dtype=arr.dtype)])
        else:
            return jnp.concatenate([arr, jnp.full((pad_width, *arr.shape[1:]), fill, dtype=arr.dtype)])

    is_active = jnp.concatenate([
        jnp.ones(n_initial, dtype=bool),
        jnp.zeros(max_agents - n_initial, dtype=bool),
    ])
    species_arr = pad(species_init, max_agents)
    agent_ids = jnp.arange(max_agents, dtype=jnp.int32)
    parent_ids = jnp.full(max_agents, -1, dtype=jnp.int32)
    ages = jnp.zeros(max_agents, dtype=jnp.int32)
    energies = pad(energies_init, max_agents)
    reward_weights = pad(reward_weights_init, max_agents)
    radii_arr = pad(radii_init, max_agents)

    # --- Policy params: initialize for all slots ---
    # Initialize one to get tree structure, then tile and re-init active slots
    policy_keys = jax.random.split(policy_key, max_agents)
    dummy_params, dummy_opt = init_policy(policy_keys[0], config)

    # Tile dummy to (max_agents, ...) for each leaf
    all_params = jtu.tree_map(
        lambda p: jnp.tile(p[None, ...], (max_agents, *([1] * p.ndim))),
        dummy_params,
    )
    all_opt = jtu.tree_map(
        lambda p: jnp.tile(p[None, ...], (max_agents, *([1] * p.ndim))),
        dummy_opt,
    )

    # Re-initialize each active agent's params with unique key
    for i in range(n_initial):
        params_i, opt_i = init_policy(policy_keys[i], config)
        all_params = jtu.tree_map(lambda s, v: s.at[i].set(v), all_params, params_i)
        all_opt = jtu.tree_map(lambda s, v: s.at[i].set(v), all_opt, opt_i)

    # --- Rollout buffers ---
    rollout_obs = jnp.zeros((max_agents, rollout_steps, obs_dim))
    rollout_actions = jnp.zeros((max_agents, rollout_steps, 2))
    rollout_log_probs = jnp.zeros((max_agents, rollout_steps))
    rollout_rewards = jnp.zeros((max_agents, rollout_steps))
    rollout_values = jnp.zeros((max_agents, rollout_steps))
    rollout_dones = jnp.zeros((max_agents, rollout_steps), dtype=bool)
    rollout_ptrs = jnp.zeros(max_agents, dtype=jnp.int32)

    # --- Temporal reward buffer (Axis 3) ---
    k = config.get("reward_context_window", 1)
    obs_buffer = jnp.zeros((max_agents, k, 4))

    # --- LSTM hidden state (Axis 4) ---
    lstm_hidden_size = config.get("lstm_hidden_size", 64)
    lstm_hidden = jnp.zeros((max_agents, 2, lstm_hidden_size))
    rollout_init_hidden = jnp.zeros((max_agents, 2, lstm_hidden_size))

    # --- Food ---
    n_food_init = config["food_initial"]
    rng_key, food_key = jax.random.split(rng_key)
    food_pos = jax.random.uniform(food_key, (food_max, 2), minval=0.0, maxval=float(world_size))
    food_active = jnp.arange(food_max) < n_food_init

    # --- Physics (phyjax2d) ---
    # Build the Space and initial state using existing infrastructure
    space, _ = _build_physics(config, n_agent_slots=max_agents)

    # Build phyjax2d stated from our arrays
    stated = space.zeros_state()
    circle_state = stated.get("circle")

    # Set positions for all slots (padding already has zeros)
    all_positions = pad(positions, max_agents)
    all_angles = pad(angles, max_agents)

    import phyjax2d as pj
    circle_state = circle_state.replace(
        p=pj.Position(angle=all_angles, xy=all_positions),
        v=pj.Velocity(angle=jnp.zeros(max_agents), xy=jnp.zeros((max_agents, 2))),
        is_active=is_active,
    )
    stated = stated.replace(circle=circle_state)
    solver = space.init_solver()

    # Force application points
    act_p1_vec = pj.Vec2d(0, prey_radius).rotated(math.pi * 0.75)
    act_p2_vec = pj.Vec2d(0, prey_radius).rotated(-math.pi * 0.75)
    act_p1 = jnp.tile(jnp.array(act_p1_vec), (max_agents, 1))
    act_p2 = jnp.tile(jnp.array(act_p2_vec), (max_agents, 1))

    # Act ratio
    pred_ratio = (pred_radius ** 2) / (prey_radius ** 2)
    act_ratio = jnp.where(
        species_arr[:, None] == 1,
        jnp.full((max_agents, 1), pred_ratio),
        jnp.ones((max_agents, 1)),
    )

    # Predator eat-timer: 0 means "ready to catch". Starts at 0 so initial
    # predators can catch on step 0 (matches emevo's reset value).
    predator_eat_timer = jnp.zeros(max_agents, dtype=jnp.int32)

    return SimState(
        is_active=is_active,
        species=species_arr,
        agent_ids=agent_ids,
        parent_ids=parent_ids,
        ages=ages,
        energies=energies,
        reward_weights=reward_weights,
        policy_params=all_params,
        policy_opt_states=all_opt,
        rollout_obs=rollout_obs,
        rollout_actions=rollout_actions,
        rollout_log_probs=rollout_log_probs,
        rollout_rewards=rollout_rewards,
        rollout_values=rollout_values,
        rollout_dones=rollout_dones,
        rollout_ptrs=rollout_ptrs,
        food_positions=food_pos,
        food_active=food_active,
        food_internal=jnp.float32(n_food_init),
        phyjax_stated=stated,
        phyjax_solver=solver,
        act_p1=act_p1,
        act_p2=act_p2,
        act_ratio=act_ratio,
        radii=radii_arr,
        obs_buffer=obs_buffer,
        lstm_hidden=lstm_hidden,
        rollout_init_hidden=rollout_init_hidden,
        predator_eat_timer=predator_eat_timer,
        rng_key=rng_key,
        step=jnp.int32(0),
        next_agent_id=jnp.int32(n_initial),
    )


# ---------------------------------------------------------------------------
# Conversion: old WorldState ↔ SimState
# ---------------------------------------------------------------------------

def worldstate_to_simstate(world, config: dict) -> SimState:
    """Convert old WorldState + AgentState list to SimState.

    Used during transition for cross-validation tests.
    """
    physics = world.physics
    max_agents = physics["max_agents"]
    food_max = config["food_max"]
    rollout_steps = config["rollout_steps"]
    obs_dim = config["obs_dim"]

    circle_state = physics["stated"].get("circle")

    # Agent arrays from physics dict (already SoA)
    is_active = circle_state.is_active
    species = physics["species"]
    radii = physics["radii"]

    # Agent IDs, parent IDs, ages, energies from agent list
    agent_ids = jnp.zeros(max_agents, dtype=jnp.int32)
    parent_ids = jnp.full(max_agents, -1, dtype=jnp.int32)
    ages = jnp.zeros(max_agents, dtype=jnp.int32)
    energies = jnp.zeros(max_agents)
    reward_weights = jnp.zeros((max_agents, 4))

    # Build indexing arrays
    slots = []
    aid_vals = []
    pid_vals = []
    age_vals = []
    e_vals = []
    rw_vals = []

    for agent in world.agents:
        slot = physics["agent_id_to_slot"].get(agent.agent_id)
        if slot is not None:
            slots.append(slot)
            aid_vals.append(agent.agent_id)
            pid_vals.append(agent.parent_id)
            age_vals.append(agent.age)
            e_vals.append(agent.energy)
            rw_vals.append(jnp.asarray(agent.reward_weights))

    if slots:
        s = jnp.array(slots)
        agent_ids = agent_ids.at[s].set(jnp.array(aid_vals, dtype=jnp.int32))
        parent_ids = parent_ids.at[s].set(jnp.array(pid_vals, dtype=jnp.int32))
        ages = ages.at[s].set(jnp.array(age_vals, dtype=jnp.int32))
        energies = energies.at[s].set(jnp.array(e_vals))
        reward_weights = reward_weights.at[s].set(jnp.stack(rw_vals))

    # Policy params: stack from agents (only active agents have real params)
    # Initialize all to dummy first, then fill active slots
    dummy_params, dummy_opt = init_policy(jax.random.PRNGKey(0), config)

    # Create stacked arrays of dummy params
    all_params = jtu.tree_map(
        lambda p: jnp.tile(p[None, ...], (max_agents, *([1] * p.ndim))),
        dummy_params,
    )
    all_opt = jtu.tree_map(
        lambda p: jnp.tile(p[None, ...], (max_agents, *([1] * p.ndim))),
        dummy_opt,
    )

    # Fill in active agents' real params
    for agent in world.agents:
        slot = physics["agent_id_to_slot"].get(agent.agent_id)
        if slot is not None and agent.policy_params is not None:
            all_params = jtu.tree_map(
                lambda stack, single: stack.at[slot].set(single),
                all_params, agent.policy_params,
            )
            if agent.policy_opt_state is not None:
                all_opt = jtu.tree_map(
                    lambda stack, single: stack.at[slot].set(single),
                    all_opt, agent.policy_opt_state,
                )

    # Rollout buffers: initialize empty (conversion doesn't preserve rollout state)
    rollout_obs = jnp.zeros((max_agents, rollout_steps, obs_dim))
    rollout_actions = jnp.zeros((max_agents, rollout_steps, 2))
    rollout_log_probs = jnp.zeros((max_agents, rollout_steps))
    rollout_rewards = jnp.zeros((max_agents, rollout_steps))
    rollout_values = jnp.zeros((max_agents, rollout_steps))
    rollout_dones = jnp.zeros((max_agents, rollout_steps), dtype=bool)
    rollout_ptrs = jnp.zeros(max_agents, dtype=jnp.int32)

    # Food
    if world.food_positions is not None and len(world.food_positions) > 0:
        n_food = len(world.food_positions)
        food_pos = jnp.array(world.food_positions[:food_max])
        pad_n = food_max - food_pos.shape[0]
        if pad_n > 0:
            food_pos = jnp.concatenate([food_pos, jnp.zeros((pad_n, 2))], axis=0)
        food_active = jnp.zeros(food_max, dtype=bool).at[:n_food].set(True)
    else:
        food_pos = jnp.zeros((food_max, 2))
        food_active = jnp.zeros(food_max, dtype=bool)

    # Next agent ID
    max_aid = max((a.agent_id for a in world.agents), default=-1) + 1

    # Temporal and LSTM fields (zero-initialized for legacy conversion)
    k = config.get("reward_context_window", 1)
    lstm_hidden_size = config.get("lstm_hidden_size", 64)

    return SimState(
        is_active=is_active,
        species=species,
        agent_ids=agent_ids,
        parent_ids=parent_ids,
        ages=ages,
        energies=energies,
        reward_weights=reward_weights,
        policy_params=all_params,
        policy_opt_states=all_opt,
        rollout_obs=rollout_obs,
        rollout_actions=rollout_actions,
        rollout_log_probs=rollout_log_probs,
        rollout_rewards=rollout_rewards,
        rollout_values=rollout_values,
        rollout_dones=rollout_dones,
        rollout_ptrs=rollout_ptrs,
        food_positions=food_pos,
        food_active=food_active,
        food_internal=jnp.float32(world.food_internal),
        phyjax_stated=physics["stated"],
        phyjax_solver=physics["solver"],
        act_p1=physics["act_p1"],
        act_p2=physics["act_p2"],
        act_ratio=physics["act_ratio"],
        radii=radii,
        obs_buffer=jnp.zeros((max_agents, k, 4)),
        lstm_hidden=jnp.zeros((max_agents, 2, lstm_hidden_size)),
        rollout_init_hidden=jnp.zeros((max_agents, 2, lstm_hidden_size)),
        predator_eat_timer=jnp.zeros(max_agents, dtype=jnp.int32),
        rng_key=world.rng_key if world.rng_key is not None else jax.random.PRNGKey(0),
        step=jnp.int32(world.step),
        next_agent_id=jnp.int32(max_aid),
    )
