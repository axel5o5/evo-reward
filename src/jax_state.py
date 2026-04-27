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
    # Axis 1 / Axis 3: MLP/temporal reward genome stacked per-agent.
    # Empty dict {} when reward_type == "linear" (no pytree leaves, so tree
    # ops over SimState skip it). For reward_type == "mlp", a Flax params
    # PyTree where each leaf has shape (max_agents, ...). Same pattern as
    # policy_params.
    reward_mlp_params: dict
    # Same shape contract as reward_mlp_params, but for the (k*4 → h → h → 1)
    # temporal MLP genome (Axis 3). Empty {} for linear/mlp runs.
    reward_temporal_params: dict

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

    # --- Cumulative event counters (D21 logging) ---
    # Monotonic int32 counters. The runner diffs them against a stashed
    # "last log" snapshot to produce per-interval event counts in progress.json.
    # Reading them forces a host sync once per log_interval (alongside the
    # population block) so no per-step cost.
    cum_catches: jnp.ndarray        # scalar int32: prey killed by predators
    cum_deaths: jnp.ndarray         # scalar int32: all deaths (catches + hazard + starvation)
    cum_feedings: jnp.ndarray       # scalar int32: prey×food contact events


# ---------------------------------------------------------------------------
# Initialization (pure JAX, no old WorldState dependency)
# ---------------------------------------------------------------------------

def init_simstate(config: dict, rng_key) -> SimState:
    """Create initial SimState from config. Pure JAX — no Python objects."""
    from src.environment import _build_physics, _init_physics_state, AgentState
    import math
    import numpy as np

    prey_cap = config["prey_cap"]
    pred_cap = config["predator_cap"]
    max_agents = prey_cap + pred_cap
    food_max = config["food_max"]
    rollout_steps = config["rollout_steps"]
    obs_dim = config["obs_dim"]
    prey_radius = config["prey_radius"]
    pred_radius = config["predator_radius"]
    from src.environment import world_bounds
    world_x, world_y = world_bounds(config)

    # --- Slot layout (D19 fix) ------------------------------------------------
    # Species are bound to disjoint slot ranges, matching the physics builder:
    #   prey slots:     [0, prey_cap)                 — physics radius 10
    #   predator slots: [prey_cap, prey_cap+pred_cap) — physics radius 14
    # The initial population fills the LOW end of each range; higher indices
    # in each range are inactive reserves for future births. This keeps the
    # species invariant tied to slot-index, not just the SimState.species
    # field, so collisions and act_ratio/inertia all agree with the body size.
    n_prey = config["prey_initial"]
    n_pred = config["predator_initial"]
    n_initial = n_prey + n_pred
    assert n_prey <= prey_cap, "prey_initial > prey_cap"
    assert n_pred <= pred_cap, "predator_initial > predator_cap"

    # Active slots (Python lists, used for per-slot initializations below).
    prey_slots = list(range(n_prey))
    pred_slots = list(range(prey_cap, prey_cap + n_pred))
    active_slots = prey_slots + pred_slots  # length = n_initial

    rng_key, pos_key, angle_key, genome_key, policy_key = jax.random.split(rng_key, 5)

    # Positions: random within world bounds (with margin), only for active slots.
    margin = max(prey_radius, pred_radius) * 2
    positions_initial = jax.random.uniform(
        pos_key, (n_initial, 2),
        minval=jnp.array([margin, margin]),
        maxval=jnp.array([world_x - margin, world_y - margin]),
    )
    angles_initial = jax.random.uniform(
        angle_key, (n_initial,), minval=-jnp.pi, maxval=jnp.pi,
    )

    # Species array keyed by SLOT, not by order-of-initial-population.
    slot_idx = jnp.arange(max_agents, dtype=jnp.int32)
    species_arr = jnp.where(slot_idx < prey_cap, 0, 1).astype(jnp.int32)
    radii_arr = jnp.where(species_arr == 0, prey_radius, pred_radius).astype(jnp.float32)

    # is_active: True at the low indices of each species range.
    is_active = (
        (slot_idx < n_prey)
        | ((slot_idx >= prey_cap) & (slot_idx < prey_cap + n_pred))
    )

    # Scatter initial positions/angles into the full (max_agents,) arrays.
    active_slots_arr = jnp.asarray(active_slots, dtype=jnp.int32)
    all_positions = jnp.zeros((max_agents, 2)).at[active_slots_arr].set(positions_initial)
    all_angles = jnp.zeros(max_agents).at[active_slots_arr].set(angles_initial)

    # Agent IDs: first n_initial ids assigned to active slots in order.
    # next_agent_id (below) continues from n_initial, so no collisions with
    # future births.
    agent_ids = jnp.zeros(max_agents, dtype=jnp.int32).at[active_slots_arr].set(
        jnp.arange(n_initial, dtype=jnp.int32)
    )
    parent_ids = jnp.full(max_agents, -1, dtype=jnp.int32)
    ages = jnp.zeros(max_agents, dtype=jnp.int32)

    # Energies: initial_energy for active slots, 0 elsewhere.
    initial_energy = config.get("initial_energy", 100.0)
    energies = jnp.where(is_active, initial_energy, 0.0).astype(jnp.float32)

    # Reward-weight genomes: N(0, init_std) for active slots; zeros elsewhere
    # (masking by is_active downstream makes the exact value inconsequential,
    # but zeros keep logged stats clean).
    genome_keys = jax.random.split(genome_key, n_initial)
    active_reward_weights = jax.vmap(
        lambda k: jax.random.normal(k, (4,)) * config["reward_weights_init_std"]
    )(genome_keys)
    reward_weights = jnp.zeros((max_agents, 4)).at[active_slots_arr].set(active_reward_weights)

    # --- Reward-MLP genome (Axis 1) -----------------------------------------
    # Linear runs leave this as {} (no leaves; tree ops no-op). MLP runs
    # stack Flax-init'd params for every slot so the array shape stays
    # static — same pattern as policy_params below.
    reward_type = config.get("reward_type", "linear")
    reward_mlp_params: dict = {}
    reward_temporal_params: dict = {}
    if reward_type == "linear":
        pass
    elif reward_type == "mlp":
        from src.reward import init_mlp_genome
        rng_key, mlp_key = jax.random.split(rng_key)
        mlp_keys = jax.random.split(mlp_key, max_agents)
        reward_mlp_params = jax.vmap(
            lambda k: init_mlp_genome(k, config)
        )(mlp_keys)
    elif reward_type == "temporal":
        from src.reward import init_temporal_genome
        rng_key, t_key = jax.random.split(rng_key)
        t_keys = jax.random.split(t_key, max_agents)
        reward_temporal_params = jax.vmap(
            lambda k: init_temporal_genome(k, config)
        )(t_keys)
    else:
        raise ValueError(f"reward_type {reward_type!r} not recognized")

    # --- Policy params: tile a dummy, then re-init active slots ---------------
    policy_keys = jax.random.split(policy_key, n_initial)
    dummy_params, dummy_opt = init_policy(jax.random.PRNGKey(0), config)

    all_params = jtu.tree_map(
        lambda p: jnp.tile(p[None, ...], (max_agents, *([1] * p.ndim))),
        dummy_params,
    )
    all_opt = jtu.tree_map(
        lambda p: jnp.tile(p[None, ...], (max_agents, *([1] * p.ndim))),
        dummy_opt,
    )

    # Re-initialize each active agent's params with a unique key.
    for local_i, slot in enumerate(active_slots):
        params_i, opt_i = init_policy(policy_keys[local_i], config)
        all_params = jtu.tree_map(lambda s, v: s.at[slot].set(v), all_params, params_i)
        all_opt = jtu.tree_map(lambda s, v: s.at[slot].set(v), all_opt, opt_i)

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
    food_pos = jax.random.uniform(
        food_key, (food_max, 2),
        minval=jnp.array([0.0, 0.0]),
        maxval=jnp.array([world_x, world_y]),
    )
    food_active = jnp.arange(food_max) < n_food_init

    # --- Physics (phyjax2d) ---
    # Build the Space and initial state using existing infrastructure
    space, _ = _build_physics(config, n_agent_slots=max_agents)

    # Build phyjax2d stated from our arrays
    stated = space.zeros_state()
    circle_state = stated.get("circle")

    # Positions/angles were scattered into full-length arrays above; nothing
    # more to do here beyond handing them to the phyjax2d circle state.

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
        reward_mlp_params=reward_mlp_params,
        reward_temporal_params=reward_temporal_params,
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
        cum_catches=jnp.int32(0),
        cum_deaths=jnp.int32(0),
        cum_feedings=jnp.int32(0),
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
        reward_mlp_params={},
        reward_temporal_params={},
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
        cum_catches=jnp.int32(0),
        cum_deaths=jnp.int32(0),
        cum_feedings=jnp.int32(0),
    )
