"""
lifecycle.py
------------
Energy updates, hazard h(t,e), birth b(e), birth/death processing,
and food regeneration.

All formulas match emevo gecco2026 branch source code.
Energy cost parameters use CODE values (not paper Table 2):
  prey c_b = 1e-4, c_a = 2.5e-6
  predator d_b = 4e-3, d_a = 5e-5
"""

import math

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Hazard (death probability per step)
# ---------------------------------------------------------------------------

def hazard_prob(age: int, energy: float, species: int, config: dict) -> float:
    """
    h(t, e) = kappa_h * (1 - 1/(1 + alpha_e * exp(-beta_h * e))) * alpha_t * exp(beta_t * t)

    Parameters from config, species-specific where noted:
        kappa_h = config["kappa_h"]
        alpha_e = config["alpha_e"]
        beta_h  = config["beta_h"]
        alpha_t = config["alpha_t_prey"] or config["alpha_t_pred"]
        beta_t  = config["beta_t_prey"] or config["beta_t_pred"]

    Returns: float in [0, 1], clamped.
    """
    kappa_h = config["kappa_h"]
    alpha_e = config["alpha_e"]
    beta_h = config["beta_h"]

    if species == 0:  # prey
        alpha_t = config["alpha_t_prey"]
        beta_t = config["beta_t_prey"]
    else:  # predator
        alpha_t = config["alpha_t_pred"]
        beta_t = config["beta_t_pred"]

    # Energy-dependent term: increases as energy drops
    # 1 - 1/(1 + alpha_e * exp(-beta_h * e))
    energy_term = 1.0 - 1.0 / (1.0 + alpha_e * math.exp(-beta_h * energy))

    # Age-dependent term: increases with age
    age_term = alpha_t * math.exp(beta_t * age)

    h = kappa_h * energy_term * age_term

    # Clamp to valid probability range
    return max(0.0, min(1.0, h))


# ---------------------------------------------------------------------------
# Birth (reproduction probability per step)
# ---------------------------------------------------------------------------

def birth_prob(energy: float, species: int, config: dict) -> float:
    """
    b(e) = kappa_b / (1 + exp(zeta - beta_b * e))

    Parameters:
        kappa_b = config["kappa_b"]
        beta_b  = config["beta_b"]
        zeta    = config["zeta_b_prey"] or config["zeta_b_pred"]

    Returns: float in [0, kappa_b], probability of reproduction this step.
    """
    kappa_b = config["kappa_b"]
    beta_b = config["beta_b"]

    if species == 0:  # prey
        zeta = config["zeta_b_prey"]
    else:  # predator
        zeta = config["zeta_b_pred"]

    exponent = zeta - beta_b * energy
    # Guard against overflow in exp for very negative energy
    if exponent > 700:
        return 0.0
    b = kappa_b / (1.0 + math.exp(exponent))
    return b


# ---------------------------------------------------------------------------
# Energy updates
# ---------------------------------------------------------------------------

def update_energy_prey(
    energy: float,
    n_eaten: int,
    action_norm: float,
    config: dict,
) -> float:
    """
    Prey energy update per step:
      delta_e = n_eaten * e_food - c_a * ||action|| - c_b
    Energy capped at energy_capacity.

    Uses CODE values: c_b = 1e-4, c_a = 2.5e-6.
    """
    e_food = config["prey_e_food"]
    c_b = config["prey_c_b"]
    c_a = config["prey_c_a"]
    cap = config.get("energy_capacity", 1000.0)

    delta = n_eaten * e_food - c_a * action_norm - c_b
    new_energy = energy + delta
    return min(new_energy, cap)


def update_energy_predator(
    energy: float,
    prey_energies_caught: list,
    action_norm: float,
    config: dict,
) -> float:
    """
    Predator energy update per step:
      delta_e = sum(eta * prey_energy_at_catch) - d_a * ||action|| - d_b
    Energy capped at energy_capacity.
    """
    d_b = config["predator_d_b"]
    d_a = config["predator_d_a"]
    eta = config["predator_eta"]
    cap = config.get("energy_capacity", 1000.0)

    food_gain = sum(eta * pe for pe in prey_energies_caught)
    delta = food_gain - d_a * action_norm - d_b
    new_energy = energy + delta
    return min(new_energy, cap)


def update_energies(world, eating_events: dict, actions_taken: dict, config: dict):
    """
    Apply energy update equation to all agents (vectorized).

    eating_events: {agent_id: n_consumed (int)} for prey,
                   or {agent_id: [(prey_id, prey_energy), ...]} for predators.
    actions_taken: {agent_id: jnp.ndarray shape (2,)}

    Returns updated WorldState with new energies.
    Agents with e < 0 are NOT yet removed here.
    """
    import numpy as np

    n = len(world.agents)
    if n == 0:
        return world

    e_food = config["prey_e_food"]
    c_b = config["prey_c_b"]
    c_a = config["prey_c_a"]
    d_b = config["predator_d_b"]
    d_a = config["predator_d_a"]
    eta = config["predator_eta"]
    cap = config.get("energy_capacity", 1000.0)

    # Extract arrays (one Python pass)
    energies = np.empty(n)
    species = np.empty(n, dtype=np.int32)
    action_norms = np.empty(n)
    food_gains = np.zeros(n)

    for i, agent in enumerate(world.agents):
        energies[i] = agent.energy
        species[i] = agent.species
        action = actions_taken.get(agent.agent_id)
        action_norms[i] = float(np.linalg.norm(np.asarray(action))) if action is not None else 0.0

        ev = eating_events.get(agent.agent_id)
        if ev is not None:
            if agent.species == 0:
                food_gains[i] = (ev if isinstance(ev, int) else 0) * e_food
            elif isinstance(ev, list):
                food_gains[i] = eta * sum(pe for _, pe in ev)

    # Vectorized energy update
    prey_mask = species == 0
    delta = np.where(prey_mask,
                     food_gains - c_a * action_norms - c_b,
                     food_gains - d_a * action_norms - d_b)
    new_energies = np.minimum(energies + delta, cap)

    # Write back (one Python pass)
    for i, agent in enumerate(world.agents):
        agent.energy = float(new_energies[i])
        agent.age += 1

    return world


# ---------------------------------------------------------------------------
# Birth and death processing
# ---------------------------------------------------------------------------

def _batch_hazard_prob(ages, energies, species, config):
    """Vectorized hazard probability for all agents. Returns (n,) array."""
    kappa_h = config["kappa_h"]
    alpha_e = config["alpha_e"]
    beta_h = config["beta_h"]

    alpha_t = np.where(species == 0, config["alpha_t_prey"], config["alpha_t_pred"])
    beta_t = np.where(species == 0, config["beta_t_prey"], config["beta_t_pred"])

    energy_term = 1.0 - 1.0 / (1.0 + alpha_e * np.exp(np.clip(-beta_h * energies, -700, 700)))
    age_term = alpha_t * np.exp(np.clip(beta_t * ages, -700, 700))
    h = kappa_h * energy_term * age_term
    return np.clip(h, 0.0, 1.0)


def _batch_birth_prob(energies, species, config):
    """Vectorized birth probability for all agents. Returns (n,) array."""
    kappa_b = config["kappa_b"]
    beta_b = config["beta_b"]
    zeta = np.where(species == 0, config["zeta_b_prey"], config["zeta_b_pred"])

    exponent = np.clip(zeta - beta_b * energies, -700, 700)
    return kappa_b / (1.0 + np.exp(exponent))


def process_births_and_deaths(world, rng_key, config: dict):
    """
    1. Kill agents: remove if e < 0; else kill with prob h(age, energy).
    2. Reproduce agents: for surviving agents, reproduce with prob b(energy).
       - Check population cap before creating offspring.
       - Parent loses energy * energy_share_ratio.
       - Offspring created via spawn_offspring (from evolution.py).
    3. Enforce caps.

    Hazard and birth probabilities are computed in batch (vectorized numpy).
    Only spawn_offspring remains per-agent (creates AgentState objects).

    Returns:
        world: updated WorldState
        dead_ids: list of agent IDs that died
        born_ids: list of agent IDs that were born
    """
    energy_share_ratio = config["energy_share_ratio"]
    prey_cap = config.get("prey_cap", 450)
    predator_cap = config.get("predator_cap", 50)

    n = len(world.agents)
    if n == 0:
        return world, [], []

    # --- Step 1: Death processing (vectorized hazard) ---
    ages = np.array([a.age for a in world.agents])
    energies = np.array([a.energy for a in world.agents])
    species = np.array([a.species for a in world.agents])

    h_all = _batch_hazard_prob(ages, energies, species, config)

    rng_key, death_key = jax.random.split(rng_key)
    death_randoms = np.array(jax.random.uniform(death_key, shape=(n,)))

    dead_mask = (energies < 0) | (death_randoms < h_all)

    dead_ids = [world.agents[i].agent_id for i in range(n) if dead_mask[i]]
    survivors = [world.agents[i] for i in range(n) if not dead_mask[i]]

    # --- Step 2: Birth processing (vectorized birth prob) ---
    n_surv = len(survivors)
    if n_surv == 0:
        world.agents = []
        world.rng_key = rng_key
        return world, dead_ids, []

    surv_energies = np.array([a.energy for a in survivors])
    surv_species = np.array([a.species for a in survivors])

    b_all = _batch_birth_prob(surv_energies, surv_species, config)

    rng_key, birth_key = jax.random.split(rng_key)
    birth_randoms = np.array(jax.random.uniform(birth_key, shape=(n_surv,)))

    wants_birth = birth_randoms < b_all

    # Apply population caps
    prey_count = int(np.sum(surv_species == 0))
    pred_count = int(np.sum(surv_species == 1))

    max_id = max((a.agent_id for a in survivors), default=-1)
    next_id = max_id + 1

    born_ids = []
    newborns = []

    for i in range(n_surv):
        if not wants_birth[i]:
            continue

        agent = survivors[i]
        if agent.species == 0 and prey_count >= prey_cap:
            continue
        if agent.species == 1 and pred_count >= predator_cap:
            continue

        rng_key, spawn_key = jax.random.split(rng_key)

        try:
            from src.evolution import spawn_offspring
            child = spawn_offspring(agent, next_id, spawn_key, config)
        except (ImportError, NotImplementedError):
            child = _minimal_offspring(agent, next_id, spawn_key, config)

        agent.energy -= agent.energy * energy_share_ratio

        newborns.append(child)
        born_ids.append(next_id)
        next_id += 1

        if agent.species == 0:
            prey_count += 1
        else:
            pred_count += 1

    world.agents = survivors + newborns
    world.rng_key = rng_key

    return world, dead_ids, born_ids


def _minimal_offspring(parent, new_id, rng_key, config):
    """Fallback offspring creation when evolution.py is not available."""
    from dataclasses import dataclass, field

    # Import AgentState if available, otherwise create a simple namespace
    try:
        from src.environment import AgentState
    except (ImportError, AttributeError):
        @dataclass
        class AgentState:
            agent_id: int = 0
            species: int = 0
            parent_id: int = -1
            position: object = None
            velocity: object = None
            angle: float = 0.0
            ang_vel: float = 0.0
            age: int = 0
            energy: float = 0.0
            reward_weights: object = None
            policy_params: object = None
            policy_opt_state: object = None
            rollout: object = None

    spawn_spread = config.get("spawn_spread", 100.0)
    world_size = config.get("world_size", 960)
    energy_share_ratio = config["energy_share_ratio"]

    k1, k2, k3, k4 = jax.random.split(rng_key, 4)

    # Position: Gaussian around parent, clamped to world
    parent_pos = parent.position if parent.position is not None else jnp.array([world_size / 2, world_size / 2])
    offset = jax.random.normal(k1, shape=(2,)) * spawn_spread
    child_pos = jnp.clip(parent_pos + offset, 0.0, float(world_size))

    # Angle: uniform
    child_angle = float(jax.random.uniform(k2, minval=-math.pi, maxval=math.pi))

    # Reward weights: mutated from parent (simple Gaussian fallback)
    parent_weights = parent.reward_weights if parent.reward_weights is not None else jnp.zeros(4)
    delta = jax.random.normal(k3, shape=(4,)) * config.get("mutation_scale", 0.4)
    child_weights = jnp.clip(
        parent_weights + delta,
        -config.get("weight_clip", 100.0),
        config.get("weight_clip", 100.0),
    )

    return AgentState(
        agent_id=new_id,
        species=parent.species,
        parent_id=parent.agent_id,
        position=child_pos,
        velocity=jnp.zeros(2),
        angle=child_angle,
        ang_vel=0.0,
        age=0,
        energy=parent.energy * energy_share_ratio,
        reward_weights=child_weights,
        policy_params=None,
        policy_opt_state=None,
        rollout=None,
    )


# ---------------------------------------------------------------------------
# Food regeneration
# ---------------------------------------------------------------------------

def regenerate_food(world, config: dict):
    """
    n_{t+1} = min(n_t + g, n_max)

    Where g = config["food_growth_rate"] = 0.5 per step.
    n_max = config["food_max"] = 600.

    Spawn new food items at random positions when floor(n_{t+1}) > current
    item count. Maximum new items per step: config["food_max_regen_per_step"] = 10.
    """
    growth_rate = config["food_growth_rate"]
    food_max = config["food_max"]
    max_regen = config.get("food_max_regen_per_step", 10)
    world_size = config.get("world_size", 960)

    # Update internal counter
    new_internal = min(world.food_internal + growth_rate, float(food_max))

    # How many food items should exist
    target_count = int(new_internal)
    current_count = len(world.food_positions) if world.food_positions is not None else 0

    # Spawn new items (up to max_regen per step)
    n_to_spawn = min(target_count - current_count, max_regen)
    n_to_spawn = max(0, n_to_spawn)

    if n_to_spawn > 0:
        rng_key, spawn_key = jax.random.split(world.rng_key)
        new_positions = jax.random.uniform(
            spawn_key,
            shape=(n_to_spawn, 2),
            minval=0.0,
            maxval=float(world_size),
        )
        if world.food_positions is not None and current_count > 0:
            world.food_positions = jnp.concatenate(
                [world.food_positions, new_positions], axis=0
            )
        else:
            world.food_positions = new_positions
        world.rng_key = rng_key

    world.food_internal = new_internal
    return world
