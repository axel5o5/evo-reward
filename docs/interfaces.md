# Module Interfaces and Data Contracts
### `evo-reward` — Phase 0 ground truth

*This document pins every inter-module boundary before any code is written. A coding agent implementing any module must produce outputs that match these contracts exactly. If the emevo source code deviates from anything here, emevo wins — update this document and note the deviation in `docs/emevo-diff.md`.*

---

## Core Data Structures

### `AgentState`

Complete runtime state of one agent.

```python
@dataclass
class AgentState:
    # Identity
    agent_id:       int            # unique, monotonically increasing across births
    species:        int            # 0 = prey, 1 = predator
    parent_id:      int            # -1 if initial population

    # Physics (managed by phyjax2d — do not write directly)
    position:       jnp.ndarray   # shape (2,), float32, world units in [0, 960]
    velocity:       jnp.ndarray   # shape (2,), float32, world units/step
    angle:          float          # radians, [-π, π]
    ang_vel:        float          # radians/step

    # Lifecycle
    age:            int            # steps lived, incremented each step
    energy:         float          # current energy, float32

    # Genome — THE ONLY HERITABLE COMPONENT
    reward_weights: jnp.ndarray   # shape (4,), float32
                                   # FIXED ORDER: [w_eat, w_act, w_prey, w_pred]
                                   # prey agent:     w_prey = conspecifics (other prey)
                                   #                 w_pred = predators
                                   # predator agent: w_prey = prey (food source)
                                   #                 w_pred = other predators

    # Policy — NOT inherited, reset fresh at every birth
    policy_params:     PyTree      # Flax parameter tree for this agent's MLP
    policy_opt_state:  PyTree      # Adam optimizer state

    # PPO rollout buffer — reset at birth, filled over N=1024 steps
    rollout:        RolloutBuffer
```

**The `[w_eat, w_act, w_prey, w_pred]` ordering is canonical and immutable.** It must be identical in reward computation, mutation, logging, checkpointing, and visualization. Never permute it.

---

### `RolloutBuffer`

One agent's PPO experience. Allocated at birth, written step-by-step.

```python
@dataclass
class RolloutBuffer:
    observations:  jnp.ndarray   # shape (N, obs_dim), float32
    actions:       jnp.ndarray   # shape (N, 2), float32 — [f_left, f_right]
    log_probs:     jnp.ndarray   # shape (N,), float32
    rewards:       jnp.ndarray   # shape (N,), float32
    values:        jnp.ndarray   # shape (N,), float32
    dones:         jnp.ndarray   # shape (N,), bool — True only at death
    ptr:           int            # next write index, 0..N-1
    full:          bool           # True once ptr has wrapped
```

`N = config["rollout_steps"] = 1024`. `obs_dim` from config (see Observation Vector section).

---

### `WorldState`

Complete simulation state at one timestep.

```python
@dataclass
class WorldState:
    step:           int                     # global step counter
    agents:         list[AgentState]        # all living agents (prey + predator mixed)
    food_internal:  float                   # real-valued food counter n_t
    food_positions: jnp.ndarray            # shape (n_active_food, 2), float32
    rng_key:        jax.random.PRNGKey
```

`agents` is a Python list (dynamic length). For JIT-compiled inner loops, convert to fixed-capacity masked arrays inside the compiled function — do not expose that representation at this interface level.

---

### `Checkpoint`

Saved to disk every `checkpoint_interval_steps`.

```python
@dataclass
class Checkpoint:
    step:           int
    config:         dict
    seed:           int
    # Agent state serialized as flat arrays (not list of AgentState)
    agent_ids:      np.ndarray    # (n_agents,) int
    species:        np.ndarray    # (n_agents,) int
    ages:           np.ndarray    # (n_agents,) int
    energies:       np.ndarray    # (n_agents,) float32
    reward_weights: np.ndarray    # (n_agents, 4) float32
    parent_ids:     np.ndarray    # (n_agents,) int
    metrics_slice:  dict          # metrics logged since last checkpoint
```

Policy parameters are NOT saved per-step checkpoint (too large). Save separately at generation boundaries only if behavioral analysis requires it.

---

### `MetricsLog`

Accumulated during training; saved to `metrics.npz` at run end.

```python
@dataclass
class MetricsLog:
    # Logged every log_interval_steps (default 10,000)
    steps:                 list[int]
    prey_population:       list[int]
    predator_population:   list[int]
    prey_mean_energy:      list[float]
    predator_mean_energy:  list[float]
    # Reward weight trajectories (reproduces K&D Figure 7)
    prey_mean_w_eat:       list[float]
    prey_mean_w_act:       list[float]
    prey_mean_w_prey:      list[float]
    prey_mean_w_pred:      list[float]
    prey_std_w_eat:        list[float]
    prey_std_w_act:        list[float]
    prey_std_w_prey:       list[float]
    prey_std_w_pred:       list[float]
    pred_mean_w_eat:       list[float]
    pred_mean_w_act:       list[float]
    pred_mean_w_prey:      list[float]
    pred_mean_w_pred:      list[float]
    pred_std_w_eat:        list[float]
    pred_std_w_act:        list[float]
    pred_std_w_prey:       list[float]
    pred_std_w_pred:       list[float]
    # Ecological metrics
    capture_rate:          list[float]       # prey caught per step, rolling avg
    food_consumption_rate: list[float]       # food eaten per step, rolling avg
    # Birth log for phylogenetic tree reconstruction
    birth_log: list[tuple[int, int, int]]   # (step, child_id, parent_id)
```

---

## Observation Vector

**The most critical contract in the codebase.** Every module touching observations uses this exact layout. Do not reorder without updating this document, the config `obs_dim`, and all tests.

### Baseline (`social_obs: position_only`)

**CONFIRMED against emevo gecco2026 branch** (`circle_foraging.py:740-748`, `circle_foraging_with_predator.py:466-473`). The observation is a flattened NamedTuple with **per-type sensor channels**. In emevo, it is structured as `CFObs(sensor, collision, velocity, angle, angular_velocity, energy)` and flattened via `as_array()`.

```
Index    Field                   Shape   Range      Notes
─────────────────────────────────────────────────────────────────────────────
0–127    proximity_sensors       (32,4)  [-1, 1]    32 sensors × 4 per-type channels.
                                                    Channel order: [prey, predator, food, wall].
                                                    Winner-take-all: only closest type positive;
                                                    others = -1.0. Flattened row-major:
                                                    [s0_prey, s0_pred, s0_food, s0_wall, s1_prey...].
                                                    FOV: 120° ("wide"), range: 200 units.

128–199  tactile_collision       (4,18)  {0, 1}     4 per-type channels × 18 bins at 20° spacing.
                                                    Channel order: [conspecific, other_species,
                                                    food, wall]. Binary contact per type per bin.
                                                    For prey: conspecific=other prey,
                                                    other_species=predator. Vice versa for predators.

200–201  velocity                (2,)    [-10, 10]  2D velocity (vx, vy), world units/step.
                                                    MAX_VELOCITY = 10.0.

202      angle                   (1,)    [-2π, 2π]  Agent heading in radians.

203      angular_velocity        (1,)    [-π/10,π/10] Angular velocity, radians/step.

204      energy                  (1,)    [0, 1000]  Raw energy value (not normalized).
                                                    Capped at energy_capacity = 1000.0.
─────────────────────────────────────────────────────────────────────────────
TOTAL    obs_dim = 205  (128 + 72 + 2 + 1 + 1 + 1)
```

**Reward stimulus extraction** (for the 4-weight linear reward genome):
- `max_s_prey`: max (or mean) over 32 sensors of channel 0 (prey), clipped ≥0. Source: `cf_predator.py:60-69`.
- `max_s_pred`: max (or mean) over 32 sensors of channel 1 (predator), clipped ≥0.
- Note: emevo default uses **mean** over sensors (`sensor_agg_type="mean"`), not max. The paper describes "most prominent" which may suggest max. Need to confirm which was used for the paper experiments.

```yaml
obs_dim: 205   # CONFIRMED: 128 + 72 + 2 + 1 + 1 + 1
```

### Extension: Social Observation (`social_obs: position_heading_velocity`)

Appended after the baseline block. Baseline indices do not change.

```
Index       Field                       Notes
─────────────────────────────────────────────────────────────
205         conspecific_1_heading       radians [-π, π], 0-padded if fewer
206         conspecific_1_speed         ‖v‖, 0-padded if fewer
207         conspecific_2_heading
208         conspecific_2_speed
...         up to N_max_neighbors pairs, zero-padded
─────────────────────────────────────────────────────────────
```

**This extension is NOT built in Phase 0.** The baseline layout is designed so extensions only append — never insert into existing indices.

---

## Module Function Signatures

### `environment.py`

```python
def init_world(config: dict, rng_key: jax.random.PRNGKey) -> WorldState:
    """
    Initialize world:
    - config["prey_initial"] prey at random positions, energy = config["prey_e_initial"]
    - config["predator_initial"] predators at random positions
    - config["food_max"] food items at random positions
    - food_internal = config["food_max"]
    All agents: reward_weights ~ N(0, config["reward_weights_init_std"])
                policy_params = fresh random init
    """

def step_physics(world: WorldState, actions: dict[int, jnp.ndarray]) -> WorldState:
    """
    Apply one phyjax2d physics step.
    actions: {agent_id: jnp.ndarray shape (2,)} — [f_left, f_right] for all living agents.
    Motor output clipped to [-20, 80] per component before physics step.
    Returns updated WorldState with new positions/velocities/angles.
    Does NOT handle eating, energy, or birth/death.
    """

def get_sensor_readings(world: WorldState, agent_id: int) -> dict:
    """
    Returns raw sensor readings for one agent:
    {
        "proximity": jnp.ndarray shape (32,),  # inverse distance, [0,1]
        "tactile":   jnp.ndarray shape (18,),  # binary contact
    }
    """

def check_eating(world: WorldState) -> dict[int, int]:
    """
    Check all eating/catching events this step.
    Prey: eats food item if within 120° forward range contact.
    Predator: catches prey if within mouth range (40–80 units, 60° arc).
    Returns: {agent_id: n_items_consumed}
    Does NOT modify world state — lifecycle.py applies energy changes.
    """
```

---

### `agents.py`

```python
def get_observation(world: WorldState, agent_id: int, config: dict) -> jnp.ndarray:
    """
    Build full observation vector for one agent.
    Returns: shape (config["obs_dim"],), float32
    Exact layout: see Observation Vector section above.
    config["social_obs"] controls whether social channels are appended.
    """

def get_stimulus_scalars(world: WorldState, agent_id: int) -> dict[str, float]:
    """
    Extract the four reward-relevant scalars from the current world state.
    Returns:
    {
        "n_eaten":    int,    # items consumed this step (from check_eating result)
        "motor_norm": float,  # ‖f_this_step‖ / F, normalized
        "max_s_prey": float,  # max proximity reading for conspecifics
        "max_s_pred": float,  # max proximity reading for opposing species
    }
    Separated from compute_reward so both linear and MLP genomes can use it.
    Caller must pass in the eating result from check_eating().
    """

def compute_reward(
    genome: jnp.ndarray,       # shape (4,) reward weights
    stimuli: dict[str, float], # from get_stimulus_scalars()
) -> float:
    """
    Apply K&D reward equation:
        r = w_eat * n_eaten
          + 0.01 * w_act * motor_norm
          + 0.1  * w_prey * max_s_prey
          + 0.1  * w_pred * max_s_pred

    genome order: [w_eat, w_act, w_prey, w_pred]
    Returns: float scalar
    """
```

---

### `reward.py`

```python
def init_genome(rng_key: jax.random.PRNGKey, config: dict) -> jnp.ndarray:
    """
    Returns: shape (4,), float32
    Each weight ~ N(0, config["reward_weights_init_std"]) = N(0, 0.1)
    Order: [w_eat, w_act, w_prey, w_pred]
    """

def compute_linear_reward(
    genome: jnp.ndarray,       # shape (4,)
    n_eaten: int,
    motor_norm: float,         # ‖f‖ / F, already normalized
    max_s_prey: float,
    max_s_pred: float,
) -> float:
    """
    Pure function. All stimuli pre-extracted.
    r = genome[0]*n_eaten + 0.01*genome[1]*motor_norm
      + 0.1*genome[2]*max_s_prey + 0.1*genome[3]*max_s_pred
    """

# Extension stubs — raise NotImplementedError in Phase 0
def init_mlp_genome(rng_key, config): raise NotImplementedError("Phase 2")
def compute_mlp_reward(genome, stimuli): raise NotImplementedError("Phase 2")
def init_temporal_genome(rng_key, config): raise NotImplementedError("Phase 2")
def compute_temporal_reward(genome, obs_window): raise NotImplementedError("Phase 2")
```

---

### `evolution.py`

```python
def mutate_genome(
    parent_genome: jnp.ndarray,   # shape (4,)
    rng_key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """
    child = clip(parent + delta, -config["weight_clip"], config["weight_clip"])
    delta ~ StudentT(df=config["mutation_df"], scale=config["mutation_scale"])
         = StudentT(df=2, scale=0.4)
    Sampled independently per weight.

    Implementation note: use scipy.stats.t(df=2, scale=0.4).rvs(4) for
    correctness verification in tests. For JAX JIT, use the JAX equivalent
    or a pre-sampled noise array. Do NOT use jax.random.normal() — the
    distribution must have heavy tails (df=2, not Gaussian).

    Returns: shape (4,), float32
    """

def spawn_offspring(
    parent: AgentState,
    new_id: int,
    rng_key: jax.random.PRNGKey,
    config: dict,
) -> AgentState:
    """
    Create offspring AgentState:
    - agent_id:       new_id
    - species:        parent.species
    - parent_id:      parent.agent_id
    - position:       N(parent.position, config["spawn_spread"] * I)
    - velocity:       zeros
    - angle:          uniform [-π, π]
    - ang_vel:        0.0
    - age:            0
    - energy:         parent.energy * config["energy_share_ratio"]
    - reward_weights: mutate_genome(parent.reward_weights, rng_key)
    - policy_params:  fresh random init (NOT copied from parent)
    - policy_opt_state: fresh init
    - rollout:        fresh empty RolloutBuffer
    """
```

---

### `lifecycle.py`

```python
def hazard_prob(age: int, energy: float, species: int, config: dict) -> float:
    """
    h(t, e) = κ_h * (1 - 1/(1 + α_e·exp(-β_h·e))) * α_t·exp(β_t·t)

    Parameters from config, species-specific where noted:
        κ_h    = config["kappa_h"]          (same for both)
        α_e    = config["alpha_e"]          (same for both)
        β_h    = config["beta_h"]           (same for both)
        α_t    = config["alpha_t_prey"] or config["alpha_t_pred"]
        β_t    = config["beta_t_prey"] or config["beta_t_pred"]

    Returns: float in [0, 1], probability of death this step.
    """

def birth_prob(energy: float, species: int, config: dict) -> float:
    """
    b(e) = κ_b / (1 + exp(ζ - β_b·e))

    Parameters:
        κ_b = config["kappa_b"]
        β_b = config["beta_b"]
        ζ   = config["zeta_b_prey"] or config["zeta_b_pred"]

    Returns: float in [0, 1], probability of reproduction this step.
    """

def update_energies(
    world: WorldState,
    eating_events: dict[int, int],  # {agent_id: n_consumed}
    actions_taken: dict[int, jnp.ndarray],  # {agent_id: action}
    config: dict,
) -> WorldState:
    """
    Apply energy update equation to all agents:
      Prey:     Δe = n_eaten * e_food - c_a * ‖action‖ - c_b
      Predator: Δe = Σ(η * prey_energy_at_catch) - d_a * ‖action‖ - d_b
    Predator energy gain requires knowing prey energy at moment of catch.
    Pass prey energy via eating_events or as a separate dict.
    Returns updated WorldState (agents with e < 0 NOT yet removed — that
    happens in process_births_and_deaths).
    """

def process_births_and_deaths(
    world: WorldState,
    rng_key: jax.random.PRNGKey,
    config: dict,
) -> tuple[WorldState, list[int], list[int]]:
    """
    1. Kill agents: remove if e < 0; else kill with prob h(age, energy).
    2. Reproduce agents: for each surviving agent, reproduce with prob b(energy).
       - Check population cap before creating offspring.
       - Parent loses energy * energy_share_ratio.
       - Offspring created via spawn_offspring().
    3. Enforce caps: if at cap, skip reproduction (do not kill existing agents).

    Returns:
        WorldState: updated (dead agents removed, newborns added)
        list[int]:  IDs of agents that died this step
        list[int]:  IDs of agents born this step
    """

def regenerate_food(world: WorldState, config: dict) -> WorldState:
    """
    n_{t+1} = min(n_t + g - n_eaten_this_step, n_max)
    Spawn new food items at random positions when floor(n_{t+1}) > current item count.
    g = config["food_growth_rate"] = 0.02
    n_max = config["food_max"] = 100
    """
```

---

### `policy.py`

```python
def init_policy(rng_key: jax.random.PRNGKey, config: dict) -> tuple[PyTree, PyTree]:
    """
    Fresh 3-layer MLP policy:
    - Input:   config["obs_dim"]
    - Hidden:  [config["policy_hidden_size"]] * (config["policy_n_layers"] - 1)
               = [64, 64] with tanh activations
    - Policy head:  2 output units (action means)
    - Value head:   1 output unit (state value estimate)
    - log_std:      learned parameter, shape (2,), state-independent
    Returns: (params, opt_state) where opt_state is Adam state for PPO updates.
    """

def sample_action(
    params: PyTree,
    obs: jnp.ndarray,
    rng_key: jax.random.PRNGKey,
) -> tuple[jnp.ndarray, float, float]:
    """
    Forward pass + action sample.
    Returns: (action, log_prob, value_estimate)
    action shape: (2,), float32, clipped to [-20, 80] per component.
    """

def policy_forward(
    params: PyTree,
    obs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Deterministic forward pass (no sampling).
    Returns: (action_mean: (2,), log_std: (2,), value: float)
    """

# Extension stub
def init_lstm_policy(rng_key, config): raise NotImplementedError("Phase 2")
def sample_action_lstm(params, obs, hidden_state, rng_key): raise NotImplementedError("Phase 2")
```

---

### `ppo.py`

```python
def compute_gae(
    rewards: jnp.ndarray,    # (N,)
    values: jnp.ndarray,     # (N,)
    dones: jnp.ndarray,      # (N,) bool
    last_value: float,
    config: dict,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generalized Advantage Estimation.
    last_value = V(s_{N+1}); use 0.0 if agent died at end of rollout.
    Returns: (advantages (N,), returns (N,))
    γ = config["gamma"] = 0.999
    λ = config["gae_lambda"] = 0.95
    """

def ppo_update(
    params: PyTree,
    opt_state: PyTree,
    rollout: RolloutBuffer,
    config: dict,
) -> tuple[PyTree, PyTree, dict]:
    """
    Run PPO update on a completed rollout.
    - config["ppo_epochs"] = 10 passes over the buffer
    - Minibatch size: config["minibatch_size"] = 256
    - Clip epsilon: config["clip_epsilon"] = 0.2
    - Entropy coef: config["entropy_coef"] = 0.001
    - Adam lr: config["lr"] = 3e-4, eps: config["adam_eps"] = 1e-7
    Returns: (new_params, new_opt_state, info)
    info keys: "policy_loss", "value_loss", "entropy", "approx_kl"
    """
```

---

### `metrics.py`

```python
def log_step(log: MetricsLog, world: WorldState, config: dict) -> MetricsLog:
    """Append current step's aggregate metrics. Returns new log (immutable)."""

def record_birth(log: MetricsLog, step: int, child_id: int, parent_id: int) -> MetricsLog:
    """Append birth event to birth_log. Returns new log."""

def save_checkpoint(world: WorldState, log: MetricsLog, config: dict, seed: int, out_dir: str) -> None:
    """
    Path: {out_dir}/{experiment_name}/seed_{seed}/step_{step:08d}.pkl
    """

def save_metrics(log: MetricsLog, config: dict, seed: int, out_dir: str) -> None:
    """
    Path: {out_dir}/{experiment_name}/seed_{seed}/metrics.npz
    All list fields saved as numpy arrays with their field names as keys.
    """

def load_checkpoint(path: str) -> Checkpoint: ...
def load_metrics(path: str) -> MetricsLog: ...
```

---

## Simulation Loop (main.py / run_experiment.py)

The outer loop ties everything together in this exact order per step:

```python
for step in range(config["total_steps"]):
    # 1. Collect observations and sample actions for all agents
    actions = {}
    for agent in world.agents:
        obs = agents.get_observation(world, agent.agent_id, config)
        action, log_prob, value = policy.sample_action(agent.policy_params, obs, rng)
        actions[agent.agent_id] = action
        # Write to rollout buffer
        agent.rollout = write_to_buffer(agent.rollout, obs, action, log_prob, value)

    # 2. Physics step
    world = environment.step_physics(world, actions)

    # 3. Check eating events
    eating_events = environment.check_eating(world)

    # 4. Compute rewards and write to rollout
    for agent in world.agents:
        stimuli = agents.get_stimulus_scalars(world, agent.agent_id)
        stimuli["n_eaten"] = eating_events.get(agent.agent_id, 0)
        stimuli["motor_norm"] = jnp.linalg.norm(actions[agent.agent_id]) / config["max_motor_norm"]
        r = agents.compute_reward(agent.reward_weights, stimuli)
        agent.rollout = write_reward_to_buffer(agent.rollout, r)

    # 5. Energy update
    world = lifecycle.update_energies(world, eating_events, actions, config)

    # 6. Birth and death
    world, dead_ids, born_ids = lifecycle.process_births_and_deaths(world, rng, config)

    # 7. Food regeneration
    world = lifecycle.regenerate_food(world, config)

    # 8. PPO update (when rollout is full)
    for agent in world.agents:
        if agent.rollout.full:
            last_obs = agents.get_observation(world, agent.agent_id, config)
            _, _, last_value = policy.policy_forward(agent.policy_params, last_obs)
            agent.policy_params, agent.policy_opt_state, _ = ppo.ppo_update(
                agent.policy_params, agent.policy_opt_state, agent.rollout, config
            )
            agent.rollout = reset_rollout_buffer(config)

    # 9. Log and checkpoint
    if step % config["log_interval_steps"] == 0:
        log = metrics.log_step(log, world, config)
    for child_id, parent_id in zip(born_ids, [world_before_birth.get_parent(c) for c in born_ids]):
        log = metrics.record_birth(log, step, child_id, parent_id)
    if step % config["checkpoint_interval_steps"] == 0:
        metrics.save_checkpoint(world, log, config, seed, out_dir)
```

---

## Configuration Keys Reference

All required keys, their types, and valid ranges:

```python
CONFIG_SCHEMA = {
    # Identity
    "experiment_name":           str,
    "policy_mode":               Literal["independent", "shared"],
    "lifecycle_mode":            Literal["continuous", "generational"],
    "reward_type":               Literal["linear", "mlp"],
    "social_obs":                Literal["position_only", "position_heading_velocity"],
    "policy_type":               Literal["mlp", "lstm"],
    "coevolution_mode":          Literal["concurrent", "alternating"],
    # World
    "world_size":                int,      # 960
    "total_steps":               int,      # 10_240_000
    "obs_dim":                   int,      # 205 — CONFIRMED
    # Population
    "prey_initial":              int,      # 150
    "predator_initial":          int,      # 10
    "prey_cap":                  int,      # 450
    "predator_cap":              int,      # 50
    "prey_e_initial":            float,    # starting energy for initial population
    # Bodies
    "prey_radius":               float,    # 10.0
    "predator_radius":           float,    # 14.0
    "max_motor_norm":            float,    # 114.0
    # Sensors
    "n_proximity_sensors":       int,      # 32
    "proximity_fov_deg":         float,    # 120.0
    "proximity_max_range":       float,    # 120.0
    "n_tactile_sensors":         int,      # 18
    "tactile_spacing_deg":       float,    # 20.0
    # Food
    "food_max":                  int,      # 600 — CONFIRMED
    "food_initial":              int,      # 40 — CONFIRMED
    "food_growth_rate":          float,    # 0.5 — CONFIRMED
    # Energy — prey
    "prey_e_food":               float,    # 1.0
    "prey_c_b":                  float,    # 2.5e-3
    "prey_c_a":                  float,    # 1.0e-4
    # Energy — predator
    "predator_d_b":              float,    # 4.0e-3
    "predator_d_a":              float,    # 5.0e-5
    "predator_eta":              float,    # 0.6 — CONFIRMED
    # Predator mouth
    "predator_mouth_deg":        float,    # 60.0 (medium)
    "predator_mouth_range_min":  float,    # 40.0
    "predator_mouth_range_max":  float,    # 80.0
    # Hazard
    "kappa_h":                   float,    # 0.01
    "alpha_e":                   float,    # 0.02
    "beta_h":                    float,    # 0.2
    "alpha_t_prey":              float,    # 4e-7
    "alpha_t_pred":              float,    # 2e-7
    "beta_t_prey":               float,    # 2e-6
    "beta_t_pred":               float,    # 4e-6
    # Birth
    "kappa_b":                   float,    # 1e-3
    "beta_b":                    float,    # 0.1
    "zeta_b_prey":               float,    # 10.0
    "zeta_b_pred":               float,    # 100.0
    # Offspring
    "energy_share_ratio":        float,    # 0.4 — CONFIRMED
    "spawn_spread":              float,    # 100.0 — CONFIRMED
    # Genome
    "reward_weights_init_std":   float,    # 0.1
    "mutation_df":               int,      # 2
    "mutation_scale":            float,    # 0.4
    "weight_clip":               float,    # 100.0
    # Policy
    "policy_hidden_size":        int,      # 64
    "policy_n_layers":           int,      # 3
    # PPO
    "gamma":                     float,    # 0.999
    "rollout_steps":             int,      # 1024
    "minibatch_size":            int,      # 256
    "ppo_epochs":                int,      # 10
    "clip_epsilon":              float,    # 0.2
    "entropy_coef":              float,    # 0.001
    "gae_lambda":                float,    # 0.95
    "lr":                        float,    # 3e-4
    "adam_eps":                  float,    # 1e-7
    # Logging
    "checkpoint_interval_steps": int,      # 25_000
    "log_interval_steps":        int,      # 10_000
    "seed":                      int,
}
```

---

## Runtime Invariants

Checked in tests and as debug assertions:

1. `prey_count <= config["prey_cap"]` and `predator_count <= config["predator_cap"]` at all times
2. All `agent.energy >= 0` at the start of each step
3. All `agent.age` increments by exactly 1 per step
4. All `agent.reward_weights` in `[-weight_clip, weight_clip]`
5. `rollout.ptr` in `[0, rollout_steps)` at all times
6. All observation values are finite (no NaN, no Inf)
7. All reward values are finite
8. Every `agent.agent_id` is unique across all living agents
9. `world.food_internal >= 0.0`
10. `len(world.food_positions)` equals `floor(world.food_internal)` at the start of each step

---

## Open Questions — ALL RESOLVED

All resolved against emevo gecco2026 branch (`github.com/oist/emevo`, branch `gecco2026`). Reference config: `config/env/20251122-predator-square.toml`.

| # | Question | Answer |
|---|----------|--------|
| 1 | Proximity sensors: single or per-type? | **Per-type: 4 channels** [prey, predator, food, wall]. obs_dim contribution: 32×4=128. |
| 2 | Velocity: scalar or 2D? | **2D (vx, vy)**. obs_dim contribution: 2. |
| 3 | Action clipping range? | **[-20, 80]**, same as 2024. Mapped via sigmoid_scale. |
| 4 | Value network architecture? | **Shared trunk** (2 hidden layers, 64 units, tanh), separate heads. |
| 5 | energy_share_ratio? | **0.4**. Parent loses 0.4×e, child gets 0.4×e. |
| 6 | spawn_spread? | **100.0** world units (neighbor_stddev). |
| 7 | Observation normalization? | **None**. Raw observations to network. |
| 8 | Initial population energy? | **100.0** for both prey and predators. |
