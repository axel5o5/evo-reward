"""
sac_state.py
------------
SacState: the parallel pytree that holds per-agent SAC state alongside
SimState. Kept separate from SimState so the PPO path (and its
rollout_* fields, policy_params, policy_opt_states) is untouched.

Per-agent indexing matches SimState — slot i in SacState corresponds to
slot i in SimState. All arrays are statically shaped to
(max_agents, ...) so the whole pytree is JIT-compatible.

Fields:
  - Networks (stacked per-agent pytrees):
      actor_params, q1_params, q2_params,
      q1_target_params, q2_target_params
  - Temperature (per-agent scalar):
      log_alpha
  - Optimizer states (stacked per-agent pytrees):
      actor_opt_state, q1_opt_state, q2_opt_state, alpha_opt_state
  - Replay ring (per-agent ring of capacity `replay_capacity`):
      replay_obs (max_agents, capacity, obs_dim)
      replay_action (max_agents, capacity, 2)
      replay_reward (max_agents, capacity)
      replay_next_obs (max_agents, capacity, obs_dim)
      replay_done (max_agents, capacity)
      replay_ptr (max_agents,)  - next write position (mod capacity)
      replay_size (max_agents,) - filled count, capped at capacity

Construction:
  init_sacstate(config, rng_key) -> SacState
  reset_sac_slot(sac_state, slot_idx, rng_key, config) -> SacState
    (used on agent birth — re-initializes one slot's networks +
    optimizers and zeros that slot's replay)
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import struct

from src.sac_networks import init_sac_actor, init_q_network, init_alpha


def _max_agents(config: dict) -> int:
    """Population capacity sum. Matches SimState's max_agents convention."""
    return int(config["prey_cap"]) + int(config["predator_cap"])


def _replay_capacity(config: dict) -> int:
    return int(config.get("sac_replay_capacity", 4096))


# ---------------------------------------------------------------------------
# Pytree
# ---------------------------------------------------------------------------

@struct.dataclass
class SacState:
    # --- Networks (each leaf shape (max_agents, ...)) ---
    actor_params: dict
    q1_params: dict
    q2_params: dict
    q1_target_params: dict
    q2_target_params: dict

    # --- Temperature, per agent ---
    log_alpha: jnp.ndarray              # (max_agents,) float32

    # --- Optimizer states (stacked per-agent pytrees) ---
    actor_opt_state: tuple
    q1_opt_state: tuple
    q2_opt_state: tuple
    alpha_opt_state: tuple

    # --- Replay ring buffer ---
    replay_obs: jnp.ndarray             # (max_agents, capacity, obs_dim)
    replay_action: jnp.ndarray          # (max_agents, capacity, 2)
    replay_reward: jnp.ndarray          # (max_agents, capacity)
    replay_next_obs: jnp.ndarray        # (max_agents, capacity, obs_dim)
    replay_done: jnp.ndarray            # (max_agents, capacity)
    replay_ptr: jnp.ndarray             # (max_agents,) int32
    replay_size: jnp.ndarray            # (max_agents,) int32


# ---------------------------------------------------------------------------
# Stacking helpers
# ---------------------------------------------------------------------------

def _stack(per_agent_trees):
    """Stack a list of pytrees of identical structure along a new axis 0."""
    return jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *per_agent_trees)


def _init_single_agent_nets(rng_key, config: dict):
    """Build one agent's full (actor, q1, q2, q1_target, q2_target,
    log_alpha) + their optimizer states.

    Targets are initialized identical to online (the standard SAC start);
    they'll drift via Polyak as the online critics learn.

    Returns a dict with the 11 leaves expected by `init_sacstate`.
    """
    k_actor, k_q1, k_q2 = jax.random.split(rng_key, 3)
    actor_params, actor_opt = init_sac_actor(k_actor, config)
    q1_params, q1_opt = init_q_network(k_q1, config)
    q2_params, q2_opt = init_q_network(k_q2, config)
    log_alpha, alpha_opt = init_alpha(config)

    return {
        "actor_params": actor_params,
        "q1_params": q1_params,
        "q2_params": q2_params,
        # Targets start as identical copies. tree_map(identity) gets a
        # fresh pytree without sharing storage — important so Polyak
        # updates don't accidentally mutate the online tree.
        "q1_target_params": jtu.tree_map(lambda x: x, q1_params),
        "q2_target_params": jtu.tree_map(lambda x: x, q2_params),
        "log_alpha": log_alpha,
        "actor_opt_state": actor_opt,
        "q1_opt_state": q1_opt,
        "q2_opt_state": q2_opt,
        "alpha_opt_state": alpha_opt,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_sacstate(config: dict, rng_key) -> SacState:
    """Allocate a fresh SacState for `max_agents` slots.

    All slots get newly-initialized networks. The user is expected to
    call `reset_sac_slot` for any slot whose agent is later replaced at
    birth, so each newborn gets fresh weights rather than inheriting the
    previous occupant's policy.

    Replay buffers start at size=0 (warmup will be respected by the
    update fn).
    """
    n = _max_agents(config)
    cap = _replay_capacity(config)
    obs_dim = int(config["obs_dim"])

    keys = jax.random.split(rng_key, n)
    per_agent_dicts = [_init_single_agent_nets(k, config) for k in keys]

    actor_params = _stack([d["actor_params"] for d in per_agent_dicts])
    q1_params = _stack([d["q1_params"] for d in per_agent_dicts])
    q2_params = _stack([d["q2_params"] for d in per_agent_dicts])
    q1_target = _stack([d["q1_target_params"] for d in per_agent_dicts])
    q2_target = _stack([d["q2_target_params"] for d in per_agent_dicts])
    log_alpha = jnp.stack([d["log_alpha"] for d in per_agent_dicts], axis=0)
    actor_opt = _stack([d["actor_opt_state"] for d in per_agent_dicts])
    q1_opt = _stack([d["q1_opt_state"] for d in per_agent_dicts])
    q2_opt = _stack([d["q2_opt_state"] for d in per_agent_dicts])
    alpha_opt = _stack([d["alpha_opt_state"] for d in per_agent_dicts])

    return SacState(
        actor_params=actor_params,
        q1_params=q1_params,
        q2_params=q2_params,
        q1_target_params=q1_target,
        q2_target_params=q2_target,
        log_alpha=log_alpha,
        actor_opt_state=actor_opt,
        q1_opt_state=q1_opt,
        q2_opt_state=q2_opt,
        alpha_opt_state=alpha_opt,
        replay_obs=jnp.zeros((n, cap, obs_dim), dtype=jnp.float32),
        replay_action=jnp.zeros((n, cap, 2), dtype=jnp.float32),
        replay_reward=jnp.zeros((n, cap), dtype=jnp.float32),
        replay_next_obs=jnp.zeros((n, cap, obs_dim), dtype=jnp.float32),
        replay_done=jnp.zeros((n, cap), dtype=jnp.float32),
        replay_ptr=jnp.zeros((n,), dtype=jnp.int32),
        replay_size=jnp.zeros((n,), dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------
#
# Mirrors the SimState checkpoint pattern in scripts/run_experiment_jax.py:
# flatten the pytree to a list of leaves, save each as `leaf_<i>` in an
# .npz. To load, we rebuild a fresh template via init_sacstate and then
# tree_unflatten with the saved leaves. The template is needed because
# the .npz doesn't preserve the pytree structure on its own.
#
# This means: on resume, the user must pass the same config (so the
# template matches the saved shapes). Mismatch → unflatten fails loudly
# rather than silently restoring wrong-shaped state.

def save_sac_state(sac_state: SacState, path) -> None:
    """Flatten SacState to leaves and np.savez. The .npz will contain
    `leaf_0` ... `leaf_N`; the count varies with optimizer + network
    pytree structure."""
    import numpy as np
    leaves = jtu.tree_leaves(sac_state)
    np.savez(
        str(path),
        **{f"leaf_{i}": np.asarray(l) for i, l in enumerate(leaves)},
    )


def load_sac_state(path, config: dict) -> SacState:
    """Rebuild a SacState from a .npz produced by save_sac_state.

    Requires `config` so we can construct a template SacState of the
    right shape (init_sacstate). Verifies the saved leaf count matches
    the template's leaf count and raises ValueError on mismatch — this
    is the common signal that the config drifted between save and load.
    """
    import numpy as np
    template = init_sacstate(config, jax.random.PRNGKey(0))
    full_leaves, treedef = jtu.tree_flatten(template)

    data = np.load(str(path), allow_pickle=False)
    n = sum(1 for k in data.files if k.startswith("leaf_"))
    if n != len(full_leaves):
        raise ValueError(
            f"sac checkpoint {path}: leaf count {n} != template {len(full_leaves)}. "
            f"Config must match the one used at save time."
        )
    loaded = []
    for i, expected in enumerate(full_leaves):
        arr = data[f"leaf_{i}"]
        if arr.shape != expected.shape:
            raise ValueError(
                f"sac checkpoint {path}: leaf_{i} shape {arr.shape} != "
                f"template {expected.shape}. Config must match the one "
                f"used at save time."
            )
        loaded.append(jnp.asarray(arr))
    return jtu.tree_unflatten(treedef, loaded)


def reset_sac_slot(sac_state: SacState, slot_idx, rng_key, config: dict) -> SacState:
    """Re-initialize one slot at agent birth.

    Mirrors the PPO birth path's `init_policy` call but for all 5
    networks + 4 optimizer states + log_alpha. Also clears that slot's
    replay buffer + ptr + size so the newborn doesn't inherit the
    previous occupant's transitions.

    `slot_idx` may be a traced int32 (called from inside jax.jit on
    handle_birth), so we use lax.dynamic_update / .at[].set forms.
    """
    fresh = _init_single_agent_nets(rng_key, config)
    cap = int(sac_state.replay_obs.shape[1])
    obs_dim = int(sac_state.replay_obs.shape[2])

    def set_slot(stacked, leaf_value):
        # leaf_value has the unbatched shape; we set stacked[slot_idx] = leaf_value.
        return stacked.at[slot_idx].set(leaf_value)

    new_actor = jtu.tree_map(set_slot, sac_state.actor_params, fresh["actor_params"])
    new_q1 = jtu.tree_map(set_slot, sac_state.q1_params, fresh["q1_params"])
    new_q2 = jtu.tree_map(set_slot, sac_state.q2_params, fresh["q2_params"])
    new_q1_t = jtu.tree_map(set_slot, sac_state.q1_target_params, fresh["q1_target_params"])
    new_q2_t = jtu.tree_map(set_slot, sac_state.q2_target_params, fresh["q2_target_params"])
    new_log_alpha = sac_state.log_alpha.at[slot_idx].set(fresh["log_alpha"])
    new_actor_opt = jtu.tree_map(set_slot, sac_state.actor_opt_state, fresh["actor_opt_state"])
    new_q1_opt = jtu.tree_map(set_slot, sac_state.q1_opt_state, fresh["q1_opt_state"])
    new_q2_opt = jtu.tree_map(set_slot, sac_state.q2_opt_state, fresh["q2_opt_state"])
    new_alpha_opt = jtu.tree_map(set_slot, sac_state.alpha_opt_state, fresh["alpha_opt_state"])

    new_replay_obs = sac_state.replay_obs.at[slot_idx].set(jnp.zeros((cap, obs_dim)))
    new_replay_action = sac_state.replay_action.at[slot_idx].set(jnp.zeros((cap, 2)))
    new_replay_reward = sac_state.replay_reward.at[slot_idx].set(jnp.zeros((cap,)))
    new_replay_next_obs = sac_state.replay_next_obs.at[slot_idx].set(jnp.zeros((cap, obs_dim)))
    new_replay_done = sac_state.replay_done.at[slot_idx].set(jnp.zeros((cap,)))
    new_replay_ptr = sac_state.replay_ptr.at[slot_idx].set(jnp.int32(0))
    new_replay_size = sac_state.replay_size.at[slot_idx].set(jnp.int32(0))

    return sac_state.replace(
        actor_params=new_actor,
        q1_params=new_q1, q2_params=new_q2,
        q1_target_params=new_q1_t, q2_target_params=new_q2_t,
        log_alpha=new_log_alpha,
        actor_opt_state=new_actor_opt,
        q1_opt_state=new_q1_opt, q2_opt_state=new_q2_opt,
        alpha_opt_state=new_alpha_opt,
        replay_obs=new_replay_obs,
        replay_action=new_replay_action,
        replay_reward=new_replay_reward,
        replay_next_obs=new_replay_next_obs,
        replay_done=new_replay_done,
        replay_ptr=new_replay_ptr,
        replay_size=new_replay_size,
    )
