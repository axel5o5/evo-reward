"""
test_sac_integration.py
-----------------------
Verifies the chunk-2 pieces (SacState pytree + sac_runtime.py) work
together end-to-end. These tests stand in for the eventual sim_step
integration — they drive the runtime hooks on synthetic transitions
and check the state evolves correctly.

What's covered:
  - init_sacstate produces a SacState with all expected shapes.
  - sample_actions yields actions in [-20, 80] range, one per slot.
  - write_transitions writes to the right rows at the right ptrs and
    correctly bumps ptr / size for active agents only.
  - Ring wraparound: after capacity writes, ptr returns to 0 and size
    caps at capacity.
  - reset_sac_slot zeros that slot's replay and randomizes its actor.
  - Full pipeline: init → 200 (sample, write, update) cycles drives
    critic MSE down on a stationary reward.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.sac_state import init_sacstate, reset_sac_slot, save_sac_state, load_sac_state
from src.sac_runtime import build_sac_runtime
from src.sac_networks import QNetwork


# ---------------------------------------------------------------------------
# Test config — small so things fit and run fast.
# ---------------------------------------------------------------------------

def _config():
    return {
        "prey_cap": 4,
        "predator_cap": 2,
        "obs_dim": 6,
        "policy_hidden_size": 16,
        "lr": 3e-4,
        "adam_eps": 1e-7,
        "gamma": 0.0,                     # collapse Bellman target → r (for crisp critic test)
        "sac_actor_lr": 3e-4,
        "sac_critic_lr": 3e-3,
        "sac_alpha_lr": 0.0,
        "sac_target_tau": 0.05,
        "sac_minibatch_size": 16,
        "sac_replay_min_size": 8,
        "sac_replay_capacity": 32,        # tiny so ring wraparound test is cheap
        "sac_target_entropy": -2.0,
        "sac_initial_log_alpha": -20.0,   # alpha ≈ 0 for clean critic test
        "lr_schedule_enable": False,
        "lr_prey_multiplier": 1.0,
        "lr_pred_multiplier": 1.0,
    }


def _max_agents(cfg):
    return cfg["prey_cap"] + cfg["predator_cap"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_init_sacstate_shapes():
    cfg = _config()
    n = _max_agents(cfg)
    cap = cfg["sac_replay_capacity"]
    obs_dim = cfg["obs_dim"]

    state = init_sacstate(cfg, jax.random.PRNGKey(0))

    # Replay shapes.
    assert state.replay_obs.shape == (n, cap, obs_dim)
    assert state.replay_action.shape == (n, cap, 2)
    assert state.replay_reward.shape == (n, cap)
    assert state.replay_next_obs.shape == (n, cap, obs_dim)
    assert state.replay_done.shape == (n, cap)
    assert state.replay_ptr.shape == (n,) and state.replay_ptr.dtype == jnp.int32
    assert state.replay_size.shape == (n,) and state.replay_size.dtype == jnp.int32
    # Replays start empty.
    assert int(state.replay_size.sum()) == 0
    assert int(state.replay_ptr.sum()) == 0

    # log_alpha is per-agent scalar.
    assert state.log_alpha.shape == (n,)

    # Networks: each leaf should have a leading max_agents dim.
    for name in ("actor_params", "q1_params", "q2_params",
                 "q1_target_params", "q2_target_params"):
        leaves = jax.tree_util.tree_leaves(getattr(state, name))
        for lf in leaves:
            assert lf.shape[0] == n, f"{name}: leading dim {lf.shape[0]} != {n}"


def test_sample_actions_in_range():
    cfg = _config()
    n = _max_agents(cfg)
    state = init_sacstate(cfg, jax.random.PRNGKey(1))
    runtime = build_sac_runtime(cfg)

    obs = jax.random.normal(jax.random.PRNGKey(2), (n, cfg["obs_dim"]))
    actions = runtime["sample_actions"](state, obs, jax.random.PRNGKey(3))

    assert actions.shape == (n, 2)
    a = np.asarray(actions)
    # sigmoid_scale maps to (-20, 80) open interval. Allow tiny tolerance
    # for fp edge cases.
    assert a.min() > -20.0 - 1e-3, f"action below -20: {a.min()}"
    assert a.max() < 80.0 + 1e-3, f"action above 80: {a.max()}"


def test_write_transitions_advances_ptr_and_size_for_active_only():
    cfg = _config()
    n = _max_agents(cfg)
    state = init_sacstate(cfg, jax.random.PRNGKey(1))
    runtime = build_sac_runtime(cfg)

    obs = jax.random.normal(jax.random.PRNGKey(2), (n, cfg["obs_dim"]))
    actions = jax.random.uniform(jax.random.PRNGKey(3), (n, 2), minval=-20, maxval=80)
    rewards = jnp.arange(n, dtype=jnp.float32)
    next_obs = jax.random.normal(jax.random.PRNGKey(4), (n, cfg["obs_dim"]))
    dones = jnp.zeros((n,), dtype=jnp.float32)
    # Mark agent 0 and 2 active; 1, 3, 4, 5 inactive.
    is_active = jnp.array([True, False, True, False, False, False])

    new_state = runtime["write_transitions"](
        state, obs, actions, rewards, next_obs, dones, is_active,
    )

    # Active agents' ptr should be 1, size should be 1.
    assert int(new_state.replay_ptr[0]) == 1
    assert int(new_state.replay_size[0]) == 1
    assert int(new_state.replay_ptr[2]) == 1
    assert int(new_state.replay_size[2]) == 1
    # Inactive agents should still be at 0.
    for i in (1, 3, 4, 5):
        assert int(new_state.replay_ptr[i]) == 0, f"slot {i}: ptr advanced"
        assert int(new_state.replay_size[i]) == 0, f"slot {i}: size advanced"

    # Reward at slot 0, ring index 0 should equal rewards[0] (= 0.0); at
    # slot 2 it should equal rewards[2] (= 2.0).
    assert float(new_state.replay_reward[0, 0]) == 0.0
    assert float(new_state.replay_reward[2, 0]) == 2.0
    # Inactive slot 1 should still have a 0 at ring index 0 (never written).
    assert float(new_state.replay_reward[1, 0]) == 0.0


def test_replay_wraparound_at_capacity():
    cfg = _config()
    cap = cfg["sac_replay_capacity"]
    n = _max_agents(cfg)
    state = init_sacstate(cfg, jax.random.PRNGKey(1))
    runtime = build_sac_runtime(cfg)

    is_active = jnp.ones((n,), dtype=jnp.bool_)
    rng = jax.random.PRNGKey(0)

    # Fill cap+5 transitions per agent. Distinct rewards each step (cap+5
    # values per agent) so we can verify which got overwritten.
    for t in range(cap + 5):
        rng, k_obs, k_act, k_next = jax.random.split(rng, 4)
        obs = jax.random.normal(k_obs, (n, cfg["obs_dim"]))
        actions = jax.random.uniform(k_act, (n, 2), minval=-20, maxval=80)
        rewards = jnp.full((n,), float(t), dtype=jnp.float32)
        next_obs = jax.random.normal(k_next, (n, cfg["obs_dim"]))
        dones = jnp.zeros((n,), dtype=jnp.float32)
        state = runtime["write_transitions"](
            state, obs, actions, rewards, next_obs, dones, is_active,
        )

    # After cap+5 writes: size capped at cap, ptr at (cap+5) % cap = 5.
    for i in range(n):
        assert int(state.replay_size[i]) == cap
        assert int(state.replay_ptr[i]) == (cap + 5) % cap

    # Ring indices [0..4] should hold the most-recent (overwritten) writes
    # cap..cap+4. Ring indices [5..cap-1] should hold the older writes
    # 5..cap-1 (the survivors of the cap+5-step rollover).
    rewards_ring = np.asarray(state.replay_reward[0])
    expected = np.concatenate([
        np.arange(cap, cap + 5, dtype=np.float32),
        np.arange(5, cap, dtype=np.float32),
    ])
    assert np.array_equal(rewards_ring, expected), \
        f"ring layout wrong:\n  got      {rewards_ring}\n  expected {expected}"


def test_reset_sac_slot_clears_replay_and_randomizes_actor():
    cfg = _config()
    n = _max_agents(cfg)
    state = init_sacstate(cfg, jax.random.PRNGKey(1))
    runtime = build_sac_runtime(cfg)

    # Fill slot 0 with some transitions.
    is_active = jnp.array([True] + [False] * (n - 1))
    rng = jax.random.PRNGKey(0)
    for t in range(5):
        rng, k = jax.random.split(rng)
        obs = jax.random.normal(k, (n, cfg["obs_dim"]))
        actions = jnp.zeros((n, 2))
        rewards = jnp.ones((n,)) * t
        next_obs = jnp.zeros((n, cfg["obs_dim"]))
        dones = jnp.zeros((n,))
        state = runtime["write_transitions"](
            state, obs, actions, rewards, next_obs, dones, is_active,
        )
    assert int(state.replay_size[0]) == 5

    # Snapshot slot-0 actor params for later comparison.
    actor_slot_0_pre = jax.tree_util.tree_map(
        lambda x: np.asarray(x[0]), state.actor_params,
    )

    # Reset slot 0 with a different RNG.
    state = reset_sac_slot(state, 0, jax.random.PRNGKey(42), cfg)

    # Replay for slot 0 is wiped.
    assert int(state.replay_size[0]) == 0
    assert int(state.replay_ptr[0]) == 0
    assert float(state.replay_reward[0].sum()) == 0.0

    # Actor params for slot 0 are different from before (probabilistically
    # certain with different RNG and >1 param).
    actor_slot_0_post = jax.tree_util.tree_map(
        lambda x: np.asarray(x[0]), state.actor_params,
    )
    any_diff = any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree_util.tree_leaves(actor_slot_0_pre),
            jax.tree_util.tree_leaves(actor_slot_0_post),
        )
    )
    assert any_diff, "reset_sac_slot did not re-randomize actor params"

    # Other slots untouched.
    for i in range(1, n):
        assert int(state.replay_size[i]) == 0  # they were never written either


def test_save_load_roundtrip(tmp_path=None):
    """Save and load a SacState and verify it reconstructs identically.

    Drive a few sample/write/update cycles first so the state isn't
    trivially zeros — any silent shape mismatch in the loader would then
    show up as a value diff.
    """
    import tempfile
    cfg = _config()
    cfg["sac_minibatch_size"] = 8
    cfg["sac_replay_min_size"] = 8

    rng = jax.random.PRNGKey(1)
    state = init_sacstate(cfg, rng)
    runtime = build_sac_runtime(cfg)
    n = _max_agents(cfg)

    is_active = jnp.ones((n,), dtype=jnp.bool_)
    ages = jnp.zeros((n,), dtype=jnp.int32)
    species = jnp.zeros((n,), dtype=jnp.int32)
    rng = jax.random.PRNGKey(99)
    for t in range(20):
        rng, k_act, k_obs, k_upd = jax.random.split(rng, 4)
        obs = jax.random.normal(k_obs, (n, cfg["obs_dim"]))
        actions = runtime["sample_actions"](state, obs, k_act)
        rewards = obs.sum(axis=1)
        next_obs = jnp.zeros((n, cfg["obs_dim"]))
        dones = jnp.zeros((n,))
        state = runtime["write_transitions"](
            state, obs, actions, rewards, next_obs, dones, is_active,
        )
        state = runtime["step_update"](state, is_active, ages, species, k_upd)

    # Snapshot all leaves to numpy BEFORE save (which uses np.asarray
    # internally — that's fine, but the post-load comparison needs a
    # stable reference).
    pre_leaves = {
        "actor_params": jax.tree_util.tree_map(np.asarray, state.actor_params),
        "q1_params": jax.tree_util.tree_map(np.asarray, state.q1_params),
        "q2_params": jax.tree_util.tree_map(np.asarray, state.q2_params),
        "q1_target_params": jax.tree_util.tree_map(np.asarray, state.q1_target_params),
        "q2_target_params": jax.tree_util.tree_map(np.asarray, state.q2_target_params),
        "log_alpha": np.asarray(state.log_alpha),
        "replay_obs": np.asarray(state.replay_obs),
        "replay_action": np.asarray(state.replay_action),
        "replay_reward": np.asarray(state.replay_reward),
        "replay_next_obs": np.asarray(state.replay_next_obs),
        "replay_done": np.asarray(state.replay_done),
        "replay_ptr": np.asarray(state.replay_ptr),
        "replay_size": np.asarray(state.replay_size),
    }

    with tempfile.TemporaryDirectory() as tdir:
        path = Path(tdir) / "sac.npz"
        save_sac_state(state, path)
        loaded = load_sac_state(path, cfg)

    # Per-field comparison.
    for name in pre_leaves:
        v = getattr(loaded, name)
        if isinstance(v, dict):  # params pytree
            old_leaves = jax.tree_util.tree_leaves(pre_leaves[name])
            new_leaves = jax.tree_util.tree_leaves(jax.tree_util.tree_map(np.asarray, v))
            for a, b in zip(old_leaves, new_leaves):
                assert np.array_equal(a, b), f"{name}: leaf mismatch after roundtrip"
        else:
            assert np.array_equal(pre_leaves[name], np.asarray(v)), \
                f"{name}: array mismatch after roundtrip"


def test_save_load_mismatched_config_raises():
    """Saving with one config, loading with an incompatible one should
    raise ValueError rather than silently corrupting state. Changing
    obs_dim doesn't add/remove pytree leaves — it just resizes some —
    so the loader's shape check (not just leaf-count) is what catches
    this case."""
    import tempfile
    cfg = _config()
    state = init_sacstate(cfg, jax.random.PRNGKey(0))

    bad_cfg = dict(cfg)
    bad_cfg["obs_dim"] = cfg["obs_dim"] + 1   # changes per-leaf shape

    with tempfile.TemporaryDirectory() as tdir:
        path = Path(tdir) / "sac.npz"
        save_sac_state(state, path)
        try:
            load_sac_state(path, bad_cfg)
            raise AssertionError("expected ValueError on config mismatch")
        except ValueError as e:
            assert "shape" in str(e) or "leaf count" in str(e), \
                f"unexpected error message: {e}"


def test_end_to_end_critic_learns_through_runtime():
    """init → 200 cycles of (sample, write, update) on synthetic, fixed-
    reward transitions. With γ=0 and α≈0, the Bellman target is just r,
    so the critic should learn to predict r."""
    cfg = _config()
    cfg["sac_minibatch_size"] = 8       # small so we can warmup faster
    cfg["sac_replay_min_size"] = 8
    n = _max_agents(cfg)
    state = init_sacstate(cfg, jax.random.PRNGKey(1))
    runtime = build_sac_runtime(cfg)
    qnet = QNetwork(hidden_size=cfg["policy_hidden_size"])

    is_active = jnp.ones((n,), dtype=jnp.bool_)
    ages = jnp.zeros((n,), dtype=jnp.int32)
    species = jnp.zeros((n,), dtype=jnp.int32)

    # Build a fixed reference dataset for slot 0 — we'll evaluate the
    # critic on it. Reward function: r = sum(obs).
    eval_obs = np.asarray(jax.random.normal(jax.random.PRNGKey(7), (32, cfg["obs_dim"])))
    eval_actions = np.asarray(jax.random.uniform(
        jax.random.PRNGKey(8), (32, 2), minval=-20, maxval=80,
    ))
    eval_rewards = eval_obs.sum(axis=1)

    def slot0_mse(actor_q1_params):
        q_params = jax.tree_util.tree_map(lambda x: x[0], actor_q1_params)
        preds = jax.vmap(lambda o, a: qnet.apply(q_params, o, a))(
            jnp.asarray(eval_obs), jnp.asarray(eval_actions),
        )
        return float(jnp.mean((preds - jnp.asarray(eval_rewards)) ** 2))

    initial = slot0_mse(state.q1_params)

    rng = jax.random.PRNGKey(0)
    for t in range(200):
        rng, k_act, k_obs, k_upd = jax.random.split(rng, 4)
        # Synthetic transitions where r = sum(obs).
        obs = jax.random.normal(k_obs, (n, cfg["obs_dim"]))
        actions = runtime["sample_actions"](state, obs, k_act)
        rewards = obs.sum(axis=1)
        next_obs = jnp.zeros((n, cfg["obs_dim"]))
        dones = jnp.zeros((n,))
        state = runtime["write_transitions"](
            state, obs, actions, rewards, next_obs, dones, is_active,
        )
        state = runtime["step_update"](state, is_active, ages, species, k_upd)

    final = slot0_mse(state.q1_params)
    assert final < initial * 0.5, (
        f"critic didn't learn via runtime: initial={initial:.3f} "
        f"final={final:.3f}"
    )


if __name__ == "__main__":
    test_init_sacstate_shapes();                         print("ok: init shapes")
    test_sample_actions_in_range();                      print("ok: sample range")
    test_write_transitions_advances_ptr_and_size_for_active_only(); print("ok: write gating")
    test_replay_wraparound_at_capacity();                print("ok: ring wraparound")
    test_reset_sac_slot_clears_replay_and_randomizes_actor(); print("ok: birth reset")
    test_save_load_roundtrip();                          print("ok: save/load roundtrip")
    test_save_load_mismatched_config_raises();           print("ok: save/load config-mismatch error")
    test_end_to_end_critic_learns_through_runtime();     print("ok: end-to-end critic learns")
    print("\nAll SAC integration tests passed.")
