"""
test_jax_sac_math.py
--------------------
Verify the SAC math module in isolation, before any SimState integration.

We construct a minimal config + a per-agent state pytree by hand, drive
the JIT-compiled update one or more times on synthetic data, and check
that:

  - Reparam sample log-prob is differentiable w.r.t. (mean, log_std).
  - Inactive agents are not touched.
  - Agents with replay_size < min_replay_size are not touched (warmup).
  - One update step changes actor / critic / target / log_alpha for
    active warmed-up agents.
  - Target critics drift toward online critics (Polyak), not the other
    way around.
  - Repeated updates on a stationary reward signal drive critic loss down.
  - Alpha autotune moves log_alpha in the direction needed to push policy
    entropy toward the target.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.sac_networks import (
    SACActorNetwork, QNetwork, reparam_sample,
    init_sac_actor, init_q_network, init_alpha,
)
from src.jax_sac import build_sac_update_fn


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _minimal_config():
    return {
        "policy_hidden_size": 32,    # small for speed
        "obs_dim": 8,
        "lr": 3e-4,
        "adam_eps": 1e-7,
        "gamma": 0.99,
        "sac_actor_lr": 3e-4,
        "sac_critic_lr": 3e-4,
        "sac_alpha_lr": 3e-4,
        "sac_target_tau": 0.05,      # bigger tau so tests see movement faster
        "sac_minibatch_size": 16,
        "sac_replay_min_size": 8,
        "sac_target_entropy": -2.0,
        "sac_initial_log_alpha": 0.0,
        # PPO-shared knobs (defaults; unused here but read at build time)
        "lr_schedule_enable": False,
        "lr_prey_multiplier": 1.0,
        "lr_pred_multiplier": 1.0,
    }


def _make_per_agent_state(rng, config, n_agents, replay_capacity):
    """Build the SimState-shaped pytree expected by maybe_sac_update_all,
    by initializing each agent's actor/critics/alpha and stacking."""
    obs_dim = config["obs_dim"]

    actor_params, actor_opt = [], []
    q1_params, q1_opt = [], []
    q2_params, q2_opt = [], []
    log_alphas, alpha_opts = [], []
    for i in range(n_agents):
        k1, k2, k3, rng = jax.random.split(rng, 4)
        ap, ao = init_sac_actor(k1, config)
        actor_params.append(ap); actor_opt.append(ao)
        qp1, qo1 = init_q_network(k2, config)
        q1_params.append(qp1); q1_opt.append(qo1)
        qp2, qo2 = init_q_network(k3, config)
        q2_params.append(qp2); q2_opt.append(qo2)
        la, ao_alpha = init_alpha(config)
        log_alphas.append(la); alpha_opts.append(ao_alpha)

    def stack(trees):
        return jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *trees,
        )

    state = {
        "actor_params": stack(actor_params),
        "q1_params": stack(q1_params),
        "q2_params": stack(q2_params),
        # Targets start identical to online.
        "q1_target_params": stack(q1_params),
        "q2_target_params": stack(q2_params),
        "log_alpha": jnp.stack(log_alphas, axis=0),
        "actor_opt": stack(actor_opt),
        "q1_opt": stack(q1_opt),
        "q2_opt": stack(q2_opt),
        "alpha_opt": stack(alpha_opts),
    }

    # Replay arrays — zeros; tests fill in transitions as needed.
    state["replay_obs"] = jnp.zeros((n_agents, replay_capacity, obs_dim))
    state["replay_action"] = jnp.zeros((n_agents, replay_capacity, 2))
    state["replay_reward"] = jnp.zeros((n_agents, replay_capacity))
    state["replay_next_obs"] = jnp.zeros((n_agents, replay_capacity, obs_dim))
    state["replay_done"] = jnp.zeros((n_agents, replay_capacity), dtype=jnp.float32)
    state["replay_size"] = jnp.zeros((n_agents,), dtype=jnp.int32)

    state["is_active"] = jnp.ones((n_agents,), dtype=jnp.bool_)
    state["ages"] = jnp.zeros((n_agents,), dtype=jnp.int32)
    state["species"] = jnp.zeros((n_agents,), dtype=jnp.int32)
    return state


def _fill_replay(state, agent_idx, rng, n_transitions, reward_fn=None):
    """Write `n_transitions` random transitions into one agent's buffer."""
    obs_dim = state["replay_obs"].shape[2]
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    obs = jax.random.normal(k1, (n_transitions, obs_dim))
    actions = jax.random.uniform(k2, (n_transitions, 2), minval=-20.0, maxval=80.0)
    next_obs = jax.random.normal(k3, (n_transitions, obs_dim))
    if reward_fn is None:
        rewards = jax.random.normal(k4, (n_transitions,))
    else:
        rewards = jax.vmap(reward_fn)(obs, actions)
    dones = jnp.zeros((n_transitions,), dtype=jnp.float32)

    new_state = {**state}
    new_state["replay_obs"] = new_state["replay_obs"].at[agent_idx, :n_transitions].set(obs)
    new_state["replay_action"] = new_state["replay_action"].at[agent_idx, :n_transitions].set(actions)
    new_state["replay_reward"] = new_state["replay_reward"].at[agent_idx, :n_transitions].set(rewards)
    new_state["replay_next_obs"] = new_state["replay_next_obs"].at[agent_idx, :n_transitions].set(next_obs)
    new_state["replay_done"] = new_state["replay_done"].at[agent_idx, :n_transitions].set(dones)
    new_state["replay_size"] = new_state["replay_size"].at[agent_idx].set(n_transitions)
    return new_state


def _step(update_fn, state, rng):
    rng_keys = jax.random.split(rng, state["actor_params"]["params"]["Dense_0"]["kernel"].shape[0])
    out = update_fn(
        state["actor_params"], state["q1_params"], state["q2_params"],
        state["q1_target_params"], state["q2_target_params"], state["log_alpha"],
        state["actor_opt"], state["q1_opt"], state["q2_opt"], state["alpha_opt"],
        state["replay_obs"], state["replay_action"], state["replay_reward"],
        state["replay_next_obs"], state["replay_done"],
        state["replay_size"], state["is_active"], rng_keys,
        state["ages"], state["species"],
    )
    new_state = dict(state)
    (new_state["actor_params"], new_state["q1_params"], new_state["q2_params"],
     new_state["q1_target_params"], new_state["q2_target_params"], new_state["log_alpha"],
     new_state["actor_opt"], new_state["q1_opt"], new_state["q2_opt"],
     new_state["alpha_opt"]) = out
    return new_state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reparam_sample_is_differentiable():
    """Gradients of log_prob and action w.r.t. mean and log_std must be
    finite. This is what makes the reparameterized actor loss work at all."""
    rng = jax.random.PRNGKey(0)
    mean = jnp.array([0.5, -0.2])
    log_std = jnp.array([-0.5, 0.1])

    def f(m, ls):
        a, lp, _ = reparam_sample(m, ls, rng)
        # arbitrary scalar mixing both action and log_prob
        return jnp.sum(a) + lp

    g_mean, g_log_std = jax.grad(f, argnums=(0, 1))(mean, log_std)
    assert jnp.all(jnp.isfinite(g_mean)), f"mean grads not finite: {g_mean}"
    assert jnp.all(jnp.isfinite(g_log_std)), f"log_std grads not finite: {g_log_std}"


def test_inactive_agents_are_untouched():
    cfg = _minimal_config()
    rng = jax.random.PRNGKey(1)
    state = _make_per_agent_state(rng, cfg, n_agents=2, replay_capacity=64)
    # Fill both replays with enough transitions, but mark agent 1 inactive.
    state = _fill_replay(state, 0, jax.random.PRNGKey(10), n_transitions=32)
    state = _fill_replay(state, 1, jax.random.PRNGKey(11), n_transitions=32)
    state["is_active"] = jnp.array([True, False])

    # Snapshot agent-1 slices to numpy BEFORE the JIT call — donate_argnums
    # invalidates the input pytrees as soon as the update fires.
    keys_to_check = ("actor_params", "q1_params", "q2_params",
                     "q1_target_params", "q2_target_params")
    snaps = {
        k: jax.tree_util.tree_map(lambda x: np.asarray(x[1]), state[k])
        for k in keys_to_check
    }
    log_alpha_snap = float(state["log_alpha"][1])

    fn = build_sac_update_fn(cfg)
    new = _step(fn, state, jax.random.PRNGKey(2))

    for k in keys_to_check:
        leaves_old = jax.tree_util.tree_leaves(snaps[k])
        leaves_new = jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda x: np.asarray(x[1]), new[k])
        )
        for lo, ln in zip(leaves_old, leaves_new):
            assert np.array_equal(lo, ln), f"inactive agent: {k} changed"
    assert log_alpha_snap == float(new["log_alpha"][1])


def test_warmup_blocks_update():
    """An active agent with replay_size < min_replay_size must not be updated."""
    cfg = _minimal_config()
    cfg["sac_replay_min_size"] = 32
    rng = jax.random.PRNGKey(1)
    state = _make_per_agent_state(rng, cfg, n_agents=1, replay_capacity=64)
    state = _fill_replay(state, 0, jax.random.PRNGKey(10), n_transitions=16)  # < 32

    # Snapshot before donate_argnums invalidates it.
    actor_snap = jax.tree_util.tree_map(lambda x: np.asarray(x), state["actor_params"])

    fn = build_sac_update_fn(cfg)
    new = _step(fn, state, jax.random.PRNGKey(2))

    a_old = jax.tree_util.tree_leaves(actor_snap)
    a_new = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda x: np.asarray(x), new["actor_params"])
    )
    for lo, ln in zip(a_old, a_new):
        assert np.array_equal(lo, ln), "warmup violated: actor changed"


def test_one_update_moves_active_agent():
    cfg = _minimal_config()
    rng = jax.random.PRNGKey(1)
    state = _make_per_agent_state(rng, cfg, n_agents=1, replay_capacity=64)
    state = _fill_replay(state, 0, jax.random.PRNGKey(10), n_transitions=32)

    snaps = {
        k: jax.tree_util.tree_map(lambda x: np.asarray(x), state[k])
        for k in ("actor_params", "q1_params", "q2_params")
    }

    fn = build_sac_update_fn(cfg)
    new = _step(fn, state, jax.random.PRNGKey(2))

    def any_leaf_changed(t_old, t_new):
        lo = jax.tree_util.tree_leaves(t_old)
        ln = jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda x: np.asarray(x), t_new)
        )
        return any(not np.array_equal(a, b) for a, b in zip(lo, ln))

    assert any_leaf_changed(snaps["actor_params"], new["actor_params"]), "actor didn't move"
    assert any_leaf_changed(snaps["q1_params"], new["q1_params"]), "Q1 didn't move"
    assert any_leaf_changed(snaps["q2_params"], new["q2_params"]), "Q2 didn't move"


def test_target_polyak_drifts_toward_online():
    """After one update, target = (1-tau)*old_target + tau*new_online,
    so target should land between old_target and new_online element-wise."""
    cfg = _minimal_config()
    rng = jax.random.PRNGKey(1)
    state = _make_per_agent_state(rng, cfg, n_agents=1, replay_capacity=64)
    state = _fill_replay(state, 0, jax.random.PRNGKey(10), n_transitions=32)

    tau = cfg["sac_target_tau"]
    # Snapshot pre-update target (donate_argnums will invalidate state).
    q1_target_old_np = jax.tree_util.tree_map(
        lambda x: np.asarray(x), state["q1_target_params"],
    )

    fn = build_sac_update_fn(cfg)
    new = _step(fn, state, jax.random.PRNGKey(2))

    new_q1_np = jax.tree_util.tree_map(
        lambda x: np.asarray(x), new["q1_params"],
    )
    expected_np = jax.tree_util.tree_map(
        lambda t, o: (1.0 - tau) * t + tau * o,
        q1_target_old_np, new_q1_np,
    )
    new_target_np = jax.tree_util.tree_map(
        lambda x: np.asarray(x), new["q1_target_params"],
    )
    flat_exp = jax.tree_util.tree_leaves(expected_np)
    flat_got = jax.tree_util.tree_leaves(new_target_np)
    for e, g in zip(flat_exp, flat_got):
        assert np.allclose(e, g, atol=1e-5), \
            f"Polyak mismatch: max diff {np.max(np.abs(e - g))}"


def test_critic_loss_decreases_on_stationary_rewards():
    """With γ=0 and α≈0 (alpha frozen), the Bellman target collapses to
    just the immediate reward: y = r. The critic should then learn to
    predict r from (s, a). Verify the residual drops substantially over
    50 updates.

    Why we can't just use the default γ=0.99: under SAC's full target
    y = r + γ·(min Q'(s', a') - α log π(a'|s')), the bootstrap term plus
    the entropy bonus push y well above r at init, so Q(s,a) drifts
    *away* from r even when the critic is learning correctly against y.
    """
    cfg = _minimal_config()
    cfg["sac_replay_min_size"] = 8
    cfg["gamma"] = 0.0                       # collapse Bellman target → r
    cfg["sac_initial_log_alpha"] = -20.0     # alpha ≈ 2e-9 → negligible
    cfg["sac_alpha_lr"] = 0.0                # freeze alpha
    cfg["sac_critic_lr"] = 3e-3              # 10× default for fast convergence
    cfg["sac_target_tau"] = 0.05
    rng = jax.random.PRNGKey(1)
    state = _make_per_agent_state(rng, cfg, n_agents=1, replay_capacity=64)
    state = _fill_replay(
        state, 0, jax.random.PRNGKey(10), n_transitions=64,
        reward_fn=lambda o, a: jnp.sum(o),
    )

    qnet = QNetwork(hidden_size=cfg["policy_hidden_size"])
    eval_obs = np.asarray(state["replay_obs"][0, :64])
    eval_actions = np.asarray(state["replay_action"][0, :64])
    eval_rewards = np.asarray(state["replay_reward"][0, :64])

    def mean_sq_residual(q_params_stacked):
        q_params = jax.tree_util.tree_map(lambda x: x[0], q_params_stacked)
        preds = jax.vmap(lambda o, a: qnet.apply(q_params, o, a))(
            jnp.asarray(eval_obs), jnp.asarray(eval_actions),
        )
        return jnp.mean((preds - jnp.asarray(eval_rewards)) ** 2)

    initial_err = float(mean_sq_residual(state["q1_params"]))

    fn = build_sac_update_fn(cfg)
    cur = state
    rng = jax.random.PRNGKey(99)
    for i in range(300):
        rng, sub = jax.random.split(rng)
        cur = _step(fn, cur, sub)

    final_err = float(mean_sq_residual(cur["q1_params"]))
    assert final_err < initial_err * 0.5, (
        f"critic didn't learn: initial={initial_err:.3f} final={final_err:.3f}"
    )


def test_alpha_moves_toward_target_entropy():
    """target_entropy in SAC is a *lower-bound constraint* on H(π).

    Setup: target_entropy = +10 forces H(π) >= +10. An initial 2-D
    Gaussian actor (log_std ≈ 0) has H ≈ 2.84 — well below 10. So the
    autotune should *increase* α to penalize low entropy harder.

    Direction-only check; magnitude depends on alpha_lr and is just a
    smoke test that the autotune is wired to the right sign.
    """
    cfg = _minimal_config()
    cfg["sac_target_entropy"] = 10.0    # demand high entropy → α must grow
    cfg["sac_alpha_lr"] = 1e-2          # bigger LR so 40 steps suffices
    cfg["sac_replay_min_size"] = 8

    rng = jax.random.PRNGKey(1)
    state = _make_per_agent_state(rng, cfg, n_agents=1, replay_capacity=64)
    state = _fill_replay(state, 0, jax.random.PRNGKey(10), n_transitions=32)

    log_alpha_0 = float(state["log_alpha"][0])

    fn = build_sac_update_fn(cfg)
    cur = state
    rng = jax.random.PRNGKey(77)
    for i in range(40):
        rng, sub = jax.random.split(rng)
        cur = _step(fn, cur, sub)
    log_alpha_T = float(cur["log_alpha"][0])

    assert log_alpha_T > log_alpha_0, (
        f"alpha didn't increase under unsatisfied entropy constraint: "
        f"start={log_alpha_0:.4f} end={log_alpha_T:.4f}"
    )


if __name__ == "__main__":
    # Allow running this file directly for quick smoke without pytest.
    test_reparam_sample_is_differentiable(); print("ok: reparam grads")
    test_inactive_agents_are_untouched();    print("ok: inactive frozen")
    test_warmup_blocks_update();             print("ok: warmup blocks update")
    test_one_update_moves_active_agent();    print("ok: active agent updates")
    test_target_polyak_drifts_toward_online(); print("ok: Polyak math")
    test_critic_loss_decreases_on_stationary_rewards(); print("ok: critic learns")
    test_alpha_moves_toward_target_entropy(); print("ok: alpha autotune direction")
    print("\nAll SAC math tests passed.")
