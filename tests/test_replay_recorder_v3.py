"""
test_replay_recorder_v3.py
--------------------------
Roundtrip checks for the v3 recorder format:

* MLP runs emit `reward_genomes_byid` + `reward_genomes_idmap` and tag
  `genome_arch="mlp"` in meta. Per-frame `reward_weights` is dropped.
* Linear runs are unchanged (per-frame `reward_weights`, no genome rows).
* The flat genome row reconstructs the per-layer kernels/biases via the
  saved layout, and a fresh forward pass through the rebuilt MLP matches
  what `compute_mlp_reward` produces on the original PyTree — which is
  the contract the dashboard's TS forward pass also relies on.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest

from scripts.replay_recorder import ReplayRecorder
from src.jax_state import init_simstate
from src.reward import compute_mlp_reward, init_mlp_genome


@pytest.fixture
def mlp_config():
    """Tiny MLP-reward config — small population for a fast capture loop."""
    from tests.test_phase0 import config as _config_factory

    # The phase0 fixture is a pytest fixture itself, so we have to build the
    # dict by hand (calling the factory directly returns a generator).
    cfg = {
        "experiment_name": "test_recorder_v3_mlp",
        "world_size": 480,
        "total_steps": 1_000,
        "obs_dim": 205,
        "prey_initial": 8,
        "predator_initial": 2,
        "prey_cap": 16,
        "predator_cap": 4,
        "prey_radius": 10.0,
        "predator_radius": 14.0,
        "max_motor_norm": 114.0,
        "n_proximity_sensors": 32,
        "n_proximity_channels": 4,
        "proximity_fov_deg": 120.0,
        "proximity_max_range": 200.0,
        "n_tactile_sensors": 18,
        "n_tactile_channels": 4,
        "tactile_spacing_deg": 20.0,
        "food_max": 60,
        "food_initial": 20,
        "food_growth_rate": 0.5,
        "food_max_regen_per_step": 5,
        "energy_capacity": 1000.0,
        "prey_e_food": 1.0,
        "prey_c_b": 1.0e-4,
        "prey_c_a": 2.5e-6,
        "predator_d_b": 4.0e-3,
        "predator_d_a": 5.0e-5,
        "predator_eta": 0.6,
        "predator_mouth_tactile_bins": [0, 1, 17],
        "predator_eat_interval": 10,
        "kappa_h": 0.01,
        "alpha_e": 0.02,
        "beta_h": 0.2,
        "alpha_t_prey": 4.0e-7,
        "alpha_t_pred": 2.0e-7,
        "beta_t_prey": 4.0e-6,
        "beta_t_pred": 4.0e-6,
        "kappa_b": 1.0e-3,
        "beta_b": 0.4,
        "zeta_b_prey": 15.0,
        "zeta_b_pred": 100.0,
        "energy_share_ratio": 0.4,
        "spawn_spread": 100.0,
        "prey_e_initial": 100.0,
        "predator_e_initial": 100.0,
        "reward_weights_init_std": 0.1,
        "mutation_df": 2,
        "mutation_scale": 0.4,
        "weight_clip": 100.0,
        "policy_hidden_size": 32,
        "policy_n_hidden_layers": 2,
        "action_clip_low": -20.0,
        "action_clip_high": 80.0,
        "action_mapping": "sigmoid",
        "gamma": 0.999,
        "rollout_steps": 1024,
        "minibatch_size": 256,
        "ppo_epochs": 10,
        "clip_epsilon": 0.2,
        "entropy_coef": 0.001,
        "gae_lambda": 0.95,
        "lr": 3.0e-4,
        "adam_eps": 1.0e-7,
        "vf_coef": 0.5,
        "checkpoint_interval_steps": 25_000,
        "log_interval_steps": 10_000,
        "seed": 0,
        # MLP-reward axis flag — what we're testing.
        "reward_type": "mlp",
        "mlp_hidden_size": 8,
        "mlp_mutation_scale": 0.01,
        "mlp_weight_clip": 10.0,
        # Recording: tiny so the test is fast.
        "replay_record_interval_steps": 5,
        "replay_record_length_steps": 5,
        "replay_quantize": True,
    }
    return cfg


def _read_section(buf: bytes, sect: dict) -> np.ndarray:
    dtype_map = {
        "float32": np.float32, "int32": np.int32,
        "uint16": np.uint16, "uint8": np.uint8, "int8": np.int8,
    }
    dtype = dtype_map[sect["dtype"]]
    raw = buf[sect["offset"]: sect["offset"] + sect["length"]]
    return np.frombuffer(raw, dtype=dtype).reshape(sect["shape"])


def _unflatten_mlp(flat: np.ndarray, layout: list[dict]) -> dict:
    """Re-nest a flat genome row using the saved layout. Mirrors what the
    JS replayLoader does."""
    out: dict = {}
    for entry in layout:
        path = entry["path"]
        shape = entry["shape"]
        size = int(np.prod(shape)) if shape else 1
        slice_ = flat[entry["offset"]: entry["offset"] + size].reshape(shape)
        cur = out
        for k in path[:-1]:
            cur = cur.setdefault(k, {})
        cur[path[-1]] = slice_
    return out


def test_mlp_recorder_writes_v3_with_genomes(tmp_path: Path, mlp_config):
    """Capture a 5-frame window and verify v3 sections + meta."""
    state = init_simstate(mlp_config, jax.random.PRNGKey(0))

    recorder = ReplayRecorder(
        mlp_config, "test_v3_mlp", seed=0,
        local_out_root=tmp_path / "replays",
    )
    assert recorder.enabled
    assert recorder.genome_arch == "mlp"
    # Layout maths: 4→8→8→1 = 8 + 32 + 8 + 64 + 1 + 8 = 121.
    assert recorder.genome_dim == 121
    assert recorder.genome_shape == {"input_dim": 4, "hidden_size": 8, "output_dim": 1}

    # Walk through frames 1..5 — flush triggers at step==interval==5 with
    # in_window across 1..5, so the buffer has 5 frames.
    for step in range(1, 6):
        recorder.step(state, step)

    out_dirs = sorted((tmp_path / "replays").iterdir())
    assert len(out_dirs) == 1, "expected exactly one flushed window"
    out_dir = out_dirs[0]

    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["version"] == 3
    assert meta["genome_arch"] == "mlp"
    assert meta["genome_dim"] == 121
    # MLP runs drop the per-frame reward_weights section.
    assert "reward_weights" not in meta["sections"]
    # Genome sections present.
    assert "reward_genomes_byid" in meta["sections"]
    assert "reward_genomes_idmap" in meta["sections"]
    assert meta["sections"]["reward_genomes_byid"]["dtype"] == "float32"
    assert meta["sections"]["reward_genomes_idmap"]["dtype"] == "int32"

    # Decode the genome rows + idmap from the binary.
    bin_buf = (out_dir / "frames.bin").read_bytes()
    rows = _read_section(bin_buf, meta["sections"]["reward_genomes_byid"])
    idmap = _read_section(bin_buf, meta["sections"]["reward_genomes_idmap"])
    n_unique = rows.shape[0]
    assert rows.shape == (n_unique, 121)
    assert idmap.shape == (n_unique,)
    assert n_unique == 10  # initial pop = 8 prey + 2 pred, no births in 5 steps
    # ids should be contiguous 0..9 in the seed-0 init.
    np.testing.assert_array_equal(np.sort(idmap), np.arange(10))

    # Reconstruct one agent's MLP from its flat row using the saved layout
    # and verify it forward-passes to the same value as the original PyTree.
    target_id = int(idmap[0])
    flat_row = rows[0]
    rebuilt = _unflatten_mlp(flat_row, meta["genome_layout"])
    # Wrap in 'params' — Flax expects that outer key for nn.Module.apply.
    rebuilt_pytree = {"params": jtu.tree_map(jnp.asarray, rebuilt)}

    # Find the active slot that held target_id at step 0. Inactive slots
    # share the default id=0, so masking by is_active is mandatory.
    ids_np = np.asarray(state.agent_ids)
    active_np = np.asarray(state.is_active)
    matches = np.where((ids_np == target_id) & active_np)[0]
    assert matches.size == 1
    slot = int(matches[0])
    original_pytree = jtu.tree_map(lambda leaf: leaf[slot], state.reward_mlp_params)

    stim = jnp.array([0.5, 0.3, 0.2, 0.1])
    expected = float(compute_mlp_reward(original_pytree, stim))
    actual = float(compute_mlp_reward(rebuilt_pytree, stim))
    assert np.isclose(expected, actual, atol=1e-6), (
        f"rebuilt genome forward pass {actual} != original {expected}"
    )


def test_linear_recorder_keeps_v2_shape_under_v3(tmp_path: Path, mlp_config):
    """Linear runs must still emit per-frame reward_weights and skip the
    new genome_byid sections — only the meta version bump and a few empty
    fields differ from v2."""
    cfg = dict(mlp_config)
    cfg["reward_type"] = "linear"
    cfg.pop("mlp_hidden_size", None)
    cfg.pop("mlp_mutation_scale", None)
    cfg.pop("mlp_weight_clip", None)
    state = init_simstate(cfg, jax.random.PRNGKey(0))

    recorder = ReplayRecorder(
        cfg, "test_v3_linear", seed=0, local_out_root=tmp_path / "replays",
    )
    for step in range(1, 6):
        recorder.step(state, step)

    out_dir = sorted((tmp_path / "replays").iterdir())[0]
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["version"] == 3
    assert meta["genome_arch"] == "linear"
    assert meta["genome_dim"] == 0
    assert "reward_weights" in meta["sections"]  # kept for linear
    assert "reward_genomes_byid" not in meta["sections"]
    assert "reward_genomes_idmap" not in meta["sections"]


def test_recorder_resets_seen_genomes_after_flush(tmp_path: Path, mlp_config):
    """seen_genomes must clear on flush so the next window doesn't keep
    stale rows for agents that died between windows."""
    state = init_simstate(mlp_config, jax.random.PRNGKey(0))
    recorder = ReplayRecorder(
        mlp_config, "test_v3_mlp_reset", seed=0,
        local_out_root=tmp_path / "replays",
    )
    for step in range(1, 6):
        recorder.step(state, step)
    assert recorder.seen_genomes == {}, "flush should clear the per-window cache"
