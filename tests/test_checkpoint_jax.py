"""
test_checkpoint_jax.py
----------------------
Tests for src/jax_checkpoint.py — full-SimState serialization and resume.

These are the correctness gate for the --resume flag on
scripts/run_experiment_jax.py. The determinism test is the important
one: it proves that saving a SimState mid-run, loading it back into a
fresh template, and stepping forward produces a bit-identical trajectory
to the uninterrupted version.

Run: pytest tests/test_checkpoint_jax.py -v
"""

import os

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest

from src.environment import _build_physics
from src.jax_checkpoint import (
    AsyncCheckpointWriter,
    checkpoint_path,
    find_latest_checkpoint,
    load_simstate,
    rotate_checkpoints,
    save_simstate,
)
from src.jax_sim import build_sim_step
from src.jax_state import init_simstate


@pytest.fixture
def config():
    """Full baseline config, copied from tests/test_phase0.py for independence."""
    return {
        "experiment_name": "test_checkpoint",
        "world_size": 960,
        "total_steps": 10_240_000,
        "obs_dim": 205,
        "prey_initial": 10,
        "predator_initial": 2,
        "prey_cap": 20,
        "predator_cap": 5,
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
        "social_obs": "position_only",
        "food_max": 100,
        "food_initial": 10,
        "food_growth_rate": 0.5,
        "food_max_regen_per_step": 10,
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
        "initial_energy": 100.0,
        "reward_weights_init_std": 0.1,
        "mutation_df": 2,
        "mutation_scale": 0.4,
        "weight_clip": 100.0,
        "policy_hidden_size": 64,
        "policy_n_hidden_layers": 2,
        "action_clip_low": -20.0,
        "action_clip_high": 80.0,
        "action_mapping": "sigmoid",
        "gamma": 0.999,
        "rollout_steps": 64,
        "minibatch_size": 16,
        "ppo_epochs": 2,
        "clip_epsilon": 0.2,
        "entropy_coef": 0.001,
        "gae_lambda": 0.95,
        "lr": 3.0e-4,
        "adam_eps": 1.0e-7,
        "vf_coef": 0.5,
        "checkpoint_interval_steps": 25_000,
        "log_interval_steps": 10_000,
        "seed": 0,
    }


def _tree_arrays_equal(a, b) -> bool:
    """Element-wise equality across every leaf of two pytrees."""
    leaves_a, tree_a = jtu.tree_flatten(a)
    leaves_b, tree_b = jtu.tree_flatten(b)
    if tree_a != tree_b:
        return False
    for la, lb in zip(leaves_a, leaves_b):
        la_arr = jnp.asarray(la)
        lb_arr = jnp.asarray(lb)
        if la_arr.shape != lb_arr.shape or la_arr.dtype != lb_arr.dtype:
            return False
        if not bool(jnp.all(la_arr == lb_arr)):
            return False
    return True


class TestFullSimStateRoundtrip:

    def test_roundtrip_preserves_all_fields(self, config, tmp_path):
        """Save a SimState, reload it into a fresh template, assert pytree equal."""
        state = init_simstate(config, jax.random.PRNGKey(7))
        path = str(tmp_path / "step_00000000.npz")

        save_simstate(state, path)

        template = init_simstate(config, jax.random.PRNGKey(999))
        loaded = load_simstate(path, template)

        assert _tree_arrays_equal(state, loaded), (
            "Roundtrip mismatch: loaded state not equal to saved state"
        )

    def test_roundtrip_atomic(self, config, tmp_path):
        """No .tmp file remains after save; only the final path exists."""
        state = init_simstate(config, jax.random.PRNGKey(0))
        path = str(tmp_path / "step_00000000.npz")
        save_simstate(state, path)

        leftover = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftover == [], f"Atomic write left stray temp files: {leftover}"


class TestResumeDeterminism:

    def test_resume_matches_uninterrupted(self, config, tmp_path):
        """The hero test: resuming from a checkpoint reproduces the same trajectory.

        Run A: init → N+M sim_step_core calls.
        Run B: init → N steps → save → load into fresh template → M steps.
        After both runs, every SimState leaf must match element-wise.
        """
        N = 30
        M = 30

        # --- Run A: continuous ---
        space_a, _ = _build_physics(config)
        sim_step_a, _ = build_sim_step(config, space_a)
        state_a = init_simstate(config, jax.random.PRNGKey(0))
        for _ in range(N + M):
            state_a = sim_step_a(state_a)
        jax.block_until_ready(state_a.step)

        # --- Run B: stop, checkpoint, resume ---
        space_b, _ = _build_physics(config)
        sim_step_b, _ = build_sim_step(config, space_b)
        state_b = init_simstate(config, jax.random.PRNGKey(0))
        for _ in range(N):
            state_b = sim_step_b(state_b)

        path = str(tmp_path / f"step_{N:08d}.npz")
        save_simstate(state_b, path)

        # Fresh template with a different seed to prove load actually writes
        template = init_simstate(config, jax.random.PRNGKey(12345))
        state_b = load_simstate(path, template)

        for _ in range(M):
            state_b = sim_step_b(state_b)
        jax.block_until_ready(state_b.step)

        assert int(state_a.step) == int(state_b.step) == N + M
        assert _tree_arrays_equal(state_a, state_b), (
            "Resumed trajectory diverges from uninterrupted run — "
            "some SimState field is not being saved/restored correctly"
        )


class TestCheckpointRotation:

    def test_keeps_latest_n(self, tmp_path):
        """With keep=3 and 5 saves on disk, only the 3 highest-step files remain."""
        for step in (10, 20, 30, 40, 50):
            (tmp_path / f"step_{step:08d}.npz").write_bytes(b"stub")

        rotate_checkpoints(str(tmp_path), keep=3)

        remaining = sorted(p.name for p in tmp_path.iterdir())
        assert remaining == [
            "step_00000030.npz",
            "step_00000040.npz",
            "step_00000050.npz",
        ]

    def test_find_latest(self, tmp_path):
        """find_latest_checkpoint returns the highest-step file."""
        for step in (100, 500, 200):
            (tmp_path / f"step_{step:08d}.npz").write_bytes(b"stub")

        latest = find_latest_checkpoint(str(tmp_path))
        assert latest is not None
        assert latest.endswith("step_00000500.npz")

    def test_find_latest_empty_dir(self, tmp_path):
        assert find_latest_checkpoint(str(tmp_path)) is None

    def test_find_latest_missing_dir(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        assert find_latest_checkpoint(str(missing)) is None

    def test_checkpoint_path_format(self, tmp_path):
        path = checkpoint_path(str(tmp_path), "my_exp", seed=3, step=1234)
        assert path.endswith("my_exp/seed_3/checkpoints/step_00001234.npz")


class TestLegacyFormatRejected:

    def test_npz_rejected_with_clear_message(self, config, tmp_path):
        """Feeding an old-format .npz to the loader raises a clear error."""
        npz_path = str(tmp_path / "step_00000100.npz")
        np.savez(npz_path, step=100, reward_weights=np.zeros((10, 4)))

        template = init_simstate(config, jax.random.PRNGKey(0))
        with pytest.raises(ValueError, match="legacy partial checkpoint"):
            load_simstate(npz_path, template)


class TestAsyncCheckpointWriter:
    """Coverage for the background-thread checkpoint writer.

    These tests pin the safety properties that make async checkpoints
    behavior-equivalent to sync save_simstate:
      1. After close(), every requested file is on disk and bit-equal.
      2. No .tmp leftovers (atomic visibility).
      3. The snapshot is decoupled at submit() time — mutating sim_state
         after submit cannot affect the saved file.
      4. Rotation runs after each write, never deletes a successor first.
      5. Resume from an async-saved checkpoint reproduces a continuous
         trajectory bit-identically.
      6. Worker exceptions surface to the caller (no silent disk errors).
    """

    def test_writes_complete_after_close(self, config, tmp_path):
        states = []
        writer = AsyncCheckpointWriter()
        try:
            for n in range(3):
                state = init_simstate(config, jax.random.PRNGKey(n))
                path = str(tmp_path / f"step_{n*10:08d}.npz")
                writer.submit(state, path)
                states.append((state, path))
        finally:
            writer.close()

        template = init_simstate(config, jax.random.PRNGKey(99))
        for state, path in states:
            assert os.path.exists(path), f"Async write didn't land: {path}"
            loaded = load_simstate(path, template)
            assert _tree_arrays_equal(state, loaded), (
                f"Async-saved checkpoint at {path} doesn't roundtrip"
            )

    def test_no_temp_files_after_close(self, config, tmp_path):
        writer = AsyncCheckpointWriter()
        try:
            for n in range(3):
                state = init_simstate(config, jax.random.PRNGKey(n))
                path = str(tmp_path / f"step_{n*10:08d}.npz")
                writer.submit(state, path)
        finally:
            writer.close()

        leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == [], f"Atomic write left stray temp files: {leftovers}"

    def test_snapshot_decoupled_from_caller_state(self, config, tmp_path):
        """The safety-critical test for async checkpoints.

        After submit() returns, the simulation loop must be free to keep
        stepping sim_state without affecting the file that gets written.
        We submit at step=0, then step the sim 50× more, then close. The
        file must reflect step=0, never step=50.
        """
        space, _ = _build_physics(config)
        sim_step, _ = build_sim_step(config, space)
        state = init_simstate(config, jax.random.PRNGKey(0))

        path = str(tmp_path / "step_00000000.npz")
        writer = AsyncCheckpointWriter()
        try:
            writer.submit(state, path)

            # Mutate sim_state aggressively while the write may still be
            # in flight — proves the worker is operating on its own snapshot.
            for _ in range(50):
                state = sim_step(state)
            jax.block_until_ready(state.step)
        finally:
            writer.close()

        template = init_simstate(config, jax.random.PRNGKey(999))
        loaded = load_simstate(path, template)
        assert int(loaded.step) == 0, (
            f"Async writer captured a stale or post-step state: "
            f"got step={int(loaded.step)}, expected 0"
        )

    def test_rotation_runs_after_write(self, config, tmp_path):
        """With rotate_keep=3 and 5 submits, the 3 highest-step files survive."""
        writer = AsyncCheckpointWriter()
        try:
            for n in range(5):
                state = init_simstate(config, jax.random.PRNGKey(n))
                path = str(tmp_path / f"step_{n*10:08d}.npz")
                writer.submit(
                    state, path, rotate_dir=str(tmp_path), rotate_keep=3,
                )
        finally:
            writer.close()

        remaining = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".npz")
        assert remaining == [
            "step_00000020.npz",
            "step_00000030.npz",
            "step_00000040.npz",
        ], f"Rotation kept wrong files: {remaining}"

    def test_resume_after_async_save_matches_uninterrupted(self, config, tmp_path):
        """End-to-end: async-saved checkpoint resumes bit-equal to continuous run.

        Mirrors test_resume_matches_uninterrupted in TestResumeDeterminism,
        but routes the save through AsyncCheckpointWriter.
        """
        N = 30
        M = 30

        space_a, _ = _build_physics(config)
        sim_step_a, _ = build_sim_step(config, space_a)
        state_a = init_simstate(config, jax.random.PRNGKey(0))
        for _ in range(N + M):
            state_a = sim_step_a(state_a)
        jax.block_until_ready(state_a.step)

        space_b, _ = _build_physics(config)
        sim_step_b, _ = build_sim_step(config, space_b)
        state_b = init_simstate(config, jax.random.PRNGKey(0))
        for _ in range(N):
            state_b = sim_step_b(state_b)

        path = str(tmp_path / f"step_{N:08d}.npz")
        writer = AsyncCheckpointWriter()
        try:
            writer.submit(state_b, path)
        finally:
            writer.close()

        template = init_simstate(config, jax.random.PRNGKey(12345))
        state_b = load_simstate(path, template)

        for _ in range(M):
            state_b = sim_step_b(state_b)
        jax.block_until_ready(state_b.step)

        assert int(state_a.step) == int(state_b.step) == N + M
        assert _tree_arrays_equal(state_a, state_b), (
            "Async-saved + resumed trajectory diverges from uninterrupted run"
        )

    def test_propagates_worker_exceptions(self, config, tmp_path):
        """Disk errors on the worker thread must surface, not be swallowed.

        Submitting to a non-existent directory triggers a FileNotFoundError
        inside _write_npz_atomic; the next submit() (or close()) must raise.
        """
        bad_path = str(tmp_path / "does_not_exist" / "step_00000000.npz")
        state = init_simstate(config, jax.random.PRNGKey(0))

        writer = AsyncCheckpointWriter()
        writer.submit(state, bad_path)
        # The worker raises asynchronously; close() flushes and re-raises.
        with pytest.raises(FileNotFoundError):
            writer.close()
