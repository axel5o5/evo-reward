"""
test_params_match_emevo.py
--------------------------
Pin hard-coded parameters in our configs / source against the emevo
gecco2026 branch (the K&D 2025 reference implementation).

If anyone rolls one of these values (hazard coefficients, birth
exponents, reward mix, mouth geometry, physics iter count) into a
different number by accident, these tests fail loudly — we've already
had Phase 1a runs quietly drift because of drift in these values.
"""

import os
import pathlib

import yaml

from src.environment import N_PHYSICS_ITER


def _load_baseline():
    root = pathlib.Path(__file__).resolve().parents[1]
    with open(root / "configs/baseline_faithful.yaml") as f:
        return yaml.safe_load(f)


# Values here come from the emevo repo's gecco2026 branch:
#   cf_predator.py (CfPredatorConfig defaults)
#   TOML configs under emevo/reward_fn/*.toml used to build K&D 2025 runs
# Any deviation should be deliberate and documented in docs/emevo-diff.md.


def test_prey_hazard_params():
    cfg = _load_baseline()
    assert cfg["kappa_h"] == 0.01
    assert cfg["alpha_e"] == 0.02
    assert cfg["beta_h"] == 0.2
    assert cfg["alpha_t_prey"] == 4.0e-7
    assert cfg["beta_t_prey"] == 4.0e-6


def test_predator_hazard_params():
    cfg = _load_baseline()
    assert cfg["alpha_t_pred"] == 2.0e-7
    assert cfg["beta_t_pred"] == 4.0e-6


def test_birth_params():
    cfg = _load_baseline()
    assert cfg["kappa_b"] == 1.0e-3
    assert cfg["beta_b"] == 0.4
    assert cfg["zeta_b_prey"] == 15.0
    assert cfg["zeta_b_pred"] == 100.0


def test_reward_coefs_match():
    """Reward-stimulus coefficients should be [1.0, 0.01, 0.1, 0.1] —
    matches emevo's cf_predator.py defaults
    (act_reward_coef=0.01, sensor_reward_coef=0.1, pair applied to
    prey + predator sensor channels)."""
    import src.jax_sim as sim_mod
    src_text = pathlib.Path(sim_mod.__file__).read_text()
    # Grep for the one assignment so future refactors that hide this in
    # a helper still stay pinned.
    needle = "jnp.array([1.0, 0.01, 0.1, 0.1])"
    assert needle in src_text, (
        "reward coefs changed from emevo defaults — see "
        "cf_predator.py:523-524 (act_reward_coef=0.01, "
        "sensor_reward_coef=0.1)"
    )


def test_predator_eat_interval():
    cfg = _load_baseline()
    assert cfg["predator_eat_interval"] == 10


def test_predator_mouth_tactile_bins():
    """emevo's cf_predator.py uses predator_mouth_range = [0, 1, 17]
    (60° front arc over 18 × 20° bins)."""
    cfg = _load_baseline()
    assert cfg["predator_mouth_tactile_bins"] == [0, 1, 17]


def test_physics_iter_count():
    """emevo nstep runs N=5 physics substeps per sim step."""
    assert N_PHYSICS_ITER == 5
