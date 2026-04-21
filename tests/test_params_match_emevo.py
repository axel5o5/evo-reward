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


# Values here come from the K&D 2025 paper text (arXiv:2507.09992, v2 Feb 2026)
# cross-checked against emevo's gecco2026 branch. When paper text and emevo
# endpoint code disagree, we follow paper text. See docs/emevo-diff.md D22
# for the full contradiction between paper and emevo `cf_predator.py` defaults.


def test_prey_hazard_params():
    cfg = _load_baseline()
    assert cfg["kappa_h"] == 0.01
    assert cfg["alpha_e"] == 0.02
    assert cfg["beta_h"] == 0.2
    # alpha_t_prey = 4e-7 (paper Table 3; endpoint TOML agrees)
    assert cfg["alpha_t_prey"] == 4.0e-7
    # beta_t_prey = 2e-6 (paper Table 3). Endpoint TOML 20240916 has 4e-6 —
    # deliberately using paper value (D22).
    assert cfg["beta_t_prey"] == 2.0e-6


def test_predator_hazard_params():
    cfg = _load_baseline()
    assert cfg["alpha_t_pred"] == 2.0e-7
    assert cfg["beta_t_pred"] == 4.0e-6


def test_birth_params():
    cfg = _load_baseline()
    assert cfg["kappa_b"] == 1.0e-3
    # beta_b = 0.4 (all endpoint TOMLs + consistent with paper Fig 19 plot;
    # paper Table 3 prints 0.1 but Fig 19's saturation point rules that out).
    assert cfg["beta_b"] == 0.4
    # zeta_b_prey = 10 (paper Table 3 + Fig 19 + paper text "30 units required
    # for prey birth" near saturation). Endpoint TOML 20240916 has 15. D22.
    assert cfg["zeta_b_prey"] == 10.0
    assert cfg["zeta_b_pred"] == 100.0


def test_food_radius_present_and_used_in_contact():
    """D22 regression: food_radius must be in config AND check_eating_jax
    must use it for the prey-food contact threshold. Pre-D22 we used only
    prey_radius (10) which made the effective contact area 51% of emevo's
    (prey_r + food_r = 14)."""
    cfg = _load_baseline()
    assert cfg["food_radius"] == 4.0
    import src.jax_food as food_mod
    src_text = pathlib.Path(food_mod.__file__).read_text()
    assert "food_contact_dist" in src_text, (
        "check_eating_jax should compute food_contact_dist = prey_r + food_r"
    )
    assert 'config.get("food_radius"' in src_text, (
        "check_eating_jax should read food_radius from config"
    )


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
