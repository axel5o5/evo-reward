"""Smoke tests for DDB / DDM stability scaffolds.

DDB (density-dependent breeding):
    - Threshold scaling: zeta_b -> zeta_b * factor
    - Rate boost (with ddb_max_boost > 1.0): kappa_b -> kappa_b / max(factor, 1/max_boost)

DDM (density-dependent metabolism):
    - Decay scaling: predator d_b -> d_b * factor
"""
import jax.numpy as jnp
import pytest

from src.jax_lifecycle import _batch_birth_prob_jax, _ddb_factor


@pytest.fixture
def base_config():
    return dict(
        kappa_b=1e-3,
        beta_b=0.4,
        zeta_b_prey=15.0,
        zeta_b_pred=100.0,
        prey_e_food=1.0,
        prey_c_b=1e-4,
        prey_c_a=2.5e-6,
        predator_d_b=4e-3,
        predator_d_a=5e-5,
        predator_eta=0.5,
        energy_capacity=1000.0,
    )


def test_ddb_factor_curve_at_calibrated_thresholds():
    # pred T=4: N=4 -> 0.50, N=8 -> 0.80
    assert float(_ddb_factor(jnp.int32(4), 4.0, 0.0)) == pytest.approx(0.5, abs=1e-3)
    assert float(_ddb_factor(jnp.int32(8), 4.0, 0.0)) == pytest.approx(0.8, abs=1e-3)
    # prey T=40: N=40 -> 0.50, N=80 -> 0.80
    assert float(_ddb_factor(jnp.int32(40), 40.0, 0.0)) == pytest.approx(0.5, abs=1e-3)
    assert float(_ddb_factor(jnp.int32(80), 40.0, 0.0)) == pytest.approx(0.8, abs=1e-3)


def test_ddb_factor_floor_pins_below_threshold():
    # floor=0 lets curve approach 0 at N=1
    f1_zero = float(_ddb_factor(jnp.int32(1), 4.0, 0.0))
    f1_third = float(_ddb_factor(jnp.int32(1), 4.0, 0.3))
    assert f1_zero == pytest.approx(1.0 / 17.0, abs=1e-3)  # ~0.059
    assert f1_third == pytest.approx(0.3, abs=1e-3)


def test_birth_prob_no_boost_when_max_boost_default(base_config):
    """Regression: ddb_max_boost defaults to 1.0 => kappa_b unchanged."""
    cfg = {**base_config, "stability_mechanism": "ddb",
           "ddb_pred_threshold": 4.0, "ddb_prey_threshold": 40.0,
           "ddb_floor": 0.0}
    species = jnp.array([0, 1])  # prey, predator
    energies = jnp.array([100.0, 1000.0])  # well above any threshold
    probs = _batch_birth_prob_jax(energies, species, cfg,
                                  prey_count=jnp.int32(100), pred_count=jnp.int32(20))
    # Healthy populations -> factor near 1, kappa_b ~unchanged
    p_prey, p_pred = float(probs[0]), float(probs[1])
    assert p_prey == pytest.approx(1e-3, rel=0.05)
    assert p_pred == pytest.approx(1e-3, rel=0.05)


def test_birth_prob_boost_at_low_predator_pop(base_config):
    """With max_boost=50 and pred N=1, kappa_b should boost ~17x (1/factor)."""
    cfg = {**base_config, "stability_mechanism": "ddb",
           "ddb_pred_threshold": 4.0, "ddb_prey_threshold": 40.0,
           "ddb_floor": 0.0, "ddb_max_boost": 50.0}
    species = jnp.array([1])  # predator
    energies = jnp.array([1000.0])  # saturated above threshold
    probs = _batch_birth_prob_jax(energies, species, cfg,
                                  prey_count=jnp.int32(100), pred_count=jnp.int32(1))
    p = float(probs[0])
    # factor at N=1, T=4 = 1/17 ≈ 0.0588 -> boost = 17 -> kappa_b_eff ≈ 17e-3
    assert p == pytest.approx(17e-3, rel=0.05)


def test_birth_prob_boost_capped_by_max_boost(base_config):
    """At extreme low N (factor << 1/max_boost), boost is capped at max_boost."""
    cfg = {**base_config, "stability_mechanism": "ddb",
           "ddb_pred_threshold": 4.0, "ddb_prey_threshold": 40.0,
           "ddb_floor": 0.0, "ddb_max_boost": 50.0}
    species = jnp.array([0])  # prey
    energies = jnp.array([1000.0])
    # At prey N=1, T=40: factor = 1/1601 ≈ 0.000625 -> uncapped boost would be 1600x
    # Cap = 50 -> kappa_b_eff = 50e-3
    probs = _batch_birth_prob_jax(energies, species, cfg,
                                  prey_count=jnp.int32(1), pred_count=jnp.int32(20))
    p = float(probs[0])
    assert p == pytest.approx(50e-3, rel=0.05)


def test_birth_prob_boost_fades_to_one_at_healthy_pop(base_config):
    """At healthy pop (factor near 1), boost should be ~1 (selection intact)."""
    cfg = {**base_config, "stability_mechanism": "ddb",
           "ddb_pred_threshold": 4.0, "ddb_prey_threshold": 40.0,
           "ddb_floor": 0.0, "ddb_max_boost": 50.0}
    species = jnp.array([1])  # predator
    energies = jnp.array([1000.0])
    # Pred N=20, T=4: factor = 400/416 ≈ 0.96 -> boost ≈ 1.04
    probs = _batch_birth_prob_jax(energies, species, cfg,
                                  prey_count=jnp.int32(100), pred_count=jnp.int32(20))
    p = float(probs[0])
    assert p == pytest.approx(1.04e-3, rel=0.05)


def test_birth_threshold_still_lowered_with_boost(base_config):
    """DDB threshold scaling is independent of rate boost — both apply.

    At pred N=1, T=4, floor=0: factor=0.0588, zeta_eff=5.88, kappa_b_eff=17e-3.
    Without DDB, predator at energy 50 would be far below the zeta=100 bar
    (sigmoid ≈ 0). With DDB threshold scaling, energy 50 is way past zeta_eff=5.88,
    so birth prob saturates near kappa_b_eff = 17e-3.
    """
    cfg_with_ddb = {**base_config, "stability_mechanism": "ddb",
                    "ddb_pred_threshold": 4.0, "ddb_prey_threshold": 40.0,
                    "ddb_floor": 0.0, "ddb_max_boost": 50.0}
    cfg_no_ddb = {**base_config, "stability_mechanism": "none"}

    species = jnp.array([1])
    energies = jnp.array([50.0])

    p_with = float(_batch_birth_prob_jax(
        energies, species, cfg_with_ddb,
        prey_count=jnp.int32(100), pred_count=jnp.int32(1))[0])
    p_no = float(_batch_birth_prob_jax(energies, species, cfg_no_ddb)[0])

    # With DDB: kappa_b_eff (17e-3) * sigmoid(50-5.88, near 1) ≈ 17e-3
    assert p_with == pytest.approx(17e-3, rel=0.05)
    # Without DDB: kappa_b (1e-3) * sigmoid(50-100, ~0) ≈ near 0
    assert p_no < 1e-9
    # Combined effect is dramatic
    assert p_with / max(p_no, 1e-30) > 1e6
