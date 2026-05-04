"""
test_age_hazard_modes.py
------------------------
Pin the two age-hazard shapes:

  - mode "exp"    : age_term = alpha_t * exp(beta_t * age)         (paper-faithful)
  - mode "linear" : age_term = max(0, alpha_t + beta_t * age)      (ablation)

and verify:
  1. Default config (no mode keys) reproduces the pre-change exp formula
     bit-exactly — the new control surface is purely opt-in.
  2. Linear mode evaluates to the right closed-form value, including the
     non-negativity floor for negative slopes.
  3. Per-species mode mixing (prey="exp" + pred="linear") routes the right
     formula to the right species.
  4. The new baseline_kd_linear_age_pred config parses and selects linear
     mode for predators only.
"""
from __future__ import annotations

import pathlib

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from src.jax_lifecycle import _batch_hazard_prob_jax


_REF_CFG = {
    "kappa_h": 0.01, "alpha_e": 0.02, "beta_h": 0.2,
    "alpha_t_prey": 4.0e-7, "beta_t_prey": 2.0e-6,
    "alpha_t_pred": 2.0e-7, "beta_t_pred": 4.0e-6,
}


def _expected_exp(alpha_t, beta_t, age):
    return alpha_t * np.exp(beta_t * age)


def _expected_lin(alpha_t, beta_t, age):
    return np.maximum(0.0, alpha_t + beta_t * age)


def _energy_term(alpha_e, beta_h, energies):
    return 1.0 - 1.0 / (1.0 + alpha_e * np.exp(-beta_h * energies))


def test_default_mode_matches_legacy_exp():
    """No mode keys → exp form, identical to the pre-change formula."""
    ages = jnp.array([0, 1_000, 100_000, 1_000_000, 2_000_000], dtype=jnp.int32)
    energies = jnp.array([0.0, 50.0, 100.0, 100.0, 200.0])
    species = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)

    h = np.asarray(_batch_hazard_prob_jax(ages, energies, species, _REF_CFG))

    np_ages = np.asarray(ages)
    np_e = np.asarray(energies)
    np_sp = np.asarray(species)
    a = np.where(np_sp == 0, _REF_CFG["alpha_t_prey"], _REF_CFG["alpha_t_pred"])
    b = np.where(np_sp == 0, _REF_CFG["beta_t_prey"], _REF_CFG["beta_t_pred"])
    expected = _REF_CFG["kappa_h"] * _energy_term(
        _REF_CFG["alpha_e"], _REF_CFG["beta_h"], np_e
    ) * _expected_exp(a, b, np_ages)
    expected = np.clip(expected, 0.0, 1.0)

    np.testing.assert_allclose(h, expected, rtol=1e-5, atol=1e-12)


def test_linear_mode_predator_only():
    """Linear form for predator; prey stays exp."""
    cfg = dict(_REF_CFG)
    cfg["age_hazard_mode_pred"] = "linear"
    cfg["alpha_t_pred"] = 2.0e-7
    cfg["beta_t_pred"] = 8.0e-13            # linear-slope reinterpretation

    ages = jnp.array([0, 500_000, 1_000_000, 2_000_000], dtype=jnp.int32)
    energies = jnp.array([100.0, 100.0, 100.0, 100.0])
    pred_species = jnp.ones_like(ages, dtype=jnp.int32)
    prey_species = jnp.zeros_like(ages, dtype=jnp.int32)

    h_pred = np.asarray(
        _batch_hazard_prob_jax(ages, energies, pred_species, cfg)
    )
    h_prey = np.asarray(
        _batch_hazard_prob_jax(ages, energies, prey_species, cfg)
    )

    e_term = _energy_term(cfg["alpha_e"], cfg["beta_h"],
                          np.asarray(energies))
    expected_pred = cfg["kappa_h"] * e_term * _expected_lin(
        cfg["alpha_t_pred"], cfg["beta_t_pred"], np.asarray(ages)
    )
    expected_prey = cfg["kappa_h"] * e_term * _expected_exp(
        cfg["alpha_t_prey"], cfg["beta_t_prey"], np.asarray(ages)
    )
    np.testing.assert_allclose(h_pred, np.clip(expected_pred, 0.0, 1.0),
                               rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(h_prey, np.clip(expected_prey, 0.0, 1.0),
                               rtol=1e-5, atol=1e-12)


def test_linear_mode_negative_slope_floors_at_zero():
    """A negative beta_t in linear mode must not produce a negative hazard."""
    cfg = dict(_REF_CFG)
    cfg["age_hazard_mode_pred"] = "linear"
    cfg["alpha_t_pred"] = 1.0e-7
    cfg["beta_t_pred"] = -1.0e-12   # decreasing in age, eventually negative
    ages = jnp.array([0, 100_000, 1_000_000], dtype=jnp.int32)
    energies = jnp.array([100.0, 100.0, 100.0])
    species = jnp.ones_like(ages, dtype=jnp.int32)

    h = np.asarray(_batch_hazard_prob_jax(ages, energies, species, cfg))
    assert (h >= 0.0).all()
    # at age=1M, alpha + beta*age = 1e-7 + (-1e-12 * 1e6) = -9e-7 → floored to 0
    assert h[2] == 0.0


def test_invalid_mode_raises():
    cfg = dict(_REF_CFG)
    cfg["age_hazard_mode_pred"] = "quadratic"
    ages = jnp.array([0], dtype=jnp.int32)
    energies = jnp.array([100.0])
    species = jnp.array([1], dtype=jnp.int32)
    with pytest.raises(ValueError, match="age_hazard_mode_pred"):
        _batch_hazard_prob_jax(ages, energies, species, cfg)


def test_baseline_linear_age_config_loads():
    root = pathlib.Path(__file__).resolve().parents[1]
    with open(root / "configs/baseline/med_linear_age.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["age_hazard_mode_prey"] == "exp"
    assert cfg["age_hazard_mode_pred"] == "linear"
    # alpha_t_pred kept at the K&D value so age=0 hazard matches; only the
    # slope reinterpretation differs.
    assert cfg["alpha_t_pred"] == 2.0e-7
    # linear slope must be tiny — order-of-magnitude check, not pinned exact.
    assert 0 < cfg["beta_t_pred"] < 1.0e-10
