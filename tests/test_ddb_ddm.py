"""Smoke tests for DDB / DDM stability scaffolds.

DDB (density-dependent breeding):
    - Threshold scaling: zeta_b -> zeta_b * factor
    - Rate boost: kappa_b -> kappa_b / max(factor, kappa_b)
      (§15.22 — natural kappa_b floor instead of ddb_max_boost cap;
       at integer pops with sane T, factor >> kappa_b so the floor
       only matters for proof-of-validity at extinction)

DDM (density-dependent metabolism):
    - Decay scaling per species: d_b -> d_b * factor(N_pred, T_ddm_pred)
                                  c_b -> c_b * factor(N_prey, T_ddm_prey)
      (§15.22 — symmetric across species)
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


def test_birth_prob_mild_boost_at_healthy_pop(base_config):
    """At healthy pop the 1/factor boost is near 1 — selection intact.

    §15.22: removed the ddb_max_boost cap. Rate boost is always applied
    when DDB is enabled, but at high pop the factor is near 1 so the
    boost is near 1× (no meaningful effect).
    """
    cfg = {**base_config, "stability_mechanism": "ddb",
           "ddb_pred_threshold": 4.0, "ddb_prey_threshold": 40.0,
           "ddb_floor": 0.0}
    species = jnp.array([0, 1])  # prey, predator
    energies = jnp.array([100.0, 1000.0])  # well above any threshold
    probs = _batch_birth_prob_jax(energies, species, cfg,
                                  prey_count=jnp.int32(100), pred_count=jnp.int32(20))
    p_prey, p_pred = float(probs[0]), float(probs[1])
    # prey N=100, T=40: factor = 10000/(10000+1600) = 0.862 -> boost = 1.16
    # pred N=20,  T=4:  factor = 400/(400+16)     = 0.962 -> boost = 1.04
    assert p_prey == pytest.approx(1.16e-3, rel=0.05)
    assert p_pred == pytest.approx(1.04e-3, rel=0.05)


def test_birth_prob_boost_at_low_predator_pop(base_config):
    """At pred N=1, kappa_b should boost ~17x (1/factor)."""
    cfg = {**base_config, "stability_mechanism": "ddb",
           "ddb_pred_threshold": 4.0, "ddb_prey_threshold": 40.0,
           "ddb_floor": 0.0}
    species = jnp.array([1])  # predator
    energies = jnp.array([1000.0])  # saturated above threshold
    probs = _batch_birth_prob_jax(energies, species, cfg,
                                  prey_count=jnp.int32(100), pred_count=jnp.int32(1))
    p = float(probs[0])
    # factor at N=1, T=4 = 1/17 ≈ 0.0588 -> boost = 17 -> kappa_b_eff ≈ 17e-3
    assert p == pytest.approx(17e-3, rel=0.05)


def test_birth_prob_no_cap_at_extreme_low_pop(base_config):
    """§15.22: cap removed. At extreme low N, P_birth saturates at the
    natural validity ceiling (~1.0) — sigmoid denom keeps it ≤ 1 regardless."""
    cfg = {**base_config, "stability_mechanism": "ddb",
           "ddb_pred_threshold": 4.0, "ddb_prey_threshold": 40.0,
           "ddb_floor": 0.0}
    species = jnp.array([0])  # prey
    energies = jnp.array([1000.0])  # sigmoid heavily saturated
    # At prey N=1, T=40: factor = 1/1601 ≈ 6.25e-4. Natural kappa_b floor
    # caps boost at 1/kappa_b = 1000x → kappa_eff = 1.0. With saturated
    # sigmoid (E=1000 >> zeta_eff/β), P_birth ≈ 1.0.
    probs = _batch_birth_prob_jax(energies, species, cfg,
                                  prey_count=jnp.int32(1), pred_count=jnp.int32(20))
    p = float(probs[0])
    assert p == pytest.approx(1.0, rel=0.01)


def test_new_config_keys_match_legacy_keys(base_config):
    """§15.23: the new density_*_threshold_* keys produce identical output
    to the legacy ddb_*/ddm_*_threshold keys."""
    cfg_legacy = {**base_config, "stability_mechanism": "ddb",
                  "ddb_pred_threshold": 4.0, "ddb_prey_threshold": 40.0,
                  "ddb_floor": 0.0, "ddb_boost_distribution_alpha": 0.5}
    cfg_new = {**base_config, "stability_mechanism": "ddb",
               "density_breeding_threshold_pred": 4.0,
               "density_breeding_threshold_prey": 40.0,
               "density_factor_floor": 0.0,
               "breeding_share_alpha": 0.5}
    species = jnp.array([0, 1])
    energies = jnp.array([100.0, 1000.0])
    is_active = jnp.array([True, True])
    p_legacy = _batch_birth_prob_jax(
        energies, species, cfg_legacy,
        prey_count=jnp.int32(20), pred_count=jnp.int32(2), is_active=is_active)
    p_new = _batch_birth_prob_jax(
        energies, species, cfg_new,
        prey_count=jnp.int32(20), pred_count=jnp.int32(2), is_active=is_active)
    assert float(p_legacy[0]) == pytest.approx(float(p_new[0]), rel=1e-6)
    assert float(p_legacy[1]) == pytest.approx(float(p_new[1]), rel=1e-6)


def test_birth_prob_legacy_max_boost_silently_ignored(base_config):
    """§15.22: configs that still set ddb_max_boost should not error.
    The knob is silently ignored; behavior matches a config without it."""
    cfg_legacy = {**base_config, "stability_mechanism": "ddb",
                  "ddb_pred_threshold": 4.0, "ddb_prey_threshold": 40.0,
                  "ddb_floor": 0.0, "ddb_max_boost": 50.0}
    cfg_clean = {**base_config, "stability_mechanism": "ddb",
                 "ddb_pred_threshold": 4.0, "ddb_prey_threshold": 40.0,
                 "ddb_floor": 0.0}
    species = jnp.array([1])
    energies = jnp.array([1000.0])
    p_legacy = float(_batch_birth_prob_jax(
        energies, species, cfg_legacy,
        prey_count=jnp.int32(100), pred_count=jnp.int32(1))[0])
    p_clean = float(_batch_birth_prob_jax(
        energies, species, cfg_clean,
        prey_count=jnp.int32(100), pred_count=jnp.int32(1))[0])
    assert p_legacy == pytest.approx(p_clean, rel=1e-6)


def test_energy_weighted_boost_redistributes_to_high_energy(base_config):
    """With energy_weighted distribution, high-energy predator gets the boost,
    low-energy ones get little/none. Total breeding pressure conserved.
    """
    cfg = {**base_config, "stability_mechanism": "ddb",
           "ddb_pred_threshold": 10.0, "ddb_prey_threshold": 100.0,
           "ddb_floor": 0.0, "ddb_max_boost": 50.0,
           "ddb_boost_distribution": "energy_weighted"}
    # 3 predators, 2 prey (ignored in pred-side analysis)
    species = jnp.array([1, 1, 1, 0, 0])
    is_active = jnp.array([True, True, True, True, True])
    # Predator energies: 800 (top), 100, 100. Prey: 100, 100.
    energies = jnp.array([800.0, 100.0, 100.0, 100.0, 100.0])
    probs = _batch_birth_prob_jax(
        energies, species, cfg,
        prey_count=jnp.int32(2), pred_count=jnp.int32(3),
        is_active=is_active,
    )
    p_top, p_mid1, p_mid2, _, _ = [float(x) for x in probs]

    # All 3 are well above zeta_eff (10*0.083 = 0.83); sigmoid ≈ 1.
    # Pred factor at N=3, T=10: 9/109 = 0.0826, boost_uniform = 1/0.0826 = 12.1
    # Total boost budget for predator species: 3 * 12.1 = 36.3
    # Energy shares: 800/(800+100+100) = 0.80, 100/1000 = 0.10, 100/1000 = 0.10
    # Per-agent boost: top = 36.3 * 0.80 = 29.0, mid = 36.3 * 0.10 = 3.6
    # kappa_b_eff_top ≈ 29e-3, kappa_b_eff_mid ≈ 3.6e-3
    assert p_top == pytest.approx(29e-3, rel=0.10)
    assert p_mid1 == pytest.approx(3.6e-3, rel=0.10)
    assert p_mid2 == pytest.approx(3.6e-3, rel=0.10)
    # Top breeds ~8x more than each low-energy peer
    assert p_top / p_mid1 > 6


def test_energy_weighted_total_budget_preserved(base_config):
    """At uniform energies within species, energy_weighted == uniform."""
    cfg_uni = {**base_config, "stability_mechanism": "ddb",
               "ddb_pred_threshold": 10.0, "ddb_prey_threshold": 100.0,
               "ddb_floor": 0.0, "ddb_max_boost": 50.0,
               "ddb_boost_distribution": "uniform"}
    cfg_ew = {**cfg_uni, "ddb_boost_distribution": "energy_weighted"}

    species = jnp.array([1, 1, 1])
    is_active = jnp.array([True, True, True])
    energies = jnp.array([200.0, 200.0, 200.0])  # all equal

    p_uni = _batch_birth_prob_jax(
        energies, species, cfg_uni,
        prey_count=jnp.int32(0), pred_count=jnp.int32(3),
        is_active=is_active,
    )
    p_ew = _batch_birth_prob_jax(
        energies, species, cfg_ew,
        prey_count=jnp.int32(0), pred_count=jnp.int32(3),
        is_active=is_active,
    )
    # At uniform energies, distributions should match
    for u, e in zip(p_uni, p_ew):
        assert float(u) == pytest.approx(float(e), rel=0.01)
    # And total breeding pressure across the species should be the same
    assert float(jnp.sum(p_uni)) == pytest.approx(float(jnp.sum(p_ew)), rel=0.01)


def test_alpha_zero_equals_uniform(base_config):
    """alpha=0 reproduces uniform-distribution behavior (legacy compat)."""
    cfg_a = {**base_config, "stability_mechanism": "ddb",
             "ddb_pred_threshold": 10.0, "ddb_prey_threshold": 100.0,
             "ddb_floor": 0.0, "ddb_max_boost": 50.0,
             "ddb_boost_distribution_alpha": 0.0}
    cfg_b = {**base_config, "stability_mechanism": "ddb",
             "ddb_pred_threshold": 10.0, "ddb_prey_threshold": 100.0,
             "ddb_floor": 0.0, "ddb_max_boost": 50.0,
             "ddb_boost_distribution": "uniform"}
    species = jnp.array([1, 1, 1])
    is_active = jnp.array([True, True, True])
    energies = jnp.array([800.0, 100.0, 100.0])
    p_a = _batch_birth_prob_jax(energies, species, cfg_a,
                                prey_count=jnp.int32(0), pred_count=jnp.int32(3),
                                is_active=is_active)
    p_b = _batch_birth_prob_jax(energies, species, cfg_b,
                                prey_count=jnp.int32(0), pred_count=jnp.int32(3),
                                is_active=is_active)
    for x, y in zip(p_a, p_b):
        assert float(x) == pytest.approx(float(y), rel=0.01)


def test_alpha_half_equals_energy_weighted(base_config):
    """alpha=0.5 reproduces energy_weighted (linear) behavior."""
    cfg_a = {**base_config, "stability_mechanism": "ddb",
             "ddb_pred_threshold": 10.0, "ddb_prey_threshold": 100.0,
             "ddb_floor": 0.0, "ddb_max_boost": 50.0,
             "ddb_boost_distribution_alpha": 0.5}
    cfg_b = {**base_config, "stability_mechanism": "ddb",
             "ddb_pred_threshold": 10.0, "ddb_prey_threshold": 100.0,
             "ddb_floor": 0.0, "ddb_max_boost": 50.0,
             "ddb_boost_distribution": "energy_weighted"}
    species = jnp.array([1, 1, 1])
    is_active = jnp.array([True, True, True])
    energies = jnp.array([800.0, 100.0, 100.0])
    p_a = _batch_birth_prob_jax(energies, species, cfg_a,
                                prey_count=jnp.int32(0), pred_count=jnp.int32(3),
                                is_active=is_active)
    p_b = _batch_birth_prob_jax(energies, species, cfg_b,
                                prey_count=jnp.int32(0), pred_count=jnp.int32(3),
                                is_active=is_active)
    for x, y in zip(p_a, p_b):
        assert float(x) == pytest.approx(float(y), rel=0.02)


def test_alpha_near_one_concentrates_to_top_predator(base_config):
    """alpha=0.95 → almost all breeding rate goes to the top-energy agent."""
    cfg = {**base_config, "stability_mechanism": "ddb",
           "ddb_pred_threshold": 10.0, "ddb_prey_threshold": 100.0,
           "ddb_floor": 0.0, "ddb_max_boost": 50.0,
           "ddb_boost_distribution_alpha": 0.95}
    species = jnp.array([1, 1, 1, 1])
    is_active = jnp.array([True, True, True, True])
    energies = jnp.array([800.0, 200.0, 200.0, 200.0])
    probs = _batch_birth_prob_jax(energies, species, cfg,
                                  prey_count=jnp.int32(0), pred_count=jnp.int32(4),
                                  is_active=is_active)
    # k = 0.95 / 0.05 = 19. share_top = 800^19 / (800^19 + 3*200^19) ≈ 1.0
    # Top should get nearly the entire species budget.
    p_top = float(probs[0])
    p_others = sum(float(x) for x in probs[1:])
    assert p_top / (p_top + p_others) > 0.95


def test_alpha_intermediate_smooth_interpolation(base_config):
    """alpha=0.7 falls between linear and winner-take-all in concentration."""
    species = jnp.array([1, 1, 1])
    is_active = jnp.array([True, True, True])
    energies = jnp.array([800.0, 100.0, 100.0])

    def top_share(alpha):
        cfg = {**base_config, "stability_mechanism": "ddb",
               "ddb_pred_threshold": 10.0, "ddb_prey_threshold": 100.0,
               "ddb_floor": 0.0, "ddb_max_boost": 50.0,
               "ddb_boost_distribution_alpha": alpha}
        probs = _batch_birth_prob_jax(energies, species, cfg,
                                      prey_count=jnp.int32(0), pred_count=jnp.int32(3),
                                      is_active=is_active)
        return float(probs[0]) / float(jnp.sum(probs))

    s0 = top_share(0.0)    # uniform: top should be 1/3 ≈ 0.33
    s5 = top_share(0.5)    # linear: top should be 800/1000 = 0.80
    s7 = top_share(0.7)    # in between higher: should be > 0.80
    s9 = top_share(0.95)   # near winner: should be ≈ 1.0

    # Monotonic increase
    assert s0 < s5 < s7 < s9
    assert s0 == pytest.approx(0.33, abs=0.05)
    assert s5 == pytest.approx(0.80, abs=0.02)
    assert s9 > 0.95


def test_energy_weighted_no_breeding_for_zero_energy_predator(base_config):
    """An active predator with zero energy gets ~zero breeding rate even
    when scaffolds are otherwise active. Selection pressure intact."""
    cfg = {**base_config, "stability_mechanism": "ddb",
           "ddb_pred_threshold": 10.0, "ddb_prey_threshold": 100.0,
           "ddb_floor": 0.0, "ddb_max_boost": 50.0,
           "ddb_boost_distribution": "energy_weighted"}
    species = jnp.array([1, 1])
    is_active = jnp.array([True, True])
    energies = jnp.array([1000.0, 0.0])  # one full, one starving
    probs = _batch_birth_prob_jax(
        energies, species, cfg,
        prey_count=jnp.int32(0), pred_count=jnp.int32(2),
        is_active=is_active,
    )
    p_full, p_starve = float(probs[0]), float(probs[1])
    assert p_full > 1e-3  # Top one breeds normally or better
    assert p_starve < 1e-6  # Starving one barely breeds (sigmoid + zero rate)


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
