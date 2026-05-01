"""Unit tests for src/config_utils.py."""
import pytest

from src.config_utils import resolve_scale_dependent_params


def test_food_growth_rate_at_paper_world_unchanged():
    cfg = {"world_size": 960, "food_growth_rate_at_960sq": 0.5}
    resolve_scale_dependent_params(cfg)
    assert cfg["food_growth_rate"] == pytest.approx(0.5)


def test_food_growth_rate_scales_with_area():
    cfg = {"world_size": 880, "food_growth_rate_at_960sq": 0.5}
    resolve_scale_dependent_params(cfg)
    expected = 0.5 * (880 / 960) ** 2
    assert cfg["food_growth_rate"] == pytest.approx(expected)


def test_food_growth_rate_scales_for_smaller_world():
    cfg = {"world_size": 480, "food_growth_rate_at_960sq": 0.5}
    resolve_scale_dependent_params(cfg)
    assert cfg["food_growth_rate"] == pytest.approx(0.5 * 0.25)


def test_legacy_absolute_key_untouched():
    cfg = {"world_size": 880, "food_growth_rate": 0.5}
    resolve_scale_dependent_params(cfg)
    assert cfg["food_growth_rate"] == 0.5
    assert "food_growth_rate_at_960sq" not in cfg


def test_both_keys_raises():
    cfg = {
        "world_size": 880,
        "food_growth_rate": 0.5,
        "food_growth_rate_at_960sq": 0.5,
    }
    with pytest.raises(ValueError, match="both"):
        resolve_scale_dependent_params(cfg)


def test_missing_world_size_raises():
    cfg = {"food_growth_rate_at_960sq": 0.5}
    with pytest.raises(ValueError, match="world_size"):
        resolve_scale_dependent_params(cfg)
