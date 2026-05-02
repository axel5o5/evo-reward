import pytest

from src.save_schedule import parse_schedule, interval_at


def test_int_legacy_form():
    sched = parse_schedule(20_000, default_interval=100_000)
    assert sched == [(0, 20_000)]
    assert interval_at(sched, 0) == 20_000
    assert interval_at(sched, 1_000_000) == 20_000


def test_none_uses_default():
    sched = parse_schedule(None, default_interval=50_000)
    assert sched == [(0, 50_000)]


def test_taper_schedule():
    sched = parse_schedule(
        [
            {"after": 0, "interval": 20_000},
            {"after": 200_000, "interval": 50_000},
            {"after": 1_000_000, "interval": 100_000},
        ],
        default_interval=999,
    )
    assert sched == [(0, 20_000), (200_000, 50_000), (1_000_000, 100_000)]
    assert interval_at(sched, 0) == 20_000
    assert interval_at(sched, 199_999) == 20_000
    assert interval_at(sched, 200_000) == 50_000
    assert interval_at(sched, 999_999) == 50_000
    assert interval_at(sched, 1_000_000) == 100_000
    assert interval_at(sched, 5_000_000) == 100_000


def test_unsorted_input_is_sorted():
    sched = parse_schedule(
        [
            {"after": 1_000_000, "interval": 100_000},
            {"after": 0, "interval": 20_000},
            {"after": 200_000, "interval": 50_000},
        ],
        default_interval=0,
    )
    assert [s[0] for s in sched] == [0, 200_000, 1_000_000]


def test_must_start_at_zero():
    with pytest.raises(ValueError, match="must start with after=0"):
        parse_schedule(
            [{"after": 100, "interval": 20_000}],
            default_interval=0,
        )


def test_empty_list_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        parse_schedule([], default_interval=0)


def test_bool_rejected():
    with pytest.raises(ValueError, match="bool"):
        parse_schedule(True, default_interval=0)


def test_negative_interval_rejected():
    with pytest.raises(ValueError, match="positive"):
        parse_schedule(
            [{"after": 0, "interval": -10}],
            default_interval=0,
        )


def test_missing_keys_rejected():
    with pytest.raises(ValueError, match="must be dicts"):
        parse_schedule(
            [{"after": 0}],
            default_interval=0,
        )
