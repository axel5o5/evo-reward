"""Stepped save cadence for checkpoints and replays.

A schedule is a list of (after_step, interval) tuples sorted ascending
by after_step. The interval that applies at any given step is the one
attached to the largest after_step <= step. Schedules must start at
after=0.

Config key may be either:
  - int (legacy):           20000             → [(0, 20000)]
  - list of dicts (taper):  [{after: 0, interval: 20000}, ...]

Both forms are accepted everywhere a schedule is consumed.
"""
from __future__ import annotations

from typing import List, Tuple, Union

Schedule = List[Tuple[int, int]]
RawScheduleValue = Union[int, list, None]


def parse_schedule(config_value: RawScheduleValue, default_interval: int) -> Schedule:
    if config_value is None:
        return [(0, int(default_interval))]
    if isinstance(config_value, bool):
        raise ValueError(f"save schedule must be int or list, got bool: {config_value!r}")
    if isinstance(config_value, int):
        return [(0, int(config_value))]
    if isinstance(config_value, list):
        if not config_value:
            raise ValueError("save schedule list cannot be empty")
        entries: Schedule = []
        for d in config_value:
            if not isinstance(d, dict) or "after" not in d or "interval" not in d:
                raise ValueError(
                    f"save schedule entries must be dicts with keys 'after' and "
                    f"'interval'; got {d!r}"
                )
            entries.append((int(d["after"]), int(d["interval"])))
        entries.sort(key=lambda x: x[0])
        if entries[0][0] != 0:
            raise ValueError("save schedule must start with after=0")
        for _, interval in entries:
            if interval <= 0:
                raise ValueError(f"save schedule intervals must be positive; got {interval}")
        return entries
    raise ValueError(f"unrecognized save schedule: {config_value!r}")


def interval_at(schedule: Schedule, step: int) -> int:
    interval = schedule[0][1]
    for after, ival in schedule:
        if step >= after:
            interval = ival
        else:
            break
    return interval
