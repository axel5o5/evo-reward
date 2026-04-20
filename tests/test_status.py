"""
test_status.py
--------------
Unit tests for scripts/status.py — the log-line regex is the main thing
worth covering, since it's the parse path for every monitoring check.
Everything else in status.py is thin wrappers around gcloud subprocess
calls, which is validated live on a real VM rather than in unit tests.

Run: pytest tests/test_status.py -v
"""

import os
import sys

import pytest

# status.py lives in scripts/, not src/, so add it to the import path.
_STATUS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _STATUS_DIR)

import status  # noqa: E402


class TestParseProgress:

    def test_basic_line(self):
        line = ("Step     5000/10240000 | prey=301 pred=50 food=600 | "
                "E=187.2 | w_pred=-0.02±0.66 w_prey=+0.02±0.56 | "
                "34.9 sps | 143s")
        p = status.parse_latest_progress(line)
        assert p is not None
        assert p["step"] == "5000"
        assert p["total"] == "10240000"
        assert p["prey"] == "301"
        assert p["pred"] == "50"
        assert p["food"] == "600"
        assert p["energy"] == "187.2"
        assert p["wpd_mean"] == "-0.02"
        assert p["wpd_std"] == "0.66"
        assert p["wpy_mean"] == "+0.02"
        assert p["wpy_std"] == "0.56"
        assert p["sps"] == "34.9"
        assert p["elapsed"] == "143"

    def test_returns_latest_when_multiple(self):
        log = "\n".join([
            "Step     5000/10240000 | prey=301 pred=50 food=600 | E=187.2 | "
            "w_pred=-0.02±0.66 w_prey=+0.02±0.56 | 34.9 sps | 143s",
            "Step    10000/10240000 | prey=305 pred=48 food=598 | E=190.0 | "
            "w_pred=-0.03±0.67 w_prey=+0.04±0.58 | 35.2 sps | 285s",
        ])
        p = status.parse_latest_progress(log)
        assert p["step"] == "10000"

    def test_handles_negative_energy(self):
        line = ("Step 100/200 | prey= 10 pred= 2 food= 20 | E=-5.0 | "
                "w_pred=-0.01±0.10 w_prey=+0.01±0.10 | 5.0 sps | 20s")
        p = status.parse_latest_progress(line)
        assert p is not None
        assert p["energy"] == "-5.0"

    def test_handles_padded_step_numbers(self):
        # Runner uses right-padded step numbers like "Step       100/10240000"
        line = ("Step      100/10240000 | prey= 50 pred=  5 food= 40 | "
                "E=100.0 | w_pred=+0.00±0.10 w_prey=+0.00±0.10 | "
                "2.0 sps | 50s")
        p = status.parse_latest_progress(line)
        assert p is not None
        assert p["step"] == "100"

    def test_non_progress_line_returns_none(self):
        assert status.parse_latest_progress("Starting experiment") is None
        assert status.parse_latest_progress("Compiling sim_step_core...") is None
        assert status.parse_latest_progress("") is None


class TestFmtDuration:

    def test_none_returns_dash(self):
        assert status.fmt_duration(None) == "—"

    def test_seconds_format(self):
        assert status.fmt_duration(0) == "00:00:00"
        assert status.fmt_duration(65) == "00:01:05"
        assert status.fmt_duration(3661) == "01:01:01"

    def test_days_format_when_over_24h(self):
        assert status.fmt_duration(86400 + 3600 * 2) == "1d 02h"
        assert status.fmt_duration(86400 * 3 + 3600 * 15) == "3d 15h"


class TestCountRetries:

    def test_no_retries(self):
        log = "Step 5000/10240000 | prey=301 pred=50 ..."
        assert status.count_retries(log) == 0

    def test_counts_retry_lines(self):
        log = "\n".join([
            "Step 5000/...",
            "[phase1a] exit 1 at Mon Apr 20 14:30:00 UTC 2026, retrying in 30s",
            "Step 5000/...",
            "[phase1a] exit 137 at Mon Apr 20 15:00:00 UTC 2026, retrying in 30s",
            "Step 5000/...",
        ])
        assert status.count_retries(log) == 2
