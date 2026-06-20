"""Calendar lap-time table (Phase 4 spec §2b "Calendar benchmark", §c; plan §B).

Written from the benchmark's public behavior and the spec: one row per pool circuit, each row
scored against THAT circuit's pole, delta = best_lap − pole, a missing pole flagged and skipped
(never zero-divided), and the table saved as JSON + CSV. The per-circuit metric logic is
``evaluate`` (tested elsewhere); here we check the loop, the pole/config resolution, the
aggregation, and the save — not the policy quality.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from f1rl.train.calendar_benchmark import (
    _cfg_for_circuit,
    _save_table,
    aggregate,
    benchmark_circuit,
)
from f1rl.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"
TRACKS_DIR = REPO_ROOT / "data" / "tracks"

pytestmark = pytest.mark.skipif(
    not (TRACKS_DIR / "red_bull_ring.npz").exists(),
    reason="cached track 'red_bull_ring' required for the calendar benchmark test",
)


class _ZeroModel:
    """A stand-in policy that always brakes/coasts (so the episode never completes a lap)."""

    def predict(self, obs, deterministic=True):  # noqa: D401, ANN001
        return np.zeros(2, dtype=np.float32), None


def _base_cfg():
    cfg = load_config("experiment/calendar_dynamic", config_root=CONFIG_ROOT)
    cfg.env.max_steps = 40  # keep the per-circuit episode short for the test
    return cfg


def test_cfg_for_circuit_binds_single_circuit_and_pole():
    cfg = _cfg_for_circuit(_base_cfg(), "monza")
    assert cfg.track_id == "monza"
    assert list(cfg.circuits.pool) == ["monza"]  # the eval env binds exactly this circuit
    assert cfg.track.pole_time_s == pytest.approx(79.8)


def test_benchmark_circuit_uses_that_circuits_pole():
    row = benchmark_circuit(
        _ZeroModel(),
        _base_cfg(),
        "red_bull_ring",
        episodes=1,
        seed=0,
        obs_rms=None,
        clip_obs=10.0,
    )
    assert row["circuit"] == "red_bull_ring"
    assert row["pole_time_s"] == pytest.approx(64.3)
    assert row["pole_missing"] is False
    # No lap completed in 40 braking steps -> best lap and delta are NaN, not a bogus number.
    assert math.isnan(float(row["best_lap_time"]))
    assert math.isnan(float(row["delta_to_pole"]))


def test_aggregate_skips_missing_pole_and_nan_laps():
    rows = [
        _row("a", best=70.0, pole=64.3, delta=5.7, two_x=1.0, missing=False),
        _row("b", best=float("nan"), pole=80.0, delta=float("nan"), two_x=0.0, missing=False),
        _row("c", best=50.0, pole=0.0, delta=float("nan"), two_x=0.0, missing=True),
    ]
    agg = aggregate(rows)
    assert agg["n_circuits"] == 3
    assert agg["n_completed"] == 2  # a and c completed a lap; b did not
    assert agg["mean_delta_to_pole"] == pytest.approx(5.7)  # only 'a' has a real delta
    assert agg["worst_delta_to_pole"] == pytest.approx(5.7)
    assert agg["worst_circuit"] == "a"


def test_save_table_writes_json_and_csv(tmp_path):
    rows = [_row("a", best=70.0, pole=64.3, delta=5.7, two_x=1.0, missing=False)]
    table = {"rows": rows, "aggregates": aggregate(rows)}
    paths = _save_table(table, tmp_path)
    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).exists()
    loaded = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert loaded["rows"][0]["circuit"] == "a"
    assert loaded["aggregates"]["n_circuits"] == 1
    csv_text = Path(paths["csv"]).read_text(encoding="utf-8")
    assert "circuit" in csv_text and "a" in csv_text


def _row(cid, *, best, pole, delta, two_x, missing):
    return {
        "circuit": cid,
        "best_lap_time": best,
        "pole_time_s": pole,
        "delta_to_pole": delta,
        "beat_pole_rate": 0.0,
        "beat_2x_pole_rate": two_x,
        "off_track_count": 0.0,
        "completed_laps": 0.0 if math.isnan(float(best)) else 1.0,
        "pole_missing": missing,
    }
