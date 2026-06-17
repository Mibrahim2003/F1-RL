"""Lap-time benchmark + calibration contract (spec §1d/§3, plan "Lap-time benchmark").

Covers the delta-to-pole / gap-to-pole math and missing-pole handling on
``f1rl.train.evaluate`` (built directly from the public ``EpisodeMetrics`` / ``EvalResult``
schema, no model needed), and the deterministic lap-time estimator in
``f1rl.train.calibrate`` (positive, monotone in grip, lands near pole for the calibrated
config). Maps to: "lap time + delta correct against a known reference; the 2*pole milestone
flag fires; missing pole is skipped/flagged, never crashes."
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from f1rl.train.evaluate import EpisodeMetrics, EvalResult

_TRACKS_DIR = Path(__file__).resolve().parents[1] / "data" / "tracks"
pytestmark = pytest.mark.skipif(
    not (_TRACKS_DIR / "red_bull_ring.npz").exists(),
    reason="cached track 'red_bull_ring' not found in data/tracks/",
)

POLE = 64.3


def _finish(metrics: EpisodeMetrics, best_lap: float, pole: float) -> EpisodeMetrics:
    """Apply the same end-of-episode vs-pole bookkeeping run_episode does."""
    metrics.best_lap_time = best_lap
    metrics.pole_missing = pole <= 0.0
    if best_lap is not None and pole > 0.0:
        metrics.beat_pole = best_lap <= pole
        metrics.beat_2x_pole = best_lap <= 2.0 * pole
        metrics.lap_delta_to_pole = float(best_lap) - float(pole)
    return metrics


# --- delta / gap math -------------------------------------------------------------------


def test_lap_delta_to_pole_sign_and_value():
    slower = _finish(EpisodeMetrics(), best_lap=70.0, pole=POLE)
    assert slower.lap_delta_to_pole == pytest.approx(70.0 - POLE)
    assert slower.lap_delta_to_pole > 0  # slower than pole -> positive
    assert slower.beat_pole is False

    faster = _finish(EpisodeMetrics(), best_lap=63.0, pole=POLE)
    assert faster.lap_delta_to_pole == pytest.approx(63.0 - POLE)
    assert faster.beat_pole is True


def test_two_x_pole_milestone_flag_fires():
    # Below 2*pole but above pole: beat_2x_pole True, beat_pole False.
    m = _finish(EpisodeMetrics(), best_lap=1.5 * POLE, pole=POLE)
    assert m.beat_2x_pole is True
    assert m.beat_pole is False
    # Above 2*pole: neither.
    slow = _finish(EpisodeMetrics(), best_lap=2.5 * POLE, pole=POLE)
    assert slow.beat_2x_pole is False


def test_summary_gap_to_pole_uses_best_episode():
    result = EvalResult(
        episodes=[
            _finish(EpisodeMetrics(), best_lap=72.0, pole=POLE),
            _finish(EpisodeMetrics(), best_lap=68.0, pole=POLE),
        ]
    )
    summary = result.summary(POLE)
    assert summary["eval/best_lap_time"] == pytest.approx(68.0)
    assert summary["eval/gap_to_pole"] == pytest.approx(68.0 - POLE)
    assert summary["eval/pole_missing"] == 0.0


# --- missing pole: flagged, never crashes ------------------------------------------------


def test_missing_pole_is_flagged_not_divided():
    m = _finish(EpisodeMetrics(), best_lap=70.0, pole=0.0)
    assert m.pole_missing is True
    assert m.lap_delta_to_pole is None  # skipped, not computed against zero
    d = m.as_dict()
    assert d["eval/pole_missing"] == 1.0
    assert math.isnan(d["eval/lap_delta_to_pole"])


def test_summary_missing_pole_does_not_crash():
    result = EvalResult(episodes=[_finish(EpisodeMetrics(), best_lap=70.0, pole=0.0)])
    summary = result.summary(0.0)
    assert summary["eval/pole_missing"] == 1.0
    assert math.isnan(summary["eval/gap_to_pole"])


# --- deterministic calibration estimator -------------------------------------------------


def test_estimate_lap_time_positive_and_finite(track):
    from f1rl.train.calibrate import estimate_lap_time, inputs_from_cfg
    from f1rl.utils.config import load_config

    cfg = load_config("experiment/rbr_dynamic")
    t = estimate_lap_time(track, inputs_from_cfg(cfg))
    assert math.isfinite(t)
    assert t > 0.0


def test_higher_grip_lowers_estimated_lap_time(track):
    import dataclasses

    from f1rl.train.calibrate import estimate_lap_time, inputs_from_cfg
    from f1rl.utils.config import load_config

    cfg = load_config("experiment/rbr_dynamic")
    base = inputs_from_cfg(cfg)
    low = estimate_lap_time(track, dataclasses.replace(base, grip=1.0))
    high = estimate_lap_time(track, dataclasses.replace(base, grip=2.5))
    assert high < low  # more grip -> faster lap


def test_calibrated_config_lands_near_pole(track):
    # The calibrated rbr_dynamic config should estimate a clean optimal lap near the pole
    # (the whole point of calibration — a fair benchmark). Generous tolerance.
    from f1rl.train.calibrate import estimate_lap_time, inputs_from_cfg
    from f1rl.utils.config import load_config

    cfg = load_config("experiment/rbr_dynamic")
    t = estimate_lap_time(track, inputs_from_cfg(cfg))
    assert abs(t - POLE) < 6.0, f"calibrated lap estimate {t:.2f}s far from pole {POLE}s"
