"""Trajectory recorder: record -> save -> load round-trip, and schema validation."""

from __future__ import annotations

import json

import pytest

from f1rl.sim.recorder import (
    TrajectoryError,
    TrajectoryRecorder,
    load_trajectory,
    validate_trajectory,
)


def _frame(t, x):
    return (
        t,
        {"x": x, "y": 0.0, "yaw": 0.0, "speed": x * 2},
        {"speed_kmh": round(x * 7.2), "lap_time": t, "lap": 1},
    )


def test_record_save_load_roundtrip(tmp_path):
    rec = TrajectoryRecorder(track_id="oval", dt=0.05, seed=42)
    for i in range(5):
        rec.append(*_frame(i * 0.05, float(i)))
    path = rec.save(tmp_path / "run.json")

    loaded = load_trajectory(path)
    assert loaded["meta"]["track_id"] == "oval"
    assert loaded["meta"]["dt"] == 0.05
    assert loaded["frames"] == rec.frames


def test_missing_file_raises():
    with pytest.raises(TrajectoryError):
        load_trajectory("does-not-exist.json")


def test_invalid_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(TrajectoryError):
        load_trajectory(p)


def test_validation_rejects_bad_structure(tmp_path):
    # Missing required car keys.
    bad = {"meta": {"track_id": "oval", "dt": 0.05}, "frames": [{"t": 0.0, "car": {"x": 1.0}}]}
    p = tmp_path / "bad2.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(TrajectoryError):
        load_trajectory(p)


def test_validation_rejects_empty_frames():
    with pytest.raises(TrajectoryError):
        validate_trajectory({"meta": {"track_id": "oval", "dt": 0.05}, "frames": []})
