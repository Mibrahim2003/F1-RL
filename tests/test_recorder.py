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


# ----- Phase 5: multi-car (field) recorder ----------------------------------------------


def _car_entry(i, x):
    return {
        "id": f"car_{i}",
        "team": i,
        "x": x,
        "y": 0.0,
        "yaw": 0.0,
        "speed": x,
        "telemetry": {"speed_kmh": round(x * 3.6), "lap_time": 0.0, "progress": 0.0},
    }


def test_multi_car_record_save_load_roundtrip(tmp_path):
    rec = TrajectoryRecorder(track_id="monza", dt=0.05, seed=7, n_agents=3)
    for f in range(4):
        rec.append_cars(f * 0.05, [_car_entry(i, float(f + i)) for i in range(3)])
    path = rec.save(tmp_path / "field.json")

    loaded = load_trajectory(path)
    assert loaded["meta"]["n_agents"] == 3
    assert len(loaded["frames"][0]["cars"]) == 3
    assert loaded["frames"] == rec.frames


def test_validation_accepts_multi_car_frame():
    data = {
        "meta": {"track_id": "x", "dt": 0.05},
        "frames": [{"t": 0.0, "cars": [{"x": 1.0, "y": 0.0, "yaw": 0.0, "speed": 1.0}]}],
    }
    assert validate_trajectory(data) is data


def test_validation_rejects_empty_cars_list():
    with pytest.raises(TrajectoryError):
        validate_trajectory(
            {"meta": {"track_id": "x", "dt": 0.05}, "frames": [{"t": 0.0, "cars": []}]}
        )


def test_validation_rejects_car_missing_keys_in_multi_car_frame():
    with pytest.raises(TrajectoryError):
        validate_trajectory(
            {"meta": {"track_id": "x", "dt": 0.05}, "frames": [{"t": 0.0, "cars": [{"x": 1.0}]}]}
        )
