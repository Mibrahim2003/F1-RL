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


# ----- Phase 6: racing fields ride along under per-car telemetry -------------------------


def test_racing_fields_round_trip_under_telemetry(tmp_path):
    # race_position / gap_ahead_s / contact live under each car's freeform telemetry, so the
    # format is unchanged and they replay verbatim (backward compatible superset).
    rec = TrajectoryRecorder(track_id="monza", dt=0.05, seed=1, n_agents=2)
    for f in range(3):
        cars = [
            {
                "id": f"car_{i}",
                "team": i,
                "x": float(f + i),
                "y": 0.0,
                "yaw": 0.0,
                "speed": 50.0,
                "telemetry": {
                    "speed_kmh": 180,
                    "race_position": i + 1,
                    "gap_ahead_s": None if i == 0 else round(0.3 * i, 3),
                    "contact": 0.0,
                },
            }
            for i in range(2)
        ]
        rec.append_cars(f * 0.05, cars)
    loaded = load_trajectory(rec.save(tmp_path / "race.json"))

    leader = loaded["frames"][1]["cars"][0]["telemetry"]
    chaser = loaded["frames"][1]["cars"][1]["telemetry"]
    assert leader["race_position"] == 1 and leader["gap_ahead_s"] is None
    assert chaser["race_position"] == 2 and chaser["gap_ahead_s"] == pytest.approx(0.3)
    assert loaded["frames"] == rec.frames  # exact round-trip
