"""Server: track/meta HTTP routes, the catalog, track switching, surface saving, the WS
input->state round-trip, and input clamping."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from f1rl.server.app import create_app, scan_checkpoints
from f1rl.server.messages import InputMessage, PolicyMessage, parse_client_message
from f1rl.track.build import BuildConfig, build_from_points, save_track
from f1rl.track.schema import Track
from f1rl.utils.config import load_config


def _client() -> TestClient:
    return TestClient(create_app())


def _build_synthetic(track_id: str, cache_dir: Path) -> None:
    """Write a synthetic ellipse circuit to ``cache_dir/<track_id>.npz`` for server tests."""
    a, b, n = 500.0, 300.0, 400
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([a * np.cos(th), b * np.sin(th)])
    peri = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    cfg = BuildConfig(id=track_id, country="Italy", official_length_m=peri)
    track, report = build_from_points(pts, cfg, source="fastf1")
    save_track(track, report, cache_dir=cache_dir)


def _client_with_tracks(tmp_path: Path, *track_ids: str) -> TestClient:
    """App whose ``tracks_dir`` is a temp dir holding the given built circuits."""
    for tid in track_ids:
        _build_synthetic(tid, tmp_path)
    cfg = load_config("default")
    cfg.server.tracks_dir = str(tmp_path)
    return TestClient(create_app(cfg))


def _recv_event(ws, event: str, max_n: int = 40) -> dict:
    """Read frames until the named event arrives (state frames may interleave)."""
    for _ in range(max_n):
        msg = ws.receive_json()
        if msg.get("type") == "event" and msg.get("event") == event:
            return msg
    raise AssertionError(f"event '{event}' not received")


def test_get_track_returns_geometry():
    with _client() as client:
        resp = client.get("/track/oval")
        assert resp.status_code == 200
        data = resp.json()
        assert data["closed"] is True
        centerline = data["centerline"]
        assert isinstance(centerline, list) and centerline
        assert len(centerline[0]) == 2  # [x, y] pairs


def test_get_track_unknown_id_404():
    with _client() as client:
        # An id with no cached .npz (monza et al. are now built circuits) → 404.
        assert client.get("/track/no_such_circuit").status_code == 404


def test_get_meta():
    with _client() as client:
        meta = client.get("/api/meta").json()
        assert meta["control_hz"] == 20
        assert isinstance(meta["total_laps"], int)
        assert isinstance(meta["pole_str"], str)
        assert ":" in meta["pole_str"]


def test_ws_input_to_state_accelerates():
    with _client() as client, client.websocket_connect("/ws/sim") as ws:
        ws.send_json({"type": "mode", "mode": "manual"})
        ws.send_json({"type": "input", "throttle": 1.0})
        speeds = []
        for _ in range(6):
            frame = ws.receive_json()
            assert frame["type"] == "state"
            speeds.append(frame["car"]["speed"])
        # The car accelerates under full throttle from rest.
        assert speeds[-1] > speeds[0]


def test_ws_input_clamping_no_crash():
    with _client() as client, client.websocket_connect("/ws/sim") as ws:
        ws.send_json({"type": "input", "steer": 5.0, "throttle": 9.0})
        for _ in range(3):
            frame = ws.receive_json()
            assert frame["type"] == "state"


def test_input_message_clamps_axes():
    msg = InputMessage(steer=5.0, throttle=9.0, brake=-2.0)
    assert msg.steer == 1.0
    assert msg.throttle == 1.0
    assert msg.brake == 0.0
    assert msg.longitudinal == 1.0


def test_parse_unknown_message_returns_none():
    assert parse_client_message({"type": "bogus"}) is None
    assert parse_client_message({}) is None
    assert parse_client_message({"type": "control", "action": "nope"}) is None


# ----- Phase 2: catalog, track switching, surface editing --------------------------------


def test_api_tracks_catalog_lists_oval_and_built(tmp_path):
    with _client_with_tracks(tmp_path, "monza") as client:
        cat = client.get("/api/tracks").json()["tracks"]
        ids = {t["id"] for t in cat}
        assert "oval" in ids and "monza" in ids
        monza = next(t for t in cat if t["id"] == "monza")
        assert monza["country"] == "Italy"
        assert monza["turns"] >= 1
        assert "low_confidence" in monza and "length" in monza


def test_get_built_track_returns_bands(tmp_path):
    with _client_with_tracks(tmp_path, "monza") as client:
        data = client.get("/track/monza").json()
        assert data["closed"] is True
        for key in ("kerb_width", "grass_width", "gravel_width", "half_width_left"):
            assert isinstance(data[key], list) and data[key]
        assert data["source"] == "fastf1"


def test_get_unbuilt_track_404(tmp_path):
    # 'spa' has a config but no .npz in the temp dir.
    with _client_with_tracks(tmp_path, "monza") as client:
        assert client.get("/track/spa").status_code == 404


def test_ws_track_switch(tmp_path):
    client = _client_with_tracks(tmp_path, "monza")
    with client, client.websocket_connect("/ws/sim") as ws:
        ws.send_json({"type": "track", "id": "monza"})
        ev = _recv_event(ws, "track_changed")
        assert ev["id"] == "monza"
        # the event carries the new circuit's pace meta (from configs/track/monza.yaml)
        assert ev["pole_time_s"] > 0
        assert ev["total_laps"] >= 1
        assert ":" in ev["pole_str"]


def test_ws_track_switch_unknown_emits_error(tmp_path):
    client = _client_with_tracks(tmp_path, "monza")
    with client, client.websocket_connect("/ws/sim") as ws:
        ws.send_json({"type": "track", "id": "spa"})
        ev = _recv_event(ws, "track_error")
        assert ev["id"] == "spa"


def test_surface_save_round_trip(tmp_path):
    with _client_with_tracks(tmp_path, "monza") as client:
        resp = client.post(
            "/track/monza/surfaces", json={"half_width_left": 9.0, "kerb_width": 1.5}
        )
        assert resp.status_code == 200 and resp.json()["ok"] is True
        saved = Track.from_npz(tmp_path / "monza.npz")
        assert np.allclose(saved.half_width_left, 9.0)
        assert np.allclose(saved.kerb_width, 1.5)
        # A backup of the previous cache is kept for rollback.
        assert (tmp_path / "monza.npz.bak").exists()


def test_surface_save_rejects_out_of_bounds(tmp_path):
    with _client_with_tracks(tmp_path, "monza") as client:
        # 99 m half-width is outside [0.5, 25] → pydantic 422.
        resp = client.post("/track/monza/surfaces", json={"half_width_left": 99.0})
        assert resp.status_code == 422


def test_surface_save_oval_rejected(tmp_path):
    with _client_with_tracks(tmp_path, "monza") as client:
        assert client.post("/track/oval/surfaces", json={"kerb_width": 2.0}).status_code == 400


def test_surface_save_unbuilt_404(tmp_path):
    with _client_with_tracks(tmp_path, "monza") as client:
        assert client.post("/track/spa/surfaces", json={"kerb_width": 2.0}).status_code == 404


# ----- Phase 3a: checkpoint catalog + watch-live policy picker ----------------------------


def _write_fake_checkpoint(
    root: Path, rel_id: str, *, total_timesteps: int = 1000, obs_version: int = 1
) -> Path:
    """Write a minimal checkpoint dir (model.zip + meta.json) for catalog/scan tests.

    The ``model.zip`` is a stub — these tests cover the scanner and the listing route, not a
    real torch load, so no SB3 dependency is pulled in.
    """
    ckpt = root / rel_id
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "model.zip").write_bytes(b"stub")
    meta = {
        "total_timesteps": total_timesteps,
        "circuit_id": "red_bull_ring",
        "obs_version": obs_version,
        "action_shape": [2],
        "seed": 42,
    }
    (ckpt / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return ckpt


def _client_with_checkpoints(tmp_path: Path) -> TestClient:
    cfg = load_config("default")
    cfg.server.checkpoints_dir = str(tmp_path)
    return TestClient(create_app(cfg))


def test_scan_checkpoints_finds_nested(tmp_path):
    _write_fake_checkpoint(tmp_path, "runA/checkpoints/final", total_timesteps=5000)
    _write_fake_checkpoint(tmp_path, "runB/checkpoints/early", total_timesteps=1000)
    # A dir missing model.zip is not a checkpoint.
    (tmp_path / "not_a_ckpt").mkdir()
    (tmp_path / "not_a_ckpt" / "meta.json").write_text("{}", encoding="utf-8")

    items = scan_checkpoints(tmp_path)
    ids = {it["id"] for it in items}
    assert ids == {"runA/checkpoints/final", "runB/checkpoints/early"}
    # Sorted least-trained first within a circuit (early before final).
    assert items[0]["total_timesteps"] == 1000
    assert items[0]["circuit_id"] == "red_bull_ring"
    assert items[0]["obs_version"] == 1


def test_scan_checkpoints_missing_root_empty(tmp_path):
    assert scan_checkpoints(tmp_path / "nope") == []


def test_api_checkpoints_route(tmp_path):
    _write_fake_checkpoint(tmp_path, "runA/final")
    with _client_with_checkpoints(tmp_path) as client:
        data = client.get("/api/checkpoints").json()["checkpoints"]
        assert len(data) == 1
        entry = data[0]
        assert entry["id"] == "runA/final"
        assert set(entry) == {"id", "total_timesteps", "circuit_id", "obs_version"}


def test_policy_message_parses():
    msg = parse_client_message({"type": "policy", "source": "autopilot"})
    assert isinstance(msg, PolicyMessage)
    assert msg.source == "autopilot" and msg.id is None
    msg2 = parse_client_message({"type": "policy", "source": "checkpoint", "id": "runA/final"})
    assert isinstance(msg2, PolicyMessage)
    assert msg2.source == "checkpoint" and msg2.id == "runA/final"


def test_policy_message_rejects_bad_source():
    assert parse_client_message({"type": "policy", "source": "bogus"}) is None


def test_ws_policy_autopilot_roundtrip(tmp_path):
    # Switching to the autopilot needs no checkpoint load (no torch), so this stays fast.
    with _client_with_checkpoints(tmp_path) as client, client.websocket_connect("/ws/sim") as ws:
        ws.send_json({"type": "policy", "source": "autopilot"})
        ev = _recv_event(ws, "policy_changed")
        assert ev["source"] == "autopilot"


# ----- Phase 4: calendar lap-time table result view --------------------------------------


def test_api_calendar_404_when_absent(tmp_path):
    cfg = load_config("default")
    cfg.server.calendar_path = str(tmp_path / "nope.json")
    with TestClient(create_app(cfg)) as client:
        assert client.get("/api/calendar").status_code == 404


def test_api_calendar_serves_saved_table(tmp_path):
    table = {
        "rows": [
            {
                "circuit": "monza",
                "best_lap_time": 81.0,
                "pole_time_s": 79.8,
                "delta_to_pole": 1.2,
                "beat_2x_pole_rate": 1.0,
                "pole_missing": False,
            }
        ],
        "aggregates": {"n_circuits": 1, "n_completed": 1},
    }
    path = tmp_path / "calendar_benchmark.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    cfg = load_config("default")
    cfg.server.calendar_path = str(path)
    with TestClient(create_app(cfg)) as client:
        data = client.get("/api/calendar").json()
        assert data["rows"][0]["circuit"] == "monza"
        assert data["aggregates"]["n_circuits"] == 1


def test_ws_policy_bad_checkpoint_falls_back(tmp_path):
    # A missing/unknown checkpoint id must surface policy_error, never crash the socket.
    with _client_with_checkpoints(tmp_path) as client, client.websocket_connect("/ws/sim") as ws:
        ws.send_json({"type": "policy", "source": "checkpoint", "id": "does/not/exist"})
        ev = _recv_event(ws, "policy_error")
        assert ev["id"] == "does/not/exist"
        # The sim keeps streaming state frames after the error (socket alive).
        ws.send_json({"type": "mode", "mode": "watch"})
        frame = ws.receive_json()
        assert frame["type"] in {"state", "event"}
