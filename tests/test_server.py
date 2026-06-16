"""Server: track/meta HTTP routes, the catalog, track switching, surface saving, the WS
input->state round-trip, and input clamping."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from f1rl.server.app import create_app
from f1rl.server.messages import InputMessage, parse_client_message
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
        assert client.get("/track/monza").status_code == 404


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
