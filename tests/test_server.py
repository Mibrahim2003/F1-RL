"""Server: track/meta HTTP routes, the WS input->state round-trip, and input clamping."""

from __future__ import annotations

from fastapi.testclient import TestClient

from f1rl.server.app import create_app
from f1rl.server.messages import InputMessage, parse_client_message


def _client() -> TestClient:
    return TestClient(create_app())


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
