"""FastAPI backend: the Python sim engine is the single source of truth.

The browser renders state and sends input over a WebSocket; nothing here imports a
renderer. Each ``/ws/sim`` connection owns its own :class:`~f1rl.sim.loop.SimLoop` and
:class:`~f1rl.sim.autopilot.CenterlineAutopilot`, advancing the sim on a fixed clock and
streaming state frames. HTTP routes serve the static track geometry, run metadata, and
recorded trajectories for replay.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import DictConfig

from f1rl.physics.kinematic import KinematicBicycle, KinematicParams
from f1rl.server.messages import (
    ControlMessage,
    InputMessage,
    ModeMessage,
    RecordMessage,
    parse_client_message,
)
from f1rl.sim.autopilot import CenterlineAutopilot
from f1rl.sim.loop import SimConfig, SimLoop
from f1rl.sim.recorder import TrajectoryError, TrajectoryRecorder, load_trajectory
from f1rl.track.loader import load_track
from f1rl.utils.config import load_config


def _format_lap_time(seconds: float) -> str:
    """Format ``seconds`` as ``m:ss.mmm`` (e.g. ``1:27.503``)."""
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes}:{rem:06.3f}"


def create_app(cfg: DictConfig | None = None) -> FastAPI:
    """Build the FastAPI app with shared track/physics objects from ``cfg``."""
    if cfg is None:
        cfg = load_config("default")

    track = load_track(cfg.track_id, cfg.track)
    params = KinematicParams.from_config(cfg.physics)
    physics = KinematicBicycle(params)
    sim_cfg = SimConfig.from_config(cfg.sim)
    pole_time_s = float(cfg.track.pole_time_s)
    total_laps = int(cfg.track.total_laps)
    recordings_dir = Path(cfg.server.recordings_dir)

    app = FastAPI(title="f1rl backend")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/track/{track_id}")
    def get_track(track_id: str) -> dict[str, Any]:
        if track_id != track.name:
            raise HTTPException(status_code=404, detail=f"unknown track '{track_id}'")
        return track.to_api_dict()

    @app.get("/api/meta")
    def get_meta() -> dict[str, Any]:
        return {
            "track_id": track.name,
            "control_hz": sim_cfg.control_hz,
            "pole_time_s": pole_time_s,
            "total_laps": total_laps,
            "pole_str": _format_lap_time(pole_time_s),
        }

    @app.get("/recordings")
    def list_recordings() -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        if recordings_dir.is_dir():
            for p in sorted(recordings_dir.glob("*.json")):
                try:
                    data = load_trajectory(p)
                except TrajectoryError:
                    continue
                created = (
                    data.get("meta", {}).get("created")
                    or datetime.fromtimestamp(p.stat().st_mtime, tz=UTC).isoformat()
                )
                items.append(
                    {"id": p.stem, "created": created, "frames": len(data.get("frames", []))}
                )
        return {"recordings": items}

    @app.get("/recordings/{rec_id}")
    def get_recording(rec_id: str) -> dict[str, Any]:
        path = recordings_dir / f"{rec_id}.json"
        try:
            return load_trajectory(path)
        except TrajectoryError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.websocket("/ws/sim")
    async def ws_sim(ws: WebSocket) -> None:
        await ws.accept()
        loop = SimLoop(physics, track, sim_cfg, pole_time_s, total_laps)
        autopilot = CenterlineAutopilot(track, params.max_steer)
        session = _Session()

        async def send_loop() -> None:
            dt = sim_cfg.dt_control
            while True:
                await asyncio.sleep(dt)
                if not session.running or session.mode not in {"manual", "watch"}:
                    continue
                frame: dict[str, Any] | None = None
                for _ in range(session.speed):
                    if session.mode == "manual":
                        steer = session.latest_input.steer
                        longitudinal = session.latest_input.longitudinal
                    else:
                        steer, longitudinal = autopilot.control(loop.state)
                    frame = loop.step(steer, longitudinal)
                    if session.recorder is not None:
                        session.recorder.append(frame["t"], frame["car"], frame["telemetry"])
                if frame is not None:
                    await ws.send_json(frame)

        async def recv_loop() -> None:
            while True:
                data = await ws.receive_json()
                msg = parse_client_message(data)
                if msg is None:
                    continue
                if isinstance(msg, InputMessage):
                    session.latest_input = msg
                    if msg.reset:
                        loop.reset()
                elif isinstance(msg, ModeMessage):
                    session.mode = msg.mode
                elif isinstance(msg, ControlMessage):
                    if msg.action == "play":
                        session.running = True
                    elif msg.action == "pause":
                        session.running = False
                    elif msg.action == "restart":
                        loop.reset()
                    if msg.speed is not None:
                        session.speed = msg.speed
                elif isinstance(msg, RecordMessage):
                    if msg.action == "start":
                        session.recorder = TrajectoryRecorder(
                            track.name, sim_cfg.dt_control, int(cfg.seed)
                        )
                    elif msg.action == "stop" and session.recorder is not None:
                        if len(session.recorder) > 0:
                            stem = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                            session.recorder.save(recordings_dir / f"{stem}.json")
                            await ws.send_json(
                                {"type": "event", "event": "recording_saved", "id": stem}
                            )
                        session.recorder = None

        send_task = asyncio.ensure_future(send_loop())
        recv_task = asyncio.ensure_future(recv_loop())
        try:
            await asyncio.gather(send_task, recv_task)
        except WebSocketDisconnect:
            pass
        finally:
            send_task.cancel()
            recv_task.cancel()

    return app


class _Session:
    """Per-connection mutable session state for the live sim socket."""

    def __init__(self) -> None:
        self.mode: str = "manual"
        self.running: bool = True
        self.speed: int = 1
        self.latest_input: InputMessage = InputMessage()
        self.recorder: TrajectoryRecorder | None = None


app = create_app()
