"""FastAPI backend: the Python sim engine is the single source of truth.

The browser renders state and sends input over a WebSocket; nothing here imports a
renderer. Each ``/ws/sim`` connection owns its own :class:`~f1rl.sim.loop.SimLoop` and a
watch-mode driver (the :class:`~f1rl.sim.autopilot.CenterlineAutopilot` by default, or a
:class:`~f1rl.sim.policy_pilot.PolicyPilot` when a trained checkpoint is picked — same
``control(state)`` interface), advancing the sim on a fixed clock and streaming state frames.
A ``track`` message rebuilds those per-session so the user can switch circuits live (Phase 2);
a ``policy`` message swaps the driver. HTTP routes serve the circuit catalog, the checkpoint
catalog, static track geometry, run metadata, recorded trajectories, and the surface-editor
save endpoint.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import DictConfig

from f1rl.physics.kinematic import KinematicBicycle, KinematicParams
from f1rl.server.messages import (
    ControlMessage,
    InputMessage,
    ModeMessage,
    PolicyMessage,
    RecordMessage,
    SurfaceEdit,
    TrackMessage,
    parse_client_message,
)
from f1rl.sim.autopilot import CenterlineAutopilot
from f1rl.sim.loop import SimConfig, SimLoop
from f1rl.sim.policy_pilot import PolicyPilot
from f1rl.sim.recorder import TrajectoryError, TrajectoryRecorder, load_trajectory
from f1rl.track.loader import DEFAULT_TRACKS_DIR, list_tracks, load_track
from f1rl.track.schema import Track
from f1rl.utils.config import load_config, load_track_config

# Checkpoint sidecar files (mirrors f1rl.train.checkpointing — the format's single source).
_META_FILE = "meta.json"
_MODEL_FILE = "model.zip"

# Band arrays the surface editor can overwrite (each set uniformly to the edited scalar).
_BAND_FIELDS = (
    "half_width_left",
    "half_width_right",
    "kerb_width",
    "grass_width",
    "gravel_width",
)


def _format_lap_time(seconds: float) -> str:
    """Format ``seconds`` as ``m:ss.mmm`` (e.g. ``1:27.503``)."""
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes}:{rem:06.3f}"


def _is_checkpoint_dir(path: Path) -> bool:
    """A checkpoint directory holds both ``model.zip`` and ``meta.json``."""
    return (path / _MODEL_FILE).is_file() and (path / _META_FILE).is_file()


def scan_checkpoints(root: Path) -> list[dict[str, Any]]:
    """Find every checkpoint directory under ``root`` and read its meta sidecar.

    Walks ``root`` recursively (runs nest checkpoints under ``<run>/checkpoints/<name>``),
    returning one entry per dir holding ``model.zip`` + ``meta.json``. Each entry's ``id`` is
    the path relative to ``root`` (POSIX-style, so the client can echo it back unchanged). A
    malformed ``meta.json`` is skipped rather than breaking the whole listing.
    """
    if not root.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for meta_path in sorted(root.rglob(_META_FILE)):
        ckpt = meta_path.parent
        if not _is_checkpoint_dir(ckpt):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        items.append(
            {
                "id": ckpt.relative_to(root).as_posix(),
                "total_timesteps": int(meta.get("total_timesteps", 0)),
                "circuit_id": str(meta.get("circuit_id", "unknown")),
                "obs_version": int(meta.get("obs_version", 0)),
            }
        )
    # Surface least-trained first so an early/late comparison reads naturally in the dropdown.
    items.sort(key=lambda e: (e["circuit_id"], e["total_timesteps"]))
    return items


def _resolve_checkpoint(root: Path, ckpt_id: str) -> Path:
    """Resolve a client checkpoint ``id`` to a directory under ``root`` (no escaping ``root``).

    Raises:
        FileNotFoundError: If the id is empty, escapes ``root``, or is not a checkpoint dir.
    """
    if not ckpt_id:
        raise FileNotFoundError("empty checkpoint id")
    candidate = (root / ckpt_id).resolve()
    root_resolved = root.resolve()
    if root_resolved != candidate and root_resolved not in candidate.parents:
        raise FileNotFoundError(f"checkpoint id '{ckpt_id}' escapes the checkpoints root")
    if not _is_checkpoint_dir(candidate):
        raise FileNotFoundError(f"no checkpoint at '{ckpt_id}'")
    return candidate


def create_app(cfg: DictConfig | None = None) -> FastAPI:
    """Build the FastAPI app with shared physics + per-circuit track loading from ``cfg``."""
    if cfg is None:
        cfg = load_config("default")

    params = KinematicParams.from_config(cfg.physics)
    physics = KinematicBicycle(params)
    sim_cfg = SimConfig.from_config(cfg.sim)
    default_track_id = str(cfg.track_id)
    recordings_dir = Path(cfg.server.recordings_dir)
    tracks_dir = Path(cfg.server.get("tracks_dir", str(DEFAULT_TRACKS_DIR)))
    checkpoints_dir = Path(cfg.server.get("checkpoints_dir", "runs"))

    def track_meta(track_id: str) -> tuple[Track, float, int]:
        """Load a circuit + its pace meta. Raises ``FileNotFoundError`` if not built."""
        if track_id == "oval":
            tk = load_track("oval", cfg.track if default_track_id == "oval" else None)
            tcfg = cfg.track if default_track_id == "oval" else load_track_config("oval").track
        else:
            tk = load_track(track_id, tracks_dir=tracks_dir)
            tcfg = load_track_config(track_id).track
        pole = float(tcfg.get("pole_time_s", 60.0))
        laps = int(tcfg.get("total_laps", 1))
        return tk, pole, laps

    app = FastAPI(title="f1rl backend")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/tracks")
    def get_tracks() -> dict[str, Any]:
        return {"tracks": list_tracks(tracks_dir)}

    @app.get("/api/checkpoints")
    def get_checkpoints() -> dict[str, Any]:
        """List trained checkpoints under the configured root, for the watch-live picker."""
        return {"checkpoints": scan_checkpoints(checkpoints_dir)}

    @app.get("/track/{track_id}")
    def get_track(track_id: str) -> dict[str, Any]:
        try:
            tk, _, _ = track_meta(track_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return tk.to_api_dict()

    @app.get("/api/meta")
    def get_meta() -> dict[str, Any]:
        _, pole, laps = track_meta(default_track_id)
        return {
            "track_id": default_track_id,
            "control_hz": sim_cfg.control_hz,
            "pole_time_s": pole,
            "total_laps": laps,
            "pole_str": _format_lap_time(pole),
        }

    @app.post("/track/{track_id}/surfaces")
    def save_surfaces(track_id: str, edit: SurfaceEdit) -> dict[str, Any]:
        if track_id == "oval":
            raise HTTPException(status_code=400, detail="the procedural oval has no cached file")
        cache = tracks_dir / f"{track_id}.npz"
        if not cache.exists():
            raise HTTPException(status_code=404, detail=f"track '{track_id}' not built")
        track = Track.from_npz(cache)
        n = len(track.centerline)
        for field in _BAND_FIELDS:
            value = getattr(edit, field)
            if value is not None:
                setattr(track, field, np.full(n, float(value)))
        # Backup previous cache before overwriting, so a bad edit is reversible (rollback).
        shutil.copy2(cache, cache.with_suffix(".npz.bak"))
        track.save_npz(cache)
        return {"ok": True, "id": track_id, "condition": edit.condition}

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
        session = _Session()
        sim = _SimState()

        def set_track(track_id: str) -> None:
            tk, pole, laps = track_meta(track_id)
            sim.track = tk
            sim.loop = SimLoop(physics, tk, sim_cfg, pole, laps)
            # Switching circuits drops any active policy back to the centerline autopilot:
            # a checkpoint may have been trained on a different circuit (re-pick to re-apply).
            sim.autopilot = CenterlineAutopilot(tk, params.max_steer)
            sim.policy_id = None
            sim.track_id = track_id
            sim.pole_time_s = pole
            sim.total_laps = laps

        async def apply_policy(msg: PolicyMessage) -> None:
            """Swap the watch-mode driver: a trained checkpoint or the centerline autopilot.

            A bad/missing checkpoint or an obs-version mismatch never crashes the socket — it
            sends a ``policy_error`` event and leaves the centerline autopilot in place.
            """
            if msg.source == "autopilot":
                sim.autopilot = CenterlineAutopilot(sim.track, params.max_steer)
                sim.policy_id = None
                await ws.send_json(
                    {"type": "event", "event": "policy_changed", "source": "autopilot"}
                )
                return

            ckpt_id = msg.id or ""
            try:
                ckpt_path = _resolve_checkpoint(checkpoints_dir, ckpt_id)
                pilot = PolicyPilot(ckpt_path, sim.track, cfg)
            except Exception as exc:  # CheckpointError, FileNotFoundError, load failure
                # Never crash the socket: report the error and stay on the autopilot.
                sim.autopilot = CenterlineAutopilot(sim.track, params.max_steer)
                sim.policy_id = None
                await ws.send_json(
                    {
                        "type": "event",
                        "event": "policy_error",
                        "id": ckpt_id,
                        "message": str(exc),
                    }
                )
                return
            sim.autopilot = pilot
            sim.policy_id = ckpt_id
            await ws.send_json(
                {
                    "type": "event",
                    "event": "policy_changed",
                    "source": "checkpoint",
                    "id": ckpt_id,
                    "circuit_id": str(pilot.meta.get("circuit_id", "")),
                    "total_timesteps": int(pilot.meta.get("total_timesteps", 0)),
                }
            )

        set_track(default_track_id)

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
                        steer, longitudinal = sim.autopilot.control(sim.loop.state)
                    frame = sim.loop.step(steer, longitudinal)
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
                        sim.loop.reset()
                elif isinstance(msg, ModeMessage):
                    session.mode = msg.mode
                elif isinstance(msg, TrackMessage):
                    try:
                        set_track(msg.id)
                        await ws.send_json(
                            {
                                "type": "event",
                                "event": "track_changed",
                                "id": msg.id,
                                "control_hz": sim_cfg.control_hz,
                                "pole_time_s": sim.pole_time_s,
                                "total_laps": sim.total_laps,
                                "pole_str": _format_lap_time(sim.pole_time_s),
                            }
                        )
                    except FileNotFoundError:
                        await ws.send_json({"type": "event", "event": "track_error", "id": msg.id})
                elif isinstance(msg, PolicyMessage):
                    await apply_policy(msg)
                elif isinstance(msg, ControlMessage):
                    if msg.action == "play":
                        session.running = True
                    elif msg.action == "pause":
                        session.running = False
                    elif msg.action == "restart":
                        sim.loop.reset()
                    if msg.speed is not None:
                        session.speed = msg.speed
                elif isinstance(msg, RecordMessage):
                    if msg.action == "start":
                        session.recorder = TrajectoryRecorder(
                            sim.track_id, sim_cfg.dt_control, int(cfg.seed)
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


class _SimState:
    """Per-connection sim objects, rebuilt when the session switches circuits.

    ``autopilot`` holds whatever drives the car in watch mode — the centerline
    :class:`CenterlineAutopilot` by default, or a :class:`PolicyPilot` once a checkpoint is
    picked. Both expose the same ``control(state)`` signature, so ``send_loop`` is unchanged.
    """

    def __init__(self) -> None:
        self.track_id: str = ""
        self.track: Track | None = None
        self.loop: SimLoop = None  # type: ignore[assignment]
        self.autopilot: CenterlineAutopilot | PolicyPilot = None  # type: ignore[assignment]
        self.policy_id: str | None = None
        self.pole_time_s: float = 0.0
        self.total_laps: int = 1


app = create_app()
