# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A 2D top-down Formula 1 simulator where every car is driven by a learned (PPO) policy, racing real 2026 circuits. **Phases 1 through 4 are complete:** the interactive web app (FastAPI/WebSocket backend, fixed-step sim loop, lap timing, trajectory recorder, centerline autopilot, Vite/Canvas frontend with manual-drive/watch/replay), the offline FastF1+OSM track pipeline with real circuits cached under `data/tracks/`, the single-agent `RacingEnv` (ObservationV2, RewardV2), the swappable kinematic→dynamic physics with the grip pipeline (tires/weather/surface), and the PPO training core (checkpoint/resume, curriculum, W&B logging, lap-time benchmark vs the real pole). **Phase 4 (one policy across many circuits)** turns the env's single track into a config-driven **circuit pool** (`env/pool.py`) sampled per reset, widens it over the curriculum, warm-starts the Phase 3b checkpoint (obs unchanged → `OBS_VERSION` stays 2), and emits the lap-time-vs-pole calendar table (`train/calendar_benchmark.py`, served at `GET /api/calendar`, toggled with `T` in the watch view); it lives on branch `phase-4-many-circuits` (see `.claude/specs/phase-4-many-circuits.md`). **Phase 5 (many cars on track)** is built on branch `phase-5-many-cars` (see `.claude/specs/phase-5-many-cars.md` + its `-plan.md`): the multi-agent `RacingParallelEnv` (`env/multi_agent.py`) puts N homogeneous cars on one shared circuit (per-car `LapTimer`; the single-car step factored into `step_one_car`/`reset_car` + `CarRuntime` in `env/single_agent.py`), trains one shared policy through SuperSuit → SB3 PPO (`env/factory.make_selfplay_vec_env`, `train/selfplay.py`) warm-started from the Phase 4 generalist, and renders the whole field live (`cars[]` frame, `FieldSimLoop`, per-team colors, timing tower by track-position gap). The observation is **unchanged** (`OBS_VERSION` stays 2 — no nearby-car block, no collisions, no racing rewards); field size is a per-run constant grown across warm-started runs; the bar is infrastructure + the render, not a learning gain. **Phase 6 (racing for real) is built** on branch `phase-6-racing` (see `.claude/specs/phase-6-racing.md` + its `-plan.md`): the observation becomes **ObservationV3** (`OBS_VERSION 2→3`, length `22 + K*5`, the v2 prefix byte-identical + a K-nearest-cars neighbor block at the tail, local/relative only — `env/observations.py` `build_neighbor_block`); a **field-level two-disc collision pass** (`env/collisions.py` `resolve_collisions`, snapshot-then-apply, order-independent, `PhysicsModel.step` untouched); **`reward_v3`** = the v2 core − a graded contact penalty + a zero-sum overtake/defend term (`env/rewards.py`, every weight in config, reduces to `reward_v2` with no contact / constant rank); the multi-agent `step` reordered advance → collide → rank → finalize-with-block (`env/single_agent.py` `advance_car_physics`/`finalize_car_step`, `env/multi_agent.py` `rank_and_overtakes`); a **grown input-layer warm start** that transplants the Phase 5 v2 driver into the v3 policy (`train/warmstart.py` `grow_policy`, `selfplay.py --warm-start`; a v2 `--resume` is refused); a reward-weight **coexist→race curriculum**; racing metrics (`train/selfplay_eval.py`) + the race-aware live app (`race_position` + gap-to-ahead in the timing tower). The `collision:` config defaults `enabled: false`, so a pre-Phase-6 config keeps the Phase 5 parade. **Unlike Phase 5, the bar is a behavioral gain — the race has to look like a race.**

## Source of truth — read these first

Two documents in `.claude/` govern every engineering decision. Read them before writing code:

- **`.claude/PROJECT_VISION.md`** — the goal and the feel to aim for.
- **`.claude/TECHNICAL_DESIGN.md`** — the **authoritative** engineering spec: stack, units, physics, env contract, reward design, repo layout (section 13), conventions (section 14), and the phased build order (section 15).
- **`.claude/specs/`** — per-phase specs and implementation plans (e.g. `phase-2-realistic-tracks.md` + its `-plan.md`). The plan is the file-by-file build order for the phase in progress; read it alongside the design doc before touching that phase's code.

If a change contradicts `TECHNICAL_DESIGN.md`, update the doc in the same change — the decision and the doc move together. Do not silently swap a library, change units, or restructure a contract.

## Commands

The project uses Python 3.12 in a local venv (`.venv/`). Use the venv interpreter directly; activation is optional.

```bash
# Run anything in the env
.venv/Scripts/python.exe <script.py>

# Install / reinstall (editable). FastF1 lives in the 'trackbuild' extra (Phase 2+ only).
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pip install -e ".[dev,trackbuild]"

# Lint + format (Ruff: line-length 100, target py310 — matches the >=3.10 support floor)
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m ruff format .

# Tests (pytest; testpaths = tests/)
.venv/Scripts/python.exe -m pytest                       # all
.venv/Scripts/python.exe -m pytest tests/test_env_api.py # one file
.venv/Scripts/python.exe -m pytest -k env_checker        # one test by name
```

Note: this is a Windows machine; the shell is PowerShell but a Bash tool is available. Two Pythons are managed by the `py` install manager (3.14 default, 3.12 for this project) — the venv is pinned to 3.12, so `requires-python` is capped at `<3.13` to avoid missing-wheel problems (torch/SB3/fastf1 lag the newest Python).

The Phase 1 interactive app has two halves. Launch the Python backend, then the web frontend:

```bash
# Backend: FastAPI on uvicorn (needs an editable install or PYTHONPATH=src)
.venv/Scripts/python.exe -m uvicorn f1rl.server.app:app

# Frontend: Vite dev server (needs Node/npm)
cd web && npm install && npm run dev   # then open the printed localhost URL

# Frontend tests (Vitest + jsdom — headless DOM tests for the vanilla-TS UI modules)
cd web && npm test                                    # all UI tests (vitest run)
cd web && npx vitest run src/ui/config_panel.test.ts  # one file
# Frontend typecheck / build gate
cd web && npx tsc --noEmit             # typecheck only (test files live in src/, so they gate too)
cd web && npm run build                # tsc && vite build
```

Training, evaluation, and the offline pipelines are argparse modules run with `-m` (each takes `--config <experiment-name>` from `configs/experiment/` and trailing `key=value` dotlist overrides):

```bash
# Single-circuit PPO (Phase 3) / self-play field PPO (Phase 5); --resume warm-starts a checkpoint
.venv/Scripts/python.exe -m f1rl.train.train --config rbr_dynamic
.venv/Scripts/python.exe -m f1rl.train.selfplay --config calendar_selfplay --resume runs/<ckpt>

# Score a checkpoint vs the real pole on every pool circuit → out/calendar_benchmark.json (GET /api/calendar)
.venv/Scripts/python.exe -m f1rl.train.calendar_benchmark --checkpoint runs/<ckpt>
# Evaluate one checkpoint deterministically, optional mp4 of the first episode
.venv/Scripts/python.exe -m f1rl.train.evaluate --checkpoint runs/<ckpt> --video out/lap.mp4

# Build real circuits offline (needs the trackbuild extra + network on first run; FastF1/OSM)
.venv/Scripts/python.exe scripts/build_all_tracks.py            # all circuits
.venv/Scripts/python.exe scripts/build_all_tracks.py monza spa  # a subset
# Re-bake the web payloads from existing .npz (runtime-safe: no FastF1, no network)
.venv/Scripts/python.exe scripts/bake_track_json.py
```

## Architecture (the big picture)

Target layout is `src/f1rl/` with a strict separation of concerns (full tree in `TECHNICAL_DESIGN.md` §13). The design's load-bearing ideas, which span multiple modules:

- **Simulate in real SI meters, everywhere internally.** Convert units only at two boundaries: config input and rendering output. This is what keeps cars and tracks in true proportion with no manual scaling, and lets lap times compare directly to the real pole.
- **The grip pipeline is the central abstraction** (`physics/tires.py`). Grip is one scalar that gates tire force via a friction circle. Every realism feature (tires, weather, surface) is just a multiplier on grip — adding a feature means writing one factor function, not touching the physics core.
- **Physics is a swappable pure function behind one interface.** `PhysicsModel.step(state, steer, longitudinal, grip, dt) -> CarState` (`physics/base.py`). Kinematic bicycle model first, dynamic model swapped in later behind the same interface. No globals, no rendering, no track lookups inside the step. The env never depends on a concrete model.
- **The env owns everything and must pass the Gymnasium env checker.** `RacingEnv(gymnasium.Env)` (single agent) and `RacingParallelEnv(pettingzoo.ParallelEnv)` (multi-agent, shared-policy self-play). A test enforces `gymnasium.utils.env_checker.check_env`.
- **Observations are local/relative only — never absolute position.** This is what lets one policy generalize across the whole calendar.
- **Reward forward progress and speed; never reward centerline proximity.** The racing line (late apexes, full track width) must emerge from the reward, not be hand-fed. Lap time is the evaluation scoreboard, not the training signal.
- **Tracks are built offline and cached, including the form the web app consumes.** FastF1 is a **build-time-only** dependency that produces `data/tracks/<name>.npz`; it must never be imported in the training loop or called over the network during training. The build also bakes the web backend's served payloads next to each `.npz` — per-circuit `<id>.api.json` (the verbatim `GET /track/<id>` body) and `_catalog.json` (`GET /api/tracks`) — which the server reads as bytes verbatim instead of reloading numpy and re-serializing JSON per request. `save_track`, `build_all_tracks.py`, and the surface editor keep these in sync; `scripts/bake_track_json.py` regenerates them from existing `.npz` with no network. All three (`.npz`, `.api.json`, `_catalog.json`) are committed so circuits are preloaded on app open; the routes fall back to recomputing from the `.npz` if a payload is missing.
- **The interactive surface is a web app.** A Vite + TypeScript frontend with an HTML5 Canvas 2D viewport talks to a local FastAPI/uvicorn backend over a WebSocket that streams car state at the control rate. The backend runs the sim loop (`src/f1rl/sim/`) and serves track geometry and recorded trajectories; the frontend (`web/`) only renders and sends input. Four modes: manual drive, configure, watch live, replay.
- **The web frontend is never in the training hot path.** Training draws nothing for speed, and the web app is never imported by it. Visibility comes from (a) an offscreen eval callback rendering one episode to mp4 (headless Pygame + imageio, unchanged by the web pivot), and (b) the interactive web app loading a checkpoint and running the agent live. The recorded-trajectory JSON format is the shared interchange across live sim, replay, and eval clips.

## Web frontend (`web/`)

Vanilla TypeScript + Vite, no UI framework. `web/src/main.ts` is the wiring hub: it owns the module instances and a small observable `Store` (`state.ts`), subscribes the DOM to state changes, and routes UI/keyboard/socket events. The pieces:

- `net/socket.ts` (`SimSocket`) — typed `/ws/sim` client with backoff auto-reconnect. `send()` **silently drops** anything pushed before the socket is `OPEN`.
- `viewport/renderer.ts` (`Renderer` + `camera.ts`) — Canvas 2D; world-meter `Path2D` geometry built once per track, ~60 fps with interpolation between 20 Hz state frames. Renders a single car **or** a field — a frame with `cars[].length > 1` flips `fieldMode` and uses per-car interpolation buffers. The layout is fluid (the stage fills the window; no fixed 1920×1080 + transform scale).
- `ui/` — `TrackSelector`; `ConfigPanel` (the configure-mode **accordion**: surface sliders + dry/wet under ROAD CONDITION, the car glyph, the **driver picker**, and the **cars-on-track** field size — the driver/field controls were once a separate `PolicyPicker` overlay floating on the viewport, now folded in so they no longer block the track); `StartLights` (`lights.ts`, the F1 five-light start gantry); `CalendarPanel` (Phase 4 result table). `input/keyboard.ts` is manual-drive only (arrows/WASD), enabled solely in manual mode.
- `hud/`, `replay/` — telemetry HUD and the replay scrubber/player.

**The server starts every `/ws/sim` connection fresh** — default `manual` mode, single car (`_Session`/`_SimState` in `server/app.py`). The client is the source of truth for mode/field, so it resyncs the server **in the socket's `onOpen`** (`syncServer` in `main.ts`), never right after `connect()` — a message sent while the socket is still `CONNECTING` is dropped, and this race also bites after every auto-reconnect. Entering `configure` pauses the server (`sendControl("pause")`); returning to watch/manual must `sendControl("play")` or the field stays frozen while the client shows it running.

**The app boots into `configure` (paused)** — `Store`'s default mode, with `syncServer` sending `pause` on connect — so the user sets the race up before anything runs. The only seamless way out is the config panel's **SAVE & RACE** button (`startRace` in `main.ts`): it POSTs surfaces only if a slider/condition actually changed (`dirty` flag — avoids needless `.npz`/`.bak` writes), commits a typed-but-not-SET field size, runs the `StartLights` sequence, then `setMode("watch")` to play — the race begins exactly at lights-out. Driver and field-size changes apply **live over the socket** during configure (the server keeps the active policy + `n_agents` across a `mode` message), so they are not gated behind the save.

## Conventions that are easy to get wrong

- **Config-driven: no tuning constant lives in logic.** Reward weights, grip values, and physics parameters all live in config (OmegaConf + YAML under `configs/`). Reward shaping is the most-iterated part of the project — keep every weight in config.
- One seeding utility seeds Python, NumPy, and PyTorch together; record the seed with every run. Checkpoints must round-trip exactly (weights, optimizer, obs-normalization stats, timestep count, config, RNG states).
- Build in the **phase order** of §15 (visibility-first: app/viewer → tracks → one car/one circuit → many circuits → many cars → racing → pit stops). Each phase ends with a watchable artifact. The observation-vector length is fixed per phase; when it changes, retraining is expected.
- Type hints and short docstrings on public functions; small, testable modules.

## Repo / git notes

- The design docs intentionally live in `.claude/` (committed, and auto-loaded as context) rather than the repo root.
- `.claude/settings.local.json`, `.venv/`, the FastF1 cache (`data/raw_telemetry/`), and training artifacts (`wandb/`, checkpoints, `*.mp4`) are gitignored.
- Commit identity is set repo-locally to `Mibrahim2003`.
- **`.gitignore` venv patterns are anchored to the repo root (`/env/`, `/ENV/`, `/venv/`, `/.venv/`) on purpose.** An earlier unanchored `env/` matched the source package `src/f1rl/env/` case-insensitively on Windows and silently dropped the whole env package from every commit (it was missing from `main` until Phase 4 re-added it). Never reintroduce an unanchored `env/`/`ENV/` ignore, and after touching `.gitignore` confirm `git ls-files src/f1rl/env/` still lists the package.
