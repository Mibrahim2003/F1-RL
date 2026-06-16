# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A 2D top-down Formula 1 simulator where every car is driven by a learned (PPO) policy, racing real 2026 circuits. Currently **early stage**: the design is complete and the package skeleton exists, but the simulation/RL code is not yet written.

## Source of truth — read these first

Two documents in `.claude/` govern every engineering decision. Read them before writing code:

- **`.claude/PROJECT_VISION.md`** — the goal and the feel to aim for.
- **`.claude/TECHNICAL_DESIGN.md`** — the **authoritative** engineering spec: stack, units, physics, env contract, reward design, repo layout (section 13), conventions (section 14), and the phased build order (section 15).

If a change contradicts `TECHNICAL_DESIGN.md`, update the doc in the same change — the decision and the doc move together. Do not silently swap a library, change units, or restructure a contract.

## Commands

The project uses Python 3.12 in a local venv (`.venv/`). Use the venv interpreter directly; activation is optional.

```bash
# Run anything in the env
.venv/Scripts/python.exe <script.py>

# Install / reinstall (editable). FastF1 lives in the 'trackbuild' extra (Phase 2+ only).
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pip install -e ".[dev,trackbuild]"

# Lint + format (Ruff: line-length 100, target py312)
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
```

## Architecture (the big picture)

Target layout is `src/f1rl/` with a strict separation of concerns (full tree in `TECHNICAL_DESIGN.md` §13). The design's load-bearing ideas, which span multiple modules:

- **Simulate in real SI meters, everywhere internally.** Convert units only at two boundaries: config input and rendering output. This is what keeps cars and tracks in true proportion with no manual scaling, and lets lap times compare directly to the real pole.
- **The grip pipeline is the central abstraction** (`physics/tires.py`). Grip is one scalar that gates tire force via a friction circle. Every realism feature (tires, weather, surface) is just a multiplier on grip — adding a feature means writing one factor function, not touching the physics core.
- **Physics is a swappable pure function behind one interface.** `PhysicsModel.step(state, steer, longitudinal, grip, dt) -> CarState` (`physics/base.py`). Kinematic bicycle model first, dynamic model swapped in later behind the same interface. No globals, no rendering, no track lookups inside the step. The env never depends on a concrete model.
- **The env owns everything and must pass the Gymnasium env checker.** `RacingEnv(gymnasium.Env)` (single agent) and `RacingParallelEnv(pettingzoo.ParallelEnv)` (multi-agent, shared-policy self-play). A test enforces `gymnasium.utils.env_checker.check_env`.
- **Observations are local/relative only — never absolute position.** This is what lets one policy generalize across the whole calendar.
- **Reward forward progress and speed; never reward centerline proximity.** The racing line (late apexes, full track width) must emerge from the reward, not be hand-fed. Lap time is the evaluation scoreboard, not the training signal.
- **Tracks are built offline and cached.** FastF1 is a **build-time-only** dependency that produces `data/tracks/<name>.npz`; it must never be imported in the training loop or called over the network during training.
- **The interactive surface is a web app.** A Vite + TypeScript frontend with an HTML5 Canvas 2D viewport talks to a local FastAPI/uvicorn backend over a WebSocket that streams car state at the control rate. The backend runs the sim loop (`src/f1rl/sim/`) and serves track geometry and recorded trajectories; the frontend (`web/`) only renders and sends input. Four modes: manual drive, configure, watch live, replay.
- **The web frontend is never in the training hot path.** Training draws nothing for speed, and the web app is never imported by it. Visibility comes from (a) an offscreen eval callback rendering one episode to mp4 (headless Pygame + imageio, unchanged by the web pivot), and (b) the interactive web app loading a checkpoint and running the agent live. The recorded-trajectory JSON format is the shared interchange across live sim, replay, and eval clips.

## Conventions that are easy to get wrong

- **Config-driven: no tuning constant lives in logic.** Reward weights, grip values, and physics parameters all live in config (OmegaConf + YAML under `configs/`). Reward shaping is the most-iterated part of the project — keep every weight in config.
- One seeding utility seeds Python, NumPy, and PyTorch together; record the seed with every run. Checkpoints must round-trip exactly (weights, optimizer, obs-normalization stats, timestep count, config, RNG states).
- Build in the **phase order** of §15 (visibility-first: app/viewer → tracks → one car/one circuit → many circuits → many cars → racing → pit stops). Each phase ends with a watchable artifact. The observation-vector length is fixed per phase; when it changes, retraining is expected.
- Type hints and short docstrings on public functions; small, testable modules.

## Repo / git notes

- The design docs intentionally live in `.claude/` (committed, and auto-loaded as context) rather than the repo root.
- `.claude/settings.local.json`, `.venv/`, the FastF1 cache (`data/raw_telemetry/`), and training artifacts (`wandb/`, checkpoints, `*.mp4`) are gitignored.
- Commit identity is set repo-locally to `Mibrahim2003`.
