# Technical Design

This file is the authoritative engineering specification for the project. Every major decision is made here, with a reason. Read `PROJECT_VISION.md` first for the goal, then follow this document for the build.

---

## 0. How to use this document (coding agent, read first)

- Treat this file as the source of truth for engineering decisions.
- Build in the phase order in section 15. Each phase ends with a working, watchable artifact.
- Follow the conventions in section 14 on every file you write.
- Do not swap a chosen library, change the units, or restructure a contract without updating this document first. The decision and the doc move together.
- When a decision needs tuning later (reward weights, grip values, physics constants), the value lives in a config file, never hardcoded in logic.

---

## 1. Scope and non-goals

In scope: a 2D top-down Formula 1 simulator where every car is driven by a learned policy, racing real 2026 circuits, with staged realism (tires, weather, surface, off-track penalties, pit stops).

Non-goals, fixed:

- No 3D rendering and no game engine (no Unity, no Godot).
- No external physics engine for the core car model. The physics is custom (reason in section 5).
- No live frame drawing inside the training loop, for speed. The interactive app still shows everything through manual driving, live runs of a trained agent, and recorded replays.
- No reward for staying near the centerline. The racing line emerges from rewarding speed and progress.
- No network calls inside the training loop. Track data is built offline and cached.
- Elevation is optional, default off, lowest priority (invisible in a top-down view).
- Pit-stop decisions start as a scripted rule, not a learned action.

---

## 2. Technology stack (decisions)

| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.10+ | Standard for RL, matches the Colab and Kaggle runtimes. |
| Math and physics | NumPy | Fast array math for state, vectors, and the track. |
| Single-agent env API | Gymnasium | Current standard, replaces the old Gym, works with the algorithm library. |
| Multi-agent env API | PettingZoo (ParallelEnv) | Standard multi-agent API, all cars step at once. |
| Multi-agent wrappers | SuperSuit | Vectorizes and adapts the PettingZoo env for training. |
| Algorithm | PPO via Stable-Baselines3 | Stable on-policy control, scales with parallel envs, fastest route to a working agent. |
| Deep learning backend | PyTorch | Backend for Stable-Baselines3. |
| Track data source | FastF1 | Real circuit position telemetry, used offline to build track geometry. |
| Curve fitting | SciPy (splprep, splev) | Resample and smooth the centerline at uniform spacing. |
| Interactive app frontend | Vite + TypeScript, HTML5 Canvas 2D | Recreates the approved broadcast-grade design as a web app; the Canvas viewport draws the top-down scene from Python track geometry in meters. No framework. |
| Interactive app backend | FastAPI on uvicorn (WebSocket) | A local server streams car state at the control rate over a WebSocket and accepts manual-drive input; serves track geometry and recorded trajectories over HTTP. |
| Offscreen eval-clip rendering | Pygame | Fast 2D drawing, runs headless on the cloud through a dummy video driver. Used only for the training eval clips, never for the interactive app. |
| Video encoding | imageio with imageio-ffmpeg | Pip-installable, compiles frames to mp4 with no system ffmpeg needed. |
| Experiment tracking | Weights and Biases | Logs to the cloud, so a session disconnect never loses your curves. |
| Config | OmegaConf with YAML files | Typed config plus command-line overrides, no heavy app framework. |
| Lint and format | Ruff | One fast tool for style and linting. |
| Tests | Pytest | Small targeted tests, including the env API check. |

Rejected alternatives, on purpose: SAC was passed over for PPO because PPO is simpler to tune and pairs better with many parallel envs. A physics engine (Box2D, Pymunk) was passed over because a closed engine blocks the grip, tire, weather, and surface modifiers the project needs. JAX was passed over for now and held as a later optimization (section 17).

---

## 3. Units and coordinate system

- All internal values use SI units: meters, meters per second, radians, seconds, newtons, kilograms.
- World frame is right-handed, x to the right, y up. The renderer flips y for the screen.
- Headings are radians in the standard math sense.
- Track coordinates from FastF1 arrive in meters. Recenter so the track centroid sits near the origin. Do not rescale them.
- Simulation runs at a fixed control step of 20 Hz, so the action timestep is `dt_control = 0.05` seconds.
- Physics integrates with 5 substeps per control step at `dt_physics = 0.01` seconds, using semi-implicit Euler. Substeps keep the dynamic model stable at high speed.
- The renderer scales meters to pixels with a single pixels-per-meter factor chosen to fit the track bounding box, with a margin. Unit conversion happens only at two boundaries: config input and rendering output.

This single rule, simulate in real meters, is what makes the car and the track stay in true proportion with no manual scaling, which is the realism the vision asks for.

---

## 4. The grip pipeline (central abstraction)

Grip is one scalar that gates how much force a tire produces. Every realism feature is a multiplier on grip. Build this from the first physics commit and set the multipliers to constants until each feature lands.

```python
def effective_grip(mu_base, compound, wear, weather, surface_zone):
    return (mu_base
            * tire_factor(compound, wear)    # soft > medium > hard, falls as wear rises
            * weather_factor(weather)         # dry 1.0, damp ~0.8, wet ~0.6
            * surface_factor(surface_zone))   # asphalt 1.0, kerb ~0.9, grass ~0.4, gravel ~0.3
```

The friction circle uses this grip to cap the combined longitudinal and lateral tire force:

```
max_force = grip * mass * g          # base limit
# optional aero term added later:
max_force += 0.5 * rho * Cl * area * speed**2
```

Tires, weather, and surface all reduce to changing one number. Adding a feature means writing one factor function, not touching the physics core.

---

## 5. Physics model

Decision: write the car model. A 2D dynamic bicycle model with a friction-circle force limit gives the most realism while staying fast and fully controllable. A closed engine cannot expose the grip pipeline above.

Build in two steps behind one interface:

1. Kinematic bicycle model first. No tire slip, turning radius from wheelbase, steering, and speed. Use this to get the environment, observations, rewards, and the first learned agent working.
2. Dynamic bicycle model second. Lateral velocity, yaw rate, slip angles, tire forces clamped to the friction circle. Swap this in behind the same interface with no env changes. Start the tire force as a linear model. Upgrade to a simplified Pacejka curve only if the feel needs more.

Car state and the physics interface:

```python
@dataclass
class CarState:
    x: float          # meters, world frame
    y: float          # meters, world frame
    yaw: float        # radians
    vx: float         # m/s, longitudinal in body frame
    vy: float         # m/s, lateral in body frame
    yaw_rate: float   # rad/s
    tire_wear: float  # 0..1
    compound: int     # 0 soft, 1 medium, 2 hard, 3 intermediate, 4 wet

class PhysicsModel(Protocol):
    def step(self, state: CarState, steer: float, longitudinal: float,
             grip: float, dt: float) -> CarState:
        ...
```

The env computes `grip` from the grip pipeline and passes a single scalar in. The physics step is a pure function of state, controls, grip, and dt. No globals, no rendering, no track lookups inside the step.

Physics parameters live in config with realistic F1-scale defaults, all tunable: wheelbase about 3.6 m, mass about 770 to 800 kg, max steer about 18 degrees at the wheels, plus engine force, brake force, drag, rolling resistance, and an optional downforce coefficient. None of these are load-bearing constants. Tune them in config.

---

## 6. Track representation and build pipeline

The simulator consumes a processed `Track` object. FastF1 is a build-time dependency only. Training loads cached files and never calls the network.

```python
@dataclass
class Track:
    name: str
    centerline: np.ndarray        # (N, 2) meters
    tangent: np.ndarray           # (N, 2) unit vectors
    normal: np.ndarray            # (N, 2) unit vectors, point left
    s: np.ndarray                 # (N,) cumulative arc length, meters
    curvature: np.ndarray         # (N,) signed, 1/m
    half_width_left: np.ndarray   # (N,) meters to the asphalt edge
    half_width_right: np.ndarray  # (N,)
    runoff_width: np.ndarray      # (N,) meters of grass or gravel beyond the edge
    gradient: np.ndarray          # (N,) slope, default zeros
    closed: bool                  # the lap loops
```

Offline build pipeline, run once per circuit, output cached to `data/tracks/<name>.npz`:

1. Load a clean fast lap with FastF1 and read the X and Y position trace.
2. Resample to uniform arc-length spacing of about 3 m with a SciPy spline.
3. Smooth to remove telemetry noise.
4. Assign track width. Default to a constant total width near 12 m, set per circuit in a track config where known.
5. Compute tangent, normal, arc length, and signed curvature along the centerline.
6. Define surface bands: asphalt inside the half-widths, a thin kerb band at the edge, then a runoff band of grass or gravel.
7. Save the processed `Track`. The build script does every circuit in one pass.

Data caveats, documented and accepted: the centerline is real and accurate. Width, kerbs, and runoff are approximations, set in config, which is fine for the racing behavior the project wants. Gradient defaults to zero.

One real gap: the new Madrid circuit has no historical telemetry, since 2026 is its first season. Build that track from a manual outline or a public map trace into the same `Track` schema. Flag any circuit that lacks telemetry rather than guessing.

A track configuration UI lives in the interactive app (section 11). Load a circuit, see the centerline and boundaries, and set the asphalt width, the kerb bands, the grass, and the sand or gravel zones, plus the dry or wet condition, then save the result back to the cached `Track`. The build pipeline produces the geometry. The UI shapes the surfaces and conditions by hand.

---

## 7. Observation space

Decision: the policy sees local, relative features only, never absolute position. A policy trained on local features drives any circuit, because the inputs look the same everywhere. This is what lets one policy generalize across the whole calendar.

Version 1, single car, fixed length about 15:

- Longitudinal speed, normalized by a reference speed.
- Heading error between the car heading and the track tangent at the nearest centerline point.
- Signed lateral offset from the centerline, divided by the half-width. This is input only. It is never rewarded toward zero.
- Track curvature sampled at 5 lookahead distances ahead along the centerline, for example 10, 25, 50, 100, and 150 m. This is how the car sees the upcoming corners.
- 7 rangefinder beams cast from the car to the asphalt edge at fixed angles, for example minus 90 to plus 90 degrees, each giving normalized distance to the edge.

Extensions added in later phases, which change the vector length and require retraining at that point:

- Tire wear and a compound indicator.
- A grip or weather indicator.
- The relative position and velocity of the K nearest cars, expressed in the car body frame.

The observation version is fixed per phase. When the vector changes, retrain. This is expected and clean.

---

## 8. Action space

Continuous, two values, each in the range minus 1 to plus 1:

```
action[0] -> steering   : maps to [-max_steer, +max_steer] radians
action[1] -> longitudinal: >= 0 throttle scaled by max engine force
                           <  0 brake scaled by max brake force
```

A later pit decision is a separate discrete output, added in the pit phase. The main driving action stays two continuous values.

---

## 9. Reward design

Principle, restated as a rule: reward forward progress and speed, penalize leaving the track and contact, never reward centerline proximity. The fast line then earns the most reward, so late apexes and full track width appear on their own.

Version 1 per step:

```
ds         = progress along centerline arc length since the last step (meters)
off        = meters past the asphalt edge, 0 while on asphalt
reward = w_progress * ds
       - w_offtrack * offtrack_penalty(off)     # graded, see below
       - w_step                                  # small constant, discourages dawdling
       - w_reverse * max(0, -ds)                 # penalize going backward
```

Graded off-track penalty, which gives the behavior the vision asks for:

- On asphalt: zero penalty.
- Slightly past the edge, within the runoff band: a small penalty that grows with distance off. Grass and gravel also cut grip and add drag, so a small excursion already loses time on its own.
- Far past the runoff, or stopped off track, or wrong way for too long: large penalty and episode termination.

Termination: success on completing the target lap count, failure on a large off-track or wrong-way event, truncation on a step limit. Contact penalty enters in the multi-agent phase.

Reward shaping is the single most iterated part of this project. Keep every weight in config and expect to tune it more than any other piece.

Lap-time benchmark. The simulation runs in real meters with real-scale physics, so a lap time comes out in real seconds and compares directly to the real pole lap for the circuit. Use lap time as the evaluation scoreboard, not the training signal. The dense progress reward drives learning. The aspiration is the real pole time. The first practical milestone is twice the pole time while the agent is still learning, tightened over training. How close to the pole is reachable depends on the car physics you tune, since the car has a top speed and a grip ceiling you set, so calibrate the car so a clean optimal lap lands near the pole and the comparison stays fair. Log lap time against the pole and against twice the pole for every circuit.

---

## 10. Environment contract

Single agent:

- `RacingEnv(gymnasium.Env)` with `reset(seed)` and `step(action)` returning `(obs, reward, terminated, truncated, info)`.
- Config-driven: track, physics model, conditions, reward weights, target laps.
- Owns the car state, the physics stepper, the track, the conditions, the reward, the termination logic, and a trajectory recorder.
- Must pass `gymnasium.utils.env_checker.check_env`. A test enforces this.
- `step` never renders. The recorder logs per-step state for offline video.

Multi agent, racing phase:

- `RacingParallelEnv(pettingzoo.ParallelEnv)`. All cars step at once. Cars are homogeneous, same observation and action structure.
- Training uses one shared policy with parameter sharing and self-play. Every car runs the same weights and learns against the current policy. This is the tractable path to many cars and avoids heavy multi-agent machinery.
- Scale the field gradually: 2 cars, then 4, then the full 22.

Shared rules:

- Parallel env copies run through Stable-Baselines3 vectorized envs for PPO throughput. The cloud gives few CPU cores, which limits this and motivates the JAX option in section 17.
- Determinism: one seeding utility seeds Python, NumPy, and PyTorch. Record the seed with every run.

---

## 11. Rendering pipeline

The interactive application is the primary surface of the project and the first thing built (section 15, phase 1). You see every change here.

- A web app: a Vite + TypeScript frontend with an HTML5 Canvas 2D viewport, talking to a local FastAPI/uvicorn backend over a WebSocket. The backend runs the simulation loop and streams car state at the control rate (20 Hz); the frontend draws at its own frame rate, interpolating between the last two streamed states. The viewport has a camera you pan and zoom, plus a HUD (top bar, timing tower, telemetry bar) recreating the approved broadcast-style design.
- Four modes, unchanged in spirit: manual drive with keyboard control (the browser sends input messages over the socket), configure to set track surfaces and conditions and save them to the `Track`, watch live to run a policy in real time, and replay to play back a recorded run.
- Watch live works for any policy, trained or untrained, because running a policy forward is cheap. This is how you see a car drive without waiting on training.
- Drawing: the Canvas draws the asphalt ribbon between the boundaries, the kerb bands, the grass or gravel background, and each car as an oriented shape colored by team, all from Python track geometry in meters through a meters→pixels camera transform. A debug overlay draws the centerline and the rangefinder beams.

The heavy training loop draws nothing, for speed, and the web frontend is never in that hot path. Two paths keep full visibility. The evaluation callback runs offscreen on the cloud, with the SDL video driver set to dummy, renders one episode to an mp4 through Pygame and imageio, and logs the clip — this headless eval-clip path is unchanged by the web pivot. After training, the app loads a checkpoint and runs the agent live. The app stays your view into everything. Only the long training loop runs unseen.

The recorded-trajectory JSON format (section 10's recorder output) is the shared interchange between the live sim, the replay viewer, and the cloud eval-clip renderer: all three consume the same frame stream, so a run recorded in any mode replays identically anywhere.

---

## 12. Training, checkpointing, and logging

- `train.py` runs from a config, logs to Weights and Biases, and saves checkpoints on a schedule.
- A checkpoint holds the model weights, the optimizer state, the observation normalization stats, the total timestep count, the config, and the RNG states. Saving and resuming must round-trip exactly.
- `--resume <path>` continues a run after a disconnect. Assume the session can die at any moment and checkpoint frequently.
- Checkpoints write to Google Drive on Colab or to notebook output on Kaggle.
- An evaluation callback periodically runs one deterministic episode, records it, renders a short clip, and logs the clip and the metrics to Weights and Biases.
- Logged metrics: episode return, lap time against the pole and twice the pole, off-track count, contact count in the racing phase, and the learning curves.

---

## 13. Repository structure

```
f1-rl/
  README.md
  PROJECT_VISION.md
  TECHNICAL_DESIGN.md
  pyproject.toml                # dependencies and tooling
  configs/
    default.yaml             # seed, sim, physics, server, and track_id (selects the track file)
    track/<circuit>.yaml     # per-circuit geometry, pole time, lap count; merged under cfg.track
    experiment/<name>.yaml   # per-run overrides for the trainer; arrives in Phase 3 with train.py
  data/
    raw_telemetry/              # FastF1 cache, gitignored
    tracks/                     # processed Track files, .npz
  recordings/                   # recorded-trajectory JSON, gitignored
  web/                          # interactive app frontend (Vite + TypeScript)
    index.html                  # the broadcast-style stage shell
    src/
      main.ts                   # bootstrap, stage scaling, UI state machine
      tokens.css                # design tokens + fonts
      net/socket.ts             # WebSocket client to /ws/sim
      input/keyboard.ts         # arrows + WASD -> input messages
      viewport/camera.ts        # meters<->pixels transform, pan/zoom/follow
      viewport/renderer.ts      # Canvas 2D draw, interpolating 20 Hz states
      hud/telemetry.ts          # tower + telemetry bar from state frames
      replay/player.ts          # load + play/pause/scrub a trajectory
  src/f1rl/
    __init__.py
    physics/
      base.py                   # CarState, PhysicsModel interface
      kinematic.py
      dynamic.py
      tires.py                  # grip pipeline and tire model
    track/
      schema.py                 # Track dataclass
      oval.py                   # procedural oval Track (Phase 1)
      build.py                  # FastF1 to processed Track, offline
      loader.py                 # load cached Track
    sim/
      loop.py                   # fixed-step simulation loop
      timing.py                 # lap timing and delta-to-pole
      recorder.py               # trajectory recording (JSON)
      autopilot.py              # centerline pure-pursuit for watch-live
    server/
      app.py                    # FastAPI app: WS /ws/sim, GET /track, GET /recordings
      messages.py               # Pydantic client/server message models
    env/
      single_agent.py           # RacingEnv
      multi_agent.py            # RacingParallelEnv
      observations.py
      rewards.py
      conditions.py             # weather and surface state
    render/
      renderer.py               # offscreen frames to mp4 for eval clips (training only)
    train/
      train.py
      evaluate.py
      callbacks.py              # checkpoint, eval video, wandb
      selfplay.py               # shared-policy self-play
    utils/
      seeding.py
      config.py                 # OmegaConf load
  scripts/
    build_all_tracks.py
    render_episode.py
  notebooks/
    colab_train.ipynb           # clone repo, run train.py on GPU
  tests/
    test_env_api.py             # gymnasium env_checker
    test_physics.py
    test_track_build.py
```

The interactive app is split across the top-level `web/` frontend and two Python packages: `server/` (the FastAPI app and its message models) and `sim/` (the fixed-step loop, lap timing, the trajectory recorder, and the centerline autopilot for watch-live). The `recorder.py` lives under `sim/` because the recorded-trajectory JSON it produces is the shared interchange across live sim, replay, and eval clips. `render/renderer.py` stays under the package as the offscreen eval-clip renderer used by training only; it is never imported by the app or the training hot path.

---

## 14. Coding conventions and rules

- Config-driven. No tuning constant lives in logic. Reward weights, grip values, and physics parameters sit in config.
- The env must pass the Gymnasium env checker. A test enforces it.
- One seeding utility seeds everything. Every run is reproducible from a config and a seed.
- Physics stays behind the `PhysicsModel` interface. The env never depends on a specific model.
- The renderer is never imported in the training hot path. Rendering reads recorded trajectories.
- FastF1 runs only at build time. Training loads cached tracks.
- The physics step is a pure function of state, controls, grip, and dt.
- SI units everywhere internally. Convert only at the config-input boundary and the rendering output boundary.
- Type hints and short docstrings on public functions. Small, testable modules.
- Ruff for lint and format. Pytest for tests. Both wired into `pyproject.toml`.

---

## 15. Build order (phases, each ends with a working artifact you watch in the app)

This order is visibility-first. You see every change in the application as you make it. Earlier phases are dependencies for later ones.

Phase 1, the application and viewer. Build the interactive 2D top-down app as a web app: a Vite + TypeScript frontend with an HTML5 Canvas 2D viewport in the approved broadcast-style UI, talking to a local FastAPI/uvicorn backend over a WebSocket. The Canvas draws the track from Python track geometry in meters through a meters→pixels camera you pan and zoom, with cars as real-proportion shapes and a HUD (top bar, timing tower, telemetry bar). Add a manual drive mode where the backend runs the sim loop and the browser sends keyboard input over the socket, to feel the physics by hand. Add a replay mode that plays a recorded-trajectory run. Use a procedurally generated oval track and the kinematic physics for now. This app is the surface for every later feature. Artifact: drive a car around an oval in the app and replay a saved run.

Phase 2, tracks and the track configuration UI. Build the offline FastF1 pipeline and bring in every real 2026 circuit as a cached `Track`. Build the configuration screen inside the app: load a circuit, see the centerline and boundaries, and set the asphalt width, the kerbs, the grass, and the sand or gravel, plus dry or wet, then save back to the `Track`. Build the new Madrid circuit from a manual outline, since it has no telemetry. Artifact: every circuit loads in the app and you shape its surfaces by hand.

Phase 3, one car on one circuit. Drop an AI car on a configured circuit. At first it runs on an untrained policy and drifts, spins, and loses the track while you watch in the app. Then add observations v1, rewards v1, PPO training with checkpoint and resume, and Weights and Biases. Pull checkpoints and run the car live in the app to watch it improve. Once the first clean laps appear, upgrade to the dynamic physics with the grip pipeline, tires, and weather, swapped in behind the interface, then retrain. Benchmark the lap time against the pole. Artifact: a trained agent lapping one real circuit with realistic physics and a lap-time score against the real pole.

Phase 4, one car on many circuits. Confirm the observations are fully track-agnostic, then sample a different circuit each episode so one policy handles every track. Artifact: one policy driving every circuit, with a lap-time table against the pole per circuit.

Phase 5, many cars on track. Add the multi-agent env on PettingZoo with shared-policy self-play, and put the full field on track with no racing rules yet. Scale from 2 cars to 4 to the full 22. Artifact: a grid lapping a circuit together in the app.

Phase 6, racing for real. Add the nearby-car observations, collision detection, the contact penalty, and rewards for overtaking and defending. The accidents, the blocking, and the wheel-to-wheel racing emerge here. This is the hardest phase. Budget the most time. Artifact: a full grid racing with overtaking and defending.

Phase 7, pit stops and polish (optional capstone). Add a pit lane with a speed limit and a stop time cost, then a scripted pit rule that pits when tire wear crosses a threshold. Make the pit decision a learned action only as a stretch goal. Add team colors and a richer HUD, record the showcase videos, and write the README. Artifact: a portfolio-ready race with strategy.

---

## 16. Local and cloud workflow

Claude Code runs on the laptop and edits the codebase. Training runs on Colab or Kaggle for the GPU. A Git repo on GitHub connects the two.

- Develop and edit locally with Claude Code, then push to GitHub.
- The notebook clones or pulls the repo, installs dependencies, and runs `train.py` on the GPU.
- Checkpoints and videos return through Drive or notebook output.
- The whole simulator, the track build, the renderer, and small smoke runs all work on the laptop CPU with no GPU. Only the long training jobs go to the cloud.

---

## 17. Deferred optimizations

Held back on purpose. Do not start these until the matching problem appears.

- JAX environment on the GPU. If training is too slow because the cloud gives few CPU cores, rewrite the simulator in JAX so thousands of races run on the GPU at once. This is the real GPU lever for this project.
- SAC instead of PPO. Consider only if sample efficiency is the bottleneck and parallel envs stay limited.
- Pacejka tire model. Upgrade from the linear tire model only if the feel needs more.
- Elevation. Add a simple gradient effect on speed only if you want the physics. It stays invisible in a top-down view, so it is the lowest priority.