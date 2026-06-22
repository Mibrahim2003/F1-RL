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

Grip is one scalar that gates how much force a tire produces. Every realism feature is a multiplier on grip.

```python
def effective_grip(mu_base: float, compound: int, wear: float, weather: int, surface_zone: int) -> float:
    return (mu_base
            * tire_factor(compound, wear)     # falls as wear rises
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

**On `mu_base` (read this before trusting the lap-time benchmark).** `mu_base` is an **effective
base-grip coefficient**, not a literal tire–road friction coefficient. It lumps mechanical tire
grip together with a baseline aero/ground-effect contribution, and it is **calibrated per car**
(`f1rl.train.calibrate`) so that a clean optimal lap lands near the real pole — see §9. In
practice the calibrated value runs well above a physical road μ (the tuned `red_bull_ring` run
uses `mu_base ≈ 1.95`, and the easy-grip curriculum stage uses `≈ 2.3`), because in this
lumped-parameter model `mu_base` also stands in for grip that a higher-fidelity model would get
from downforce. The speed-dependent `downforce_coeff * v²` term in the friction circle adds on
top of it. This is a deliberate modeling choice: the lap-time-vs-pole comparison is fair
*because the car is calibrated to it*, not because `mu_base` is a measured friction value. Do
not describe `mu_base` as "dry-asphalt friction" in code or config comments.

---

## 5. Physics model

Decision: write the car model. A 2D dynamic bicycle model with a friction-circle force limit gives the most realism while staying fast and fully controllable. A closed engine cannot expose the grip pipeline above.

Build in two steps behind one interface:

1. Kinematic bicycle model first. No tire slip, turning radius from wheelbase, steering, and speed. Use this to get the environment, observations, rewards, and the first learned agent working.
2. Dynamic bicycle model second. Uses a semi-implicit Euler integration scheme with 5 substeps (`dt=0.01`).
   - Computes tire slip angles and linear cornering stiffness.
   - Restricts total tire force using a **Friction Circle**: `clamp_to_circle(fx, fy, max_grip_force)`.
   - Incorporates a low-speed kinematic fallback (`v_blend`, `v_eps`) to prevent singularity when `vx` approaches zero.

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

Calibration note: the grip level (`mu_base`), `downforce_coeff`, `max_engine_force`, and
`max_brake_force` are the levers `f1rl.train.calibrate` adjusts so a clean optimal lap lands
near the real pole. As covered in §4, the calibrated `mu_base` is a lumped effective-grip
coefficient and runs above a physical road μ — that is intentional, not a bug. If a future
upgrade wants literal-physics grip, lower `mu_base` toward ~1.5–1.8 and let `downforce_coeff`
carry high-speed cornering, then re-run calibration and retrain.

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
    kerb_width: np.ndarray        # (N,) red/white band past the asphalt edge
    grass_width: np.ndarray       # (N,) green band past the kerb
    gravel_width: np.ndarray      # (N,) sand/gravel band, where present
    gradient: np.ndarray          # (N,) slope, default zeros
    closed: bool                  # the lap loops
    country: str                  # for the selector + flag
    official_length_m: float      # published length, the scale check
    source: str                   # "fastf1" | "osm" | "fastf1+osm" | "manual" | "procedural"
    surface_zones: np.ndarray | None  # optional per-sample runoff label (0 grass, 1 gravel)
```

> Phase-2 change (this is the authoritative schema): the single Phase-1 `runoff_width`
> band is split into explicit `kerb_width` / `grass_width` / `gravel_width` bands (each a
> width measured outward past the asphalt edge), and the build metadata `country`,
> `official_length_m`, and `source` are added. `Track.length_error` and `Track.low_confidence`
> are derived. The cached form round-trips via `Track.save_npz` / `Track.from_npz`.

Offline build pipeline, run once per circuit, output cached to `data/tracks/<name>.npz`:

1. Acquire the shape from the best source: a clean FastF1 fast lap's X/Y position trace.
2. Acquire width from OpenStreetMap (Overpass) where mapped — Shapely offsets the asphalt
   edge polygons against the centerline — else fall back to a per-circuit config constant.
3. Resample to uniform arc-length spacing of about 2–3 m with a periodic SciPy spline, smooth.
4. Recenter to the centroid origin, in meters.
5. Compute tangent, normal, arc length, and signed curvature (shared `track/geometry.py`).
6. Define surface bands: asphalt inside the half-widths, a thin kerb band at the edge, then
   grass and (where `surface_zones` says) gravel runoff, all from config.
7. Validate: arc length within tolerance of `official_length_m`, Shapely `is_simple` edge
   check, positive bounded widths. Flag low-confidence, never crash.
8. Save the processed `Track` plus a row in `data/tracks/_build_report.json`. The build
   script does every circuit in one pass and isolates per-circuit failures.

Data caveats, documented and accepted: the centerline is real and accurate. Width, kerbs,
and runoff are approximations, from OSM where mapped and config otherwise, refined by hand in
the config UI — fine for the racing behavior the project wants. Gradient defaults to zero.

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

**ObservationV2 (Phase 3b Dynamic Model)**: Fixed length of 22 (OBS_VERSION = 2). This is the
**only** layout the code produces now; V1 (length 15) is historical — it is described above for
the record, but `observations.py` hard-codes `OBS_DIM = 22` and there is no V1 builder left.
Index ranges below use Python slice notation (half-open), matching `observations.py`:
- `[0:15]` The exact 15-length slice from V1 (speed, heading error, lateral offset, 5 curvature lookaheads, 7 rangefinders).
- `[15]` Tire wear (0.0 to 1.0).
- `[16:21]` Tire compound — **5-wide** one-hot (indices 16–20: Soft, Medium, Hard, Intermediate, Wet).
- `[21]` Grip/Weather indicator (e.g., 1.0 for dry, lower for wet).
- The relative position and velocity of the K nearest cars (added in later phases).

The observation version is fixed per phase. When the vector layout changes, the schema enforces retraining by refusing a checkpoint resume on a mismatched version (`validate_checkpoint`). Because only V2 exists in code, any pre-V2 checkpoint is intentionally unloadable.

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

Circuit pool, Phase 4 (one car on many circuits):

- The single bound track becomes a **config-driven circuit pool** (`env/pool.py` `CircuitPool`). A `circuits:` config block (`pool: [ids]`, `sampling: uniform|weighted`, `weights`, `pin_per_worker`) lists built `.npz` ids; an empty/absent pool falls back to the single `track_id`, reproducing the Phase 3b one-circuit behavior exactly.
- Each pool id is loaded and precomputed **once** per worker (`Track` + `EdgeCache` + `LapTimer` + resolved pole). `reset` draws one id from `self.np_random` (reproducible from the seed; each worker draws independently) and rebinds `self.track`/`self.edge_cache`/`self.lap_timer`/`self.track_id` + the active pole together — the per-reset cost is a lookup, never a rebuild. `reset`/`step` `info` carry `circuit_id` and `pole_time_s`. No `reset`/`step` signature change; **the observation is unchanged (ObservationV2, length 22, `OBS_VERSION = 2`)**, so the Phase 3b checkpoint warm-starts.
- **Per-circuit pole resolves from `configs/track/<id>.yaml` (`pole_time_s`)**, never from the geometry `.npz`; a missing/non-positive pole is flagged `pole_missing` and its delta is skipped (never divided by zero). The resolver is runtime-safe (YAML only, no FastF1, no network).
- `set_track_pool(circuits)` — env method mirroring `apply_conditions`: narrows/widens the **active** sampling set from the next `reset` (validates ids are in the built pool), never rebuilds. The curriculum calls it via `VecEnv.env_method` to widen the pool easy → full calendar; sampling-side only, no obs change, no mid-run retrain.

Multi agent — the field, Phase 5 (many cars, no racing rules yet):

- `RacingParallelEnv(pettingzoo.ParallelEnv)` (`env/multi_agent.py`). N homogeneous cars on **one shared circuit** step at once; `reset(seed)` returns `(obs, info)` **dicts keyed by agent id** (`car_0…car_{N-1}`) and `step(actions)` returns per-agent `(obs, rewards, terminations, truncations, infos)` dicts. `observation_space(agent)`/`action_space(agent)` are the **unchanged** ObservationV2 length-22 Box and 2-D action Box (every agent identical). It passes `pettingzoo.test.parallel_api_test`.
- **The observation is unchanged (`OBS_VERSION = 2`, no nearby-car block, no collisions, no racing rewards — those are Phase 6).** Each car sees only the track, so the Phase 4 generalist warm-starts directly as the shared brain.
- **The single-car update is factored once** into `step_one_car`/`reset_car` over a per-car `CarRuntime` (`env/single_agent.py`); `RacingEnv` is a thin one-car wrapper and `RacingParallelEnv` reuses the same unit per car. **What is per-car vs per-circuit is load-bearing:** per car = `CarState`, its **own `LapTimer` instance** (`LapTimer(track, pole)`, *never* the pooled `CircuitEntry.lap_timer`), and the projection state (`prev_s`/`grip_idx`/`grip_lat`/`wrong_way_count`); per circuit (read-only, shared by the field) = the pool entry's `Track`/`EdgeCache`/resolved pole. One car's lap never advances another's.
- **Constant SuperSuit-visible width via `black_death_v3`.** The raw env follows standard PettingZoo: a terminated/truncated car is removed from `self.agents` on the next step (so it passes `parallel_api_test`). The *constant* agent width the SuperSuit vectorizer requires comes from the `black_death_v3` **wrapper** (it re-pads removed agents to zero), not the raw env. One car finishing or failing never ends the others; the episode ends when `self.agents` is empty or the per-car step limit fires.
- **`grid:` config block** (field layout): `n_agents` (field size), `reset_mode` (`scattered` train = a distinct seeded centerline index per car; `grid` eval/demo = distinct non-overlapping two-column slots laid out forward from the S/F line), `grid_spacing_m`, `grid_lateral_m`, `team_colors` (render only, never observed). Both reset modes are seeded/reproducible. A non-positive `n_agents` and an unbuilt circuit id are refused.
- **One shared policy, parameter sharing**, vectorized PettingZoo → SuperSuit (`black_death_v3` → `pettingzoo_env_to_vec_env_v1` → `concat_vec_envs_v1`) → Stable-Baselines3 PPO (`env/factory.py` `make_selfplay_vec_env`, `train/selfplay.py`). Every car's transition trains the same weights; with no mutual observation this equals raising `n_envs` for *learning* (Phase 5 buys infrastructure + the render, not a learning gain).
- **Field size (`n_agents`) is a per-run constant, grown across warm-started runs (2 → 4 → 22)** — it sets the vector-env width (`n_agents * n_copies`), which the in-place curriculum (`VecEnv.env_method`, sampling-side, no rebuild) cannot change without breaking the PPO rollout buffer. The **circuit-pool widening stays an in-place curriculum knob** (Phase 4, unchanged). `n_agents` is **not** a `CurriculumStage` field. SuperSuit's `ConcatVecEnv` has no `env_method`, so the curriculum reaches the raw field envs via `factory.raw_parallel_envs` (in-process `num_cpus=0` only).

Shared rules:

- Parallel env copies run through Stable-Baselines3 vectorized envs for PPO throughput. The cloud gives few CPU cores, which limits this and motivates the JAX option in section 17. SuperSuit steps all N cars **sequentially inside one process**, so a process's step cost scales ~linearly with N on the dynamic model and is **not** amortized by `SubprocVecEnv` fan-out — measure SPS (field vs equal-width single-agent) before committing field sizes.
- The checkpoint meta adds `n_agents` (the constant field size the run trained on; 1 = single-agent). It is **not** validated on resume — the per-agent obs/action spaces match across widths, so a smaller-field checkpoint warm-starts a larger-field run.
- Determinism: one seeding utility seeds Python, NumPy, and PyTorch. Record the seed with every run. The per-episode circuit draw and both reset modes use `self.np_random` (reproducible from the seed).

---

## 11. Rendering pipeline

The interactive application is the primary surface of the project and the first thing built (section 15, phase 1). You see every change here.

- A web app: a Vite + TypeScript frontend with an HTML5 Canvas 2D viewport, talking to a local FastAPI/uvicorn backend over a WebSocket. The backend runs the simulation loop and streams car state at the control rate (20 Hz); the frontend draws at its own frame rate, interpolating between the last two streamed states. The viewport has a camera you pan and zoom, plus a HUD (top bar, timing tower, telemetry bar) recreating the approved broadcast-style design.
- Four modes, unchanged in spirit: manual drive with keyboard control (the browser sends input messages over the socket), configure to set track surfaces and conditions and save them to the `Track`, watch live to run a policy in real time, and replay to play back a recorded run.
- Watch live works for any policy, trained or untrained, because running a policy forward is cheap. This is how you see a car drive without waiting on training.
- Drawing: the Canvas draws the asphalt ribbon between the boundaries, the kerb bands, the grass or gravel background, and each car as an oriented shape colored by team, all from Python track geometry in meters through a meters→pixels camera transform. A debug overlay draws the centerline and the rangefinder beams.

The heavy training loop draws nothing, for speed, and the web frontend is never in that hot path. Two paths keep full visibility. The evaluation callback runs offscreen on the cloud, with the SDL video driver set to dummy, renders one episode to an mp4 through Pygame and imageio, and logs the clip — this headless eval-clip path is unchanged by the web pivot. After training, the app loads a checkpoint and runs the agent live. The app stays your view into everything. Only the long training loop runs unseen.

The recorded-trajectory JSON format (section 10's recorder output) is the shared interchange between the live sim, the replay viewer, and the cloud eval-clip renderer: all three consume the same frame stream, so a run recorded in any mode replays identically anywhere. Phase 5: the live frame and the recorder carry a `cars: [{id, x, y, yaw, speed, team, telemetry}, …]` array (a single car is a one-element array, so the Phase 1/4 one-car path is unchanged); the field is driven live by `FieldSimLoop` (N cars, one shared pilot) and the frontend draws every car colored by team, follows the leader, and lists the field in the timing tower by track-position gap.

---

## 12. Training, checkpointing, and logging

- `train.py` runs from a config, logs to Weights and Biases, and saves checkpoints on a schedule.
- A checkpoint holds the model weights, the optimizer state, the observation normalization stats, the total timestep count, the config, and the RNG states. Saving and resuming must round-trip exactly.
- `--resume <path>` continues a run after a disconnect. Assume the session can die at any moment and checkpoint frequently.
- Checkpoints write to Google Drive on Colab or to notebook output on Kaggle.
- An evaluation callback periodically runs one deterministic episode, records it, renders a short clip, and logs the clip and the metrics to Weights and Biases.
- Logged metrics: episode return, lap time against the pole and twice the pole, off-track count, contact count in the racing phase, and the learning curves.
- **Curriculum Learning**: Staged realism is enforced through a config-driven `CurriculumCallback`. The callback steps through stages (e.g., Dry -> Damp -> Full Dynamic with Tire Wear) based on the current timestep, calling an `apply_conditions` hook on the environment without halting the training loop.

---

## 13. Repository structure

```
f1-rl/
  README.md
  CLAUDE.md                     # repo guidance for the coding agent
  DISCREPANCIES.md              # standing doc/code audit + resolution log
  pyproject.toml                # dependencies and tooling
  .claude/                      # design docs live here (committed, auto-loaded as context)
    PROJECT_VISION.md
    TECHNICAL_DESIGN.md         # this file
    specs/<phase>.md            # per-phase specs
    plan/<phase>-plan.md        # per-phase file-by-file build plans
    agents/<role>.md            # subagent definitions
  configs/
    default.yaml             # global defaults: seed, sim, physics, obs, reward, env, grip pipeline
    track/<circuit>.yaml     # per-circuit geometry, pole time, lap count; merged under cfg.track
    experiment/<name>.yaml   # per-run trainer config (rbr_ppo = Phase 3a, rbr_dynamic = Phase 3b)
  data/
    raw_telemetry/              # FastF1 cache, gitignored
    tracks/                     # processed Track files, .npz, + _build_report.json
                                #   + pre-baked web payloads: <id>.api.json (GET /track body)
                                #   and _catalog.json (GET /api/tracks body), committed so the
                                #   server serves a file read instead of reloading/serializing
                                #   numpy per request (scripts/bake_track_json.py regenerates)
  recordings/                   # recorded-trajectory JSON, gitignored
  web/                          # interactive app frontend (Vite + TypeScript)
    index.html                  # the broadcast-style stage shell
    src/
      main.ts                   # bootstrap, stage scaling, UI state machine
      state.ts                  # UI state machine + app store
      types.ts                  # shared TS types (track, frame, messages)
      format.ts                 # number/time formatting helpers
      tokens.css  styles.css    # design tokens + fonts + layout
      net/socket.ts             # WebSocket client to /ws/sim
      input/keyboard.ts         # arrows + WASD -> input messages
      viewport/camera.ts        # meters<->pixels transform, pan/zoom/follow
      viewport/renderer.ts      # Canvas 2D draw, 20 Hz interp; field draw per team (Phase 5)
      hud/telemetry.ts          # tower + telemetry bar; field tower by track-position gap (Phase 5)
      replay/player.ts          # load + play/pause/scrub a trajectory (single-car or field)
      ui/selector.ts            # track selector
      ui/config_panel.ts        # surface/condition editor (Phase 2)
      ui/policy_picker.ts       # checkpoint picker for watch-live (Phase 3)
  src/f1rl/
    __init__.py
    physics/
      base.py                   # CarState, PhysicsModel interface
      kinematic.py
      dynamic.py                # dynamic bicycle + friction circle
      tires.py                  # grip pipeline and tire model
      factory.py                # make_physics: "kinematic" | "dynamic"
    track/
      schema.py                 # Track dataclass
      geometry.py               # tangent/normal/arc-length/curvature (shared)
      oval.py                   # procedural oval Track (Phase 1)
      build.py                  # FastF1 + OSM to processed Track, offline
      loader.py                 # load cached Track
    sim/
      loop.py                   # fixed-step simulation loop
      timing.py                 # lap timing and delta-to-pole
      recorder.py               # trajectory recording (JSON)
      autopilot.py              # centerline pure-pursuit for watch-live
      policy_pilot.py           # checkpoint-driven pilot for watch-live
    server/
      app.py                    # FastAPI app: WS /ws/sim, GET /track, GET /recordings, policies
      messages.py               # Pydantic client/server message models
    env/
      single_agent.py           # RacingEnv + factored step_one_car/reset_car/CarRuntime (Phase 5)
      observations.py           # ObservationV2 builder (length 22)
      rewards.py                # reward_v1 / reward_v2 (progress core)
      conditions.py             # weather/surface state + grip provider
      pool.py                   # CircuitPool: per-id Track/EdgeCache/LapTimer/pole (Phase 4)
      factory.py                # make_vec_env + make_selfplay_vec_env (SuperSuit→SB3, Phase 5)
      multi_agent.py            # RacingParallelEnv: N-car field, per-car LapTimer (Phase 5)
    render/
      renderer.py               # offscreen frames to mp4 for eval clips (training only)
    train/
      train.py                  # config-driven PPO entrypoint, checkpoint/resume
      evaluate.py               # deterministic eval episode + metrics + clip
      callbacks.py              # checkpoint, eval video, wandb
      checkpointing.py          # save/load + obs-version/action-shape validation
      curriculum.py             # CurriculumCallback (conditions + circuit pool by timestep)
      calendar_benchmark.py     # lap-time-vs-pole table across the pool (Phase 4 artifact)
      benchmark.py              # steps-per-second sweep
      calibrate.py              # tune grip/engine/brake so optimal lap ~ pole
      wandb_logger.py           # W&B with local-CSV fallback
      selfplay.py               # shared-policy self-play PPO + throughput check (Phase 5)
      selfplay_eval.py          # multi-car field eval driver (cars[] trajectory + metrics, Phase 5)
    utils/
      seeding.py
      config.py                 # OmegaConf load
  scripts/
    build_all_tracks.py
    benchmark_sps.py
  tests/                        # pytest: env_checker, physics (kinematic+dynamic), grip,
                                # obs, rewards, termination, track build/oval, timing,
                                # recorder, config, seeding, checkpoint, curriculum, server,
                                # smoke-train (kinematic+dynamic), lap-benchmark,
                                # circuit-pool, env-sampling, calendar-benchmark (Phase 4),
                                # multi-agent-env, selfplay-smoke (Phase 5)
```

The interactive app is split across the top-level `web/` frontend and two Python packages: `server/` (the FastAPI app and its message models) and `sim/` (the fixed-step loop, lap timing, the trajectory recorder, and the centerline autopilot for watch-live). The `recorder.py` lives under `sim/` because the recorded-trajectory JSON it produces is the shared interchange across live sim, replay, and eval clips. `render/renderer.py` stays under the package as the offscreen eval-clip renderer used by training only; it is never imported by the app or the training hot path.

The tree reflects the repo **as built through Phase 5** (`env/multi_agent.py`,
`env/factory.make_selfplay_vec_env`, `train/selfplay.py`, `train/selfplay_eval.py`, and the
`grid:`/`selfplay:` config blocks are all present). Eval-clip rendering is invoked through
`train/evaluate.py --video` (there is no separate `scripts/render_episode.py`); the Phase 5
field produces a multi-car `cars[]` trajectory replayed in the web app rather than a multi-car
mp4. There is no `notebooks/` directory yet — the Colab/Kaggle path clones the repo and runs
`python -m f1rl.train.train` (single-agent) or `python -m f1rl.train.selfplay` (field) directly.

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

Phase 4, one car on many circuits. Confirm the observations are fully track-agnostic, then sample a different circuit each episode so one policy handles every track. Artifact: one policy driving every circuit, with a lap-time table against the pole per circuit. As-built contracts: the `circuits:` pool config block and per-episode sampling (§10 circuit pool); per-circuit pole from `configs/track/<id>.yaml`; the curriculum stage gains an optional `circuits:` list that `set_track_pool` widens to the full calendar; the lap-time table (`train/calendar_benchmark.py`, one row per circuit — achieved / pole / delta / 2×-pole flag) is saved as JSON+CSV under `out/` and served at `GET /api/calendar` for the result view (toggle with T). The checkpoint format is unchanged (`obs_version` stays 2 → the Phase 3b checkpoint warm-starts; `--resume` continues the timestep count); `meta.json` `circuit_id` records the pool descriptor (e.g. `"calendar"`).

Phase 5, many cars on track. Add the multi-agent env on PettingZoo with shared-policy self-play, and put the full field on track with no racing rules yet. Scale from 2 cars to 4 to the full 22. Artifact: a grid lapping a circuit together in the app. As-built contracts: `RacingParallelEnv` (`env/multi_agent.py`) holds N homogeneous cars on one shared circuit, reusing the factored `step_one_car`/`reset_car` per car with a **per-car `LapTimer`** (never the pooled entry's), passing `parallel_api_test`; the **observation is unchanged** (`OBS_VERSION = 2`, no nearby-car block); the `grid:` config block sets field size + reset mode (`scattered` train / `grid` eval); SuperSuit (`black_death_v3` for constant width → `pettingzoo_env_to_vec_env_v1` → `concat_vec_envs_v1`) → SB3 PPO trains one shared policy (`make_selfplay_vec_env`, `train/selfplay.py`), warm-starting the Phase 4 generalist; **field size is a per-run constant grown across warm-started runs** (the circuit-pool widening stays an in-place curriculum, broadcast to the raw envs); the checkpoint meta adds `n_agents` (not validated across widths); the live frame carries a `cars: [...]` array (single car = one-element, backward compatible) driven by `FieldSimLoop`, with one multi-car recorder and the timing tower listing every car by track-position gap. **The bar is infrastructure + the render, not a learning gain** (none expected with no mutual observation).

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