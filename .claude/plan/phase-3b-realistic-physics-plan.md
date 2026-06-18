# Phase 3b Implementation Plan — Realistic Physics & Lap-Time Benchmark

Companion to `.claude/specs/phase-3b-realistic-physics.md` (the spec). This is the **how**:
concrete, dependency-ordered, file-by-file build order grounded in the real Phase 1/2/3a
code, dispatched through the subagent roster in spec §5. Branch:
`phase-3b-realistic-physics`.

> Authoritative engineering doc remains `.claude/TECHNICAL_DESIGN.md` (§4 grip pipeline,
> §5 physics model + `PhysicsModel` interface, §7 observations, §9 reward, §10 env contract,
> §12 training). Where this plan fixes a contract the design leaves open (the exact
> ObservationV2 layout, the dynamic-model equations and friction-circle reduction, the grip
> pipeline factor signatures, the curriculum schema), update `TECHNICAL_DESIGN.md` in the
> **same commit** — the decision and the doc move together (CLAUDE.md rule).

The headline: replace the kinematic model with a **dynamic bicycle model + friction circle**
behind the unchanged `PhysicsModel` interface, wire the **grip pipeline** (tires × weather ×
surface) into the env and the live loop, grow the observation to **v2 (length 22)**, retrain
on a **curriculum**, **calibrate** the car so a clean optimal lap lands near the pole, and
**benchmark** lap time against the real pole.

---

## Confirmed / assumed decisions (resolve the spec open questions)

All are config values, reversible; flagged where they cost a retrain.

1. **Circuit = `red_bull_ring`** (unchanged from 3a). Built, clean (length_error 0.55 %,
   `pole_time_s: 64.3`, `total_laps: 71`). No new track work (spec non-goal). Phase 4
   generalizes; here we stay on one circuit.
2. **Physics = dynamic bicycle + friction circle**, linear tires first (Pacejka deferred per
   §2g/§17). Selected via `cfg.physics.model: dynamic` through the existing
   `make_physics` factory — `make_physics` currently raises `NotImplementedError` for
   `"dynamic"`; this phase implements it. **No `reset`/`step` signature change anywhere.**
3. **ObservationV2 = fixed length 22**, `OBS_VERSION = 2`:
   `ObservationV1(15) + tire_wear(1) + compound_onehot(5) + grip_indicator(1)`. The length
   change forces a retrain (expected, clean); the bumped `OBS_VERSION` makes the checkpoint
   loader refuse every v1 checkpoint automatically (`validate_checkpoint`, already built).
4. **Action unchanged** — `Box(-1, +1, shape=(2,))` = `[steer, longitudinal]` (§8). The
   action contract is stable, so 3a checkpoints fail only on `obs_version`, which is correct.
5. **Grip pipeline** = `mu_base · tire(compound, wear) · weather · surface`, one scalar gating
   the friction circle (§4). `tires.py` holds the **pure** factor functions (no `Track`); the
   **env owns the surface lookup** (it does track queries) and computes the scalar each step,
   then passes it into `physics.step` — physics stays a pure function of `(state, steer,
   longitudinal, grip, dt)`.
6. **Wear lives in `CarState.tire_wear`** (already in the struct) and advances **inside the
   dynamic step** (it has the slip/load the wear rate needs); `compound` is set at reset from
   config. Grip for a step is computed by the env from the **pre-step** wear/surface/weather
   and passed in; the step uses it for the friction circle and separately advances wear for
   the next step. Document this ordering.
7. **Curriculum = single run, config-driven scheduler** (`CurriculumCallback`) that ramps
   conditions (grip → wear → weather) at timestep thresholds via `VecEnv.set_attr` /
   `env_method`. One run, one set of curves. Fallback: sequential stage configs chained by
   `--resume` (warm start is fine *within* obs-v2; only the v1→v2 jump forbids it).
8. **Retrain from scratch** (spec §b, §g): obs and dynamics both changed; no warm start from
   the 3a policy.
9. **Calibration is deterministic and agent-independent**: a forward–backward
   **velocity-profile lap-time estimator** over the track curvature under the configured
   grip/engine/downforce limits gives the achievable clean-lap time as a pure function of the
   physics config. Calibrate by sweeping the lever params (downforce, `mu_base`, engine force)
   until that estimate lands near 64.3 s — *before* burning training compute.
10. **Everything tunable in config** under a new `configs/experiment/rbr_dynamic.yaml` plus new
    `physics`/`tires`/`weather`/`curriculum`/`reward` keys. No tuning constant in logic.
11. **Device = local CPU** by default (installed torch is CPU-only, 4 cores). The dynamic step
    is heavier than kinematic; re-run the existing SPS benchmark (`train/benchmark.py`) on the
    dynamic model to re-make the local-vs-cloud call with data (spec §3a).

---

## Phase 1/2/3a baseline (verified in code — what we build on)

- `physics/base.py` — `CarState` already carries the **full dynamic struct** (`vx, vy,
  yaw_rate, tire_wear, compound`) and `PhysicsModel` Protocol
  `step(state, steer, longitudinal, grip, dt) -> CarState`. **No struct change needed.**
- `physics/kinematic.py` — `KinematicBicycle`, `KinematicParams.from_config`. Stays as the
  Phase-1 model and the low-speed blend target for the dynamic model.
- `physics/factory.py` — `make_physics(cfg)` selects on `cfg.physics.model`; `"dynamic"`
  branch currently raises `NotImplementedError`. **This is the single swap point.**
- `env/single_agent.py` — `RacingEnv` owns its own substep loop, builds physics via
  `make_physics`, computes off-track from `track_query`, calls `reward_v1`, builds obs via
  `build_observation`. Passes `self.conditions.grip` (constant `1.0`) into `step` — **this is
  the one line that becomes the grip-pipeline call.**
- `env/conditions.py` — tiny `Conditions(grip, wet)` holder, *designed to expand* (its
  docstring says so) without an env signature change. Grows into the grip provider.
- `env/observations.py` — pure NumPy (no torch/gym in the hot path), `OBS_VERSION = 1`,
  `OBS_DIM = 15`, `build_observation`, `track_query` (shared projection), `observation_space()`.
  Reused verbatim by `PolicyPilot`. Extend to v2 here.
- `env/rewards.py` — `reward_v1` + `RewardWeights.from_config`, progress-only, never
  centerline. Extend to `reward_v2` (additive optional shaping) here.
- `env/factory.py` — `make_vec_env` (SubprocVecEnv + VecNormalize). Unchanged.
- `train/checkpointing.py` — checkpoint format + `validate_checkpoint(meta)` already enforces
  `obs_version`/action shape with a clear error. Bumping `OBS_VERSION` to 2 is all that's
  needed to refuse v1 checkpoints.
- `train/evaluate.py` — already computes `best_lap_time`, `beat_pole`, `beat_2x_pole`,
  off-track count, steps-to-first-clean-lap, and a `summary(pole)` with `pole_time_s` /
  `two_x_pole_time_s`. The lap-benchmark is ~80 % built; Part 2 adds the **delta** and
  missing-pole handling.
- `train/callbacks.py` — `CheckpointCallback` + `EvalVideoCallback`. The curriculum callback
  slots in beside them.
- `sim/loop.py` — `SimLoop` (watch-live engine) takes a `PhysicsModel` and passes the
  **constant** `cfg.sim.grip` into `step`, emits the telemetry frame. For a realistic live
  view it must use the dynamic model + the grip provider and surface the new readouts.
- `sim/policy_pilot.py` — `PolicyPilot` loads a checkpoint, builds obs via the **shared**
  `env/observations.py`, validates `obs_version`. Picks up obs-v2 for free once the builder
  grows (it imports the builder, not a copy).
- `server/app.py` — **builds `KinematicBicycle` directly** (`app.py:127`) and serves the
  watch-live `SimLoop`. Must route physics through `make_physics(cfg)` for the dynamic swap.
- `server/messages.py` — `SurfaceEdit` already carries a `condition: dry|wet` field
  (forward-compat, no-op in Phase 2). Phase 3b makes it live.
- `web/src/hud/telemetry.ts` — HUD readouts; tyre is a single static compound dot, no wear/
  grip. Add compound, wear, and grip/weather readouts here.

**Gaps to create:** `src/f1rl/physics/dynamic.py`, `src/f1rl/physics/tires.py`,
`src/f1rl/train/calibrate.py`, the curriculum callback, obs-v2 + reward-v2 extensions, the
grip-provider extension to `conditions.py`, `configs/experiment/rbr_dynamic.yaml` + new config
blocks, the new tests, and the re-created `.claude/agents/` roster (deleted in the working
tree).

---

## Contracts fixed before any code (foundation)

### Dynamic bicycle model (`physics/dynamic.py`, `DynamicBicycle(PhysicsModel)`)

State body frame: `vx` (longitudinal), `vy` (lateral), `r = yaw_rate`. Params from config
(`DynamicParams.from_config`): `mass m`, yaw inertia `Iz`, CG-to-front `lf`, CG-to-rear `lr`
(`lf + lr = wheelbase`), front/rear cornering stiffness `Cf`/`Cr`, `max_steer`,
`max_engine_force`, `max_brake_force`, `drag_coeff`, `rolling_coeff`, `downforce_coeff`
(= `0.5·ρ·Cl·A`), `wear_rate`, low-speed blend speed `v_blend`. All SI, all tunable.

Per control substep (`dt = dt_physics`, semi-implicit Euler):

1. **Commands** → `delta = steer · max_steer`; longitudinal force `Fx_drive` from
   throttle·engine or brake (as the kinematic model), minus `drag·vx·|vx|` and rolling
   resistance.
2. **Slip angles** (guard the singularity at `vx → 0`):
   `vx_safe = max(vx, v_eps)`;
   `alpha_f = atan2(vy + lf·r, vx_safe) − delta`;
   `alpha_r = atan2(vy − lr·r, vx_safe)`.
3. **Linear lateral tire forces**: `Fyf = −Cf·alpha_f`, `Fyr = −Cr·alpha_r`.
4. **Friction circle (THE central abstraction, must be literally testable):**
   `max_force = grip · m · g + downforce_coeff · vx²` (the optional aero term, §4).
   Reduce **each axle's** combined force so `hypot(Fx_axle, Fy_axle) ≤ max_force_axle`: if a
   tire is over the circle, scale its longitudinal **and** lateral components by
   `max_force / hypot(...)`. The unit test asserts the realized combined force never exceeds
   `max_force` (+tolerance). Keep the reduction simple and explicit so the test is exact.
5. **EoM** (body frame, with centripetal coupling):
   `ax = (Fx_total − Fyf·sin delta)/m + vy·r`;
   `ay = (Fyf·cos delta + Fyr)/m − vx·r`;
   `r_dot = (lf·Fyf·cos delta − lr·Fyr)/Iz`.
   Integrate `vx += ax·dt`, `vy += ay·dt`, `r += r_dot·dt`.
6. **Low-speed blend (Kong et al. 2015):** below `v_blend` blend the dynamic `vy`/`r` toward
   the **kinematic** prediction (no-slip `r = vx·tan delta / L`, `vy = 0`) with a smooth
   weight, so standstill and pit-crawl speeds stay stable and never produce NaN. Clamp the car
   out of reverse as the kinematic model does (no reverse gear this phase).
7. **Pose**: `yaw += r·dt`; `x += (vx·cos yaw − vy·sin yaw)·dt`;
   `y += (vx·sin yaw + vy·cos yaw)·dt`.
8. **Wear**: `tire_wear = clip(tire_wear + wear_rate · slip_load · dt, 0, 1)` where
   `slip_load` grows with combined slip and speed (config-shaped). `wear_rate = 0` disables
   wear (early curriculum). Return the new `CarState` with updated `vx, vy, r, tire_wear`,
   carrying `compound` unchanged.

Pure function: no globals, no rendering, no track lookups inside `step` (CLAUDE.md rule). The
env passes the precomputed `grip` scalar in.

### Grip pipeline (`physics/tires.py`, pure — no `Track`)

```python
def effective_grip(mu_base, compound, wear, weather, surface_zone) -> float:
    return mu_base * tire_factor(compound, wear) * weather_factor(weather) * surface_factor(surface_zone)
```

- `tire_factor(compound, wear)` — per-compound base (soft > medium > hard > inter > wet) times
  a wear falloff `(1 − wear_falloff·wear)` (or quadratic), all from config. Monotone: wear ↑ →
  grip ↓; soft > medium > hard at equal wear.
- `weather_factor(weather)` — dry `1.0`, damp `~0.8`, wet `~0.6` (config).
- `surface_factor(surface_zone)` — asphalt `1.0`, kerb `~0.9`, grass `~0.4`, gravel `~0.3`
  (config). Surface also bleeds longitudinal time via the existing off-track penalty + drag.

`TireParams.from_config` / `WeatherParams` / `SurfaceParams` hold the tables. Everything in
`configs` under a `tires:` / `weather:` block. `compound` indices match `CarState`:
`0 soft, 1 medium, 2 hard, 3 intermediate, 4 wet`.

### Surface lookup + grip provider (`env/conditions.py`, extended)

`Conditions` grows a method `grip_at(track, nearest_idx, signed_lateral, wear, compound)` that:
classifies the surface zone from `|signed_lateral|` against `half_width` + `kerb_width` +
`grass_width` + `gravel_width` at `nearest_idx` (asphalt inside half-width; kerb in the kerb
band; then grass/gravel per `surface_zones`), reads the current `weather`, and returns
`effective_grip(...)`. Pure NumPy, no torch/gym, so both `RacingEnv` **and** `SimLoop` import
it and agree. The env already has `nearest_idx`/`signed_lateral`/`half_width` from
`track_query` each step — feed them straight in (no second projection).

### ObservationV2 (`OBS_VERSION = 2`, `OBS_DIM = 22`)

| idx | field | definition | normalization |
|----|----|----|----|
| 0–14 | `ObservationV1` | unchanged (speed, heading err, lateral, curvature[5], beams[7]) | as v1 |
| 15 | `tire_wear` | `state.tire_wear` | already `[0,1]` |
| 16–20 | `compound_onehot[5]` | one-hot of `state.compound` (0 soft … 4 wet) | `{0,1}` |
| 21 | `grip_indicator` | effective grip at the car (folds weather + surface + wear + compound) | `/ mu_base`, clipped to a generous Box |

`observation_space()` returns `Box(shape=(22,))` with v1 bounds plus `[0,1]` for wear,
`{0,1}` for the one-hot, and a bounded grip range. The builder clips so `check_env` never sees
an out-of-space value. The grip indicator reuses `Conditions.grip_at` so train/serve agree.

### RewardV2 (`reward_v2`, additive over the v1 core)

Same progress core (`w_progress·ds − offtrack − w_step − w_reverse·max(0,−ds)`), **never
centerline-seeking**. Adds optional, config-gated shaping for the harder dynamics (defaults
that keep behavior ≈ v1 so a regression is opt-in):

- `w_slip` — small penalty on excessive lateral slip / spin (e.g. `|alpha_r|` or `|vy/vx|`
  beyond a threshold) to discourage overdriving past the grip limit. Default `0`.
- The graded off-track penalty stays; grass/gravel now *also* cut grip via the pipeline, so a
  slide off already loses lap time on its own (the realism the spec asks for).

Keep every weight in `configs`. `reward_v2` returns the same `(float, terms)` breakdown shape
for logging. Select via `cfg.reward.version: 2` (default `1` stays back-compatible).

### Curriculum (`train/curriculum.py` or a callback in `train/callbacks.py`)

Config-driven stage table, applied to the live `VecEnv` by timestep:

```yaml
curriculum:
  enabled: true
  stages:
    - start_step: 0          # learn the line on easy grip
      mu_base: 1.30          # high grip
      wear_rate: 0.0         # no wear
      weather: dry
    - start_step: 600000     # nominal grip, still dry, no wear
      mu_base: 1.05
      wear_rate: 0.0
      weather: dry
    - start_step: 1200000    # introduce wear
      mu_base: 1.05
      wear_rate: 0.02
      weather: dry
    - start_step: 1800000    # introduce weather (sample wet episodes)
      mu_base: 1.05
      wear_rate: 0.02
      weather: sampled       # P(wet) from cfg
```

`CurriculumCallback._on_step` finds the active stage for `num_timesteps` and pushes the
overrides into every worker via `self.training_env.set_attr(...)` / `env_method(...)` (the env
re-reads them in `reset`/`step`). The stage values are conditions only — they never touch the
obs layout, so no retrain mid-run.

### Lap-time benchmark (extend `train/evaluate.py`)

`evaluate` already yields `best_lap_time`, `beat_pole`, `beat_2x_pole`. Add:
`lap_delta_to_pole = best_lap_time − pole` (positive = slower), surface it in `summary`, and
log a `eval/gap_to_pole` so its **downward trend** is visible. **Missing pole** (`pole<=0`):
skip the delta and set a `eval/pole_missing = 1` flag rather than dividing by zero (spec error
state). The first milestone is `best_lap ≤ 2·pole`, then tighten.

### Calibration (`train/calibrate.py`, new — deterministic, agent-free)

A pure lap-time estimator from physics config + track geometry:
`v_limit(s) = sqrt(max_force(v) / (m·|kappa(s)|))` per sample (lateral grip limit, with the
aero term making `max_force` speed-dependent — solve/iterate), then a **forward–backward
pass** bounding acceleration by engine force and braking by brake force, each within the
friction circle, to get a feasible speed profile and integrate `dt = ds / v` for a clean
optimal lap time. Sweep a lever (`downforce_coeff`, then `mu_base`, then `max_engine_force`)
and report the value that lands the estimate near `pole_time_s`. CLI prints a small table;
the chosen values go into `configs/experiment/rbr_dynamic.yaml`. This makes the score *fair*
(spec §1d, §4-Deliberation) without spending training compute to discover miscalibration.

### Checkpoint (`OBS_VERSION = 2`)

No format change. Bumping `OBS_VERSION` in `observations.py` makes `validate_checkpoint`
refuse every Phase-3a (v1) checkpoint with the existing clear message — exactly the spec's
"loader refuses a mismatched policy". `meta.json` already records `obs_version`.

---

## Build order (dependency-first), mapped to subagents

### Step 0 — scaffold (main thread)
- Create branch `phase-3b-realistic-physics`.
- **Re-create the `.claude/agents/` roster for 3b** (physics-engineer, env-engineer,
  training-engineer, app-integration-engineer, test-engineer, reviewer) — the 3a defs were
  deleted in the working tree. Each agent file's body **must open with the directive to call
  the `/caveman` skill before any work** (spec §5, emphasized, non-negotiable), then its
  narrow scope, "read first" list, tasks, rules, and done-gate. Point each at this plan + the
  spec + the relevant `TECHNICAL_DESIGN.md` sections.
- Add the new config blocks (`tires`, `weather`, `curriculum`, dynamic `physics` keys,
  `reward.version`) to `configs/default.yaml` with safe defaults, and create
  `configs/experiment/rbr_dynamic.yaml` (extends the 3a run, sets `physics.model: dynamic`).

### Step A — physics-engineer · `src/f1rl/physics/` (critical path, the main 3b role)
Build the dynamic model + grip pipeline behind the unchanged interface.

- **`tires.py`** — `effective_grip` + `tire_factor` / `weather_factor` / `surface_factor` and
  their `*Params.from_config`. **Pure**, no `Track`, no torch. Compound indices match
  `CarState`.
- **`dynamic.py`** — `DynamicBicycle(PhysicsModel)` + `DynamicParams.from_config`, per the
  formulation above (slip angles with the `vx→0` guard, linear tires, **friction-circle
  reduction**, EoM, low-speed kinematic blend, wear update). Pure `step`.
- **`factory.py`** — implement the `"dynamic"` branch of `make_physics` (build
  `DynamicBicycle(DynamicParams.from_config(physics_cfg))`); keep `"kinematic"` intact.

**Gate:** `tests/test_physics_dynamic.py` + `tests/test_grip_pipeline.py` pass; friction
circle caps combined force; each grip factor monotone in the right direction; no NaN at
`vx≈0`; `make_physics(model="dynamic")` returns a working model; `ruff` clean.

### Step B — env-engineer · `src/f1rl/env/` (depends on A's `tires.py` signatures)
Grow obs to v2, reward to v2, wire the grip pipeline — **no `reset`/`step` signature change.**

- **`conditions.py`** — extend `Conditions` with the surface classifier + `grip_at(...)` grip
  provider and a `weather` field; `from_config` reads `tires`/`weather`. Pure NumPy.
- **`observations.py`** — `OBS_VERSION = 2`, `OBS_DIM = 22`; append `tire_wear`,
  `compound_onehot[5]`, `grip_indicator` (via `Conditions.grip_at`); widen
  `observation_space()`; keep the v1 slice byte-identical so only the tail is new.
- **`rewards.py`** — add `reward_v2` (v1 core + optional `w_slip`), `RewardWeights` gains the
  new weight + `version`. Never centerline.
- **`single_agent.py`** — replace the constant `self.conditions.grip` passed into `step` with
  the **per-step pipeline grip** from `Conditions.grip_at` (reuse the existing `track_query`
  outputs — no second projection); set `compound` at reset from config; select `reward_v2`
  when `cfg.reward.version == 2`; pass `grip_indicator` into the obs build. Keep the env
  rendering-free and recorder-optional.

**Gate:** `gymnasium.utils.env_checker.check_env(RacingEnv(dynamic cfg))` passes; obs ∈ space
at length 22; `tests/test_observations.py` (v2) + `tests/test_env_api.py` green; a random
rollout on the dynamic model runs N steps without error or NaN.

### Step C — training-engineer · `src/f1rl/train/` + `configs/experiment/`
Depends on B (env on dynamic + obs v2).

- **`calibrate.py`** (new) — the forward–backward lap-time estimator + lever sweep; CLI prints
  the calibration table. **Run first** and bake the chosen params into `rbr_dynamic.yaml`.
- **`evaluate.py`** — add `lap_delta_to_pole` + `gap_to_pole` to `EpisodeMetrics`/`summary`;
  handle missing pole (flag, don't divide).
- **`curriculum.py`** / `callbacks.py` — `CurriculumCallback` ramping conditions by timestep
  via `set_attr`/`env_method` from the `curriculum:` config table.
- **`observations.py` bump already done in B** → `OBS_VERSION = 2` makes `validate_checkpoint`
  refuse v1 checkpoints; confirm with a test.
- **`configs/experiment/rbr_dynamic.yaml`** — `physics.model: dynamic` + dynamic params +
  `tires`/`weather`/`curriculum`/`reward.version: 2` + calibrated levers + PPO/eval/checkpoint
  blocks (start from `rbr_ppo.yaml`). Bump `wandb.tags`/`group` to `phase-3b`.
- Re-run **`train/benchmark.py`** on the dynamic model to re-make the local-vs-cloud call.
- Then the **retrain** + a **tiny-budget smoke run** proving reward trends up and resume
  continues the timestep count.

**Gate:** calibration table recorded; smoke run reward non-degenerate on dynamic physics;
checkpoint resumes; one eval mp4 + `gap_to_pole` logged; `2·pole` reached as training
progresses.

### Step D — app-integration-engineer · server + `web/` (depends on C's checkpoint, A's physics)
Make the watch-live view realistic and show the new readouts.

- **`server/app.py`** — build watch-live physics via **`make_physics(cfg)`** (not a direct
  `KinematicBicycle`) so `physics.model: dynamic` drives the live car; have `SimLoop` use the
  **grip provider** so surface/weather change the live grip. Bad/missing checkpoint or obs
  mismatch still falls back to the autopilot and never crashes (existing path).
- **`sim/loop.py`** — compute the per-step grip from `Conditions.grip_at` (like the env)
  instead of the constant `cfg.sim.grip`; add `compound`, `tire_wear`, and `grip` to the
  telemetry frame.
- **`server/messages.py`** — make the `SurfaceEdit.condition` (dry/wet) actually set the live
  weather; optionally a small message to pick compound/weather for the watch session.
- **`web/src/hud/telemetry.ts`** (+ `types.ts`, `format.ts`) — render the tyre **compound**
  (dot color by compound), **wear %**, a **grip/weather** indicator, and the lap time + delta,
  colored by the Phase-1 timing colors. Reuse the debug overlay to *see* grip-limited driving.

**Gate:** watch-live runs a dynamic-physics checkpoint; tyres wear and weather visibly change
the driving; the readouts and the delta-to-pole show; switching wet/dry changes grip live.

### Step E — test-engineer (independent) · `tests/`
Writes from the **spec contracts and public signatures only** — not implementation internals —
so tests verify requirements unbiased (spec §5, §c). `/caveman` first.

- `test_physics_dynamic.py` — friction circle caps combined force ≤ `grip·m·g (+aero)`; turning
  produces `yaw_rate`/`vy`; straight-line ≈ kinematic; **stable at `vx≈0`** (no NaN/blowup);
  higher grip → tighter stable cornering speed.
- `test_grip_pipeline.py` — `effective_grip` is the product of the four factors; wear ↑ →
  grip ↓; soft > medium > hard; wet < dry; grass/gravel < asphalt; bounds sane.
- `test_observations.py` (v2) — `OBS_DIM == 22`, `OBS_VERSION == 2`, in-space; compound one-hot
  valid + matches `state.compound`; wear ∈ `[0,1]`; grip indicator bounded; v1 slice unchanged.
- `test_env_api.py` — `check_env` passes on the dynamic env; action space unchanged.
- `test_checkpoint.py` — a v1 (`obs_version=1`) checkpoint is **refused with a clear message**;
  a fresh v2 checkpoint round-trips weights + vecnorm + timestep exactly.
- `test_lap_benchmark.py` — lap time + delta correct against a known reference; `2·pole`
  milestone flag fires; missing pole is skipped/flagged, never crashes.
- `test_curriculum.py` — the scheduler activates the right stage at each threshold and pushes
  conditions into the workers.
- `test_smoke_train_dynamic.py` (integration, tiny budget) — `learn()` runs on dynamic
  physics, reward trend non-degenerate, checkpoint resumes and continues the timestep count.

### Step F — reviewer · gate every diff (read-only)
`/caveman` first. Checklist per task: config-driven (no magic constant in logic); SI units;
**physics behind `PhysicsModel`, no `reset`/`step` signature change**; `tires.py` pure (no
`Track`); env passes the checker at obs **v2**; `OBS_VERSION` bumped **deliberately** to 2 and
the loader refuses v1; **no rendering in `env.step` or the training hot path**; FastF1 not
imported under `env/` or `train/`; grip pipeline is one scalar gating the friction circle;
shared obs builder reused by the server (not reimplemented); seed recorded; checkpoint
round-trips. Runs `pytest` + `ruff check` + `ruff format --check`. Pass/fail with reasons;
blocks the merge on any violation or red test/lint.

---

## Dispatch DAG (dependency order)

```
0. scaffold (branch, re-create agent defs w/ /caveman, config blocks)
1. physics-engineer (A): tires.py + dynamic.py + make_physics 'dynamic'   ──┐  test-engineer (E)
   (critical path)                                                          │  starts in parallel,
2.                                   dynamic model + grip pipeline ready    │  writing failing tests
3. env-engineer (B): conditions grip provider, obs v2, reward v2, env wire  │  from the spec contracts
4. training-engineer (C): calibrate FIRST → rbr_dynamic.yaml → curriculum   ┘
                          → benchmark → retrain + smoke
5. app-integration-engineer (D): make_physics in server, SimLoop grip,
                                  telemetry readouts   (needs C's ckpt + A's physics)
6. reviewer (F) gates each merge; final full suite + ruff; PR with run summary,
   curves, calibration table, and the lap-time-vs-pole benchmark.
```

physics-engineer is the critical path (the main 3b role). env-engineer is sequential on the
grip-pipeline signatures. training and app integration follow the env and the checkpoint.
test-engineer runs concurrently from the contracts; reviewer gates throughout. **Every
subagent calls `/caveman` before starting (spec §5).**

---

## Definition of done (spec §2e, §3b)

- `DynamicBicycle` swaps in behind `PhysicsModel` with **no env API change**; `make_physics`
  builds it from `physics.model: dynamic`.
- The car drives on the dynamic physics and **loses grip when overdriven**; tyres wear and
  weather change the grip, visibly, in watch-live.
- The grip pipeline is **one scalar** = `mu_base · tire · weather · surface`, gating the
  friction circle; `tires.py` is pure.
- ObservationV2 (length 22, `OBS_VERSION = 2`) passes `check_env`; v1 checkpoints are refused.
- The agent **laps `red_bull_ring` cleanly** on dynamic physics, retrained from scratch under
  the curriculum.
- A **lap-time score vs the real pole (64.3 s)** is reported, **`2·pole` reached**, and the
  **gap to pole is closing** over training; the delta is logged.
- The car is **calibrated** so a clean optimal lap lands near the pole (calibration table in
  the PR).
- Resume works (checkpoint round-trips exactly; `--resume` continues the timestep count).
- Full test suite + `ruff` green. PR carries the run summary, the wandb curves, the SPS
  re-benchmark, the calibration table, and the lap-time-vs-pole benchmark.

---

## Risks & open items

- **`vx → 0` singularity** in the slip-angle `atan2` is the #1 dynamic-model bug — the
  `vx_safe` guard + low-speed kinematic blend are mandatory, and `test_physics_dynamic.py`
  must cover standstill and pit-crawl speeds. (spec §b: integrate with substeps for stability.)
- **Friction-circle correctness is the load-bearing realism** — keep the reduction explicit so
  the unit test can assert combined force ≤ `max_force` exactly. A wrong reduction reads as
  "car never loses grip" or "car is undriveable".
- **Reward re-shaping is the most-iterated work** (spec §4a) — keep `reward_v2` ≈ v1 by default
  (`w_slip=0`), all weights in config; a collapsing return → revert to the last good checkpoint
  and reshape, don't tune in logic.
- **Calibration before compute** — running `calibrate.py` first avoids training a fast policy
  on a car that physically can't approach the pole (or trivially beats it), which would make
  the benchmark meaningless (spec open question §4b).
- **Dynamic step is heavier than kinematic** — re-run the SPS benchmark; throughput, not the
  algorithm, is the limiter on 4 CPU cores (spec §3a). Device is a config change if it moves
  to cloud.
- **Grip provider must agree between env and `SimLoop`** — both import the *same*
  `Conditions.grip_at` (no second implementation), or train/serve grip skews and the live view
  lies about the policy. Mirror the existing shared-obs-builder discipline.
- **Server still builds `KinematicBicycle` directly** — easy to miss; the `make_physics`
  re-route in `app.py` is required for the dynamic watch-live, and is on the reviewer's list.
- **Pacejka deferred** — linear tires first (§2g, §17); upgrade only if the feel is too plain.
  The friction-circle interface is unchanged either way, so it's a `tires.py`-local swap later.
```