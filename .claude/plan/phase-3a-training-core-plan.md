# Phase 3a Implementation Plan — Training Core

Companion to `.claude/specs/phase-3a-training-core.md` (the spec). This is the **how**:
concrete, dependency-ordered, file-by-file build order grounded in the real Phase 1/2 code,
dispatched through the subagent roster in spec §5. Branch: `phase-3a-training-core`.

> Authoritative engineering doc remains `.claude/TECHNICAL_DESIGN.md` (§7 observations,
> §8 actions, §9 reward, §10 env contract, §11 rendering, §12 training). Where this plan
> fixes a contract the design leaves open (the exact ObservationV1 layout, the checkpoint
> sidecar), `TECHNICAL_DESIGN.md` is updated in the same commit — the decision and the doc
> move together (CLAUDE.md rule).

---

## Confirmed / assumed decisions (override spec open questions)

These resolve the spec's open questions with the obvious defaults so the build is unblocked.
All are config values and reversible; flagged where they cost a retrain.

1. **Circuit = `red_bull_ring`.** Already built and clean: 1431 samples, length_error
   0.55 %, `low_confidence: false`, `pole_time_s: 64.3`, `total_laps: 71`. Short, simple,
   the spec's own proposal. No new track work.
2. **Physics = kinematic only.** `KinematicBicycle` behind `PhysicsModel`. Dynamic model,
   grip pipeline, tires, weather are **Part 2** (`03b`). `grip` stays the constant `1.0`.
3. **Device = local CPU by default.** Installed torch is `2.12.0+cpu`, `cuda=False`, and the
   laptop has **4 cores**, so `SubprocVecEnv` is capped at ~4–8 envs. The SPS benchmark
   (built first under training-engineer) makes the local-vs-cloud call with data, per spec.
4. **ObservationV1 = fixed length 15**, `OBS_VERSION = 1`:
   `[speed_norm, heading_error, lateral_offset_norm, curvature_lookahead[5], edge_beam[7]]`.
   Changing this length later forces a retrain (expected, clean).
5. **Action = `Box(-1, +1, shape=(2,))`** — `[steer, longitudinal]`, exactly §8.
6. **Start on the real circuit** with start-state randomization. No oval warm-up unless
   learning stalls (then it's a one-line `track_id` change, the resume mechanism already
   exists).
7. **Algorithm = SB3 PPO + `VecNormalize`** (obs normalization), `MlpPolicy`. PPO per §2.
8. **Everything tunable lives in config** under a new `configs/experiment/<name>.yaml`:
   reward weights, PPO hyperparameters, obs parameters (`ref_speed`, lookahead distances,
   beam angles/range), env limits (`max_steps`, `target_laps`), eval/checkpoint cadence,
   wandb settings, device. No tuning constant in logic.

---

## Phase 1/2 baseline (verified in code — what we build on)

- `physics/base.py` — `CarState` (full dynamic struct, kinematic uses the subset),
  `PhysicsModel` Protocol: `step(state, steer, longitudinal, grip, dt) -> CarState`, pure.
- `physics/kinematic.py` — `KinematicBicycle`, `KinematicParams.from_config`. Pure step,
  `grip` ignored, no reverse (clamps `v>=0`). Spins/drifts naturally at speed → the
  "untrained car fails" story works on this model.
- `track/schema.py` — `Track`: `centerline`, `tangent`, `normal` (left), `s` (arc length),
  `curvature` (signed 1/m), `half_width_left/right`, kerb/grass/gravel bands,
  `nearest_index(x, y)` (brute force), `to_api_dict`, `save_npz`/`from_npz`.
- `track/geometry.py` — `arc_length`, `frames`, `signed_curvature`, `derive_geometry`
  (seam-safe). Reuse for any lookahead resampling math.
- `track/loader.py` — `load_track(id)`, `list_tracks`. `red_bull_ring.npz` present.
- `sim/loop.py` — `SimLoop` owns car+physics+`LapTimer`, 5×0.01 s substeps, emits a JSON
  state frame, never renders. The env replicates this stepping internally (does **not**
  import `SimLoop` — it owns its own loop to stay rendering-free and recorder-optional).
- `sim/timing.py` — `LapTimer`: arc-length lap detection (wrap `>0.7L → <0.3L`),
  best/last lap, delta-to-pole. Env reuses the same wrap logic for lap-success termination.
- `sim/recorder.py` — `TrajectoryRecorder` + `load_trajectory`: the shared JSON interchange
  for live sim, replay, and eval clips. The env's eval path and the cloud clip renderer both
  emit this exact format.
- `sim/autopilot.py` — `CenterlineAutopilot.control(state) -> (steer, longitudinal)`. The
  live-policy pilot mirrors this interface so the server drops it into the same slot.
- `server/app.py` — per-`/ws/sim`-session `SimLoop` + autopilot; `mode` `manual|watch`;
  `send_loop` calls `sim.autopilot.control(state)` in `watch`. `TrackMessage` switches
  circuits live. **No policy/checkpoint path yet.**
- `utils/seeding.py` — `seed_everything` (Python + NumPy + torch). `utils/config.py` —
  `load_config` merges `configs/track/<id>.yaml` under `cfg.track`; supports dotlist
  overrides and `load_track_config`.
- `pyproject.toml` — SB3, torch, gymnasium, wandb, omegaconf already core deps. Installed:
  SB3 2.9.0, gymnasium 1.3.0, torch 2.12 CPU, 4 cores.

**Gaps to create:** `src/f1rl/env/`, `src/f1rl/train/`, `src/f1rl/render/`,
`configs/experiment/`, `.claude/agents/`, the new tests.

---

## Contracts fixed before any code (foundation)

### ObservationV1 (`OBS_VERSION = 1`, `OBS_DIM = 15`)

| idx | field | definition | normalization |
|----|----|----|----|
| 0 | `speed_norm` | `vx / ref_speed` | `ref_speed` from config (~92 m/s top speed) |
| 1 | `heading_error` | `wrap(yaw − atan2(tangent_y, tangent_x))` at nearest sample | `/ π` |
| 2 | `lateral_offset_norm` | `((pos − center) · left_normal) / half_width_on_that_side` | signed, **input only, never rewarded toward 0** |
| 3–7 | `curvature_lookahead[5]` | signed curvature sampled at `s + {10,25,50,100,150} m` (wrap on closed loop) | `× curvature_scale` |
| 8–14 | `edge_beam[7]` | rangefinder distance car→asphalt edge at angles `{−90,−60,−30,0,30,60,90}°` rel. to `yaw` | `/ beam_max`, clipped `[0,1]` |

`observation_space = Box(low, high, shape=(15,))` with generous bounds; the builder clips
so `env_checker` never sees an out-of-space value.

### Action (`Box(-1, +1, shape=(2,))`)
`action[0]→steer·max_steer`, `action[1]≥0→throttle`, `<0→brake` (matches `KinematicBicycle`).

### Checkpoint (resume-exact)
SB3 `model.zip` (weights + optimizer + torch RNG) **+** `vecnormalize.pkl` (obs stats) **+**
`meta.json` sidecar: `{total_timesteps, circuit_id, obs_version, seed, config_snapshot,
sb3_version, numpy_rng_state}`. The loader **refuses** a mismatched `obs_version` or action
shape with a clear message before resuming (spec §Data-Model validation).

---

## Build order (dependency-first), mapped to subagents

### Step 0 — scaffold (main thread)
- Create branch `phase-3a-training-core`.
- Write the 6 agent definitions under `.claude/agents/` from the briefs in the section
  below. Each gets a narrow scope + its own context (token saving, unbiased tests).
- Add empty packages: `src/f1rl/env/__init__.py`, `src/f1rl/train/__init__.py`,
  `src/f1rl/render/__init__.py`, `configs/experiment/`.

### Step A — env-engineer · `src/f1rl/env/` (passes the Gymnasium checker)
Dependency root for everything else. Files:

- **`observations.py`** — *pure NumPy, no gym, no torch* (so the server reuses it verbatim).
  - `OBS_VERSION = 1`, `OBS_DIM = 15`, `observation_space()` → `Box`.
  - `ObsParams.from_config` (`ref_speed`, `lookahead_m=[10,25,50,100,150]`,
    `beam_angles_deg`, `beam_max`, `curvature_scale`).
  - `track_query(track, x, y, yaw)` → `(nearest_idx, s_along, signed_lateral, half_width,
    heading_error)` — shared by reward + server (one nearest-point projection, not duplicated).
  - `sample_curvature_ahead(track, s, lookahead_m)` — interp on `track.s`/`track.curvature`,
    wrap for closed loop (reuse arc-length conventions from `geometry.py`).
  - `cast_beams(track, x, y, yaw, angles, beam_max)` — ray vs the two asphalt-edge polylines
    (`centerline ± normal·half_width`); precompute the edge polylines once per track and
    cache. Vectorized over beams.
  - `build_observation(track, state, params, edge_cache) -> np.ndarray(15,)` (clipped).
- **`rewards.py`** — `RewardWeights.from_config` (`w_progress, w_offtrack, w_step,
  w_reverse`, off-track shape params). `reward_v1(prev_s, cur_s, off_track_m, length, ...)
  -> (float, terms: dict)`. `ds` = signed arc-length progress (wrap-aware); graded
  `offtrack_penalty(off_track_m)`; `−w_step`; `−w_reverse·max(0,−ds)`. Returns term breakdown
  for logging. **Never** rewards centerline proximity.
- **`conditions.py`** — minimal `Conditions` holder (dry, `grip=1.0`) so Part 2 expands it
  without an env signature change. Keep tiny.
- **`single_agent.py`** — `RacingEnv(gymnasium.Env)`:
  - `__init__(track, physics, sim params, obs_params, reward_weights, target_laps,
    max_steps, start_randomization, rng)` — all config-driven.
  - `reset(seed)`: seed via `seed_everything`-compatible RNG; **start-state randomization** —
    random centerline index, place on centerline, heading = tangent, low/zero start speed;
    init lap tracker; return `(obs, info)`.
  - `step(action)`: map action → controls; run `substeps` physics substeps; recompute
    `s_along` + `off_track_m` (meters past asphalt edge, 0 on asphalt); `reward_v1`;
    **termination** — success on `completed_laps == target_laps`, failure on large off-track
    or wrong-way (sustained negative progress / heading reversal), **truncation** on
    `max_steps`; build obs; `info` carries `lap_time`, `off_track`, `progress`, term
    breakdown. **Never renders.**
  - Optional `recorder` hook: when set, append the shared-format frame each step (eval only).
- **`factory.py`** — `make_env(cfg, seed, rank)` (single `RacingEnv` for `check_env`/eval)
  and `make_vec_env(cfg, n_envs, seed)` → `SubprocVecEnv` wrapped in `VecNormalize`
  (obs-norm on, reward-norm per config). Per-env seed = `base_seed + rank`. This is the seam
  training-engineer and the test author build against.

**Gate:** `gymnasium.utils.env_checker.check_env(RacingEnv(...))` passes; a random-policy
smoke loop runs N steps without error.

### Step B — physics-engineer · `src/f1rl/physics/` (light, parallel with A)
- Confirm `KinematicBicycle` satisfies `PhysicsModel` cleanly and `grip` is threaded
  (ignored now) so the Part 2 dynamic model swaps in with **no env change**.
- Add `physics/factory.py` → `make_physics(cfg)` selecting on `cfg.physics.model`
  (`"kinematic"` now; `"dynamic"` reserved for Part 2). Env + sim + server construct physics
  through this one factory.
- Keep `tests/test_physics_kinematic.py` green.

**Gate:** interface-clean confirmation; physics tests pass.

### Step C — training-engineer · `src/f1rl/train/` + `src/f1rl/render/` + `configs/experiment/`
Depends on Step A's `factory.py`. **Build the benchmark first** (spec: run it early).

- **`train/benchmark.py`** (also wired as `scripts/benchmark_sps.py`) — measure steps/sec for
  `n_envs ∈ {1,2,4,8}` on this machine; print an SPS table. **Run before tuning the budget**
  to make the local-vs-cloud decision with data, and record it in the PR.
- **`configs/experiment/rbr_ppo.yaml`** — extends `default`; sets `track_id: red_bull_ring`,
  `n_envs`, `total_timesteps` (start 2–5 M), PPO hyperparameters (`n_steps, batch_size,
  gamma, gae_lambda, ent_coef, vf_coef, learning_rate, clip_range, n_epochs`), obs params,
  reward weights, env limits, eval/checkpoint cadence, `wandb`, `device: cpu`.
- **`train/checkpointing.py`** — `save_checkpoint` / `load_checkpoint` (model + VecNormalize
  + meta sidecar); `validate_checkpoint(meta)` enforces `obs_version`/action shape, clear
  error on mismatch. **Reused by the server** (Step D) — single source for the format.
- **`train/callbacks.py`** —
  - `CheckpointCallback`: every `checkpoint_freq` steps, atomic save (model + vecnorm +
    meta); keep last K + best.
  - `EvalVideoCallback`: every `eval_freq`, run one **deterministic** episode with a
    `TrajectoryRecorder`, render an mp4 via `render/renderer.py`, log clip + metrics
    (episode return, lap time vs pole 64.3 s, off-track count, steps-to-first-clean-lap) to
    wandb.
  - Lap-time logging as a plain metric (pole benchmark proper is Part 2).
- **`train/wandb_logger.py`** — wandb init with **offline/local-CSV fallback**
  (`WANDB_MODE=offline`) so an outage never loses curves.
- **`train/train.py`** — load config + dotlist overrides; `seed_everything`; build VecEnv via
  `env.factory`; build PPO (`MlpPolicy`, hyperparams + `device` from config); attach
  callbacks; `model.learn(total_timesteps)`. **`--resume <path>`** loads model + VecNormalize
  + timestep count and continues (`reset_num_timesteps=False`). Device-agnostic: identical
  local and cloud, same checkpoints both ways.
- **`train/evaluate.py`** — load a checkpoint, run deterministic episodes, report metrics,
  optionally emit a trajectory JSON / mp4. Shared by the eval callback and a CLI.
- **`render/renderer.py`** — offscreen Pygame (`SDL_VIDEODRIVER=dummy`) → mp4 via imageio,
  drawing the asphalt ribbon (`centerline ± half_width`) + oriented car glyph from a recorded
  trajectory. **Training-only; never imported by `env/` or the training hot path.**

**Gate:** a tiny-budget smoke run shows reward trending up; a checkpoint resumes and
continues the timestep count; one eval mp4 is logged.

### Step D — app-integration-engineer · server + web (live checkpoint viewing)
Depends on the checkpoint format (C) and `env/observations.py` (A).

- **`sim/policy_pilot.py`** (new, beside `autopilot.py`) — `PolicyPilot(checkpoint_path)`:
  load model + VecNormalize stats + meta (validate `obs_version`/action shape); each step
  build ObservationV1 from the live `CarState` via **the same `env/observations.py`**,
  normalize with the saved stats, `model.predict(deterministic=True)` →
  `control(state) -> (steer, longitudinal)`. Same interface as `CenterlineAutopilot`, so it
  drops into the server's existing slot.
- **`server/messages.py`** — `PolicyMessage` (`type:"policy"`, `source:"autopilot"|
  "checkpoint"`, `id`) + parser.
- **`server/app.py`** — `GET /api/checkpoints` (scan the checkpoints dir); on `PolicyMessage`
  set `sim.autopilot = PolicyPilot(...)` or back to `CenterlineAutopilot`. Bad/missing
  checkpoint or obs mismatch → `event` message + fall back to autopilot, **never crash**.
  `send_loop` is unchanged (still calls `.control(state)`).
- **`web/src/`** — watch-mode checkpoint picker (dropdown from `/api/checkpoints`) sending
  `PolicyMessage`; reuse the debug overlay (centerline + beams) to *see* the policy; pick an
  early vs late checkpoint to watch improvement. `types.ts` additions.

**Gate:** watch-live runs a chosen checkpoint; switching checkpoints shows the agent improve.

### Step E — test-engineer (independent) · `tests/`
Writes from the spec contracts and public signatures **only** — not implementation
internals — so tests verify requirements unbiased. Maps to spec §c:

- `test_env_api.py` — `check_env(RacingEnv)` passes; obs ∈ space; action space shape/bounds.
- `test_observations.py` — obs shape 15 + in-bounds; straight section `heading_error≈0`;
  lateral-offset **sign** correct each side; beams ∈ `[0,1]`; curvature lookahead picks the
  upcoming corner's sign.
- `test_rewards.py` — reward rises with forward progress; penalized off track; reverse
  penalized; on-asphalt `off=0`; centerline proximity never rewarded.
- `test_termination.py` — large off-track terminates; wrong-way terminates; lap-count success
  terminates; step limit truncates; lap detection fires once per lap.
- `test_checkpoint.py` — save→load round-trips weights + vecnorm + timestep **exactly**;
  `obs_version` mismatch refused with a clear error.
- `test_seeding.py` — fixed seed → identical rollout.
- `test_smoke_train.py` (integration, tiny budget) — `learn()` runs without crashing, reward
  trend non-degenerate, checkpoint resumes and continues the timestep count.

### Step F — reviewer · gate every diff (read-only)
Checklist per task: config-driven (no magic constants in logic), SI units, **no rendering in
`env.step` or the training hot path**, env passes the checker, `obs_version` stable, physics
behind the interface, FastF1 **not** imported under `train/` or `env/`, seed recorded,
checkpoint round-trips. Runs `pytest` + `ruff check` + `ruff format --check`. Pass/fail with
reasons; gates each merge.

---

## Dispatch DAG (dependency order)

```
0. scaffold (branch, agent defs, empty packages)
1. physics-engineer (B, quick)  ──┐         test-engineer (E) starts in parallel,
   env-engineer (A) ──────────────┤            writing failing tests from the spec
2.                          env factory ready
3. training-engineer (C): benchmark FIRST → device call → train/callbacks/eval/render → smoke
4. app-integration-engineer (D): PolicyPilot + server + web   (needs C's checkpoint + A's obs)
5. reviewer (F) gates each merge; final full suite + ruff; PR with run summary + curves
```

env-engineer is the critical path. test-engineer runs concurrently from contracts.
physics-engineer is a short confirm. training and app integration are sequential on the
checkpoint format. reviewer gates throughout.

---

## Definition of done (spec §3b + §2c)

- `RacingEnv` passes `gymnasium.utils.env_checker.check_env`.
- The agent laps `red_bull_ring` **cleanly** on kinematic physics.
- Improvement is **visible across checkpoints** in the app's watch-live mode.
- **Resume works** (checkpoint round-trips exactly; `--resume` continues the timestep count).
- SPS benchmark recorded; lap time logged against pole (64.3 s) and 2× pole.
- Full test suite + ruff green. PR carries the run summary and the wandb curves.

---

## Risks & open items

- **4 cores, CPU torch** → throughput is the real limiter; the benchmark (Step C, first)
  decides whether long runs move to cloud. Env API + checkpoint schema are stable, so the
  device is a config change (spec §g migration).
- **Reward shaping is the most-iterated work** — expect several passes; all weights stay in
  `configs/experiment/`. A collapsing return → revert to the last good checkpoint and reshape.
- **VecNormalize stats must travel with the checkpoint** — otherwise eval and live inference
  mismatch the training distribution. Enforced by the checkpoint format + `PolicyPilot` reuse
  of the saved stats.
- **Beam-cast cost** in `env.step` — precompute per-track edge polylines once and vectorize
  over beams to keep steps fast.
- **Shared obs builder** (`env/observations.py`) is imported by both training and the server;
  keep it pure (no gym/torch) so the server never pulls the training stack into the hot path.
- If learning stalls on the real circuit, the spec's oval warm-up is a one-line `track_id`
  switch resumed from the same checkpoint — built-in, not a rewrite.
