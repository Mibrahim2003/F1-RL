# Phase 5 Implementation Plan — Many Cars on Track

Companion to `.claude/specs/phase-5-many-cars.md` (the spec). This is the **how**: concrete,
dependency-ordered, file-by-file build order grounded in the real Phase 1–4 code, dispatched
through the subagent roster in spec §5. Branch: `phase-5-many-cars` (cut from `main` after Phase 4
merges).

> Authoritative engineering doc remains `.claude/TECHNICAL_DESIGN.md` (§7 observations —
> "local, relative features only … relative position and velocity of the K nearest cars (added in
> later phases)"; §10 env contract — `RacingParallelEnv`, shared-policy self-play, scale 2→4→22,
> SuperSuit; §13 repo layout — planned `env/multi_agent.py`, `train/selfplay.py`; §15 build order —
> Phase 5 = field on track, **no racing rules yet**). Where this plan fixes a contract the design
> leaves open (the `grid:` config block, the per-agent dict contract, the black-death/constant-width
> mechanism, the per-run field-size step-up, the multi-car live-frame and recorder schema, the
> `n_agents` checkpoint field), update `TECHNICAL_DESIGN.md` in the **same commit** — the decision
> and the doc move together (CLAUDE.md rule).

The headline: keep the Phase 4 policy, physics, observation, and reward **unchanged**; factor the
single-car update out of `RacingEnv` into a reusable per-car unit; build a `RacingParallelEnv`
(PettingZoo `ParallelEnv`) that holds **N homogeneous cars on one shared circuit**, each with its
**own** car state and **own `LapTimer`**, sharing only the pool entry's read-only `Track` /
`EdgeCache` / pole; train **one shared policy** through SuperSuit → Stable-Baselines3 PPO with
parameter sharing; **warm-start the Phase 4 generalist** (obs is unchanged → legal); grow the field
**2 → 4 → 22 across successive warm-started runs** (field size is a per-run constant, not a
mid-run curriculum knob); and **render the whole field** in the app. **No nearby-car observation,
no collisions, no racing rewards** — those are Phase 6.

---

## Confirmed / assumed decisions (resolve the spec open questions)

All are config values or per-run constants, reversible; none costs a forced retrain (the
observation is unchanged, `OBS_VERSION` stays 2).

1. **Observation = ObservationV2, UNCHANGED and locked.** Length 22, `OBS_VERSION = 2`. No
   nearby-car block this phase (design §7 reserves it for "later phases" → Phase 6). Each car
   observes only the track, exactly as Phase 4. Because the version does not move,
   `validate_checkpoint` accepts the Phase 4 generalist → **warm start is legal**. A test asserts
   the per-agent obs equals the single-agent obs for the same `CarState` on the same circuit.

2. **Physics = unchanged.** The dynamic bicycle model + grip pipeline from Phase 3b are reused
   as-is behind `PhysicsModel`, once per car. No `physics/` work this phase; no change to the
   `PhysicsModel.step` signature.

3. **The multi-agent stack is already installed — the gating task is PIN + verify, not "resolve
   from scratch."** `pyproject.toml:22-24` already lists `pettingzoo`, `supersuit`,
   `stable-baselines3` (unpinned). The venv currently has a **working** matrix:
   `pettingzoo 1.26.1`, `SuperSuit 3.11.0`, `stable_baselines3 2.9.0`, `gymnasium 1.3.0`,
   `torch 2.12.0`, and `from pettingzoo.test import parallel_api_test` /
   `from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1, black_death_v3` /
   `from stable_baselines3 import PPO` all import cleanly. So the dependency-matrix step **pins
   these exact versions** in `pyproject.toml` and proves a SuperSuit round-trip on a trivial
   `ParallelEnv`. (Spec §1g/§2b say PettingZoo/SuperSuit are "not yet in `pyproject.toml`" — that
   is stale; they are present but unpinned. Update that spec line in the same change.)

4. **Constant SuperSuit-visible width via `black_death_v3`; the raw env follows standard
   PettingZoo.** SuperSuit's `pettingzoo_env_to_vec_env_v1` requires a **fixed agent set** across
   steps. The clean, standard way to get that with per-car early death: the **raw
   `RacingParallelEnv` removes a terminated/truncated car from `self.agents` on the next step**
   (so it passes `parallel_api_test`), and **`black_death_v3` wraps it in the training stack to
   re-pad removed agents to zero** obs/reward, giving the vectorizer constant width. The
   constant-width guarantee lives in the SuperSuit wrapper, not in the raw env. This implements the
   spec's chosen "constant-width + `black_death_v3`" decision concretely; read the spec's "never
   removed from `self.agents`" as **"the SuperSuit-visible width is constant"** and align that
   wording in the same change. One car finishing or failing never ends the others; the episode ends
   when all cars are done or the step limit fires.

5. **Per-car `LapTimer` instances; the pool entry is read-only.** The Phase 4 `CircuitEntry` holds
   **one** `LapTimer` (`pool.py:129`, `lap_timer=LapTimer(track, pole)`), and `LapTimer` is
   stateful (`completed_laps`, `lap_start_t`, `_prev_s`). N cars on one circuit map to one entry,
   so reusing `entry.lap_timer` would have every car stomp one shared lap state. The parallel env
   **constructs one `LapTimer(track, pole)` per car** (cheap) and reuses the entry only for its
   read-only `Track`, `EdgeCache`, and resolved pole. `entry.lap_timer` is **not** reused per car.
   The same trap applies to every per-step projection field — `_prev_s`, `_grip_idx`, `_grip_lat`,
   `_wrong_way_count`, and `CarState` (`single_agent.py:181-186`, re-seeded in `reset` at
   `:220-222`): these are **car state**, carried per car by the factored step, never read off the
   env.

6. **Field size is a per-run constant, stepped across warm-started runs — NOT a curriculum knob.**
   `n_agents` sets the vector-env **width**. The Phase 4 curriculum mechanism is in-place
   (`CurriculumCallback._maybe_apply` pushes via `training_env.env_method(...)`,
   `curriculum.py:102-111`) and cannot change vec-env width without a full rebuild that would break
   the PPO rollout buffer / `n_steps`. So the field grows **2 → 4 → … → 22 by launching successive
   runs**, each a fixed `n_agents`, each warm-started from the prior smaller-field run (obs/action
   spaces match → SB3 `set_env` accepts a different width). The **circuit-pool widening stays an
   in-place curriculum knob** (Phase 4 `set_track_pool` / `CurriculumStage.circuits`, unchanged).
   `n_agents` is **not** a `CurriculumStage` field.

7. **Two reset modes, both seeded and config-selectable.** `reset_mode = scattered` (training:
   a seeded random centerline index per car — reuse the existing `_sample_start_index` /
   `start_randomize` path, `single_agent.py:410-416 — for state-space coverage, the field smeared
   around the lap) and `reset_mode = grid` (eval/demo: seeded **distinct, non-overlapping** slots
   and headings near the start/finish line). Do not promise a tidy grid on the training reset.

8. **Circuit pool reused unchanged, per episode, shared by the whole field.** The grid draws **one**
   circuit per episode from the Phase 4 `CircuitPool` (`pool.sample(self.np_random)`); every car in
   the episode shares that entry's read-only `Track` / `EdgeCache` / pole (per-car lap timers, per
   #5). Per-worker independent draw, reproducible from the seed.

9. **New experiment config `configs/experiment/calendar_selfplay.yaml`**, extending
   `calendar_dynamic.yaml`: adds the `grid:` block (`n_agents`, `reset_mode`, `grid_spacing_m`,
   `team_colors`), the warm-start checkpoint path, and the self-play vectorization knobs
   (`n_copies` SuperSuit copies); keeps the Phase 4 `physics` / `tires` / `weather` / `reward` /
   `obs` / `curriculum` (pool-widening) / PPO blocks. Bump `wandb.group`/`tags` to `phase-5`. No
   tuning constant in logic.

10. **Device = local CPU by default. Throughput is the wall of the phase.** SuperSuit steps all N
    cars **sequentially inside one process** (the `ParallelEnv.step` loops the agents), so a
    process's step cost scales ~linearly with N on the slow dynamic model and is **not** amortized
    by process fan-out. Get a **steps-per-second estimate before committing to field sizes** (the §2
    throughput check): SPS of `n_copies × N`-car SuperSuit vs an equal-core single-agent `n_envs`
    run. This sizes the field ceiling and the laptop-vs-cloud line.

11. **Live frame: a `cars` array, backward compatible.** The WebSocket frame carries
    `cars: [{id, x, y, yaw, speed, telemetry}, …]` instead of the single `car` object
    (`sim/loop.py:112-138`). A single car is a one-element `cars` array, so the Phase 1/4 one-car
    path keeps working. Gap in the timing tower is **track-position based** (arc-length `s`,
    rendered as distance or time-behind-leader by progress) — not a race gap (Phase 6).

12. **One multi-car recorder.** Extend the trajectory recorder (the env `_record`,
    `single_agent.py:418-433`, and `sim/loop.py`) to a single recorder writing per-car entries
    under a `cars` key per frame (matching the live frame); the replay player loads and scrubs the
    multi-car trajectory. Not one recorder per car.

---

## Phase 1–4 baseline (verified in code — what we build on)

- `env/single_agent.py` — `RacingEnv(gym.Env)`. The **single-car update we factor**:
  - `__init__` (`:114-187`) builds the `CircuitPool` once (`:144`), binds an initial circuit via
    `_bind_circuit` (`:153`, `:308-322` — sets `self.track`/`track_id`/`edge_cache`/`lap_timer`/
    `_pole` from `entry`), and initializes the per-step projection state `_prev_s`/`_wrong_way_count`/
    `_grip_idx`/`_grip_lat` (`:181-186`).
  - `reset` (`:190-236`) draws a circuit (`pool.sample`, `:198`), rebinds, resolves weather, samples
    a start index (`:202`, `_sample_start_index` `:410-416`), builds `CarState`, **resets the lap
    timer** (`:217`) and **re-seeds the projection state** (`:220-222`), returns
    `info["circuit_id"]`/`["pole_time_s"]`.
  - `step` (`:238-304`) maps the action, runs `substeps` physics steps, **recomputes the projection
    once** (`track_query`, `:252`) shared by reward + off-track + obs + next grip, updates the lap
    timer (`:259`), computes `reward_v1`/`reward_v2` (`:261-269`), tracks wrong-way (`:273-276`),
    checks termination (`:278`, `_check_termination` `:387-397`), builds obs, returns the 5-tuple.
    **This entire body is the per-car unit to extract** — it reads only `self.track`,
    `self.edge_cache`, `self.lap_timer`, `self._pole`, and the per-car projection/`CarState`.
  - `apply_conditions` (`:335-357`) and `set_track_pool` (`:324-331`) are the **in-place curriculum
    hooks** called on workers via `VecEnv.env_method` — reuse them unchanged for the pool widening;
    do **not** add a field-size hook here (decision #6).
- `env/pool.py` — `CircuitPool` / `CircuitEntry` (`:84-185`). `CircuitEntry.lap_timer` is the
  **per-entry** timer (`:89`, built `:129`) the parallel env must **not** reuse per car (#5). Reuse
  `entry.track` / `entry.edge_cache` / `entry.pole_time_s` (read-only), `pool.sample` (`:168-178`),
  `pool.set_active` (`:149-166`), `resolve_pole` (`:40-69`), `pool_ids_from_config` (`:72-81`)
  unchanged.
- `env/observations.py` — `OBS_VERSION = 2`, `OBS_DIM = 22`, `build_observation` / `track_query` /
  `observation_space()` / `build_edge_cache`. Local/relative only — **the property to lock, not
  change**. The per-agent `observation_space(agent)` returns this same Box.
- `env/factory.py` — `make_env` (`:23-33`), `make_vec_env` (`:46-80`, `SubprocVecEnv` +
  `VecNormalize`, per-worker seed `seed + rank`). The **single-agent** path; Phase 5 adds a
  **parallel** builder alongside it (SuperSuit), leaving this untouched.
- `train/curriculum.py` — `CurriculumStage` (`:21-31`, already carries `circuits: tuple | None`),
  `CurriculumCallback._maybe_apply` (`:93-127`, pushes `apply_conditions` + `set_track_pool` via
  `env_method`, `:102-111`). **In-place, sampling-side, no rebuild** — reuse for pool widening;
  never for field size.
- `train/train.py` — config-driven PPO entry. Builds the venv via `make_vec_env` (`:173`); resume
  path is `load_checkpoint(resume, env=venv)` + `model.set_env(venv)` + `reset_num_timesteps=False`
  (`:179-187`); callbacks assembled in `_make_callbacks` (`:101-137`). The self-play entry mirrors
  this against the SuperSuit venv.
- `train/checkpointing.py` — `meta.json` records `obs_version` (stays 2) and `circuit_id`;
  `validate_checkpoint` refuses an obs/action mismatch (per-agent spaces unchanged → warm start
  passes). Add an `n_agents` meta field (the constant field size the run trained on).
- `sim/loop.py` — `SimLoop` drives **one** car; `_frame` (`:112-138`) returns a single `car`
  object. Phase 5 extends the live path to a `cars` array (#11).
- `server/app.py` / `web/` — backend + frontend, one car, one timing row today. Phase 5 streams the
  field and lists every car.

**Gaps to create:** the per-car step factoring in `single_agent.py`; the new `env/multi_agent.py`
(`RacingParallelEnv`); a parallel vec-env builder (SuperSuit) in `env/factory.py`; the new
`train/selfplay.py` + a **multi-car offscreen eval driver** (a `ParallelEnv` is not a Gym env, so
`evaluate.py` cannot drive it); `configs/experiment/calendar_selfplay.yaml` + a `grid:` block in
`configs/default.yaml`; the multi-car live frame + one multi-car recorder + multi-car replay in
`sim/`+`server/`+`web/`; the new tests; pinned deps in `pyproject.toml`; and the `.claude/agents/`
roster for Phase 5.

---

## Contracts fixed before any code (foundation)

### Dependency matrix (`pyproject.toml`, gating)

Pin the verified working matrix and prove the round-trip **before any env code**:

```toml
# core deps (already present, now pinned)
"pettingzoo==1.26.1",
"supersuit==3.11.0",
"stable-baselines3==2.9.0",
# gymnasium / torch already resolved transitively; pin if the install drifts
```

Smoke gate (must pass): `from pettingzoo.test import parallel_api_test`,
`from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1, black_death_v3`, and a
**SuperSuit round-trip on a trivial 2-agent `ParallelEnv`** (`black_death_v3` →
`pettingzoo_env_to_vec_env_v1` → `concat_vec_envs_v1(..., num_vec_envs=2)` → one `PPO.learn` of a
few hundred steps without error). No feature code starts until this is green and pinned.

### Per-car step factoring (`env/single_agent.py`)

Extract the body of `RacingEnv.step` (`:238-304`) and the per-car parts of `reset` into a reusable
unit keyed by **per-car state**, so `RacingEnv` and `RacingParallelEnv` call one implementation:

```
CarRuntime (per car):                 # NOT pooled, NOT env-level
  state: CarState
  lap_timer: LapTimer                 # its OWN instance (LapTimer(track, pole)), never entry.lap_timer
  prev_s, grip_idx, grip_lat: float
  wrong_way_count: int
  done: bool                          # frozen after termination (black_death pads it)

step_one_car(track, edge_cache, pole, physics, sim, limits, reward_weights,
             conditions, obs_params, car: CarRuntime, action) -> (obs, reward, terminated, truncated, info)
  # identical math to RacingEnv.step today, reading car.* and the read-only circuit args
```

`RacingEnv` becomes a thin one-car wrapper over `step_one_car` (its contract, obs, action space,
and `check_env` pass-through unchanged — verify with the existing `test_env_api.py`). No
`reset`/`step` signature change on `RacingEnv`; a one-car field reproduces it.

### `RacingParallelEnv` (`env/multi_agent.py`, new)

```
RacingParallelEnv(pettingzoo.ParallelEnv):
  __init__(cfg, n_agents, seed):
    pool = CircuitPool(...)           # built once, as Phase 4 (read-only Track/EdgeCache/pole)
    physics/sim/limits/reward/obs/conditions params built once (shared, read-only)
    possible_agents = [f"car_{i}" for i in range(n_agents)]
    one CarRuntime per agent, each with its OWN LapTimer
    grid_cfg = GridParams.from_config(cfg)   # n_agents, reset_mode, grid_spacing_m, team_colors

  observation_space(agent) -> observation_space()      # the unchanged length-22 Box, same for all
  action_space(agent)      -> Box(-1,1,(2,))           # unchanged, same for all

  reset(seed) -> (obs:{agent:vec}, info:{agent:{circuit_id, pole_time_s, team}}):
    self.agents = list(self.possible_agents)
    cid = pool.sample(self.np_random); bind read-only track/edge_cache/pole for the field
    place cars by reset_mode (scattered = per-car _sample_start_index; grid = distinct slots @ S/F)
    reset EACH car's OWN lap_timer + projection state

  step(actions:{agent:a}) -> (obs, rewards, terminations, truncations, infos):   # per-agent dicts
    for agent in self.agents: step_one_car(read-only circuit, car[agent], actions[agent])
    a car that terminates/truncates is REMOVED from self.agents on the next step (PettingZoo std)
    episode ends when self.agents is empty or the step limit fires

  passes pettingzoo.test.parallel_api_test
```

The **read-only circuit** (Track/EdgeCache/pole) is shared by the field; **everything mutable is
per car** (decision #5). No nearby-car data enters any obs; no collision is computed.

### Self-play vectorization (`env/factory.py` + `train/selfplay.py`, new)

Parallel builder alongside `make_vec_env`, leaving the single-agent path untouched:

```
make_selfplay_vec_env(cfg, n_agents, n_copies, seed) -> VecNormalize:
  thunk: RacingParallelEnv(cfg, n_agents, seed+rank)
       → black_death_v3(env)                              # constant SuperSuit-visible width
       → pettingzoo_env_to_vec_env_v1(env)
  concat_vec_envs_v1(vec, num_vec_envs=n_copies, num_cpus=..., base_class="stable_baselines3")
       → VecMonitor → VecNormalize(norm_obs=True, norm_reward=cfg.env.norm_reward, clip/gamma from cfg)
```

`train/selfplay.py` mirrors `train.py`: load `calendar_selfplay.yaml`, `seed_everything`, build the
SuperSuit venv, **warm-start the Phase 4 generalist** (`load_checkpoint(resume, env=venv)` +
`set_env` — obs/action spaces match across field widths, so a 2-car checkpoint loads into a 4-car
run), attach the checkpoint + curriculum (pool-widening only) callbacks + the multi-car eval driver,
`model.learn(reset_num_timesteps=False)`. Field size is fixed per run (`grid.n_agents`); the
**cross-run step-up** is a documented convention (`--resume` the prior smaller-field run), not code.

### Multi-car live frame + recorder (`sim/`, `server/`, `web/`)

- `sim/loop.py` `_frame` (`:112-138`) emits `cars: [{id, x, y, yaw, speed, telemetry}, …]`; a single
  car is a one-element array (backward compatible, decision #11).
- The recorder writes one trajectory with per-car entries under a `cars` key per frame (decision
  #12); the replay player scrubs it.
- The timing tower lists every car with lap time and **track-position gap** (arc-length `s`).

### Checkpoint (near-unchanged format)

`obs_version` stays 2 → `validate_checkpoint` accepts the Phase 4 generalist (warm start legal). Add
`n_agents` to `meta.json` (records the constant field size). Round-trip (weights, optimizer,
vecnorm, timestep, RNG) unchanged.

### Observation lock (no change, test only)

`test_observations.py` / the multi-agent test assert: the per-agent obs equals the single-agent obs
for the same `CarState` on the same circuit; no nearby-car data; `OBS_DIM == 22`,
`OBS_VERSION == 2`. This **locks** the property the warm start relies on.

---

## Build order (dependency-first), mapped to subagents

### Step 0 — scaffold (main thread)
- Create branch `phase-5-many-cars` from `main` (after Phase 4 merges).
- Create the `.claude/agents/` roster for Phase 5 (dependency-matrix, multiagent-env-engineer,
  selfplay-training-engineer, app-integration-engineer, test-engineer, reviewer — **no
  physics-engineer**). `/caveman` is **opt-in** per agent (spec §5), not a forced pre-call. Point
  each at this plan + the spec + `TECHNICAL_DESIGN.md` §7/§10/§13/§15.
- Add the `grid:` block to `configs/default.yaml` (safe default `n_agents: 1`, `reset_mode:
  scattered` ⇒ single-car fallback) and create `configs/experiment/calendar_selfplay.yaml`
  extending `calendar_dynamic.yaml`.

### Step A — dependency-matrix (first gate, may be the main thread)
Pin `pettingzoo==1.26.1` / `supersuit==3.11.0` / `stable-baselines3==2.9.0` in `pyproject.toml`;
run the SuperSuit round-trip smoke gate (above). **No env code starts until green.** Also correct
the spec lines that say PettingZoo/SuperSuit are "not yet in `pyproject.toml`."

**Gate:** the round-trip + `parallel_api_test` on a trivial env pass; versions pinned.

### Step B — multiagent-env-engineer · `src/f1rl/env/` (critical path, the main Phase 5 role)
Factor the per-car step; build `RacingParallelEnv`; add the SuperSuit vec builder.
- **`single_agent.py`** — extract `step_one_car` + `CarRuntime`; make `RacingEnv` a thin wrapper;
  no contract change.
- **`multi_agent.py`** (new) — `RacingParallelEnv` (contract above): N cars, **per-car `LapTimer`**,
  read-only shared circuit, two reset modes, standard PettingZoo per-car removal, per-agent dicts.
- **`factory.py`** — add `make_selfplay_vec_env` (black_death_v3 → vec → concat → VecNormalize);
  leave `make_vec_env` untouched.

**Gate:** `parallel_api_test(RacingParallelEnv(...))` passes; per-agent obs ∈ the length-22 Box and
equals the single-agent obs for the same state; one car's lap **does not** advance another car's
`completed_laps`; the agent set width stays constant through SuperSuit even when a car terminates
early; a one-car field reproduces `RacingEnv`; `ruff` clean.

### Step C — selfplay-training-engineer · `src/f1rl/train/` + `configs/experiment/` (depends on B)
- **`selfplay.py`** (new) — the self-play entry (above): SuperSuit venv, warm-start the Phase 4
  generalist, pool-widening curriculum reused in-place, field size a per-run constant with the
  cross-run step-up convention, checkpoint/resume.
- **Multi-car eval driver** (new, in `train/`) — drives `RacingParallelEnv` for the offscreen mp4
  (since `evaluate.py` drives a Gym env, which a `ParallelEnv` is not). Owns the field eval clip.
- **`calendar_selfplay.yaml`** — `grid:` block (`n_agents`, `reset_mode`, `grid_spacing_m`,
  `team_colors`), `n_copies`, warm-start path, `wandb` group `phase-5`.
- **Throughput check** — SPS of `n_copies × N`-car SuperSuit vs equal-core single-agent `n_envs`,
  reported **before** committing field sizes.
- **Smoke run** — a tiny-budget run completes and the return is **non-degenerate** (this only proves
  the env is trainable; it is expected to match the equivalent single-agent `n_envs` run — no
  learning gain this phase).

**Gate:** smoke run completes non-degenerate; Phase 4 generalist warm-starts without an obs-version
error; a 2-car checkpoint resumes into a 4-car run; SPS number reported; checkpoint round-trips with
`n_agents`.

### Step D — app-integration-engineer · server + `sim/` + `web/` (depends on B's env, C's checkpoint)
- **`sim/loop.py` / `server/app.py`** — drive the field; stream `cars: [...]` (one car = one-element
  array, one-car path preserved); the same checkpoint drives every car.
- **recorder + replay** — one multi-car recorder (`cars` key); the replay player scrubs it.
- **`web/`** — render every car colored by team; the timing tower lists every car with lap time and
  the track-position gap.

**Gate:** a field of 2 then 4 cars laps a circuit live for one checkpoint; every car appears in the
tower; the `grid` reset does not spawn cars on top of each other; a recorded multi-car run replays;
the one-car live view still works.

### Step E — test-engineer (independent) · `tests/`
Writes from the **spec contracts and public signatures only** — not implementation internals.
- `test_multi_agent_env.py` — `parallel_api_test` passes; per-agent obs/action spaces are the
  unchanged Box; per-agent obs == single-agent obs for the same state (obs unchanged, track-only);
  **per-car lap timer** (one car's lap does not advance another's `completed_laps`/`progress`);
  independent per-car done flags (one car's failure does not end the others); the per-episode
  circuit draw is shared by the field and reproducible from the seed; `reset_mode=grid` places
  distinct non-overlapping slots, `reset_mode=scattered` distinct seeded indices, both reproducible;
  a non-positive `n_agents` and an unbuilt circuit id are refused.
- `test_selfplay_smoke.py` — SuperSuit wraps the env into an SB3-trainable vec env; a short run
  completes with a non-degenerate return; **constant SuperSuit-visible width** even when a car
  terminates early (black-death padding).
- `test_checkpoint.py` (extend) — the Phase 4 generalist (`obs_version=2`) **passes**
  `validate_checkpoint`; a fresh self-play checkpoint round-trips weights + vecnorm + timestep +
  `n_agents`; a smaller-field checkpoint loads into a larger-field venv (warm start across widths).
- `test_recorder.py` (extend) / replay — the multi-car recorder writes per-car `cars` entries and
  replays identically.

### Step F — reviewer · gate every diff (read-only)
Checklist: config-driven (no magic constant in logic); SI units; **no observation change and
`OBS_VERSION` stays 2** (no nearby-car block, no collisions, no racing rewards this phase); no
physics-interface change; homogeneous agents (one shared obs/action space); per-car core reused not
duplicated; **a per-car `LapTimer`, never the pooled `entry.lap_timer`**; **constant
SuperSuit-visible width** (black_death_v3; raw env removes dead agents per PettingZoo); **field size
a per-run constant, no in-place field-size curriculum** (pool widening stays in-place); the
single-agent path and the one-car live view unbroken; deterministic seeding for the circuit draw and
both reset modes; runtime-safe loader (no FastF1 under `env/`/`train/`). Runs `pytest` (incl.
`parallel_api_test`) + `ruff check` + `ruff format --check`. Pass/fail with reasons; blocks on any
violation or red test/lint. **No physics-engineer this phase — physics is unchanged.**

---

## Dispatch DAG (dependency order)

```
0. scaffold (branch, agent defs, grid config + calendar_selfplay.yaml)
A. dependency-matrix (GATE): pin pettingzoo/supersuit/sb3 + SuperSuit round-trip   ── no env code until green
B. multiagent-env-engineer: per-car step factoring + RacingParallelEnv            ──┐  test-engineer (E)
   + make_selfplay_vec_env   (critical path; the main Phase 5 role)                 │  starts in parallel,
C. selfplay-training-engineer: SuperSuit→SB3 shared policy + warm-start +          │  writing failing tests
   multi-car eval driver + throughput check + smoke   (needs B + A)                 │  from the spec contracts
D. app-integration-engineer: cars[] live frame + multi-car recorder/replay +      ┘
   field render + timing tower   (needs B's env + C's checkpoint)
F. reviewer (E) gates each merge; final full suite + ruff; PR with run summary,
   curves, SPS number, and a clip of the field lapping a circuit.
```

dependency-matrix is the **first gate** (a broken matrix sinks the phase). multiagent-env-engineer
is the critical path. selfplay-training-engineer is sequential on the env + the SuperSuit builder.
app integration follows the env + checkpoint. test-engineer runs concurrently from the contracts;
reviewer gates throughout.

---

## Definition of done (spec §2e, §3b)

- The observation is **confirmed unchanged and locked** (length 22, `OBS_VERSION = 2`, no nearby-car
  block, no collisions) by a test.
- `RacingParallelEnv` holds **N homogeneous cars on one shared circuit**, passes `parallel_api_test`,
  keeps **per-car lap state independent** (per-car `LapTimer`), and reproduces `RacingEnv` on a
  one-car field.
- The **SuperSuit-visible agent width is constant** (black_death_v3) even with per-car early death;
  one car's failure never ends the others.
- One **shared policy** trains through SuperSuit → SB3 with parameter sharing, **warm-started from
  the Phase 4 generalist** (no obs-version error), and the field scales **2 → 4 across warm-started
  runs** (full 22 where compute allows) — field size a **per-run constant**, pool widening **in-place**.
- The field **renders and times correctly** in the app (cars[] frame, per-team colors, timing tower
  with track-position gap), one multi-car recorder replays, and the **one-car path still works**.
- An **SPS number** (SuperSuit field vs single-agent `n_envs`) is measured and the field ceiling
  documented.
- Full test suite + `ruff` green. PR carries the run summary, the curves, the SPS number, and a clip
  of the field lapping a circuit. **The bar is infrastructure correctness and the render, not a
  learning gain** (none is expected this phase).

---

## Risks & open items

- **Shared `LapTimer` / shared projection state** is the #1 multi-agent bug — `entry.lap_timer` is
  one instance per pool entry (`pool.py:129`), and `_prev_s`/`_grip_idx`/`_grip_lat`/
  `_wrong_way_count`/`CarState` are car state. The factored step must carry **all** of these per
  car; the test asserts one car's lap does not move another's `completed_laps`.
- **`black_death_v3` vs `parallel_api_test`** — the raw env must follow standard PettingZoo (remove
  dead agents next step) so the API test passes; the constant width comes from the `black_death_v3`
  **wrapper**, not the raw env. Wiring black_death into the env itself, or keeping dead agents in
  `self.agents`, risks failing `parallel_api_test`.
- **Throughput is the wall** — SuperSuit steps N cars sequentially in one process on the slow
  dynamic model; a 22-car process costs ~22× per-process step cost, not amortized by fan-out. The
  SPS check gates the field ceiling before scaling; the JAX env (design §17) is the real lever and
  Phase 5 is where that pressure first appears.
- **Cross-run field-size warm start** — SB3 `set_env` accepts a different vec width because the
  per-agent obs/action spaces match; verify a 2-car checkpoint loads into a 4-car venv in
  `test_checkpoint.py`. A field that drives worse as it grows → step the field up more slowly across
  runs, or revert the warm start.
- **Reproducible draws** — the per-episode circuit draw and both reset modes must use
  `self.np_random` (seeded by `reset`), not a module RNG (CLAUDE.md seeding rule).
- **Live-frame backward compatibility** — keep one frame type with a `cars` array (single car = one
  element) so the Phase 1/4 one-car view keeps working; the open question (versioned frame vs one
  array) is settled here as the one-array path.
- **Spec drift to fix in the same change** — (a) PettingZoo/SuperSuit are already in
  `pyproject.toml` (unpinned), not "not yet"; (b) read the spec's "never removed from `self.agents`"
  as "constant SuperSuit-visible width via the wrapper." Update `TECHNICAL_DESIGN.md` §10/§13 with
  the `grid:` block, the per-agent dict contract, the black-death/constant-width mechanism, the
  per-run field-size step-up, the multi-car frame/recorder schema, and the `n_agents` checkpoint
  field — in the commit that introduces each.
```
