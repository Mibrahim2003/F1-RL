# Phase 4 Implementation Plan — One Car on Many Circuits

Companion to `.claude/specs/phase-4-many-circuits.md` (the spec). This is the **how**:
concrete, dependency-ordered, file-by-file build order grounded in the real Phase 1/2/3a/3b
code, dispatched through the subagent roster in spec §5. Branch: `phase-4-many-circuits`.

> Authoritative engineering doc remains `.claude/TECHNICAL_DESIGN.md` (§7 observations —
> "local, relative features only … this is what lets one policy generalize across the whole
> calendar", §10 env contract, §12 training/checkpointing, §15 build order — Phase 4). Where
> this plan fixes a contract the design leaves open (the circuit-pool config schema, the
> per-episode sampling contract, the per-circuit pole resolution, the curriculum pool-widening
> field, the lap-time-table format), update `TECHNICAL_DESIGN.md` in the **same commit** — the
> decision and the doc move together (CLAUDE.md rule).

The headline: keep the Phase 3b policy, physics, and observation **unchanged**; turn the env's
single bound track into a **config-driven circuit pool** that **samples a circuit per reset**;
precompute the per-track work once per pool circuit; **widen the pool easy → full calendar**
with the existing curriculum mechanism; **warm-start and continue** the 3b checkpoint (obs is
unchanged, so no forced retrain); and emit a **lap-time table vs the pole, one row per
circuit** as the artifact.

---

## Confirmed / assumed decisions (resolve the spec open questions)

All are config values, reversible; none costs a forced retrain (the observation is unchanged).

1. **Observation = ObservationV2, UNCHANGED.** Length 22, `OBS_VERSION = 2`. `observations.py`
   is already local/relative only (speed, heading error, signed lateral over half-width,
   curvature lookahead, edge beams, tire tail) — **no absolute position anywhere** (design §7).
   Phase 4 *confirms and locks* this with a test; it does **not** change the vector. Because
   `OBS_VERSION` does not move, `validate_checkpoint` accepts the 3b checkpoint → **warm start
   is legal** (unlike the 3a→3b jump, which bumped the version and forced a retrain).
2. **Physics = unchanged.** The dynamic bicycle model + grip pipeline from 3b are reused as-is
   behind `PhysicsModel`. No `physics/` work this phase, no `reset`/`step` signature change.
3. **Circuit pool = the built `.npz` circuits** under `data/tracks/` (19 today: albert_park,
   bahrain, baku, catalunya, cota, hungaroring, interlagos, jeddah, las_vegas, lusail,
   marina_bay, mexico_city, miami, monaco, montreal, monza, red_bull_ring, shanghai,
   silverstone). The pool is a **config list of ids**, so it is trivially editable.
   `spa`/`suzuka`/`zandvoort`/`yas_marina`/`madring` have a `configs/track/` file but **no built
   `.npz`** — excluded from the pool until built (the loader already raises `FileNotFoundError`
   with the build hint; the pool builder surfaces it, never silently drops a circuit).
4. **Per-episode sampling, uniform, from the env RNG.** Each `reset` draws one circuit id with
   `self.np_random` so the draw is reproducible from the run seed; each parallel worker draws
   independently for diverse rollouts. Optional per-id sampling weights are config, default
   uniform. Fallback (config flag): pin one circuit per worker if per-episode swapping ever
   costs too much.
5. **Per-circuit pole comes from `configs/track/<id>.yaml`** (`pole_time_s`), **not** from the
   geometry `.npz` (the `Track` schema stores `official_length_m` but no pole). The pool builder
   resolves each id's pole once at construction and binds it to that circuit's `LapTimer`. A
   missing/non-positive pole flags that circuit and skips its delta (never compare against zero)
   — the existing `evaluate.py` `pole_missing` path.
6. **Curriculum widens the pool**, reusing the 3b mechanism. `CurriculumStage` gains an optional
   `circuits: [ids]`; the callback pushes the active stage's pool into every worker via a new
   `env_method("set_track_pool", ...)`, exactly as it already pushes grip/wear/weather via
   `apply_conditions`. Conditions and pool are both *sampling*-side — they never touch the obs
   layout, so there is no mid-run retrain.
7. **Warm start from the 3b checkpoint, then continue** on the pool (`--resume` continues the
   timestep count; obs version matches). Retrain from scratch is the fallback if the warm start
   fails to generalize.
8. **New experiment config `configs/experiment/calendar_dynamic.yaml`**, extending
   `rbr_dynamic.yaml`: adds the `circuits:` pool block, the pool-widening `curriculum.stages`,
   and the warm-start checkpoint path; keeps the 3b `physics`/`tires`/`weather`/`reward`/PPO
   blocks. Bump `wandb.tags`/`group` to `phase-4`. No tuning constant in logic.
9. **Device = local CPU** by default (as 3b). The per-*step* cost is unchanged (same physics);
   the new cost is **RAM** — N tracks + N edge caches + N lap timers held per worker. Cap N per
   worker via the curriculum (start small) and document the ceiling; re-run `train/benchmark.py`
   only to confirm sampling adds no per-step regression.

---

## Phase 1/2/3a/3b baseline (verified in code — what we build on)

- `env/single_agent.py` — `RacingEnv` **binds one track at `__init__`**
  (`single_agent.py:137-162`): `self.track = load_track(track_id, …)`, `self.track_id`,
  `self.edge_cache = build_edge_cache(self.track)` (`:149`), and
  `self.lap_timer = LapTimer(self.track, pole_time)` (`:162`, pole via
  `_track_get(cfg, "pole_time_s", 0.0)`). **These four are the entire per-circuit binding.**
  - `reset()` (`:185-223`) reads `self.track.centerline/tangent/s`, calls
    `_sample_start_index` (`:372`, uses `self.np_random.integers(0, n)`), and
    `self.lap_timer.reset()`. It does **not** re-pick a circuit — the swap point is the top of
    `reset`.
  - `step()` (`:225-291`) reads `self.track` and `self.lap_timer` only through instance attrs,
    so **rebinding them in `reset` is sufficient; `step` needs no change.**
  - `apply_conditions(...)` (`:295-317`) is the **exact precedent** for the curriculum hook: an
    env method called on workers via `VecEnv.env_method`, takes effect from the next
    `reset`/`step`, never touches the obs layout. `set_track_pool` mirrors it.
- `env/observations.py` — `OBS_VERSION = 2`, `OBS_DIM = 22`, `build_observation`/`track_query`/
  `observation_space()`, `build_edge_cache(track)` (the per-track precompute the beams cast
  against). **Local/relative only — the track-agnostic property to lock, not change.**
- `env/factory.py` — `make_env` / `make_vec_env` (SubprocVecEnv + VecNormalize), per-worker
  seed `seed + rank`. The pool is constructed inside `RacingEnv`, so the factory is largely
  unchanged (it passes the same `cfg`); each worker builds its own pool and seeds its own draw.
- `track/loader.py` — `load_track(id)` (runtime-safe, **no FastF1**, reads `data/tracks/<id>.npz`,
  raises `FileNotFoundError` + build hint for an unbuilt id) and `list_tracks()` (catalog). The
  pool builder calls `load_track` per id.
- `train/curriculum.py` — `CurriculumStage(start_step, mu_base, wear_rate, weather)`,
  `parse_stages`, `active_stage`, `CurriculumCallback` (pushes via
  `training_env.env_method("apply_conditions", …)`). **Extend the dataclass + parser + callback
  with `circuits`.**
- `train/evaluate.py` — `run_episode`/`evaluate` already compute `best_lap_time`,
  `lap_delta_to_pole`, `beat_pole`, `beat_2x_pole`, `pole_missing`, and a `summary(pole)`. It
  reads the pole from `cfg.track.pole_time_s` via `_track_pole` and **builds one `RacingEnv` on
  one circuit.** The calendar benchmark loops this per pool circuit with each circuit's pole.
- `train/checkpointing.py` — `meta.json` records `obs_version` (still 2) and a single
  `circuit_id`; `validate_checkpoint` refuses an obs/action mismatch. Format unchanged;
  `circuit_id` records the **pool descriptor** (e.g. `"calendar"`) for a multi-circuit run.
- `sim/loop.py` / `sim/policy_pilot.py` / `server/app.py` — watch-live engine + policy loader +
  backend. `app.py` loads a track by id and runs `SimLoop`; the selector already lists circuits
  (`list_tracks`). Phase 4 lets the **same checkpoint** drive **any** built circuit and names
  the current circuit + pole in the HUD.

**Gaps to create:** the circuit-pool builder + per-episode sampling + rebind + `set_track_pool`
in `env/single_agent.py`; the `circuits` field across `train/curriculum.py`; a calendar
benchmark (`train/calendar_benchmark.py`, new) emitting the lap-time table;
`configs/experiment/calendar_dynamic.yaml` + a `circuits:` config block in `default.yaml`; the
new/extended tests; the live-view circuit-name + pole readouts; and the `.claude/agents/` roster
for Phase 4 (env-engineer, training-engineer, app-integration-engineer, test-engineer,
reviewer).

---

## Contracts fixed before any code (foundation)

### Circuit pool + per-episode sampling (`env/single_agent.py`)

Config block (root, alongside the existing single `track_id`):

```yaml
circuits:
  pool: [red_bull_ring, monza, catalunya, silverstone, …]   # built .npz ids only
  sampling: uniform        # uniform | weighted
  weights: null            # optional {id: w}; null => uniform
  pin_per_worker: false    # fallback: each worker fixed to one pool circuit
```

`RacingEnv.__init__` builds a **`CircuitPool`**: for each id, `load_track(id)`,
`build_edge_cache(track)`, resolve `pole_time_s` (per-circuit, see below), and build a
`LapTimer(track, pole)`. Store as parallel dicts keyed by id (built **once**). Back-compat: an
empty/absent `circuits.pool` falls back to the single `track_id` → a **one-circuit pool**, so
the Phase 3b behavior is reproduced exactly.

`reset()` change (top of the method, before `_sample_start_index`):

1. Draw `cid` from the active pool with `self.np_random` (respecting `sampling`/`weights`; or the
   pinned id when `pin_per_worker`).
2. Rebind the active per-circuit state from the dicts: `self.track`, `self.edge_cache`,
   `self.lap_timer`, `self.track_id`, and the active pole. (These are the only four bindings;
   `step` reads them through `self.*` and needs no change.)
3. Proceed with the existing `_resolve_weather` → `_sample_start_index(self.track)` → spawn →
   `self.lap_timer.reset()`. The draw is reproducible from the seed; no absolute position leaks
   across the swap (obs is built fresh from the new track each episode).

`reset` returns `info["circuit_id"] = cid` (and keeps `start_index`) so eval/clip/HUD know the
drawn circuit.

### Per-circuit pole resolution (pool builder)

The pole lives in `configs/track/<id>.yaml` (`pole_time_s`), not the `.npz`. The pool builder
resolves it per id once (load that track's config node, read `pole_time_s`, default `0.0`). Bind
it to that circuit's `LapTimer`. `pole <= 0` → flag the circuit `pole_missing` and skip its
delta downstream (reuse `evaluate.py`'s existing handling). Keep this resolution **runtime-safe**
(read YAML, no FastF1, no network).

### Curriculum pool-widening (`train/curriculum.py`)

Extend the existing stage table with an optional `circuits` list:

```yaml
curriculum:
  enabled: true
  stages:
    - start_step: 0          # learn on a few easy, wide circuits
      circuits: [red_bull_ring, monza]
      mu_base: 1.20
      weather: dry
    - start_step: 1500000    # widen to a mid set
      circuits: [red_bull_ring, monza, catalunya, silverstone, bahrain, cota]
    - start_step: 4000000    # full calendar
      circuits: []           # empty/"all" => the full configured pool
      weather: sampled
```

- `CurriculumStage` gains `circuits: list[str] | None`; `parse_stages` reads it; `active_stage`
  is unchanged.
- `CurriculumCallback._maybe_apply` additionally calls
  `self.training_env.env_method("set_track_pool", circuits=stage.circuits)` when the stage sets a
  pool (alongside the existing `apply_conditions`). An empty/`None`/`"all"` value means the full
  configured pool.
- `RacingEnv.set_track_pool(circuits)` — new env method mirroring `apply_conditions`: validate
  every id is in the built pool, set the **active** sampling list (takes effect next `reset`),
  never rebuild the per-circuit dicts (already built for the full configured pool). Pure, no obs
  change → safe mid-run.

### Lap-time table benchmark (`train/calendar_benchmark.py`, new)

A deterministic, agent-driven sweep over the configured pool:

- For each pool circuit: build a one-circuit `cfg` (set `track_id` = id, point `cfg.track` at
  that circuit's config so `_track_pole` resolves), run the existing `evaluate(...)` for K
  deterministic episodes with the **saved VecNormalize stats**, and collect
  `best_lap_time`, `pole_time_s`, `lap_delta_to_pole`, `beat_pole`, `beat_2x_pole`,
  `pole_missing`, `off_track_count`.
- Assemble a table (one row per circuit) + aggregates (mean delta, worst-circuit delta,
  beat-2×-pole rate). Print it, log per-circuit scalars + aggregates to W&B, and **save the
  table** (JSON + CSV under `out/`) as the phase artifact.
- CLI: `python -m f1rl.train.calendar_benchmark --checkpoint <dir> --config experiment/calendar_dynamic --episodes 2`.

This reuses `evaluate.py` verbatim per circuit (no duplicate metric logic); the new code is the
loop, the per-circuit pole/config resolution, and the table assembly/save.

### Checkpoint (unchanged format)

No schema change. `obs_version` stays 2 → `validate_checkpoint` **accepts** the 3b checkpoint, so
the warm-start `--resume` loads cleanly. `meta.json` `circuit_id` records the pool descriptor
(e.g. `"calendar"`) rather than one track. Round-trip (weights, optimizer, vecnorm, timestep,
RNG) is unchanged.

### Track-agnostic observation lock (no change, test only)

`test_observations.py` gains: the observation contains no absolute world coordinate; the **same
`CarState`** projected on two different circuits yields same-shape, in-bounds vectors; `OBS_DIM
== 22` and `OBS_VERSION == 2` hold. This **locks** the property the whole phase relies on; it
asserts the vector did **not** change.

---

## Build order (dependency-first), mapped to subagents

BEFORE DISPATCHING ANY AGENT USE THE /caveman SKILL IN THERE PROMPT.

### Step 0 — scaffold (main thread)
- Create branch `phase-4-many-circuits`.
- Create the `.claude/agents/` roster for Phase 4 (env-engineer, training-engineer,
  app-integration-engineer, test-engineer, reviewer — **no physics-engineer this phase**). Each
  agent file's body **must open with the directive to call the `/caveman` skill before any
  work** (spec §5, emphasized, non-negotiable), then its narrow scope, "read first" list, tasks,
  rules, and done-gate. Point each at this plan + the spec + the relevant `TECHNICAL_DESIGN.md`
  sections (§7, §10, §12, §15).
- Add the `circuits:` config block to `configs/default.yaml` (safe default: empty pool ⇒
  single-`track_id` fallback) and create `configs/experiment/calendar_dynamic.yaml` extending
  `rbr_dynamic.yaml` (pool list + pool-widening `curriculum.stages` + warm-start checkpoint path
  + `wandb` group `phase-4`).

### Step A — env-engineer · `src/f1rl/env/` (critical path, the main Phase 4 role)
Turn the single bound track into a sampled pool — **no `reset`/`step` signature change, no obs
change.**

- **`single_agent.py`** — build the `CircuitPool` in `__init__` (per-id `Track` + `EdgeCache` +
  `LapTimer` + resolved pole, built once); draw a circuit from `self.np_random` at the top of
  `reset` and rebind `self.track`/`self.edge_cache`/`self.lap_timer`/`self.track_id` + active
  pole; add the `set_track_pool(circuits)` env method (mirror `apply_conditions`); return
  `info["circuit_id"]`. Empty pool ⇒ one-circuit fallback (3b behavior preserved).
- **Per-circuit pole resolution** — a small helper (in the env or a `pool.py` module) that reads
  `configs/track/<id>.yaml`'s `pole_time_s`; runtime-safe, no FastF1.
- **`factory.py`** — confirm `make_vec_env` needs no change (each worker builds its own pool from
  the same `cfg`, seeds its own draw via `seed + rank`); adjust only if pool construction needs a
  per-worker hook.

**Gate:** `gymnasium.utils.env_checker.check_env(RacingEnv(pool cfg))` passes; obs ∈ space at
length 22 on every pool circuit; consecutive `reset`s draw different circuits and are
reproducible from a fixed seed; the four bindings swap together with no stale per-track state; a
one-circuit pool reproduces the single-circuit rollout exactly; `ruff` clean.

### Step B — training-engineer · `src/f1rl/train/` + `configs/experiment/` (depends on A)
Widen the pool, warm-start, and emit the table.

- **`curriculum.py`** — add `circuits` to `CurriculumStage`/`parse_stages`; make
  `CurriculumCallback` push `set_track_pool` per active stage (alongside `apply_conditions`).
- **`calendar_benchmark.py`** (new) — the per-circuit `evaluate` loop + table assembly + W&B
  logging + JSON/CSV save (contract above).
- **`train.py` / `configs/experiment/calendar_dynamic.yaml`** — wire the warm-start `--resume`
  from the 3b checkpoint (obs v2 matches), set the `circuits:` pool + pool-widening
  `curriculum.stages`; the eval callback rotates the eval circuit. Then a **tiny-budget smoke
  run** proving reward trends up across circuits and resume continues the timestep count, and the
  calendar benchmark produces a table.

**Gate:** smoke run reward non-degenerate on the pool; 3b checkpoint warm-starts without an
obs-version error; curriculum activates the right pool at each threshold; calendar benchmark
emits a row per pool circuit with each circuit's pole + delta; checkpoint resumes.

### Step C — app-integration-engineer · server + `web/` (depends on A's env, B's checkpoint)
- **`server/app.py` / `sim/loop.py`** — let the **same** checkpoint drive **any** built circuit
  (the selector already lists circuits); load the picked circuit's `Track` + pole for the live
  session. Bad/missing checkpoint still falls back to autopilot, never crashes.
- **`web/src/hud/telemetry.ts`** (+ `types.ts`) — show the **current circuit name** and its
  **pole** alongside the existing lap time + delta (Phase-1 timing colors). Render the calendar
  lap-time table as the phase result view.

**Gate:** one checkpoint drives two different circuits live; the HUD names the right circuit +
pole; the calendar table renders.

### Step D — test-engineer (independent) · `tests/`
Writes from the **spec contracts and public signatures only** — not implementation internals
(spec §5, §c). `/caveman` first.

- `test_observations.py` (track-agnostic lock) — no absolute position; same `CarState` on two
  circuits ⇒ same-shape, in-bounds; `OBS_DIM == 22`, `OBS_VERSION == 2` (vector unchanged).
- `test_circuit_pool.py` — pool builder loads every configured id; an unbuilt id raises the
  clear `FileNotFoundError` build hint; an id with no track config / non-positive pole is
  flagged, not zero-divided; per-circuit pole resolves from `configs/track/<id>.yaml`.
- `test_env_sampling.py` — `reset` draws varying circuits across episodes; fixed seed ⇒
  reproducible draw sequence; the rebind swaps track/edge_cache/lap_timer/pole together;
  `set_track_pool` changes the active draw set from the next reset; one-circuit pool ≡ 3b.
- `test_env_api.py` — `check_env` passes on a pool env; action space unchanged.
- `test_curriculum.py` — a stage with `circuits` pushes `set_track_pool`; the right pool is
  active at each threshold; empty `circuits` ⇒ full pool.
- `test_calendar_benchmark.py` — the loop yields one row per pool circuit; each row uses that
  circuit's pole; delta = `best_lap − pole`; missing pole skipped/flagged; table saved.
- `test_checkpoint.py` — the 3b (`obs_version=2`) checkpoint **passes** `validate_checkpoint`
  (warm start legal); a fresh pool-run checkpoint round-trips weights + vecnorm + timestep.

### Step E — reviewer · gate every diff (read-only)
`/caveman` first. Checklist per task: config-driven (no magic constant in logic); SI units;
**no `reset`/`step` signature change, no observation change, `OBS_VERSION` stays 2**; **no
absolute position in the observation**; per-track precompute built **once** (no edge-cache
rebuild per reset); per-circuit pole resolved from config (never divided by zero); pool draw
uses `self.np_random` (reproducible); runtime-safe loader (**no FastF1 under `env/` or
`train/`**, no network); shared obs builder reused by the server (not reimplemented); one-circuit
pool reproduces 3b; checkpoint round-trips. Runs `pytest` + `ruff check` + `ruff format --check`.
Pass/fail with reasons; blocks the merge on any violation or red test/lint.

---

## Dispatch DAG (dependency order)

```
0. scaffold (branch, agent defs w/ /caveman, circuits config + calendar_dynamic.yaml)
1. env-engineer (A): CircuitPool + per-reset draw + rebind + set_track_pool   ──┐  test-engineer (D)
   (critical path; the main Phase 4 role)                                       │  starts in parallel,
2.                              pool-sampling env ready (one-circuit ≡ 3b)       │  writing failing tests
3. training-engineer (B): curriculum.circuits + warm-start resume +            │  from the spec contracts
                          calendar_benchmark → table → smoke                    ┘
4. app-integration-engineer (C): same ckpt → any circuit, HUD circuit + pole,
                                 calendar table   (needs A's env + B's ckpt)
5. reviewer (E) gates each merge; final full suite + ruff; PR with run summary,
   curves, and the lap-time-vs-pole table across the calendar.
```

env-engineer is the critical path (the main Phase 4 role). training-engineer is sequential on
the pool/`set_track_pool` signatures. app integration follows the env + checkpoint.
test-engineer runs concurrently from the contracts; reviewer gates throughout. **Every subagent
calls `/caveman` before starting (spec §5).** No physics-engineer this phase — physics is
unchanged.

---

## Definition of done (spec §2e, §3b)

- The observation is **confirmed track-agnostic and locked** by a test; the vector is
  **unchanged** (length 22, `OBS_VERSION = 2`).
- The env holds a **config-driven circuit pool** and **samples a different circuit each reset**,
  rebinding track/edge_cache/lap_timer/pole together; a one-circuit pool reproduces 3b exactly.
- Per-track precompute is built **once per circuit**; the per-reset cost is a lookup.
- Each circuit's **pole resolves from its config**; a missing pole is flagged, never zero-divided.
- The **curriculum widens the pool** easy → full calendar via `set_track_pool`, with no obs
  change and no mid-run retrain.
- The Phase 3b checkpoint **warm-starts** on the pool (`validate_checkpoint` accepts obs v2) and
  `--resume` continues the timestep count.
- **One policy laps every pool circuit**, and a **lap-time table vs the pole, one row per
  circuit** (achieved / pole / delta / 2× flag), is logged and saved; `2·pole` is reached across
  the bulk of the calendar and the deltas close over training.
- Full test suite + `ruff` green. PR carries the run summary, the wandb curves, and the
  lap-time-vs-pole calendar table.

---

## Risks & open items

- **Stale per-track state after a swap** is the #1 pool bug — `step` reads `self.track`,
  `self.edge_cache`, `self.lap_timer`, and the cached `_grip_idx`/`_grip_lat`/`_prev_s` through
  instance attrs. `reset` must rebind **all four** bindings *and* re-seed the per-step projection
  state for the drawn circuit before the first `step`. `test_env_sampling.py` must assert no
  carry-over (e.g. lap timer not spuriously firing across a swap).
- **Per-circuit pole lives in config, not the `.npz`** — easy to miss and silently score against
  pole `0`. The pool builder resolves it from `configs/track/<id>.yaml`; the reviewer checklist
  enforces it; the missing-pole path is the existing `evaluate.py` flag.
- **RAM, not steps, is the new cost** — N tracks + N edge caches + N lap timers per worker ×
  `n_envs`. Start the pool small via the curriculum and document the ceiling; the SPS re-bench
  confirms the per-step rate is unchanged (same physics).
- **Generalization needs more steps than one circuit** (spec §4) — the curriculum widening and
  the warm start are the levers; a circuit stuck far off its pole → adjust the stage pool or the
  sampling weights in config, not in logic.
- **Reproducible per-episode draw** — the draw must use `self.np_random` (seeded by `reset`), not
  a module RNG, or runs stop being reproducible from the seed (CLAUDE.md seeding rule).
- **Unbuilt circuits in the pool** (spa/suzuka/zandvoort/yas_marina/madring have configs, no
  `.npz`) — the pool builder must fail loud with the build hint, never silently shrink the pool.
- **Loader stays runtime-safe** — the pool builder and pole resolver must not import FastF1 or
  touch the network (design + CLAUDE.md); they read cached `.npz` + YAML only.
```
