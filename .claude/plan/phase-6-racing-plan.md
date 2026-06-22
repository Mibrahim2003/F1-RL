# Phase 6 Implementation Plan — Racing for Real

Companion to `.claude/specs/phase-6-racing.md` (the spec). This is the **how**: the concrete,
dependency-ordered, file-by-file build order grounded in the real Phase 1–5 code, dispatched through
the subagent roster in spec §5. Branch: `phase-6-racing` (cut from `main` after Phase 5 merges).

> Authoritative engineering doc remains `.claude/TECHNICAL_DESIGN.md` (§7 observations — the reserved
> nearby-car block, "relative position and velocity of the K nearest cars (added in later phases)";
> §5/§9/§10 env + reward + physics; §15 build order — Phase 6 = nearby-car obs, collisions, contact
> penalty, overtake/defend rewards). Where this plan fixes a contract the design leaves open (the
> ObservationV3 layout + `OBS_VERSION = 3`, the `collision:` config block + two-disc model, the
> field-level collision pass + advance/finalize split, `reward_v3`, the grown warm start, the
> reward-weight curriculum, the racing info/metrics), **update `TECHNICAL_DESIGN.md` in the same
> commit** — the decision and the doc move together (CLAUDE.md rule).

The headline: keep the Phase 5 physics, circuit pool, per-car `CarRuntime`, SuperSuit→SB3 stack, and
field render. **Append** a fixed K-nearest-cars block to the observation (`OBS_VERSION 2→3`, `[0:22]`
byte-identical). **Split** the per-car step into `advance_car_physics` + `finalize_car_step` and run
a field-level `resolve_collisions` between them (two discs/car, snapshot-then-apply, order-independent;
`PhysicsModel.step` untouched). Add `reward_v3` = `reward_v2` core − graded contact penalty + zero-sum
position term. **Warm-start by growing the input layer** (the obs changed, so a silent resume is
refused; transplant the Phase 5 driver). Then the racing curriculum, the racing metrics, and the
race-aware app. **Reward balance is the budgeted core of the phase, not a final polish step.**

---

## Confirmed / assumed decisions (resolve the spec open questions)

All new values are config; none is a tuning constant in logic. The obs change is the one forced
retrain (`OBS_VERSION 3`); everything else is reversible by config.

1. **ObservationV3 = ObservationV2 + a tail neighbor block.** `[0:22]` unchanged and byte-identical
   to v2 (the warm-start prefix property). Tail `[22 : 22+K·F]` = K nearest cars, **nearest-first,
   zero-padded** with a validity bit, **local/relative in the observer body frame**, capped at a
   sensing range R. Defaults (config): `K = 4`, per-neighbor `F = 5` = `[dx_body/R, dy_body/R,
   dvx_body/ref, dvy_body/ref, valid]`, `R = 50 m` → **`OBS_DIM = 42`, `OBS_VERSION = 3`**.
   Body-frame encode: forward axis `(cos yaw, sin yaw)`, left axis `(−sin yaw, cos yaw)`; `CarState`
   velocity is body-frame so the encoder maps each car's `(vx,vy)` to world, takes the delta, then
   rotates into the observer frame.

2. **Collision = two discs per car, field-level pass, physics pure.** Front + rear disc, centers at
   `±disc_offset_m` along the body axis, radius `disc_radius_m` (defaults 1.25 m / 1.0 m). Contact
   when any disc-pair gap `< 2·r`. **`PhysicsModel.step` is NOT touched** — the pass lives in the env
   between physics and finalize. **Snapshot-then-apply**: read all post-physics states, compute every
   pair's correction against the snapshot, then apply the summed corrections (order-independent,
   reproducible). Single resolution pass per step (soft residual overlap accepted in pileups).

3. **`reward_v3` = `reward_v2` core − graded contact penalty + zero-sum position term.** Contact:
   `−w_contact · (closing_mps / contact_soft_mps)^contact_exp` from the car's contact record
   (symmetric default; `w_contact_fault` extra share, default 0). Position: `+w_overtake · places`
   where `places` is the field-computed net places gained this step **gated to genuine swaps** (pair
   order flipped AND current total-progress gap `< overtake_battle_range_m`), zero-sum across the
   swapping pair. Dense gap term `w_gap` **opt-in, default 0** (locked from the spec question). A
   one-car field / single agent → no contact, constant rank → `reward_v3 == reward_v2`.

4. **Contact penalized, not terminal by default.** `collision.crashout_enabled = false`; when on, a
   contact with `closing_mps > collision.crashout_closing_speed_mps` ends that car with the existing
   `env.failure_reward` (same path as off-track in `_check_termination`). Done cars leave
   `self.agents` (Phase 5, unchanged) and are excluded from the collision + neighbor passes.

5. **Warm start = grow the input layer; `--resume` of a v2 checkpoint is refused.** `OBS_VERSION 3`
   makes `validate_checkpoint` raise on a Phase 5 (v2) checkpoint — correct. A new `--warm-start`
   path loads the v2 model with `validate=False`, builds a fresh v3 PPO, copies every weight except
   the policy/value **input layer** (`mlp_extractor.policy_net.0`, `mlp_extractor.value_net.0`),
   copies their weight columns `[0:22]` and **zero-initializes** columns `[22:42]`, copies biases, and
   grows the `VecNormalize` `obs_rms` to width 42 (new dims `mean 0 / var 1`). From scratch is the
   fallback.

6. **Curriculum gains reward-weight ramping.** `CurriculumStage` gets optional `w_contact` /
   `w_overtake`; the in-place transport (Phase 5 `SelfPlayCurriculumCallback` → `raw_parallel_envs` →
   `apply_conditions`) also pushes reward-weight overrides, so a run learns to **coexist** (contact
   penalty on, overtake reward low) before it learns to **fight** (overtake reward ramped up).
   Field-size scaling stays **cross-run** (Phase 5, unchanged); conditions + pool widening stay
   in-place.

7. **Field scaling + circuit pool reused unchanged.** Field size a per-run constant grown across
   warm-started runs; one circuit per episode shared by the field; per-car lap timers; read-only pool
   entry. All Phase 5, untouched.

8. **No new third-party dependency.** Collisions + neighbor search are NumPy. The pinned Phase 5
   matrix (`pettingzoo==1.26.1` / `supersuit==3.11.0` / `stable-baselines3==2.9.0`) stands.

9. **New experiment config `configs/experiment/calendar_racing.yaml`**, extending
   `calendar_selfplay.yaml`: adds `collision:`, bumps `reward.version` to 3 + the racing weights, adds
   the `obs` neighbor params, extends the curriculum stages with reward-weight ramps, bumps
   `wandb.group`/`tags` to `phase-6`. No tuning constant in logic.

---

## Phase 5 baseline (verified in code — what we build on)

- **`env/single_agent.py`** — `step_one_car` (`:228-304`) is the per-car unit to **split**: map
  action (`:241`) → grip (`:243`) → substeps physics (`:246-248`) → `track_query` projection
  (`:252-258`) → lap timer (`:260`) → `reward_v2`/`reward_v1` (`:262-271`) → wrong-way (`:273-277`) →
  `_check_termination` (`:279-283`, fn `:338-350`) → `_build_obs` (`:285`, fn `:328-335`) → info
  (`:286-303`). `CarRuntime` (`:123-141`) is the per-car mutable state to **extend** with the contact
  record + `prev_rank`. `reset_car` (`:181-225`); `_build_obs` is the single seam where the neighbor
  block enters (`build_observation(...)`, `:333-335`). `CarStepConfig.from_config` (`:164-178`) is
  where `CollisionParams` + the new reward weights get built once.
- **`env/multi_agent.py`** — `RacingParallelEnv.step` (`:177-212`) loops `step_one_car` per live agent
  (`:195-205`) and drops done agents (`:211`). This loop is **reordered** to advance → collide → rank
  → finalize. `reset` (`:140-175`) builds per-car own `LapTimer` (`:159`). `GridParams` (`:56-83`) is
  the field-layout block to sit `collision:` alongside (or in its own block).
- **`env/observations.py`** — `OBS_VERSION = 2`, `OBS_DIM = 22` (`:36-37`), `build_observation`
  (`:254-316`) writes the tail at `:300-306`, `observation_space()` (`:119-133`). The **append point**
  is after `[21]`; `track_query` (`:161-192`) is reused for the v2 prefix. Keep the builder
  field-agnostic — it gets a precomputed block, never the field.
- **`env/rewards.py`** — `reward_v2` (`:152-173`) wraps `reward_v1` (`:102-139`) + an opt-in `slip`
  term (the **exact pattern** `reward_v3` follows: wrap v2, add config-gated terms). `RewardWeights`
  (`:36-70`) — extend with the racing weights. `signed_progress` (`:73-85`).
- **`env/factory.py`** — `make_selfplay_vec_env` (`:108-144`, `black_death_v3` →
  `pettingzoo_env_to_vec_env_v1` → `concat_vec_envs_v1` → `VecMonitor` → `VecNormalize`),
  `raw_parallel_envs` (`:174-192`, the curriculum's reach to the raw field envs). **Unchanged** — the
  obs/action spaces flow through automatically at the new length.
- **`train/selfplay.py`** — `selfplay()` (`:142-195`); warm-start at `:167-176` uses
  `load_checkpoint(resume, env=venv)` which **validates** → add a separate `--warm-start` branch.
  `SelfPlayCurriculumCallback` (`:46-95`) broadcasts via `raw_parallel_envs` (`:65-78`) — extend to
  push reward weights. `_make_callbacks` (`:108-139`).
- **`train/curriculum.py`** — `CurriculumStage` (`:21-31`), `parse_stages` (`:34-62`) — add
  `w_contact`/`w_overtake`.
- **`train/checkpointing.py`** — `validate_checkpoint` (`:249-278`) refuses an `obs_version` mismatch
  (`:260-266`) — this is what makes `--resume` of a v2 ckpt fail (correct). `build_meta` (`:91-131`)
  records `obs_version` (`:124`, now 3). `load_checkpoint` (`:192-246`, `validate=False` is the escape
  hatch the warm start uses).
- **`train/selfplay_eval.py`** — `run_field_episode` (`:90-174`) records `cars[]` (`:125-148`) +
  per-car metrics; `FieldResult.summary` (`:55-76`). Extend with racing metrics + `race_position`.
- **`train/train.py`** — `build_model` (`:86-98`, `PPO("MlpPolicy", ...)`), `_ppo_kwargs` (`:65-83`).
  The grown warm start builds a v3 PPO with these same kwargs, then transplants.
- **`sim/loop.py`** — `FieldSimLoop.step` (`:271-306`) **already** computes
  `total_progress = (completed_laps + progress) · length` (`:280`) and `gap_m = leader − total`
  (`:293-295`) — the **same rank formula** the env's overtake term uses. Add `race_position` +
  gap-to-ahead here. `_frame` (`:169-186`).
- **`sim/timing.py`** — `LapTimer.update` (`:51-75`) gives `completed_laps` + `progress` (`s/L`);
  total progress for rank = `(completed_laps + progress)·length`.
- **`sim/recorder.py`** — `append_cars` (`:55-61`) per-car `{id,x,y,yaw,speed,team,telemetry}`;
  backward-compatible superset. Racing fields go under per-car `telemetry`.
- **`configs/default.yaml`** — `grid:` (`:26`), `selfplay:` (`:45`), `obs:` (`:86`), `reward:`
  (`:94`), `env:` (`:106`), `curriculum:` (`:146`). **`configs/experiment/calendar_selfplay.yaml`** —
  the file `calendar_racing.yaml` extends.

**Gaps to create:** the neighbor block in `observations.py`; a new `env/collisions.py`; the
`advance`/`finalize` split + reordered field step in `single_agent.py`/`multi_agent.py`; `reward_v3`
in `rewards.py`; the `collision:` config block + `reward.version 3` weights + obs neighbor params; a
`--warm-start` grow-the-input-layer path (`train/warmstart.py` + `selfplay.py` branch); the curriculum
reward-weight ramp; the racing metrics in `selfplay_eval.py`; race position + gap-to-ahead in
`sim/loop.py` + `server/` + `web/`; `calendar_racing.yaml`; the new tests; and the `.claude/agents/`
roster for Phase 6.

---

## Contracts fixed before any code (foundation)

### ObservationV3 (`env/observations.py`)

```
OBS_VERSION = 3
N_NEIGHBORS (K) and NEIGHBOR_FEATS (F) from config (obs.k_neighbors, fixed F=5)
OBS_DIM = 22 + K*F                                   # default 42

build_neighbor_block(observer: CarState, others: list[CarState], params) -> np.ndarray  # (K*F,)
  # K nearest by center distance within params.neighbor_range_m, nearest-first, zero-padded.
  # per neighbor, in observer body frame: [dx/R, dy/R, dvx/ref, dvy/ref, valid]; clip to Box.

build_observation(track, state, params, edge_cache=None, grip_indicator=None,
                  neighbor_block=None) -> np.ndarray
  # writes [0:22] EXACTLY as v2; writes [22:22+K*F] = neighbor_block (zeros when None).

observation_space()  -> Box(low, high, shape=(OBS_DIM,))   # tail bounds: dx/dy [-1,1],
                                                            # dv [-2,2], valid [0,1]
```

`[0:22]` byte-identical to v2 (a test asserts it); no absolute position in the tail (local/relative
only); a single car / one-car field → all-zero block.

### Collision pass (`env/collisions.py`, new)

```
@dataclass(frozen=True) CollisionParams:
  enabled, body, disc_radius_m, disc_offset_m, restitution, friction, push_fraction,
  crashout_enabled, crashout_closing_speed_mps    # all from cfg.collision

@dataclass ContactRecord: impulse: float; closing_mps: float; count: int   # per car, per step

resolve_collisions(states: list[CarState], params) -> list[ContactRecord]
  # 1. snapshot world positions + the two disc centers per car
  # 2. for each unordered live pair, min over the 4 disc-disc combos; contact if gap < 2r
  # 3. equal-mass response vs the SNAPSHOT: push-apart (push_fraction, split evenly) along the
  #    contact normal + impulse j = -(1+restitution)*v_n/2 (v_n = closing speed), tangential
  #    friction damp; accumulate per-car position+velocity corrections
  # 4. apply summed corrections to each CarState; map the world-frame velocity delta back into
  #    body (vx,vy) via the car yaw; yaw/yaw_rate unchanged (contact spin = future)
  # order-independent (all corrections computed from the snapshot), reproducible from the seed
```

`PhysicsModel.step` is **not** imported or touched. Done/removed cars are not in `states`.

### Per-car step split (`env/single_agent.py`)

```
advance_car_physics(entry, cfg, car, action) -> None
  # map action -> grip -> substeps physics (mutates car.state); car.t/step_count++; NO project/reward

finalize_car_step(entry, cfg, car, *, neighbor_block=None, places=0) -> (obs, reward, term, trunc, info)
  # project once -> lap timer -> reward_v3(... contact=car.contact, places=places) -> wrong-way ->
  # _check_termination (+ optional crashout from car.contact.closing_mps) -> build obs WITH
  # neighbor_block -> info (+ race_position/gap/contact/overtakes). Identical math to today otherwise.

step_one_car(entry, cfg, car, action) -> (...)            # = advance + finalize(no block, places=0)
  # RacingEnv keeps calling step_one_car -> its contract, obs prefix, and check_env pass-through
  # unchanged at the new length (one car can't collide; reward_v3 reduces to reward_v2).
```

### Reordered field step (`env/multi_agent.py`)

```
RacingParallelEnv.step(actions):
  for a in self.agents: advance_car_physics(entry, cfg, cars[a], actions[a])      # independent
  records = resolve_collisions([cars[a].state for a in self.agents], coll_params) # field-level
  assign records to cars[a].contact
  totals = {a: (cars[a].lap_timer.completed_laps + progress_a) * length}          # rank by total
  ranks_now, places = rank_and_overtakes(totals, prev_ranks, battle_range)        # zero-sum, gated
  for a in self.agents:
    block = build_neighbor_block(cars[a].state, [other live states], obs_params)
    obs[a], rew[a], term[a], trunc[a], info[a] = finalize_car_step(
        entry, cfg, cars[a], neighbor_block=block, places=places[a])
    info[a]["race_position"] = ranks_now[a]; info[a]["gap_ahead_s"] = ...
  update prev_ranks; drop done agents from self.agents (unchanged); episode ends as Phase 5
```

### `reward_v3` (`env/rewards.py`)

```
RewardWeights += w_contact, contact_soft_mps, contact_exp, w_contact_fault,
                 w_overtake, overtake_battle_range_m, w_gap
reward_v3(prev_s, cur_s, off_m, length, weights, slip=0, contact=ContactRecord(), places=0, gap_delta=0):
  reward, terms = reward_v2(prev_s, cur_s, off_m, length, weights, slip)
  terms["contact"]  = -weights.w_contact  * contact_cost(contact, weights)
  terms["overtake"] = +weights.w_overtake * places
  terms["gap"]      = +weights.w_gap      * gap_delta      # w_gap = 0 default -> 0
  reward += contact + overtake + gap; terms["total"] = reward
  # contact==empty and places==0 -> reward_v3 == reward_v2 (single agent / one-car field)
```

### Grown warm start (`train/warmstart.py` + `selfplay.py`)

```
grow_policy(src_ckpt, target_venv, cfg, seed) -> (model_v3, meta_src)
  src_model, meta = load_checkpoint(src_ckpt, env=None, validate=False)   # accept v2
  model = build_model(cfg, target_venv, seed)                            # fresh v3 PPO, same kwargs
  copy every src state_dict tensor into model EXCEPT:
    mlp_extractor.policy_net.0.{weight,bias}, mlp_extractor.value_net.0.{weight,bias}
  for those two layers: new.weight[:, 0:22] = src.weight; new.weight[:, 22:] = 0; new.bias = src.bias
  grow target VecNormalize obs_rms (mean/var/count) to OBS_DIM: copy 0:22, new dims mean 0 / var 1
  return model, meta
# selfplay.py: add --warm-start (distinct from --resume); --resume still validates (v3->v3 only)
```

### Checkpoint (near-unchanged)

`build_meta` records `obs_version = 3`; `--resume` of a v2 ckpt is **refused** by `validate_checkpoint`
(correct); the grown warm start is the explicit transplant path. Round-trip (weights, optimizer,
vecnorm, timestep, RNG, `n_agents`) unchanged. A v3 self-play checkpoint resumes normally.

### Race-aware app (`sim/loop.py`, `server/`, `web/`)

- `FieldSimLoop.step` already has `total_progress` + `gap_m` (`:280`, `:293`); add `race_position`
  (rank index of sorted totals) and convert/augment to **gap-to-car-ahead** per car; thread both into
  the per-car `telemetry`.
- `server/app.py` field driver carries the new fields through unchanged (it forwards the frame).
- `web/`: `hud/telemetry.ts` lists running order **P1…PN** with gap-to-ahead; the renderer needs no
  new primitive (cars bounce because their state changes). Optional debug overlay: neighbor links +
  collision discs.
- `recorder.append_cars` carries the fields under per-car `telemetry`; replay unchanged in format.

---

## Build order (dependency-first), mapped to subagents

### Step 0 — scaffold (main thread)
- Branch `phase-6-racing` from `main` (after Phase 5 merges).
- `.claude/agents/` roster for Phase 6 (observation-engineer, collision-engineer,
  multiagent-env-engineer, reward-engineer, selfplay-training-engineer, app-integration-engineer,
  test-engineer, reviewer). `/caveman` opt-in per agent (spec §5). Point each at this plan + the spec +
  `TECHNICAL_DESIGN.md` §5/§7/§9/§10/§15.
- Add the `collision:` block to `configs/default.yaml` (`enabled: false` default ⇒ Phase 5 parade
  preserved), the `obs.k_neighbors`/`obs.neighbor_range_m` keys, and the `reward` racing weights (with
  `reward.version` left at the per-experiment value). Create `configs/experiment/calendar_racing.yaml`
  extending `calendar_selfplay.yaml`.

### Step A — observation-engineer · `env/observations.py` (foundation, parallel with B)
Append the neighbor block: `build_neighbor_block`, `build_observation(neighbor_block=...)`, bump
`OBS_VERSION 2→3` + `OBS_DIM`, extend `observation_space()`, read `K`/`R` from `ObsParams`.
**Gate:** `[0:22]` byte-identical to v2 for the same state; the block is local/relative (field
translate/rotate invariant up to the observer frame), nearest-first, zero-padded, range-capped; a lone
car → all-zero block; obs ∈ the length-42 Box; `ruff` clean.

### Step B — collision-engineer · `env/collisions.py` (foundation, parallel with A)
Build `CollisionParams`, `ContactRecord`, the two-disc bodies, and `resolve_collisions`
(snapshot-then-apply, equal-mass impulse, body-frame velocity map-back). **Does not import
`PhysicsModel`.**
**Gate:** overlapping cars are pushed apart; closing cars do not tunnel at representative speeds; the
response is equal-and-opposite and **order-independent** (any agent order / shuffled ids → identical
post-step states); a glancing hit records a smaller impulse than a hard hit; the single-car physics
tests are untouched; `ruff` clean.

### Step C — multiagent-env-engineer · `env/single_agent.py` + `multi_agent.py` (needs A + B)
Split `step_one_car` into `advance_car_physics` + `finalize_car_step` (keep `step_one_car` = advance +
finalize so `RacingEnv` is unchanged); extend `CarRuntime` with the contact record + `prev_rank`;
reorder `RacingParallelEnv.step` to advance → `resolve_collisions` → rank/overtakes →
finalize-with-block; add `rank_and_overtakes` (zero-sum, battle-range gated); wire `CollisionParams` +
the racing weights through `CarStepConfig`; add the crashout branch to `_check_termination`; add the
racing info fields.
**Gate:** `parallel_api_test(RacingParallelEnv(...))` passes with collisions + the block on; a one-car
field reproduces `RacingEnv` (and `reward_v3 == reward_v2`); per-agent obs == single-agent obs for the
same state when alone; the SuperSuit-visible width stays constant under `black_death_v3` when a car
crashes out; `check_env(RacingEnv)` passes at length 42; `ruff` clean.

### Step D — reward-engineer · `env/rewards.py` (parallel with C, needs the ContactRecord shape from B)
Add the racing weights to `RewardWeights`; write `reward_v3` (wrap `reward_v2`, add the graded contact
term + the zero-sum overtake term + the opt-in gap term). Never centerline-seeking; never hand-codes
blame.
**Gate:** `reward_v3` with an empty contact and `places=0` equals `reward_v2`; a contact subtracts a
penalty scaling with closing speed; `+w_overtake` per place gained, `−w_overtake` per place lost,
zero-sum across the pair; `w_gap=0` → no gap term; lateral never enters; `ruff` clean.

### Step E — selfplay-training-engineer · `train/` + `configs/experiment/` (needs C + D)
- `train/warmstart.py` (`grow_policy`) + a `--warm-start` branch in `selfplay.py` (distinct from the
  still-validating `--resume`).
- `curriculum.py` + `SelfPlayCurriculumCallback`: ramp `w_contact`/`w_overtake` in-place via
  `raw_parallel_envs`.
- `selfplay_eval.py`: racing metrics (overtakes/race, contact rate + mean impulse, finishing order,
  `race_position`); start eval in `grid` reset.
- `calendar_racing.yaml`: `collision:`, `reward.version 3` + weights, obs neighbor params, the
  coexist→race curriculum, `wandb` group `phase-6`.
- **Re-run the throughput check** (`selfplay --throughput`) with collisions on: field SPS vs the Phase
  5 blind field vs an equal-width single-agent run.
- Smoke run: a tiny-budget grown-warm-start run completes and the return is non-degenerate.
**Gate:** the grown warm start produces a v3 policy whose action on a no-neighbor obs matches the Phase
5 policy on the same `[0:22]`; the obs stats grow to width 42; a v3 checkpoint round-trips and
`--resume`s; the curriculum ramp moves the per-step reward; an SPS number is reported; `ruff` clean.

### Step F — app-integration-engineer · `sim/` + `server/` + `web/` (needs C's env + E's checkpoint)
Add `race_position` + gap-to-ahead to `FieldSimLoop` and the per-car `telemetry`; list the running
order P1…PN in the timing tower; confirm contact is visible (cars bounce); keep the recorder/replay
format backward compatible. Optional debug overlay (neighbor links, collision discs).
**Gate:** a field races live for a v3 checkpoint with a changing order and real gaps; contact is
visible; a recorded race replays with positions/gaps; the one-car live view still works.

### Step G — test-engineer (independent) · `tests/` (parallel from the spec contracts)
`test_observations.py` (obs prefix unchanged, local/relative block, padding/range);
`test_collisions.py` (push-apart, no-tunnel, order-independence, impulse grading, dead-car exclusion);
`test_multi_agent_env.py` (parallel_api_test with racing on, one-car field == RacingEnv, zero-sum
overtake, crashout opt-in, constant width); `test_rewards.py` (`reward_v3 == reward_v2` baseline,
contact + overtake terms); `test_warmstart.py` / `test_checkpoint.py` (v2 `--resume` refused; grown
warm start preserves the driver + grows vecnorm + round-trips v3); `test_recorder.py` (racing fields
carried + replayed). Written from the spec + public signatures only.

### Step H — reviewer (read-only, gates every merge)
Checklist: config-driven (every weight/geometry in config); SI units + world frame; **obs `[0:22]`
unchanged and `OBS_VERSION = 3`**; **no absolute position in the block**; **`PhysicsModel.step`
untouched**; **collisions order-independent + reproducible**; **`reward_v3` reduces to `reward_v2`**
and never centerline-seeking / hand-codes racecraft; v2 `--resume` refused + grown warm start the
explicit path; per-car `LapTimer` / constant-width SuperSuit / done-agent removal preserved; the
single-agent path + one-car live view unbroken; deterministic seeding for the draw, both reset modes,
and the collision pass; runtime-safe loader (no FastF1 under `env/`/`train/`). Runs `pytest` (incl.
`parallel_api_test`) + `ruff check` + `ruff format --check`. Blocks on any violation or red test/lint.

---

## Dispatch DAG (dependency order)

```
0. scaffold (branch, agent defs, collision/obs/reward config + calendar_racing.yaml)
A. observation-engineer: ObservationV3 neighbor block (OBS_VERSION 3)  ─┐  parallel foundation
B. collision-engineer:   env/collisions.py resolve_collisions          ─┘
C. multiagent-env-engineer: advance/finalize split + reordered field   ──┐ needs A+B
   step + rank/overtakes + crashout   (critical path)                    │  test-engineer (G)
D. reward-engineer:      reward_v3 (contact + zero-sum overtake)        ──┘  starts in parallel,
E. selfplay-training-engineer: grown warm start + reward-weight             writing failing tests
   curriculum + racing metrics + throughput + smoke   (needs C+D)           from the spec contracts
F. app-integration-engineer: race position + gap-to-ahead + tower      (needs C's env + E's ckpt)
H. reviewer (G) gates each merge; final full suite + ruff; PR with the run summary, the curves,
   the racing metrics, the SPS number, and a clip of the grid racing (>=1 clear overtake + defend).
```

observation-engineer + collision-engineer are independent foundations (parallel).
multiagent-env-engineer is the critical path (threads both into the field step). reward-engineer is
parallel with it (needs only the `ContactRecord` shape). selfplay-training-engineer is sequential on
the env + reward. app follows the env + a checkpoint. test-engineer runs concurrently from the
contracts; reviewer gates throughout.

---

## Definition of done (spec §2e, §3b — behavioral, not just green tests)

- The observation is `OBS_VERSION = 3`, `[0:22]` proven byte-identical to v2, the neighbor block
  local/relative and zero-padded — by test.
- Collisions detect + resolve contact (cars bounce, lose speed, **do not tunnel**), order-independent
  and reproducible; `PhysicsModel.step` unchanged.
- `reward_v3` reduces to `reward_v2` with no contact / constant rank; the contact penalty and the
  zero-sum overtake/defend term are config-weighted and verified.
- The grown warm start reproduces the Phase 5 driver before learning the racing; a v2 `--resume` is
  refused; v3 checkpoints round-trip and resume.
- One shared policy **races** a field on a real circuit with **visible, genuine overtaking and
  defending that emerge from the reward** (no hand-coded racecraft); the contact rate is a sane
  non-zero level (committed but not ramming); the running order + real gaps render and replay; the
  field is demonstrated at 2 and 4 cars (full 22 where compute allows).
- An SPS number with collisions on is measured and the field ceiling documented. Full suite + `ruff`
  green. **Unlike Phase 5, the bar is a learning/behavioral gain — the race has to look like a race.**
- `TECHNICAL_DESIGN.md` §5/§7/§9/§10/§12/§15 updated in the same commits that introduce each contract.

---

## Risks & open items

- **Reward balance is THE risk and the budgeted core.** Too much `w_overtake` → ramming/dirty driving;
  too much `w_contact` → timid cars that won't race. Watch contact rate + overtake count **together**;
  ramp via the coexist→race curriculum; every weight in config. This dominates the phase's time — not
  a final polish step.
- **Self-play non-stationarity.** Each car learns against a moving target (itself). Parameter sharing +
  PPO is the mitigation; the warm start + gentle curriculum + more steps are the levers if it diverges;
  revert to the Phase 5 driver if it collapses.
- **Tunneling at the control step.** Fast closing cars could skip past each other in one 0.05 s step.
  The two-disc swept length helps; if it bites, add a sub-step collision check or a swept test (noted,
  not built first). The no-tunnel test gates this.
- **Warm-start surgery correctness.** The transplant must target the right tensors
  (`mlp_extractor.policy_net.0` / `value_net.0` for `MlpPolicy` + `FlattenExtractor`) and grow the
  VecNormalize `obs_rms` consistently, or the "competent driver" property is lost. `test_warmstart.py`
  asserts action-match on a no-neighbor obs.
- **Obs-stat growth vs normalization.** New dims start `mean 0 / var 1`; the zero-init weight columns
  mean the neighbor block contributes 0 at step one regardless — consistent, but verify the grown
  `obs_rms` shape matches the new policy input.
- **O(N²) collision + neighbor cost** on the already-sequential SuperSuit field step. Cheap at N=22 in
  vectorized NumPy but **measured** by the re-run SPS check before scaling; it is the new pressure on
  the §17 JAX trigger.
- **Reproducibility.** The collision pass (snapshot-then-apply) and the overtake ranking (sorted
  totals, ties broken by id) must be deterministic; the circuit draw + both reset modes already use
  `self.np_random` (CLAUDE.md seeding rule) — keep the new passes off any module RNG.
- **Spec/design drift to fix in the same change** — update `TECHNICAL_DESIGN.md` §7 (the neighbor block
  + `OBS_VERSION 3`), §5/§10 (the env collision pass, physics still pure), §9 (contact + overtake/defend
  reward), §12 (racing metrics + grown warm start), §15 (Phase 6 as-built), each in the commit that
  introduces it.
