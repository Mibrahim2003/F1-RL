# Phase 6 Spec: Racing for Real

Status: draft for plan mode. Branch: `phase-6-racing` (to be cut from `main` after Phase 5 merges). Depends on Phase 5 (`phase-5-many-cars.md`) for the multi-agent env, shared-policy self-play, the SuperSuit→SB3 stack, the per-car `CarRuntime` / `step_one_car` factoring, the multi-car live frame, and the field render; on Phase 4 (`phase-4-many-circuits.md`) for the circuit pool, the curriculum mechanism, and the generalist; on Phase 3b for the dynamic physics, the grip pipeline, and the observation; on Phase 3a for the training loop, the checkpoint format, and the live view; and on Phase 1/2 for the app and the cached circuits.

**This is the hardest phase. It carries the most risk and should get the most time.** Phase 5 put a field on track but the cars were blind to each other — its bar was explicitly "infrastructure and the render, not a learning gain." Phase 6 is the inverse: it is where genuine multi-agent learning happens. The cars finally see each other, touch each other, and are rewarded for position. **Wheel-to-wheel racing, blocking, and the accidents must emerge here, and the make-or-break is reward shaping**, the single most-iterated part of the whole project (TECHNICAL_DESIGN.md §9). Budget accordingly.

---

## 1. Introduction

### a. Overview, Problem Description, Summary

Through Phase 5 a full grid laps a circuit together, but it is a parade, not a race. Every car is driven by the same shared policy, each one sees only the track, and none of them knows another car exists: no nearby-car observation, no collision, no contact penalty, no positional reward. They drive past each other and overlap on screen with no consequence. That was a deliberate split (TECHNICAL_DESIGN.md §15) — isolate the multi-agent-env risk (Phase 5) from the racing-interaction risk (Phase 6) so each is debuggable on its own.

Phase 6 closes that gap. It adds the four things that turn a field into a race:

1. **Nearby-car observations** — each car sees the relative position and velocity of the K nearest cars, local and relative only (the block design §7 has reserved since Phase 1). This is the first observation change since Phase 3b, so `OBS_VERSION` bumps **2 → 3**.
2. **Collision detection and response** — cars are solid bodies. They cannot drive through each other; contact pushes them apart and scrubs speed.
3. **The contact penalty** — touching another car costs reward, graded by how hard, so clean racing pays and dirty racing does not.
4. **Overtaking and defending rewards** — a positional, zero-sum term: gaining a place pays, losing one costs. This is what makes a car hunt the car ahead and defend from the car behind.

Suggested solution, in one line: extend the Phase 5 `RacingParallelEnv` with a field-level **collision pass** between the per-car physics and the per-car finalize, append a fixed-length **K-nearest-cars block** to the observation (`OBS_VERSION = 3`), add **contact** and **position-change** terms to a new `reward_v3`, warm-start the Phase 5 competent-driver policy by **growing its input layer** (the obs changed, so a silent resume is refused — this is a deliberate transplant), and let the racecraft emerge from the reward rather than hand-coding any of it.

The hard constraint, restated from §9 and held all the way through this phase: **the racing line, the blocking, and the overtakes must emerge from the reward, never be hand-fed.** We reward outcomes — track position relative to other cars, clean fast laps, the cost of contact — and let PPO discover that late apexes, covering the inside line, and a committed dive into a braking zone are the way to earn them. We do not script a defensive line, a "let-by," or a passing trajectory.

Stakeholders: you, as the developer and primary user. Secondary: future portfolio reviewers, who watch the artifact this whole project was for — a full grid of cars racing a real circuit, fighting for position, seen from above.

### b. Glossary or Terminology

- Nearby-car / neighbor block: the fixed-length tail of the observation encoding the K nearest other cars, each as relative position + relative velocity in the observer's body frame, plus a validity bit. Local and relative only, so it does not break the track-agnostic generalization.
- ObservationV3: ObservationV2 (length 22) with the neighbor block appended. `OBS_VERSION = 3`, `OBS_DIM = 22 + K·F`.
- Collision body: the simplified solid shape a car occupies for contact tests — here two discs (front + rear) swept along the car's body axis (chosen over a single disc, which is a poor fit for a long thin car, and over a full oriented box, which is the later fidelity upgrade).
- Collision pass: the field-level step, run once per control step after every car's physics has advanced and before any car's observation/reward is finalized, that detects overlaps and applies the contact response. It is the one place cars are coupled.
- Contact response: the positional push-apart + velocity impulse (restitution + friction) that separates two overlapping cars and bleeds their closing speed. Equal-mass, reproducible, applied to `CarState`.
- Contact penalty: the per-car reward term that charges for contact this step, graded by the impulse / closing speed.
- Race position / rank: a car's order in the field by total progress (`completed_laps · length + s_along`) among the currently-live cars. Lower rank number = further ahead.
- Overtake / defend reward: the zero-sum position-change term — `+w_overtake` per place gained, `−w_overtake` per place lost — gated to genuine wheel-to-wheel swaps (both cars within a battle range), so overtaking and defending are two sides of one weight.
- Input-layer transplant / grown warm start: warm-starting Phase 6 from the Phase 5 policy by copying every weight except the policy/value input layer, where the columns for the unchanged inputs 0–21 are copied and the new neighbor columns are zero-initialized — so the car drives exactly as well as Phase 5 on step one and only has to learn to use the neighbor block. Distinct from a silent resume (which the loader refuses on a version mismatch).
- Self-play non-stationarity: every car learns against copies of a policy that is itself changing, so the environment each car faces is non-stationary. Parameter sharing + PPO is the standard mitigation; it is the central learning risk of this phase.

### c. Context or Background

- Why worth solving: this is the dream (PROJECT_VISION.md). "Then the field comes alive. Cars defend their position. Cars hunt the car ahead and look for a way past. Some moves work and some end in contact." Everything before Phase 6 was scaffolding for this. The artifact of this phase — a full grid racing with overtaking and defending — is the thing the whole project exists to produce, and the headline of any showcase video or resume line.
- Origin: Phase 6 of the visibility-first build order (TECHNICAL_DESIGN.md §15), the racing-rules section, and the long-reserved nearby-car block of the observation contract (§7) and the contact penalty noted in the reward design (§9, "Contact penalty enters in the multi-agent phase").
- How it affects the goals: it is the core of all three goals. Fun (a real race with surprises every time, the stated reason the project exists), resume (emergent multi-agent racecraft from self-play is a strong, rare headline), and learning (genuine multi-agent RL, self-play non-stationarity, collision physics, and reward shaping for interaction are the deepest skills in the build).
- Past efforts: Phase 3 produced the resumable checkpoint and the competent single-car driver; Phase 4 made it a generalist across the calendar; Phase 5 put N of them on one circuit under one shared policy with the multi-agent env, the per-car `CarRuntime` factoring, and the field render — but blind and non-interacting. Phase 6 reuses all of that and adds sight, contact, and competition on top.
- Roadmap fit: Phase 7 (pit stops and strategy) is the only thing after this and is optional. Phase 6 is the last phase that has to work for the vision to be real. Everything Phase 7 adds is a layer on a working race.
- Technical strategy fit: the observation stays **local and relative** (§7) — the neighbor block is relative position/velocity in the car's own frame, so one policy still drives any circuit with any car around it. The physics step stays a **pure single-car function** (§5) — collisions are resolved by the env in a field-level pass, never inside `PhysicsModel.step`. The reward stays **outcome-based and config-driven** (§9, §14) — every new weight lives in config and the racecraft emerges. The change is real (the first obs bump since 3b, the first inter-car coupling, the first genuine self-play), but it is contained to the env's field-level seam, the observation tail, and the reward.

### d. Goals or Product and Technical Requirements

Product requirements as user stories:

- As the user, I start a race and watch a full grid fight for position — cars catch the car ahead, dive down the inside, and complete real overtakes, so the screen finally shows wheel-to-wheel racing.
- As the user, I see cars defend — a car under attack covers the inside line and holds position, so a pass has to be earned, not handed over.
- As the user, I see contact have consequences — cars that touch bounce apart and lose speed, a clumsy lunge ends in a mistake, and the order reshuffles, so the racing has the honesty the vision asks for.
- As the user, I read the running order — the timing tower shows P1…PN with real gaps to the car ahead, updating as positions change, so I can follow the battle.
- As the user, I can replay a recorded race and watch a specific overtake again, so a good moment is reviewable.

Technical requirements, functional and required:

- **Nearby-car observation, fixed length, local and relative.** Each car's observation gains a fixed-length block encoding the K nearest other cars, each as relative position and relative velocity **in the observer's body frame** plus a validity bit, sorted nearest-first, zero-padded (validity 0) when fewer than K neighbors are within the sensing range. No absolute position ever enters (§7). The block is appended at the tail (indices `[22:]`) so the unchanged track features keep their exact indices `[0:22]`. `OBS_VERSION = 3`, `OBS_DIM = 22 + K·F`, both K and the feature set config-driven.
- **The observation builder stays field-agnostic.** `build_observation` (the heavy NumPy builder reused verbatim by the server) must not learn about the field. It accepts a precomputed neighbor block (defaulting to all-zero = no neighbors) and writes it into the tail. The field env owns "who are my neighbors and how do I encode them"; the single-agent `RacingEnv` passes an empty block, so a lone car simply always observes zero valid neighbors and the same length vector. Homogeneous spaces across both envs (required for the shared policy) are preserved.
- **Collision detection and response, field-level, physics pure.** Cars are solid collision bodies (two discs per car, config-driven geometry). After every car's substep physics has advanced and **before** any car's projection/reward/obs is finalized, a single field-level collision pass detects overlapping live cars and applies a contact response: a positional push-apart to remove penetration and a velocity impulse (restitution + tangential friction) that scrubs closing speed, equal-mass, applied to `CarState` and mapped back into each car's body frame. **`PhysicsModel.step` is not touched** — it stays the pure single-car function of §5; the coupling lives only in the env's collision pass.
- **The per-car step splits to admit the collision pass.** The Phase 5 `step_one_car` runs physics → project → timing → reward → terminate → obs in one body. Phase 6 splits it into **`advance_car_physics`** (map action, grip, substeps — per car, independent) and **`finalize_car_step`** (project, timing, reward including the new contact/position terms, terminate, obs — per car), with the field-level **`resolve_collisions`** between them. `step_one_car` is kept as `advance` + (empty collision) + `finalize` so `RacingEnv` (one car, cannot collide) reproduces its exact Phase 5 behavior. No per-car math is duplicated.
- **Reproducible, order-independent collisions.** The collision pass snapshots all post-physics states, computes every pair's correction against that snapshot, then applies the summed corrections, so the result does not depend on agent iteration order and is reproducible from the seed (CLAUDE.md seeding rule). A single resolution pass per step is accepted (deep pileups may leave a small residual overlap, resolved over subsequent steps — a soft constraint, not a hard solver).
- **Contact penalty, graded and config-driven.** A new `reward_v3` adds a contact term: `−w_contact · contact_cost`, where `contact_cost` is graded by the impulse / closing speed of the contact this step (a light brush costs little, a heavy hit costs a lot), shape and weight from config. Symmetric between the two cars by default; an optional fault-asymmetry weight (default 0) can later charge the car driving into the contact more. Never centerline-seeking; never hand-codes blame.
- **Overtaking and defending reward, zero-sum and emergent.** `reward_v3` adds a position-change term: `+w_overtake` per race place gained this step and `−w_overtake` per place lost, where race position is the rank by total progress among live cars, **gated to genuine wheel-to-wheel swaps** (both cars within a configured battle range when they swap) so lapping and far-apart rank shuffles do not pay. The same weight produces both behaviors: chasing the reward of a place gained is overtaking; avoiding the cost of a place lost is defending. The reward never encodes a racing line, a defensive line, or a passing trajectory — those emerge.
- **Reward balance is the deliverable, not an afterthought.** The contact penalty and the overtake reward pull against each other (passing pays, contact costs), and against the progress core. Too much overtake reward and cars ram or drive dirty; too much contact penalty and cars hang back and refuse to race. Finding the balance that yields committed-but-clean racing is the central work of this phase. Every weight is in config (`w_contact`, `w_overtake`, `w_gap`, the contact-cost shape, the battle range, restitution, the collision geometry) and is expected to be tuned more than any other piece.
- **Warm start by growing the input layer; a silent resume is refused.** Because `OBS_VERSION` moves 2 → 3, `validate_checkpoint` refuses a plain resume of the Phase 5 checkpoint (correct — the obs layout changed). Phase 6 provides a deliberate **grown warm start**: build the Phase 6 policy, copy every Phase 5 weight except the policy/value input layer, copy the input-layer columns for inputs 0–21 and zero-initialize the new neighbor columns, and grow the `VecNormalize` obs statistics to the new width (new dims start mean 0 / var 1). The result drives exactly as well as Phase 5 on the first step and only has to learn the neighbor block. Training from scratch is the documented fallback.
- **Field-size scaling and the circuit pool reused unchanged.** The field still scales as a per-run constant grown across warm-started runs (Phase 5 decision, unchanged), and each episode still draws one circuit from the Phase 4 pool shared by the whole field. Per-car lap timers and per-car runtime state remain per car; the pool entry stays read-only (`Track`/`EdgeCache`/pole).
- **Curriculum for racing.** Introduce the racing pressure gradually rather than all at once on top of a blind driver. The curriculum mechanism (Phase 4/5, in-place) is extended to ramp **reward weights** (`w_contact`, `w_overtake`) over timesteps in addition to conditions and the circuit pool, so a run can learn to coexist without crashing before it is pushed to fight for position. (Cross-run field-size staging stays the mechanism for field size; the existing conditions + pool widening stay in-place.)
- **Termination on contact is opt-in.** By default contact is penalized but not terminal — cars race through it. A config-gated severe-crash termination (a single contact whose impulse/closing speed exceeds a threshold ends that car's episode with the failure penalty) is available and **disabled by default**, so a big accident taking a car out can be turned on once the racing is stable. Done cars leave the live set and are excluded from the collision and neighbor passes (no ghost-car pileups; a stopped car as a track hazard is future work).
- **The multi-agent API and constant-width training still hold.** `RacingParallelEnv` still passes `pettingzoo.test.parallel_api_test`, still removes done agents from `self.agents` on the next step, and still relies on `black_death_v3` for the constant SuperSuit-visible width. The neighbor and collision passes operate only on live agents; the padding of dead agents is the wrapper's concern, unchanged.
- **Race-aware live view and replay.** The live frame and recorder still carry the `cars[]` array (Phase 5), now with a race **position** and a real **gap to the car ahead** per car; the timing tower lists the running order P1…PN; contact is visible because the cars actually bounce. Replay of a multi-car race is unchanged in format.
- Deterministic seeding, the seed recorded in every run and checkpoint; the circuit draw, both reset modes, the collision pass, and any contact stochasticity reproducible from the seed.
- Runs on the laptop CPU with no GPU, headless and device-agnostic, reusing the Phase 5 training setup; the full 22-car race reaches the cloud if the laptop cannot.

### e. Non-Goals or Out of Scope

- **No high-fidelity collision model.** Two discs per car with an equal-mass impulse response is the deliberate first model. A full oriented-bounding-box or convex-shape collision, contact-induced yaw spin, and momentum transfer by real per-car mass are explicit fidelity upgrades for later, taken only if the feel needs them (mirrors kinematic-before-dynamic and the deferred Pacejka model, §17).
- **No hand-coded racecraft.** No scripted blocking line, no scripted defensive moves, no scripted passing trajectory, no rule that one car yields to another. All of it emerges from the reward (§9 is load-bearing here).
- **No team orders, no driver personalities, no heterogeneous skill.** Every car is the same shared policy. Per-car strategy or skill differences are future work.
- **No racing flags, penalties, or stewarding.** No blue flags, no track-limits enforcement beyond the existing off-track reward, no penalty for causing a collision beyond the contact reward term (the optional fault-asymmetry weight is the only nod to blame, and it defaults off).
- **No pit stops, tire-strategy decisions, or fuel.** Phase 7.
- **No new track data, no physics-interface change, no action-space change.** The action stays `Box(-1,1,(2,))`. The dynamic model and grip pipeline are reused unchanged behind `PhysicsModel`.
- **No stopped-car-as-obstacle.** A car that crashes out or finishes leaves the collision/neighbor field for the rest of the episode; it is not left on track as a hazard.
- No JAX/GPU env (the O(N²) collision + neighbor cost on top of the already-sequential SuperSuit field is a new pressure on the §17 trigger, noted but not acted on here). No mobile.

### f. Future Goals

- Higher-fidelity collisions (oriented boxes, contact spin, real per-car mass and momentum) if the racing feel needs it.
- Fault-aware penalties and simple stewarding (track-limits, causing-a-collision) once clean racing is stable.
- A stopped or damaged car as a persistent track hazard (debris, a slow car to avoid).
- Heterogeneous fields — per-car compound, tire age, or strategy differences — now that cars interact, feeding Phase 7 strategy.
- Phase 7: pit stops and strategy, layered on the working race.
- Density/aggression curriculum (tighter grids, closer starts, ramped overtake reward) tuned for the kind of racing you want to watch.

### g. Assumptions

- Phase 5 is complete and merged: `RacingParallelEnv`, the per-car `CarRuntime` / `step_one_car` / `reset_car` factoring, `make_selfplay_vec_env`, `train/selfplay.py`, the multi-car eval driver, the `cars[]` live frame and field render, and the `n_agents` checkpoint field all work, and a shared policy laps a circuit with a field.
- The Phase 5 (or Phase 4) checkpoint is a competent driver whose observation indices 0–21 are exactly the Phase 6 indices 0–21, so the grown warm start is a sound transplant (the new block is purely additive at the tail).
- PettingZoo / SuperSuit / SB3 / Gymnasium / PyTorch are the pinned working matrix from Phase 5 (`pettingzoo==1.26.1`, `supersuit==3.11.0`, `stable-baselines3==2.9.0`, `gymnasium 1.3.0`, `torch 2.12.0`); no new third-party dependency is required (collisions and neighbor search are NumPy).
- The dynamic physics, the grip pipeline, and the circuit pool are unchanged and reused.
- The laptop has the cores and RAM to train a small interacting field; the full 22-car race reaches the cloud if not. The O(N²) collision/neighbor cost is small at N=22 with vectorized NumPy but is measured, not assumed (the throughput check is re-run with collisions on).
- Self-play with parameter sharing and PPO is stable enough to learn interactive racing; the curriculum and the warm start are the levers if it is not.

---

## 2. Solutions

### a. Current or Existing Solution

Phase 5 runs a blind field. `RacingParallelEnv` (`env/multi_agent.py`) holds N cars on one circuit, and its `step` loops `step_one_car` over the live agents independently — each car reads only its own `CarRuntime` and the shared read-only circuit, computes its own ObservationV2 (length 22, no neighbor data), its own progress reward (`reward_v1`/`reward_v2`), and its own termination. No car's state ever touches another's. The cars overlap on screen with no physical interaction. The live frame and recorder carry a `cars[]` array and the tower lists every car by a track-position gap, but there is no race position and no real gap to the car ahead.

- Pros: a working multi-agent env that passes `parallel_api_test`, a shared policy warm-started from the generalist, the per-car factoring already in place (`advance`/`finalize` is a clean split of an already-factored unit), and the field render and multi-car recorder done.
- Cons: it is a parade. No car sees another, nothing collides, nothing is at stake. The central image of the vision — cars fighting for position — cannot be shown, and the genuine self-play dynamic the project is for is dormant.

### b. Suggested or Proposed Solution

- **Add the neighbor block to the observation (`OBS_VERSION = 3`).** Append a fixed-length K-nearest-cars block to the observation tail, encoded local/relative in the observer's body frame, zero-padded with a validity bit. Keep `build_observation` field-agnostic: it takes a precomputed block (default zeros) and writes it into `[22:]`. The field env computes each car's block from the other cars' states; the single-agent env passes zeros. `observation_space()` returns the new length for both envs.
- **Resolve collisions in a field-level pass.** Split the per-car step into `advance_car_physics` and `finalize_car_step`, and run `resolve_collisions` between them. Cars are two discs each; overlapping live pairs are pushed apart and given a restitution+friction impulse against a snapshot of the post-physics states (order-independent, reproducible). `PhysicsModel.step` is untouched. Each car's contact record (impulse, closing speed, count) is stashed on its `CarRuntime` for the reward and the info.
- **Add contact and position terms to `reward_v3`.** `reward_v3` = the `reward_v2` progress core + a graded contact penalty (from the car's contact record) + a zero-sum position-change term (from the field's race ranking, gated to genuine swaps). Single-agent and a one-car field both reduce `reward_v3` to `reward_v2` (no contact, constant rank). Every weight is config.
- **Warm-start by growing the input layer.** Because the obs grew, the loader refuses a silent resume; instead a dedicated utility transplants the Phase 5 policy — all weights except the input layer copied directly, the input layer's unchanged columns copied and the neighbor columns zero-initialized, and the `VecNormalize` obs stats grown — so the Phase 6 policy starts as a competent driver and learns only the racing on top. From scratch is the fallback.
- **Race the field with a curriculum.** Reuse the Phase 5 self-play stack (`make_selfplay_vec_env`, `train/selfplay.py`) unchanged in shape. Extend the curriculum to ramp the reward weights (`w_contact`, `w_overtake`) over timesteps so a run learns to coexist before it learns to fight, and keep the field-size scaling across warm-started runs and the conditions/pool widening in place.
- **Make the view race-aware.** Add race position and a real gap-to-ahead to the per-car frame, list the running order in the tower, and let contact be visible through the bounce. The frame and recorder formats stay backward compatible (a `cars[]` array with extra per-car fields).

External components the solution interacts with or alters: the Python env (`multi_agent.py`, `single_agent.py`), the observation builder (`observations.py`), the reward (`rewards.py`), the training scripts and curriculum (`train/`), the checkpoint warm-start path, the live backend and frontend, the recorder/replay format, Weights and Biases, and the local filesystem. No new third-party dependency.

Pros of the proposed solution: real racing emerges, the observation stays local/relative and the physics stays pure (both generalization and the swappable-physics contract preserved), the warm start keeps the competent driver instead of throwing it away, and the env becomes the exact substrate Phase 7 strategy plugs into.

Cons of the proposed solution: this is the first obs bump since 3b, so the clean Phase 5 resume is gone and the warm start is a deliberate transplant (more code, a real test surface); self-play is genuinely non-stationary now, so training can be unstable and slow to converge on good racecraft; **reward balance is a hard, iterative tuning problem with degenerate failure modes on both sides** (ramming vs. timidity); and the O(N²) collision/neighbor cost adds to the already-sequential SuperSuit field-step cost. None of these is avoidable — they are the substance of "the hardest phase."

#### Data Model and Schema Changes

```
ObservationV3:                 # CHANGED — OBS_VERSION 2 -> 3, OBS_DIM 22 -> 22 + K*F
  [0:22]   ObservationV2, UNCHANGED   # track-local features keep their exact indices
  [22:22+K*F]  neighbor block          # K nearest cars, nearest-first, zero-padded
    per neighbor (F features, body frame of the observer):
      dx_body / R        # relative position forward,  clipped [-1, 1]
      dy_body / R        # relative position left,      clipped [-1, 1]
      dvx_body / ref     # relative velocity forward,   clipped [-2, 2]
      dvy_body / ref     # relative velocity left,      clipped [-2, 2]
      valid              # 1 if this slot holds a real neighbor within range R, else 0
  # defaults (all config): K = 4, F = 5, sensing range R = 50 m. -> OBS_DIM = 42.

Action:                        # UNCHANGED, Box(-1, 1, shape=(2,))

collision: config block (new)
  enabled: true
  body: two_disc               # two_disc (default) | disc  (obb = future)
  disc_radius_m: 1.0           # per-disc radius
  disc_offset_m: 1.25          # front/rear disc offset along the body axis from center
  restitution: 0.1             # 0 = fully inelastic, 1 = elastic (F1 contact ~ low)
  friction: 0.3                # tangential damping at contact, [0,1]
  push_fraction: 1.0           # fraction of penetration removed per step, split evenly
  crashout_enabled: false      # opt-in severe-crash termination
  crashout_closing_speed_mps:  # threshold; a contact above this ends the car (when enabled)

reward: config block (extended) — reward_v3 = reward_v2 core + contact + position
  version: 3
  # contact penalty
  w_contact: <float>
  contact_soft_mps: <float>    # closing-speed scale; small brush << hard hit
  contact_exp: <float>         # >1 makes light contact cheap, heavy contact costly
  w_contact_fault: 0.0         # optional extra share for the car driving into contact (default 0)
  # overtaking / defending (zero-sum position change)
  w_overtake: <float>          # + per place gained, - per place lost (genuine swaps only)
  overtake_battle_range_m: <float>  # both cars within this track-gap to count the swap
  # optional dense gap shaping (opt-in, like w_slip; default 0)
  w_gap: 0.0
  # (unchanged v1/v2 terms: w_progress, w_offtrack, w_step, w_reverse, offtrack_*, w_slip, slip_threshold)

CarRuntime (extended, per car):     # Phase 5 fields + the contact record this step
  ... state, lap_timer, prev_s, grip_idx, grip_lat, wrong_way_count, t, step_count, done ...
  contact_impulse: float = 0.0      # aggregate impulse magnitude this step
  contact_closing_mps: float = 0.0  # max closing speed of any contact this step
  contact_count: int = 0            # number of contacts this step
  prev_rank: int = 0                # race rank last step, for the position-change term

curriculum stage (extended): may also carry reward-weight overrides
  start_step, mu_base, wear_rate, weather, circuits (all Phase 4/5, unchanged)
  w_contact: float | None           # NEW: ramp the contact penalty in-place
  w_overtake: float | None          # NEW: ramp the overtake reward in-place

Per-agent step info (extended): adds racing fields for the view + metrics
  ... lap_time, off_track, progress, completed_laps, ... (Phase 5) ...
  race_position: int                # rank in the field (1 = leader)
  gap_ahead_s: float | None         # time/!distance gap to the car directly ahead
  contact: float                    # impulse magnitude this step (0 = clean)
  overtakes: int                    # places gained this step (for metrics)

Checkpoint meta:
  obs_version                       # NOW 3 — a Phase 5 (v2) checkpoint is refused for silent resume;
                                    #   the grown warm start is the explicit, separate path
  n_agents                          # field size (Phase 5, unchanged)
  ... rest of the §12 schema unchanged ...
```

Modified data: the observation grows a neighbor block and bumps its version; a new `collision:` config block; `reward_v3` and its weights; the `CarRuntime` contact record and previous rank; the curriculum stage gains optional reward-weight overrides; the per-agent info gains racing fields. Unchanged: the action shape, the physics interface and the dynamic model, the circuit pool entries (read-only Track/EdgeCache/pole), the per-car lap timer, the checkpoint format other than `obs_version`, and the constant-width SuperSuit mechanism. Validation: the checkpoint loader **refuses** an `obs_version` 2 checkpoint for a normal resume (correct on a layout change) and the grown warm start is the explicit transplant path; the env still refuses a non-positive `n_agents` and an unbuilt circuit id; the collision block validates its geometry (positive radius, non-negative restitution/friction in range).

#### Business Logic

- **Env construction:** as Phase 5 (pool once, shared read-only circuit, per-car runtime), plus build the collision params and the neighbor-block params once from config. `observation_space()` now returns the length-`(22+K·F)` Box for both envs.
- **Reset:** as Phase 5 (draw one circuit for the field, resolve weather, place cars by `reset_mode`, per-car own lap timer + projection state), plus initialize each car's contact record to zero and seed `prev_rank` from the start placement's race order. Each car's first observation includes its neighbor block (computed from the placed field).
- **Step (the reordered field step):**
  1. **Advance** each live car's physics independently (`advance_car_physics`: map action → grip → substeps). No projection, no reward yet.
  2. **Resolve collisions** once over the live field (`resolve_collisions`): snapshot post-physics states, find overlapping two-disc pairs, compute equal-mass push-apart + restitution/friction impulses against the snapshot, apply the summed corrections to each `CarState` (mapping the world-frame velocity change back into body frame), and write each car's contact record.
  3. **Rank** the live field by total progress, gated battle ranges noted, to get each car's `race_position` and the per-car places gained/lost vs `prev_rank`.
  4. **Finalize** each live car (`finalize_car_step`: project once, lap timing, build the neighbor block from the post-collision field, `reward_v3` using the contact record + the position change, wrong-way, termination including the optional crash-out, build ObservationV3, assemble info with the racing fields). Update `prev_rank`.
  5. Drop done agents from `self.agents` for the next step (standard PettingZoo, unchanged); the episode ends when the live set is empty or the step limit fires.
- **Single-agent path:** `RacingEnv` calls `advance` then `finalize` with no collision pass and an empty neighbor block; `reward_v3` with one car has no contact and a constant rank, so it equals `reward_v2`. The env checker still passes (at the new obs length). A one-car field reproduces this.
- **Self-play training:** unchanged in shape (`make_selfplay_vec_env`, parameter sharing, `black_death_v3`), warm-started by the grown transplant from the Phase 5 policy. The curriculum now also ramps `w_contact`/`w_overtake` in-place (reward weights are read each step from the shared step config, so an in-place swap takes effect from the next step — same transport as conditions).
- **Field scaling:** across warm-started runs, unchanged (a per-run constant `n_agents`).
- **Error states:** a non-positive `n_agents`, an unbuilt circuit id, an invalid collision geometry, and a silent resume of a v2 checkpoint are all refused with specific messages. The grown warm start refuses a source whose non-neighbor layout is not a prefix of the target (so a genuinely incompatible policy cannot be transplanted).
- **Failure scenarios:** the policy learns to ram (raise `w_contact` or ramp the overtake reward in later, slow the field-size step-up); the policy learns to hang back and not race (lower `w_contact`, raise `w_overtake`, tighten the grid); self-play diverges (revert to the warm start, more steps, gentler curriculum); the laptop stalls on throughput (cloud, smaller field, re-run the SPS check with collisions on).
- **Limitations:** equal-mass two-disc collisions, single-pass resolution (soft residual overlap in pileups), no contact spin, no stewarding, and a homogeneous field.

#### Presentation Layer

- User requirements: watch a grid race for position, read the running order and real gaps, see contact happen and cost time, and replay a battle.
- UI changes: the per-car frame gains `race_position` and `gap_ahead_s`; the timing tower lists the running order P1…PN with the gap to the car ahead (the Phase 5 track-position gap becomes a true race gap by position); contact needs no new primitive — the cars visibly bounce because their states actually change. An optional debug overlay can draw the neighbor links and the collision discs.
- Web concerns: the same WebSocket and `cars[]` frame, a few more scalar fields per car. The canvas already interpolates per car; positions and gaps are HUD text. No new rendering primitive is required for the core artifact.
- UI states: a race running live with a changing order, and a recorded race replaying with the same order/gaps.
- Error handling: a clear message when a circuit or checkpoint cannot load; the viewport never crashes on a partial or bad frame; a missing pole degrades the gap display gracefully.

#### Other questions to answer

- How will it scale: the collision pass is O(N²) pair tests and the neighbor block is O(N²) nearest-neighbor, both vectorized in NumPy and cheap at N=22, but they sit on top of the already-sequential SuperSuit field step, so the SPS check is re-run with collisions on to re-confirm the field ceiling and the laptop-vs-cloud line before scaling up. The checkpoint stays resumable and portable into Phase 7.
- Limitations: as above; the JAX env (§17) is the real lever if the interacting-field throughput becomes the wall, and the O(N²) cost is the new pressure on that trigger.
- Recovery on failure: resume from the last checkpoint (saved often); the grown warm start and the curriculum are the levers if the racing degenerates.
- Future requirements: this is the substrate for Phase 7 — pit entry/exit cars, strategy differences, and tire-state heterogeneity all plug into the per-car runtime, the neighbor block, and the contact pass built here.

### c. Test Plan

Tests are written by an independent test subagent from this spec and the public interfaces, not from the implementation, so they verify the requirements and stay unbiased. See section 5.

- **Observation tests:** `OBS_DIM == 22 + K·F`, `OBS_VERSION == 3`; the `[0:22]` slice of a car with no neighbors is **byte-identical** to the Phase 5 ObservationV2 for the same state on the same circuit (the unchanged-prefix property the warm start relies on); the neighbor block is local/relative — translating and rotating the whole field leaves each car's block invariant up to the observer's frame, and **no absolute position** appears; the block is sorted nearest-first, zero-padded with `valid=0` when fewer than K neighbors are within R, and a neighbor beyond R is excluded; a single car (and a one-car field) observe an all-zero block; the per-agent observation still lies in the declared Box.
- **Collision tests:** two cars placed overlapping are pushed apart so they no longer overlap after a step; two cars on a head-on/closing path do not pass through each other (no tunneling at the control step for representative speeds); the contact response conserves the expected symmetry (equal-mass, equal-and-opposite correction) and **is order-independent** (resolving the field in any agent order, or with shuffled ids, yields the same post-step states — reproducible from the seed); a glancing contact costs less than a hard hit (the contact record's impulse/closing speed scales correctly); done/removed cars are excluded from collisions; `PhysicsModel.step` is unchanged (the single-car physics tests still pass).
- **Reward tests:** `reward_v3` with no contact and a constant rank equals `reward_v2` (so the single-agent path and a one-car field are unchanged); a contact this step subtracts a penalty graded by the impulse/closing speed; a place gained adds `+w_overtake` and a place lost subtracts it, **zero-sum** across the two cars that swap, and only when both are within the battle range; the dense gap term is zero at its default weight; the lateral offset still never enters the reward.
- **Warm-start tests:** `validate_checkpoint` **refuses** an `obs_version=2` checkpoint for a silent resume with a specific message; the grown warm start produces a Phase 6 policy whose action on a no-neighbor observation **matches the Phase 5 policy's** action on the same `[0:22]` features (the transplant preserves the driver), grows the `VecNormalize` obs stats to the new width, and round-trips as a fresh Phase 6 checkpoint (`obs_version=3`); a transplant from an incompatible layout is refused.
- **Multi-agent API tests:** `RacingParallelEnv` still passes `parallel_api_test` with collisions and the neighbor block on; the SuperSuit-visible width stays constant under `black_death_v3` when a car crashes out early; per-agent dicts still carry the racing info fields.
- **Integration tests:** a short self-play smoke run with collisions + `reward_v3` completes without error and the return is non-degenerate; the grown warm start trains without an obs-version error; a checkpoint resumes and continues; the curriculum ramps `w_contact`/`w_overtake` in-place and the change is observed in the per-step reward; the app drives a multi-car race live and a recorded race replays with positions and gaps.
- **Throughput check (re-run):** SPS of the interacting field (collisions + neighbor block on) vs the Phase 5 blind field and vs an equal-width single-agent `n_envs` run, to re-size the field ceiling with the new O(N²) cost before scaling.
- **QA / behavioral (the real bar):** confirm that over training, on a held-out eval race, the field produces **genuine overtakes and defends** (positions change through wheel-to-wheel moves, not just pace), the **contact rate trends to a sane non-zero level** (not zero = too timid, not constant = ramming), cars **do not tunnel** through each other, and the racing looks believable to watch. This behavioral bar — not just a green test suite — is the definition of done.

### d. Monitoring and Alerting Plan

- Logging: Weights and Biases primary, local logs fallback (unchanged).
- Metrics: mean/per-car lap time and delta-to-pole; **overtakes per race and per car**; **contact rate and mean contact impulse**; **positions gained/lost and the final finishing order**; the race-gap spread; off-track count; episode return per car and aggregated; the ramped reward weights (`w_contact`, `w_overtake`) over timesteps; steps per second with collisions on; and the learning curves. The eval driver (`selfplay_eval.py`) is extended to compute the racing metrics and to start in `grid` reset for a clean racing demo.
- Observability: the learning curves, periodic **multi-car race eval clips** on a rotating circuit (now showing real position changes), and the on-screen debug overlay (neighbor links, collision discs, running order).
- Alerting: none external. Watch the contact rate and the overtake count together — a contact rate spiking with overtakes flat is ramming (raise `w_contact` / slow the overtake ramp); overtakes near zero with contact near zero is timidity (raise `w_overtake` / tighten the grid). A collapsing return when the racing pressure ramps is the signal to soften the curriculum or revert the warm start.

### e. Release, Roll-out, and Deployment Plan

- Branch `phase-6-racing`. Merge when one shared policy **races** a field on a real circuit with **visible, genuine overtaking and defending**, contact is detected and penalized and the cars do not tunnel, the running order and real gaps render and replay correctly, the grown warm start reproduces the Phase 5 driver before learning the racing, the obs-version bump and the refused-silent-resume hold under test, and the suite + linter are green. Demonstrate at least 2 and 4 cars racing (the full 22 where compute allows). The PR carries the run summary, the curves, the racing metrics, and a clip of the grid racing with at least one clear overtake and one clear defend.

### f. Rollback Plan

- Liabilities: the obs bump invalidates a silent Phase 5 resume (intended, but a real break if the warm start is wrong); the collision pass couples cars and could destabilize training or tunnel at speed; the reward change could produce degenerate racing; and the frontend change could break the Phase 5 field view.
- Reduce liabilities: keep main working, develop on the branch, tag the working commit, checkpoint often, and keep the last good Phase 5 checkpoint and policy.
- Prevent spread: `PhysicsModel.step`, the action space, the circuit pool, the per-car lap timer, and the constant-width SuperSuit mechanism are unchanged, so the physics, the tracks, and the training plumbing keep working independently of the racing change. The single-agent `RacingEnv` and a one-car field reduce `reward_v3` to `reward_v2` and observe an empty neighbor block, so the solo path is a clean fallback. The collision pass is behind `collision.enabled` and the racing terms behind their weights, so the field can be reverted to the Phase 5 blind parade by config alone for triage. Revert the merge or restore the tagged commit + the last good checkpoint if needed; delete the branch after a clean merge and a passing suite.

### g. Alternate Solutions or Designs

- **Neighbor block appended at the tail (obs bump) vs. a from-scratch retrain vs. a frozen-driver + separate racing head.** Appending at the tail with a grown warm start is chosen: it keeps the unchanged track features at their exact indices, preserves the competent driver, and only learns the racing on top. A from-scratch retrain is the fallback (correct but wasteful). A frozen driving body with a separate racing head was rejected as more architecture for no clear gain when the input-layer transplant already achieves "keep the driver, learn the racing."
- **Two discs per car vs. one disc vs. an oriented box.** Two discs are chosen: nearly as cheap as one disc but a far better fit for a long thin car (it distinguishes rear-end from side contact), where a single disc makes a long car a fat circle and an OBB is the later fidelity upgrade taken only if the feel needs it.
- **Env-level collision pass vs. collisions inside the physics step.** The field-level pass is required: `PhysicsModel.step` is a pure single-car function by design (§5), and collisions couple cars, so they cannot live inside the step without breaking the swappable-physics contract. The reordered field step (advance → collide → finalize) is the seam.
- **Zero-sum position-change overtake reward vs. dense gap-closing vs. pure emergence.** The zero-sum position term is chosen: it directly rewards both overtaking and defending with one weight, sums to ~0 across the field (no reward inflation, no incentive to game the metric), and ties to genuine swaps. Dense gap-closing is kept as an opt-in (`w_gap`, default 0) because it bootstraps early learning but risks rewarding tailgating without committing. Pure emergence (no positional term, only "go fast + don't crash") was rejected because §15 explicitly asks for overtaking and defending rewards and the sparse signal is hard to learn from alone — but the line against hand-coding *behavior* (lines, blocking, yields) is held absolutely; we reward the outcome, not the maneuver.
- **Symmetric contact penalty vs. fault-based blame.** Symmetric is the default — it is simple, robust, and avoids a brittle fault model — with an optional fault-asymmetry weight (default 0) to charge the car driving into contact more, turned on only once clean racing is stable.
- **Reward-weight curriculum (coexist → race) vs. full racing pressure from step zero.** Ramping `w_contact`/`w_overtake` in-place is chosen so a warm-started driver learns to coexist without crashing before it is pushed to fight; full pressure from the start is the fallback if the ramp adds little.
- **Migration:** a one-car field and the single-agent env reduce `reward_v3` to `reward_v2` and observe an empty neighbor block, and `collision.enabled=false` plus zero racing weights reproduce the Phase 5 blind field exactly — so the change is backward compatible at the env contract and reversible by config.

---

## 3. Success Evaluation

### a. Impact

- Security: local training and a localhost WebSocket, no exposure, validated checkpoints (a v2 checkpoint refused for silent resume, the grown warm start the explicit path).
- Performance: the dynamic per-car step is unchanged, but the field step adds an O(N²) collision pass and an O(N²) neighbor-block build, both vectorized NumPy and cheap at N=22, sitting on the already-sequential SuperSuit field step. Re-run the SPS check with collisions on to re-confirm the ceiling; this O(N²) cost is the new pressure on the §17 JAX trigger.
- Cost: zero on the laptop, free tiers on the cloud if used; the full 22-car race and the longer self-play convergence are the likely reasons to use the cloud.
- Impact on other components: delivers the genuine racing env and the emergent racecraft that the whole project was built toward, and the exact substrate Phase 7 strategy plugs into. The observation version moves to 3 (the first bump since 3b), forcing the warm-start transplant — every later checkpoint is `obs_version=3`.

### b. Metrics

- Capture: overtakes per race and per car, contact rate and mean impulse, positions gained/lost and the finishing order, race-gap spread, mean/per-car lap time and delta-to-pole, off-track count, episode return, the ramped reward weights, and SPS with collisions on.
- Tools: Weights and Biases, the extended field eval clips (now showing real position changes), and the on-screen debug overlay.
- **Definition of done (behavioral, not just green tests):** one shared policy races a field on a real circuit with **visible, genuine overtaking and defending** that emerge from the reward (no hand-coded racecraft); contact is detected and penalized, cars bounce and lose time and do not tunnel; the contact rate is a sane non-zero level (committed but not ramming); the running order and real gaps render and replay correctly; the grown warm start reproduces the Phase 5 driver before learning the racing; the obs bump and the refused silent resume hold; the field is demonstrated at 2 and 4 cars (the full 22 where compute allows); and the suite + linter are green. **Unlike Phase 5, the bar is a learning/behavioral gain — the race has to actually look like a race.**

---

## 4. Deliberation

### a. Discussion

- The single biggest open question is reward balance: the relative weights of progress, contact penalty, and overtake reward that yield committed-but-clean wheel-to-wheel racing, and whether the dense gap term is needed to bootstrap it. This is expected to dominate the phase's time.
- The collision fidelity that looks right from above — whether two discs suffice or the racing needs an oriented box and contact spin to feel honest.
- The K, the sensing range R, and the per-neighbor feature set that give the car enough awareness to race without bloating the observation or hurting generalization.
- How much racing pressure a warm-started driver can take at once, and the curriculum cadence (coexist → race, field 2 → 4 → 22) that keeps self-play stable.
- Whether self-play with parameter sharing converges to interesting racecraft or to a dull equilibrium (everyone drives the same line and no one passes), and what breaks the symmetry if it does (start crowding, the overtake reward, opponent variety).

### b. Decided before coding (resolve in the plan / review)

- **The observation bumps to `OBS_VERSION = 3`**, with the neighbor block appended at the tail so `[0:22]` is unchanged; a silent resume of a v2 checkpoint is refused and the grown input-layer warm start is the explicit transplant path.
- **Collisions are a field-level pass** between `advance_car_physics` and `finalize_car_step`; `PhysicsModel.step` is untouched. Two discs per car, equal-mass impulse, snapshot-then-apply for order-independence.
- **`reward_v3` = `reward_v2` core + graded contact penalty + zero-sum position-change term**, all weights in config; a one-car field/single agent reduces it to `reward_v2`. No hand-coded racecraft.
- **Contact is penalized but not terminal by default**; severe-crash termination is opt-in (`collision.crashout_enabled`, default off). Done cars leave the collision/neighbor field.
- **The curriculum gains reward-weight ramping** (`w_contact`, `w_overtake`) in-place; field-size scaling stays cross-run; conditions and pool widening stay in-place.

### c. Open Questions

- The exact reward weights and curriculum cadence (set during training; the most-iterated values).
- The collision geometry (disc radius/offset), restitution, and friction that look right from above.
- K, R, and the per-neighbor feature encoding (body-frame dx/dy/dvx/dvy + valid vs. a range/bearing encoding — the latter is more rotation-natural but has a wrap discontinuity behind the car).
- Whether the dense gap term (`w_gap`) is needed to bootstrap, and at what weight.
- Whether the symmetric contact penalty suffices or the fault-asymmetry weight is needed for clean racing.
- The measured throughput ceiling for the interacting full field on the laptop vs the cloud (sized by the re-run SPS check).
- Whether a contact-induced yaw spin is needed for the racing to feel honest, or whether the velocity-only response is enough for the top-down view.

---

## 5. Implementation Subagents

These become definitions under `.claude/agents/` (replacing or extending the Phase 5 roster for this branch). Each has a narrow scope and its own context, which saves tokens, since the main thread never carries the whole job. The test author is independent, so tests verify the spec and stay unbiased.

How they work together. The plan you build dispatches tasks in dependency order. Feature subagents read this spec and the technical design. The test subagent reads only this spec and the public interfaces, not the implementation. The reviewer gates each merge.

`/caveman` is **opt-in** for these subagents, not a forced pre-call (the caveman rules already exempt code, commits, and PRs — the bulk of subagent output). Set it in an agent file only where terse status chatter actually helps; default off.

### Roster

- **observation-engineer.** Scope: `src/f1rl/env/observations.py` (+ the obs-version constant). Appends the fixed-length K-nearest-cars block to ObservationV3, keeping the builder **field-agnostic**: a new `build_neighbor_block(observer_state, others, params)` encodes neighbors local/relative in the observer's body frame (sorted nearest-first, zero-padded with a validity bit, capped at range R), and `build_observation` takes a precomputed block (default zeros) and writes it into `[22:]`. Bumps `OBS_VERSION` 2 → 3 and `OBS_DIM`, extends `observation_space()`. Inputs: this spec §2 data model, design §7. Tools: read/edit, run the obs tests. Output: ObservationV3 with `[0:22]` byte-identical to v2 and the neighbor block local/relative and zero-padded — the property the warm start relies on.

- **collision-engineer** (physics-adjacent; the physics role returns this phase). Scope: a new `src/f1rl/env/collisions.py` (+ the `collision:` config block). Builds the two-disc collision-body model and `resolve_collisions(cars, params)` — a field-level pass that snapshots post-physics states, detects overlapping live pairs, applies equal-mass push-apart + restitution/friction impulses against the snapshot (order-independent, reproducible), maps the world-frame velocity change back into each car's body frame, and writes each car's contact record. **Does not touch `PhysicsModel.step`.** Inputs: this spec §2, design §5 (physics purity), §3 (units/coords). Tools: read/edit, run the collision tests. Output: a pure, reproducible collision pass that pushes cars apart, scrubs closing speed, and records contact — physics interface unchanged.

- **multiagent-env-engineer.** Scope: `src/f1rl/env/single_agent.py` + `multi_agent.py`. Splits `step_one_car` into `advance_car_physics` + `finalize_car_step` (keeping `step_one_car` = advance + empty-collision + finalize so `RacingEnv` is unchanged), reorders `RacingParallelEnv.step` to advance → resolve_collisions → rank → finalize, threads each car's neighbor block (from the post-collision field) into `finalize`, computes the race ranking + per-car place changes, and adds the racing fields to the per-agent info. Reuses the per-car core, the per-car lap timer, the read-only pool entry, and the standard PettingZoo done-agent removal + `black_death_v3`. Inputs: this spec §2 business logic, design §10. Tools: read/edit, run `parallel_api_test`. Output: a field env that detects/resolves contact, gives each car its neighbor block, ranks the field, passes `parallel_api_test`, and reduces to Phase 5 on a one-car field.

- **reward-engineer.** Scope: `src/f1rl/env/rewards.py`. Adds `reward_v3` = the `reward_v2` core + a graded contact penalty (from the contact record) + the zero-sum position-change term (gated to genuine swaps) + the opt-in dense gap term, every weight in config, reducing to `reward_v2` with no contact and a constant rank. Never centerline-seeking, never hand-codes blame or a racing line. Inputs: this spec §2, design §9. Tools: read/edit, run the reward tests. Output: `reward_v3` with the contact and overtake/defend terms and their config weights, equal to `reward_v2` in the single-car case.

- **selfplay-training-engineer.** Scope: `src/f1rl/train/` (the grown warm start, the curriculum extension, `selfplay.py`/eval) + `configs/experiment/`. Implements the **input-layer transplant** warm start (copy all weights except the policy/value input layer, copy inputs 0–21, zero-init the neighbor columns, grow the `VecNormalize` obs stats), extends the curriculum to ramp `w_contact`/`w_overtake` in-place, extends the field eval driver + metrics for the racing stats (overtakes, contact rate, finishing order), adds the Phase 6 experiment config, and re-runs the SPS check with collisions on. Inputs: this spec §2/§3, design §12. Tools: read/edit, a short smoke run, W&B. Output: the grown warm start, the racing curriculum, the racing metrics, a Phase 6 config, a smoke run, and a fresh SPS number. Depends on the env, the reward, and the obs.

- **app-integration-engineer.** Scope: the live backend (`server/`, `sim/`) and the frontend (`web/`). Adds race position + the real gap-to-ahead to the per-car frame, lists the running order P1…PN in the timing tower, ensures contact is visible (the cars bounce), and keeps the recorder/replay format backward compatible. Optional: a debug overlay of neighbor links + collision discs. Inputs: this spec §2 presentation, the Phase 5 app. Tools: read/edit, manual browser check. Output: a race-aware live view and replay. Depends on the env and a checkpoint.

- **test-engineer, independent.** Scope: `tests/`. Writes the unit and integration tests in §c from this spec and the public interfaces and schemas, not from the implementation source. Forbidden from reading implementation internals to shape tests. Inputs: this spec, the interface signatures, the schemas. Tools: read the spec/signatures, read/write the tests folder, run the suite. Output: an unbiased suite mapped to the acceptance criteria — the obs-prefix-unchanged, local/relative neighbor, order-independent collision, no-tunnel, zero-sum overtake, refused-silent-resume, and grown-warm-start-preserves-driver contracts especially.

- **reviewer.** Scope: read-only review + the test run. Reviews each diff against this spec and the conventions: config-driven values (every new weight/geometry in config); SI units and the world frame; **the obs `[0:22]` prefix unchanged and `OBS_VERSION = 3`**; **no absolute position in the neighbor block** (local/relative only); **`PhysicsModel.step` untouched** (collisions only in the env pass); **collisions order-independent and reproducible**; **`reward_v3` reduces to `reward_v2`** with no contact/constant rank and never centerline-seeking or hand-coding racecraft; the silent resume of a v2 checkpoint refused and the grown warm start the explicit path; per-car lap timer / constant-width SuperSuit / done-agent removal preserved; the single-agent path and a one-car field unbroken; deterministic seeding for the draw, both reset modes, and the collision pass. Runs the full suite (incl. `parallel_api_test`), the linter, and the formatter; pass/fail with reasons; blocks on any violation or red test/lint.

### Notes

- Token saving comes from the narrow scopes and the separate contexts.
- Unbiased tests come from the test-engineer working from the spec contracts, separate from the agents that wrote the code.
- Dependency order: observation-engineer and collision-engineer first (independent, parallel), then multiagent-env-engineer (threads both into the field step) and reward-engineer (in parallel), then selfplay-training-engineer (warm start + curriculum + metrics), then app-integration-engineer, with test-engineer working in parallel from the spec and reviewer gating each merge.
- The physics role returns this phase as the **collision-engineer** (collision response is physics-adjacent), but `PhysicsModel.step` and the dynamic model are unchanged — the collision math lives in the env, not the physics core.

---

## 6. Companion plan

The file-by-file build order lives in `.claude/plan/phase-6-racing-plan.md` (to be written in plan mode), grounded in the real Phase 5 code. Its **chronological gate order**:

1. **Lock the contracts the rest of the phase depends on** — the ObservationV3 layout (`OBS_VERSION = 3`, the neighbor block at the tail, `[0:22]` unchanged) and the reordered field step (advance → resolve_collisions → rank → finalize) — and prove the obs-prefix-unchanged property with a test before training anything.
2. **The collision pass** (two-disc bodies, snapshot-then-apply, order-independent) and **the neighbor block** (local/relative, zero-padded) — the two new mechanisms, each behind a test, before they are wired into the field step.
3. **`reward_v3`** (contact + zero-sum overtake/defend, config-weighted, reducing to `reward_v2`), then **the grown input-layer warm start** (refuse the silent v2 resume; transplant the Phase 5 driver), then **the racing curriculum** and **the racing metrics/eval**, then **the race-aware app**, then the dispatch DAG.
4. **Re-run the throughput check with collisions on** before committing field sizes, and treat **reward-balance tuning as the budgeted core of the phase**, not a final polish step.

Where this phase fixes a contract the design leaves open (the neighbor-block layout and `OBS_VERSION = 3`, the `collision:` config block and the two-disc model, the field-level collision pass and the advance/finalize split, `reward_v3` and its weights, the grown warm start, the reward-weight curriculum, and the racing info/metrics), update `TECHNICAL_DESIGN.md` in the **same commit** — the decision and the doc move together (CLAUDE.md rule). The specific design edits: §7 (the neighbor block defined, `OBS_VERSION = 3`), §5/§10 (the env's collision pass, physics still pure), §9 (the contact penalty and the overtake/defend reward), §12 (the racing metrics and the grown warm start), and §15 (Phase 6 as-built).
