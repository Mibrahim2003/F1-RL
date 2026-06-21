# Phase 5 Spec: Many Cars on Track

Status: draft for plan mode. Branch: `phase-5-many-cars` (to be cut from `main` after Phase 4 merges). Depends on Phase 4 (`phase-4-many-circuits.md`) for the generalist policy, the circuit pool, and the curriculum mechanism; on Phase 3b (`phase-3b-realistic-physics.md`) for the dynamic physics, the grip pipeline, and ObservationV2; on Phase 3a for the training loop, the checkpoint format, and the live view; and on Phase 1/2 for the app and the cached circuits.

---

## 1. Introduction

### a. Overview, Problem Description, Summary

Put a full field on track. Through Phase 4 a single policy laps the whole calendar, but one car at a time — the env owns exactly one car and the live view streams exactly one car. Phase 5 turns that one car into a grid: a multi-agent environment where many homogeneous cars step at once, all driven by one shared policy, all lapping the same circuit together, then renders the whole field in the app.

This phase is the **infrastructure** phase for racing, not racing itself. Per the build order (TECHNICAL_DESIGN.md §15) the field goes on track **with no racing rules yet**: the cars do not see each other, there are no collisions, and there is no contact penalty or overtaking reward. Those are Phase 6. Phase 5 isolates the new risk — the multi-agent env, shared-policy self-play, and rendering many cars — from the racing-interaction risk that follows it.

Because the cars do not yet observe each other, **the observation is unchanged**: ObservationV2, length 22, `OBS_VERSION = 2`. Every car sees only the track, exactly as in Phase 4. The nearby-car block (relative position and velocity of the K nearest cars, reserved in design §7) is **not** added here — it lands in Phase 6 and bumps `OBS_VERSION` then. Because the observation is unchanged, the Phase 4 generalist checkpoint warm-starts directly as the shared brain.

Suggested solution, in one line: build a `RacingParallelEnv` (PettingZoo `ParallelEnv`) that holds N independent cars on one shared circuit and reuses the single-agent physics / observation / reward per car, train one shared policy across it with parameter sharing through SuperSuit + Stable-Baselines3 PPO, scale the field 2 → 4 → 22, warm-start the Phase 4 policy, and stream the whole field to the app.

Stakeholders: you, as the developer and primary user. Secondary: future portfolio reviewers, who will watch a full grid lap a real circuit together — the first frame that looks like a Grand Prix.

### b. Glossary or Terminology

- Multi-agent env: an environment that steps many agents at once and returns per-agent observations, rewards, and done flags. Here every agent is one car.
- PettingZoo ParallelEnv: the standard parallel multi-agent API; all agents submit actions and step simultaneously.
- SuperSuit: the adapter that vectorizes a PettingZoo ParallelEnv into a vector env Stable-Baselines3 can train against.
- Shared policy / parameter sharing: one set of network weights drives every car; each car's transitions are training data for the same policy.
- Self-play: training a policy against copies of itself. In Phase 5 the copies share a track but do not yet interact, so the genuine react-to-others dynamic is dormant until Phase 6 adds nearby-car observations and contact.
- Homogeneous agents: every car has the same observation shape, the same action shape, and the same dynamics — required for one shared policy.
- Field / grid: the set of cars on track in one episode (2, 4, …, up to 22).
- Grid slot: a distinct start position and heading assigned to each car at reset so the field does not spawn on top of itself.
- Generalist checkpoint: the Phase 4 policy that laps any circuit; the warm-start weights for the shared brain.
- Non-interacting field: many cars sharing a track with no mutual observation and no collision — the Phase 5 artifact, distinct from racing (Phase 6).

### c. Context or Background

- Why worth solving: a single car, however good, is not the dream. The vision (PROJECT_VISION.md) is twenty-two cars on a real circuit seen from above. Phase 5 is the step where the screen finally fills with a field. It is also the load-bearing infrastructure for every later phase: racing (Phase 6) and pit strategy (Phase 7) both run inside the multi-agent env this phase builds.
- Origin: Phase 5 of the visibility-first build order in TECHNICAL_DESIGN.md §15, and the multi-agent half of the env contract in §10.
- How it affects the goals: it serves fun (a grid on track is a far stronger demo than one car), resume (a working PettingZoo + shared-policy self-play pipeline is a clear headline), and learning (the multi-agent env, parameter sharing, and the SuperSuit + SB3 vectorization are the deep skill of this phase).
- Past efforts: Phase 3 produced the resumable checkpoint and the live view. Phase 4 produced the generalist policy and the circuit pool. Phase 5 reuses both — the generalist as the shared brain, the pool as the per-episode circuit draw — and adds the second car onward.
- Roadmap fit: Phase 6 adds nearby-car observations, collisions, the contact penalty, and overtaking/defending rewards on top of this field. Phase 7 adds pit stops. Both stand on the multi-agent env built here.
- Technical strategy fit: the observation is local and relative by design (§7) and the cars do not yet see each other, so Phase 5 changes the **number of cars and the training topology**, not the observation. The physics interface, the observation vector, the action space, and the checkpoint schema are unchanged; the single-agent `RacingEnv` stays as the one-car path.

### d. Goals or Product and Technical Requirements

Product requirements as user stories:

- As the user, I start a session and watch a full field of cars lap a real circuit together, so the race finally looks like a Grand Prix from above.
- As the user, I pick the field size (2, 4, or the full 22) and the circuit, and the same shared policy drives every car, so I can see the grid scale.
- As the user, I see the timing tower list every car with its lap time and gap, so I can read the field at a glance. (Gap is defined by **track position** — arc-length `s` along the centerline, rendered as distance or as time-behind-leader by progress — since the cars do not yet race; a true race gap arrives with Phase 6.)
- As the user, I can replay a recorded multi-car run, so a saved grid plays back identically.

Technical requirements, functional and required:

- A multi-agent env. `RacingParallelEnv(pettingzoo.ParallelEnv)` holds N homogeneous cars on one circuit, steps all cars at once, and returns per-agent `(obs, reward, terminated, truncated, info)` dicts keyed by agent id. It must pass PettingZoo's `parallel_api_test`, the multi-agent analogue of the Gymnasium env checker.
- Per-car reuse of the single-agent core. Each car reuses the existing physics step, the ObservationV2 builder, the progress reward, the conditions / grip pipeline, lap timing, and the termination logic. No per-car logic is duplicated; the single-car step is factored so both `RacingEnv` and `RacingParallelEnv` call it. **What is per-car versus per-circuit is load-bearing:** the per-step projection state (`_prev_s`, `_grip_idx`, `_grip_lat`, `_wrong_way_count`, the `CarState`) is **car state**, not circuit state, so the factored step must carry it per car, not read it off the env. See blocker note below on the lap timer.
- The lap timer is **per car, not pooled**. The Phase 4 `CircuitPool` gives each pool entry **one** `LapTimer` (`CircuitEntry.lap_timer`, `pool.py:129`), and `LapTimer` is stateful (`completed_laps`, `lap_start_t`, `_prev_s`). N cars on one circuit map to one pool entry, so reusing `entry.lap_timer` would have every car stomp a single shared lap state. The parallel env must construct **one `LapTimer` per car** (`LapTimer(track, pole)` is cheap) and reuse the pool entry only for its **read-only** `Track`, `EdgeCache`, and resolved pole. `entry.lap_timer` is **not** reused per car.
- Observation unchanged and locked. ObservationV2 stays length 22, `OBS_VERSION = 2`. Each car observes only the track (no nearby-car block this phase). A test asserts the per-agent observation is the same vector the single-agent env produces for the same car state on the same circuit.
- Per-car episodes inside a constant-width field. Each car has its own state, lap timer, progress, reward, and done flag; one car finishing or failing does not end the others. **The SuperSuit-visible agent width stays constant for the whole episode.** **As built:** the raw `RacingParallelEnv` follows standard PettingZoo — a finished/crashed car is removed from `self.agents` on the next step (so it passes `parallel_api_test`) — and the *constant* width comes from the `black_death_v3` **wrapper** in the training stack, which re-pads removed agents to zero obs/reward. (The earlier "not removed from `self.agents` mid-episode" wording meant the SuperSuit-visible width, which the wrapper guarantees, not the raw env.) The env episode ends when `self.agents` is empty (all cars done) or the step limit fires.
- Constant agent set for the vectorizer. SuperSuit's `pettingzoo_env_to_vec_env_v1` requires a **fixed agent set** across steps; agents dying at different steps breaks the conversion unless padded. The chosen and built path: per-car early death is allowed in the raw env, and `black_death_v3` zero-pads removed agents so the vectorizer sees a fixed width. The constant-width guarantee lives in the wrapper, not the raw env.
- Grid start, with two reset modes. Spawn layout depends on the purpose and the two conflict on one reset: **training reset = scattered starts** (a random centerline index per car, the existing `start_randomize`, for state-space coverage — the field smeared around the whole lap), and **eval/demo reset = a real grid** (cars bunched in distinct, non-overlapping slots and headings near the start/finish line). Both are config-selectable and seeded; do not promise a tidy grid on the training reset.
- One shared policy with parameter sharing. Training routes the ParallelEnv through SuperSuit (`pettingzoo_env_to_vec_env_v1` + `concat_vec_envs_v1`) into a Stable-Baselines3 vector env so a single PPO policy learns from every car's transitions. Self-play uses the current shared weights for every car. **Honest framing:** with no mutual observation and no collisions, the cars are non-interacting, so for *learning* this is identical to raising `n_envs` on the Phase 4 single-agent path — Phase 5 buys **infrastructure and the render, not a learning gain** (see §2b cons and §3b).
- Field size is a **per-run constant, stepped across warm-started runs — not a mid-run curriculum knob**. `n_agents` sets the vector-env **width**, which the in-place curriculum mechanism (`VecEnv.env_method`, sampling-side, no rebuild — `curriculum.py:102-111`) cannot change without a full vector-env rebuild that would break the PPO rollout buffer / `n_steps`. So the field is grown 2 → 4 → 22 by **launching successive runs**, each a fixed field size warm-started from the prior. The **circuit-pool widening stays an in-place curriculum knob** (Phase 4, unchanged); **field-size widening does not**.
- Warm start, then continue. The shared policy warm-starts from the Phase 4 generalist checkpoint (observation unchanged, checkpoint validates), then continues; retraining from scratch is the fallback.
- Circuit pool reuse. The grid samples a circuit per episode from the Phase 4 circuit pool; all cars in an episode share the drawn circuit — sharing the entry's **read-only** `Track`, `EdgeCache`, and pole only (per-car lap timers, per above).
- Dependency version matrix resolved **first**. PettingZoo + SuperSuit + the pinned Stable-Baselines3 / Gymnasium / PyTorch must be a working, installed matrix on the venv (py3.12, `<3.13`) **before any env code is written**. SuperSuit is lightly maintained and lags the Gymnasium API; a broken matrix sinks the phase, so it is the first gating task in the plan, not an "during training" detail.
- Throughput estimate **before committing**. SuperSuit steps all N cars **sequentially inside one Python process** (`ParallelEnv.step` loops the agents); the dynamic model is the slow path, so a 22-car process costs ~22× the per-process step cost and that is **not** amortized by `SubprocVecEnv` fan-out. Get a steps-per-second estimate (current `n_envs × 1-car` versus a few SuperSuit copies × N cars on the same cores) before building, so the field-size ceiling and the laptop-vs-cloud line are known up front.
- Deterministic seeding, with the seed recorded in every run and checkpoint; the per-episode circuit draw and the per-car start randomization are reproducible from the seed.
- Multi-car live view and replay. The backend streams every car's state each control step, the frontend renders the full field colored by team and lists every car in the timing tower, and the recorder/replay format carries many cars.
- Runs on the laptop CPU with no GPU, headless and device-agnostic, reusing the Phase 3/4 training setup.

### e. Non-Goals or Out of Scope

- **No nearby-car observations.** The K-nearest-cars block (design §7) is Phase 6; the observation stays ObservationV2 (length 22, `OBS_VERSION = 2`) here. Cars do not see each other.
- **No collisions and no contact penalty.** Cars may overlap on screen with no physical interaction; collision detection and the contact penalty are Phase 6.
- **No overtaking or defending rewards.** The reward stays the per-car progress core; racing-interaction rewards are Phase 6.
- No observation or physics change otherwise. The dynamic model, the grip pipeline, and ObservationV2 are reused untouched.
- No new track data work. The circuits and surfaces are Phase 2's; this phase samples the pool already built.
- No pit stops. Phase 7.
- No JAX or GPU-accelerated environment. A later optimization (§17); the N-cars × n-envs throughput cost is the trigger to revisit it, not this phase.
- No mobile.

### f. Future Goals

- Phase 6, racing for real: nearby-car observations (bumping `OBS_VERSION`), collision detection, the contact penalty, and overtaking/defending rewards, built inside this multi-agent env.
- Phase 7, pit stops and strategy, on top of the racing field.
- Heterogeneous fields later (per-car compound or strategy differences) once racing works, if worth it.
- Curriculum on field density (tighter grids, closer spawns) once cars interact.

### g. Assumptions

- Phase 4 is complete and merged: the generalist checkpoint, the circuit pool, the curriculum mechanism, ObservationV2, the dynamic physics, the checkpoint format, and the live view all work.
- PettingZoo and SuperSuit are installable in the project venv and version-compatible with the pinned Stable-Baselines3 / Gymnasium / PyTorch (design §2 names them as the chosen stack). **As built (Phase 5):** they were already listed in `pyproject.toml` but unpinned; the dependency-matrix step pinned the verified-working matrix `pettingzoo==1.26.1` / `supersuit==3.11.0` / `stable-baselines3==2.9.0` (with `gymnasium 1.3.0`, `torch 2.12.0`).
- The built circuits under `data/tracks/` load through the runtime-safe loader (no FastF1, no network).
- The shared-policy warm start is legal because the observation and action shapes are unchanged from Phase 4.
- The laptop has several CPU cores and enough RAM to hold the field; the full 22-car grid is reached on the cloud if the laptop cannot.

---

## 2. Solutions

### a. Current or Existing Solution

Phase 4 runs one car. `RacingEnv` (`env/single_agent.py`) owns a single `CarState`, a single lap timer, and a single progress reward, samples a circuit per reset from the `CircuitPool`, and steps one car per `step`. The live path (`sim/loop.py`, `server/app.py`) drives one car and the WebSocket frame carries a single `car` object; the frontend renders one car and one timing row.

- Pros: a working generalist policy on realistic physics with a per-circuit lap-time table, a clean per-circuit binding, and a shared observation builder reused by the server.
- Cons: exactly one car. There is no field, no multi-agent env, and no shared-policy self-play, so the central image of the vision — a full grid — cannot be shown.

### b. Suggested or Proposed Solution

- Factor the single-car step into a reusable unit. Extract the per-car update (map action → grip → substep physics → project → reward → termination → per-car info) so `RacingEnv` and the new `RacingParallelEnv` share one implementation. The single-agent env keeps its contract unchanged.
- Build `RacingParallelEnv` (PettingZoo `ParallelEnv`). It holds N cars on one circuit drawn from the Phase 4 pool at reset, assigns each car a grid slot, steps every car with the shared per-car unit, and returns per-agent dicts. Per-agent observation and action spaces are the unchanged ObservationV2 Box and the unchanged 2-D action Box. It passes `parallel_api_test`.
- Keep the cars independent. Each car has its own state, lap timer, progress, reward, and done flag. No car observes another and no collision is computed (Phase 6). One car's failure removes that agent from the field for the rest of the episode without ending the others.
- Train one shared policy. Route the ParallelEnv through SuperSuit (`pettingzoo_env_to_vec_env_v1` + `concat_vec_envs_v1`) into a Stable-Baselines3 vector env wrapped in `VecNormalize`, and train PPO with parameter sharing so every car feeds the same policy. Warm-start from the Phase 4 generalist checkpoint.
- Scale the field with the curriculum. Reuse the Phase 3b/4 stage-table mechanism to grow the field size (and reuse the circuit-pool widening) over timesteps, so training starts at 2 cars and widens to the full 22.
- Render the field. Stream every car's state each control step as a `cars` array, render the full grid colored by team, list every car in the timing tower, and extend the recorder/replay format to many cars.

External components the solution interacts with or alters: the Python env and training scripts, the new PettingZoo + SuperSuit dependencies, the live backend and frontend, the recorder/replay format, the Weights and Biases service, and the local filesystem. New third-party dependencies: PettingZoo and SuperSuit (both named in design §2).

Pros of the proposed solution: a real field on track, one shared policy warm-started from the generalist, no observation or physics change, and the exact env Phase 6 racing is built into.

Cons of the proposed solution: the new multi-agent stack (PettingZoo + SuperSuit + SB3) is the main integration risk and its **version matrix must be resolved first** (SuperSuit lags Gymnasium); throughput is worse than it looks because SuperSuit steps all N cars **sequentially in one process** on the slow dynamic model (~N× per-process cost, not amortized by `SubprocVecEnv`); and, with no mutual observation and no contact, the cars are independent, so this phase delivers **zero learning gain** over raising `n_envs` on the Phase 4 path — the payoff is the multi-agent infrastructure and the field render, and genuine self-play stays dormant until Phase 6.

#### Data Model and Schema Changes

```
ObservationV2:            # UNCHANGED, length 22, OBS_VERSION = 2
  ... as Phase 4 ...      # local/relative track features only, no nearby-car block yet

Action:                   # UNCHANGED, Box(-1, 1, shape=(2,))  steering, longitudinal

RacingParallelEnv (new):
  agents: [car_0, car_1, ...]      # N homogeneous agents, CONSTANT width for the episode
  observation_space(agent) -> ObservationV2 Box   # same for every agent
  action_space(agent)      -> 2-D action Box       # same for every agent
  reset(seed) -> (obs: {agent: vec}, info: {agent: {...}})
  step(actions: {agent: a}) -> (obs, rewards, terminations, truncations, infos)  # per-agent dicts
  # per-car (NOT pooled): CarState, LapTimer instance, _prev_s/_grip_idx/_grip_lat/_wrong_way_count
  # per-circuit (pool entry, read-only, shared by the field): Track, EdgeCache, pole

Field config (new block, e.g. `grid:`):
  n_agents: 2                # field size — a PER-RUN CONSTANT (not a curriculum knob; see below)
  reset_mode: scattered      # scattered (train: random idx per car) | grid (eval/demo: slots @ S/F)
  grid_spacing_m: <float>    # spacing between grid slots (grid mode)
  team_colors: [...]         # render only; not seen by the policy

Curriculum stage (extended, Phase 4 reuse — IN-PLACE only):
  start_step
  circuits: [id, ...]        # Phase 4 pool widening, reused (sampling-side, in-place)
  ... existing condition overrides (mu_base, wear_rate, weather) ...
  # NOTE: n_agents is NOT a stage field. Field size changes the vec-env width, which the
  # in-place env_method mechanism cannot do without a rebuild -> stepped across runs instead.

Recorder (multi-car):
  one recorder per run, frames keyed `cars: [ {id, x, y, yaw, speed, telemetry}, ... ]`
  (matches the live `cars[]` frame; replaces the single-car `car` object)

Checkpoint meta:
  ... as Phase 4 ...
  obs_version               # still 2, so the Phase 4 generalist warm-starts cleanly
  circuit_id                # pool descriptor (e.g. "calendar"), as Phase 4
  n_agents                  # records the (constant) field size the run trained on
```

Modified data: a new multi-agent env alongside the single-agent one, a field config block (size + reset mode), a multi-car recorder/frame, and an `n_agents` field in the checkpoint meta. The curriculum stage is **unchanged** (no `n_agents` field — field size is stepped across runs, not per stage). Unchanged: the observation vector and version, the action shape, the physics interface, the circuit pool entries (Track/EdgeCache/pole are read-only and shared; lap timers are per car), the per-circuit pole resolution, and the rest of the checkpoint schema. Validation: the checkpoint loader still refuses a mismatched `obs_version` or action shape; the parallel env refuses a non-positive `n_agents` and a circuit with no built `.npz`.

#### Business Logic

- Env construction: build the circuit pool once (as Phase 4, read-only Track/EdgeCache/pole per entry); build the shared per-car physics and obs/reward params once; hold N car slots, each with **its own `LapTimer` instance** (not the pool entry's).
- Reset: draw one circuit from the pool for the whole field; resolve weather; place the cars by `reset_mode` (`scattered` = a seeded random centerline index per car for coverage; `grid` = seeded distinct, non-overlapping slots near the start/finish for eval/demo); reset **each car's own lap timer** and per-car projection state; return per-agent observations and info (each carrying `circuit_id` and `pole_time_s`).
- Step: for **every agent in the constant-width set**, run the shared per-car unit (map action → grip → substeps → project → progress reward → per-car termination); a car whose episode ended is **frozen and zero-padded** (`black_death_v3` / equivalent), **not** removed from `self.agents`, so the vectorizer keeps a fixed width; assemble per-agent obs/reward/terminated/truncated/info dicts; end the env episode when all cars are done or the step limit fires.
- Self-play training: the ParallelEnv is vectorized by SuperSuit (`pettingzoo_env_to_vec_env_v1` + `concat_vec_envs_v1`) so every car's transition trains the one shared policy; the policy warm-starts from the Phase 4 checkpoint. No mutual observation, no contact → for learning this equals more `n_envs` on the Phase 4 path (no learning gain this phase).
- Field scaling: **across runs, not mid-run.** Each run is launched with a fixed `n_agents` and warm-started from the prior smaller-field run (2 → 4 → … → 22). Field size cannot be an in-place curriculum knob because it changes the vector-env width (a rebuild that breaks the PPO rollout buffer). The circuit-pool widening **does** stay an in-place curriculum knob (Phase 4, unchanged).
- Error states: a non-positive `n_agents` is refused; an unbuilt circuit id is refused with the build hint; a checkpoint with a mismatched observation version is refused as before.
- Failure scenarios: the laptop stops (frequent checkpoints, clean resume), a Weights and Biases outage (local logging), and a shared policy that drives worse in a field than alone (curriculum field-size widening, more steps, or revert to the Phase 4 checkpoint).
- Limitations: no interaction yet, the dynamic model is slower per step, and N cars × n_envs copies cost RAM and throughput.

#### Presentation Layer

- User requirements: pick a field size and circuit, watch the whole grid lap together, read every car in the timing tower, and replay a multi-car run.
- UI changes: the live frame carries a `cars` array instead of a single `car`; the renderer draws every car as an oriented shape colored by team; the timing tower lists every car with lap time and gap; the replay player loads and scrubs a multi-car trajectory.
- Web concerns: the same WebSocket streams more state per frame (N cars); the canvas already interpolates between frames and now interpolates per car. No new rendering primitive, only an array.
- UI states: a field selected and lapping live, and a recorded multi-car run replaying.
- Error handling: a clear message when a circuit or checkpoint cannot load; the viewport never crashes on a bad or partial frame.

#### Other questions to answer

- How will it scale: SuperSuit + SB3 vector envs use the laptop cores; the full 22-car grid moves to the cloud if the laptop runs out of RAM or throughput. The checkpoint stays resumable and portable into Phase 6.
- Limitations: as above; the JAX env (§17) is the real lever if N-cars throughput becomes the bottleneck, and Phase 5 is where that pressure first appears.
- Recovery on failure: resume from the last checkpoint, saved often.
- Future requirements: this env is the substrate for Phase 6 racing — the nearby-car observation block, collisions, and the contact penalty plug into the per-car step and the per-agent obs built here.

### c. Test Plan

Tests are written by an independent test subagent from this spec and the public interfaces, not from the implementation, so they verify the requirements and stay unbiased. See section 5.

- Unit tests: the parallel env passes `parallel_api_test`; per-agent observation and action spaces are the unchanged ObservationV2 Box and 2-D action Box; the per-agent observation equals the single-agent observation for the same car state on the same circuit (observation unchanged and track-only — no nearby-car data); **the lap timer is per car** — one car completing a lap does **not** advance another car's `completed_laps`/`progress` (the shared-timer trap); the **agent set width is constant across steps** even when a car terminates early (the dead car is padded, not dropped); `reset_mode=grid` places N distinct, non-overlapping slots and `reset_mode=scattered` places distinct seeded indices, both reproducible from a fixed seed; each car has an independent progress / done flag (one car's failure does not end the others); the per-episode circuit draw is shared by the whole field and reproducible from the seed; a non-positive `n_agents` and an unbuilt circuit id are refused.
- Integration tests: SuperSuit wraps the ParallelEnv into an SB3-trainable vector env and a short smoke run **completes without error and the shared-policy return is non-degenerate** with a field (note: a rising curve here only proves the env is trainable, **not** a multi-agent learning gain — it is expected to match the equivalent single-agent `n_envs` run); the Phase 4 generalist checkpoint warm-starts on the multi-agent run without an obs-version error; a checkpoint resumes and continues; the app drives a multi-car field live for one checkpoint; a recorded multi-car run replays.
- Throughput check: measure steps per second for a SuperSuit field of N cars versus an equal-core single-agent `n_envs` run on the same machine, to size the field-size ceiling and the laptop-vs-cloud line before scaling up.
- QA: confirm a field of 2, then 4, then more cars laps a circuit cleanly together; confirm every car appears in the timing tower with believable laps; confirm the `grid` reset does not spawn cars on top of each other.

### d. Monitoring and Alerting Plan

- Logging: Weights and Biases primary, local logs fallback.
- Metrics: mean and per-car lap time, mean and best delta to the pole across the field, the (constant) field size of the run, episode return per car and aggregated, off-track count across the field, steps per second (with the field-size penalty), and the learning curves.
- Observability: the learning curves, periodic **multi-car eval clips** of the field on a rotating circuit, and the on-screen debug overlay. **Owner note:** the existing `train/evaluate.py` drives a Gymnasium env, which a `ParallelEnv` is not, so the multi-car offscreen mp4 needs a **dedicated multi-car eval driver** — assigned to the selfplay-training-engineer scope (below). If that driver is descoped, drop the field eval clip from monitoring rather than leaving it unowned.
- Alerting: none external. Watch the curves and the per-car spread. A collapsing return when the field grows, or cars that drive well alone but poorly in a grid, is the signal to **step the field up more slowly across runs** or revert the warm start.

### e. Release, Roll-out, and Deployment Plan

- Branch `phase-5-many-cars`. Merge when one shared policy laps a circuit with a field (at least 2 and 4 cars demonstrated, the full 22 reached where compute allows), the field renders and times correctly in the app, the Phase 4 generalist warm-starts cleanly, and the suite and linter are green. The PR description carries the run summary, the curves, and a clip of the field lapping a circuit.

### f. Rollback Plan

- Liabilities: a new multi-agent stack could break training throughput or correctness, and a frontend change could break the existing one-car live view.
- Reduce liabilities: keep main working, develop on the branch, tag the working commit, checkpoint often, and keep the last good Phase 4 checkpoint.
- Prevent spread: the single-agent `RacingEnv`, the observation, the physics interface, the action space, and the checkpoint schema are unchanged, so the one-car path keeps working independently of the new multi-agent path. The live frame is the main shared surface — keep it backward compatible (a single car is a one-element `cars` array) or version the frame. Revert the merge or restore the tagged commit and the last good checkpoint if needed. Delete the branch after a clean merge and a passing test.

### g. Alternate Solutions or Designs

- PettingZoo + SuperSuit + SB3 versus a hand-rolled multi-agent loop. The PettingZoo path is chosen because it is the named stack (design §2) and SuperSuit gives the SB3 vectorization for free; a hand-rolled loop is rejected as reinventing the adapter.
- Shared policy with parameter sharing versus independent policies per car. Parameter sharing is chosen — it is the tractable path to 22 cars and the design's decision (§10); independent policies are rejected as 22× the training cost with no payoff for a homogeneous field.
- One ParallelEnv of N cars versus N single-agent envs glued together. The ParallelEnv is chosen because Phase 6 needs a true multi-agent env (cars must see and hit each other); gluing single-agent envs is rejected as a dead end that cannot carry interaction.
- Adding the field but no interaction now versus jumping straight to racing. Splitting the field (Phase 5) from racing (Phase 6) is the design's decision (§15) — it isolates the multi-agent-env risk from the racing-interaction risk so each is debuggable on its own.
- Field-size curriculum versus the full 22 from step zero. Widening is chosen to ease learning and keep early throughput high; training straight on 22 is the fallback if the curriculum adds little.
- Migration: a one-car field reproduces the single-agent behavior closely (a one-element field on the same circuit and reward), so the change is backward compatible at the env contract and the live frame.

---

## 3. Success Evaluation

### a. Impact

- Security: local training and a localhost WebSocket, no exposure, and validated checkpoints.
- Performance: the dynamic model is unchanged per car-step, but SuperSuit steps the field **sequentially within one process**, so a process's step cost scales ~linearly with N and is **not** amortized by `SubprocVecEnv` fan-out; throughput therefore drops sharply with field size, and RAM rises with the field. This is **the wall of the phase** — get the SPS estimate before committing (§2 test plan). The JAX env (§17) is the real lever, and Phase 5 is where that pressure first appears.
- Cost: zero on the laptop, free tiers on the cloud if used; the full 22-car field is the most likely reason to use the cloud GPU/CPU.
- Impact on other components: delivers the multi-agent env and the shared-policy self-play pipeline that Phase 6 racing and Phase 7 pit strategy are built into.

### b. Metrics

- Capture: mean and per-car lap time and delta to the pole with a field, field size reached, episode return per car and aggregated, off-track count across the field, and steps per second versus field size.
- Tools: Weights and Biases, the eval clips, and the on-screen overlay.
- Definition of done: one shared policy laps a circuit with a field, scaled from 2 to 4 cars **across warm-started runs** (and to the full 22 where compute allows), the field renders and times correctly in the app, the Phase 4 generalist warm-starts cleanly, the per-car lap times stay believable as the field grows, and the per-car lap timer / constant-width-field / observation-unchanged contracts hold under test. The bar is **infrastructure correctness and the render**, not a learning gain (none is expected this phase).

---

## 4. Deliberation

### a. Discussion

- How large a field the laptop can train before throughput or RAM forces the cloud, and how big a field-size step (2 → 4 → …) each warm-started run should take.
- Whether the shared policy degrades in a field even without interaction (e.g. start-state crowding) and what cadence of field-size steps keeps it stable.
- How much of the Phase 6 racing surface to leave hooks for now (the per-car step and the per-agent obs/info are the seams) without building any of it.

### b. Decided before coding (were open; resolved by the review)

- **Field size is a per-run constant, stepped across warm-started runs** — not an in-place curriculum knob (it changes the vector-env width). Locked in §1d / §2 business logic.
- **Constant-width field with `black_death_v3`** (per-car early death allowed, dead cars zero-padded), because SuperSuit's vectorizer needs a fixed agent set. No mid-episode agent removal.
- **Per-car `LapTimer` instances**; the pool entry supplies only read-only Track/EdgeCache/pole.
- **The dependency version matrix (PettingZoo / SuperSuit / SB3 / Gymnasium / PyTorch on py3.12) is the first plan task** — resolved and installed before any env code, not "during training".

### c. Open Questions

- The exact field-size steps and the spawn spacing, set during training.
- The grid-slot layout for `reset_mode=grid` (a real starting grid along the start straight versus spaced centerline indices) and whether the `scattered` train reset jitters each car independently (it should, for coverage).
- The measured memory and throughput ceiling for the full 22-car field on the laptop versus the cloud (sized by the §2 throughput check).
- The live-frame schema choice: keep one frame type with a `cars` array (single car = one-element array) versus a versioned frame, keeping the one-car path working.

---

## 5. Implementation Subagents

These become definitions under `.claude/agents/` (replacing or extending the Phase 4 roster for this branch). Each has a narrow scope and its own context, which saves tokens, since the main thread never carries the whole job. The test author is independent, so tests verify the spec and stay unbiased.

How they work together. The plan you build dispatches tasks in dependency order. Feature subagents read this spec and the technical design. The test subagent reads only this spec and the public interfaces, not the implementation. The reviewer gates each merge.

`/caveman` is **opt-in** for these subagents, not a forced pre-call. The caveman rules already exempt code, commits, and PRs (the bulk of subagent output), so a mandatory `/caveman` before every task is wasted ceremony and a wasted tool call. Set it in an agent file only where terse status chatter actually helps; default off.

### Roster

- dependency-matrix (first, gating). Scope: `pyproject.toml` and the venv. Resolves and installs a working PettingZoo + SuperSuit + Stable-Baselines3 + Gymnasium + PyTorch matrix on py3.12 (`<3.13`), confirms `parallel_api_test` and a `pettingzoo_env_to_vec_env_v1` round-trip import and run on a trivial ParallelEnv, and pins the versions. **No feature code starts until this passes** (SuperSuit lags Gymnasium; a broken matrix sinks the phase). May be the main thread rather than a subagent. Output: pinned, installed, smoke-imported deps.

- multiagent-env-engineer. Scope: `src/f1rl/env/` — the new `multi_agent.py` and the per-car step factoring in `single_agent.py`, plus the env construction seam in `factory.py`. The main role in Phase 5. Factors the single-car step into a reusable unit **carrying per-car state** (`CarState`, a **per-car `LapTimer` instance**, `_prev_s`/`_grip_idx`/`_grip_lat`/`_wrong_way_count`); builds `RacingParallelEnv(pettingzoo.ParallelEnv)` holding N cars on one pooled circuit (reusing the entry's **read-only** Track/EdgeCache/pole, **not** its lap timer) with **constant-width agents** (`black_death_v3`-compatible: early death freezes/zero-pads, never removes mid-episode) and **two reset modes** (`scattered` train / `grid` eval); returns per-agent dicts; reuses the unchanged observation / physics / reward per car. No observation change, no nearby-car block, no collisions, no field-size curriculum hook (field size is a constructor constant). Inputs: this spec and technical design §7, §10. Tools: read and edit files, run PettingZoo `parallel_api_test`. Output: a parallel env that passes the API test, keeps per-car lap state independent, and reproduces the single-agent per-car behavior on a one-car field.

- selfplay-training-engineer. Scope: `src/f1rl/train/` (new `selfplay.py`, a multi-car eval driver) + `configs/experiment/`. Wires PettingZoo → SuperSuit (`pettingzoo_env_to_vec_env_v1` + `concat_vec_envs_v1`, `black_death_v3`) → Stable-Baselines3 for one shared policy with parameter sharing, warm-starts the Phase 4 generalist checkpoint, sets the field size as a **per-run constant** with a **cross-run step-up** convention (each run warm-starts the prior smaller field; **no in-place field-size curriculum**), keeps the Phase 4 circuit-pool widening in-place, **owns the multi-car offscreen eval clip** (a `ParallelEnv` is not a Gym env, so `evaluate.py` cannot drive it), and runs a smoke run plus the throughput check (SPS of N-car field vs equal-core single-agent `n_envs`). Inputs: this spec and technical design §10, §12, §15. Tools: read and edit files, run a short smoke training run, log to Weights and Biases. Output: the self-play training script, the multi-car eval driver, a grid experiment config, a smoke run, and an SPS number. Depends on the env and the dependency matrix.

- app-integration-engineer. Scope: the live backend (`server/`, `sim/`) and the frontend (`web/`). Streams every car each control step as a `cars` array (single car = one-element array, keeping the one-car path working), renders the full field colored by team, lists every car in the timing tower with a track-position **gap**, and extends the recorder to **one multi-car recorder** (per-car entries under a `cars` key) and the replay player to many cars. Inputs: this spec and the Phase 1/3/4 app. Tools: read and edit files in the backend and frontend, manual check in the browser. Output: a multi-car live view and replay. Depends on the env and a checkpoint.

- test-engineer, independent. Scope: `tests/`. Writes the unit and integration tests in section c from this spec and the public interfaces and schemas, not from the implementation source. Forbidden from reading implementation internals to shape tests, so the tests check the contracts. Inputs: this spec, the interface signatures, and the data schemas. Tools: read the spec and the signatures, read and write the tests folder, run the suite. Output: an unbiased test suite mapped to the acceptance criteria.

- reviewer. Scope: read-only review and the test run. Reviews each diff against this spec and the conventions: config-driven values, SI units, **no observation change and `OBS_VERSION` stays 2 (no nearby-car block, no collisions this phase)**, no physics-interface change, homogeneous agents (one shared observation/action space), per-car core reused not duplicated, **a per-car `LapTimer` (never the pooled entry's)**, **constant-width agent set (no mid-episode removal)**, **field size as a per-run constant (no in-place field-size curriculum)**, the single-agent path unbroken, deterministic seeding for the circuit draw and both reset modes, and the runtime-safe loader (no FastF1 in the training loop). Runs the full suite (including `parallel_api_test`), the linter, and the formatter, and reports pass/fail with reasons. Writes no feature code. Inputs: the diffs, this spec, and the conventions. Tools: read files, run tests, run the linter.

Note on the physics role: Phase 5 changes no physics. The dynamic model and the grip pipeline from Phase 3b are reused unchanged behind the `PhysicsModel` interface, once per car, so there is no physics-engineer task this phase.

### Notes

- Token saving comes from the narrow scopes and the separate contexts.
- Unbiased tests come from the test-engineer working from the spec contracts, separate from the agents that wrote the code.
- Dependency order: multiagent-env-engineer first, then selfplay-training-engineer, then app-integration-engineer, with test-engineer working in parallel from the spec and reviewer gating each merge.

---

## 6. Companion plan

The file-by-file build order lives in `.claude/plan/phase-5-many-cars-plan.md` (to be written in plan mode), grounded in the real Phase 4 code. Its **chronological gate order** is fixed by the review above:

1. **Resolve and pin the dependency version matrix** (PettingZoo / SuperSuit / SB3 / Gymnasium / PyTorch on py3.12) and smoke-import `parallel_api_test` + the SuperSuit vec round-trip — **before any env code**.
2. **Throughput estimate** (SPS of an N-car SuperSuit field vs equal-core single-agent `n_envs`) — before committing to field sizes.
3. Then the per-car step factoring (carrying per-car lap timer + projection state), the `RacingParallelEnv` contract (constant-width agents via `black_death_v3`, two reset modes), the SuperSuit + SB3 wiring, the cross-run field-size step-up, the multi-car live frame + one multi-car recorder, and the dispatch DAG.

Where this phase fixes a contract the design leaves open (the `grid:` config block with `n_agents`/`reset_mode`, the per-agent dict contract, the constant-width + `black_death_v3` choice, the per-run field-size step-up, the multi-car live-frame and recorder schema, the `n_agents` checkpoint field), update `TECHNICAL_DESIGN.md` in the **same commit** — the decision and the doc move together (CLAUDE.md rule). Note §13's planned entries (`env/multi_agent.py`, `train/selfplay.py`) and §10's "shared rules" gain the new dependencies and the per-car-timer / constant-width contracts.
