# Phase 4 Spec: One Car on Many Circuits

Status: draft for plan mode. Branch: `phase-4-many-circuits`. Depends on Phase 3a (`phase-3a-training-core.md`) for the training loop, the checkpoint format, and the live view; on Phase 3b (`phase-3b-realistic-physics.md`) for the dynamic physics, the grip pipeline, ObservationV2, and the lap-time-vs-pole benchmark; and on Phase 2 for the cached real circuits under `data/tracks/`.

---

## 1. Introduction

### a. Overview, Problem Description, Summary

Make one policy drive the whole calendar. Through Phase 3b a single agent learns one circuit at a time, bound to one track for the life of the env. Phase 4 keeps that agent and that physics, confirms the observation is fully track-agnostic, and then samples a different circuit each episode so a single policy generalizes across every built circuit instead of overfitting one.

This part isolates generalization, the new risk, from the physics and the observation, which do not change. The observation vector stays ObservationV2 (length 22, `OBS_VERSION = 2`), so the policy can warm-start from the Phase 3b checkpoint rather than retrain from scratch.

Suggested solution, in one line: turn the env's single bound track into a sampled pool, draw a circuit per reset, rebind the per-track precompute, widen the pool from easy to the full calendar with the existing curriculum mechanism, continue training the Phase 3b policy, and score a lap-time table against the pole for every circuit.

Stakeholders: you, as the developer and primary user. Secondary: future portfolio reviewers, who will watch one policy lap circuit after circuit.

### b. Glossary or Terminology

- Track-agnostic observation: an observation built only from local, relative features, never absolute position, so it looks the same on every circuit.
- Circuit pool: the set of built circuits an env may sample from on reset.
- Per-episode sampling: drawing a circuit from the pool at each `reset`, so consecutive episodes run different tracks.
- Generalization: one policy driving circuits it trains on, without per-circuit weights.
- Pole lap: the real fastest lap for a circuit, the per-circuit benchmark; it lives in that circuit's config, not in its cached geometry.
- Lap-time table: the artifact, a per-circuit row of achieved lap, pole, and delta across the whole pool.
- Curriculum: training on an easier setting first, then harder; here the difficulty axis is the size and hardness of the circuit pool.
- Warm start: continuing the Phase 3b policy instead of retraining from random weights, possible because the observation is unchanged.
- Edge cache: the per-track precomputed asphalt-edge polylines the rangefinder beams cast against; one per circuit in the pool.
- Subagent: a focused Claude Code agent with its own context, role, and tools.

### c. Context or Background

- Why worth solving: a driver that only knows one circuit is a memorizer, not a driver. One policy that laps the whole calendar is the proof that the local-feature observation generalizes, and it is the most watchable result yet, the same car on Monaco, Monza, and Silverstone back to back.
- Origin: Phase 4 of the visibility-first build order in TECHNICAL_DESIGN.md section 15.
- How it affects the goals: it serves fun, one car touring every circuit is a strong demo, resume, generalization across the calendar is a clear headline, and learning, the generalization work and the curriculum design are the deep skill here.
- Past efforts: Phase 3a built the resumable checkpoint and a circuit switch on purpose, so Phase 4 continues the same policy on more circuits. Phase 3b proved that policy on realistic physics and added the pole benchmark on one circuit.
- Roadmap fit: Phase 5 adds many cars with shared-policy self-play, Phase 6 adds racing rules, Phase 7 adds pit stops. All of them stand on a policy that already drives any circuit.
- Technical strategy fit: the observation is local and relative by design (section 7, "this is what lets one policy generalize across the whole calendar"), so generalization is a sampling and training change, not an observation change. The env contract, the physics interface, and the checkpoint format are unchanged.

### d. Goals or Product and Technical Requirements

Product requirements as user stories:

- As the user, I load one checkpoint and watch the same policy lap any circuit I pick, so generalization is visible.
- As the user, I see a lap-time table with a row per circuit, achieved lap, pole, and delta, so I can measure how the one policy does across the calendar.
- As the user, I see the live view name the circuit currently loaded and its pole, so I know which track the one policy is driving.

Technical requirements, functional and required:

- Confirm and lock the track-agnostic observation. ObservationV2 stays length 22 and `OBS_VERSION = 2`; a test asserts the observation carries no absolute position and that the same car state on two different circuits yields same-shape, in-bounds vectors. The observation does not change in this phase.
- A circuit pool in config and per-episode sampling. The env holds a pool of circuit ids, draws one on each `reset` from the env RNG, and rebinds the track, the edge cache, the lap timer, and the pole for the drawn circuit. No absolute position leaks across the swap.
- Per-circuit precompute, built once and reused. The edge cache (and any per-track precompute) is computed once per pool circuit and cached, never rebuilt every reset, so sampling stays cheap.
- Per-circuit pole resolution. Each circuit's pole comes from its `configs/track/<id>.yaml` (the geometry `.npz` does not store the pole), so the benchmark compares against the right pole per circuit.
- A curriculum over the pool. Reuse the Phase 3b curriculum mechanism (a config stage table applied to the workers by timestep threshold) to widen the pool from a few easy circuits to the full calendar, so the agent generalizes gradually.
- Warm start, then continue. Continue the Phase 3b policy across the pool, since the observation is unchanged and the checkpoint validates; retraining from scratch is the fallback, not the default.
- The lap-time table benchmark. A deterministic evaluation runs one clean lap on every pool circuit, reads each pole, and emits a table of achieved lap, pole, delta, and whether twice the pole is reached, logged and saved.
- Deterministic seeding, with the seed recorded in every run and checkpoint; the per-episode circuit draw is reproducible from the seed.
- Runs on the laptop CPU with no GPU, headless and device-agnostic, reusing the Phase 3 training setup and the vectorized-env stack.

### e. Non-Goals or Out of Scope

- No observation change. ObservationV2 stays as built; if it ever changes, that is a deliberate `OBS_VERSION` bump and a retrain, not part of this phase.
- No physics change. The dynamic model and the grip pipeline from Phase 3b are reused untouched behind the `PhysicsModel` interface.
- No new track data work. Phase 2 owns the circuits and the surfaces; this phase only samples the circuits already built under `data/tracks/`.
- No multiple cars and no racing. Phase 5 and Phase 6.
- No pit stops. Phase 7.
- No JAX or GPU-accelerated environment. A later optimization.
- No mobile.

### f. Future Goals

- Phase 5, many cars on track, the multi-agent env with shared-policy self-play, reusing this generalist policy as the shared brain.
- Per-circuit fine-tuning from the generalist checkpoint, only if a circuit stays stubbornly slow.
- A held-out circuit split to measure transfer to circuits never trained on, if the calendar grows.

### g. Assumptions

- Phase 3a and Phase 3b are complete and merged, the training loop, the dynamic physics, ObservationV2, the checkpoint format, the curriculum mechanism, the lap-time-vs-pole metrics, and the live view all work.
- The built circuits under `data/tracks/` load through the runtime-safe loader, which never imports FastF1 and never touches the network.
- Each pool circuit has a config under `configs/track/<id>.yaml` carrying its pole; circuits with a config but no built `.npz` are excluded from the pool until built.
- The laptop has several CPU cores for parallel envs, and a Weights and Biases account logs from the laptop.

---

## 2. Solutions

### a. Current or Existing Solution

The Phase 3b agent drives realistic physics on one circuit. The env binds a single track at construction (`single_agent.py`), so `reset` always replays the same circuit, the edge cache, lap timer, and pole are all bound to that one track, and the benchmark scores one circuit.

- Pros: a working, learning agent on realistic physics with a pole benchmark on one circuit.
- Cons: the policy can overfit one circuit, and there is no measure of generalization across the calendar.

### b. Suggested or Proposed Solution

- Confirm the observation is track-agnostic and lock it with a test. ObservationV2 is built from local, relative features only (speed, heading error, signed lateral offset over half-width, curvature lookahead, edge-distance beams, tire tail), with no absolute position. The observation does not change; `OBS_VERSION` stays 2.
- Turn the env's single bound track into a sampled pool. The env reads a list of circuit ids from config, holds a loaded track and a precomputed edge cache per id, and on each `reset` draws one circuit from the env RNG and rebinds the active track, edge cache, lap timer, and pole. Per-circuit precompute is built once and cached, so the per-reset cost is a lookup, not a rebuild.
- Resolve each circuit's pole from its track config. The pole lives in `configs/track/<id>.yaml`, not in the geometry `.npz`, so the env and the benchmark read the right pole for the drawn circuit.
- Widen the pool with the curriculum. Reuse the Phase 3b stage-table mechanism, applied to the workers by timestep threshold through an env method, to grow the pool from a few easy circuits to the full calendar, touching only the sampling, never the observation, so there is no mid-run retrain.
- Continue the Phase 3b policy across the pool. The observation is unchanged and the checkpoint validates, so warm-start and continue; retrain from scratch only if the warm start fails to generalize.
- Emit the lap-time table. A deterministic calendar benchmark runs one clean lap per pool circuit, reads each pole, and produces a table of achieved lap, pole, delta, and the twice-pole flag, logged to Weights and Biases and saved as the phase artifact.

External components the solution interacts with or alters: the Python env and training scripts, the track loader and configs, the Weights and Biases service, the local filesystem, and the web app for the circuit-aware live view and the table.

Dependencies: the Phase 3 training setup, Stable-Baselines3, PyTorch, Gymnasium, NumPy, Weights and Biases, and the existing physics, track, env, and app modules. No new third-party dependency.

Pros of the proposed solution: one policy across the calendar, no observation or physics change, a cheap per-reset circuit swap, and a per-circuit lap-time table.

Cons of the proposed solution: generalization is harder than one circuit, so it needs more steps and possibly curriculum tuning, and holding several tracks and edge caches in memory per worker costs RAM.

#### Data Model and Schema Changes

```
ObservationV2:           # UNCHANGED, length 22, OBS_VERSION = 2
  ... as Phase 3b ...    # local/relative only, no absolute position

CircuitPool (config):
  circuits: [id, ...]    # ids of built circuits under data/tracks/
  sampling: uniform      # per-episode draw from the env RNG (weights optional)
  # each id resolves its pole from configs/track/<id>.yaml

Curriculum stage (extended):
  start_step
  circuits: [id, ...]    # the active pool from this step on (widen over time)
  ... existing condition overrides (mu_base, wear_rate, weather) ...

Checkpoint meta:
  ... as Phase 3b ...
  obs_version            # still 2, so the Phase 3b checkpoint warm-starts cleanly
  circuit_id             # now records the pool descriptor (e.g. "calendar"), not one track
```

Modified data: the env's single track id becomes a sampled pool, the curriculum stage gains an optional pool list, and the checkpoint's `circuit_id` records the pool rather than one circuit. Unchanged: the observation vector and version, the action shape, the physics interface, and the rest of the checkpoint schema. Validation: the checkpoint loader still refuses a mismatched `obs_version` or action shape; the pool loader refuses an id with no built `.npz` and an id with no track config (so the pole is never silently zero).

#### Business Logic

- Pool construction: at env build, load every pool circuit through the runtime-safe loader and precompute its edge cache once; resolve each circuit's pole from its track config.
- Per-episode sampling: on `reset`, draw a circuit from the env RNG, rebind the active track, edge cache, lap timer, and pole, then run start-state randomization on the drawn circuit as before.
- Curriculum over the pool: a callback finds the active stage for the current timestep and pushes the stage's pool (and any condition overrides) into every worker through an env method, exactly as Phase 3b pushes conditions, so the pool widens with no observation change.
- Calendar benchmark: a deterministic evaluation iterates the pool, runs one clean lap per circuit with the saved normalization stats, reads each pole, and assembles the table of achieved lap, pole, delta, and twice-pole flag.
- Error states: a pool id with no built `.npz` is refused with the build hint, a pool id with no track config or a non-positive pole skips that circuit's delta and flags it (never compare against zero), and a checkpoint with a mismatched observation version is refused as before.
- Failure scenarios: the laptop stops, handled by frequent checkpoints and clean resume, a Weights and Biases outage, handled by local logging, and a policy that generalizes poorly, handled by curriculum widening, more steps, or reverting to retrain-from-scratch.
- Limitations: one car, the dynamic model is slower per step, and several tracks plus edge caches per worker cost RAM.

#### Presentation Layer

- User requirements: pick any circuit and watch the one policy drive it, and read the per-circuit lap-time table.
- UI changes: the watch-live circuit selector loads any built circuit for the same checkpoint, and the telemetry bar names the current circuit and its pole alongside the existing lap time and delta. The lap-time table is shown as the phase result, colored by the Phase 1 timing colors.
- Web concerns: live inference streams over the existing WebSocket and the canvas renders as before; the only new readouts are the circuit name and its pole. No new rendering work.
- UI states: a circuit selected and the one policy driving it live, and the calendar table shown.
- Error handling: a clear message when a circuit or checkpoint cannot load, and the viewport never crashes on a bad frame.

#### Other questions to answer

- How will it scale: vectorized envs use the laptop cores, each worker samples its own circuit per reset for diverse rollouts, and the checkpoint stays resumable and portable for Phase 5.
- Limitations: as above.
- Recovery on failure: resume from the last checkpoint, saved often.
- Future requirements: the generalist checkpoint becomes the shared brain for the Phase 5 multi-agent self-play.

### c. Test Plan

Tests are written by an independent test subagent from this spec and the public interfaces, not from the implementation, so they verify the requirements and stay unbiased. See section 5.

- Unit tests: the observation carries no absolute position and is same-shape and in-bounds on two different circuits with the same car state, the pool loader refuses an unbuilt id and an id with no track config, per-episode sampling draws different circuits across resets and is reproducible from a fixed seed, the rebind swaps the track, edge cache, lap timer, and pole together with no stale per-track state, each circuit's pole resolves from its config, and the curriculum picks the right pool for a given timestep.
- Integration tests: a short smoke training run on a small pool shows the reward trending up across circuits, a checkpoint resumes and continues on the pool, the Phase 3b checkpoint warm-starts on the pool without an obs-version error, the app loads one checkpoint and drives two different circuits, and the calendar benchmark produces a table with a row per pool circuit.
- QA: confirm one policy laps several circuits cleanly back to back, confirm the per-circuit deltas are believable, and confirm the live view names the right circuit and pole.

### d. Monitoring and Alerting Plan

- Logging: Weights and Biases as the primary, local logs as a fallback.
- Metrics: per-circuit lap time, per-circuit delta to the pole, per-circuit twice-pole flag, mean and worst delta across the pool, off-track count, episode return, the active pool size from the curriculum, and steps per second.
- Observability: the learning curves, periodic eval clips on a rotating circuit, the calendar table, and the on-screen debug overlay.
- Alerting: none external. Watch the curves and the worst-circuit delta. A collapsing return or one circuit stuck far off the pole is the signal to adjust the curriculum or re-shape.

### e. Release, Roll-out, and Deployment Plan

- Branch `phase-4-many-circuits`. Merge when one policy laps every pool circuit and reports a lap-time table against the pole per circuit, with twice the pole reached on the bulk of the calendar and the deltas closing over training. The PR description carries the run summary, the curves, and the calendar table.

### f. Rollback Plan

- Liabilities: a sampling or env change could break the drive loop or the app, and a training change could waste compute.
- Reduce liabilities: keep main working, develop on the branch, tag the working commit, checkpoint often, and keep the last good checkpoint.
- Prevent spread: the change is confined to the env's track binding, the curriculum, the benchmark, and the live-view readouts; the observation, the physics interface, the action space, and the checkpoint schema are unchanged, so a single-circuit env still works by setting a one-circuit pool. Revert the merge or restore the tagged commit and the last good checkpoint if needed. Delete the branch after a clean merge and a passing test.

### g. Alternate Solutions or Designs

- Warm start versus retrain. Warm start is chosen, since the observation is unchanged and the Phase 3b checkpoint validates. Retrain from scratch is the fallback if the warm start fails to generalize.
- Per-episode sampling versus per-worker fixed circuit. Per-episode sampling is chosen for the most diverse rollouts. Pinning a circuit per worker is a fallback if per-episode swapping ever costs too much.
- Curriculum pool widening versus the full calendar from step zero. Widening is chosen to ease generalization. Training straight on the full pool is the fallback if the curriculum adds little.
- Precompute and cache per circuit versus rebuild on reset. Precompute is chosen, since the edge cache is the per-reset cost; rebuilding every reset is rejected as wasteful.
- Migration: a one-circuit pool reproduces the Phase 3b behavior exactly, so the change is backward compatible and switching pools is a config change.

---

## 3. Success Evaluation

### a. Impact

- Security: local training and a localhost WebSocket, no exposure, and validated checkpoints.
- Performance: the dynamic model is slower per step, several tracks and edge caches per worker cost RAM, and vectorized envs use the laptop cores.
- Cost: zero on the laptop, free tiers on the cloud if used.
- Impact on other components: sets the generalist policy and the pool-sampling env that Phase 5's multi-agent self-play reuses.

### b. Metrics

- Capture: per-circuit final lap time and delta to the pole, the twice-pole rate across the pool, the worst-circuit delta, the gap trend over training, off-track count, and steps per second.
- Tools: Weights and Biases, the eval clips, the calendar table, and the on-screen overlay.
- Definition of done: one policy laps every pool circuit and reports a lap-time table against the pole per circuit, with twice the pole reached across the bulk of the calendar and the deltas closing over training.

---

## 4. Deliberation

### a. Discussion

- The pool order, which circuits count as easy starters and how fast to widen to the full calendar.
- How many steps generalization needs versus one circuit, and whether the warm start shortens it.
- Whether per-episode sampling or per-worker pinning gives better throughput on the laptop.

### b. Open Questions

- The exact curriculum stages, the starter circuits, and the widening thresholds, set during training.
- Whether to weight the sampling toward harder or worse-performing circuits, or keep it uniform.
- The memory ceiling, how many tracks and edge caches a worker can hold before RAM is the limit.
- Whether to hold out a circuit to measure transfer, deferred unless the calendar grows.

---

## 5. Implementation Subagents

These become definitions under `.claude/agents/`. Each has a narrow scope and its own context, which saves tokens, since the main thread never carries the whole job. The test author is independent, so tests verify the spec and stay unbiased.

How they work together. The plan you build dispatches tasks in dependency order. Feature subagents read this spec and the technical design. The test subagent reads only this spec and the public interfaces, not the implementation. The reviewer gates each merge.

ONE MAIN THING NOT TO FORGET. WHENEVER THE YOU CALL THE SUBAGENT TO WORK ON ANY PROBLEM. THIS SHOULD BE SET TO BY DEFAULT. TO CALL THE SKILL /caveman BEFORE ANY CONVERSATION. THIS IS NON-NEGOIATIABLE. THE /caveman IS THE IMPORTANT SKILL WRITE THIS IN ALL SUB-AGENTS FILES SO THEY REMEMBER BY DEFAULT WHAT TO DO.

### Roster

- env-engineer. Scope: `src/f1rl/env/` and the env construction seam in `src/f1rl/env/factory.py`. The main role in Phase 4. Turns the single bound track into a config-driven circuit pool, precomputes and caches one edge cache per pool circuit, draws a circuit from the env RNG on each `reset`, rebinds the track, edge cache, lap timer, and pole together, resolves each circuit's pole from its track config, and exposes the env method the curriculum calls to set the active pool. No observation change, no absolute position leak. Inputs: this spec and technical design sections seven and ten. Tools: read and edit files, run the Gymnasium env checker. Output: a pool-sampling env that passes the checker and reproduces the single-circuit behavior on a one-circuit pool.

- training-engineer. Scope: `src/f1rl/train/`. Extends the curriculum stage table to widen the pool over timesteps, wires the warm-start continue from the Phase 3b checkpoint, runs the retrain, and builds the calendar lap-time-table benchmark that iterates the pool and scores each circuit against its pole. Inputs: this spec and technical design sections twelve and fifteen. Tools: read and edit files, run a short smoke training run, log to Weights and Biases. Output: training and benchmark scripts and a smoke run where the reward trends up across circuits. Depends on the env.

- app-integration-engineer. Scope: the backend live path and the frontend readouts. Lets the watch-live selector drive any built circuit for one checkpoint, names the current circuit and its pole in the telemetry bar, and shows the calendar table. Inputs: this spec and the Phase 1 and Phase 3 app. Tools: read and edit files in the backend and frontend, manual check in the browser. Output: a circuit-aware live view and the table. Depends on the checkpoint and the env.

- test-engineer, independent. Scope: `tests/`. Writes the unit and integration tests in section c from this spec and the public interfaces and schemas, not from the implementation source. Forbidden from reading implementation internals to shape tests, so the tests check the contracts. Inputs: this spec, the interface signatures, and the data schemas. Tools: read the spec and the signatures, read and write the tests folder, run the suite. Output: an unbiased test suite mapped to the acceptance criteria.

- reviewer. Scope: read-only review and the test run. Reviews each diff against this spec and the conventions, config-driven values, SI units, no absolute position in the observation, no observation or physics-interface change, no rendering in the training hot path, and the runtime-safe loader (no FastF1 in the training loop), then runs the full suite and the linter and reports. Writes no feature code. Inputs: the diffs, this spec, and the conventions. Tools: read files, run tests, run the linter. Output: a pass or fail review per task, with reasons.

Note on the physics role: Phase 4 changes no physics. The dynamic model and the grip pipeline from Phase 3b are reused unchanged behind the `PhysicsModel` interface, so there is no physics-engineer task this phase.

### Notes

- Token saving comes from the narrow scopes and the separate contexts.
- Unbiased tests come from the test-engineer working from the spec contracts, separate from the agent that wrote the code.
- Dependency order: env-engineer first, then training-engineer, then app-integration-engineer, with test-engineer working in parallel from the spec and reviewer gating each merge.
