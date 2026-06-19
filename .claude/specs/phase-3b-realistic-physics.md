# Phase 3 Part 2 Spec: Realistic Physics and Lap-Time Benchmark

Status: draft for plan mode. Branch: `phase-3b-realistic-physics`. Depends on Part 1, `03a-training-core.md`, being complete, the training loop, the env, the checkpoint format, and the live view. Also depends on Phase 2, the circuits, the surface zones, and the official length and pole.

---

## 1. Introduction

### a. Overview, Problem Description, Summary

Make the world real. Part 1 proved the agent can learn to lap on the simple kinematic physics. Part 2 swaps in the dynamic physics with the grip pipeline, tires, and weather, behind the same interface, extends the observations, retrains, and benchmarks the lap time against the real pole.

Suggested solution, in one line: replace the kinematic model with the dynamic model and the grip pipeline behind the `PhysicsModel` interface, add tires and weather to the observation, retrain with a simple curriculum, and score the lap time against the pole.

Stakeholders: you, as the developer and primary user. Secondary: future portfolio reviewers, who will watch the trained car drive on realistic physics.

### b. Glossary or Terminology

- Dynamic bicycle model: a car model with lateral velocity, yaw rate, and tire forces, the realistic step.
- Friction circle: the combined grip limit on the tire force, longitudinal plus lateral.
- Grip pipeline: one grip scalar that tires, weather, and surface multiply.
- Tire compound: soft, medium, hard, intermediate, wet, each with its own grip and wear.
- Tire wear: a value from zero to one that lowers grip as it rises.
- Weather factor: a grip multiplier, dry near one, wet lower.
- Surface factor: a grip multiplier per zone, asphalt, kerb, grass, gravel.
- Curriculum: training on easier conditions first, then harder.
- Pole lap: the real fastest lap time for the circuit, the benchmark.
- Delta: the time gap to the pole lap, positive when slower.
- Calibration: tuning the car physics so a clean optimal lap lands near the real pole.
- Pacejka: a richer tire force formula, an optional upgrade.
- Subagent: a focused Claude Code agent with its own context, role, and tools.

### c. Context or Background

- Why worth solving: realistic driving is what makes the agent feel like real F1 and what makes the lap-time benchmark meaningful. A kinematic car cannot lose grip, so the racing has no edge until this part.
- Origin: Phase 3 of the visibility-first build order, the second of two parts.
- How it affects the goals: it serves fun, grip-limited driving looks like real racing, resume, the lap-time score against the pole is a clear result, and learning, the vehicle dynamics and the grip model are deep work.
- Past efforts: Part 1 trained on kinematic physics. This part replaces the model.
- Roadmap fit: Phase 4 generalizes this agent across circuits, Phase 5 adds many cars, Phase 6 adds racing. All reuse the dynamic physics and the benchmark built here.
- Technical strategy fit: the dynamic model swaps in behind the stable `PhysicsModel` interface with no env API change, so the env, the training loop, and the app keep working.

### d. Goals or Product and Technical Requirements

Product requirements as user stories:

- As the user, I see the car drive on realistic physics, losing grip when overdriven, so the driving feels like real F1.
- As the user, I see tires wear and the weather change the grip, so the world is real.
- As the user, I see a lap-time score against the real pole, so I can measure how good the agent is.

Technical requirements, functional and required:

- The dynamic bicycle model with a friction-circle force limit, behind the `PhysicsModel` interface, with no env API change.
- The grip pipeline, effective grip equals a base value times a tire factor for compound and wear, times a weather factor, times a surface factor from the Phase 2 zones.
- Observations version 2, version 1 plus tire wear, a compound indicator, and a grip or weather indicator. The vector changes, so retrain.
- Reward version 2: the same progress core plus an **opt-in** slip/spin penalty (`w_slip`, `slip_threshold`). With `w_slip = 0` (the shipped default) it is numerically identical to reward v1 — the extra shaping term is structurally present but **left un-tuned by default**; turning it on and re-shaping for the dynamics is expected follow-up, not a precondition for this phase. Never centerline-seeking.
- A simple curriculum, high grip and no wear and dry weather first, then wear and weather.
- A lap-time benchmark against the official pole, with twice the pole as the first milestone, and the delta logged.
- Physics calibration so a clean optimal lap lands near the real pole, so the score is fair.
- A retrain, since the observation and the dynamics changed. No reliance on a warm start.
- Deterministic seeding, with the seed recorded in every run and checkpoint.
- Runs on the laptop CPU with no GPU, headless and device-agnostic, reusing the Part 1 training setup.

### e. Non-Goals or Out of Scope

- No training across many circuits. That is Phase 4. The mechanism is already built in Part 1.
- No multiple cars and no racing. Phase 5 and Phase 6.
- No pit stops. Phase 7.
- No JAX or GPU-accelerated environment. A later optimization.
- No new track data work. Phase 2 owns the circuits and the surfaces.
- No elevation. Optional and low priority, default flat.
- No mobile.

### f. Future Goals

- Phase 4, one policy across all circuits, continuing this checkpoint with track sampling.
- Pacejka tires, only if the linear tire model feels too plain.
- A reward-shaping dashboard to compare runs faster.

### g. Assumptions

- Part 1 is complete, the training loop, the env, the checkpoint format, and the live view all work.
- The Phase 2 tracks carry the surface zones and the official length and pole.
- The `PhysicsModel` interface from Part 1 is stable and ready for the swap.
- A Weights and Biases account is set up and logs from the laptop.

---

## 2. Solutions

### a. Current or Existing Solution

The Part 1 agent drives on kinematic physics. The car cannot lose grip, there are no tires or weather, and there is no pole benchmark.

- Pros: a working, learning agent that laps the circuit.
- Cons: no realism in the driving, and no measure against the real pole.

### b. Suggested or Proposed Solution

- Swap the kinematic model for the dynamic bicycle model with a friction-circle limit, behind the `PhysicsModel` interface, with no env API change. Linear tires first, Pacejka optional.
- Grip pipeline: effective grip equals a base value times a tire factor for compound and wear, times a weather factor, times a surface factor for asphalt, kerb, grass, and gravel from the Phase 2 tracks. One grip scalar gates the friction circle.
- Observation version 2: add tire wear, a compound indicator, and a grip or weather indicator.
- Reward version 2: the same progress core, with an opt-in slip/spin penalty (`w_slip`, default 0 ⇒ identical to v1) reserved for re-shaping under the harder dynamics, and tire management appearing over longer episodes. The structure ships; the tuning is deferred.
- Curriculum: start with high grip and no wear and dry weather, then introduce wear and weather, so the agent learns the basics before the hard cases.
- Lap-time benchmark: compute the lap time, compare to the official pole and to twice the pole, and log the delta. Calibrate the car physics so a clean optimal lap lands near the pole.
- Retrain from scratch, since the observation and the dynamics changed. A warm start from the Part 1 policy does not transfer cleanly across a changed observation, so do not rely on it.

External components the solution interacts with or alters: the Python physics and env, the training scripts, the Weights and Biases service, the local filesystem, and the web app for the realistic live view and the score.

Dependencies: the Part 1 training setup, Stable-Baselines3, PyTorch, Gymnasium, NumPy, Weights and Biases, and the existing physics, track, env, and app modules.

Pros of the proposed solution: realistic, grip-limited driving, a single physics source behind a stable interface, and a measurable lap-time score against the pole.

Cons of the proposed solution: the dynamic model is harder for the agent and slower per step, and the reward likely needs re-shaping for the new dynamics.

#### Data Model and Schema Changes

```
ObservationV2 = ObservationV1 plus:
  tire_wear
  compound_onehot[5]
  grip_indicator

GripPipeline:
  effective_grip = base * tire(compound, wear) * weather * surface
  # friction circle: max combined tire force = effective_grip * mass * g (+ optional aero)

Checkpoint:
  ... as Part 1 ...
  obs_version            # 2 here, so a loader refuses a mismatched policy
```

Modified data: the observation grows from version 1 to version 2, and the physics model behind the interface changes from kinematic to dynamic. Validation: the loader checks the observation version before resuming, grip factors stay in sensible bounds, and the benchmark reads the official length and pole from the Track and the circuit metadata.

#### Business Logic

- Dynamic step: integrate the dynamic model with substeps, with the combined tire force capped by the friction circle using the grip scalar.
- Grip: compute effective grip from the base value, the tire factor, the weather factor, and the surface factor at the car position each step.
- Curriculum: schedule grip and weather from easy to hard over training.
- Benchmark: an evaluation callback runs a deterministic lap, records it, computes the lap time and the delta to the pole, logs the clip and the numbers, and renders a short clip.
- Error states: a checkpoint with a mismatched observation version is refused with a clear message, a missing pole skips the benchmark and flags it, and a diverged run is caught by watching the curves and reverting to the last good checkpoint.
- Failure scenarios: the laptop stops, handled by frequent checkpoints and clean resume, a Weights and Biases outage, handled by local logging, and reward collapse on the new dynamics, handled by reverting and re-shaping.
- Limitations: one car, one circuit, and the dynamic model is slower per step.

#### Presentation Layer

- User requirements: watch the car drive on realistic physics, see tires and weather take effect, and read the lap-time score.
- UI changes: the telemetry bar shows the tire compound and wear, a grip or weather indicator, the lap time, and the delta to the pole, colored by the timing colors from Phase 1. The watch-live mode runs a dynamic-physics checkpoint.
- Web concerns: live inference streams over the existing WebSocket, and the canvas renders as before, so no new rendering work beyond the new readouts.
- UI states: running a dynamic-physics checkpoint live, and a benchmark result shown.
- Error handling: a clear message when a checkpoint cannot load, and the viewport never crashes on a bad frame.

#### Other questions to answer

- How will it scale: vectorized envs use the laptop cores, and the checkpoint is resumable and portable for Phase 4.
- Limitations: as above.
- Recovery on failure: resume from the last checkpoint, saved often.
- Future requirements: the same dynamic physics and checkpoint feed the multi-circuit training in Phase 4 and the self-play in Phase 5.

### c. Test Plan

Tests are written by an independent test subagent from this spec and the public interfaces, not from the implementation, so they verify the requirements and stay unbiased. See section 5.

- Unit tests: the friction circle caps the combined tire force, each grip factor lowers grip in the right direction, tire wear lowers grip as it rises, the observation version 2 has the right shape and stays in bounds, and the lap-time and delta computation is correct against a known reference.
- Integration tests: a short smoke training run on the dynamic physics shows the reward trending up, a checkpoint resumes and continues, the app loads a dynamic-physics checkpoint and shows realistic driving and the readouts, and the benchmark produces a number against the pole.
- QA: confirm the driving feels grip-limited, confirm tires and weather visibly change the driving, and confirm the lap-time score is believable.

### d. Monitoring and Alerting Plan

- Logging: Weights and Biases as the primary, local logs as a fallback.
- Metrics: lap time, delta to the pole, whether twice the pole is reached, the gap-to-pole trend, off-track count, episode return, tire-wear behavior, and steps per second.
- Observability: the learning curves, periodic eval clips, and an on-screen debug overlay.
- Alerting: none external. Watch the curves. A collapsing return is the signal to stop and re-shape.

### e. Release, Roll-out, and Deployment Plan

- Branch `phase-3b-realistic-physics`. Merge when the agent laps on the dynamic physics with tires and weather and reports a lap-time score against the pole, with twice the pole reached and the gap to the pole closing. The PR description carries the run summary, the curves, and the benchmark.

### f. Rollback Plan

- Liabilities: a physics change could break the drive loop or the app, and a training change could waste compute.
- Reduce liabilities: keep main working, develop on the branch, tag the working commit, checkpoint often, and keep the last good checkpoint.
- Prevent spread: the physics sits behind the interface, so the physics change does not touch the env API or the app. Revert the merge or restore the tagged commit and the last good checkpoint if needed. Delete the branch after a clean merge and a passing test.

### g. Alternate Solutions or Designs

- Warm start versus retrain. Retrain is chosen, since the observation and the dynamics changed. A warm start does not transfer cleanly and is not relied on.
- Linear tires versus Pacejka. Linear first for simplicity, Pacejka as an upgrade only if the feel needs more.
- Curriculum versus no curriculum. A curriculum is chosen to ease the harder dynamics. Training straight on the full difficulty is the fallback if the curriculum adds little.
- Local versus cloud training. Local headless is the default, reused from Part 1. The cloud is the fallback for long unattended runs.
- Migration: the env API and the checkpoint schema are stable, so switching the tire model or the training device is a config change.

---

## 3. Success Evaluation

### a. Impact

- Security: local training and a localhost WebSocket, no exposure, and validated checkpoints.
- Performance: the dynamic model is slower per step than kinematic, and vectorized envs use the laptop cores.
- Cost: zero on the laptop, free tiers on the cloud if used.
- Impact on other components: sets the dynamic physics, the grip model, and the benchmark that Phases 4 through 6 reuse.

### b. Metrics

- Capture: final lap time, delta to the pole, whether twice the pole is reached, the gap-to-pole trend over training, off-track count, and steps per second.
- Tools: Weights and Biases, the eval clips, and the on-screen overlay.
- Definition of done: the agent laps on the dynamic physics with tires and weather, and reports a lap-time score against the pole, with twice the pole reached and the gap to the pole closing.

---

## 4. Deliberation

### a. Discussion

- The curriculum order and how fast to introduce wear and weather.
- How much reward re-shaping the new dynamics need.
- The physics calibration target, so a clean optimal lap lands near the real pole.

### b. Open Questions

- The calibration values, so the simulated optimal lap approaches the real pole.
- The off-track and grip penalty tuning for the dynamic model, set during shaping.
- Linear tires versus Pacejka, and when to upgrade.
- The local versus cloud decision, reused from the Part 1 benchmark.

---

## 5. Implementation Subagents

These become definitions under `.claude/agents/`. Each has a narrow scope and its own context, which saves tokens, since the main thread never carries the whole job. The test author is independent, so tests verify the spec and stay unbiased.

How they work together. The plan you build dispatches tasks in dependency order. Feature subagents read this spec and the technical design. The test subagent reads only this spec and the public interfaces, not the implementation. The reviewer gates each merge.

ONE MAIN THING NOT TO FORGET. WHENEVER THE YOU CALL THE SUBAGENT TO WORK ON ANY PROBLEM. THIS SHOULD BE SET TO BY DEFAULT. TO CALL THE SKILL /caveman BEFORE ANY CONVERSATION. THIS IS NON-NEGOIATIABLE. THE /caveman IS THE IMPORTANT SKILL WRITE THIS IN ALL SUB-AGENTS FILES SO THEY REMEMBER BY DEFAULT WHAT TO DO.

### Roster

- physics-engineer. Scope: `src/f1rl/physics/`. The main role in Part 2. Builds the dynamic bicycle model, the friction circle, the grip pipeline, the tire model, and the weather and surface factors, all behind the `PhysicsModel` interface, with no env API change. Inputs: this spec and technical design sections four and five. Tools: read and edit files, run physics unit tests. Output: dynamic physics that pass their unit checks.

- env-engineer. Scope: `src/f1rl/env/`. Extends the observation to version 2 with tire wear, compound, and grip, and adjusts the reward version 2 for the harder dynamics. Inputs: this spec and technical design sections seven through nine. Tools: read and edit files, run the Gymnasium env checker. Output: an env on observation version 2 that passes the checker. Depends on the physics.

- training-engineer. Scope: `src/f1rl/train/`. Adds the curriculum, runs the retrain, and builds the lap-time benchmark against the pole with the calibration. Inputs: this spec and technical design sections eleven and twelve. Tools: read and edit files, run a short smoke training run, log to Weights and Biases. Output: training scripts, a benchmark, and a smoke run where the reward trends up. Depends on the env.

- app-integration-engineer. Scope: the frontend readouts and the live view. Shows the tire compound and wear, the grip or weather indicator, the lap time, and the delta to the pole, and runs a dynamic-physics checkpoint live. Inputs: this spec and the Phase 1 app. Tools: read and edit files in the backend and frontend, manual check in the browser. Output: the realistic live view with the score. Depends on the checkpoint and the env.

- test-engineer, independent. Scope: `tests/`. Writes the unit and integration tests in section c from this spec and the public interfaces and schemas, not from the implementation source. Forbidden from reading implementation internals to shape tests, so the tests check the contracts. Inputs: this spec, the interface signatures, and the data schemas. Tools: read the spec and the signatures, read and write the tests folder, run the suite. Output: an unbiased test suite mapped to the acceptance criteria.

- reviewer. Scope: read-only review and the test run. Reviews each diff against this spec and the conventions, config-driven values, SI units, no rendering in the training hot path, and a stable interface, then runs the full suite and the linter and reports. Writes no feature code. Inputs: the diffs, this spec, and the conventions. Tools: read files, run tests, run the linter. Output: a pass or fail review per task, with reasons.

### Notes

- Token saving comes from the narrow scopes and the separate contexts.
- Unbiased tests come from the test-engineer working from the spec contracts, separate from the agent that wrote the code.
- Dependency order: physics-engineer first, then env-engineer, then training-engineer, then app-integration-engineer, with test-engineer working in parallel from the spec and reviewer gating each merge.