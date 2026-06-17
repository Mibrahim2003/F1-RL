# Phase 3 Part 1 Spec: Training Core

Status: draft for plan mode. Branch: `phase-3a-training-core`. Depends on Phase 2, the real circuits and the cached Track format. Reuses the Phase 1 app, viewport, engine, and WebSocket. Part 2, realistic physics, follows in `03b-realistic-physics.md`.

---

## 1. Introduction

### a. Overview, Problem Description, Summary

Get the agent to learn at all. One car learns to drive one real circuit on its own, on the simple kinematic physics. At first it runs on an untrained policy and drifts, spins, and leaves the track while you watch in the app. Then it learns, and over time it laps cleanly.

This part isolates the riskiest new thing, learning, from the physics complexity. The realistic physics waits for Part 2.

Suggested solution, in one line: a PPO training loop on the existing kinematic physics and one Phase 2 circuit, with checkpoint and resume, Weights and Biases logging, and live viewing of a checkpoint in the app.

Stakeholders: you, as the developer and primary user. Secondary: future portfolio reviewers.

### b. Glossary or Terminology

- Policy: the network that maps an observation to an action.
- Untrained policy: random weights, so the car drives badly at first.
- Observation: the vector the policy sees each step.
- Action: steering and a combined throttle and brake command.
- Reward: the per-step signal the agent learns to maximize.
- Reward shaping: tuning the reward terms to get the behavior you want.
- Episode: one run from reset to termination.
- Step or timestep: one control update at the fixed rate. The budget is counted in steps.
- PPO: Proximal Policy Optimization, the on-policy algorithm in use.
- Vectorized envs: many env copies running in parallel to feed the algorithm faster.
- Checkpoint: a saved model plus the state needed to resume exactly.
- Resume: continue a run from a checkpoint after a stop.
- Start-state randomization: resetting the car at random points along the lap.
- Headless training: training with no rendering, as a background process.
- Subagent: a focused Claude Code agent with its own context, role, and tools.

### c. Context or Background

- Why worth solving: the learned driver is the heart of the project. Everything before this built the world. This part makes a car teach itself to drive, the first real proof the idea works.
- Origin: Phase 3 of the visibility-first build order, the first of two parts.
- How it affects the goals: it serves fun, watching a car learn is the payoff, resume, a trained RL agent is the headline, and learning, the RL work is the deepest skill in the project.
- Past efforts: none. This is the first learning agent in the project.
- Roadmap fit: Part 2 makes the physics realistic, Phase 4 generalizes across circuits, Phase 5 adds many cars, Phase 6 adds racing. All of them stand on this loop.
- Technical strategy fit: physics stays single-source in Python behind the `PhysicsModel` interface, training is device-agnostic and reproducible from a config and a seed, and the recorded-trajectory format feeds the live view.

### d. Goals or Product and Technical Requirements

Product requirements as user stories:

- As the user, I drop a car on a circuit and watch the untrained policy fail, so I see the starting point.
- As the user, I start training and watch the agent improve by loading checkpoints into the app live, so progress is visible.
- As the user, I see learning curves, so I know training is working.
- As the user, I can stop and resume training without losing progress, so a closed laptop costs nothing.

Technical requirements, functional and required:

- A PPO training loop with vectorized envs across the laptop cores, and observation normalization.
- Observations version 1, a fixed-length vector.
- Reward version 1, progress-based, never rewarding centerline proximity.
- Checkpoint and resume, saving everything needed to continue exactly.
- Weights and Biases logging, with local logs as a fallback.
- Headless local training as the default, with a device-agnostic script that runs the same locally or on the cloud, and the same checkpoints across both.
- A steps-per-second benchmark on the laptop and on a cloud instance, run early, to pick the training device with data.
- Start-state randomization, the car resets at random points along the centerline, so the agent learns the whole lap, not only the run from the start line.
- The deep-repetition training philosophy applied to one circuit here. The resumable checkpoint and a circuit switch are built now, so Phase 4 continues the same policy on more circuits.
- Deterministic seeding, with the seed recorded in every run and checkpoint.
- Runs on the laptop CPU with no GPU.

### e. Non-Goals or Out of Scope

- No dynamic physics, tires, weather, or grip model. That is Part 2. Part 1 uses the kinematic model.
- No lap-time benchmark against the pole. Part 2 owns the benchmark. Part 1 logs lap time only as a basic metric.
- No training across many circuits. The mechanism is built here, the act is Phase 4.
- No multiple cars and no racing. Phase 5 and Phase 6.
- No pit stops. Phase 7.
- No JAX or GPU-accelerated environment. A later optimization.
- No new track data work. Phase 2 owns the circuits.
- No mobile.

### f. Future Goals

- Part 2, the dynamic physics, grip pipeline, tires, and weather, with the pole benchmark.
- Phase 4, one policy across all circuits, continuing this checkpoint with track sampling.
- A JAX environment for large speedups, only if training becomes the bottleneck.

### g. Assumptions

- A Phase 2 circuit loads and renders, with the cached Track format.
- The laptop has several CPU cores for parallel envs.
- A Weights and Biases account is set up and logs from the laptop.
- The kinematic physics from Phase 1 works as the Part 1 model.
- The web app can load a checkpoint and run a policy live over the existing WebSocket, the watch-live mode.

---

## 2. Solutions

### a. Current or Existing Solution

No learning agent exists. The car is driven only by the keyboard or by a simple centerline-following script from Phase 1.

- Pros: proves the live path and the drive loop.
- Cons: no learning, no autonomy.

### b. Suggested or Proposed Solution

- Pick one clean circuit, for example the Red Bull Ring, built in Phase 2. Keep the kinematic physics.
- Observation version 1: normalized speed, heading error against the track tangent at the nearest centerline point, signed lateral offset divided by the half-width as input only, track curvature at five lookahead distances, and seven rangefinder distances to the asphalt edge. Fixed length.
- Action: two continuous values, steering and a combined throttle and brake, each in minus one to plus one.
- Reward version 1: progress along the centerline as the main term, a graded off-track penalty, a small per-step penalty, and a penalty for going backward. Never reward staying near the center.
- Termination: success on completing the target laps, failure on a large off-track or wrong-way event, truncation on a step limit.
- Algorithm: PPO from Stable-Baselines3, vectorized envs across the laptop cores, observation normalization.
- Checkpoint and resume: save the model, the optimizer, the normalization stats, the timestep count, the config, and the RNG states, and resume exactly.
- Live viewing: pull a checkpoint and run the policy in the app over the WebSocket, so you watch the agent improve between runs.
- Start-state randomization: reset at random points along the centerline.
- Training budget: start around two to five million steps on the circuit, tunable, and raise it for a polished lap.

External components the solution interacts with or alters: the Python engine and env, the training scripts, the Weights and Biases service, the local filesystem for checkpoints and trajectories, and the web app for live viewing.

Dependencies: Stable-Baselines3, PyTorch, Gymnasium, NumPy, Weights and Biases, and the existing physics, track, env, and app modules.

Pros of the proposed solution: a small, focused path that proves learning works, a single physics source, device-agnostic training, and live visibility throughout.

Cons of the proposed solution: reward shaping is fiddly and the most iterated work, and training takes hours.

#### Data Model and Schema Changes

```
ObservationV1:
  speed_norm
  heading_error
  lateral_offset_norm        # input only, never rewarded toward zero
  curvature_lookahead[5]
  edge_distance_beam[7]

Action:
  steer in [-1, 1]
  longitudinal in [-1, 1]    # >= 0 throttle, < 0 brake

Checkpoint:
  model_weights
  optimizer_state
  obs_normalization_stats
  total_timesteps
  config
  rng_states
  circuit_id
  obs_version                # 1 here, so a loader checks compatibility
```

Validation: the loader checks the observation version and the action shape before resuming, the seed is recorded, and the observation stays in bounds.

#### Business Logic

- Training loop: at the fixed control rate, build the observation, step PPO, integrate the kinematic physics with substeps, compute the reward and termination, record the trajectory when evaluating, and checkpoint on a schedule.
- Randomization: random start positions along the centerline each reset.
- Device-agnostic run: one `train.py` driven by a config, the same locally and on the cloud, writing checkpoints to a configured path.
- Basic lap-time logging: log the lap time once laps complete, as a plain metric. The pole benchmark is Part 2.
- Error states: a checkpoint with a mismatched observation version is refused with a clear message, and a diverged run is caught by watching the curves and reverting to the last good checkpoint.
- Failure scenarios: the laptop stops, handled by frequent checkpoints and clean resume, a Weights and Biases outage, handled by local logging, and reward collapse, handled by reverting and re-shaping.
- Limitations: one car, one circuit, kinematic physics, and patient reward shaping.

#### Presentation Layer

- User requirements: watch the untrained car fail, watch it improve, and read the curves.
- UI changes: the watch-live mode runs a chosen checkpoint, an external Weights and Biases view shows the curves, and the telemetry bar shows the lap time.
- Web concerns: live inference streams over the existing WebSocket at the control rate, and the canvas renders as in Phase 1, so no new rendering work.
- UI states: no checkpoint selected, running a checkpoint live, and training in progress as seen through the curves.
- Error handling: a clear message when a checkpoint cannot load, and the viewport never crashes on a bad frame.

#### Other questions to answer

- How will it scale: vectorized envs use all laptop cores, and the checkpoint is resumable and portable for Phase 4.
- Limitations: as above.
- Recovery on failure: resume from the last checkpoint, saved often.
- Future requirements: the same loop and checkpoint feed Part 2 and the multi-circuit training in Phase 4.

### c. Test Plan

Tests are written by an independent test subagent from this spec and the public interfaces, not from the implementation, so they verify the requirements and stay unbiased. See section 5.

- Unit tests: the kinematic step is correct on a straight and a constant-radius turn, the observation has the right shape and stays in bounds, the reward rises with forward progress and falls off track, termination fires on a large off-track and on the wrong way, lap detection fires once per lap, the checkpoint round-trips exactly, and a fixed seed gives a reproducible run.
- Integration tests: a short smoke training run on a tiny budget shows the reward trending up, a checkpoint resumes and continues, and the app loads a checkpoint and runs the policy live.
- QA: watch the agent across checkpoints and confirm visible improvement.

### d. Monitoring and Alerting Plan

- Logging: Weights and Biases as the primary, local logs as a fallback.
- Metrics: episode return, episode length, off-track count, steps to the first clean lap, lap time, and steps per second.
- Observability: the learning curves, periodic eval clips, and an on-screen debug overlay for the live view.
- Alerting: none external. Watch the curves. A collapsing return is the signal to stop and re-shape.

### e. Release, Roll-out, and Deployment Plan

- Branch `phase-3a-training-core`. Merge when the agent laps the chosen circuit cleanly on kinematic physics, improvement is visible across checkpoints, and resume works. The PR description carries the run summary and the curves.

### f. Rollback Plan

- Liabilities: a bad training change could waste compute, and an env change could break the drive loop or the app.
- Reduce liabilities: keep main working, develop on the branch, tag the working commit, checkpoint often, and keep the last good checkpoint.
- Prevent spread: revert the merge or restore the tagged commit and the last good checkpoint if needed. Delete the branch after a clean merge and a passing test.

### g. Alternate Solutions or Designs

- PPO versus SAC. PPO is chosen for stability and parallel-env throughput. SAC is a fallback only if sample efficiency becomes the limit and parallelism stays low.
- Local versus cloud training. Local headless is the proposed default, since the bottleneck is the CPU and free cloud is core-starved. The cloud is the fallback for long unattended runs. Decide with the steps-per-second benchmark.
- Start straight on the real circuit versus a brief oval warm-up. Proposed: start on the real circuit, with the oval as a quick sanity check only if learning stalls.
- Migration: the env API and the checkpoint schema are stable, so switching the algorithm or the training device is a config change.

---

## 3. Success Evaluation

### a. Impact

- Security: local training and a localhost WebSocket, no exposure, and validated checkpoints.
- Performance: training takes hours, and vectorized envs use the laptop cores.
- Cost: zero on the laptop, free tiers on the cloud if used.
- Impact on other components: sets the training loop, the checkpoint format, and the observation and reward design that Part 2 and later phases reuse.

### b. Metrics

- Capture: steps to the first clean lap, off-track count over training, episode return, and steps per second.
- Tools: Weights and Biases, the eval clips, and the on-screen overlay.
- Definition of done: the agent laps the chosen circuit cleanly on kinematic physics, improvement is visible across checkpoints, and resume works.

---

## 4. Deliberation

### a. Discussion

- The starting training budget in steps, and how far to push for a polished lap.
- How much reward shaping to do before judging the design, since shaping is the most iterated part.
- Whether to keep a brief oval warm-up or start straight on the real circuit.

### b. Open Questions

- Which circuit for Part 1. Proposed: the Red Bull Ring, short and simple.
- The exact off-track penalty shape and the step penalty weight, set during shaping.
- The local versus cloud decision, pending the steps-per-second benchmark.
- How often to checkpoint, balancing safety against disk use.

---

## 5. Implementation Subagents

These become definitions under `.claude/agents/`. Each has a narrow scope and its own context, which saves tokens, since the main thread never carries the whole job. The test author is independent, so tests verify the spec and stay unbiased.

How they work together. The plan you build dispatches tasks in dependency order. Feature subagents read this spec and the technical design. The test subagent reads only this spec and the public interfaces, not the implementation. The reviewer gates each merge.

### Roster

- env-engineer. Scope: `src/f1rl/env/`. Builds the observation version 1 builder, the reward version 1 function, the termination logic, the start-state randomization, the trajectory recorder hookup, and the vectorized env wiring on the kinematic physics. Inputs: this spec and technical design sections seven through ten. Tools: read and edit files, run the Gymnasium env checker. Output: an env that passes the checker.

- training-engineer. Scope: `src/f1rl/train/`. Sets up PPO, the device-agnostic `train.py`, the hyperparameters in config, checkpoint and resume, Weights and Biases logging, the evaluation callback, and basic lap-time logging. Inputs: this spec and technical design sections eleven and twelve. Tools: read and edit files, run a short smoke training run, log to Weights and Biases. Output: training scripts and a smoke run where the reward trends up and a checkpoint resumes. Depends on the env.

- app-integration-engineer. Scope: the backend inference path and the frontend watch-live hooks. Wires loading a checkpoint and running the policy live over the existing WebSocket. Inputs: this spec and the Phase 1 app. Tools: read and edit files in the backend and frontend, manual check in the browser. Output: watch-live runs a trained policy. Depends on the checkpoint format and the env.

- physics-engineer. Scope: `src/f1rl/physics/`. Light role in Part 1. Confirms the kinematic model exposes the `PhysicsModel` interface cleanly, ready for the Part 2 swap. Inputs: this spec and technical design sections four and five. Tools: read and edit files, run physics unit tests. Output: a confirmed, interface-clean kinematic model.

- test-engineer, independent. Scope: `tests/`. Writes the unit and integration tests in section c from this spec and the public interfaces and schemas, not from the implementation source. Forbidden from reading implementation internals to shape tests, so the tests check the contracts. Inputs: this spec, the interface signatures, and the data schemas. Tools: read the spec and the signatures, read and write the tests folder, run the suite. Output: an unbiased test suite mapped to the acceptance criteria.

- reviewer. Scope: read-only review and the test run. Reviews each diff against this spec and the conventions, config-driven values, SI units, no rendering in the training hot path, and a stable interface, then runs the full suite and the linter and reports. Writes no feature code. Inputs: the diffs, this spec, and the conventions. Tools: read files, run tests, run the linter. Output: a pass or fail review per task, with reasons.

### Notes

- Token saving comes from the narrow scopes and the separate contexts.
- Unbiased tests come from the test-engineer working from the spec contracts, separate from the agent that wrote the code.
- Dependency order: env-engineer first, then training-engineer, then app-integration-engineer, with physics-engineer confirming the interface, test-engineer working in parallel from the spec, and reviewer gating each merge.