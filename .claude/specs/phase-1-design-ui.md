# Phase 1 Spec: Application Shell and Race Viewer

Status: draft for plan mode. Branch: `phase-1-design-ui`.

---

## 1. Introduction

### a. Overview, Problem Description, Summary

You need a visible application before anything else, because you want to see every change you make as you make it. Phase 1 builds the application shell and the race viewer: a polished F1-broadcast-style interface with a top-down viewport that renders a track and a car, lets you drive a car by keyboard to feel the physics, and replays a recorded run. No reinforcement learning yet. This screen becomes the window for every later phase.

The look was prototyped in Claude Design and approved. This phase turns that prototype into a working app with a live viewport and a real physics-driven car.

Suggested solution, in one line: a web frontend, the approved design made real, with an HTML5 Canvas viewport, driven by the Python simulation engine. A local server streams live state for manual drive. Replay reads a recorded trajectory file.

Stakeholders: you, as the sole developer and primary user. Secondary: future portfolio reviewers and recruiters who will watch the result.

### b. Glossary or Terminology

- Viewport: the canvas area that draws the track and cars from directly above.
- Top-down: a flat overhead view, no 3D.
- Kinematic bicycle model: a simple car model where the turn radius comes from wheelbase, steering, and speed, with no tire slip.
- Trajectory: an ordered list of car states over time.
- Recorded trajectory: a trajectory saved to a file for replay.
- Telemetry: the live readouts, speed, lap time, delta, lap counter.
- Timing tower: the vertical list of positions on the left, the signature F1 element.
- HUD: the on-screen readouts overlaid on the viewport.
- Pole lap: the fastest reference lap time for a circuit.
- Delta: the time gap to the pole lap, positive when slower, negative when faster.
- Tyre compound: soft, medium, hard, intermediate, wet, shown as a colored dot.
- WebSocket: a two-way live connection between the browser and the local server.
- Canvas: the browser drawing surface used for the viewport.
- Tick or step: one simulation update at the fixed control rate.
- Mode: manual drive, watch live, or replay.
- Design tokens: the named colors, type sizes, and spacing values from the approved design.

### c. Context or Background

- Why worth solving: momentum and visibility. The project stalled before with no visible artifact. Seeing progress sustains the work, so the visible app comes first on purpose.
- Origin: Phase 1 of the agreed, visibility-first build order in the technical design.
- How it affects you and the goals: the app serves all three goals. A polished visible app is the portfolio centerpiece, the daily feedback loop, and the surface that teaches you the system.
- Past efforts: earlier attempts had no working front end, so nothing felt real and the work faded. This phase fixes the order.
- Roadmap fit: Phase 1 is the foundation. Phases 2 through 7 add tracks, agents, racing, and pit strategy on top of this shell.
- Technical strategy fit: physics and the simulation live in Python as the single source of truth. The UI renders and sends input. The recorded-trajectory format becomes the shared interchange between the simulation, the replay viewer, and the later cloud clip renderer.

### d. Goals or Product and Technical Requirements

Product requirements as user stories:

- As the user, I see a polished F1-style app shell, so the project feels real from day one.
- As the user, I see a top-down view of a track and a car, so I can watch the simulation.
- As the user, I can drive a car with the keyboard, so I can feel the physics by hand.
- As the user, I can pan and zoom the viewport, so I can see the whole track or follow the car.
- As the user, I can record a run and replay it, so I can watch a past session.
- As the user, I can switch modes from the UI, manual drive, watch live, and replay.
- As the user, I see live telemetry, speed, lap time, delta, and lap counter, so the screen feels alive.

Technical requirements:

- A web frontend implementing the approved design tokens and components.
- An HTML5 Canvas viewport rendering the track and car at real proportions, with camera pan and zoom.
- A Python simulation engine with the kinematic bicycle model and a hardcoded oval track, in SI meters.
- A local server, FastAPI on uvicorn, exposing a WebSocket that streams car state at the control rate and accepts keyboard input for manual drive.
- A recorded-trajectory JSON format and a replay player in the frontend.
- A fixed control step at 20 Hz with five physics substeps, per the technical design.
- Deterministic seeding from one utility.
- Clean module boundaries, so the physics and track are reused unchanged by later phases.
- Runs locally on the laptop CPU with no GPU.

### e. Non-Goals or Out of Scope

- No reinforcement learning, no training, no policies. Phase 3 and later.
- No real circuits and no FastF1 pipeline. Phase 2. Phase 1 uses a hardcoded oval only.
- No dynamic physics, tires, weather, or surface zones beyond a flat oval. Phase 2 and 3.
- No multiple cars. Phase 5 and later.
- No cloud, no deployment, no eval-clip rendering on the cloud.
- No pit stops and no racing-rule HUD elements.
- No mobile target. Desktop browser only.

### f. Future Goals

- Real circuits and the track configuration UI, Phase 2.
- Dynamic physics with the grip pipeline, tires, and weather, Phase 3.
- RL training and live inference viewing over the same WebSocket, Phase 3.
- A cloud eval-clip renderer that reuses the trajectory format, Phase 3 and later.
- A full-grid timing tower with 22 cars, Phase 5.

### g. Assumptions

- The approved Claude Design output is available as usable HTML and CSS, or a React project.
- A modern desktop browser, Chromium or Firefox, is the target.
- Python 3.10 or newer and a Node toolchain are available locally.
- The laptop CPU handles one car at the control rate with ease.
- Free fonts are used, Saira or Chakra Petch for display, Inter for body, and a tabular monospace for numbers.

---

## 2. Solutions

### a. Current or Existing Solution

The current state is a static UI prototype from Claude Design. Visual only. No live viewport, no physics, no interaction, static placeholder data.

- Pros: establishes the look and the component system, and is approved.
- Cons: not interactive, draws no track or car, has no physics, no replay, and no live data.

### b. Suggested or Proposed Solution

A web frontend plus a Python engine, connected by a local WebSocket for live modes, with file-based recorded-trajectory replay. A Canvas viewport draws the top-down scene.

External components the solution interacts with or alters:

- The browser, for rendering and keyboard input.
- The local filesystem, for the track definition and saved trajectory files.
- The Python process, the simulation engine and the FastAPI server.

Dependencies of the proposed solution:

- Frontend: Vite, TypeScript, the Canvas 2D API, the chosen fonts, and the approved design tokens. React only if the prototype is already React.
- Backend: Python, FastAPI, uvicorn, NumPy, and the project physics and track modules.

Pros of the proposed solution:

- Uses the approved UI directly, polished and shareable, which serves the resume goal.
- Physics stays single-source in Python, so manual drive feels the real model.
- The trajectory format unifies replay now and cloud clips later.
- The viewport scales to a full grid by streaming an array of car states.

Cons of the proposed solution:

- More moving parts than one process, a frontend and a backend, a socket protocol, and serialization.
- A running local server alongside the browser.
- The smoothness of the state stream needs care.

#### Data Model and Schema

CarState, the engine state from the technical design. Phase 1 uses the kinematic subset, x, y, yaw, speed. The full struct is kept for later phases.

Track, Phase 1 minimal. A procedurally generated oval as a centerline array in meters with constant half-widths, matching the Track schema fields already in the technical design, with closed set to true.

WebSocket messages:

```json
client -> server:
  { "type": "input", "steer": -1.0, "throttle": 0.0, "brake": 0.0, "reset": false }
  { "type": "mode",  "mode": "manual" }

server -> client:
  { "type": "state", "t": 12.35,
    "car": { "x": 0.0, "y": 0.0, "yaw": 0.0, "speed": 83.2 },
    "telemetry": { "speed_kmh": 299, "lap_time": 91.42, "delta_to_pole": -0.21,
                   "lap": 2, "best_lap": 90.88 } }
```

Recorded trajectory file:

```json
{
  "meta": { "track_id": "oval", "dt": 0.05, "seed": 42, "created": "ISO-8601" },
  "frames": [
    { "t": 0.0,
      "car": { "x": 0.0, "y": 0.0, "yaw": 0.0, "speed": 0.0 },
      "telemetry": { "speed_kmh": 0, "lap_time": 0.0, "lap": 0 } }
  ]
}
```

Data validation: schema-validate a trajectory file on load, bound-check every input message, and clamp actions to their ranges.

#### Business Logic

The simulation loop, at 20 Hz:

```
loop at 20 Hz:
    input = latest_input or zero
    for substep in range(5):
        state = physics.step(state, input.steer, input.longitudinal, grip=1.0, dt=0.01)
    telemetry = update_timing(state, track)
    if recording:
        recorder.append(t, state, telemetry)
    socket.send(state_frame(state, telemetry))
```

- Lap timing: detect the start and finish crossing on the oval, accumulate lap time, track the best lap, and compute the delta against a stored pole reference. Phase 1 uses a placeholder pole time for the oval.
- Modes: manual drive takes inputs from the browser. Watch live runs a simple centerline-following script, only to prove the live path before any policy exists. Replay plays a trajectory file in the frontend, and does not need the server.
- API: a WebSocket at `/ws/sim` for live, an HTTP GET `/track/oval` for the geometry, and either an HTTP list of saved trajectories or a direct client-side load.
- Error states: socket disconnect shows a reconnect state and the backend keeps the last state. Invalid input is clamped. A missing track is an error. A corrupt trajectory raises a validation error shown in the UI.
- Failure scenarios: the server is not running, so the frontend shows engine offline while replay still works. Dropped frames are smoothed by interpolation. The server is the time authority and stamps every frame.
- Limitations: a single car, an oval only, kinematic physics only, local only.

#### Presentation Layer

- User requirements: the approved prototype screen, now live.
- UX and UI: the four regions wired to live data, the top bar, the viewport, the timing tower, and the telemetry and control bar. A mode switch. Playback controls active in replay. Pan and zoom on the viewport, with a camera-follow toggle.
- Wireframes: the approved Claude Design screen is the reference. Each region behaves as follows. The top bar shows the wordmark, the circuit name, and a session clock. The viewport draws the oval and the car, with an optional debug overlay for the centerline and the start and finish line. The timing tower shows one row in Phase 1, the single car, with placeholder fields for gap and tyre. The bottom bar shows speed, current lap time, delta to pole, lap counter, and playback controls.
- Link to the designer work: the approved Claude Design file. Add the link here.
- Web concerns: the canvas renders at 60 frames per second, decoupled from the 20 Hz simulation, and interpolates between the last two received states. The canvas uses device-pixel-ratio scaling for crisp thin lines. The layout responds to the window size. Keyboard focus is handled so driving keys do not scroll the page.
- Mobile concerns: out of scope. Desktop only.
- UI states: loading, engine online, engine offline, manual driving, watching live, replay playing, replay paused, replay scrubbing, no trajectory selected, and error.
- Error handling: clear non-blocking banners. The viewport never crashes on a bad message.

#### Other questions to answer

- How will it scale: trivial for one car. The same socket carries an array of car states for a full grid, and the canvas draws 22 cars with ease.
- Limitations: single car, oval, kinematic, local, as listed above.
- Recovery on failure: a server restart begins a fresh simulation. Replay is independent of the server. The frontend auto-reconnects the socket.
- Future requirements: the same socket later carries a trained policy driving live. The same trajectory format feeds the cloud clip renderer. The same viewport renders real tracks and a full grid.

### c. Test Plan

Each user story maps to a check.

- Unit tests, Python: the kinematic step produces a correct straight line and a correct constant-radius turn, the simulation is deterministic under a fixed seed, lap detection fires once per lap on the oval, a record then replay round-trip yields identical frames, and inputs clamp to range.
- Integration tests: a socket round-trip returns a consistent state for a given input, the track endpoint returns valid geometry, and a recorded session replays frame for frame.
- Frontend tests: the canvas draws without error, mode switching works, and replay scrubbing lands on the right frame.
- QA: manual drive feels responsive, pan and zoom are smooth, the screen matches the approved design, and numbers use tabular figures that never jitter.

### d. Monitoring and Alerting Plan

Adapted for a local development tool.

- Logging: structured Python logs for the simulation loop and the socket. A small console panel or overlay in the frontend for live diagnostics.
- Monitoring and metrics: simulation step rate, which should hold 20 Hz, render frames per second, target 60, socket round-trip latency, and dropped-frame count.
- Observability: an on-screen debug overlay showing fps, step rate, and car state, toggled with a key.
- Alerting: none external. This is a local tool, so the on-screen overlay and console warnings are the alerts.

### e. Release, Roll-out, and Deployment Plan

- Deployment architecture: runs locally. The backend on uvicorn and the frontend on the Vite dev server or a static build, both on localhost.
- Environments: local development only in Phase 1.
- Phased roll-out: built on the branch `phase-1-design-ui`. No feature flags at this scale. Merge to main when the Phase 1 artifact works and passes the manual test.
- Communicating changes: a short README entry and the pull request description. For a solo project, the PR description is the release note.

### f. Rollback Plan

- Liabilities: a broken merge could break the only working state of the app.
- Reduce liabilities: develop on `phase-1-design-ui`, keep main always working, open a PR, test, then merge. Tag the working Phase 1 commit.
- Prevent affecting other components: Phase 1 has no dependents yet. If a later issue appears, revert the merge commit or check out the tagged Phase 1 commit. Delete the feature branch only after a clean merge and a passing manual test.

### g. Alternate Solutions or Designs

Alternate 1, the Pygame desktop app, the original choice in the technical design. One Python process renders and runs the simulation, with no web and no server.

- Pros: the simplest stack, one language, interactive modes that drive the real simulation directly, and no socket.
- Cons: hard to reproduce the polished prototype look, all widgets drawn by hand, and less shareable than a web app.
- Why not chosen: the approved web prototype and the visibility and portfolio priority favor the web UI. Pygame would discard the prototype and the broadcast-grade polish.
- Migration path if web falls through: the physics, the track, and the trajectory format are language-agnostic and reused unchanged. Rebuild only the rendering and input layer in Pygame.

Alternate 2, a browser-only app with the kinematic physics written in TypeScript and no Python backend in Phase 1.

- Pros: no server, and the simplest way to ship Phase 1.
- Cons: physics would live in two languages once Python training arrives, and the manual-drive feel would not match the real model.
- Why not chosen: it breaks single-source-of-truth for physics.
- Migration: keep as a fallback only if the socket proves troublesome, and isolate the TypeScript physics behind an interface so the Python engine can replace it.

Alternate 3, a desktop shell, Electron or Tauri, wrapping the web UI with a Python sidecar.

- Pros: a native desktop feel and packaging.
- Cons: a heavier toolchain than Phase 1 needs.
- Why not chosen: premature. The plain web app suffices and can be wrapped later.

---

## 3. Success Evaluation

### a. Impact

- Security: local only. The WebSocket binds to localhost, needs no auth, and the port stays private. Inputs and trajectory files are validated.
- Performance: one car is a trivial load. Target 60 fps render, a steady 20 Hz simulation, and sub-frame socket latency on localhost.
- Cost: zero. All local and free tools.
- Impact on other components: this phase sets the engine, the trajectory format, and the rendering and input contract that every later phase reuses.

### b. Metrics

- Capture: simulation step rate, render fps, socket round-trip latency, dropped-frame count, input-to-render latency, and replay accuracy.
- Tools: the on-screen debug overlay, Python logs, and the browser performance tools.
- Definition of done: drive a car around the oval by keyboard in the polished app, see live telemetry, record a run, and replay it, with the look matching the approved prototype.

---

## 4. Deliberation

### a. Discussion

- The main item to settle: a web frontend with a Python backend, versus the Pygame app named in the current technical design. This spec proposes web, for polish, shareability, and reuse of the approved prototype, at the cost of more moving parts. Confirm this, because it changes the rendering and input sections and means updating the technical design doc to match.
- Secondary: whether watch live in Phase 1 uses a simple centerline-following script, which is recommended to prove the live path, or waits until a policy exists.
- Frontend framework: React or vanilla TypeScript, set by what Claude Design produced.

### b. Open Questions

- Web or Pygame for the application? Proposed: web. This is the one decision to confirm before plan mode.
- Did Claude Design output plain HTML and CSS, or a React project? This sets the frontend toolchain.
- The keyboard mapping for manual drive, arrows or WASD, and the throttle and brake scheme.
- A placeholder pole time for the oval, so the delta has a reference in Phase 1.
- Where recorded trajectories live, and whether the frontend loads them directly or through the backend.