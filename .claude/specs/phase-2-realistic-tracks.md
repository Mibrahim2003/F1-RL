# Phase 2 Spec: Realistic Circuits and Surfaces

Status: draft for plan mode. Branch: `phase-2-realistic-tracks`. Depends on Phase 1, the web app, the viewport, the engine, and the trajectory format.

---

## 1. Introduction

### a. Overview, Problem Description, Summary

Phase 1 gave a working web app with a drivable car on a narrow oval. Phase 2 makes the world real. Bring in every 2026 circuit, draw each one with a faithful shape and true proportions, add the real surface layers, asphalt, red and white kerbs, grass, and sand or gravel runoff, and render the car as a directional glyph, an arrow inside a circle, scaled to a real F1 car against the track. The aim is a spatially honest environment, so the later RL phases train on a realistic world from the start.

Suggested solution, in one line: an offline track-build pipeline that pulls each circuit's shape from public data, fits a centerline in real meters, assigns surface bands, validates the scale against the official lap length, and caches the result, plus a track selector and a layered surface renderer in the app.

Stakeholders: you, as the developer and primary user. Secondary: future portfolio reviewers.

### b. Glossary or Terminology

- Centerline: the line down the middle of the asphalt.
- Racing line: the path a driver takes, the source of a telemetry trace.
- Track edge or boundary: where the asphalt ends.
- Half-width: the distance from the centerline to the asphalt edge on one side.
- Kerb or rumble strip: the red and white band at the edge of the asphalt.
- Runoff: the area past the asphalt, grass or gravel.
- Gravel trap: a sand or gravel zone that slows a car that leaves the track.
- Surface zone: a labeled region, asphalt, kerb, grass, or gravel, with its own grip.
- Arc length: distance measured along the centerline.
- Official lap length: the published circuit length, used as a scale check.
- FastF1: a Python library for F1 timing and positional data.
- OpenStreetMap and Overpass: an open map and its query API, a source of track outlines.
- GPS positional data: the car X and Y trace that draws a track map.
- Proportional scale: the car-to-track size ratio, kept equal to real F1.
- Glyph: the on-screen car symbol.
- Config UI: the in-app panel to adjust a track's widths and surfaces.

### c. Context or Background

- Why worth solving: realism is the foundation. A faithful, proportional world makes the app look its best and makes later training meaningful, since the agent learns on the real shapes and the real ratios. The Phase 1 oval is a placeholder.
- Origin: Phase 2 of the visibility-first build order.
- How it affects the goals: it serves fun, recognizable circuits are a joy to watch, resume, a faithful all-circuits sim is impressive, and learning, the data pipeline and the geometry work.
- Past efforts: Phase 1 used a hardcoded oval. This phase replaces it with real circuits.
- Roadmap fit: Phase 3 adds the agent and the dynamic physics and trains on these tracks. The spatial realism here makes that path cleaner.
- Technical strategy fit: tracks are built offline and cached, then loaded by the app and later by the training env, all in real meters, exactly as the technical design sets out.

### d. Goals or Product and Technical Requirements

Product requirements as user stories:

- As the user, I select any 2026 circuit and see it drawn faithfully, so the app feels real.
- As the user, I see the real surface layers, asphalt, red and white kerbs, grass, and sand or gravel, so each track looks like real F1.
- As the user, I see the car as an arrow inside a circle, sized to a real F1 car against the track, so the proportions feel right.
- As the user, I can drive on any circuit, so I can feel each layout.
- As the user, I can adjust a track's widths and surface bands and save them, so I can correct the approximations.

Technical requirements:

- An offline pipeline producing one Track per circuit, in real meters, cached to disk, for all 22 active 2026 circuits.
- Each Track holds a resampled and smoothed centerline, tangent, normal, arc length, curvature, half-widths, a kerb band, runoff bands for grass and gravel, and a closed flag.
- A scale check against the official lap length, within a small tolerance.
- A layered surface renderer in the canvas viewport, runoff background, asphalt, kerb stripes, and an optional racing line.
- A car glyph, an arrow in a circle, drawn at the real F1 footprint, scaled by the same meters-to-pixels factor as the track.
- A track selector in the app.
- The track configuration UI to edit widths and surface bands and save back to the cached Track.
- Realistic car dimensions and a realistic top-speed envelope set in config, so proportions and pacing look right. Full dynamics stay in Phase 3.
- Runs locally on the CPU.

### e. Non-Goals or Out of Scope

- No dynamic physics, tires, weather, or grip model. Phase 3. Phase 2 keeps the kinematic model, only with realistic dimensions and speed caps.
- No reinforcement learning. Phase 3 and later.
- No multiple cars. Phase 5 and later.
- No survey-grade or centimeter accuracy. Free public data does not provide it. The target is a faithful shape, true proportions, and a correct total length to within a few percent.
- No elevation. Optional and low priority, default flat.
- No pit lane geometry yet. Phase 7.
- No mobile.

### f. Future Goals

- Per-corner width and runoff refinement as better data appears.
- Elevation profiles if a good source is found, physics only, invisible in a top-down view.
- Pit lane geometry for Phase 7.
- DRS zones and sector markers as overlays.

### g. Assumptions

- FastF1 telemetry is available for the returning circuits from recent seasons.
- OpenStreetMap has usable outlines for most circuits.
- Madrid, the Madring, is new for 2026 and has no telemetry, so it needs a manual outline.
- The official lap length per circuit is known and used as the scale check.
- A modern desktop browser and a local Python toolchain are present.

---

## 2. Solutions

### a. Current or Existing Solution

A single hardcoded oval from Phase 1, narrow, with a constant width and no real surfaces.

- Pros: simple, and it proved the viewport and the drive loop.
- Cons: not real, narrow, no kerbs or runoff, not proportional, and only one track.

### b. Suggested or Proposed Solution

An offline track-build pipeline, cached Track files, a layered renderer, a track selector, and the config UI.

External components the solution interacts with or alters: public data sources, FastF1 over the F1 timing API and OpenStreetMap over Overpass, the local filesystem for cached tracks, the app for the selector, the renderer, and the config UI, and the existing engine and viewport from Phase 1.

Data sources, per circuit, in priority order:

1. FastF1 positional data. A fast lap's GPS X and Y trace gives a faithful, recognizable shape and the total lap distance. Best for the overall shape and the scale check. Note that this is the driven racing line, not the exact centerline or the edges, and FastF1's own circuit annotations are manually made and not highly accurate, only sufficient for visualization.
2. OpenStreetMap over Overpass. Many circuits are traced from satellite imagery as ways or areas, which can give the asphalt outline and a better sense of real width than a single driven line. Best for edges and width where the mapping is good.
3. Manual tracing from satellite or an official map, for the new Madrid circuit and any poorly mapped track.

Honest accuracy statement: the shape is recognizable and the total length is validated to within a few percent of the official figure. Track width, kerbs, and runoff are approximated and refined by hand in the config UI. Centimeter or inch accuracy is not reachable from free public data, so the realistic target is a faithful shape, true proportions, and a correct length.

Dependencies: FastF1, a request library for Overpass, NumPy, SciPy for the spline resample and smooth, Shapely for offsetting the centerline into edges and bands, and the existing app and engine.

Pros of the proposed solution: every circuit, faithful and proportional, real surface layers, one cached format reused by the app now and the training env later, and a measurable scale check.

Cons of the proposed solution: data quality varies by circuit, width and runoff are approximations, Madrid needs manual work, and detailed surface polygons need efficient rendering.

#### Data Model and Schema Changes

Extend the Track schema with explicit surface bands:

```
Track:
  name, country
  centerline (N, 2)        meters
  tangent, normal (N, 2)
  s (N,)                   arc length, meters
  curvature (N,)
  half_width_left (N,)     asphalt edge, left
  half_width_right (N,)    asphalt edge, right
  kerb_width (N,)          red and white band past the asphalt edge
  grass_width (N,)         green band past the kerb
  gravel_width (N,)        sand or gravel band, where present
  surface_zones            optional per-segment overrides, gravel vs grass
  gradient (N,)            default zeros
  closed                   bool
  official_length_m        for the scale check
  source                   fastf1 or osm or manual
```

New data: the surface band widths, the official length, and the source tag. Modified data: half-widths now come from real edge data where available, not a constant.

Validation: the computed arc length sits within tolerance of `official_length_m`, the widths are positive and bounded, and the centerline is closed and does not self-intersect after smoothing.

#### Business Logic

Pipeline steps per circuit, offline, cached to `data/tracks/<name>.npz`:

1. Acquire the shape from the best source.
2. Resample to uniform spacing, about 2 to 3 m, and smooth.
3. Recenter to the origin, in meters.
4. Compute tangent, normal, arc length, and curvature.
5. Assign half-widths from OSM where available, else a sensible default near 12 m total, and the kerb, grass, and gravel band widths.
6. Validate the arc length against the official length and flag a mismatch.
7. Save the Track and a small report, the length error, the source, and the point count.

A build-all script runs every circuit and writes a summary table.

Renderer logic: draw from outside in, the grass and gravel runoff background, then the asphalt polygon between the edges, then the red and white kerb stripes along each edge, then the optional racing line, then the car glyph on top.

Car glyph: a circle sized to the real F1 footprint, about 2 m across, the car width, with a forward arrow for heading, scaled by the same meters-to-pixels factor as the track, so the car-to-track ratio is real. An oriented rectangle at the true 5 m by 2 m footprint is an option for stricter fidelity.

Error states: a source returns nothing, so fall back to the next source or flag for manual tracing. A length mismatch beyond tolerance is flagged for review. A self-intersecting centerline after smoothing triggers less smoothing or a flag.

Failure scenarios: a FastF1 cache miss or a network issue at build time, handled because the build is offline and retried and never blocks the app. Overpass rate limits, handled by caching responses. A missing official length, handled by skipping the check and flagging.

Limitations: width and runoff are approximations, one car still, and kinematic physics still.

#### Presentation Layer

- User requirements: pick a circuit, see it real, drive it, and tweak its surfaces.
- UI changes: a track selector, a searchable list or a grid of the 22 circuits with name, country, length, and turn count, the layered surface render in the viewport, and the config UI panel to adjust band widths and dry or wet, with a save action.
- Wireframes: the selector as a panel or modal. The viewport keeps its Phase 1 layout but now draws real circuits and surfaces. The config panel has numeric fields or sliders for asphalt width, kerb, grass, and gravel, with a per-segment brush as a later refinement.
- Web concerns: build the static track polygons once into cached canvas paths, so a detailed circuit holds 60 fps, and redraw only the car each frame. Use device-pixel-ratio scaling for crisp kerb stripes.
- Mobile concerns: out of scope.
- UI states: loading a track, track loaded, editing surfaces, unsaved changes, save complete, and a build error or low-confidence track flagged.
- Error handling: a circuit flagged as low confidence, from a length mismatch or a manual source, shows a small badge, so you know which tracks need a hand.

#### Other questions to answer

- How will it scale: the pipeline runs once per circuit offline, the app loads cached files, and rendering caches static paths, so all 22 circuits load fast.
- Limitations: as above.
- Recovery on failure: rebuild a single circuit without touching the others, since the cached format is stable.
- Future requirements: the same Track feeds the training env unchanged, and per-corner width and elevation slot into the existing fields.

### c. Test Plan

- Unit tests: the arc length sits within tolerance of the official length for each built circuit, the centerline is closed and smooth, offsets produce non-crossing edges on representative corners, and band widths are positive.
- Integration tests: every circuit loads into the app and renders without error, the config UI saves and reloads band changes, and driving works on a sample of circuits.
- QA: a visual side-by-side against real circuit maps for shape, kerbs, grass, and gravel that look like real F1, a car glyph in believable proportion to the track, and a selector that matches the real calendar.

### d. Monitoring and Alerting Plan

Local tool. The build script logs per-circuit length error, source, and point count, and writes a summary build report. The app shows a low-confidence badge on flagged circuits. An on-screen debug overlay shows the loaded track name, length, and render fps. No external alerting.

### e. Release, Roll-out, and Deployment Plan

- Branch `phase-2-realistic-tracks`. Build the pipeline first, then circuits in batches, a few clean ones first, then the rest, then Madrid by hand.
- Merge to main when all 22 circuits load, render, and pass the scale check, and the look is approved.
- The PR description carries the build report.

### f. Rollback Plan

- Liabilities: a bad track build or a renderer change could break the viewport for all circuits.
- Reduce liabilities: keep main working, develop on the branch, tag the working Phase 2 commit, and version the cached track files in data, so a bad rebuild is reverted by restoring the previous cache.
- Prevent spread: the renderer and the pipeline are separate, so a pipeline issue does not touch the drive loop. Revert the merge or restore the tagged commit if needed. Delete the branch after a clean merge and a passing visual test.

### g. Alternate Solutions or Designs

- Alternate 1, FastF1 only, no OSM. Simpler, one source, gives shape and length, but width and edges are guessed everywhere. Inferior on width realism. Kept as the fallback when OSM is poor.
- Alternate 2, OSM only. Good edges and width where mapped, but messy or missing for some circuits and no racing-line reference. Used as the primary for edges, paired with FastF1.
- Alternate 3, community SVG or GeoJSON track datasets. Quick to import, but licensing and accuracy vary, and the scale must still be set from the official length. A convenience source, cross-checked against the length.
- Alternate 4, fully manual tracing of all circuits. Highest control, far too slow for 22. Reserved for Madrid and flagged tracks.
- Migration: all sources feed the same Track schema, so switching a circuit's source is a per-circuit rebuild.

---

## 3. Success Evaluation

### a. Impact

- Security: build-time network calls to public data only, the app stays local, and all parsed geometry is validated and bounded.
- Performance: heavier static geometry, handled by caching canvas paths, with the car the only per-frame redraw, target 60 fps.
- Cost: zero, public data and free tools.
- Impact on other components: sets the Track format, the surface model, and the proportional scale that the Phase 3 env and renderer reuse unchanged.

### b. Metrics

- Capture: per-circuit length error versus official, point count, render fps per circuit, the count of circuits flagged low confidence, and a visual-match pass or fail per circuit.
- Tools: the build report, the debug overlay, and a manual visual checklist against real maps.
- Definition of done: all 22 active 2026 circuits load and render with asphalt, kerbs, grass, and gravel, each within the length tolerance, the car glyph in real proportion, drivable, with the config UI able to refine surfaces.

---

## 4. Deliberation

### a. Discussion

- How tight the length tolerance should be. A few percent is realistic from telemetry.
- Whether to lead with OSM edges or the FastF1 racing line per circuit. Likely per-circuit, whichever is cleaner.
- The default car glyph, a circle with an arrow, your request, versus an oriented 5 by 2 rectangle for stricter fidelity. Proposed: the circle with an arrow, with the rectangle as an option.
- How much per-segment surface detail to author by hand versus leaving uniform bands.

### b. Open Questions

- Confirm the realism target, a faithful shape, true proportions, and a length within a few percent, since centimeter accuracy is not available from free data. Is that acceptable?
- A default total track width where no edge data exists. Proposed near 12 m, wider on long straights.
- The exact list of the 22 circuits to include, and whether to also build the two called-off circuits, Bahrain and Saudi Arabia, for completeness.
- Licensing comfort for OSM data, which is open with attribution, and any community datasets.
- Realistic top speed and acceleration values for the kinematic car in this phase, so the pacing looks right before the dynamic model arrives.