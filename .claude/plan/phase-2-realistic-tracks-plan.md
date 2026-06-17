# Phase 2 Implementation Plan — Realistic Circuits & Surfaces

Companion to `.claude/specs/phase-2-realistic-tracks.md` (the spec). This is the **how**:
concrete file-by-file build order, grounded in the actual Phase 1 code. Branch:
`phase-2-realistic-tracks`.

> Authoritative engineering doc remains `.claude/TECHNICAL_DESIGN.md`. Where this plan
> changes a contract in §6 (the `Track` schema), that doc is updated in the same commit —
> per CLAUDE.md, the decision and the doc move together.

---

## Confirmed decisions (override spec defaults)

Locked with the user 2026-06-16. Also stored in memory `phase-2-decisions`.

1. **Data source: FastF1 + OSM, both, this phase.** FastF1 gives the shape + the scale
   check; the Overpass/OSM pipeline is built **now** for real asphalt-edge widths
   (Shapely nearest-edge offset from the centerline). `shapely` + `requests` are part of
   the trackbuild pipeline, not deferred. Per-circuit `source` picks the cleaner of
   fastf1-shape+osm-width vs fastf1-only.
2. **Calendar: build ALL circuits, including the called-off Bahrain and Saudi Arabia** —
   not only the 22 active.
3. **Madrid (Madring): no manual tracing.** Source from OSM. If Overpass returns nothing
   usable, flag low-confidence and skip — do not hand-draw.
4. **Car glyph default = oriented 5×2 m rectangle** (real footprint). Circle+arrow is a
   toggle option. (Spec proposed circle+arrow as default; overridden.)

---

## Phase 1 baseline (verified in code, what we build on)

- `src/f1rl/track/schema.py` — `Track` has a **single** `runoff_width` band. No
  `country`, `official_length_m`, `source`, kerb/grass/gravel split. `to_api_dict`
  mirrors this.
- `src/f1rl/track/oval.py` — procedural oval; produces a `Track`; seam-safe central-diff
  tangent/normal/arc/curvature math worth reusing.
- `src/f1rl/track/loader.py` — `load_track`; non-oval ids raise `NotImplementedError`,
  `.npz` path stubbed.
- `src/f1rl/server/app.py` — `create_app` loads **one** track at startup; each `/ws/sim`
  session builds its own `SimLoop` + `CenterlineAutopilot`. `pole_time_s`/`total_laps`
  are closure constants. Routes: `GET /track/{id}`, `GET /api/meta`, `GET /recordings`,
  `GET /recordings/{id}`. No catalog, no switch, no surface-save.
- `src/f1rl/server/messages.py` — `input`/`mode`/`control`/`record` messages +
  `parse_client_message`. No `track` message.
- `src/f1rl/sim/loop.py` — `SimLoop` owns car state, physics, `LapTimer`; emits state
  frame; never renders.
- `web/src/types.ts` — mirrors old schema (`runoff_width`).
- `web/src/viewport/renderer.ts` — draws infield + asphalt ribbon + dashed red/white kerb
  from `half_width_*`; ignores runoff; car is an angular polygon. No surface bands, no
  selector, no config UI.
- `web/src/main.ts` / `state.ts` — modes `manual`/`watch`/`replay`; fetches
  `/track/oval` hardcoded. No `configure` mode.
- `pyproject.toml` — `trackbuild` extra = FastF1 only. No Shapely/requests.
- `configs/default.yaml` + `configs/track/oval.yaml` — realistic kinematic dims already
  (mass 798, ~92 m/s top speed).

**Schema divergence to reconcile:** spec §Data-Model splits bands + adds
`country`/`official_length_m`/`source`/`surface_zones`; TECHNICAL_DESIGN §6 still shows a
single `runoff_width`. §6 is updated as part of step A.

---

## Build order (dependency-first)

### A. Schema + doc (foundation — everything ripples from here)

`src/f1rl/track/schema.py`:
- Replace `runoff_width` → `kerb_width`, `grass_width`, `gravel_width` (each `(N,)`).
- Add `country: str`, `official_length_m: float`, `source: str`
  (`fastf1` | `osm` | `manual`), `surface_zones` optional per-segment grass/gravel
  override. Keep `gradient` zeros.
- Add `length_error` property = `abs(length - official_length_m)/official_length_m`;
  `low_confidence` derived (error > tolerance, or `source == "manual"`).
- Extend `to_api_dict` → emit all bands + country + length + source + low_confidence.
- npz round-trip: `save_npz(path)` / `from_npz(path)` via `np.savez` (strings as 0-d
  arrays). Must round-trip exactly.

`src/f1rl/track/oval.py`: fill new band fields (old `runoff_width` → grass; gravel = 0;
kerb ~1 m const). Keeps Phase 1 oval working.

`.claude/TECHNICAL_DESIGN.md` §6: update the `Track` block to the new fields + caveat note.

Ripples: `web/src/types.ts`, `web/src/viewport/renderer.ts` (step F).

### B. Offline build pipeline — `src/f1rl/track/build.py`

Pure offline; FastF1 imported only here. Per spec §Business-Logic:
1. **Acquire shape** — `_from_fastf1(circuit, year)`: clean fast lap, X/Y pos trace (m),
   close loop.
2. **Acquire width** — `_from_osm(circuit)`: Overpass query for asphalt ways/areas, parse
   to edge polygons; cache raw Overpass JSON to disk (rate-limit safe). Shapely
   nearest-edge offset from the FastF1 centerline → per-sample `half_width_left/right`.
   Fall back to config constant where OSM is missing/poor.
3. **Resample + smooth** — SciPy `splprep`/`splev`, uniform ~2–3 m, periodic for closed.
4. **Recenter** to centroid origin.
5. **Geometry** — tangent/normal(left)/arc-length/signed curvature. Extract the oval's
   central-diff math into a shared `track/geometry.py` so oval + build agree.
6. **Bands** — half-widths from OSM (else config default ~6 m/side, wider on straights via
   low-curvature heuristic); kerb/grass/gravel from config; gravel only where
   `surface_zones` says.
7. **Validate** — arc length vs `official_length_m` within tolerance (default 5%);
   Shapely `is_simple` self-intersection check on edge offsets; positive bounded widths.
   Flag, never crash.
8. **Save** — `data/tracks/<name>.npz` + append row to `data/tracks/_build_report.json`
   (length error, source, point count, low_confidence).

`scripts/build_all_tracks.py`: iterate `configs/track/*.yaml`, build each, write summary
table, isolate failures (one bad circuit never blocks the rest).

### C. Per-circuit configs — `configs/track/<circuit>.yaml`

One file per circuit, **all 24** incl. Bahrain + Saudi + Madrid. Each: `id`, `country`,
`fastf1_round`/`year`, `official_length_m`, `pole_time_s`, `total_laps`, width defaults,
kerb/grass/gravel widths, `source`. Mirrors `oval.yaml`. Drives both build + server meta.

### D. Loader — `src/f1rl/track/loader.py`

Replace stub: non-oval id → `Track.from_npz(data/tracks/<id>.npz)`; clear error if missing
(tells user to run the build script). Add `list_tracks()` scanning `data/tracks/*.npz` for
lightweight catalog metadata (name, country, length, turn count, low_confidence).

### E. Server — `app.py` + `messages.py`

- `GET /api/tracks` → catalog from `list_tracks()` (selector source).
- `GET /track/{id}` → load any built track, not just the startup one.
- New `TrackMessage` (`type:"track"`, `id`) in `messages.py` + parser. On `/ws/sim`,
  switching rebuilds `SimLoop` + `CenterlineAutopilot` for the session, resets,
  re-streams. `track`/`total_laps`/`pole_time_s` become per-session, not closure consts.
- `POST /track/{id}/surfaces` → accept edited band widths (+ dry/wet), bound-check, write
  back to npz via `save_npz`. Backup previous npz first (rollback, spec §f).
- Verify kinematic pacing per circuit; tune in config only.

### F. Frontend — `web/src/`

- `types.ts`: new band fields, `country`, `source`, `low_confidence`, catalog type,
  `TrackMessage`.
- `renderer.ts` **layered surfaces**: precompute band offset polylines once on `setTrack`
  → cache as `Path2D`. Redraw only the car each frame. Draw outside-in: grass/gravel
  runoff background → asphalt → red/white kerb stripes → optional racing line → glyph.
- **Car glyph** = oriented 5×2 m rectangle (real footprint, scaled by `camera.scale`);
  circle+arrow as toggle.
- **Track selector**: searchable grid/list (name, country, length, turns) + low-confidence
  badge. Sends `TrackMessage`, re-fetches `/track/{id}`, `setTrack`, refits camera.
- **Config UI panel** (`configure` mode): sliders/fields for asphalt half-width, kerb,
  grass, gravel + dry/wet; live preview; Save → `POST /surfaces`; unsaved/saved UI states.
  Per-segment brush deferred.
- `state.ts`: add `configure` mode + UI states (loading-track, editing, unsaved, saved,
  low-confidence). `main.ts`: wire selector + config panel + glyph; debug overlay shows
  track name/length/fps.

### G. Tests — `tests/`

- `test_track_build.py`: arc length within tolerance, centerline closed+smooth, edge
  offsets non-crossing on sample corners, bands positive, npz round-trip exact.
- `test_track_oval.py`: update for new band fields.
- `test_server.py`: `/api/tracks` catalog, track-switch message, surface-save round-trip +
  bounds rejection.
- Integration: each built track loads + `to_api_dict` valid.

### H. Dependencies — `pyproject.toml`

`trackbuild` extra += `shapely`, `requests` (Overpass). Core unchanged (both build-time
only).

---

## Definition of done (spec §3b)

All 24 circuits load + render with asphalt/kerbs/grass/gravel, each within length
tolerance, car glyph in real proportion, drivable, config UI refines surfaces. Build report
in the PR description.

## Open items still worth a later pass

- Overpass width quality varies per circuit — expect hand refinement in the config UI.
- Madrid depends on OSM coverage; flagged low-confidence if unmapped.
- Per-corner / per-segment surface brush deferred to a refinement pass.
