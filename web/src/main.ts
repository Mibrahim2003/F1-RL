/**
 * Bootstrap: stage fit-to-window scaling, fetch track + meta, wire the UI
 * state machine, mode/control buttons, keyboard, socket, replay, and the
 * 60 fps render loop.
 */

import "./tokens.css";
import "./styles.css";

import { Store } from "./state.ts";
import { SimSocket } from "./net/socket.ts";
import { Keyboard } from "./input/keyboard.ts";
import { Renderer } from "./viewport/renderer.ts";
import type { GlyphStyle } from "./viewport/renderer.ts";
import { Hud } from "./hud/telemetry.ts";
import { ReplayPlayer, fetchRecording, listRecordings } from "./replay/player.ts";
import { TrackSelector } from "./ui/selector.ts";
import { ConfigPanel } from "./ui/config_panel.ts";
import { PolicyPicker } from "./ui/policy_picker.ts";
import type { Meta, Mode, SimMode, StateFrame, SurfaceEdit, Track } from "./types.ts";

const $ = (id: string): HTMLElement => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el;
};

// ---------- stage fit-to-window scaling ----------
function fitStage(): void {
  const stage = $("stage");
  const s = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
  stage.style.transform = `scale(${s})`;
}

// ---------- module instances ----------
const store = new Store();
const canvas = $("track-canvas") as HTMLCanvasElement;
const renderer = new Renderer(canvas);
const hud = new Hud();
const replay = new ReplayPlayer(renderer, hud);

// Canonical loaded track (configure-mode previews mutate clones, not this).
let currentTrack: Track | null = null;

const selector = new TrackSelector($("stage"), (id) => void switchTrack(id));
const configPanel = new ConfigPanel($("stage"), {
  onPreview: (edit) => previewSurfaces(edit),
  onSave: (edit) => saveSurfaces(edit),
  onGlyph: (style) => renderer.setGlyph(style),
});

// Watch-live: pick the centerline autopilot or a trained checkpoint (mounted in the viewport).
const policyPicker = new PolicyPicker($("viewport"), {
  onAutopilot: () => socket.sendPolicy("autopilot"),
  onCheckpoint: (id) => socket.sendPolicy("checkpoint", id),
});

const keyboard = new Keyboard(
  (input) => socket.sendInput(input.steer, input.throttle, input.brake, false),
  () => socket.sendInput(0, 0, 0, true),
);

const socket = new SimSocket({
  onState: (frame) => onStateFrame(frame),
  onEvent: (ev) => {
    if (ev.event === "recording_saved") {
      // refresh recordings list silently; replay picks newest on demand
      void listRecordings().catch(() => {});
    } else if (ev.event === "policy_error") {
      // non-fatal: server fell back to the autopilot, viewport keeps streaming
      policyPicker.reset();
      policyPicker.setStatus(
        `Checkpoint '${ev.id ?? ""}' failed to load — using autopilot.`,
        "error",
      );
    } else if (ev.event === "policy_changed") {
      if (ev.source === "checkpoint") {
        policyPicker.setStatus(
          `Policy: ${ev.id ?? ""} (${ev.circuit_id ?? "?"})`,
          "ok",
        );
      } else {
        policyPicker.setStatus("Autopilot (centerline)", "");
      }
    } else if (ev.event === "track_changed" && ev.pole_time_s !== undefined) {
      // server confirms the switch and sends the new circuit's pace meta
      hud.setMeta({
        track_id: ev.id ?? store.get().trackId,
        control_hz: ev.control_hz ?? 20,
        pole_time_s: ev.pole_time_s,
        total_laps: ev.total_laps ?? 1,
        pole_str: ev.pole_str ?? "",
      });
      if (currentTrack) updateCircuitChip(currentTrack);
    }
  },
  onOpen: () => store.set({ engineConnected: true }),
  onClose: () => store.set({ engineConnected: false }),
});

// ---------- state frame handling ----------
function onStateFrame(frame: StateFrame): void {
  // live frames only matter in manual/watch (not replay/configure)
  const mode = store.get().mode;
  if (mode === "replay" || mode === "configure") return;
  renderer.pushPose(frame.t, frame.car);
  hud.update(frame);
}

// ---------- track switching ----------
function updateCircuitChip(track: Track): void {
  $("circuit-name").textContent = track.name.toUpperCase();
  $("circuit-locale").textContent = (track.country || "—").toUpperCase();
  const chip = document.querySelector<HTMLElement>(".circuit-chip");
  chip?.classList.toggle("low-confidence", track.low_confidence);
}

async function switchTrack(id: string): Promise<void> {
  if (id === store.get().trackId && currentTrack) return;
  store.set({ loadingTrack: true });
  socket.sendTrack(id);
  try {
    const track = await fetch(`/track/${id}`).then((r) => {
      if (!r.ok) throw new Error(`/track/${id} ${r.status}`);
      return r.json() as Promise<Track>;
    });
    currentTrack = track;
    renderer.resize();
    renderer.setTrack(track, true);
    renderer.clearPoses();
    hud.reset();
    updateCircuitChip(track);
    // The server drops any active policy back to the autopilot on a circuit switch.
    policyPicker.reset();
    store.set({ trackId: id, lowConfidence: track.low_confidence, loadingTrack: false });
    if (store.get().mode === "configure") openConfigPanel();
  } catch {
    store.set({ loadingTrack: false, errorMessage: `Could not load track '${id}'.` });
  }
}

// ---------- configure-mode surface editing ----------
function applySurfaceEdit(base: Track, edit: SurfaceEdit): Track {
  const n = base.centerline.length;
  const fill = (v: number) => new Array(n).fill(v);
  return {
    ...base,
    half_width_left:
      edit.half_width_left != null ? fill(edit.half_width_left) : base.half_width_left,
    half_width_right:
      edit.half_width_right != null ? fill(edit.half_width_right) : base.half_width_right,
    kerb_width: edit.kerb_width != null ? fill(edit.kerb_width) : base.kerb_width,
    grass_width: edit.grass_width != null ? fill(edit.grass_width) : base.grass_width,
    gravel_width: edit.gravel_width != null ? fill(edit.gravel_width) : base.gravel_width,
  };
}

function previewSurfaces(edit: SurfaceEdit): void {
  if (!currentTrack) return;
  renderer.setTrack(applySurfaceEdit(currentTrack, edit), false);
  store.set({ edit: "unsaved" });
}

async function saveSurfaces(edit: SurfaceEdit): Promise<boolean> {
  const id = store.get().trackId;
  try {
    const r = await fetch(`/track/${id}/surfaces`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(edit),
    });
    if (!r.ok) return false;
    const fresh = await fetch(`/track/${id}`).then((res) => res.json() as Promise<Track>);
    currentTrack = fresh;
    renderer.setTrack(fresh, false);
    store.set({ edit: "saved", lowConfidence: fresh.low_confidence });
    return true;
  } catch {
    return false;
  }
}

function openConfigPanel(): void {
  const t = currentTrack;
  if (!t) return;
  configPanel.show({
    half_width: t.half_width_left[0] ?? 6,
    kerb_width: t.kerb_width[0] ?? 1,
    grass_width: t.grass_width[0] ?? 8,
    gravel_width: t.gravel_width.length ? Math.max(0, ...t.gravel_width) : 0,
    condition: "dry",
    glyph: renderer.getGlyph(),
    trackName: t.name,
    lowConfidence: t.low_confidence,
  });
}

// ---------- UI reactions to store ----------
store.subscribe((s) => {
  // live pill
  const liveLabel = $("live-label");
  const liveDot = $("live-dot");
  const livePill = $("live-pill");
  let color = "var(--red)";
  let label = "LIVE";
  if (s.mode === "manual") {
    color = "var(--slower)";
    label = "MANUAL";
  } else if (s.mode === "replay") {
    color = "var(--fastest)";
    label = "REPLAY";
  } else if (s.mode === "configure") {
    color = "var(--pb)";
    label = "CONFIG";
  } else if (!s.engineConnected) {
    color = "var(--text-2)";
    label = "OFFLINE";
  }
  liveLabel.style.color = color;
  liveLabel.textContent = label;
  liveDot.style.background = color;
  livePill.style.borderColor = color;

  // mode badge (shown for non-watch modes, matching prototype)
  const badge = $("mode-badge");
  const badgeText = $("mode-badge-text");
  if (s.mode === "watch") {
    badge.classList.add("hidden");
  } else {
    badge.classList.remove("hidden");
    badgeText.textContent = label;
    badgeText.style.color = color;
    badge.style.borderColor = color;
  }

  // overlays
  $("manual-overlay").classList.toggle("hidden", s.mode !== "manual");
  $("replay-scrub").classList.toggle("hidden", s.mode !== "replay");

  // watch-live policy picker is visible only while watching a live run
  if (s.mode === "watch") policyPicker.show();
  else policyPicker.hide();

  // engine-offline / error banner
  const banner = $("banner");
  const showOffline = s.ui === "engine-offline" && s.mode !== "replay";
  const showError = s.ui === "error";
  const showNoTraj = s.mode === "replay" && !replay.hasRecording() && s.engineConnected === false && noTrajectory;
  if (showError) {
    $("banner-title").textContent = "ERROR";
    $("banner-sub").textContent = s.errorMessage ?? "Something went wrong.";
    banner.classList.remove("hidden");
  } else if (showNoTraj) {
    $("banner-title").textContent = "NO TRAJECTORY";
    $("banner-sub").textContent = "No recordings are available to replay yet. Record a run in Manual or Watch mode first.";
    banner.classList.remove("hidden");
  } else if (showOffline) {
    $("banner-title").textContent = "ENGINE OFFLINE";
    $("banner-sub").textContent =
      "The simulation engine is not reachable. Start the backend, or use Replay to review a recorded run.";
    banner.classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
  }

  // run button glyph (pause when running, play when paused)
  setRunGlyph(s.running);

  // segmented active states
  syncSeg("mode-seg", "mode", s.mode);
  syncSeg("speed-seg", "speed", String(s.speed));

  // keyboard only in manual
  keyboard.setEnabled(s.mode === "manual");
});

let noTrajectory = false;

function setRunGlyph(running: boolean): void {
  const glyph = $("run-glyph");
  glyph.innerHTML = running
    ? '<rect x="3" y="2.5" width="3" height="9" fill="#E10600"/><rect x="8" y="2.5" width="3" height="9" fill="#E10600"/>'
    : '<path d="M3.5 2 L11.5 7 L3.5 12 Z" fill="#E10600"/>';
}

function syncSeg(segId: string, attr: string, value: string): void {
  const seg = $(segId);
  seg.querySelectorAll<HTMLElement>(".item").forEach((item) => {
    item.classList.toggle("active", item.dataset[attr] === value);
  });
}

// ---------- control wiring ----------
function setMode(mode: Mode): void {
  const prev = store.get().mode;
  // Leaving configure: hide the panel and discard any unsaved live preview.
  if (prev === "configure" && mode !== "configure") {
    configPanel.hide();
    if (currentTrack) renderer.setTrack(currentTrack, false);
    store.set({ edit: "clean" });
  }
  store.set({ mode });
  if (mode === "replay") {
    void enterReplay();
    return;
  }
  if (mode === "configure") {
    socket.sendControl("pause");
    store.set({ running: false });
    openConfigPanel();
    return;
  }
  socket.sendMode(mode as SimMode);
  renderer.clearPoses();
  hud.reset();
  store.set({ running: true });
}

async function enterReplay(): Promise<void> {
  store.set({ running: false });
  try {
    const list = await listRecordings();
    if (list.length === 0) {
      noTrajectory = true;
      store.set({}); // trigger re-render of banner
      return;
    }
    noTrajectory = false;
    const newest = list[list.length - 1];
    const rec = await fetchRecording(newest.id);
    replay.load(rec);
    store.set({}); // refresh banner state
  } catch {
    // backend offline: replay simply has nothing to show
    noTrajectory = true;
    store.set({});
  }
}

$("mode-seg").addEventListener("click", (e) => {
  const item = (e.target as HTMLElement).closest<HTMLElement>(".item");
  if (!item?.dataset.mode) return;
  setMode(item.dataset.mode as Mode);
});

$("speed-seg").addEventListener("click", (e) => {
  const item = (e.target as HTMLElement).closest<HTMLElement>(".item");
  if (!item?.dataset.speed) return;
  const speed = Number(item.dataset.speed) as 1 | 2 | 4;
  store.set({ speed });
  if (store.get().mode === "replay") replay.setSpeed(speed);
  else socket.sendControl("play", speed);
});

$("btn-run").addEventListener("click", () => {
  const s = store.get();
  const next = !s.running;
  store.set({ running: next });
  if (s.mode === "replay") {
    if (next) replay.play();
    else replay.pause();
  } else {
    socket.sendControl(next ? "play" : "pause");
  }
});

$("btn-restart").addEventListener("click", () => {
  const s = store.get();
  if (s.mode === "replay") {
    replay.restart();
    store.set({ running: false });
  } else {
    socket.sendControl("restart");
    hud.reset();
  }
});

// scrub bar
const scrub = $("replay-scrub");
let scrubbing = false;
function scrubToEvent(e: MouseEvent): void {
  const rect = scrub.getBoundingClientRect();
  const f = (e.clientX - rect.left) / rect.width;
  replay.seekFraction(f);
}
scrub.addEventListener("mousedown", (e) => {
  if (store.get().mode !== "replay") return;
  scrubbing = true;
  replay.pause();
  store.set({ running: false });
  scrubToEvent(e);
});
window.addEventListener("mousemove", (e) => {
  if (scrubbing) scrubToEvent(e);
});
window.addEventListener("mouseup", () => {
  scrubbing = false;
});

// ---------- viewport interaction: pan / zoom / follow / debug ----------
let dragging = false;
let lastX = 0;
let lastY = 0;
canvas.addEventListener("mousedown", (e) => {
  dragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
});
window.addEventListener("mouseup", () => {
  dragging = false;
});
window.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  // convert CSS-pixel mouse delta to canvas-internal scale
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width / (window.devicePixelRatio || 1);
  const sy = canvas.height / rect.height / (window.devicePixelRatio || 1);
  renderer.camera.panByPixels((e.clientX - lastX) * sx, (e.clientY - lastY) * sy);
  lastX = e.clientX;
  lastY = e.clientY;
});
canvas.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const anchor = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    renderer.camera.zoomAt(anchor, e.deltaY < 0 ? 1.1 : 1 / 1.1);
  },
  { passive: false },
);
window.addEventListener("keydown", (e) => {
  if (e.code === "Backquote") {
    renderer.toggleDebug();
    $("debug").classList.toggle("on", renderer.isDebugOn());
  } else if (e.code === "KeyF") {
    // toggle camera-follow (avoid hijacking WASD 'F' is unused)
    renderer.camera.setFollow(!renderer.camera.follow);
  } else if (e.code === "KeyC") {
    renderer.camera.resetView();
  } else if (e.code === "KeyG") {
    const next: GlyphStyle = renderer.getGlyph() === "rect" ? "arrow" : "rect";
    renderer.setGlyph(next);
  }
});

// circuit chip → open the track selector
document.querySelector<HTMLElement>(".circuit-chip")?.addEventListener("click", () => {
  void selector.refresh().then((ok) => {
    if (ok) selector.open(store.get().trackId);
  });
});

// ---------- render loop ----------
let lastFrame = performance.now();
function loop(now: number): void {
  const dt = now - lastFrame;
  lastFrame = now;

  if (store.get().mode === "replay") {
    const f = replay.tick(now);
    $("replay-fill").style.width = `${(f * 100).toFixed(1)}%`;
    if (!replay.isPlaying() && store.get().running) store.set({ running: false });
  }

  renderer.render(now, dt);

  if (renderer.isDebugOn()) {
    const d = renderer.getDebugInfo();
    const car = d.car;
    $("debug").textContent =
      `${d.trackName.toUpperCase()}  ${(d.trackLengthM / 1000).toFixed(3)}km\n` +
      `fps ${d.fps.toFixed(0)}  state ${d.stateHz.toFixed(1)}Hz\n` +
      (car
        ? `x ${car.x.toFixed(1)}  y ${car.y.toFixed(1)}  yaw ${((car.yaw * 180) / Math.PI).toFixed(0)}°  v ${(car.speed * 3.6).toFixed(0)}km/h`
        : "no car");
  }

  requestAnimationFrame(loop);
}

// ---------- canvas sizing ----------
function resizeCanvas(): void {
  renderer.resize();
  if (renderer.hasTrack()) {
    // keep current view but re-fit only if no track interaction yet
  }
}

// ---------- async startup ----------
async function start(): Promise<void> {
  fitStage();
  resizeCanvas();

  // fetch meta + the default circuit (gracefully degrade if backend offline)
  try {
    const meta = await fetch("/api/meta").then((r) => {
      if (!r.ok) throw new Error(`/api/meta ${r.status}`);
      return r.json() as Promise<Meta>;
    });
    const track = await fetch(`/track/${meta.track_id}`).then((r) => {
      if (!r.ok) throw new Error(`/track/${meta.track_id} ${r.status}`);
      return r.json() as Promise<Track>;
    });
    currentTrack = track;
    renderer.resize();
    renderer.setTrack(track);
    hud.setMeta(meta);
    updateCircuitChip(track);
    store.set({ trackId: meta.track_id, lowConfidence: track.low_confidence });
    void selector.refresh(); // populate the catalog for the selector
    void policyPicker.refresh(); // populate the watch-live checkpoint dropdown
  } catch {
    // no backend yet — shell still renders; banner shown via store
    hud.reset();
  }

  // start with watch mode and connect
  socket.connect();
  const startMode = store.get().mode;
  if (startMode !== "configure" && startMode !== "replay") socket.sendMode(startMode);

  keyboard.attach();
  requestAnimationFrame(loop);
}

window.addEventListener("resize", () => {
  fitStage();
  resizeCanvas();
});

void start();
