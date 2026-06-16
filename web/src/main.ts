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
import { Hud } from "./hud/telemetry.ts";
import { ReplayPlayer, fetchRecording, listRecordings } from "./replay/player.ts";
import type { Meta, Mode, StateFrame, Track } from "./types.ts";

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
    }
  },
  onOpen: () => store.set({ engineConnected: true }),
  onClose: () => store.set({ engineConnected: false }),
});

// ---------- state frame handling ----------
function onStateFrame(frame: StateFrame): void {
  // live frames only matter in manual/watch
  const mode = store.get().mode;
  if (mode === "replay") return;
  renderer.pushPose(frame.t, frame.car);
  hud.update(frame);
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
  store.set({ mode });
  if (mode === "replay") {
    void enterReplay();
  } else {
    socket.sendMode(mode);
    renderer.clearPoses();
    hud.reset();
    store.set({ running: true });
  }
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
  }
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

  // fetch track + meta (gracefully degrade if backend offline)
  try {
    const [track, meta] = await Promise.all([
      fetch("/track/oval").then((r) => {
        if (!r.ok) throw new Error(`/track/oval ${r.status}`);
        return r.json() as Promise<Track>;
      }),
      fetch("/api/meta").then((r) => {
        if (!r.ok) throw new Error(`/api/meta ${r.status}`);
        return r.json() as Promise<Meta>;
      }),
    ]);
    renderer.resize();
    renderer.setTrack(track);
    hud.setMeta(meta);
    $("circuit-name").textContent = track.name.toUpperCase();
  } catch {
    // no backend yet — shell still renders; banner shown via store
    hud.reset();
  }

  // start with watch mode and connect
  socket.connect();
  socket.sendMode(store.get().mode);

  keyboard.attach();
  requestAnimationFrame(loop);
}

window.addEventListener("resize", () => {
  fitStage();
  resizeCanvas();
});

void start();
