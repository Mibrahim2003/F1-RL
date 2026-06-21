/**
 * Canvas 2D track + car renderer (Phase 2: layered real surfaces).
 *
 * - Track geometry is built once into cached world-meter Path2D objects on
 *   `setTrack` (grass/gravel runoff, asphalt, kerbs, racing line, start/finish).
 *   Each frame the camera world→device matrix is applied via ctx.setTransform,
 *   so pan/zoom/follow only changes a matrix — the heavy geometry is never rebuilt.
 *   Line widths/dashes are recomputed per frame from the live scale (cheap).
 * - Draws outside-in: infield → grass/gravel runoff → asphalt → kerb stripes →
 *   racing line → start/finish → car glyph.
 * - Car glyph: an oriented 5×2 m rectangle at the real F1 footprint (default), or a
 *   circle+arrow (toggle). Kept visible by a minimum on-screen size when zoomed out.
 * - Runs at ~60 fps and interpolates between the last two 20 Hz state frames.
 */

import { Camera } from "./camera.ts";
import type { CarEntry, CarPose, StateFrame, Track } from "../types.ts";

// Team color palette for the field (render only; mirrors configs/default.yaml grid.team_colors).
const TEAM_COLORS = [
  "#e10600",
  "#00d2be",
  "#0600ef",
  "#ff8700",
  "#006f62",
  "#2b4562",
  "#900000",
  "#005aff",
  "#ffffff",
  "#358c75",
  "#c92d4b",
];

// Visual constants (mirror tokens.css viewport group).
const C = {
  infield: "#19220F",
  grass: "#1F3A14",
  gravel: "#B89B6B",
  kerbLight: "#D2D6DF",
  kerbRed: "#C8202B",
  asphalt: "#3A3A46",
  racingLine: "#E10600",
  startFinish: "#F2F2F7",
  carBody: "#E10600",
  carStroke: "rgba(0,0,0,0.6)",
  carWindow: "rgba(0,0,0,0.35)",
  halo: "#E10600",
};

// Real F1 footprint (meters) for the rectangle glyph.
const CAR_LENGTH_M = 5.0;
const CAR_WIDTH_M = 2.0;

export type GlyphStyle = "rect" | "arrow";

interface TimedPose {
  t: number;
  pose: CarPose;
}

export interface DebugInfo {
  fps: number;
  stateHz: number;
  car: CarPose | null;
  trackName: string;
  trackLengthM: number;
}

export class Renderer {
  readonly camera = new Camera();
  private ctx: CanvasRenderingContext2D;
  private canvas: HTMLCanvasElement;
  private dpr = 1;
  private cssW = 1;
  private cssH = 1;

  private track: Track | null = null;
  private glyph: GlyphStyle = "rect";

  // cached world-meter geometry (rebuilt only on setTrack)
  private pInfield = new Path2D();
  private pGrass = new Path2D();
  private pGravel = new Path2D();
  private pAsphalt = new Path2D();
  private pKerbL = new Path2D();
  private pKerbR = new Path2D();
  private pRacingLine = new Path2D();
  private pStartFinish = new Path2D();
  private hasGravel = false;

  private prev: TimedPose | null = null;
  private curr: TimedPose | null = null;
  private interpMs = 50; // nominal gap between 20 Hz frames

  // Phase 5 field: per-car interpolation buffers, keyed by car id.
  private fieldMode = false;
  private fieldCars = new Map<
    string,
    { prev: TimedPose | null; curr: TimedPose | null; team: number; gap: number }
  >();
  private fieldLastReceived = 0;
  private fieldInterpMs = 50;
  private fieldLastT: number | null = null;

  // debug
  private debugOn = false;
  private frameTimes: number[] = [];
  private stateTimestamps: number[] = [];
  private lastDrawnCar: CarPose | null = null;
  private lastReceived = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2D canvas context unavailable");
    this.ctx = ctx;
  }

  resize(): void {
    const rect = this.canvas.getBoundingClientRect();
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.cssW = Math.max(1, rect.width);
    this.cssH = Math.max(1, rect.height);
    this.canvas.width = Math.round(this.cssW * this.dpr);
    this.canvas.height = Math.round(this.cssH * this.dpr);
    this.camera.setViewport(this.cssW, this.cssH);
  }

  setTrack(track: Track, refit = true): void {
    this.track = track;
    this.buildPaths();
    if (refit) this.camera.fitTo(track.bounds);
  }

  hasTrack(): boolean {
    return this.track !== null;
  }

  setGlyph(style: GlyphStyle): void {
    this.glyph = style;
  }

  getGlyph(): GlyphStyle {
    return this.glyph;
  }

  /** Feed a new authoritative car pose (from socket or replay). */
  pushPose(t: number, pose: CarPose): void {
    const now = performance.now();
    this.stateTimestamps.push(now);
    if (this.stateTimestamps.length > 40) this.stateTimestamps.shift();

    if (this.curr) {
      const gap = t - this.curr.t;
      if (gap > 0 && gap < 1) this.interpMs = gap * 1000;
    }
    this.prev = this.curr;
    this.curr = { t, pose };
    this.lastReceived = now;
  }

  /** Replace pose immediately without interpolation (replay scrub / reset). */
  setPoseImmediate(t: number, pose: CarPose): void {
    this.prev = { t, pose };
    this.curr = { t, pose };
    this.lastReceived = performance.now();
  }

  /** Feed a frame (single-car or field). A one-element `cars` array is the single-car path. */
  pushFrame(frame: StateFrame): void {
    if (frame.cars && frame.cars.length > 1) {
      this.pushFieldCars(frame.t, frame.cars);
      return;
    }
    this.fieldMode = false;
    const c = frame.cars && frame.cars.length ? frame.cars[0] : frame.car;
    if (c) this.pushPose(frame.t, { x: c.x, y: c.y, yaw: c.yaw, speed: c.speed });
  }

  /** Feed a multi-car field frame; updates per-car interpolation buffers. */
  pushFieldCars(t: number, cars: CarEntry[]): void {
    this.fieldMode = true;
    const now = performance.now();
    this.stateTimestamps.push(now);
    if (this.stateTimestamps.length > 40) this.stateTimestamps.shift();
    if (this.fieldLastT !== null) {
      const gap = t - this.fieldLastT;
      if (gap > 0 && gap < 1) this.fieldInterpMs = gap * 1000;
    }
    this.fieldLastT = t;
    this.fieldLastReceived = now;

    const seen = new Set<string>();
    for (const c of cars) {
      seen.add(c.id);
      let buf = this.fieldCars.get(c.id);
      if (!buf) {
        buf = { prev: null, curr: null, team: c.team, gap: c.gap_m ?? 0 };
        this.fieldCars.set(c.id, buf);
      }
      buf.team = c.team;
      buf.gap = c.gap_m ?? 0;
      buf.prev = buf.curr;
      buf.curr = { t, pose: { x: c.x, y: c.y, yaw: c.yaw, speed: c.speed } };
    }
    // Drop cars no longer in the field (e.g. the field size shrank).
    for (const id of [...this.fieldCars.keys()]) if (!seen.has(id)) this.fieldCars.delete(id);
  }

  /** Replace the whole field immediately without interpolation (replay scrub). */
  setFieldImmediate(t: number, cars: CarEntry[]): void {
    this.fieldMode = true;
    this.fieldLastReceived = performance.now();
    this.fieldLastT = t;
    const seen = new Set<string>();
    for (const c of cars) {
      seen.add(c.id);
      const tp: TimedPose = { t, pose: { x: c.x, y: c.y, yaw: c.yaw, speed: c.speed } };
      this.fieldCars.set(c.id, { prev: tp, curr: tp, team: c.team, gap: c.gap_m ?? 0 });
    }
    for (const id of [...this.fieldCars.keys()]) if (!seen.has(id)) this.fieldCars.delete(id);
  }

  clearPoses(): void {
    this.prev = null;
    this.curr = null;
    this.fieldMode = false;
    this.fieldCars.clear();
    this.fieldLastT = null;
  }

  toggleDebug(): void {
    this.debugOn = !this.debugOn;
  }

  isDebugOn(): boolean {
    return this.debugOn;
  }

  /** Interpolated pose for the current wall-clock time. */
  private interpolatedPose(now: number): CarPose | null {
    if (!this.curr) return null;
    if (!this.prev) return this.curr.pose;
    const age = now - this.lastReceived;
    const a = Math.min(1, Math.max(0, age / this.interpMs));
    const p0 = this.prev.pose;
    const p1 = this.curr.pose;
    return {
      x: p0.x + (p1.x - p0.x) * a,
      y: p0.y + (p1.y - p0.y) * a,
      yaw: p0.yaw + wrapAngle(p1.yaw - p0.yaw) * a,
      speed: p0.speed + (p1.speed - p0.speed) * a,
    };
  }

  /** Draw a single frame. dtMs is the wall-clock delta since last frame. */
  render(now: number, dtMs: number): void {
    this.recordFrame(now);
    const ctx = this.ctx;
    // Clear in device space, then switch to the world→device matrix for everything else.
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.cssW, this.cssH);

    if (this.fieldMode && this.fieldCars.size > 0) {
      this.renderField(ctx, now, dtMs);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      return;
    }

    const pose = this.interpolatedPose(now);
    this.camera.updateFollow(pose ? { x: pose.x, y: pose.y } : null, dtMs);
    if (this.track) {
      ctx.setTransform(...this.camera.viewMatrix(this.dpr));
      this.drawTrack(ctx);
      if (pose) {
        this.drawCar(ctx, pose, C.carBody, true);
        this.lastDrawnCar = pose;
      }
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  /** Render the whole field: every car colored by team, camera follows the leader. */
  private renderField(ctx: CanvasRenderingContext2D, now: number, dtMs: number): void {
    // Leader = the car with the smallest track-position gap.
    let leaderGap = Infinity;
    let leaderPose: CarPose | null = null;
    const poses: { pose: CarPose; team: number; gap: number }[] = [];
    for (const buf of this.fieldCars.values()) {
      const pose = this.interpField(buf, now);
      if (!pose) continue;
      poses.push({ pose, team: buf.team, gap: buf.gap });
      if (buf.gap < leaderGap) {
        leaderGap = buf.gap;
        leaderPose = pose;
      }
    }
    this.camera.updateFollow(leaderPose ? { x: leaderPose.x, y: leaderPose.y } : null, dtMs);

    if (this.track) {
      ctx.setTransform(...this.camera.viewMatrix(this.dpr));
      this.drawTrack(ctx);
      for (const p of poses) {
        const color = TEAM_COLORS[p.team % TEAM_COLORS.length];
        this.drawCar(ctx, p.pose, color, p.pose === leaderPose);
      }
      this.lastDrawnCar = leaderPose;
    }
    // The field's timing tower is the HUD's HTML tower (driven from the state frame's cars[]).
  }

  private interpField(
    buf: { prev: TimedPose | null; curr: TimedPose | null },
    now: number,
  ): CarPose | null {
    if (!buf.curr) return null;
    if (!buf.prev) return buf.curr.pose;
    const a = Math.min(1, Math.max(0, (now - this.fieldLastReceived) / this.fieldInterpMs));
    const p0 = buf.prev.pose;
    const p1 = buf.curr.pose;
    return {
      x: p0.x + (p1.x - p0.x) * a,
      y: p0.y + (p1.y - p0.y) * a,
      yaw: p0.yaw + wrapAngle(p1.yaw - p0.yaw) * a,
      speed: p0.speed + (p1.speed - p0.speed) * a,
    };
  }


  getDebugInfo(): DebugInfo {
    return {
      fps: this.fps(),
      stateHz: this.stateHz(),
      car: this.lastDrawnCar,
      trackName: this.track?.name ?? "",
      trackLengthM: this.track?.length ?? 0,
    };
  }

  // ---------- track geometry (built once per track, in world meters) ----------

  private buildPaths(): void {
    const tk = this.track;
    if (!tk) return;
    const off = (i: number, w: number): [number, number] => [
      tk.centerline[i][0] + tk.normal[i][0] * w,
      tk.centerline[i][1] + tk.normal[i][1] * w,
    ];

    const edgeL: [number, number][] = [];
    const edgeR: [number, number][] = [];
    const runoffL: [number, number][] = [];
    const runoffR: [number, number][] = [];
    for (let i = 0; i < tk.centerline.length; i++) {
      const hl = tk.half_width_left[i];
      const hr = tk.half_width_right[i];
      // grass is the outer runoff extent past the kerb; gravel sits within it where present.
      const outL = hl + tk.kerb_width[i] + tk.grass_width[i];
      const outR = hr + tk.kerb_width[i] + tk.grass_width[i];
      edgeL.push(off(i, hl));
      edgeR.push(off(i, -hr));
      runoffL.push(off(i, outL));
      runoffR.push(off(i, -outR));
    }

    // Infield: the region enclosed by the centerline.
    this.pInfield = ringPath(tk.centerline);
    // Grass runoff: ring between the outer runoff edge and the asphalt edge (even-odd hole).
    this.pGrass = ribbonPath(runoffL, runoffR);
    // Asphalt ribbon between the two asphalt edges.
    this.pAsphalt = ribbonPath(edgeL, edgeR);
    // Kerb stripes follow each asphalt edge.
    this.pKerbL = openPath(edgeL, true);
    this.pKerbR = openPath(edgeR, true);
    // Racing line along the centerline.
    this.pRacingLine = openPath(tk.centerline, tk.closed);
    // Gravel patches: quads over the kerb→gravel band wherever gravel_width > 0.
    this.buildGravel(tk);
    // Start/finish line across the track at sample 0.
    this.buildStartFinish(edgeL[0], edgeR[0]);
  }

  private buildGravel(tk: Track): void {
    const path = new Path2D();
    const n = tk.centerline.length;
    let any = false;
    const at = (i: number, w: number): [number, number] => [
      tk.centerline[i][0] + tk.normal[i][0] * w,
      tk.centerline[i][1] + tk.normal[i][1] * w,
    ];
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      for (const side of [1, -1]) {
        const gi = tk.gravel_width[i];
        const gj = tk.gravel_width[j];
        if (gi <= 0 && gj <= 0) continue;
        any = true;
        const hi = side > 0 ? tk.half_width_left[i] : tk.half_width_right[i];
        const hj = side > 0 ? tk.half_width_left[j] : tk.half_width_right[j];
        const innerI = side * (hi + tk.kerb_width[i]);
        const innerJ = side * (hj + tk.kerb_width[j]);
        const outerI = side * (hi + tk.kerb_width[i] + Math.max(gi, 0));
        const outerJ = side * (hj + tk.kerb_width[j] + Math.max(gj, 0));
        const a = at(i, innerI);
        const b = at(i, outerI);
        const c = at(j, outerJ);
        const d = at(j, innerJ);
        path.moveTo(a[0], a[1]);
        path.lineTo(b[0], b[1]);
        path.lineTo(c[0], c[1]);
        path.lineTo(d[0], d[1]);
        path.closePath();
      }
    }
    this.pGravel = path;
    this.hasGravel = any;
  }

  private buildStartFinish(a: [number, number], b: [number, number]): void {
    const p = new Path2D();
    p.moveTo(a[0], a[1]);
    p.lineTo(b[0], b[1]);
    this.pStartFinish = p;
  }

  private drawTrack(ctx: CanvasRenderingContext2D): void {
    const scale = this.camera.scale;
    const px = (n: number) => n / scale; // n device-independent px expressed in world meters

    // 1) infield
    ctx.fillStyle = C.infield;
    ctx.fill(this.pInfield);

    // 2) grass runoff ring (even-odd: outer loop minus asphalt loop)
    ctx.fillStyle = C.grass;
    ctx.fill(this.pGrass, "evenodd");

    // 3) gravel patches over the grass at corners
    if (this.hasGravel) {
      ctx.fillStyle = C.gravel;
      ctx.fill(this.pGravel);
    }

    // 4) asphalt ribbon
    ctx.fillStyle = C.asphalt;
    ctx.fill(this.pAsphalt, "evenodd");

    // 5) kerb stripes (alternating red/white dashes) along both edges
    this.drawKerb(ctx, this.pKerbL, px);
    this.drawKerb(ctx, this.pKerbR, px);

    // 6) faint dashed racing line
    ctx.save();
    ctx.strokeStyle = C.racingLine;
    ctx.globalAlpha = 0.3;
    ctx.lineWidth = Math.max(px(1), 0.4);
    ctx.setLineDash([Math.max(px(2), 1.6), Math.max(px(4), 3.4)]);
    ctx.stroke(this.pRacingLine);
    ctx.restore();

    // 7) start/finish
    ctx.save();
    ctx.strokeStyle = C.startFinish;
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = Math.max(px(2), 0.6);
    ctx.stroke(this.pStartFinish);
    ctx.restore();
  }

  private drawKerb(ctx: CanvasRenderingContext2D, edge: Path2D, px: (n: number) => number): void {
    const lw = Math.max(px(2), 0.9); // ~kerb band thickness in world meters, min 2 px
    const dash = Math.max(px(6), 3.0);
    ctx.save();
    ctx.lineWidth = lw;
    ctx.lineJoin = "round";
    ctx.setLineDash([dash, dash]);

    ctx.strokeStyle = C.kerbLight;
    ctx.lineDashOffset = 0;
    ctx.stroke(edge);

    ctx.strokeStyle = C.kerbRed;
    ctx.lineDashOffset = dash;
    ctx.stroke(edge);
    ctx.restore();
  }

  // ---------- car ----------

  private drawCar(
    ctx: CanvasRenderingContext2D,
    pose: CarPose,
    color: string = C.carBody,
    halo = true,
  ): void {
    const scale = this.camera.scale;
    // Real 5×2 m footprint, but never below ~9 px long so it stays visible when zoomed out.
    const len = Math.max(CAR_LENGTH_M, 9 / scale);
    const wid = len * (CAR_WIDTH_M / CAR_LENGTH_M);
    const px = (n: number) => n / scale;

    ctx.save();
    ctx.translate(pose.x, pose.y);
    ctx.rotate(pose.yaw); // world is y-up; the view matrix's y-flip makes CCW read correctly

    // featured-car halo (leader only in field mode)
    if (halo) {
      ctx.beginPath();
      ctx.arc(0, 0, len * 0.95, 0, Math.PI * 2);
      ctx.strokeStyle = C.halo;
      ctx.globalAlpha = 0.4;
      ctx.lineWidth = Math.max(px(2), len * 0.05);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    if (this.glyph === "arrow") this.drawArrowGlyph(ctx, len, px, color);
    else this.drawRectGlyph(ctx, len, wid, px, color);

    ctx.restore();
  }

  private drawRectGlyph(
    ctx: CanvasRenderingContext2D,
    len: number,
    wid: number,
    px: (n: number) => number,
    color: string = C.carBody,
  ): void {
    const hl = len / 2;
    const hw = wid / 2;
    ctx.beginPath();
    ctx.moveTo(hl, 0); // nose
    ctx.lineTo(hl * 0.6, -hw);
    ctx.lineTo(-hl, -hw);
    ctx.lineTo(-hl, hw);
    ctx.lineTo(hl * 0.6, hw);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = C.carStroke;
    ctx.lineWidth = Math.max(px(1), len * 0.03);
    ctx.stroke();
    // cockpit hint
    ctx.beginPath();
    ctx.ellipse(-hl * 0.1, 0, len * 0.12, wid * 0.22, 0, 0, Math.PI * 2);
    ctx.fillStyle = C.carWindow;
    ctx.fill();
  }

  private drawArrowGlyph(
    ctx: CanvasRenderingContext2D,
    len: number,
    px: (n: number) => number,
    color: string = C.carBody,
  ): void {
    const r = len * 0.42;
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = C.carStroke;
    ctx.lineWidth = Math.max(px(1), len * 0.03);
    ctx.stroke();
    // forward arrow
    ctx.beginPath();
    ctx.moveTo(r * 1.4, 0);
    ctx.lineTo(r * 0.2, -r * 0.6);
    ctx.lineTo(r * 0.2, r * 0.6);
    ctx.closePath();
    ctx.fillStyle = C.startFinish;
    ctx.fill();
  }

  // ---------- debug metrics ----------

  private recordFrame(now: number): void {
    this.frameTimes.push(now);
    if (this.frameTimes.length > 60) this.frameTimes.shift();
  }

  private fps(): number {
    if (this.frameTimes.length < 2) return 0;
    const span = this.frameTimes[this.frameTimes.length - 1] - this.frameTimes[0];
    return span > 0 ? ((this.frameTimes.length - 1) / span) * 1000 : 0;
  }

  private stateHz(): number {
    if (this.stateTimestamps.length < 2) return 0;
    const span =
      this.stateTimestamps[this.stateTimestamps.length - 1] - this.stateTimestamps[0];
    return span > 0 ? ((this.stateTimestamps.length - 1) / span) * 1000 : 0;
  }
}

// ---------- Path2D builders (world meters) ----------

function ringPath(pts: [number, number][]): Path2D {
  const p = new Path2D();
  if (pts.length === 0) return p;
  p.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) p.lineTo(pts[i][0], pts[i][1]);
  p.closePath();
  return p;
}

/** Closed ribbon between two edge polylines (fill with "evenodd" for a hole). */
function ribbonPath(outer: [number, number][], inner: [number, number][]): Path2D {
  const p = new Path2D();
  if (outer.length === 0) return p;
  p.moveTo(outer[0][0], outer[0][1]);
  for (let i = 1; i < outer.length; i++) p.lineTo(outer[i][0], outer[i][1]);
  p.closePath();
  if (inner.length) {
    p.moveTo(inner[0][0], inner[0][1]);
    for (let i = 1; i < inner.length; i++) p.lineTo(inner[i][0], inner[i][1]);
    p.closePath();
  }
  return p;
}

function openPath(pts: [number, number][], closed: boolean): Path2D {
  const p = new Path2D();
  if (pts.length === 0) return p;
  p.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) p.lineTo(pts[i][0], pts[i][1]);
  if (closed) p.closePath();
  return p;
}

function wrapAngle(a: number): number {
  let x = a;
  while (x > Math.PI) x -= 2 * Math.PI;
  while (x < -Math.PI) x += 2 * Math.PI;
  return x;
}
