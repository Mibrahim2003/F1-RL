/**
 * Canvas 2D track + car renderer.
 *
 * - Draws the broadcast look from /track/oval geometry (meters):
 *   infield, asphalt ribbon, red/white kerbs, faint dashed racing line,
 *   start/finish line.
 * - Runs at ~60 fps and interpolates between the last two 20 Hz state
 *   frames (lerp position, wrap-aware lerp of yaw).
 * - DPR-aware so thin lines stay crisp.
 */

import { Camera } from "./camera.ts";
import type { CarPose, Track } from "../types.ts";

// Visual constants (mirror tokens.css viewport group).
const C = {
  infield: "#19220F",
  kerbLight: "#D2D6DF",
  kerbRed: "#C8202B",
  asphalt: "#3A3A46",
  racingLine: "#E10600",
  startFinish: "#F2F2F7",
  carBody: "#E10600",
  carStroke: "rgba(0,0,0,0.5)",
  halo: "#E10600",
};

interface TimedPose {
  t: number;
  pose: CarPose;
}

export interface DebugInfo {
  fps: number;
  stateHz: number;
  car: CarPose | null;
}

export class Renderer {
  readonly camera = new Camera();
  private ctx: Canvas2DContext;
  private canvas: HTMLCanvasElement;
  private dpr = 1;
  private cssW = 1;
  private cssH = 1;

  private track: Track | null = null;
  // left/right edge polylines (world meters), precomputed when track loads
  private edgeL: [number, number][] = [];
  private edgeR: [number, number][] = [];

  private prev: TimedPose | null = null;
  private curr: TimedPose | null = null;
  private interpMs = 50; // nominal gap between 20 Hz frames

  // debug
  private debugOn = false;
  private frameTimes: number[] = [];
  private stateTimestamps: number[] = [];
  private lastDrawnCar: CarPose | null = null;

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

  setTrack(track: Track): void {
    this.track = track;
    this.precomputeEdges();
    this.camera.fitTo(track.bounds);
  }

  hasTrack(): boolean {
    return this.track !== null;
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

  clearPoses(): void {
    this.prev = null;
    this.curr = null;
  }

  toggleDebug(): void {
    this.debugOn = !this.debugOn;
  }

  isDebugOn(): boolean {
    return this.debugOn;
  }

  private lastReceived = 0;

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
    const pose = this.interpolatedPose(now);
    this.camera.updateFollow(pose ? { x: pose.x, y: pose.y } : null, dtMs);

    const ctx = this.ctx;
    ctx.save();
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.cssW, this.cssH);

    if (this.track) {
      this.drawTrack(ctx);
      if (pose) {
        this.drawCar(ctx, pose);
        this.lastDrawnCar = pose;
      }
    }
    ctx.restore();
  }

  getDebugInfo(): DebugInfo {
    return { fps: this.fps(), stateHz: this.stateHz(), car: this.lastDrawnCar };
  }

  // ---------- track drawing ----------

  private precomputeEdges(): void {
    const tk = this.track;
    if (!tk) return;
    this.edgeL = tk.centerline.map((c, i) => {
      const n = tk.normal[i];
      const w = tk.half_width_left[i];
      return [c[0] + n[0] * w, c[1] + n[1] * w] as [number, number];
    });
    this.edgeR = tk.centerline.map((c, i) => {
      const n = tk.normal[i];
      const w = tk.half_width_right[i];
      return [c[0] - n[0] * w, c[1] - n[1] * w] as [number, number];
    });
  }

  private drawTrack(ctx: CanvasRenderingContext2D): void {
    const tk = this.track;
    if (!tk) return;

    // 1) infield fill (region enclosed by the centerline)
    ctx.beginPath();
    this.tracePolyline(ctx, tk.centerline, true);
    ctx.fillStyle = C.infield;
    ctx.fill();

    // 2) asphalt ribbon between the two edge polylines
    ctx.beginPath();
    this.tracePolyline(ctx, this.edgeL, true);
    this.tracePolyline(ctx, this.edgeR, true);
    ctx.fillStyle = C.asphalt;
    ctx.fill("evenodd");

    // 3) kerb bands along both edges (alternating red/white dashes)
    this.drawKerb(ctx, this.edgeL);
    this.drawKerb(ctx, this.edgeR);

    // 4) faint dashed racing line along the centerline
    ctx.save();
    ctx.beginPath();
    this.tracePolyline(ctx, tk.centerline, tk.closed);
    ctx.strokeStyle = C.racingLine;
    ctx.globalAlpha = 0.3;
    ctx.lineWidth = Math.max(1, 0.4 * this.camera.scale);
    ctx.setLineDash([1.6 * this.camera.scale, 3.4 * this.camera.scale]);
    ctx.stroke();
    ctx.restore();

    // 5) start/finish line (perpendicular to track at s=0)
    this.drawStartFinish(ctx);
  }

  private drawKerb(ctx: CanvasRenderingContext2D, edge: [number, number][]): void {
    const seg = 3.0; // meters per kerb segment
    ctx.save();
    ctx.lineWidth = Math.max(1.5, 0.9 * this.camera.scale);
    ctx.lineJoin = "round";
    const dashPx = seg * this.camera.scale;
    ctx.setLineDash([dashPx, dashPx]);

    ctx.beginPath();
    this.tracePolyline(ctx, edge, true);
    ctx.strokeStyle = C.kerbLight;
    ctx.lineDashOffset = 0;
    ctx.stroke();

    ctx.beginPath();
    this.tracePolyline(ctx, edge, true);
    ctx.strokeStyle = C.kerbRed;
    ctx.lineDashOffset = dashPx;
    ctx.stroke();
    ctx.restore();
  }

  private drawStartFinish(ctx: CanvasRenderingContext2D): void {
    const tk = this.track;
    if (!tk) return;
    const sf = tk.start_finish;
    const p = { x: sf.point[0], y: sf.point[1] };
    const n = sf.normal; // points across the track
    const wl = tk.half_width_left[0] ?? 8;
    const wr = tk.half_width_right[0] ?? 8;
    const a = this.camera.worldToScreen({ x: p.x + n[0] * wl, y: p.y + n[1] * wl });
    const b = this.camera.worldToScreen({ x: p.x - n[0] * wr, y: p.y - n[1] * wr });
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = C.startFinish;
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = Math.max(2, 0.6 * this.camera.scale);
    ctx.stroke();
    ctx.restore();
  }

  private tracePolyline(
    ctx: CanvasRenderingContext2D,
    pts: [number, number][],
    closed: boolean,
  ): void {
    if (pts.length === 0) return;
    const first = this.camera.worldToScreen({ x: pts[0][0], y: pts[0][1] });
    ctx.moveTo(first.x, first.y);
    for (let i = 1; i < pts.length; i++) {
      const s = this.camera.worldToScreen({ x: pts[i][0], y: pts[i][1] });
      ctx.lineTo(s.x, s.y);
    }
    if (closed) ctx.closePath();
  }

  // ---------- car drawing ----------

  private drawCar(ctx: CanvasRenderingContext2D, pose: CarPose): void {
    const s = this.camera.worldToScreen({ x: pose.x, y: pose.y });
    // world yaw is CCW with y-up; screen y is flipped, so negate.
    const screenAngle = -pose.yaw;
    // size the car ~5 m long; keep a sensible minimum on screen
    const px = Math.max(8, this.camera.scale);

    ctx.save();
    ctx.translate(s.x, s.y);
    ctx.rotate(screenAngle);

    // featured-car red halo
    ctx.beginPath();
    ctx.arc(0, 0, px * 1.7, 0, Math.PI * 2);
    ctx.strokeStyle = C.halo;
    ctx.globalAlpha = 0.45;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.globalAlpha = 1;

    // angular body polygon (prototype points scaled by px/11)
    const u = px / 11;
    ctx.beginPath();
    ctx.moveTo(-8 * u, -5.5 * u);
    ctx.lineTo(11 * u, 0);
    ctx.lineTo(-8 * u, 5.5 * u);
    ctx.lineTo(-3.5 * u, 0);
    ctx.closePath();
    ctx.fillStyle = C.carBody;
    ctx.fill();
    ctx.strokeStyle = C.carStroke;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.restore();
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

type Canvas2DContext = CanvasRenderingContext2D;

function wrapAngle(a: number): number {
  let x = a;
  while (x > Math.PI) x -= 2 * Math.PI;
  while (x < -Math.PI) x += 2 * Math.PI;
  return x;
}
