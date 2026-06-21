/**
 * Replay player: fetches a recording from the backend and drives the SAME
 * renderer + HUD as the live path, with play/pause and scrub. Works while
 * the live socket is idle (replay does not stream from /ws).
 */

import type { Hud } from "../hud/telemetry.ts";
import type { Renderer } from "../viewport/renderer.ts";
import type {
  CarEntry,
  Recording,
  RecordingSummary,
  StateFrame,
  Telemetry,
} from "../types.ts";

const EMPTY_TELEMETRY: Telemetry = {
  speed_kmh: 0,
  lap_time: 0,
  delta_to_pole: 0,
  lap: 1,
  lap_total: 1,
  best_lap: -1,
  last_lap: -1,
  progress: 0,
};

export async function listRecordings(): Promise<RecordingSummary[]> {
  const res = await fetch("/recordings");
  if (!res.ok) throw new Error(`/recordings -> ${res.status}`);
  const data = (await res.json()) as { recordings: RecordingSummary[] };
  return data.recordings;
}

export async function fetchRecording(id: string): Promise<Recording> {
  const res = await fetch(`/recordings/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`/recordings/${id} -> ${res.status}`);
  return (await res.json()) as Recording;
}

export class ReplayPlayer {
  private renderer: Renderer;
  private hud: Hud;
  private recording: Recording | null = null;
  private startWall = 0;
  private startT = 0;
  private playing = false;
  private speed = 1;
  private rafScrub = 0; // 0..1 set while scrubbing

  constructor(renderer: Renderer, hud: Hud) {
    this.renderer = renderer;
    this.hud = hud;
  }

  hasRecording(): boolean {
    return this.recording !== null && this.recording.frames.length > 0;
  }

  load(rec: Recording): void {
    this.recording = rec;
    this.playing = false;
    this.startT = rec.frames.length ? rec.frames[0].t : 0;
    this.startWall = performance.now();
    this.rafScrub = 0;
    this.applyAtFraction(0);
  }

  setSpeed(speed: number): void {
    // preserve current position when speed changes
    const f = this.currentFraction();
    this.speed = speed;
    this.seekFraction(f);
  }

  play(): void {
    if (!this.hasRecording()) return;
    // if at the end, restart
    if (this.currentFraction() >= 0.999) this.seekFraction(0);
    this.playing = true;
    this.startWall = performance.now();
    this.startT = this.fractionToT(this.currentFraction());
  }

  pause(): void {
    this.playing = false;
    this.rafScrub = this.currentFraction();
  }

  restart(): void {
    this.seekFraction(0);
    this.playing = false;
  }

  isPlaying(): boolean {
    return this.playing;
  }

  /** Scrub to a 0..1 fraction (from the scrub bar). */
  seekFraction(f: number): void {
    const clamped = Math.min(1, Math.max(0, f));
    this.rafScrub = clamped;
    if (this.playing) {
      this.startWall = performance.now();
      this.startT = this.fractionToT(clamped);
    }
    this.applyAtFraction(clamped);
  }

  /** Advance playback; returns current fraction for the scrub UI. */
  tick(now: number): number {
    if (!this.recording || this.recording.frames.length === 0) return 0;
    if (this.playing) {
      const elapsed = ((now - this.startWall) / 1000) * this.speed;
      const t = this.startT + elapsed;
      const f = this.tToFraction(t);
      if (f >= 1) {
        this.playing = false;
        this.applyAtFraction(1);
        return 1;
      }
      this.applyAtFraction(f);
      return f;
    }
    return this.rafScrub;
  }

  currentFraction(): number {
    return this.rafScrub;
  }

  // ---------- internals ----------

  private duration(): number {
    const f = this.recording!.frames;
    return f.length ? f[f.length - 1].t - f[0].t : 0;
  }

  private fractionToT(f: number): number {
    const frames = this.recording!.frames;
    return frames[0].t + f * this.duration();
  }

  private tToFraction(t: number): number {
    const d = this.duration();
    if (d <= 0) return 1;
    return (t - this.recording!.frames[0].t) / d;
  }

  private applyAtFraction(f: number): void {
    const rec = this.recording;
    if (!rec || rec.frames.length === 0) return;
    this.rafScrub = f;
    const t = this.fractionToT(f);
    const { a, b, alpha } = this.bracket(t);

    // Multi-car (field) frame: interpolate every car by id and render the whole field.
    if (a.cars && a.cars.length) {
      const bById = new Map((b.cars ?? []).map((c) => [c.id, c]));
      const cars: CarEntry[] = a.cars.map((ca) => {
        const cb = bById.get(ca.id) ?? ca;
        return {
          ...ca,
          x: ca.x + (cb.x - ca.x) * alpha,
          y: ca.y + (cb.y - ca.y) * alpha,
          yaw: ca.yaw + wrapAngle(cb.yaw - ca.yaw) * alpha,
          speed: ca.speed + (cb.speed - ca.speed) * alpha,
        };
      });
      this.renderer.setFieldImmediate(t, cars);
      const near = alpha < 0.5 ? a : b;
      this.hud.update({ type: "state", t, cars: near.cars });
      return;
    }

    // Single-car frame (legacy recordings).
    const aCar = a.car;
    const bCar = b.car ?? aCar;
    if (!aCar || !bCar) return;
    const pose = {
      x: aCar.x + (bCar.x - aCar.x) * alpha,
      y: aCar.y + (bCar.y - aCar.y) * alpha,
      yaw: aCar.yaw + wrapAngle(bCar.yaw - aCar.yaw) * alpha,
      speed: aCar.speed + (bCar.speed - aCar.speed) * alpha,
    };
    this.renderer.setPoseImmediate(t, pose);

    // synthesize a StateFrame for the HUD from the nearer keyframe's telemetry
    const near = alpha < 0.5 ? a : b;
    const tm = { ...EMPTY_TELEMETRY, ...near.telemetry } as Telemetry;
    tm.speed_kmh = pose.speed * 3.6;
    const synthetic: StateFrame = { type: "state", t, car: pose, telemetry: tm };
    this.hud.update(synthetic);
  }

  private bracket(t: number): {
    a: Recording["frames"][number];
    b: Recording["frames"][number];
    alpha: number;
  } {
    const frames = this.recording!.frames;
    if (t <= frames[0].t) return { a: frames[0], b: frames[0], alpha: 0 };
    const last = frames[frames.length - 1];
    if (t >= last.t) return { a: last, b: last, alpha: 0 };
    // linear scan is fine for Phase 1 recording sizes
    let i = 0;
    while (i < frames.length - 1 && frames[i + 1].t < t) i++;
    const a = frames[i];
    const b = frames[Math.min(i + 1, frames.length - 1)];
    const span = b.t - a.t;
    const alpha = span > 0 ? (t - a.t) / span : 0;
    return { a, b, alpha };
  }
}

function wrapAngle(a: number): number {
  let x = a;
  while (x > Math.PI) x -= 2 * Math.PI;
  while (x < -Math.PI) x += 2 * Math.PI;
  return x;
}
