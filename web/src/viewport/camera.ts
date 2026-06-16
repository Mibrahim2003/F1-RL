/**
 * Camera: meters <-> pixels transform with fit-to-track, drag pan,
 * wheel zoom, and a smooth camera-follow toggle.
 *
 * World is in SI meters (y up in track space). Canvas pixels have y down,
 * so the world->screen transform flips y.
 */

import type { Bounds } from "../types.ts";

export interface Vec2 {
  x: number;
  y: number;
}

export class Camera {
  /** pixels per meter */
  scale = 1;
  /** world-space point centered in the viewport */
  center: Vec2 = { x: 0, y: 0 };
  follow = false;

  private viewW = 1;
  private viewH = 1;
  private fitScale = 1;
  private fitCenter: Vec2 = { x: 0, y: 0 };
  private followTarget: Vec2 | null = null;

  /** CSS-pixel viewport size (not device pixels). */
  setViewport(w: number, h: number): void {
    this.viewW = Math.max(1, w);
    this.viewH = Math.max(1, h);
  }

  /** Compute a scale + center that fits the whole track with margin. */
  fitTo(bounds: Bounds, marginFrac = 0.08): void {
    const w = bounds.max_x - bounds.min_x;
    const h = bounds.max_y - bounds.min_y;
    const cx = (bounds.min_x + bounds.max_x) / 2;
    const cy = (bounds.min_y + bounds.max_y) / 2;
    const sx = this.viewW / (w * (1 + marginFrac * 2));
    const sy = this.viewH / (h * (1 + marginFrac * 2));
    this.fitScale = Math.min(sx, sy);
    this.fitCenter = { x: cx, y: cy };
    this.scale = this.fitScale;
    this.center = { ...this.fitCenter };
  }

  resetView(): void {
    this.scale = this.fitScale;
    this.center = { ...this.fitCenter };
    this.follow = false;
  }

  /**
   * The world→device transform as a 6-tuple [a,b,c,d,e,f] for ctx.setTransform,
   * folding in the device-pixel ratio. World y is up; screen y is down, so d is negative.
   * Lets the renderer draw cached world-meter Path2D geometry directly (pan/zoom aware).
   */
  viewMatrix(dpr: number): [number, number, number, number, number, number] {
    const s = this.scale;
    return [
      dpr * s,
      0,
      0,
      -dpr * s,
      dpr * (this.viewW / 2 - this.center.x * s),
      dpr * (this.viewH / 2 + this.center.y * s),
    ];
  }

  worldToScreen(p: Vec2): Vec2 {
    return {
      x: this.viewW / 2 + (p.x - this.center.x) * this.scale,
      y: this.viewH / 2 - (p.y - this.center.y) * this.scale,
    };
  }

  screenToWorld(p: Vec2): Vec2 {
    return {
      x: this.center.x + (p.x - this.viewW / 2) / this.scale,
      y: this.center.y - (p.y - this.viewH / 2) / this.scale,
    };
  }

  /** Pan by a screen-pixel delta (drag). */
  panByPixels(dx: number, dy: number): void {
    this.center.x -= dx / this.scale;
    this.center.y += dy / this.scale;
    this.follow = false;
  }

  /** Zoom about a screen anchor point (wheel). factor > 1 zooms in. */
  zoomAt(anchor: Vec2, factor: number): void {
    const before = this.screenToWorld(anchor);
    this.scale = clamp(this.scale * factor, this.fitScale * 0.25, this.fitScale * 20);
    const after = this.screenToWorld(anchor);
    this.center.x += before.x - after.x;
    this.center.y += before.y - after.y;
  }

  setFollow(on: boolean): void {
    this.follow = on;
    if (!on) this.followTarget = null;
  }

  /** Provide the latest car position; call once per frame before draw. */
  updateFollow(target: Vec2 | null, dtMs: number): void {
    if (!this.follow || !target) return;
    this.followTarget = target;
    // critically-damped-ish exponential smoothing toward the target
    const t = 1 - Math.exp(-dtMs / 140);
    this.center.x += (this.followTarget.x - this.center.x) * t;
    this.center.y += (this.followTarget.y - this.center.y) * t;
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
