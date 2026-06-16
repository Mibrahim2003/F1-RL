/** Formatting helpers for tabular numeric readouts. */

const EMPTY_TIME = "-:--.---";

/** Lap time in seconds -> "m:ss.mmm". Negative / non-finite -> placeholder. */
export function formatLapTime(s: number): string {
  if (!Number.isFinite(s) || s < 0) return EMPTY_TIME;
  const m = Math.floor(s / 60);
  const rem = s - m * 60;
  const secs = rem.toFixed(3);
  return `${m}:${rem < 10 ? "0" : ""}${secs}`;
}

/** Session clock seconds -> "h:mm:ss". */
export function formatClock(s: number): string {
  const total = Math.max(0, Math.floor(s));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sec = total % 60;
  return `${h}:${m < 10 ? "0" : ""}${m}:${sec < 10 ? "0" : ""}${sec}`;
}

/** Signed delta in seconds -> "+0.000" / "-0.214". */
export function formatDelta(s: number): string {
  if (!Number.isFinite(s)) return "+0.000";
  const sign = s >= 0 ? "+" : "-";
  return `${sign}${Math.abs(s).toFixed(3)}`;
}

/** Speed km/h -> zero-padded 3-digit string. */
export function formatSpeed(kmh: number): string {
  const v = Number.isFinite(kmh) ? Math.max(0, Math.round(kmh)) : 0;
  return String(v).padStart(3, "0");
}

/**
 * Delta-to-pole color per the design's reserved meaning:
 * ahead of pole -> purple, slightly behind -> yellow, ~level -> neutral.
 */
export function deltaColor(delta: number): string {
  if (!Number.isFinite(delta)) return "var(--text-2)";
  if (delta < -0.03) return "var(--fastest)";
  if (delta > 0.03) return "var(--slower)";
  return "var(--text-2)";
}
