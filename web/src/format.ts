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

type Compound = "soft" | "medium" | "hard" | "intermediate" | "wet";

/** One-letter tyre badge for a compound (Phase 3b). */
export function compoundLetter(c: Compound | undefined): string {
  return { soft: "S", medium: "M", hard: "H", intermediate: "I", wet: "W" }[c ?? "soft"];
}

/** Tyre-dot color for a compound, reusing the design tokens. */
export function compoundColor(c: Compound | undefined): string {
  const map: Record<Compound, string> = {
    soft: "var(--tyre-s)",
    medium: "var(--tyre-m)",
    hard: "var(--tyre-h)",
    intermediate: "var(--tyre-i, #43b047)",
    wet: "var(--tyre-w, #2f6fd0)",
  };
  return map[c ?? "soft"];
}

/** Tyre wear 0..1 -> "NN %" (placeholder when absent). */
export function formatWear(wear: number | undefined): string {
  if (wear === undefined || !Number.isFinite(wear)) return "— %";
  return `${Math.round(Math.max(0, Math.min(1, wear)) * 100)} %`;
}

/** Effective grip scalar -> 2-decimal readout (placeholder when absent). */
export function formatGrip(grip: number | undefined): string {
  if (grip === undefined || !Number.isFinite(grip)) return "—";
  return grip.toFixed(2);
}
