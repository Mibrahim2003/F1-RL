/**
 * HUD: drives the bottom bar, the timing tower, and the top-bar clock from
 * state frames. Tabular figures keep digits from shifting.
 *
 * Phase 1: one car, kinematic (no gearbox / tyres). Gear shows "—",
 * tyre is a single static compound dot, sectors/DRS/FL stay placeholder.
 * The tower renders data-driven so 22 rows can drop in later.
 */

import {
  compoundColor,
  compoundLetter,
  deltaColor,
  formatClock,
  formatDelta,
  formatGrip,
  formatLapTime,
  formatSpeed,
  formatWear,
} from "../format.ts";
import type { CarEntry, Meta, StateFrame } from "../types.ts";

// Team color palette for the field tower (mirrors configs/default.yaml grid.team_colors).
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

interface TowerRowData {
  pos: number;
  num: string;
  code: string;
  teamColor: string;
  tyreColor: string;
  gap: string;
  gapColor: string;
  posColor: string;
  codeColor: string;
  isFastest: boolean;
  selected: boolean;
}

const $ = (id: string): HTMLElement => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el;
};

export class Hud {
  setMeta(meta: Meta): void {
    $("lap-total").textContent = String(meta.total_laps);
    $("pole-str").textContent = meta.pole_str || formatLapTime(meta.pole_time_s);
    $("circuit-name").textContent = meta.track_id.toUpperCase();
    this.renderTower(this.defaultRow());
  }

  /** Update HUD + tower + clock from a live or replayed state frame. */
  update(frame: StateFrame): void {
    // The leader's telemetry drives the bottom bar; field frames carry it at the top level.
    const tm = (frame.telemetry ?? frame.cars?.[0]?.telemetry) as
      | StateFrame["telemetry"]
      | undefined;
    if (!tm) return;

    // bottom bar
    $("speed").textContent = formatSpeed(tm.speed_kmh);
    $("gear").textContent = "—"; // kinematic: no gearbox
    $("lap-time").textContent = formatLapTime(tm.lap_time);
    $("last-lap").textContent = formatLapTime(tm.last_lap);

    const delta = $("delta");
    delta.textContent = formatDelta(tm.delta_to_pole);
    delta.style.color = deltaColor(tm.delta_to_pole);

    $("lap").textContent = String(Math.max(1, tm.lap));
    if (tm.lap_total > 0) $("lap-total").textContent = String(tm.lap_total);

    // Phase 3b: tyre compound + wear, and the live grip / weather (grip pipeline).
    const dot = $("tyre-dot");
    dot.textContent = compoundLetter(tm.compound);
    dot.style.background = compoundColor(tm.compound);
    $("tyre-wear").textContent = formatWear(tm.tire_wear);
    $("grip").textContent = formatGrip(tm.grip);
    $("weather").textContent = (tm.weather ?? "dry").toUpperCase();

    // top-bar session clock (use frame time as the running clock)
    $("clock").textContent = formatClock(frame.t);

    // timing tower: one row per car for a field (Phase 5), else the single live row.
    if (frame.cars && frame.cars.length > 1) {
      this.renderTower(...this.fieldRows(frame.cars));
    } else {
      this.renderTower({
        ...this.defaultRow(),
        tyreColor: compoundColor(tm.compound),
        gap: "LEADER",
        gapColor: "var(--text)",
      });
    }
  }

  /** Build timing-tower rows from a field frame: running order P1..PN with gap to the car ahead.
   *
   * Phase 6 frames carry a per-car `race_position` and `gap_ahead_s`; we order by race position
   * and show the real gap to the car directly ahead (falling back to the Phase 5 track-position
   * gap-to-leader for older frames that lack the racing fields). */
  private fieldRows(cars: CarEntry[]): TowerRowData[] {
    const hasRank = cars.some((c) => c.telemetry.race_position != null);
    const sorted = hasRank
      ? [...cars].sort(
          (a, b) =>
            (a.telemetry.race_position ?? 99) - (b.telemetry.race_position ?? 99) ||
            (a.gap_m ?? 0) - (b.gap_m ?? 0),
        )
      : [...cars].sort((a, b) => (a.gap_m ?? 0) - (b.gap_m ?? 0));
    return sorted.map((c, i) => {
      let gap: string;
      if (i === 0) {
        gap = "LEADER";
      } else if (hasRank && c.telemetry.gap_ahead_s != null) {
        gap = `+${c.telemetry.gap_ahead_s.toFixed(1)}s`;
      } else {
        gap = `+${(c.gap_m ?? 0).toFixed(0)}m`;
      }
      return {
        pos: i + 1,
        num: `#${String(i + 1).padStart(2, "0")}`,
        code: c.id.replace("car_", "C").toUpperCase(),
        teamColor: TEAM_COLORS[c.team % TEAM_COLORS.length],
        tyreColor: compoundColor(c.telemetry.compound),
        gap,
        gapColor: i === 0 ? "var(--text)" : "var(--text-2)",
        posColor: "var(--text)",
        codeColor: "var(--text-bright)",
        isFastest: false,
        selected: i === 0,
      };
    });
  }

  /** Reset the readouts to placeholders (e.g. before any frame arrives). */
  reset(): void {
    $("speed").textContent = "000";
    $("gear").textContent = "—";
    $("lap-time").textContent = formatLapTime(-1);
    $("last-lap").textContent = formatLapTime(-1);
    const delta = $("delta");
    delta.textContent = "+0.000";
    delta.style.color = "var(--text-2)";
    $("lap").textContent = "1";
    $("tyre-dot").textContent = "S";
    $("tyre-wear").textContent = "— %";
    $("grip").textContent = "—";
    $("weather").textContent = "DRY";
    this.renderTower(this.defaultRow());
  }

  private defaultRow(): TowerRowData {
    return {
      pos: 1,
      num: "#01",
      code: "SIM",
      teamColor: "var(--red)",
      tyreColor: "var(--tyre-s)",
      gap: "LEADER",
      gapColor: "var(--text)",
      posColor: "var(--text)",
      codeColor: "var(--text-bright)",
      isFastest: false,
      selected: true,
    };
  }

  private renderTower(...rows: TowerRowData[]): void {
    const body = $("tower-body");
    body.innerHTML = "";
    for (const r of rows) body.appendChild(this.rowEl(r));
  }

  private rowEl(r: TowerRowData): HTMLElement {
    const row = document.createElement("div");
    row.className = "tower-row";
    if (r.selected) row.style.background = "var(--raised)";

    const team = document.createElement("span");
    team.className = "team";
    team.style.background = r.teamColor;

    const pos = document.createElement("span");
    pos.className = "pos";
    pos.style.color = r.posColor;
    pos.textContent = String(r.pos);

    const num = document.createElement("span");
    num.className = "num";
    num.textContent = r.num;

    const code = document.createElement("span");
    code.className = "code";
    code.style.color = r.codeColor;
    code.textContent = r.code;

    const badges = document.createElement("span");
    badges.className = "badges";
    if (r.isFastest) {
      const fl = document.createElement("span");
      fl.className = "fl";
      fl.textContent = "FL";
      badges.appendChild(fl);
    }

    const tyreCell = document.createElement("span");
    tyreCell.className = "tyre-cell";
    const tyre = document.createElement("span");
    tyre.className = "tyre-dot";
    tyre.style.background = r.tyreColor;
    tyreCell.appendChild(tyre);

    const gap = document.createElement("span");
    gap.className = "gap";
    gap.style.color = r.gapColor;
    gap.textContent = r.gap;

    row.append(team, pos, num, code, badges, tyreCell, gap);
    return row;
  }
}
