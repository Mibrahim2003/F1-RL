/**
 * HUD: drives the bottom bar, the timing tower, and the top-bar clock from
 * state frames. Tabular figures keep digits from shifting.
 *
 * Phase 1: one car, kinematic (no gearbox / tyres). Gear shows "—",
 * tyre is a single static compound dot, sectors/DRS/FL stay placeholder.
 * The tower renders data-driven so 22 rows can drop in later.
 */

import { deltaColor, formatClock, formatDelta, formatLapTime, formatSpeed } from "../format.ts";
import type { Meta, StateFrame } from "../types.ts";

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
    const tm = frame.telemetry;

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

    // top-bar session clock (use frame time as the running clock)
    $("clock").textContent = formatClock(frame.t);

    // timing tower (single row, populated with live gap = LEADER)
    this.renderTower({
      ...this.defaultRow(),
      gap: "LEADER",
      gapColor: "var(--text)",
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
