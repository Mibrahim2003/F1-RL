/**
 * Calendar result view: the Phase 4 artifact, one row per circuit — achieved lap vs the pole
 * and the delta — fetched from GET /api/calendar (the saved calendar_benchmark table). Hidden
 * until toggled (press T in watch mode); a 404 (no table yet) shows a friendly note instead of
 * crashing. Reuses the Phase-1 timing colors for the delta column.
 *
 * Self-contained like TrackSelector: builds its own DOM under a host element, renders with
 * textContent (no injection), and never crashes on a missing/partial table.
 */

import { deltaColor, formatDelta, formatLapTime } from "../format.ts";
import type { CalendarAggregates, CalendarRow, CalendarTable } from "../types.ts";

export class CalendarPanel {
  private host: HTMLElement;
  private root!: HTMLElement;
  private body!: HTMLElement;

  constructor(host: HTMLElement) {
    this.host = host;
    this.build();
  }

  /** Toggle the panel, refetching the table each time it opens. */
  toggle(): void {
    if (this.root.classList.contains("hidden")) void this.open();
    else this.hide();
  }

  async open(): Promise<void> {
    this.root.classList.remove("hidden");
    await this.refresh();
  }

  hide(): void {
    this.root.classList.add("hidden");
  }

  /** Fetch and render the saved table; a 404 means it has not been generated yet. */
  async refresh(): Promise<void> {
    try {
      const r = await fetch("/api/calendar");
      if (r.status === 404) {
        this.renderEmpty("No calendar table yet — run f1rl.train.calendar_benchmark.");
        return;
      }
      if (!r.ok) throw new Error(`/api/calendar ${r.status}`);
      this.render((await r.json()) as CalendarTable);
    } catch {
      this.renderEmpty("Calendar table unavailable (backend offline?).");
    }
  }

  private build(): void {
    const root = document.createElement("div");
    root.className = "calendar-panel hidden";
    root.innerHTML = `
      <div class="cal-head">
        <span class="cal-title">CALENDAR — LAP TIME VS POLE</span>
        <span class="cal-close" title="close">✕</span>
      </div>
      <div class="cal-body"></div>`;
    this.host.appendChild(root);
    this.root = root;
    this.body = root.querySelector(".cal-body") as HTMLElement;
    (root.querySelector(".cal-close") as HTMLElement).addEventListener("click", () => this.hide());
  }

  private renderEmpty(msg: string): void {
    this.body.innerHTML = "";
    const el = document.createElement("div");
    el.className = "cal-empty";
    el.textContent = msg;
    this.body.appendChild(el);
  }

  private render(table: CalendarTable): void {
    const rows = table.rows ?? [];
    if (rows.length === 0) {
      this.renderEmpty("Calendar table is empty.");
      return;
    }
    this.body.innerHTML = "";
    this.body.appendChild(this.headerRow());
    for (const r of rows) this.body.appendChild(this.dataRow(r));
    if (table.aggregates) this.body.appendChild(this.footer(table.aggregates));
  }

  private headerRow(): HTMLElement {
    const row = document.createElement("div");
    row.className = "cal-row cal-head-row";
    row.append(
      this.cell("CIRCUIT", "cal-c-circuit"),
      this.cell("BEST", "cal-c-num"),
      this.cell("POLE", "cal-c-num"),
      this.cell("DELTA", "cal-c-num"),
      this.cell("2×", "cal-c-flag"),
    );
    return row;
  }

  private dataRow(r: CalendarRow): HTMLElement {
    const row = document.createElement("div");
    row.className = "cal-row";
    const missing = r.pole_missing;
    const delta = missing ? "—" : formatDelta(r.delta_to_pole);
    const twox = r.beat_2x_pole_rate > 0 ? "✓" : "·";
    row.append(
      this.cell(r.circuit, "cal-c-circuit"),
      this.cell(formatLapTime(r.best_lap_time), "cal-c-num"),
      this.cell(missing ? "—" : formatLapTime(r.pole_time_s), "cal-c-num"),
      this.cell(delta, "cal-c-num", missing ? "var(--text-3)" : deltaColor(r.delta_to_pole)),
      this.cell(twox, "cal-c-flag"),
    );
    return row;
  }

  private footer(agg: CalendarAggregates): HTMLElement {
    const row = document.createElement("div");
    row.className = "cal-foot";
    const worst = agg.worst_circuit ? ` @${agg.worst_circuit}` : "";
    row.textContent =
      `${agg.n_completed}/${agg.n_circuits} lapped · ` +
      `mean Δ ${formatDelta(agg.mean_delta_to_pole)} · ` +
      `worst Δ ${formatDelta(agg.worst_delta_to_pole)}${worst} · ` +
      `2×pole ${Math.round((agg.beat_2x_pole_rate ?? 0) * 100)}%`;
    return row;
  }

  private cell(text: string, cls: string, color?: string): HTMLElement {
    const el = document.createElement("span");
    el.className = cls;
    el.textContent = text;
    if (color) el.style.color = color;
    return el;
  }
}
