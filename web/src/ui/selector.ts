/**
 * Track selector: a searchable modal grid of every cached circuit (GET /api/tracks).
 * Each card shows name, country, length, turn count, and a low-confidence badge.
 * Builds its own DOM under a host element; emits the chosen circuit id via onSelect.
 */

import type { TrackSummary } from "../types.ts";

export class TrackSelector {
  private host: HTMLElement;
  private onSelect: (id: string) => void;
  private overlay!: HTMLElement;
  private grid!: HTMLElement;
  private search!: HTMLInputElement;
  private tracks: TrackSummary[] = [];
  private currentId = "";

  constructor(host: HTMLElement, onSelect: (id: string) => void) {
    this.host = host;
    this.onSelect = onSelect;
    this.build();
  }

  /** Fetch the catalog; safe to call repeatedly. Returns false if the backend is offline. */
  async refresh(): Promise<boolean> {
    try {
      const r = await fetch("/api/tracks");
      if (!r.ok) throw new Error(`/api/tracks ${r.status}`);
      const data = (await r.json()) as { tracks: TrackSummary[] };
      this.tracks = data.tracks;
      this.renderGrid();
      return true;
    } catch {
      return false;
    }
  }

  open(currentId: string): void {
    this.currentId = currentId;
    this.search.value = "";
    this.renderGrid();
    this.overlay.classList.remove("hidden");
    this.search.focus();
  }

  close(): void {
    this.overlay.classList.add("hidden");
  }

  isOpen(): boolean {
    return !this.overlay.classList.contains("hidden");
  }

  private build(): void {
    const overlay = document.createElement("div");
    overlay.className = "selector-overlay hidden";
    overlay.innerHTML = `
      <div class="selector-modal">
        <div class="selector-head">
          <span class="selector-title">SELECT CIRCUIT</span>
          <input class="selector-search" type="text" placeholder="Search circuit or country…" />
          <button class="selector-close" title="Close">✕</button>
        </div>
        <div class="selector-grid"></div>
      </div>`;
    this.host.appendChild(overlay);
    this.overlay = overlay;
    this.grid = overlay.querySelector(".selector-grid") as HTMLElement;
    this.search = overlay.querySelector(".selector-search") as HTMLInputElement;

    this.search.addEventListener("input", () => this.renderGrid());
    overlay.querySelector(".selector-close")?.addEventListener("click", () => this.close());
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) this.close();
    });
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.isOpen()) this.close();
    });
  }

  private renderGrid(): void {
    const q = this.search.value.trim().toLowerCase();
    const items = this.tracks.filter(
      (t) =>
        !q ||
        t.name.toLowerCase().includes(q) ||
        t.country.toLowerCase().includes(q),
    );
    this.grid.innerHTML = "";
    if (items.length === 0) {
      const empty = document.createElement("div");
      empty.className = "selector-empty";
      empty.textContent = "No circuits match. Build tracks with scripts/build_all_tracks.py.";
      this.grid.appendChild(empty);
      return;
    }
    for (const t of items) {
      const card = document.createElement("button");
      card.className = "track-card" + (t.id === this.currentId ? " active" : "");
      const km = (t.length / 1000).toFixed(3);
      const badge = t.low_confidence ? '<span class="lc-badge" title="Low confidence">!</span>' : "";
      card.innerHTML = `
        <div class="tc-top">
          <span class="tc-name">${escapeHtml(t.name.toUpperCase())}</span>${badge}
        </div>
        <span class="tc-country">${escapeHtml(t.country)}</span>
        <div class="tc-stats">
          <span>${km} km</span><span>${t.turns} turns</span>
        </div>`;
      card.addEventListener("click", () => {
        this.onSelect(t.id);
        this.close();
      });
      this.grid.appendChild(card);
    }
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!;
  });
}
