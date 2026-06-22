/**
 * Watch-live driver picker: a compact dropdown that chooses what drives the car — the
 * centerline autopilot, or a trained checkpoint from GET /api/checkpoints. Each pick sends a
 * PolicyMessage over the socket. Picking an early vs late checkpoint (sorted least-trained
 * first) is how you watch the agent improve.
 *
 * Self-contained like TrackSelector: builds its own DOM under a host element and is only
 * visible in watch mode. A backend policy_error is surfaced as a non-fatal status line, never
 * a crash — the server falls back to the autopilot and the viewport keeps streaming.
 */

import type { CheckpointSummary } from "../types.ts";

const AUTOPILOT_VALUE = "__autopilot__";

// Field-size bounds mirror the server's FieldMessage validator ([1, 22]).
const FIELD_MIN = 1;
const FIELD_MAX = 22;

export interface PolicyPickerCallbacks {
  onAutopilot: () => void;
  onCheckpoint: (id: string) => void;
  /** Set the live field size (Phase 5 many-cars); n is clamped to [1, 22]. */
  onField: (n: number) => void;
}

export class PolicyPicker {
  private host: HTMLElement;
  private cb: PolicyPickerCallbacks;
  private root!: HTMLElement;
  private select!: HTMLSelectElement;
  private status!: HTMLElement;
  private fieldInput!: HTMLInputElement;
  private fieldStatus!: HTMLElement;
  private checkpoints: CheckpointSummary[] = [];

  constructor(host: HTMLElement, cb: PolicyPickerCallbacks) {
    this.host = host;
    this.cb = cb;
    this.build();
  }

  /** Fetch the checkpoint catalog; safe to call repeatedly. False if the backend is offline. */
  async refresh(): Promise<boolean> {
    try {
      const r = await fetch("/api/checkpoints");
      if (!r.ok) throw new Error(`/api/checkpoints ${r.status}`);
      const data = (await r.json()) as { checkpoints: CheckpointSummary[] };
      this.checkpoints = data.checkpoints ?? [];
      this.renderOptions();
      return true;
    } catch {
      this.checkpoints = [];
      this.renderOptions();
      return false;
    }
  }

  show(): void {
    this.root.classList.remove("hidden");
  }

  hide(): void {
    this.root.classList.add("hidden");
  }

  /** Reflect a backend confirmation/error in the status line (non-fatal). */
  setStatus(text: string, kind: "ok" | "error" | "" = ""): void {
    this.status.textContent = text;
    this.status.dataset.kind = kind;
  }

  /** Reset the selection back to the autopilot (e.g. after a circuit switch). */
  reset(): void {
    this.select.value = AUTOPILOT_VALUE;
    this.setStatus("", "");
  }

  /** Reflect the server-confirmed field size (e.g. after a field_changed event). */
  setField(n: number): void {
    const clamped = clampField(n);
    this.fieldInput.value = String(clamped);
    this.fieldStatus.textContent =
      clamped === 1 ? "Single car" : `${clamped} cars on the grid`;
    this.fieldStatus.dataset.kind = "ok";
  }

  /** Read, clamp, and apply the field-size input — sends one field message. */
  private applyField(): void {
    const n = clampField(Number(this.fieldInput.value));
    this.fieldInput.value = String(n);
    this.fieldStatus.textContent = n === 1 ? "Single car…" : `Building ${n}-car grid…`;
    this.fieldStatus.dataset.kind = "";
    this.cb.onField(n);
  }

  private build(): void {
    const root = document.createElement("div");
    root.className = "policy-picker hidden";
    root.innerHTML = `
      <div class="pp-head">
        <span class="pp-title">DRIVER</span>
        <span class="pp-hint">backtick = beams overlay</span>
      </div>
      <select class="pp-select"></select>
      <div class="pp-status" data-kind=""></div>
      <div class="pp-field-row">
        <span class="pp-field-label">CARS ON TRACK</span>
        <input class="pp-field-input" type="number" min="${FIELD_MIN}" max="${FIELD_MAX}"
               step="1" value="1" />
        <button class="pp-field-btn" type="button">SET</button>
      </div>
      <div class="pp-field-status" data-kind=""></div>`;
    this.host.appendChild(root);
    this.root = root;
    this.select = root.querySelector(".pp-select") as HTMLSelectElement;
    this.status = root.querySelector(".pp-status") as HTMLElement;
    this.fieldInput = root.querySelector(".pp-field-input") as HTMLInputElement;
    this.fieldStatus = root.querySelector(".pp-field-status") as HTMLElement;

    this.select.addEventListener("change", () => {
      const value = this.select.value;
      if (value === AUTOPILOT_VALUE) {
        this.setStatus("Autopilot (centerline)", "");
        this.cb.onAutopilot();
      } else {
        this.setStatus("Loading checkpoint…", "");
        this.cb.onCheckpoint(value);
      }
    });

    // Apply the field size on SET click or Enter in the input.
    const btn = root.querySelector(".pp-field-btn") as HTMLButtonElement;
    btn.addEventListener("click", () => this.applyField());
    this.fieldInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        this.applyField();
      }
    });
    this.renderOptions();
  }

  private renderOptions(): void {
    const prev = this.select.value;
    this.select.innerHTML = "";
    this.select.appendChild(this.option(AUTOPILOT_VALUE, "Autopilot (centerline)"));
    for (const c of this.checkpoints) {
      const steps = formatSteps(c.total_timesteps);
      const label = `${c.id}  ·  ${c.circuit_id}  ·  ${steps}`;
      this.select.appendChild(this.option(c.id, label));
    }
    // Preserve the prior selection if it still exists, else fall back to the autopilot.
    const stillThere = Array.from(this.select.options).some((o) => o.value === prev);
    this.select.value = stillThere ? prev : AUTOPILOT_VALUE;
  }

  private option(value: string, label: string): HTMLOptionElement {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = label;
    return o;
  }
}

/** Clamp a field size into the server's valid [1, 22] range (NaN -> 1). */
function clampField(n: number): number {
  if (!Number.isFinite(n)) return FIELD_MIN;
  return Math.max(FIELD_MIN, Math.min(FIELD_MAX, Math.round(n)));
}

function formatSteps(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M steps`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k steps`;
  return `${n} steps`;
}
