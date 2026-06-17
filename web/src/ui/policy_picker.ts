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

export interface PolicyPickerCallbacks {
  onAutopilot: () => void;
  onCheckpoint: (id: string) => void;
}

export class PolicyPicker {
  private host: HTMLElement;
  private cb: PolicyPickerCallbacks;
  private root!: HTMLElement;
  private select!: HTMLSelectElement;
  private status!: HTMLElement;
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

  private build(): void {
    const root = document.createElement("div");
    root.className = "policy-picker hidden";
    root.innerHTML = `
      <div class="pp-head">
        <span class="pp-title">DRIVER</span>
        <span class="pp-hint">backtick = beams overlay</span>
      </div>
      <select class="pp-select"></select>
      <div class="pp-status" data-kind=""></div>`;
    this.host.appendChild(root);
    this.root = root;
    this.select = root.querySelector(".pp-select") as HTMLSelectElement;
    this.status = root.querySelector(".pp-status") as HTMLElement;

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

function formatSteps(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M steps`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k steps`;
  return `${n} steps`;
}
