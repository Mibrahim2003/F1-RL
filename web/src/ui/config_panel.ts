/**
 * Configure-mode panel: an accordion of collapsible main options the user expands one at a
 * time. ROAD CONDITION holds the surface sliders (asphalt half-width, kerb, grass, gravel) plus
 * the dry/wet condition; CAR GLYPH toggles the car marker; DRIVER picks the watch-mode pilot
 * (centerline autopilot or a trained checkpoint from GET /api/checkpoints); CARS ON TRACK sets
 * the live field size. Surface changes emit live previews and a SurfaceEdit on Save
 * (POST /track/{id}/surfaces, tracked unsaved/saved). Driver and field changes apply immediately
 * over the socket — they are not gated behind Save. This replaces the old watch-mode overlay so
 * the picker no longer floats over the track.
 */

import type { GlyphStyle } from "../viewport/renderer.ts";
import type { CheckpointSummary, SurfaceEdit } from "../types.ts";

const AUTOPILOT_VALUE = "__autopilot__";

// Field-size bounds mirror the server's FieldMessage validator ([1, 22]).
const FIELD_MIN = 1;
const FIELD_MAX = 22;

export interface ConfigInitial {
  half_width: number;
  kerb_width: number;
  grass_width: number;
  gravel_width: number;
  condition: "dry" | "wet";
  glyph: GlyphStyle;
  field: number;
  trackName: string;
  lowConfidence: boolean;
}

export interface ConfigCallbacks {
  onPreview: (edit: SurfaceEdit) => void;
  /** Persist the surface bands (POST only — never changes mode). Resolves false on failure. */
  onSave: (edit: SurfaceEdit) => Promise<boolean>;
  /** Leave config and resume racing in watch mode (seamless "& RACE" half of the button). */
  onResume: () => void;
  onGlyph: (style: GlyphStyle) => void;
  onAutopilot: () => void;
  onCheckpoint: (id: string) => void;
  /** Set the live field size (Phase 5 many-cars); n is clamped to [1, 22]. */
  onField: (n: number) => void;
}

interface Field {
  key: keyof SurfaceEdit;
  label: string;
  min: number;
  max: number;
  step: number;
}

const FIELDS: Field[] = [
  { key: "half_width_left", label: "ASPHALT HALF-WIDTH", min: 2, max: 15, step: 0.5 },
  { key: "kerb_width", label: "KERB", min: 0, max: 3, step: 0.1 },
  { key: "grass_width", label: "GRASS", min: 0, max: 30, step: 0.5 },
  { key: "gravel_width", label: "GRAVEL", min: 0, max: 30, step: 0.5 },
];

export class ConfigPanel {
  private host: HTMLElement;
  private cb: ConfigCallbacks;
  private root!: HTMLElement;
  private inputs = new Map<string, HTMLInputElement>();
  private vals = new Map<string, HTMLElement>();
  private statusEl!: HTMLElement;
  private saveBtn!: HTMLButtonElement;
  private driverSelect!: HTMLSelectElement;
  private driverStatus!: HTMLElement;
  private fieldInput!: HTMLInputElement;
  private fieldStatus!: HTMLElement;
  private condition: "dry" | "wet" = "dry";
  private glyph: GlyphStyle = "rect";
  private checkpoints: CheckpointSummary[] = [];
  // True once a surface slider / condition changed and not yet persisted. Gates the POST so
  // "SAVE & RACE" only writes the .npz when something actually changed; it always resumes.
  private dirty = false;
  // Last field size actually sent to the server. The CARS ON TRACK input is "pending" until
  // committed (SET, Enter, or SAVE & RACE) — comparing to this avoids a needless grid rebuild.
  private appliedField = 1;

  constructor(host: HTMLElement, cb: ConfigCallbacks) {
    this.host = host;
    this.cb = cb;
    this.build();
  }

  show(init: ConfigInitial): void {
    this.condition = init.condition;
    this.glyph = init.glyph;
    (this.root.querySelector(".cfg-track") as HTMLElement).textContent = init.trackName.toUpperCase();
    this.root.querySelector(".cfg-lowconf")?.classList.toggle("hidden", !init.lowConfidence);
    const seed: Record<string, number> = {
      half_width_left: init.half_width,
      kerb_width: init.kerb_width,
      grass_width: init.grass_width,
      gravel_width: init.gravel_width,
    };
    for (const f of FIELDS) {
      const inp = this.inputs.get(f.key)!;
      inp.value = String(seed[f.key]);
      this.vals.get(f.key)!.textContent = `${Number(inp.value).toFixed(1)} m`;
    }
    this.appliedField = clampField(init.field);
    this.fieldInput.value = String(this.appliedField);
    this.dirty = false;
    this.syncCondition();
    this.syncGlyph();
    this.setStatus("clean");
    this.root.classList.remove("hidden");
  }

  hide(): void {
    this.root.classList.add("hidden");
  }

  /** Fetch the checkpoint catalog for the DRIVER option; safe to call repeatedly. */
  async refresh(): Promise<boolean> {
    try {
      const r = await fetch("/api/checkpoints");
      if (!r.ok) throw new Error(`/api/checkpoints ${r.status}`);
      const data = (await r.json()) as { checkpoints: CheckpointSummary[] };
      this.checkpoints = data.checkpoints ?? [];
      this.renderDriverOptions();
      return true;
    } catch {
      this.checkpoints = [];
      this.renderDriverOptions();
      return false;
    }
  }

  /** Reflect a backend policy confirmation/error in the DRIVER status line (non-fatal). */
  setPolicyStatus(text: string, kind: "ok" | "error" | "" = ""): void {
    this.driverStatus.textContent = text;
    this.driverStatus.dataset.kind = kind;
  }

  /** Reset the driver selection back to the autopilot (e.g. after a circuit switch). */
  resetPolicy(): void {
    this.driverSelect.value = AUTOPILOT_VALUE;
    this.setPolicyStatus("", "");
  }

  /** Reflect the server-confirmed field size (e.g. after a field_changed event). */
  setField(n: number): void {
    const clamped = clampField(n);
    this.appliedField = clamped;
    this.fieldInput.value = String(clamped);
    this.fieldStatus.textContent =
      clamped === 1 ? "Single car" : `${clamped} cars on the grid`;
    this.fieldStatus.dataset.kind = "ok";
  }

  setStatus(state: "clean" | "unsaved" | "saving" | "saved"): void {
    const label = {
      clean: "",
      unsaved: "UNSAVED CHANGES",
      saving: "SAVING…",
      saved: "SAVED",
    }[state];
    this.statusEl.textContent = label;
    this.statusEl.dataset.state = state;
    // Always clickable (it doubles as "resume racing"); only blocked mid-save to stop a
    // double-submit. A clean state still resumes — it just skips the POST.
    this.saveBtn.disabled = state === "saving";
  }

  private build(): void {
    const root = document.createElement("div");
    root.className = "config-panel hidden";
    root.innerHTML = `
      <div class="cfg-head">
        <span class="cfg-title">CONFIG</span>
        <span class="cfg-track"></span>
        <span class="cfg-lowconf hidden" title="Low-confidence track">LOW CONF</span>
      </div>
      <div class="cfg-accordion">
        <section class="cfg-opt open" data-opt="road">
          ${optHead("ROAD CONDITION")}
          <div class="cfg-opt-body">
            <div class="cfg-fields"></div>
            <div class="cfg-row">
              <span class="cfg-label">CONDITION</span>
              <div class="cfg-seg" data-group="condition">
                <button data-val="dry" class="active">DRY</button>
                <button data-val="wet">WET</button>
              </div>
            </div>
          </div>
        </section>
        <section class="cfg-opt" data-opt="glyph">
          ${optHead("CAR GLYPH")}
          <div class="cfg-opt-body">
            <div class="cfg-row">
              <span class="cfg-label">GLYPH</span>
              <div class="cfg-seg" data-group="glyph">
                <button data-val="rect" class="active">RECT</button>
                <button data-val="arrow">ARROW</button>
              </div>
            </div>
          </div>
        </section>
        <section class="cfg-opt" data-opt="driver">
          ${optHead("DRIVER")}
          <div class="cfg-opt-body">
            <select class="cfg-driver-select"></select>
            <div class="cfg-driver-status" data-kind=""></div>
          </div>
        </section>
        <section class="cfg-opt" data-opt="field">
          ${optHead("CARS ON TRACK")}
          <div class="cfg-opt-body">
            <div class="cfg-field-row">
              <input class="cfg-field-input" type="number" min="${FIELD_MIN}" max="${FIELD_MAX}"
                     step="1" value="1" />
              <button class="cfg-field-btn" type="button">SET</button>
            </div>
            <div class="cfg-field-status" data-kind=""></div>
          </div>
        </section>
      </div>
      <div class="cfg-foot">
        <span class="cfg-status"></span>
        <button class="cfg-save">SAVE &amp; RACE</button>
      </div>`;
    this.host.appendChild(root);
    this.root = root;
    this.statusEl = root.querySelector(".cfg-status") as HTMLElement;
    this.saveBtn = root.querySelector(".cfg-save") as HTMLButtonElement;
    this.driverSelect = root.querySelector(".cfg-driver-select") as HTMLSelectElement;
    this.driverStatus = root.querySelector(".cfg-driver-status") as HTMLElement;
    this.fieldInput = root.querySelector(".cfg-field-input") as HTMLInputElement;
    this.fieldStatus = root.querySelector(".cfg-field-status") as HTMLElement;

    // Accordion: clicking an option header expands/collapses its body.
    root.querySelectorAll<HTMLElement>(".cfg-opt-head").forEach((head) => {
      head.addEventListener("click", () => {
        head.closest(".cfg-opt")?.classList.toggle("open");
      });
    });

    const fieldsEl = root.querySelector(".cfg-fields") as HTMLElement;
    for (const f of FIELDS) {
      const row = document.createElement("div");
      row.className = "cfg-field";
      row.innerHTML = `
        <div class="cf-top"><span>${f.label}</span><span class="cf-val"></span></div>
        <input type="range" min="${f.min}" max="${f.max}" step="${f.step}" />`;
      fieldsEl.appendChild(row);
      const inp = row.querySelector("input") as HTMLInputElement;
      const val = row.querySelector(".cf-val") as HTMLElement;
      this.inputs.set(f.key, inp);
      this.vals.set(f.key, val);
      inp.addEventListener("input", () => {
        val.textContent = `${Number(inp.value).toFixed(1)} m`;
        this.dirty = true;
        this.preview();
        this.setStatus("unsaved");
      });
    }

    root.querySelectorAll<HTMLElement>('.cfg-seg[data-group="condition"] button').forEach((b) => {
      b.addEventListener("click", () => {
        this.condition = b.dataset.val as "dry" | "wet";
        this.dirty = true;
        this.syncCondition();
        this.setStatus("unsaved");
      });
    });
    root.querySelectorAll<HTMLElement>('.cfg-seg[data-group="glyph"] button').forEach((b) => {
      b.addEventListener("click", () => {
        this.glyph = b.dataset.val as GlyphStyle;
        this.syncGlyph();
        this.cb.onGlyph(this.glyph);
      });
    });

    // DRIVER: picking an option swaps the watch-mode pilot immediately (live socket message).
    this.driverSelect.addEventListener("change", () => {
      const value = this.driverSelect.value;
      if (value === AUTOPILOT_VALUE) {
        this.setPolicyStatus("Autopilot (centerline)", "");
        this.cb.onAutopilot();
      } else {
        this.setPolicyStatus("Loading checkpoint…", "");
        this.cb.onCheckpoint(value);
      }
    });

    // CARS ON TRACK: commit the field size on SET click or Enter (also done on SAVE & RACE).
    const fieldBtn = root.querySelector(".cfg-field-btn") as HTMLButtonElement;
    fieldBtn.addEventListener("click", () => this.commitField());
    this.fieldInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        this.commitField();
      }
    });

    this.saveBtn.addEventListener("click", () => void this.save());
    this.renderDriverOptions();
  }

  private renderDriverOptions(): void {
    const prev = this.driverSelect.value;
    this.driverSelect.innerHTML = "";
    this.driverSelect.appendChild(option(AUTOPILOT_VALUE, "Autopilot (centerline)"));
    for (const c of this.checkpoints) {
      const steps = formatSteps(c.total_timesteps);
      const label = `${c.id}  ·  ${c.circuit_id}  ·  ${steps}`;
      this.driverSelect.appendChild(option(c.id, label));
    }
    // Preserve the prior selection if it still exists, else fall back to the autopilot.
    const stillThere = Array.from(this.driverSelect.options).some((o) => o.value === prev);
    this.driverSelect.value = stillThere ? prev : AUTOPILOT_VALUE;
  }

  /**
   * Read + clamp the CARS ON TRACK input and, if it differs from the last applied size, send one
   * field message. Idempotent: re-committing the same value is a no-op (no grid rebuild). Called
   * from SET, Enter, and SAVE & RACE — so typing a number and hitting Save alone updates the grid.
   */
  private commitField(): void {
    const n = clampField(Number(this.fieldInput.value));
    this.fieldInput.value = String(n);
    if (n === this.appliedField) return;
    this.appliedField = n;
    this.fieldStatus.textContent = n === 1 ? "Single car…" : `Building ${n}-car grid…`;
    this.fieldStatus.dataset.kind = "";
    this.cb.onField(n);
  }

  private syncCondition(): void {
    this.root
      .querySelectorAll<HTMLElement>('.cfg-seg[data-group="condition"] button')
      .forEach((b) => b.classList.toggle("active", b.dataset.val === this.condition));
  }

  private syncGlyph(): void {
    this.root
      .querySelectorAll<HTMLElement>('.cfg-seg[data-group="glyph"] button')
      .forEach((b) => b.classList.toggle("active", b.dataset.val === this.glyph));
  }

  private edit(): SurfaceEdit {
    const hw = Number(this.inputs.get("half_width_left")!.value);
    return {
      half_width_left: hw,
      half_width_right: hw,
      kerb_width: Number(this.inputs.get("kerb_width")!.value),
      grass_width: Number(this.inputs.get("grass_width")!.value),
      gravel_width: Number(this.inputs.get("gravel_width")!.value),
      condition: this.condition,
    };
  }

  private preview(): void {
    this.cb.onPreview(this.edit());
  }

  /**
   * "SAVE & RACE": persist the surfaces only if they changed, then resume racing in watch mode.
   * A failed POST stays in config (so the edit can be retried); a clean panel skips straight to
   * the resume. This is what makes leaving config seamless — no manual switch back to watch.
   */
  private async save(): Promise<void> {
    this.commitField(); // apply a typed-but-not-SET field size before resuming
    if (this.dirty) {
      this.setStatus("saving");
      const ok = await this.cb.onSave(this.edit());
      if (!ok) {
        this.setStatus("unsaved");
        return;
      }
      this.dirty = false;
      this.setStatus("saved");
    }
    this.cb.onResume();
  }
}

/** Build an option header with a label and a rotating chevron. */
function optHead(label: string): string {
  return `
    <button class="cfg-opt-head" type="button">
      <span>${label}</span>
      <svg class="cfg-chevron" width="11" height="11" viewBox="0 0 12 12">
        <path d="M2 4 L6 8 L10 4" stroke="currentColor" stroke-width="1.4" fill="none" />
      </svg>
    </button>`;
}

function option(value: string, label: string): HTMLOptionElement {
  const o = document.createElement("option");
  o.value = value;
  o.textContent = label;
  return o;
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
