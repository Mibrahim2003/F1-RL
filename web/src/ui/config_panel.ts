/**
 * Configure-mode surface editor: sliders for asphalt half-width, kerb, grass, and gravel
 * bands plus a dry/wet condition and a car-glyph toggle. Emits live previews on every change
 * and a SurfaceEdit on Save (POST /track/{id}/surfaces). Tracks unsaved/saved UI state.
 */

import type { GlyphStyle } from "../viewport/renderer.ts";
import type { SurfaceEdit } from "../types.ts";

export interface ConfigInitial {
  half_width: number;
  kerb_width: number;
  grass_width: number;
  gravel_width: number;
  condition: "dry" | "wet";
  glyph: GlyphStyle;
  trackName: string;
  lowConfidence: boolean;
}

export interface ConfigCallbacks {
  onPreview: (edit: SurfaceEdit) => void;
  onSave: (edit: SurfaceEdit) => Promise<boolean>;
  onGlyph: (style: GlyphStyle) => void;
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
  private condition: "dry" | "wet" = "dry";
  private glyph: GlyphStyle = "rect";

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
    this.syncCondition();
    this.syncGlyph();
    this.setStatus("clean");
    this.root.classList.remove("hidden");
  }

  hide(): void {
    this.root.classList.add("hidden");
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
    this.saveBtn.disabled = state === "saving" || state === "clean";
  }

  private build(): void {
    const root = document.createElement("div");
    root.className = "config-panel hidden";
    root.innerHTML = `
      <div class="cfg-head">
        <span class="cfg-title">SURFACES</span>
        <span class="cfg-track"></span>
        <span class="cfg-lowconf hidden" title="Low-confidence track">LOW CONF</span>
      </div>
      <div class="cfg-fields"></div>
      <div class="cfg-row">
        <span class="cfg-label">CONDITION</span>
        <div class="cfg-seg" data-group="condition">
          <button data-val="dry" class="active">DRY</button>
          <button data-val="wet">WET</button>
        </div>
      </div>
      <div class="cfg-row">
        <span class="cfg-label">CAR GLYPH</span>
        <div class="cfg-seg" data-group="glyph">
          <button data-val="rect" class="active">RECT</button>
          <button data-val="arrow">ARROW</button>
        </div>
      </div>
      <div class="cfg-foot">
        <span class="cfg-status"></span>
        <button class="cfg-save" disabled>SAVE</button>
      </div>`;
    this.host.appendChild(root);
    this.root = root;
    this.statusEl = root.querySelector(".cfg-status") as HTMLElement;
    this.saveBtn = root.querySelector(".cfg-save") as HTMLButtonElement;

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
        this.preview();
        this.setStatus("unsaved");
      });
    }

    root.querySelectorAll<HTMLElement>('.cfg-seg[data-group="condition"] button').forEach((b) => {
      b.addEventListener("click", () => {
        this.condition = b.dataset.val as "dry" | "wet";
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

    this.saveBtn.addEventListener("click", () => void this.save());
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

  private async save(): Promise<void> {
    this.setStatus("saving");
    const ok = await this.cb.onSave(this.edit());
    this.setStatus(ok ? "saved" : "unsaved");
  }
}
