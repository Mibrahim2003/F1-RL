/**
 * ConfigPanel DOM tests (jsdom). Covers the post-restructure config tab: the collapsible
 * main-option accordion (ROAD CONDITION / CAR GLYPH / DRIVER / CARS ON TRACK), the relocated
 * driver picker and live field-size input (formerly the watch-mode overlay), surface previews
 * and the SurfaceEdit emitted on Save, plus the [1, 22] field clamp.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ConfigPanel, type ConfigCallbacks, type ConfigInitial } from "./config_panel.ts";
import type { CheckpointSummary } from "../types.ts";

const INIT: ConfigInitial = {
  half_width: 6,
  kerb_width: 1,
  grass_width: 8,
  gravel_width: 4,
  condition: "dry",
  glyph: "rect",
  field: 1,
  trackName: "monza",
  lowConfidence: false,
};

function makeCallbacks(): ConfigCallbacks {
  return {
    onPreview: vi.fn(),
    onSave: vi.fn().mockResolvedValue(true),
    onResume: vi.fn(),
    onGlyph: vi.fn(),
    onAutopilot: vi.fn(),
    onCheckpoint: vi.fn(),
    onField: vi.fn(),
  };
}

let host: HTMLElement;
let cb: ConfigCallbacks;
let panel: ConfigPanel;

function root(): HTMLElement {
  return host.querySelector(".config-panel") as HTMLElement;
}
function q<T extends HTMLElement>(sel: string): T {
  return root().querySelector(sel) as T;
}
function opt(name: string): HTMLElement {
  return root().querySelector(`.cfg-opt[data-opt="${name}"]`) as HTMLElement;
}

beforeEach(() => {
  host = document.createElement("div");
  document.body.appendChild(host);
  cb = makeCallbacks();
  panel = new ConfigPanel(host, cb);
});

afterEach(() => {
  host.remove();
  vi.restoreAllMocks();
});

describe("structure", () => {
  it("builds hidden with the four main options in order", () => {
    expect(root().classList.contains("hidden")).toBe(true);
    const opts = Array.from(root().querySelectorAll<HTMLElement>(".cfg-opt")).map(
      (o) => o.dataset.opt,
    );
    expect(opts).toEqual(["road", "glyph", "driver", "field"]);
  });

  it("opens ROAD CONDITION by default, others collapsed", () => {
    expect(opt("road").classList.contains("open")).toBe(true);
    expect(opt("glyph").classList.contains("open")).toBe(false);
    expect(opt("driver").classList.contains("open")).toBe(false);
    expect(opt("field").classList.contains("open")).toBe(false);
  });

  it("groups the four surface sliders + dry/wet under ROAD CONDITION", () => {
    const sliders = opt("road").querySelectorAll('input[type="range"]');
    expect(sliders.length).toBe(4);
    expect(opt("road").querySelector('.cfg-seg[data-group="condition"]')).not.toBeNull();
  });

  it("clicking an option header toggles its open state", () => {
    const head = opt("driver").querySelector(".cfg-opt-head") as HTMLElement;
    expect(opt("driver").classList.contains("open")).toBe(false);
    head.click();
    expect(opt("driver").classList.contains("open")).toBe(true);
    head.click();
    expect(opt("driver").classList.contains("open")).toBe(false);
  });
});

describe("show()", () => {
  it("seeds slider/field values and unhides", () => {
    panel.show(INIT);
    expect(root().classList.contains("hidden")).toBe(false);
    expect(q<HTMLElement>(".cfg-track").textContent).toBe("MONZA");
    const ranges = Array.from(root().querySelectorAll<HTMLInputElement>('input[type="range"]'));
    expect(ranges.map((r) => Number(r.value))).toEqual([6, 1, 8, 4]);
    expect(Number(q<HTMLInputElement>(".cfg-field-input").value)).toBe(1);
    expect(q(".cfg-seg[data-group=\"condition\"] button.active").textContent).toBe("DRY");
  });

  it("starts clean but SAVE & RACE is always clickable (it doubles as resume)", () => {
    panel.show(INIT);
    expect(q<HTMLButtonElement>(".cfg-save").disabled).toBe(false);
    expect(q<HTMLButtonElement>(".cfg-save").textContent).toBe("SAVE & RACE");
    expect(q<HTMLElement>(".cfg-status").dataset.state).toBe("clean");
  });

  it("toggles the low-confidence badge", () => {
    panel.show({ ...INIT, lowConfidence: true });
    expect(q<HTMLElement>(".cfg-lowconf").classList.contains("hidden")).toBe(false);
    panel.show({ ...INIT, lowConfidence: false });
    expect(q<HTMLElement>(".cfg-lowconf").classList.contains("hidden")).toBe(true);
  });
});

describe("surface editing", () => {
  it("a slider change previews and marks unsaved", () => {
    panel.show(INIT);
    const asphalt = root().querySelector('input[type="range"]') as HTMLInputElement;
    asphalt.value = "9";
    asphalt.dispatchEvent(new Event("input"));
    expect(cb.onPreview).toHaveBeenCalledTimes(1);
    expect(q<HTMLElement>(".cfg-status").dataset.state).toBe("unsaved");
    expect(q<HTMLButtonElement>(".cfg-save").disabled).toBe(false);
  });

  it("a dirty SAVE & RACE emits the SurfaceEdit then resumes", async () => {
    panel.show(INIT);
    const asphalt = root().querySelector('input[type="range"]') as HTMLInputElement;
    asphalt.value = "9";
    asphalt.dispatchEvent(new Event("input"));
    (q(".cfg-seg[data-group=\"condition\"] button[data-val=\"wet\"]") as HTMLElement).click();

    q<HTMLButtonElement>(".cfg-save").click();
    await Promise.resolve();
    await Promise.resolve();

    expect(cb.onSave).toHaveBeenCalledTimes(1);
    const edit = (cb.onSave as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(edit.half_width_left).toBe(9);
    expect(edit.half_width_right).toBe(9); // left mirrored to right
    expect(edit.condition).toBe("wet");
    expect(q<HTMLElement>(".cfg-status").dataset.state).toBe("saved");
    expect(cb.onResume).toHaveBeenCalledTimes(1); // seamless return to watch
  });

  it("a clean SAVE & RACE skips the POST but still resumes (no needless .npz write)", () => {
    panel.show(INIT);
    q<HTMLButtonElement>(".cfg-save").click();
    expect(cb.onSave).not.toHaveBeenCalled();
    expect(cb.onResume).toHaveBeenCalledTimes(1);
  });

  it("a failed POST stays in config (unsaved, no resume) so the edit can be retried", async () => {
    (cb.onSave as ReturnType<typeof vi.fn>).mockResolvedValue(false);
    panel.show(INIT);
    const asphalt = root().querySelector('input[type="range"]') as HTMLInputElement;
    asphalt.value = "7";
    asphalt.dispatchEvent(new Event("input"));
    q<HTMLButtonElement>(".cfg-save").click();
    await Promise.resolve();
    await Promise.resolve();
    expect(q<HTMLElement>(".cfg-status").dataset.state).toBe("unsaved");
    expect(cb.onResume).not.toHaveBeenCalled();
  });

  it("a glyph/driver/field-only change resumes without a surface POST", () => {
    panel.show(INIT);
    (q('.cfg-seg[data-group="glyph"] button[data-val="arrow"]') as HTMLElement).click();
    q<HTMLInputElement>(".cfg-field-input").value = "6";
    q<HTMLButtonElement>(".cfg-field-btn").click();
    q<HTMLButtonElement>(".cfg-save").click();
    expect(cb.onSave).not.toHaveBeenCalled();
    expect(cb.onResume).toHaveBeenCalledTimes(1);
  });
});

describe("car glyph", () => {
  it("picking a glyph fires onGlyph and marks the button active", () => {
    panel.show(INIT);
    (q('.cfg-seg[data-group="glyph"] button[data-val="arrow"]') as HTMLElement).click();
    expect(cb.onGlyph).toHaveBeenCalledWith("arrow");
    expect(q('.cfg-seg[data-group="glyph"] button.active').textContent).toBe("ARROW");
  });

  it("does not touch the surface save state", () => {
    panel.show(INIT);
    (q('.cfg-seg[data-group="glyph"] button[data-val="arrow"]') as HTMLElement).click();
    expect(q<HTMLElement>(".cfg-status").dataset.state).toBe("clean");
  });
});

describe("driver picker", () => {
  it("refresh() populates the select from /api/checkpoints", async () => {
    const ckpts: CheckpointSummary[] = [
      { id: "ckpt_a", total_timesteps: 2_000_000, circuit_id: "monza", obs_version: 2 },
      { id: "ckpt_b", total_timesteps: 500, circuit_id: "spa", obs_version: 2 },
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => ({ checkpoints: ckpts }) }),
    );
    const ok = await panel.refresh();
    expect(ok).toBe(true);
    const sel = q<HTMLSelectElement>(".cfg-driver-select");
    // autopilot + the two checkpoints
    expect(sel.options.length).toBe(3);
    expect(sel.options[1].value).toBe("ckpt_a");
    expect(sel.options[1].textContent).toContain("2.0M steps");
  });

  it("refresh() failure leaves only the autopilot option", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500 }));
    const ok = await panel.refresh();
    expect(ok).toBe(false);
    const sel = q<HTMLSelectElement>(".cfg-driver-select");
    expect(sel.options.length).toBe(1);
  });

  it("selecting a checkpoint fires onCheckpoint; autopilot fires onAutopilot", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          checkpoints: [
            { id: "ckpt_a", total_timesteps: 1000, circuit_id: "monza", obs_version: 2 },
          ],
        }),
      }),
    );
    await panel.refresh();
    const sel = q<HTMLSelectElement>(".cfg-driver-select");
    sel.value = "ckpt_a";
    sel.dispatchEvent(new Event("change"));
    expect(cb.onCheckpoint).toHaveBeenCalledWith("ckpt_a");

    sel.value = "__autopilot__";
    sel.dispatchEvent(new Event("change"));
    expect(cb.onAutopilot).toHaveBeenCalledTimes(1);
  });

  it("setPolicyStatus / resetPolicy update the status line", () => {
    panel.setPolicyStatus("Policy: ckpt_a (monza)", "ok");
    expect(q<HTMLElement>(".cfg-driver-status").textContent).toBe("Policy: ckpt_a (monza)");
    expect(q<HTMLElement>(".cfg-driver-status").dataset.kind).toBe("ok");
    panel.resetPolicy();
    expect(q<HTMLSelectElement>(".cfg-driver-select").value).toBe("__autopilot__");
    expect(q<HTMLElement>(".cfg-driver-status").textContent).toBe("");
  });
});

describe("cars on track (field size)", () => {
  it("SET applies the entered size via onField", () => {
    panel.show(INIT);
    q<HTMLInputElement>(".cfg-field-input").value = "6";
    q<HTMLButtonElement>(".cfg-field-btn").click();
    expect(cb.onField).toHaveBeenCalledWith(6);
  });

  it("clamps to [1, 22] and rounds (NaN -> 1)", () => {
    panel.show(INIT);
    const input = q<HTMLInputElement>(".cfg-field-input");
    const set = q<HTMLButtonElement>(".cfg-field-btn");
    const onField = cb.onField as ReturnType<typeof vi.fn>;

    input.value = "99";
    set.click();
    expect(onField).toHaveBeenLastCalledWith(22);

    input.value = "0";
    set.click();
    expect(onField).toHaveBeenLastCalledWith(1);

    input.value = "3.7";
    set.click();
    expect(onField).toHaveBeenLastCalledWith(4);

    input.value = "abc";
    set.click();
    expect(onField).toHaveBeenLastCalledWith(1);
  });

  it("Enter in the input applies the size too", () => {
    panel.show(INIT);
    const input = q<HTMLInputElement>(".cfg-field-input");
    input.value = "5";
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    expect(cb.onField).toHaveBeenCalledWith(5);
  });

  it("SAVE & RACE commits a typed-but-not-SET size (the reported bug)", () => {
    panel.show(INIT); // starts at 1
    q<HTMLInputElement>(".cfg-field-input").value = "8";
    q<HTMLButtonElement>(".cfg-save").click();
    expect(cb.onField).toHaveBeenCalledWith(8);
    expect(cb.onResume).toHaveBeenCalledTimes(1);
  });

  it("is idempotent: SET then SAVE & RACE does not resend the same size", () => {
    panel.show(INIT);
    q<HTMLInputElement>(".cfg-field-input").value = "8";
    q<HTMLButtonElement>(".cfg-field-btn").click(); // SET
    q<HTMLButtonElement>(".cfg-save").click(); // SAVE & RACE
    expect(cb.onField).toHaveBeenCalledTimes(1);
    expect(cb.onField).toHaveBeenCalledWith(8);
  });

  it("an unchanged size is not resent on SAVE & RACE", () => {
    panel.setField(5); // server-confirmed 5 cars
    panel.show({ ...INIT, field: 5 });
    q<HTMLButtonElement>(".cfg-save").click();
    expect(cb.onField).not.toHaveBeenCalled();
  });

  it("setField reflects a server-confirmed size", () => {
    panel.setField(8);
    expect(q<HTMLInputElement>(".cfg-field-input").value).toBe("8");
    expect(q<HTMLElement>(".cfg-field-status").textContent).toBe("8 cars on the grid");
    panel.setField(1);
    expect(q<HTMLElement>(".cfg-field-status").textContent).toBe("Single car");
  });
});
