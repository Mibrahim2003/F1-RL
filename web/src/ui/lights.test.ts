/**
 * StartLights tests (jsdom + fake timers). Verifies the F1 gantry builds, illuminates the five
 * lights in sequence, then goes dark ("lights out") and resolves run() so the caller can start
 * the race exactly then.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { StartLights } from "./lights.ts";

let host: HTMLElement;
let lights: StartLights;

function root(): HTMLElement {
  return host.querySelector(".start-lights") as HTMLElement;
}
function onCount(): number {
  return root().querySelectorAll(".sl-col.on").length;
}

beforeEach(() => {
  vi.useFakeTimers();
  host = document.createElement("div");
  document.body.appendChild(host);
  lights = new StartLights(host);
});

afterEach(() => {
  vi.useRealTimers();
  host.remove();
});

it("builds hidden with five twin-lamp columns", () => {
  expect(root().classList.contains("hidden")).toBe(true);
  const cols = root().querySelectorAll(".sl-col");
  expect(cols.length).toBe(5);
  cols.forEach((c) => expect(c.querySelectorAll(".sl-lamp").length).toBe(2));
});

describe("run()", () => {
  it("shows the overlay, lights up in sequence, goes dark, then resolves", async () => {
    vi.spyOn(Math, "random").mockReturnValue(0); // shortest hold, deterministic

    let resolved = false;
    const done = lights.run().then(() => {
      resolved = true;
    });

    // Synchronous part: overlay visible, marked running, nothing lit yet.
    expect(root().classList.contains("hidden")).toBe(false);
    expect(lights.isRunning()).toBe(true);
    expect(onCount()).toBe(0);

    // After the start delay, the first light is on.
    await vi.advanceTimersByTimeAsync(500);
    expect(onCount()).toBe(1);

    // One light per second up to all five.
    await vi.advanceTimersByTimeAsync(1000 * 4);
    expect(onCount()).toBe(5);
    expect(resolved).toBe(false); // still holding lit

    // Flush the hold + lights-out flash.
    await vi.advanceTimersByTimeAsync(10000);
    expect(resolved).toBe(true);
    expect(onCount()).toBe(0); // lights out
    expect(root().classList.contains("hidden")).toBe(true);
    expect(lights.isRunning()).toBe(false);
    expect(root().querySelector(".sl-caption")?.textContent).toBe("LIGHTS OUT");

    await done;
  });

  it("is a no-op while already running (guards double triggers)", async () => {
    vi.spyOn(Math, "random").mockReturnValue(0);
    const first = lights.run();
    expect(lights.isRunning()).toBe(true);

    let secondResolvedEarly = false;
    void lights.run().then(() => {
      if (lights.isRunning()) secondResolvedEarly = true;
    });
    await Promise.resolve();
    expect(secondResolvedEarly).toBe(true); // second call returned without touching the sequence

    await vi.advanceTimersByTimeAsync(20000);
    await first;
    expect(lights.isRunning()).toBe(false);
  });
});
