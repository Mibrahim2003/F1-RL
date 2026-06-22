/** Store tests: the app must boot into the race set-up screen, not a running race. */

import { describe, expect, it } from "vitest";
import { Store } from "./state.ts";

describe("boot state", () => {
  it("starts in configure mode, paused", () => {
    const s = new Store().get();
    expect(s.mode).toBe("configure");
    expect(s.running).toBe(false);
  });

  it("derives the configure UI state once recomputed", () => {
    const store = new Store();
    store.set({}); // any update runs recomputeUI
    expect(store.get().ui).toBe("configure");
  });
});
