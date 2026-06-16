/** UI state machine. A small observable store; views subscribe to changes. */

import type { Mode } from "./types.ts";

export type UIState =
  | "loading"
  | "engine-online"
  | "engine-offline"
  | "manual"
  | "watch-live"
  | "replay-playing"
  | "replay-paused"
  | "replay-scrubbing"
  | "no-trajectory"
  | "error";

export interface AppState {
  ui: UIState;
  mode: Mode;
  engineConnected: boolean;
  running: boolean;
  speed: 1 | 2 | 4;
  errorMessage: string | null;
}

type Listener = (s: Readonly<AppState>) => void;

export class Store {
  private state: AppState = {
    ui: "loading",
    mode: "watch",
    engineConnected: false,
    running: true,
    speed: 1,
    errorMessage: null,
  };
  private listeners = new Set<Listener>();

  get(): Readonly<AppState> {
    return this.state;
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    fn(this.state);
    return () => this.listeners.delete(fn);
  }

  set(patch: Partial<AppState>): void {
    this.state = { ...this.state, ...patch };
    this.recomputeUI();
    for (const fn of this.listeners) fn(this.state);
  }

  /** Derive the high-level UI state from mode + connection + running flags. */
  private recomputeUI(): void {
    const s = this.state;
    let ui: UIState;
    if (s.errorMessage) {
      ui = "error";
    } else if (s.mode === "replay") {
      ui = s.running ? "replay-playing" : "replay-paused";
    } else if (!s.engineConnected) {
      ui = "engine-offline";
    } else if (s.mode === "manual") {
      ui = "manual";
    } else {
      ui = "watch-live";
    }
    this.state.ui = ui;
  }
}
