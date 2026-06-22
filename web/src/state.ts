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
  | "configure"
  | "loading-track"
  | "no-trajectory"
  | "error";

/** Surface-editor lifecycle for the configure panel. */
export type EditState = "clean" | "unsaved" | "saving" | "saved";

export interface AppState {
  ui: UIState;
  mode: Mode;
  engineConnected: boolean;
  running: boolean;
  speed: 1 | 2 | 4;
  errorMessage: string | null;
  trackId: string;
  loadingTrack: boolean;
  edit: EditState;
  lowConfidence: boolean;
}

type Listener = (s: Readonly<AppState>) => void;

export class Store {
  private state: AppState = {
    ui: "loading",
    // Boot into configure (race set-up), not a running watch session — the user sets conditions
    // and then triggers the start-light sequence with SAVE & RACE.
    mode: "configure",
    engineConnected: false,
    running: false,
    speed: 1,
    errorMessage: null,
    trackId: "oval",
    loadingTrack: false,
    edit: "clean",
    lowConfidence: false,
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
    } else if (s.loadingTrack) {
      ui = "loading-track";
    } else if (s.mode === "configure") {
      ui = "configure";
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
