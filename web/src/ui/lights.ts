/**
 * F1 start-light gantry: five columns of twin red lamps that illuminate left-to-right one per
 * second, hold lit, then all extinguish ("lights out") — the moment the race begins. `run()`
 * resolves at lights-out so the caller can unpause the sim exactly then. Self-contained DOM under
 * a host element (mounted in the viewport), styled to match the app tokens.
 */

const COLUMNS = 5;
const START_DELAY_MS = 500; // a beat before the first light
const STEP_MS = 1000; // one light per second (authentic F1 cadence)
const HOLD_MIN_MS = 700; // random hold once all five are lit
const HOLD_MAX_MS = 1600;
const OUT_MS = 450; // "LIGHTS OUT" flash before the overlay clears

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

export class StartLights {
  private host: HTMLElement;
  private root!: HTMLElement;
  private cols: HTMLElement[] = [];
  private caption!: HTMLElement;
  private running = false;

  constructor(host: HTMLElement) {
    this.host = host;
    this.build();
  }

  /** True while a sequence is mid-flight (guards against double triggers). */
  isRunning(): boolean {
    return this.running;
  }

  /**
   * Play the full sequence: illuminate the five lights in turn, hold, then go dark. Resolves at
   * lights-out. A no-op (resolves immediately) if a sequence is already running.
   */
  async run(): Promise<void> {
    if (this.running) return;
    this.running = true;
    this.reset();
    this.root.classList.remove("hidden");
    try {
      await sleep(START_DELAY_MS);
      for (const col of this.cols) {
        col.classList.add("on");
        await sleep(STEP_MS);
      }
      await sleep(HOLD_MIN_MS + Math.random() * (HOLD_MAX_MS - HOLD_MIN_MS));
      // Lights out — the start.
      for (const col of this.cols) col.classList.remove("on");
      this.caption.textContent = "LIGHTS OUT";
      this.caption.classList.add("go");
      await sleep(OUT_MS);
    } finally {
      this.root.classList.add("hidden");
      this.running = false;
    }
  }

  private reset(): void {
    for (const col of this.cols) col.classList.remove("on");
    this.caption.textContent = "";
    this.caption.classList.remove("go");
  }

  private build(): void {
    const root = document.createElement("div");
    root.className = "start-lights hidden";
    const gantry = document.createElement("div");
    gantry.className = "sl-gantry";
    for (let i = 0; i < COLUMNS; i++) {
      const col = document.createElement("div");
      col.className = "sl-col";
      col.innerHTML = `<span class="sl-lamp"></span><span class="sl-lamp"></span>`;
      gantry.appendChild(col);
      this.cols.push(col);
    }
    const caption = document.createElement("div");
    caption.className = "sl-caption";
    root.appendChild(gantry);
    root.appendChild(caption);
    this.host.appendChild(root);
    this.root = root;
    this.caption = caption;
  }
}
