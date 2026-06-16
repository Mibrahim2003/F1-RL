/**
 * Keyboard driving input: arrows + WASD. Combines opposing keys
 * (steer = right - left, longitudinal handled via throttle/brake).
 * Emits an input message only when the resolved action changes.
 */

export interface DriveInput {
  steer: number; // -1..1
  throttle: number; // 0..1
  brake: number; // 0..1
}

export type InputHandler = (input: DriveInput) => void;
export type ResetHandler = () => void;

const DRIVE_KEYS = new Set([
  "ArrowUp",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
  "KeyW",
  "KeyA",
  "KeyS",
  "KeyD",
]);

export class Keyboard {
  private pressed = new Set<string>();
  private last: DriveInput = { steer: 0, throttle: 0, brake: 0 };
  private enabled = false;
  private onInput: InputHandler;
  private onReset: ResetHandler;

  constructor(onInput: InputHandler, onReset: ResetHandler) {
    this.onInput = onInput;
    this.onReset = onReset;
  }

  /** Enable only in manual mode so driving keys don't fight other modes. */
  setEnabled(on: boolean): void {
    if (this.enabled === on) return;
    this.enabled = on;
    if (!on) {
      this.pressed.clear();
      this.emitIfChanged();
    }
  }

  attach(): void {
    window.addEventListener("keydown", this.onKeyDown, { passive: false });
    window.addEventListener("keyup", this.onKeyUp, { passive: false });
    window.addEventListener("blur", this.onBlur);
  }

  detach(): void {
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("keyup", this.onKeyUp);
    window.removeEventListener("blur", this.onBlur);
  }

  private onKeyDown = (e: KeyboardEvent): void => {
    if (!this.enabled) return;
    if (e.repeat) return;
    if (e.code === "KeyR") {
      e.preventDefault();
      this.onReset();
      return;
    }
    if (DRIVE_KEYS.has(e.code)) {
      e.preventDefault(); // never scroll the page while driving
      this.pressed.add(e.code);
      this.emitIfChanged();
    }
  };

  private onKeyUp = (e: KeyboardEvent): void => {
    if (DRIVE_KEYS.has(e.code)) {
      if (this.enabled) e.preventDefault();
      this.pressed.delete(e.code);
      this.emitIfChanged();
    }
  };

  private onBlur = (): void => {
    this.pressed.clear();
    this.emitIfChanged();
  };

  private resolve(): DriveInput {
    const up = this.pressed.has("ArrowUp") || this.pressed.has("KeyW");
    const down = this.pressed.has("ArrowDown") || this.pressed.has("KeyS");
    const left = this.pressed.has("ArrowLeft") || this.pressed.has("KeyA");
    const right = this.pressed.has("ArrowRight") || this.pressed.has("KeyD");
    return {
      throttle: up ? 1 : 0,
      brake: down ? 1 : 0,
      steer: (right ? 1 : 0) - (left ? 1 : 0),
    };
  }

  private emitIfChanged(): void {
    const next = this.resolve();
    if (
      next.steer === this.last.steer &&
      next.throttle === this.last.throttle &&
      next.brake === this.last.brake
    ) {
      return;
    }
    this.last = next;
    this.onInput(next);
  }
}
