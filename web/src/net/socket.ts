/** WebSocket client for /ws/sim — typed send, auto-reconnect with backoff. */

import type {
  ClientMessage,
  EventMessage,
  Mode,
  ServerMessage,
  StateFrame,
} from "../types.ts";

export interface SocketCallbacks {
  onState: (frame: StateFrame) => void;
  onEvent: (ev: EventMessage) => void;
  onOpen: () => void;
  onClose: () => void;
}

/** Build the ws:// or wss:// URL for the same-origin /ws/sim endpoint. */
function wsUrl(): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws/sim`;
}

export class SimSocket {
  private ws: WebSocket | null = null;
  private cb: SocketCallbacks;
  private backoff = 500;
  private readonly maxBackoff = 8000;
  private closedByUser = false;
  private reconnectTimer: number | null = null;

  constructor(cb: SocketCallbacks) {
    this.cb = cb;
  }

  connect(): void {
    this.closedByUser = false;
    this.open();
  }

  private open(): void {
    let ws: WebSocket;
    try {
      ws = new WebSocket(wsUrl());
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      this.backoff = 500;
      this.cb.onOpen();
    };

    ws.onmessage = (e) => {
      let msg: ServerMessage;
      try {
        msg = JSON.parse(e.data as string) as ServerMessage;
      } catch {
        return;
      }
      if (msg.type === "state") this.cb.onState(msg);
      else if (msg.type === "event") this.cb.onEvent(msg);
    };

    ws.onclose = () => {
      this.ws = null;
      this.cb.onClose();
      if (!this.closedByUser) this.scheduleReconnect();
    };

    // Errors are followed by a close event; let onclose drive reconnection.
    ws.onerror = () => {};
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer !== null) return;
    const delay = this.backoff;
    this.backoff = Math.min(this.maxBackoff, this.backoff * 2);
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.closedByUser) this.open();
    }, delay);
  }

  private send(msg: ClientMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  sendInput(steer: number, throttle: number, brake: number, reset = false): void {
    this.send({ type: "input", steer, throttle, brake, reset });
  }

  sendMode(mode: Mode): void {
    this.send({ type: "mode", mode });
  }

  sendControl(action: "play" | "pause" | "restart", speed?: 1 | 2 | 4): void {
    this.send(speed ? { type: "control", action, speed } : { type: "control", action });
  }

  sendRecord(action: "start" | "stop"): void {
    this.send({ type: "record", action });
  }

  isOpen(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  close(): void {
    this.closedByUser = true;
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
  }
}
