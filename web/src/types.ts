/** Shared types mirroring the backend contract (all geometry in meters). */

export interface Bounds {
  min_x: number;
  min_y: number;
  max_x: number;
  max_y: number;
}

export interface StartFinish {
  point: [number, number];
  tangent: [number, number];
  normal: [number, number];
}

export interface Track {
  name: string;
  closed: boolean;
  length: number;
  centerline: [number, number][];
  tangent: [number, number][];
  normal: [number, number][];
  half_width_left: number[];
  half_width_right: number[];
  runoff_width: number[];
  bounds: Bounds;
  start_finish: StartFinish;
}

export interface Meta {
  track_id: string;
  control_hz: number;
  pole_time_s: number;
  total_laps: number;
  pole_str: string;
}

export interface CarPose {
  x: number;
  y: number;
  yaw: number;
  speed: number;
}

export interface Telemetry {
  speed_kmh: number;
  lap_time: number;
  delta_to_pole: number;
  lap: number;
  lap_total: number;
  best_lap: number;
  last_lap: number;
  progress: number;
}

export interface StateFrame {
  type: "state";
  t: number;
  car: CarPose;
  telemetry: Telemetry;
}

export interface EventMessage {
  type: "event";
  event: string;
  id?: string;
}

export type ServerMessage = StateFrame | EventMessage;

export type Mode = "manual" | "watch" | "replay";

export interface InputMessage {
  type: "input";
  steer: number;
  throttle: number;
  brake: number;
  reset: boolean;
}

export interface ModeMessage {
  type: "mode";
  mode: Mode;
}

export interface ControlMessage {
  type: "control";
  action: "play" | "pause" | "restart";
  speed?: 1 | 2 | 4;
}

export interface RecordMessage {
  type: "record";
  action: "start" | "stop";
}

export type ClientMessage = InputMessage | ModeMessage | ControlMessage | RecordMessage;

/** Recording trajectory served by GET /recordings/{id}. */
export interface RecordingFrame {
  t: number;
  car: CarPose;
  telemetry: Partial<Telemetry>;
}

export interface Recording {
  meta: { track_id: string; dt: number; seed: number; created: string };
  frames: RecordingFrame[];
}

export interface RecordingSummary {
  id: string;
  created: string;
  frames: number;
}
