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
  country: string;
  closed: boolean;
  length: number;
  official_length_m: number;
  length_error: number | null;
  source: string;
  low_confidence: boolean;
  centerline: [number, number][];
  tangent: [number, number][];
  normal: [number, number][];
  half_width_left: number[];
  half_width_right: number[];
  kerb_width: number[];
  grass_width: number[];
  gravel_width: number[];
  bounds: Bounds;
  start_finish: StartFinish;
}

/** Lightweight catalog entry from GET /api/tracks (the selector source). */
export interface TrackSummary {
  id: string;
  name: string;
  country: string;
  length: number;
  official_length_m: number;
  turns: number;
  source: string;
  low_confidence: boolean;
}

/** Edited surface band widths sent to POST /track/{id}/surfaces (uniform, meters). */
export interface SurfaceEdit {
  half_width_left?: number;
  half_width_right?: number;
  kerb_width?: number;
  grass_width?: number;
  gravel_width?: number;
  condition?: "dry" | "wet";
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
  // track_changed carries the new circuit's pace meta (mirrors GET /api/meta).
  control_hz?: number;
  pole_time_s?: number;
  total_laps?: number;
  pole_str?: string;
  // policy_changed / policy_error (watch-live checkpoint picker).
  source?: "autopilot" | "checkpoint";
  circuit_id?: string;
  total_timesteps?: number;
  message?: string;
}

export type ServerMessage = StateFrame | EventMessage;

export type Mode = "manual" | "watch" | "replay" | "configure";

export interface InputMessage {
  type: "input";
  steer: number;
  throttle: number;
  brake: number;
  reset: boolean;
}

/** Server-side sim modes (no "configure"; that is a client-only view). */
export type SimMode = "manual" | "watch" | "replay";

export interface ModeMessage {
  type: "mode";
  mode: SimMode;
}

export interface TrackMessage {
  type: "track";
  id: string;
}

/** Select what drives the car in watch mode: the centerline autopilot or a checkpoint. */
export interface PolicyMessage {
  type: "policy";
  source: "autopilot" | "checkpoint";
  id?: string;
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

export type ClientMessage =
  | InputMessage
  | ModeMessage
  | ControlMessage
  | RecordMessage
  | TrackMessage
  | PolicyMessage;

/** Catalog entry from GET /api/checkpoints (the watch-live policy picker source). */
export interface CheckpointSummary {
  id: string;
  total_timesteps: number;
  circuit_id: string;
  obs_version: number;
}

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
