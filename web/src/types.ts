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
  // Phase 3b grip-pipeline readouts (optional: kinematic frames omit them).
  compound?: "soft" | "medium" | "hard" | "intermediate" | "wet";
  tire_wear?: number; // 0..1
  grip?: number; // effective grip scalar at the car
  weather?: "dry" | "damp" | "wet";
}

/** One car in a multi-car (field) frame; a single car is a one-element `cars` array. */
export interface CarEntry extends CarPose {
  id: string;
  team: number; // index into the team color palette (render only)
  gap_m?: number; // track-position gap behind the leader, meters (field frames)
  telemetry: Partial<Telemetry> & { completed_laps?: number };
}

export interface StateFrame {
  type: "state";
  t: number;
  // Legacy single-car keys (kept for backward compatibility; equal cars[leader]).
  car?: CarPose;
  telemetry?: Telemetry;
  // Phase 5: the whole field (length 1 for a single car).
  cars?: CarEntry[];
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
  // weather_changed (Phase 3b live grip).
  condition?: "dry" | "damp" | "wet";
  // field_changed (Phase 5 many-cars).
  n_agents?: number;
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

/** Set the live weather (Phase 3b grip pipeline); changes grip immediately. */
export interface WeatherMessage {
  type: "weather";
  condition: "dry" | "damp" | "wet";
}

export interface RecordMessage {
  type: "record";
  action: "start" | "stop";
}

/** Set the live field size (Phase 5 many-cars); >1 switches watch to the field view. */
export interface FieldMessage {
  type: "field";
  n_agents: number;
}

export type ClientMessage =
  | InputMessage
  | ModeMessage
  | ControlMessage
  | RecordMessage
  | TrackMessage
  | PolicyMessage
  | WeatherMessage
  | FieldMessage;

/** Catalog entry from GET /api/checkpoints (the watch-live policy picker source). */
export interface CheckpointSummary {
  id: string;
  total_timesteps: number;
  circuit_id: string;
  obs_version: number;
}

/** Recording trajectory served by GET /recordings/{id} (single-car or multi-car frames). */
export interface RecordingFrame {
  t: number;
  car?: CarPose;
  telemetry?: Partial<Telemetry>;
  cars?: CarEntry[];
}

export interface Recording {
  meta: { track_id: string; dt: number; seed: number; created: string; n_agents?: number };
  frames: RecordingFrame[];
}

export interface RecordingSummary {
  id: string;
  created: string;
  frames: number;
}

/** One circuit's row in the Phase 4 calendar table (GET /api/calendar). */
export interface CalendarRow {
  circuit: string;
  best_lap_time: number; // NaN when no clean lap was completed
  pole_time_s: number;
  delta_to_pole: number; // best_lap - pole; NaN when no lap / pole missing
  beat_pole_rate: number;
  beat_2x_pole_rate: number;
  off_track_count: number;
  completed_laps: number;
  pole_missing: boolean; // true => the circuit has no pole; skip its delta
}

/** Pool-level aggregates across the calendar (poles missing / NaN laps skipped). */
export interface CalendarAggregates {
  n_circuits: number;
  n_completed: number;
  mean_delta_to_pole: number;
  worst_delta_to_pole: number;
  worst_circuit: string | null;
  beat_pole_rate: number;
  beat_2x_pole_rate: number;
}

/** The saved calendar lap-time-vs-pole table (the Phase 4 result view). */
export interface CalendarTable {
  rows: CalendarRow[];
  aggregates: CalendarAggregates;
}
