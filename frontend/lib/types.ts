export type TeamId = "A" | "B";

export interface Player {
  id: string;
  team: TeamId;
  shirt: number;
  name: string;
  position: string;
}

export interface MatchMeta {
  seed: number;
  duration: number;
  teams: Record<TeamId, { name: string; short: string; color: string }>;
  player_order: string[];
  final_score: Record<TeamId, number>;
  initial_score?: Record<TeamId, number>;
  clock_offset?: number;
  view?: string;
  reconstruction?: { mode: string; retrospective: boolean; camera: string; observed_share: number };
  live?: { t_now: number; speed: number; finished: boolean; started?: boolean; part?: string; note: string } | null;
  parts?: { id: string; label: string }[] | null;
  vod_available?: boolean; // ground-truth match that also carries the camera-only estimate (Replay / Live)
  source?: { type: string; game: string; half: number; start: number; end: number; model: string; coverage: number; observed_share?: number; dataset?: string };
  players: Player[];
  goals: MatchEvent[];
  key_events: MatchEvent[];
}

export interface Frame {
  t: number;
  b: [number, number];
  p: TeamId | null;
  c: string | null;
  xy: ([number, number] | null)[];
  cam?: [number, number][]; // camera footprint on the pitch (camera views of real-tracking matches)
  ds?: string; // dead ball: what restarts play ("Throw-in", "Corner", ...); absent while the ball is in play
  sl?: number[]; // each player's line in the current formation (0 = back line), -1 = goalkeeper / none
  obs?: (boolean | null)[]; // CV matches: false = position estimated (player off screen)
  bs?: string; // CV matches: ball source (carried, in_flight, dead_ball, goal, kick_off, near_player, raw)
  conf?: (number | null)[]; // reconstruction: P(error <= 5 m) for inferred players
  rad?: (number | null)[]; // reconstruction: calibrated uncertainty radius (m), 0 when observed
  cut?: boolean; // camera on a close-up / replay: nobody visible this second
}

export type ViewMode = "replay" | "live" | "truth";

export interface MatchEvent {
  id: string;
  timestamp: number;
  event_type: string;
  team_id: TeamId | null;
  player_id: string | null;
  x?: number;
  y?: number;
  target_x?: number;
  target_y?: number;
  success?: boolean;
  outcome?: string;
  xg?: number;
  [k: string]: unknown;
}

export interface Evidence {
  label: string;
  detail: string;
  value: number | string;
  unit: string;
  timestamp: number;
  basis?: string; // "observed events" | "observed + inferred positions" | "coverage"
}

export interface Coverage {
  team: TeamId;
  observed_avg: number;
  inferred_avg: number;
  observed_share: number;
  inferred_mean_confidence: number;
  reliability: number;
}

export interface ChainStep {
  stage: "EVENT" | "STATE_CHANGE" | "TACTICAL_CHANGE" | "CONSEQUENCE" | "OUTCOME";
  label: string;
  timestamp: number;
}

export interface CounterfactualRow {
  metric: string;
  key: string;
  unit: string;
  actual: number | null;
  counterfactual: number | null;
}

export interface HiddenState {
  team: TeamId; window_s: [number, number]; observed_players: number; inferred_players: number;
  high_confidence_inferred: number; outfield_players: number;
  metrics: Record<"observed_only" | "reconstructed" | "truth", Record<"def_line_m" | "width_m" | "near_ball", number> | null>;
  abs_error_vs_truth: Record<string, { observed_only: number | null; reconstructed: number | null }>;
}

export interface Moment {
  hidden_state?: HiddenState;
  id: string;
  type: string;
  subtype?: string | null;
  team: TeamId;
  timestamp: number;
  end_timestamp: number;
  minute_label: string;
  title: string;
  importance: number;
  confidence: number;
  trigger: Record<string, unknown>;
  before: Record<string, unknown>;
  after: Record<string, unknown>;
  deltas: Record<string, number>;
  evidence: Evidence[];
  consequences: string[];
  outcome: { type: string; description?: string; timestamp?: number };
  chain: ChainStep[];
  replay: { start: number; end: number; focus_players: string[]; focus_team: TeamId; overlay: string };
  narrative: {
    headline: string;
    why_it_matters: string;
    visibility_note?: string | null;
    fan_stats?: { value: number | string; label: string }[];
    analyst: { summary: string; bullets: string[] };
    fan: { headline: string; text: string };
    broadcast: { lower_third: string; ticker: string };
  };
  counterfactual?: {
    question: string;
    team: TeamId;
    window_minutes: [number, number];
    rows: CounterfactualRow[];
    method: string;
  } | null;
  links: { to: string; relation: string }[];
  coverage?: Coverage;
}

export interface Story {
  nodes: { id: string; type: string; team: TeamId; timestamp: number; title: string; importance: number }[];
  edges: { from: string; to: string; relation: string; explanation: string }[];
  chapters: { team: TeamId; label: string; start: number; end: number }[];
  key_moment_ids: string[];
}

export interface Reveal {
  t: number; player_id: string; hidden_s: number; predicted: [number, number]; actual: [number, number];
  error_m: number; radius_m: number; confidence: number; inside_radius: boolean;
}

export interface MinuteRow {
  minute: number;
  press_index: number;
  possession_s: number;
  territory_s: number;
  def_line_s: number;
  def_line_oop_s: number | null;
  width_s: number;
  depth_s: number;
  compactness_s: number;
  formation: string;
  formation_lines?: string[][]; // player ids per line, back line first (goalkeeper excluded)
  visibility?: { observed_avg: number; inferred_avg: number; inferred_conf: number | null; abstain: boolean };
  [k: string]: unknown;
}

export type Timeline = Record<TeamId, MinuteRow[]>;

export interface PlayerProfile {
  player: Player;
  until: number;
  movement: { distance_km: number; top_speed_kmh: number; sprint_seconds: number };
  stats: Record<string, number>;
  moments: { id: string; title: string; minute_label: string }[];
  visibility?: { observed_seconds: number; inferred_seconds: number; observed_share: number };
  narrative: { headline: string; impact: string };
}

export type Mode = "analyst" | "fan" | "player" | "accuracy";
