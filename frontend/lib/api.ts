import type { Frame, MatchEvent, MatchMeta, Moment, PlayerProfile, Story, Timeline } from "./types";

// Deployed (Vercel services): the backend is public under /api on the same domain. Local ./run.sh sets
// NEXT_PUBLIC_API_URL=http://localhost:8000 (FastAPI also answers without the /api prefix).
export const API = process.env.NEXT_PUBLIC_API_URL ?? "/api";

// which match / part the /match/... calls are about (set by the studio); empty = the server's default match
let scopeQS = "";
export function setScope(match?: string, part?: string) {
  const q = new URLSearchParams();
  if (match) q.set("match", match);
  if (part) q.set("part", part);
  scopeQS = q.toString();
}

async function get<T>(path: string): Promise<T> {
  const scoped = path.startsWith("/match/") || path === "/match" || path.startsWith("/evaluation");
  const url = scoped && scopeQS ? `${path}${path.includes("?") ? "&" : "?"}${scopeQS}` : path;
  const r = await fetch(`${API}${url}`);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

export interface MatchBundle {
  meta: MatchMeta;
  parts?: { id: string; label: string }[] | null;
  frames: Frame[];
  events: MatchEvent[];
  moments: Moment[];
  story: Story;
  timeline: Timeline;
}

export async function loadMatch(): Promise<MatchBundle> {
  const [meta, fr, events, moments, story, timeline] = await Promise.all([
    get<MatchMeta>("/match"),
    get<{ frames: Frame[] }>("/match/frames"),
    get<MatchEvent[]>("/match/events"),
    get<Moment[]>("/match/moments"),
    get<Story>("/match/story"),
    get<Timeline>("/match/timeline"),
  ]);
  return { meta, parts: meta.parts, frames: fr.frames, events, moments, story, timeline };
}

export const loadFrames = (mode: string) => get<{ frames: Frame[] }>(`/match/frames?mode=${mode}`).then((r) => r.frames);

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const loadEvaluation = () => get<any>("/evaluation");

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const loadPlayerState = (pid: string, t: number, mode: string) => get<any>(`/match/players/${pid}/state?t=${Math.floor(t)}&mode=${mode}`);

export const loadPlayer = (pid: string, t: number, mode: string = "default") => get<PlayerProfile>(`/match/players/${pid}?t=${Math.floor(t)}&mode=${mode}`);

export const fmtClock = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

// ---------------------------------------------------------------- lobby
export interface TeamInfo { name: string; short: string; color: string }
export interface LobbyCard {
  id: string;
  kind: "tracking" | "computer_vision" | "synthetic";
  label: string;
  teams: Record<"A" | "B", TeamInfo>;
  score: Record<"A" | "B", number>;
  score_known?: boolean;
  status: "live" | "finished";
  minute?: number;
  competition?: string | null;
  date?: string | null;
  season?: string | null;
  matchday?: string | null;
  note?: string | null;
  parts: { id: string; label: string }[];
  live_part?: string;
}
export const loadLobby = () => get<LobbyCard[]>("/matches");

export interface TeamStats {
  goals: number; shots: number; shots_on_target: number; xg: number; passes: number; passes_completed: number;
  pass_accuracy: number; tackles_won: number; fouls: number; corners: number; offsides: number; yellow_cards: number;
  red_cards: number; possession: number; distance_km: number | null; press_index: number | null;
}
export interface Highlight {
  id: string; timestamp: number; end_timestamp: number; minute_label: string; type: string; team: "A" | "B";
  title: string; importance: number; confidence: number; headline?: string; summary?: string; bullets: string[];
  why?: string; coverage?: { observed_share?: number } | null;
}
export interface PartSummary {
  id: string; label: string; until: number; duration: number; clock_offset: number; view: string;
  teams: Record<"A" | "B", TeamStats & { formation: string | null; formations: { from_minute: number; formation: string }[] }>;
  goals: { timestamp: number; minute: string; team: "A" | "B"; scorer: string | null; assist: string | null }[];
  cards: { timestamp: number; minute: string; team: "A" | "B"; player: string | null; card: "red" | "yellow" }[];
  substitutions: { timestamp: number; minute: string; team: "A" | "B"; on: string; off: string }[];
  highlights: Highlight[];
  chapters: { team: "A" | "B"; label: string; start: number; end: number }[];
}
export interface PlayerLine {
  id: string; name: string; team: "A" | "B"; shirt?: number; position?: string; passes: number; completed: number;
  tackles: number; shots: number; goals: number; assists: number; xg: number; distance_km: number | null; top_speed_kmh: number | null;
}
export interface MatchSummary extends Omit<LobbyCard, "parts"> {
  parts: PartSummary[];
  team_stats: Record<"A" | "B", TeamStats>;
  players: PlayerLine[];
  live: { t_now: number; speed: number; finished: boolean; started?: boolean; part?: string; note: string } | null;
  until: number;
}
export const loadSummary = (id: string) => get<MatchSummary>(`/matches/${id}`);

export interface RevealSummary {
  n: number; mean_error_m: number; median_error_m: number; inside_80pct_radius: number; by_hidden_s: Record<string, number>;
}
export const loadReveals = () => get<{ summary: RevealSummary | null; reveals: import("./types").Reveal[] }>("/match/reveals");
