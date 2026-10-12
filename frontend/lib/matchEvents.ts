import type { MatchEvent } from "./types";

export const SET_PIECE_TYPES = new Set(["PENALTY", "FREE_KICK", "CORNER", "GOAL_KICK"]);
export const MATCH_EVENT_TYPES = new Set([
  ...SET_PIECE_TYPES, "GOAL", "SHOT", "FOUL", "THROW_IN", "OFFSIDE", "CARD",
  "RED_CARD", "SUBSTITUTION", "SUBSTITUTION_LABEL", "KICK_OFF", "BALL_OUT",
]);

export function eventLabel(event: MatchEvent): string {
  const labels: Record<string, string> = {
    PENALTY: "Penalty", FREE_KICK: "Free kick", CORNER: "Corner", GOAL_KICK: "Goal kick", GOAL: "Goal",
    SHOT: "Shot", FOUL: "Foul", THROW_IN: "Throw-in", OFFSIDE: "Offside",
    CARD: "Card", RED_CARD: "Red card", SUBSTITUTION: "Substitution",
    SUBSTITUTION_LABEL: "Substitution", KICK_OFF: "Kick-off", BALL_OUT: "Ball out",
  };
  // Preserve the source's direct/indirect distinction without inventing an outcome.
  return event.event_type === "FREE_KICK" && typeof event.label === "string"
    ? event.label : labels[event.event_type] ?? event.event_type.replaceAll("_", " ").toLowerCase();
}

export function eventBasis(event: MatchEvent): string {
  if (event.source === "soccernet_label") return "SoccerNet annotation";
  if (event.source) return String(event.source).replaceAll("_", " ");
  return "Match event feed";
}
