"use client";
import { useEffect, useState } from "react";
import { fmtClock, loadPlayerState } from "@/lib/api";
import type { MatchMeta } from "@/lib/types";

/* eslint-disable @typescript-eslint/no-explicit-any */
const EVIDENCE_LABEL: Record<string, string> = {
  previous_position: "previous trajectory", team_shape: "team shape", ball_position: "ball position",
  player_role: "player role", future_reappearance: "next sighting (retrospective)", velocity: "last velocity", observed: "camera",
};

export default function PlayerStateCard({ pid, sec, mode, meta, onClose }: { pid: string; sec: number; mode: string; meta: MatchMeta; onClose: () => void }) {
  const requestMode = mode === "truth" ? "replay" : mode;
  const [response, setResponse] = useState<{ pid: string; sec: number; mode: string; data: any } | null>(null);
  useEffect(() => {
    let alive = true;
    loadPlayerState(pid, sec, requestMode).then((data) => alive && setResponse({ pid, sec, mode: requestMode, data })).catch(() => {});
    return () => { alive = false; };
  }, [pid, sec, requestMode]);
  // A seek or mode/player change must not retain a later state's position or
  // history while the replacement request is pending. Brief past lag is safe.
  const st = response && response.pid === pid && response.mode === requestMode && response.sec <= sec && sec - response.sec <= 1 ? response.data : null;
  if (!st || st.on_pitch === false) return null;
  const pl = meta.players.find((p) => p.id === pid);
  const inferred = st.state_source === "inferred";
  const h = st.history ?? {};
  const off = meta.clock_offset ?? 0;
  return (
    <div className="pointer-events-auto bg-black/80 backdrop-blur rounded-lg px-3 py-2 text-xs w-72 border border-white/10">
      <div className="flex items-center gap-2">
        <b className="text-sm" style={{ color: pl ? meta.teams[pl.team].color : undefined }}>#{pl?.shirt} {pl?.name}</b>
        <span className={`ml-auto px-1.5 rounded text-[10px] font-bold ${inferred ? "bg-amber-400 text-black" : "bg-lime-300 text-black"}`}>{inferred ? "ESTIMATED" : meta.source?.type === "computer_vision" ? "PROJECTED" : "OBSERVED"}</span>
        <button onClick={onClose} className="text-slate-400 hover:text-white">✕</button>
      </div>
      {inferred ? (
        <div className="mt-1.5 space-y-0.5 text-slate-200">
          <div>Estimated position ({st.position.x}, {st.position.y}) — not identified in this frame.</div>
          <div>Confidence <b className="font-mono">{st.confidence == null ? "N/A" : `${Math.round(st.confidence * 100)}%`}</b> · {meta.source?.type === "computer_vision" ? "model radius" : "uncertainty"} <b className="font-mono">{st.uncertainty_radius == null ? "N/A" : `±${st.uncertainty_radius} m`}</b></div>
          {h.last_seen_t != null && <div>Last observed {fmtClock(h.last_seen_t + off)} ({h.seconds_since_seen}s ago) at ({h.last_seen_position?.[0]}, {h.last_seen_position?.[1]})</div>}
          {h.next_seen_t != null && <div>Next observed {fmtClock(h.next_seen_t + off)} (used retrospectively)</div>}
          <div className="text-slate-400">Evidence: {(st.evidence ?? []).map((e: string) => EVIDENCE_LABEL[e] ?? e).join(", ")}</div>
          {st.estimator && <div className="text-slate-500">Estimator: {st.estimator}</div>}
        </div>
      ) : (
        <div className="mt-1.5 text-slate-300">{meta.source?.type === "computer_vision" ? "Projected measurement" : "Observed position"} at ({st.position.x}, {st.position.y}). Measured {Math.round((h.observed_share_so_far ?? 0) * 100)}% of the timeline so far.
          {st.observation_timestamp != null && <div>Measurement time {fmtClock(st.observation_timestamp + off)}.</div>}</div>
      )}
      {meta.source?.type === "computer_vision" && <div className="text-[10px] text-amber-200 mt-1">{meta.source?.provider === "SkillCorner" ? "Identity and measured positions come from SkillCorner broadcast tracking (provider computer vision), not independently verified. Hidden positions are our model estimates." : "Lineup identity is assigned by role; shirt number is unverified. Camera projection may contain error. An estimate can be inside the view when detection or identity is missing."}</div>}
      {st.measurement_diagnostic?.status === "review_camera_or_identity" && <div className="text-[10px] text-amber-200 mt-1">Review camera or identity: recent projected measurements imply {st.measurement_diagnostic.apparent_speed_m_s} m/s over {st.measurement_diagnostic.elapsed_s}s. This is a measurement conflict, not verified player speed.</div>}
    </div>
  );
}
