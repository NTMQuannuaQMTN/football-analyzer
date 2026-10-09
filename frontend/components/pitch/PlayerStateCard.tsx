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
  const [st, setSt] = useState<any>(null);
  useEffect(() => {
    let alive = true;
    loadPlayerState(pid, sec, mode === "truth" ? "replay" : mode).then((s) => alive && setSt(s)).catch(() => {});
    return () => { alive = false; };
  }, [pid, sec, mode]);
  if (!st || st.on_pitch === false) return null;
  const pl = meta.players.find((p) => p.id === pid);
  const inferred = st.state_source === "inferred";
  const h = st.history ?? {};
  const off = meta.clock_offset ?? 0;
  return (
    <div className="pointer-events-auto bg-black/80 backdrop-blur rounded-lg px-3 py-2 text-xs w-72 border border-white/10">
      <div className="flex items-center gap-2">
        <b className="text-sm" style={{ color: pl ? meta.teams[pl.team].color : undefined }}>#{pl?.shirt} {pl?.name}</b>
        <span className={`ml-auto px-1.5 rounded text-[10px] font-bold ${inferred ? "bg-amber-400 text-black" : "bg-lime-300 text-black"}`}>{inferred ? "ESTIMATED" : "OBSERVED"}</span>
        <button onClick={onClose} className="text-slate-400 hover:text-white">✕</button>
      </div>
      {inferred ? (
        <div className="mt-1.5 space-y-0.5 text-slate-200">
          <div>Estimated position ({st.position.x}, {st.position.y}) — the camera is not showing him.</div>
          <div>Confidence <b className="font-mono">{Math.round(st.confidence * 100)}%</b> (P(error ≤ 5 m)) · uncertainty <b className="font-mono">±{st.uncertainty_radius} m</b></div>
          {h.last_seen_t != null && <div>Last observed {fmtClock(h.last_seen_t + off)} ({h.seconds_since_seen}s ago) at ({h.last_seen_position?.[0]}, {h.last_seen_position?.[1]})</div>}
          {h.next_seen_t != null && <div>Next observed {fmtClock(h.next_seen_t + off)} (used retrospectively)</div>}
          <div className="text-slate-400">Evidence: {(st.evidence ?? []).map((e: string) => EVIDENCE_LABEL[e] ?? e).join(", ")}</div>
          {st.estimator && <div className="text-slate-500">Estimator: {st.estimator}</div>}
        </div>
      ) : (
        <div className="mt-1.5 text-slate-300">Observed by the camera at ({st.position.x}, {st.position.y}). On camera {Math.round((h.observed_share_so_far ?? 0) * 100)}% of the match so far.</div>
      )}
    </div>
  );
}
