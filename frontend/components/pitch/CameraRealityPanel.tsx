"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { loadReveals, type RevealSummary } from "@/lib/api";
import type { Frame, MatchMeta, Reveal } from "@/lib/types";

/** "What the camera couldn't see" + prediction -> reality, for matches with a camera view and ground truth. */
export default function CameraRealityPanel({ meta, frame, cameraOn, hideInferred, onCamera, onHideInferred, onPlayReveal, active, maxT = Infinity, video = false }: {
  meta: MatchMeta;
  frame: Frame;
  cameraOn: boolean;
  hideInferred: boolean;
  onCamera: (on: boolean) => void;
  onHideInferred: (h: boolean) => void;
  onPlayReveal: (r: Reveal) => void;
  active: Reveal | null;
  maxT?: number; // live match: only reappearances that have already happened
  video?: boolean; // broadcast-video match: "actual" is the CV detection, not ground truth
}) {
  const [data, setData] = useState<{ summary: RevealSummary | null; reveals: Reveal[] } | null>(null);
  useEffect(() => { loadReveals().then(setData).catch(() => setData({ summary: null, reveals: [] })); }, []);
  const names = useMemo(() => Object.fromEntries(meta.players.map((p) => [p.id, `#${p.shirt} ${p.name}`])), [meta.players]);
  const clicks = useRef(0); // deterministic rotation through examples (repeatable demo)
  const obs = (frame.obs ?? []).filter((o) => o === true).length;
  const all = (frame.obs ?? []).filter((o) => o != null).length;

  const pick = (kind: "typical" | "hard" | "random") => {
    const rs = (data?.reveals ?? []).filter((r) => r.t + 5 <= maxT);
    if (!rs.length) return;
    let pool = rs;
    if (kind === "typical") pool = rs.filter((r) => r.hidden_s >= 10 && r.hidden_s < 30);
    if (kind === "hard") pool = rs.filter((r) => r.hidden_s >= 30);
    if (!pool.length) pool = rs;
    const k = clicks.current++;
    if (kind === "random") return onPlayReveal(pool[(k * 7919 + 13) % pool.length]);
    // not cherry-picked: an example whose error is close to the median of its group
    const errs = pool.map((r) => r.error_m).sort((a, b) => a - b);
    const med = errs[Math.floor(errs.length / 2)];
    const near = pool.filter((r) => Math.abs(r.error_m - med) <= 0.5);
    const choice = near.length ? near : pool;
    onPlayReveal(choice[k % choice.length]);
  };

  return (
    <div className="pointer-events-auto w-[300px] rounded-lg bg-black/75 backdrop-blur p-3 text-xs space-y-2 border border-white/10">
      <div className="text-[10px] uppercase tracking-widest text-amber-300 font-bold">What the camera couldn&apos;t see</div>
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-black text-white">{all ? Math.round((100 * obs) / all) : 0}%</span>
        <span className="text-slate-300">camera coverage · {obs}/{all} players on screen{frame.cut ? " · CAMERA CUT" : ""}
          {video && !frame.cam && <span className="block text-amber-300/90">No pitch view this second (replay or close-up)</span>}</span>
      </div>
      <div className="flex gap-2">
        <button onClick={() => onCamera(!cameraOn)} className={`flex-1 py-1 rounded font-semibold ${cameraOn ? "bg-white/20" : "bg-white/10 hover:bg-white/20"}`}>{cameraOn ? "Hide camera view" : "Show camera view"}</button>
        <button onClick={() => onHideInferred(!hideInferred)} className={`flex-1 py-1 rounded font-bold ${hideInferred ? "bg-lime-300 text-black" : "bg-white/10 hover:bg-white/20"}`}>
          {hideInferred ? "Reconstruct hidden players" : "Show camera only"}
        </button>
      </div>
      <div className="border-t border-white/10 pt-2">
        <div className="text-[10px] uppercase tracking-widest text-amber-300 font-bold mb-1">Prediction → reality</div>
        {data?.summary ? (
          <div className="text-slate-300 mb-1.5">
            {data.summary.n.toLocaleString()} reappearances {video ? "in this clip" : "this half"}. Live prediction error: median <b className="text-white">{data.summary.median_error_m} m</b>;
            the 80% radius contained the player <b className="text-white">{Math.round(data.summary.inside_80pct_radius * 100)}%</b> of the time.
          </div>
        ) : <div className="text-slate-500 mb-1.5">No reappearance tests for this match.</div>}
        <div className="flex gap-1.5">
          {(["typical", "hard", "random"] as const).map((k) => (
            <button key={k} disabled={!data?.reveals.length} onClick={() => pick(k)} className="flex-1 py-1 rounded bg-amber-300 text-black font-bold disabled:opacity-40 capitalize">
              {k === "hard" ? "Long hidden" : k === "random" ? "Any" : k}
            </button>
          ))}
        </div>
        {active && (
          <div className="mt-2 rounded bg-white/5 p-2 space-y-0.5 font-mono">
            <div className="font-sans text-white font-semibold">{names[active.player_id]} · hidden {active.hidden_s}s</div>
            <div>PREDICTED ({active.predicted[0]}, {active.predicted[1]}) · conf {Math.round(active.confidence * 100)}% · ±{active.radius_m} m</div>
            <div>{video ? "SEEN IN VIDEO" : "ACTUAL\u00a0\u00a0\u00a0"} ({active.actual[0]}, {active.actual[1]})</div>
            <div className="text-amber-300 font-bold">ERROR {active.error_m} m · {active.inside_radius ? "inside" : "outside"} the 80% radius</div>
          </div>
        )}
        <div className="mt-1.5 text-[10px] text-slate-500">Prediction = the live model at the moment of reappearance, using only what came before. Typical = an example near the median error.
          {video && " On video the 'actual' position is the computer-vision detection and player identities come from the line-up (often wrong), so errors are an upper bound; the radius is widened 2.67x for video."}</div>
      </div>
    </div>
  );
}
