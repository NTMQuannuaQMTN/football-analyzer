"use client";
import { memo, useState } from "react";
import type { MatchMeta, Moment } from "@/lib/types";
import { fmtClock } from "@/lib/api";
import WhatIf from "./WhatIf";

export const TYPE_META: Record<string, { label: string; icon: string }> = {
  TACTICAL_TURNING_POINT: { label: "Tactical turning point", icon: "⚡" },
  FORMATION_SHIFT: { label: "Formation shift", icon: "⇄" },
  GOAL_BUILD_UP: { label: "Goal build-up", icon: "⚽" },
  PRESSURE_BREAK: { label: "Pressure break", icon: "↗" },
  COUNTERATTACK: { label: "Counterattack", icon: "»" },
  SUBSTITUTION_IMPACT: { label: "Substitution impact", icon: "⇅" },
  RED_CARD_IMPACT: { label: "Red card impact", icon: "■" },
};

const STAGE_COLOR: Record<string, string> = {
  TACTICAL_CHANGE: "bg-violet-500/20 text-violet-200 border-violet-400/40",
  EVENT: "bg-slate-500/20 text-slate-200 border-slate-400/40",
  STATE_CHANGE: "bg-sky-500/20 text-sky-200 border-sky-400/40",
  CONSEQUENCE: "bg-amber-500/20 text-amber-200 border-amber-400/40",
  OUTCOME: "bg-lime-500/20 text-lime-200 border-lime-400/40",
};

const DELTA_LABEL: Record<string, [string, string]> = {
  press_intensity: ["Press intensity", ""],
  def_line_oop_m: ["Defensive line", "m"],
  def_line_m: ["Defensive line", "m"],
  territory_pct: ["Territory", " pts"],
  possession_pct: ["Possession", " pts"],
  final_third_recoveries: ["Final-third recoveries", ""],
  shots: ["Shots", ""],
  xg: ["xG", ""],
  width_m: ["Team width", "m"],
  depth_m: ["Team depth", "m"],
  right_press_share_pct: ["Right-side press share", " pts"],
  opp_possession_pct: ["Opp. possession", " pts"],
  opp_territory_pct: ["Opp. territory", " pts"],
  opp_shots: ["Opp. shots", ""],
  opp_xg: ["Opp. xG", ""],
  opp_final_third_recoveries: ["Opp. final-third recov.", ""],
  team_def_line_m: ["Own defensive line", "m"],
  team_width_m: ["Own width", "m"],
  team_depth_m: ["Own depth", "m"],
};

function Badge({ m, meta }: { m: Moment; meta: MatchMeta }) {
  const tm = TYPE_META[m.type] ?? { label: m.type, icon: "•" };
  return (
    <div className="flex items-center gap-2 text-[11px] uppercase tracking-widest">
      <span className="px-2 py-0.5 rounded bg-lime-300 text-black font-bold">{tm.icon} {tm.label}</span>
      <span className="font-mono text-slate-400">{m.minute_label} · {fmtClock(m.timestamp + (meta.clock_offset ?? 0))}</span>
      <span className="ml-auto font-semibold" style={{ color: meta.teams[m.team].color }}>{meta.teams[m.team].short}</span>
    </div>
  );
}

/** What the camera couldn't see: the team's shape from camera-only players vs the reconstruction vs ground truth. */
function HiddenStateBlock({ m, meta }: { m: Moment; meta: MatchMeta }) {
  const h = m.hidden_state;
  if (!h) return null;
  const rows: [string, "def_line_m" | "width_m" | "near_ball", string][] = [
    ["Defensive line (from own goal)", "def_line_m", " m"], ["Team width", "width_m", " m"], ["Players within 15 m of ball", "near_ball", ""],
  ];
  const v = (k: "observed_only" | "reconstructed" | "truth", r: "def_line_m" | "width_m" | "near_ball") => h.metrics[k]?.[r];
  return (
    <div className="rounded-lg border border-amber-300/30 bg-amber-300/5 p-2.5">
      <div className="text-[11px] uppercase tracking-widest text-amber-300 font-bold">What the camera couldn&apos;t see</div>
      <div className="text-xs text-slate-300 mt-1">
        Around this moment the camera showed <b className="text-white">{h.observed_players}</b> of {meta.teams[h.team].name}&apos;s {h.outfield_players} outfield
        players; <b className="text-white">{h.inferred_players}</b> were reconstructed ({h.high_confidence_inferred} with confidence ≥ 50%).
      </div>
      <table className="w-full mt-2 text-xs font-mono">
        <thead><tr className="text-slate-400 text-[10px] uppercase"><th className="text-left font-normal">±5 s average</th><th className="font-normal">Camera only</th><th className="font-normal text-amber-200">Reconstructed</th><th className="font-normal">Actual</th></tr></thead>
        <tbody>
          {rows.map(([label, k, unit]) => (
            <tr key={k}>
              <td className="font-sans text-slate-300 py-0.5">{label}</td>
              <td className="text-center">{v("observed_only", k) != null ? `${v("observed_only", k)}${unit}` : "–"}</td>
              <td className="text-center text-amber-200 font-bold">{v("reconstructed", k)}{unit}</td>
              <td className="text-center">{v("truth", k)}{unit}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="text-[10px] text-slate-500 mt-1">Camera only = from the players on screen; reconstructed = with hidden players estimated; actual = tracking ground truth.</div>
    </div>
  );
}

function AnalystView({ m, meta }: { m: Moment; meta: MatchMeta }) {
  const [showJson, setShowJson] = useState(false);
  const deltas = Object.entries(m.deltas ?? {}).filter(([k]) => DELTA_LABEL[k]);
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-xl font-bold leading-tight">{m.title}</h2>
        <p className="text-sm text-slate-300 mt-1">{m.narrative.why_it_matters}</p>
        <div className="flex gap-4 mt-2 text-xs text-slate-400">
          <span>Importance <b className="text-white font-mono">{Math.round(m.importance * 100)}</b></span>
          <span>Confidence <b className="text-white font-mono">{Math.round(m.confidence * 100)}%</b></span>
          {m.coverage && m.coverage.inferred_avg > 0.05 && (
            <span title="Average players of this team on camera vs estimated during the moment">
              Coverage <b className="text-white font-mono">{m.coverage.observed_avg.toFixed(0)}/{(m.coverage.observed_avg + m.coverage.inferred_avg).toFixed(0)}</b> observed
            </span>
          )}
        </div>
        {m.narrative.visibility_note && <p className="text-[11px] text-amber-200/90 mt-1">ⓘ {m.narrative.visibility_note}</p>}
      </div>

      <HiddenStateBlock m={m} meta={meta} />

      {deltas.length > 0 && (
        <div className="grid grid-cols-2 gap-1.5">
          {deltas.map(([k, v]) => {
            const [label, unit] = DELTA_LABEL[k];
            return (
              <div key={k} className="bg-white/5 rounded px-2 py-1.5">
                <div className="text-[10px] uppercase tracking-wider text-slate-400">{label}</div>
                <div className={`font-mono text-lg font-semibold ${v > 0 ? "text-lime-300" : v < 0 ? "text-rose-300" : "text-white"}`}>
                  {v > 0 ? "+" : ""}{v}{unit}
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div>
        <div className="text-[11px] uppercase tracking-widest text-slate-400 mb-1.5">Tactic → outcome</div>
        <div className="flex flex-wrap items-center gap-1">
          {m.chain.map((c, i) => (
            <span key={i} className="flex items-center gap-1">
              <span className={`text-[11px] px-1.5 py-0.5 rounded border ${STAGE_COLOR[c.stage]}`} title={c.stage}>{c.label}</span>
              {i < m.chain.length - 1 && <span className="text-slate-500">→</span>}
            </span>
          ))}
        </div>
      </div>

      <div>
        <div className="text-[11px] uppercase tracking-widest text-slate-400 mb-1.5">Why did this moment matter?</div>
        <ol className="space-y-1.5">
          {m.evidence.map((e, i) => (
            <li key={i} className="flex gap-2 text-sm">
              <span className="font-mono text-lime-300 w-4 shrink-0">{i + 1}.</span>
              <span>
                <b className="text-white">{e.label}</b>
                <span className="text-slate-300"> — {e.detail}</span>
                <span className="text-slate-500 font-mono text-[11px]"> @{fmtClock(e.timestamp + (meta.clock_offset ?? 0))}</span>
                {e.basis && e.basis !== "observed events" && (
                  <span className={`ml-1 text-[10px] px-1 rounded ${e.basis === "coverage" ? "bg-sky-500/20 text-sky-200" : "bg-amber-500/20 text-amber-200"}`}>{e.basis === "coverage" ? "visibility" : "uses inferred positions"}</span>
                )}
              </span>
            </li>
          ))}
        </ol>
        <div className="text-[11px] text-slate-500 mt-1">Every figure is computed by the analytics engine from tracking + event data.</div>
      </div>

      {m.counterfactual && <WhatIf cf={m.counterfactual} meta={meta} />}

      <div>
        <button onClick={() => setShowJson((s) => !s)} className="text-[11px] uppercase tracking-widest text-slate-400 hover:text-white">
          {showJson ? "▾" : "▸"} Machine-readable insight (overlay JSON)
        </button>
        {showJson && (
          <pre className="mt-1 text-[10px] leading-tight bg-black/50 rounded p-2 max-h-64 overflow-auto text-slate-300">
            {JSON.stringify({ id: m.id, timestamp: m.timestamp, type: m.type, team: m.team, importance: m.importance, confidence: m.confidence, title: m.title, before: m.before, after: m.after, evidence: m.evidence, outcome: m.outcome, chain: m.chain, narrative: m.narrative, links: m.links }, null, 1)}
          </pre>
        )}
      </div>
    </div>
  );
}

function FanView({ m, meta }: { m: Moment; meta: MatchMeta }) {
  const stats = m.narrative.fan_stats ?? [];
  return (
    <div className="space-y-4">
      <div className="rounded-lg p-4 bg-gradient-to-br from-white/10 to-transparent border border-white/10">
        <div className="text-2xl font-black leading-tight tracking-tight" style={{ color: meta.teams[m.team].color }}>
          {m.narrative.fan.headline}
        </div>
        <p className="text-base mt-2 text-slate-100">{m.narrative.fan.text}</p>
      </div>
      <div className="grid grid-cols-3 gap-2">
        {stats.map((e, i) => (
          <div key={i} className="text-center bg-white/5 rounded p-2">
            <div className="text-2xl font-black font-mono">{e.value}</div>
            <div className="text-[10px] uppercase tracking-wider text-slate-400">{e.label}</div>
          </div>
        ))}
      </div>
      <div className="text-[11px] uppercase tracking-widest text-slate-400">Broadcast lower third</div>
      <div className="flex items-stretch rounded overflow-hidden text-sm font-semibold">
        <div className="px-2 py-1.5 bg-lime-300 text-black">{m.minute_label}</div>
        <div className="px-3 py-1.5 bg-slate-900 border border-white/10 flex-1">{m.narrative.broadcast.lower_third}</div>
      </div>
      {m.counterfactual && <WhatIf cf={m.counterfactual} meta={meta} compact />}
    </div>
  );
}

function MomentPanel({ moment, meta, mode, live }: { moment: Moment | null; meta: MatchMeta; mode: "analyst" | "fan"; live: boolean }) {
  if (!moment) return <div className="text-slate-400 text-sm">No key moments yet. The engine is watching…</div>;
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-[11px] uppercase tracking-widest">
        {live ? <span className="text-rose-400 font-bold">● Latest insight</span> : <span className="text-lime-300 font-bold">▶ Selected moment</span>}
      </div>
      <Badge m={moment} meta={meta} />
      {mode === "analyst" ? <AnalystView m={moment} meta={meta} /> : <FanView m={moment} meta={meta} />}
    </div>
  );
}

export default memo(MomentPanel);
