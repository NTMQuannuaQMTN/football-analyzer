"use client";
import { memo } from "react";
import type { MatchMeta, Timeline } from "@/lib/types";

const LABEL: Record<string, string> = { INSUFFICIENT_VISIBILITY: "Too little visible", HIGH_PRESS: "High press", LOW_BLOCK: "Low block", DOMINANT: "Dominant", BALANCED: "Balanced" };

function label(r: Timeline["A"][number]) {
  if (r.visibility?.abstain) return "INSUFFICIENT_VISIBILITY";
  if (r.press_index >= 55 && (r.def_line_oop_s ?? 0) >= 36) return "HIGH_PRESS";
  if ((r.def_line_oop_s ?? 99) <= 22) return "LOW_BLOCK";
  if ((r.territory_s ?? 0) >= 0.72) return "DOMINANT";
  return "BALANCED";
}

function LiveState({ timeline, minute, meta }: { timeline: Timeline; minute: number; meta: MatchMeta }) {
  const k = Math.min(minute, timeline.A.length - 1);
  const rows = (["A", "B"] as const).map((team) => ({ team, r: timeline[team][k] }));
  const metrics: [string, (r: Timeline["A"][number]) => string][] = [
    ["Shape", (r) => (r.formation && r.formation !== "?" ? r.formation : "n/a")],
    ["Press", (r) => r.press_index.toFixed(0)],
    ["Def. line", (r) => (r.def_line_s != null ? `${r.def_line_s.toFixed(1)}m` : "n/a")],
    ["Width", (r) => (r.width_s != null ? `${r.width_s.toFixed(1)}m` : "n/a")],
    ["Territory", (r) => `${Math.round((r.territory_s ?? 0) * 100)}%`],
  ];
  if (rows.some(({ r }) => r.visibility && r.visibility.inferred_avg > 0.05)) {
    metrics.push(["Observed / inferred", (r) => (r.visibility ? `${r.visibility.observed_avg.toFixed(0)} / ${r.visibility.inferred_avg.toFixed(0)}` : "–")]);
    metrics.push(["Inferred conf.", (r) => (r.visibility?.inferred_conf != null ? `${Math.round(r.visibility.inferred_conf * 100)}%` : "–")]);
  }
  return (
    <div className="rounded-lg bg-white/[0.03] border border-white/10 p-2.5">
      <div className="text-[11px] uppercase tracking-widest text-slate-400 mb-1.5">Live tactical state · {k + 1 + Math.floor((meta.clock_offset ?? 0) / 60)}&apos;</div>
      <table className="w-full text-xs">
        <thead>
          <tr>
            <th />
            {rows.map(({ team, r }) => (
              <th key={team} className="text-right font-semibold" style={{ color: meta.teams[team].color }}>
                {meta.teams[team].short} <span className="ml-1 px-1 rounded bg-white/10 text-white font-normal">{LABEL[label(r)]}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="font-mono">
          {metrics.map(([name, f]) => (
            <tr key={name} className="border-t border-white/5">
              <td className="font-sans text-slate-400 py-0.5">{name}</td>
              {rows.map(({ team, r }) => <td key={team} className="text-right">{f(r)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.some(({ r }) => r.visibility?.abstain) && (
        <div className="mt-1.5 text-[11px] text-amber-300">Insufficient visibility for reliable full-team tactical analysis this minute — not claiming a tactical state.</div>
      )}
    </div>
  );
}

export default memo(LiveState);
