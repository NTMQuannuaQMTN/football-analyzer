"use client";
import { useState } from "react";
import type { MatchMeta, Moment } from "@/lib/types";

type CF = NonNullable<Moment["counterfactual"]>;

export default function WhatIf({ cf, meta, compact = false }: { cf: CF; meta: MatchMeta; compact?: boolean }) {
  const [open, setOpen] = useState(!compact);
  const rows = compact ? cf.rows.filter((r) => ["territory_pct", "shots", "goals"].includes(r.key)) : cf.rows;
  return (
    <div className="rounded-lg border border-violet-400/30 bg-violet-500/5 p-3">
      <button onClick={() => setOpen((o) => !o)} className="w-full text-left">
        <div className="text-[11px] uppercase tracking-widest text-violet-300 font-bold">{open ? "▾" : "▸"} What if? <span className="normal-case tracking-normal font-normal text-violet-200/70">synthetic counterfactual (estimated)</span></div>
        <div className="text-sm font-semibold">{cf.question}</div>
      </button>
      {open && (
        <>
          <table className="w-full mt-2 text-sm">
            <thead>
              <tr className="text-[10px] uppercase tracking-wider text-slate-400">
                <th className="text-left font-normal">{meta.teams[cf.team].short} · min {cf.window_minutes[0] + 1}–{cf.window_minutes[1]}</th>
                <th className="text-right font-normal">Actual</th>
                <th className="text-right font-normal">Synthetic counterfactual</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {rows.map((r) => {
                const better = (r.actual ?? 0) > (r.counterfactual ?? 0);
                return (
                  <tr key={r.key} className="border-t border-white/5">
                    <td className="font-sans py-0.5 text-slate-300">{r.metric}</td>
                    <td className={`text-right ${better ? "text-lime-300" : "text-white"}`}>{r.actual ?? "-"}{r.actual != null ? r.unit : ""}</td>
                    <td className="text-right text-slate-400">{r.counterfactual ?? "-"}{r.counterfactual != null ? r.unit : ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {!compact && <div className="text-[10px] text-slate-500 mt-1.5">{cf.method}</div>}
        </>
      )}
    </div>
  );
}
