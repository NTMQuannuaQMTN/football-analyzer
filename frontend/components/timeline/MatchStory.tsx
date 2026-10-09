"use client";
import { memo } from "react";
import type { MatchMeta, Moment, Story, Timeline } from "@/lib/types";

const SHORT: Record<string, string> = {
  TACTICAL_TURNING_POINT: "SHIFT", FORMATION_SHIFT: "SHAPE", GOAL_BUILD_UP: "GOAL", PRESSURE_BREAK: "BREAK",
  COUNTERATTACK: "COUNTER", SUBSTITUTION_IMPACT: "SUB", RED_CARD_IMPACT: "RED",
};
const CHAPTER: Record<string, string> = { HIGH_PRESS: "High press", LOW_BLOCK: "Low block", DOMINANT: "Dominance" };

interface Props {
  story: Story;
  moments: Moment[];
  timeline: Timeline;
  meta: MatchMeta;
  t: number;
  selectedId: string | null;
  onSelect: (m: Moment) => void;
  onSeek: (t: number) => void;
}

// smooth curve through points (Catmull-Rom as cubic Béziers)
function smoothPath(pts: [number, number][]) {
  if (pts.length < 2) return "";
  let d = `M${pts[0][0]},${pts[0][1]}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)];
    const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
    const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
    d += ` C${c1[0]},${c1[1]} ${c2[0]},${c2[1]} ${p2[0]},${p2[1]}`;
  }
  return d;
}

function MatchStory({ story, moments, timeline, meta, t, selectedId, onSelect, onSeek }: Props) {
  const D = meta.duration;
  const pct = (t: number) => `${(100 * t) / D}%`;
  const tickStep = D > 1800 ? 15 : D > 600 ? 5 : 1;
  const ticks = Array.from({ length: Math.floor(D / 60 / tickStep) + 1 }, (_, i) => i * tickStep);
  const byId = Object.fromEntries(moments.map((m) => [m.id, m]));
  const sel = selectedId ? byId[selectedId] : null;
  const linked = new Set<string>();
  if (sel) {
    story.edges.forEach((e) => {
      if (e.from === sel.id) linked.add(e.to);
      if (e.to === sel.id) linked.add(e.from);
    });
  }
  // lanes: each moment (marker + label to its right) takes the nearest free row around the axis, so moments in the
  // same minute sit one above another and neighbouring labels never overlap
  const LANES = [0, -1, 1, -2, 2];
  const span = D * 0.085; // seconds of axis one marker + label occupies (~8.5% of the width)
  const lane: Record<string, number | null> = {};
  const used: { t: number; lane: number }[] = [];
  [...moments].sort((a, b) => a.timestamp - b.timestamp).forEach((m) => {
    const free = LANES.find((l) => !used.some((u) => u.lane === l && m.timestamp - u.t < span)) ?? null;
    lane[m.id] = free;
    if (free != null) used.push({ t: m.timestamp, lane: free });
  });
  // pressing momentum: lightly smoothed (3-minute average) and drawn as a curve
  const momentum = (team: "A" | "B") => {
    const v = timeline[team].map((r) => r.press_index);
    const pts = v.map((_, k) => {
      const w = v.slice(Math.max(0, k - 1), k + 2);
      return [((k + 0.5) * 60 * 1000) / D, 40 - (w.reduce((a, b) => a + b, 0) / w.length / 100) * 36] as [number, number];
    });
    return smoothPath(pts);
  };

  return (
    <div className="h-full flex flex-col px-4 pt-2 pb-1">
      <div className="flex items-center gap-3 text-[11px] uppercase tracking-widest text-slate-400">
        <span className="text-white font-bold">Match story</span>
        <span>tactical chapters · pressing momentum · key moments · causal links</span>
        <span className="ml-auto flex gap-3 normal-case tracking-normal">
          {(["A", "B"] as const).map((t) => (
            <span key={t} className="flex items-center gap-1"><span className="w-3 h-0.5 inline-block" style={{ background: meta.teams[t].color }} />{meta.teams[t].short} press intensity</span>
          ))}
        </span>
      </div>
      <div
        className="relative flex-1 mt-1 cursor-crosshair"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          onSeek(((e.clientX - r.left) / r.width) * D);
        }}
      >
        {/* chapters */}
        {story.chapters.map((c, i) => (
          <div key={i} className="absolute h-4 rounded-sm text-[10px] px-1 overflow-hidden whitespace-nowrap font-semibold"
            style={{ left: pct(c.start), width: pct(c.end - c.start), top: c.team === "A" ? 0 : "auto", bottom: c.team === "B" ? 0 : "auto",
              background: `${meta.teams[c.team].color}33`, color: meta.teams[c.team].color, borderLeft: `2px solid ${meta.teams[c.team].color}` }}>
            {meta.teams[c.team].short} {CHAPTER[c.label] ?? c.label}
          </div>
        ))}

        {/* momentum + causal arcs */}
        <svg className="absolute left-0 right-0 top-5 h-[46px] w-full" viewBox="0 0 1000 42" preserveAspectRatio="none">
          <path d={momentum("A")} fill="none" stroke={meta.teams.A.color} strokeWidth={1.4} opacity={0.8} vectorEffect="non-scaling-stroke" />
          <path d={momentum("B")} fill="none" stroke={meta.teams.B.color} strokeWidth={1.4} opacity={0.8} vectorEffect="non-scaling-stroke" />
          {story.edges.map((e, i) => {
            const a = byId[e.from], b = byId[e.to];
            if (!a || !b) return null;
            const x1 = (a.timestamp / D) * 1000, x2 = (b.timestamp / D) * 1000;
            const hot = sel && (e.from === sel.id || e.to === sel.id);
            const h = Math.min(38, 8 + Math.abs(x2 - x1) * 0.25);
            return (
              <path key={i} d={`M${x1},42 Q${(x1 + x2) / 2},${42 - h * 2} ${x2},42`} fill="none"
                stroke={hot ? "#d9f99d" : "rgba(255,255,255,0.25)"} strokeWidth={hot ? 2 : 1} strokeDasharray={hot ? undefined : "3 3"} vectorEffect="non-scaling-stroke" />
            );
          })}
        </svg>

        {/* axis */}
        <div className="absolute left-0 right-0 top-[72px] h-px bg-white/25" />
        {ticks.map((mn) => (
          <div key={mn} className="absolute top-[76px] text-[10px] font-mono text-slate-500 -translate-x-1/2" style={{ left: pct(mn * 60) }}>{mn + Math.floor((meta.clock_offset ?? 0) / 60)}&apos;</div>
        ))}

        {/* playhead */}
        <div className="absolute top-0 bottom-0 w-px bg-lime-300/80 pointer-events-none" style={{ left: pct(Math.min(D, Math.max(0, t))) }} />

        {/* moments */}
        {moments.map((m) => {
          const size = 8 + m.importance * 8;
          const isSel = m.id === selectedId;
          const isLinked = linked.has(m.id);
          const ln = lane[m.id];
          const top = 72 + (ln ?? 0) * 19;
          return (
            <button key={m.id} title={`${m.minute_label} ${m.title}`}
              onClick={(e) => { e.stopPropagation(); onSelect(m); }}
              className="absolute -translate-x-1/2 -translate-y-1/2 group" style={{ left: pct(m.timestamp), top, zIndex: isSel ? 2 : 1 }}>
              <span className={`flex items-center justify-center rounded-full border-2 text-[9px] font-bold transition-transform group-hover:scale-125 ${isSel ? "scale-125 ring-2 ring-lime-300" : ""} ${isLinked ? "ring-2 ring-lime-200/60" : ""}`}
                style={{ width: size, height: size, background: m.type === "GOAL_BUILD_UP" ? "#fff" : meta.teams[m.team].color, borderColor: meta.teams[m.team].color, color: "#000" }}>
              </span>
              {(ln != null || isSel) && <span className={`absolute top-1/2 -translate-y-1/2 whitespace-nowrap text-[10px] font-bold tracking-wider ${isSel ? "text-lime-300" : "text-slate-300"}`}
                style={{ left: size + 3 }}>
                {SHORT[m.type]} <span style={{ color: meta.teams[m.team].color }}>{meta.teams[m.team].short}</span> <span className="font-mono text-slate-500">{m.minute_label}</span>
              </span>}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default memo(MatchStory);
