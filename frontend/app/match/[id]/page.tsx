"use client";
import Link from "next/link";
import { use, useEffect, useMemo, useState } from "react";
import { API, loadSummary, type Highlight, type MatchSummary, type PlayerLine } from "@/lib/api";

type Team = "A" | "B";
const TYPE: Record<string, { label: string; cls: string; group: string }> = {
  GOAL_BUILD_UP: { label: "GOAL", cls: "bg-white text-black", group: "Goals" },
  TACTICAL_TURNING_POINT: { label: "TACTICAL SHIFT", cls: "bg-lime-300/20 text-lime-200", group: "Tactics" },
  FORMATION_SHIFT: { label: "SHAPE CHANGE", cls: "bg-sky-400/20 text-sky-200", group: "Tactics" },
  PRESSURE_BREAK: { label: "PRESS BROKEN", cls: "bg-amber-400/20 text-amber-200", group: "Pressing" },
  COUNTERATTACK: { label: "COUNTER", cls: "bg-orange-500/20 text-orange-200", group: "Counters" },
  SUBSTITUTION_IMPACT: { label: "SUBSTITUTION", cls: "bg-violet-500/20 text-violet-200", group: "Changes" },
  RED_CARD_IMPACT: { label: "RED CARD", cls: "bg-rose-600/30 text-rose-200", group: "Changes" },
};
const FILTERS = ["All", "Goals", "Tactics", "Pressing", "Counters", "Changes"];

function StatRow({ label, a, b, fmt = (v: number) => String(v), colors }: {
  label: string; a: number | null; b: number | null; fmt?: (v: number) => string; colors: [string, string];
}) {
  if (a == null || b == null) return null;
  const tot = a + b || 1;
  return (
    <div>
      <div className="flex text-sm">
        <span className="w-16 font-mono font-bold">{fmt(a)}</span>
        <span className="flex-1 text-center text-[11px] uppercase tracking-wider text-slate-400">{label}</span>
        <span className="w-16 text-right font-mono font-bold">{fmt(b)}</span>
      </div>
      <div className="mt-1 h-1.5 rounded overflow-hidden flex bg-white/5">
        <div style={{ width: `${(a / tot) * 100}%`, background: colors[0] }} />
        <div className="flex-1" />
        <div style={{ width: `${(b / tot) * 100}%`, background: colors[1] }} />
      </div>
    </div>
  );
}

function HighlightCard({ h, s, part, partLabel }: { h: Highlight; s: MatchSummary; part: string; partLabel?: string }) {
  const [open, setOpen] = useState(false);
  const ty = TYPE[h.type] ?? { label: h.type, cls: "bg-white/10", group: "Tactics" };
  const team = s.teams[h.team];
  return (
    <div className="rounded-lg border border-white/10 bg-[#0b1222] p-3">
      <div className="flex items-center gap-2 text-xs">
        <span className="font-mono text-lime-300 font-bold w-10">{h.minute_label}</span>
        <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold tracking-wider ${ty.cls}`}>{ty.label}</span>
        <span className="flex items-center gap-1 text-slate-300"><span className="w-2 h-2 rounded-sm" style={{ background: team.color }} />{team.short}</span>
        {partLabel && <span className="text-slate-500">{partLabel}</span>}
        <Link href={`/match/${s.id}/studio?part=${part}&moment=${h.id}`}
          className="ml-auto px-2 py-1 rounded bg-lime-300 text-black font-bold hover:bg-lime-200">▶ Watch</Link>
      </div>
      <div className="mt-2 font-semibold">{h.title}</div>
      {(h.why || h.summary) && <p className="mt-1 text-sm text-slate-300">{h.why ?? h.summary}</p>}
      {h.bullets.length > 0 && (
        <>
          <button onClick={() => setOpen((o) => !o)} className="mt-2 text-[11px] text-slate-400 hover:text-white">{open ? "Hide" : "Show"} evidence ({h.bullets.length})</button>
          {open && <ul className="mt-1 text-xs text-slate-400 list-disc pl-5 space-y-0.5">{h.bullets.map((b, i) => <li key={i}>{b}</li>)}</ul>}
        </>
      )}
      <div className="mt-2 text-[10px] text-slate-500" title="Importance and evidence scores are heuristic rankings, not calibrated probabilities of accuracy">importance {Math.round(h.importance * 100)} · evidence strength {s.kind === "computer_vision" ? "N/A" : `${Math.round(h.confidence * 100)}/100`}</div>
    </div>
  );
}

function Leaders({ players, s }: { players: PlayerLine[]; s: MatchSummary }) {
  const cats: [string, (p: PlayerLine) => number | null, (v: number) => string][] = [
    ["Passes completed", (p) => p.completed, (v) => String(v)],
    ["Tackles won", (p) => p.tackles, (v) => String(v)],
    ["Expected goals", (p) => p.xg, (v) => v.toFixed(2)],
    ["Distance", (p) => p.distance_km, (v) => `${v.toFixed(1)} km`],
    ["Top speed", (p) => p.top_speed_kmh, (v) => `${v.toFixed(1)} km/h`],
  ];
  return (
    <div className="space-y-3">
      {cats.map(([label, f, fmt]) => {
        const top = players.filter((p) => f(p) != null && (f(p) as number) > 0).sort((a, b) => (f(b) as number) - (f(a) as number)).slice(0, 3);
        if (!top.length) return null;
        return (
          <div key={label}>
            <div className="text-[11px] uppercase tracking-wider text-slate-400 mb-1">{label}</div>
            {top.map((p) => (
              <div key={p.id} className="flex items-center gap-2 text-sm">
                <span className="w-2 h-2 rounded-sm" style={{ background: s.teams[p.team].color }} />
                <span className="flex-1 truncate">{p.name} <span className="text-slate-500 text-xs">#{p.shirt}</span></span>
                <span className="font-mono">{fmt(f(p) as number)}</span>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}

export default function MatchPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [s, setS] = useState<MatchSummary | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [filter, setFilter] = useState("All");
  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | undefined;
    const load = () => loadSummary(id).then((d) => {
      setS(d);
      if (d.status === "live" && !timer) timer = setInterval(load, 20000);
    }).catch((e) => setErr(String(e)));
    load();
    return () => clearInterval(timer);
  }, [id]);
  const colors: [string, string] = s ? [s.teams.A.color, s.teams.B.color] : ["#fff", "#fff"];
  const all = useMemo(() => (s ? s.parts.flatMap((p) => p.highlights.map((h) => ({ h, part: p.id, partLabel: s.parts.length > 1 ? p.label : undefined }))) : []), [s]);
  if (err && !s) return <div className="p-8 text-rose-300">Could not load this match from {API}.<br />{err}</div>;
  if (!s) return <div className="p-8 text-slate-300 animate-pulse bg-[#060a13] min-h-screen">Analysing the whole match…</div>;
  const ts = s.team_stats;
  const shown = all.filter(({ h }) => filter === "All" || (TYPE[h.type]?.group ?? "Tactics") === filter);
  const goals = s.parts.flatMap((p) => p.goals);
  const cards = s.parts.flatMap((p) => p.cards);
  const subs = s.parts.flatMap((p) => p.substitutions);
  const live = s.status === "live";
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  const firstPart = s.live?.part ?? s.parts[0]?.id;
  return (
    <main className="min-h-screen bg-[#060a13] text-slate-100">
      <header className="h-14 flex items-center gap-4 px-6 border-b border-white/10 bg-gradient-to-r from-[#0b1222] to-[#060a13]">
        <Link href="/" className="text-sm text-slate-400 hover:text-white">← Lobby</Link>
        <span className="font-black tracking-widest text-lime-300">MATCH INSIGHTS</span>
        <span className="text-[11px] uppercase tracking-widest text-slate-500">Highlights & analysis</span>
      </header>

      {/* scoreboard */}
      <section className="border-b border-white/10 bg-[#0a1020]">
        <div className="max-w-6xl mx-auto px-6 py-6">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            {live ? <span className="flex items-center gap-1.5 font-black tracking-widest text-rose-400"><span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />LIVE {s.minute}&apos;</span>
              : <span className="font-bold tracking-widest text-slate-300">{s.kind === "computer_vision" && s.parts.length < 2 ? "CLIP" : "FULL TIME"}</span>}
            <span>·</span><span>{s.competition}{s.matchday ? ` · Matchday ${s.matchday}` : ""}{s.season ? ` · ${s.season}` : ""}</span>
            <span className="ml-auto font-mono">{s.date ?? "N/A"}</span>
          </div>
          <div className="mt-4 grid grid-cols-[1fr_auto_1fr] items-start gap-6">
            {(["A", "B"] as Team[]).map((tm, i) => (
              <div key={tm} className={`${i === 1 ? "order-3 text-right" : ""}`}>
                <div className={`flex items-center gap-3 text-2xl font-bold ${i === 1 ? "justify-end" : ""}`}>
                  {i === 0 && <span className="w-4 h-4 rounded-sm" style={{ background: s.teams[tm].color }} />}
                  {s.teams[tm].name}
                  {i === 1 && <span className="w-4 h-4 rounded-sm" style={{ background: s.teams[tm].color }} />}
                </div>
                <div className="mt-2 text-sm text-slate-300 space-y-0.5">
                  {goals.filter((g) => g.team === tm).map((g, k) => <div key={k}>⚽ {g.scorer ?? "Goal"} <span className="font-mono text-slate-500">{g.minute}</span></div>)}
                  {cards.filter((c) => c.team === tm && c.card === "red").map((c, k) => <div key={k}><span className="inline-block w-2 h-3 bg-rose-600 rounded-[1px] mr-1 align-middle" />{c.player} <span className="font-mono text-slate-500">{c.minute}</span></div>)}
                </div>
              </div>
            ))}
            <div className="order-2 text-center">
              <div className="font-mono text-5xl font-black">{s.score_known === false ? "? - ?" : `${s.score.A} – ${s.score.B}`}</div>
              <div className="mt-3 flex gap-2 justify-center flex-wrap">
                <Link href={`/match/${s.id}/studio${firstPart ? `?part=${firstPart}` : ""}`} className="px-3 py-1.5 rounded bg-lime-300 text-black font-bold text-sm hover:bg-lime-200">
                  {live ? "● Watch live in studio" : "Open match studio"}
                </Link>
                {s.parts.length > 1 && s.parts.map((p) => (
                  <Link key={p.id} href={`/match/${s.id}/studio?part=${p.id}`} className="px-3 py-1.5 rounded bg-white/10 text-sm hover:bg-white/20">{p.label}</Link>
                ))}
              </div>
            </div>
          </div>
          {(s.note || s.live) && <div className="mt-4 text-[11px] text-slate-500">{s.live?.note ?? s.note}{live ? ` · analysis up to ${s.minute}'` : ""}</div>}
        </div>
      </section>

      <div className="max-w-6xl mx-auto px-6 py-6 grid gap-6 lg:grid-cols-[1fr_360px]">
        {/* highlights */}
        <section>
          <div className="flex items-center gap-2 mb-3">
            <h2 className="text-xs uppercase tracking-widest text-slate-300 font-bold">Highlights</h2>
            <span className="text-xs text-slate-500">{all.length} key moments</span>
            <div className="ml-auto flex rounded overflow-hidden border border-white/10 text-[11px]">
              {FILTERS.map((f) => (
                <button key={f} onClick={() => setFilter(f)} className={`px-2 py-1 ${filter === f ? "bg-lime-300 text-black font-bold" : "hover:bg-white/10"}`}>{f}</button>
              ))}
            </div>
          </div>
          <div className="space-y-3">
            {s.kind === "computer_vision" && <p className="text-xs text-amber-200/90">Recorded match events and reconstructed sequences have different evidence. Build-up descriptions depend on unverified player positions and identities.</p>}
            {shown.length === 0 && <div className="text-sm text-slate-500">No {filter.toLowerCase()} moments{live ? " yet" : ""}.</div>}
            {shown.map(({ h, part, partLabel }) => <HighlightCard key={`${part}-${h.id}`} h={h} s={s} part={part} partLabel={partLabel} />)}
          </div>
        </section>

        {/* analysis */}
        <aside className="space-y-6">
          <div className="rounded-lg border border-white/10 bg-[#0b1222] p-4 space-y-3">
            <div className="flex text-xs font-bold">
              <span style={{ color: colors[0] }}>{s.teams.A.short}</span>
              <span className="flex-1 text-center uppercase tracking-widest text-slate-400">Team stats</span>
              <span style={{ color: colors[1] }}>{s.teams.B.short}</span>
            </div>
            {s.kind === "computer_vision" && <p className="text-[11px] leading-relaxed text-amber-200/90">Recorded events are mixed with model estimates. Possession, passes, pressure and player totals depend on unverified reconstruction; these are not official match statistics.</p>}
            <StatRow label="Possession" a={ts.A.possession} b={ts.B.possession} fmt={pct} colors={colors} />
            <StatRow label="Shots" a={ts.A.shots} b={ts.B.shots} colors={colors} />
            <StatRow label="On target" a={ts.A.shots_on_target} b={ts.B.shots_on_target} colors={colors} />
            <StatRow label="Expected goals" a={ts.A.xg} b={ts.B.xg} fmt={(v) => v.toFixed(2)} colors={colors} />
            <StatRow label="Passes" a={ts.A.passes} b={ts.B.passes} colors={colors} />
            <StatRow label="Pass accuracy" a={ts.A.pass_accuracy} b={ts.B.pass_accuracy} fmt={pct} colors={colors} />
            <StatRow label="Tackles won" a={ts.A.tackles_won} b={ts.B.tackles_won} colors={colors} />
            <StatRow label="Press intensity" a={ts.A.press_index} b={ts.B.press_index} colors={colors} />
            <StatRow label="Distance (km)" a={ts.A.distance_km} b={ts.B.distance_km} fmt={(v) => v.toFixed(1)} colors={colors} />
            <StatRow label="Corners" a={ts.A.corners} b={ts.B.corners} colors={colors} />
            <StatRow label="Fouls" a={ts.A.fouls} b={ts.B.fouls} colors={colors} />
            <StatRow label="Yellow cards" a={ts.A.yellow_cards} b={ts.B.yellow_cards} colors={colors} />
          </div>

          <div className="rounded-lg border border-white/10 bg-[#0b1222] p-4">
            <div className="text-xs uppercase tracking-widest text-slate-400 font-bold mb-2">Shape</div>
            {s.kind === "computer_vision" && <p className="text-[11px] text-amber-200/90 mb-2">Estimated from reconstructed positions; calibration and identities are unverified.</p>}
            {s.parts.map((p) => (
              <div key={p.id} className="mb-3 last:mb-0">
                {s.parts.length > 1 && <div className="text-[11px] text-slate-500 mb-1">{p.label}</div>}
                {(["A", "B"] as Team[]).map((tm) => (
                  <div key={tm} className="flex items-center gap-2 text-sm mb-1">
                    <span className="w-10 font-bold" style={{ color: s.teams[tm].color }}>{s.teams[tm].short}</span>
                    <span className="font-mono font-bold">{p.teams[tm].formation ?? "–"}</span>
                    <span className="text-[11px] text-slate-500 truncate">mostly; also {[...new Set(p.teams[tm].formations.map((f) => f.formation))].filter((f) => f !== p.teams[tm].formation).slice(0, 3).join(", ") || "–"}</span>
                  </div>
                ))}
              </div>
            ))}
            <div className="text-[10px] text-slate-500 mt-1">Read from the back line forward, goalkeeper excluded.</div>
          </div>

          {s.parts.some((p) => p.chapters.length) && (
            <div className="rounded-lg border border-white/10 bg-[#0b1222] p-4">
              <div className="text-xs uppercase tracking-widest text-slate-400 font-bold mb-2">Tactical chapters</div>
              {s.parts.flatMap((p) => p.chapters.map((c, k) => (
                <div key={`${p.id}-${k}`} className="flex items-center gap-2 text-sm">
                  <span className="font-mono text-slate-500 w-20">{Math.floor((c.start + p.clock_offset) / 60)}&apos;–{Math.ceil((c.end + p.clock_offset) / 60)}&apos;</span>
                  <span style={{ color: s.teams[c.team].color }} className="font-bold w-10">{s.teams[c.team].short}</span>
                  <span>{({ HIGH_PRESS: "High press", LOW_BLOCK: "Low block", DOMINANT: "Dominance" } as Record<string, string>)[c.label] ?? c.label}</span>
                </div>
              )))}
            </div>
          )}

          <div className="rounded-lg border border-white/10 bg-[#0b1222] p-4">
            <div className="text-xs uppercase tracking-widest text-slate-400 font-bold mb-2">Top players</div>
            {s.kind === "computer_vision" && <p className="text-[11px] leading-relaxed text-amber-200/90 mb-2">Names follow unverified role assignments. Individual totals can belong to a different visible player.</p>}
            <Leaders players={s.players} s={s} />
          </div>

          {(subs.length > 0 || cards.length > 0) && (
            <div className="rounded-lg border border-white/10 bg-[#0b1222] p-4 text-sm space-y-1">
              <div className="text-xs uppercase tracking-widest text-slate-400 font-bold mb-2">Cards & substitutions</div>
              {[...cards.map((c) => ({ t: c.timestamp + 0, m: c.minute, team: c.team, txt: <><span className={`inline-block w-2 h-3 rounded-[1px] mr-1 align-middle ${c.card === "red" ? "bg-rose-600" : "bg-yellow-400"}`} />{c.player}</> })),
                ...subs.map((x) => ({ t: x.timestamp, m: x.minute, team: x.team, txt: <>⇅ {x.on} <span className="text-slate-500">for {x.off}</span></> }))]
                .sort((a, b) => parseInt(a.m) - parseInt(b.m))
                .map((r, k) => (
                  <div key={k} className="flex gap-2"><span className="font-mono text-slate-500 w-9">{r.m}</span><span className="w-2 h-2 mt-1.5 rounded-sm" style={{ background: s.teams[r.team].color }} /><span>{r.txt}</span></div>
                ))}
            </div>
          )}
        </aside>
      </div>
    </main>
  );
}
