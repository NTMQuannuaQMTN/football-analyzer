"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { API, loadLobby, type LobbyCard } from "@/lib/api";

const KIND_STYLE: Record<LobbyCard["kind"], string> = {
  tracking: "bg-sky-500/20 text-sky-200",
  computer_vision: "bg-violet-500/20 text-violet-200",
  synthetic: "bg-amber-400/20 text-amber-200",
};

function MatchCard({ c }: { c: LobbyCard }) {
  const live = c.status === "live";
  return (
    <Link href={`/match/${c.id}`}
      className={`group block rounded-xl border p-4 transition-colors bg-[#0b1222] hover:bg-[#101a30] ${live ? "border-rose-500/60" : "border-white/10 hover:border-white/25"}`}>
      <div className="flex items-center gap-2 text-[11px] text-slate-400">
        {live ? (
          <span className="flex items-center gap-1.5 font-black tracking-widest text-rose-400">
            <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />LIVE {c.minute}&apos;
          </span>
        ) : (
          <span className="font-bold tracking-widest text-slate-300">{c.kind === "computer_vision" && c.parts.length < 2 ? "CLIP" : "FT"}</span>
        )}
        <span>·</span>
        <span className="truncate">{c.competition}{c.matchday ? ` · Matchday ${c.matchday}` : ""}</span>
        <span className="ml-auto font-mono whitespace-nowrap">{c.date ?? "N/A"}</span>
      </div>
      <div className="mt-3 space-y-2">
        {(["A", "B"] as const).map((tm) => (
          <div key={tm} className="flex items-center gap-3">
            <span className="w-3 h-3 rounded-sm shrink-0" style={{ background: c.teams[tm].color }} />
            <span className="flex-1 font-semibold text-lg truncate">{c.teams[tm].name}</span>
            <span className={`font-mono text-2xl font-bold ${c.score_known !== false && c.score[tm] > c.score[tm === "A" ? "B" : "A"] ? "text-white" : "text-slate-400"}`}>{c.score_known === false ? "?" : c.score[tm]}</span>
          </div>
        ))}
      </div>
      <div className="mt-3 flex items-center gap-2 flex-wrap">
        <span className={`px-2 py-0.5 rounded text-[10px] font-bold tracking-wider ${KIND_STYLE[c.kind]}`}>{c.label.toUpperCase()}</span>
        {c.parts.length > 1 && <span className="text-[11px] text-slate-500">{c.parts.map((p) => p.label).join(" · ")}</span>}
        <span className="ml-auto text-xs text-lime-300 opacity-0 group-hover:opacity-100 transition-opacity">Highlights & analysis →</span>
      </div>
      {c.note && <div className="mt-2 text-[11px] text-slate-500">{c.note}</div>}
    </Link>
  );
}

export default function Lobby() {
  const [cards, setCards] = useState<LobbyCard[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let id: ReturnType<typeof setInterval> | undefined;
    const load = () => loadLobby().then((c) => {
      setCards(c);
      // refresh the live minute / score only while a match is live (every poll is a server call)
      if (c.some((x) => x.status === "live")) id ??= setInterval(load, 15000);
      else if (id) { clearInterval(id); id = undefined; }
    }).catch((e) => setErr(String(e)));
    load();
    return () => { if (id) clearInterval(id); };
  }, []);
  if (err && !cards) return <div className="p-8 text-rose-300">Could not reach the intelligence API at {API}. Start the backend first.<br />{err}</div>;
  const live = (cards ?? []).filter((c) => c.status === "live");
  const past = (cards ?? []).filter((c) => c.status !== "live");
  return (
    <main className="min-h-screen bg-[#060a13] text-slate-100">
      <header className="h-14 flex items-center gap-4 px-6 border-b border-white/10 bg-gradient-to-r from-[#0b1222] to-[#060a13]">
        <span className="font-black tracking-widest text-lime-300">MATCH INSIGHTS</span>
        <span className="text-[11px] uppercase tracking-widest text-slate-500">Lobby</span>
      </header>
      <div className="max-w-6xl mx-auto px-6 py-6 space-y-8">
        {!cards && <div className="text-slate-400 animate-pulse">Loading matches…</div>}
        {live.length > 0 && (
          <section>
            <h2 className="text-xs uppercase tracking-widest text-rose-400 font-bold mb-3">Live now</h2>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">{live.map((c) => <MatchCard key={c.id} c={c} />)}</div>
          </section>
        )}
        {past.length > 0 && (
          <section>
            <h2 className="text-xs uppercase tracking-widest text-slate-400 font-bold mb-3">Past matches</h2>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">{past.map((c) => <MatchCard key={c.id} c={c} />)}</div>
          </section>
        )}
        {cards && (
          <p className="text-[11px] text-slate-600 max-w-3xl">
            Real tracking: DFL / IDSSE open data (CC BY 4.0). Broadcast-derived data: existing SoccerNet assets and SkillCorner partial tracking (MIT).
            The live feed replays a recorded match. Broadcast estimates do not provide independent hidden-player ground truth.
          </p>
        )}
      </div>
    </main>
  );
}
