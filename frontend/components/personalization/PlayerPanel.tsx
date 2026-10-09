"use client";
import { memo, useEffect, useState } from "react";
import { loadPlayer } from "@/lib/api";
import type { MatchMeta, Moment, PlayerProfile } from "@/lib/types";

function PlayerPanel({ meta, playerId, minute, onSelect, onMoment, moments }: {
  meta: MatchMeta;
  playerId: string | null;
  minute: number;
  onSelect: (pid: string) => void;
  onMoment: (m: Moment) => void;
  moments: Moment[];
}) {
  const [prof, setProf] = useState<PlayerProfile | null>(null);
  useEffect(() => {
    if (!playerId) return;
    let alive = true;
    loadPlayer(playerId, Math.max(60, minute * 60 + 59)).then((p) => alive && setProf(p)).catch(() => {});
    return () => { alive = false; };
  }, [playerId, minute]);

  const s = prof?.stats;
  const tiles: [string, string | number][] = prof && s ? [
    ["Distance", `${prof.movement.distance_km} km`],
    ["Top speed", `${prof.movement.top_speed_kmh} km/h`],
    ["Sprints", `${prof.movement.sprint_seconds}s`],
    ["Passes", `${s.completed}/${s.passes}`],
    ["Progressive passes", s.progressive_passes],
    ["Pressure actions", s.pressures],
    ["Recoveries", s.recoveries],
    ["Final-third recov.", s.final_third_recoveries],
    ["Shots / xG", `${s.shots} / ${s.xg}`],
    ["Goals / assists", `${s.goals} / ${s.assists}`],
  ] : [];

  return (
    <div className="space-y-3">
      <select value={playerId ?? ""} onChange={(e) => onSelect(e.target.value)} className="w-full bg-slate-900 border border-white/15 rounded px-2 py-1.5 text-sm">
        <option value="" disabled>Select a player (or click one on the pitch)</option>
        {(["A", "B"] as const).map((team) => (
          <optgroup key={team} label={meta.teams[team].name}>
            {meta.players.filter((p) => p.team === team).map((p) => (
              <option key={p.id} value={p.id}>#{p.shirt} {p.name} · {p.position}</option>
            ))}
          </optgroup>
        ))}
      </select>
      {prof && prof.player.id === playerId && (
        <>
          <div>
            <div className="text-xl font-black tracking-tight" style={{ color: meta.teams[prof.player.team].color }}>{prof.narrative.headline}</div>
            <div className="text-xs text-slate-400">#{prof.player.shirt} · {meta.teams[prof.player.team].name} · up to {minute + 1}&apos;</div>
            {prof.visibility && prof.visibility.inferred_seconds > 0 && (
              <div className="text-[11px] text-amber-200/90 mt-1">On camera {Math.round(prof.visibility.observed_share * 100)}% of the time; movement stats include estimated off-screen positions.</div>
            )}
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            {tiles.map(([k, v]) => (
              <div key={k} className="bg-white/5 rounded px-2 py-1.5">
                <div className="text-[10px] uppercase tracking-wider text-slate-400">{k}</div>
                <div className="font-mono text-lg font-semibold">{v}</div>
              </div>
            ))}
          </div>
          <div>
            <div className="text-[11px] uppercase tracking-widest text-slate-400 mb-1">Tactical impact</div>
            <p className="text-sm">{prof.narrative.impact}</p>
          </div>
          {prof.moments.length > 0 && (
            <div>
              <div className="text-[11px] uppercase tracking-widest text-slate-400 mb-1">Involved in key moments</div>
              <div className="space-y-1">
                {prof.moments.map((pm) => {
                  const m = moments.find((x) => x.id === pm.id);
                  return (
                    <button key={pm.id} onClick={() => m && onMoment(m)} className="block w-full text-left text-sm px-2 py-1 rounded bg-white/5 hover:bg-white/10">
                      <span className="font-mono text-lime-300 mr-2">{pm.minute_label}</span>{pm.title} <span className="text-slate-500">▶ replay</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default memo(PlayerPanel);
