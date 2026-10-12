"use client";
import { useMemo } from "react";
import { fmtClock } from "@/lib/api";
import { eventBasis, eventLabel, MATCH_EVENT_TYPES } from "@/lib/matchEvents";
import type { MatchEvent, MatchMeta } from "@/lib/types";

export default function MatchEvents({ events, meta, onSeek, onClose }: {
  events: MatchEvent[]; meta: MatchMeta; onSeek: (t: number) => void; onClose: () => void;
}) {
  const rows = useMemo(() => events.filter(e => MATCH_EVENT_TYPES.has(e.event_type))
    .sort((a, b) => a.timestamp - b.timestamp), [events]);
  return <div role="dialog" aria-label="Match events" className="pointer-events-auto rounded-lg border border-slate-600 bg-slate-950/95 shadow-xl p-3 text-xs w-[560px] max-w-full">
    <div className="flex items-center justify-between mb-2"><h2 className="font-bold text-white">Match events</h2>
      <button type="button" onClick={onClose} className="rounded border border-slate-600 px-2 py-1">Close events</button></div>
    <p className="text-slate-400 mb-2">Jump to a recorded event. A penalty label does not establish its outcome or taker. Event timing and projected player positions are separate evidence.</p>
    <div className="max-h-[min(50vh,400px)] overflow-auto">
      <table className="w-full border-collapse text-left">
        <thead className="sticky top-0 bg-slate-950"><tr>{["Clock", "Event", "Team", "Source"].map(label => <th key={label} className="p-2 border-b border-slate-600">{label}</th>)}</tr></thead>
        <tbody>{rows.map(event => <tr key={event.id} className="border-b border-white/10">
          <td className="p-2 font-mono"><button type="button" onClick={() => onSeek(event.timestamp)} className="text-lime-300 underline" aria-label={`Jump to ${eventLabel(event)} at ${fmtClock(event.timestamp + (meta.clock_offset ?? 0))}`}>{fmtClock(event.timestamp + (meta.clock_offset ?? 0))}</button></td>
          <td className="p-2">{eventLabel(event)}</td>
          <td className="p-2">{event.team_id ? meta.teams[event.team_id].short : "N/A"}</td>
          <td className="p-2 text-slate-400">{eventBasis(event)}</td>
        </tr>)}</tbody>
      </table>
      {!rows.length && <p className="p-2 text-slate-400">No recorded events in the available window.</p>}
    </div>
  </div>;
}
