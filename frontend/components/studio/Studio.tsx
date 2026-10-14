"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import Pitch from "@/components/pitch/Pitch";
import MomentPanel from "@/components/insights/MomentPanel";
import LiveState from "@/components/insights/LiveState";
import PlayerPanel from "@/components/personalization/PlayerPanel";
import MatchStory from "@/components/timeline/MatchStory";
import MatchEvents from "@/components/studio/MatchEvents";
import { eventLabel, MATCH_EVENT_TYPES } from "@/lib/matchEvents";
import AccuracyPanel from "@/components/insights/AccuracyPanel";
import WhatIf from "@/components/insights/WhatIf";
import PlayerStateCard from "@/components/pitch/PlayerStateCard";
import CameraRealityPanel from "@/components/pitch/CameraRealityPanel";
import Link from "next/link";
import { API, fmtClock, loadFrames, loadMatch, setScope, type MatchBundle } from "@/lib/api";
import { useReplayClock } from "@/lib/useReplayClock";
import type { Frame, MatchEvent, Mode, Moment, Reveal, TeamId, ViewMode } from "@/lib/types";

const SPEEDS = [1, 4, 8, 16, 32, 64];
const EMPTY_FRAMES: Frame[] = [];
const FEED_TYPES = new Set([...MATCH_EVENT_TYPES, "TACKLE"]);

function eventLine(e: MatchEvent, names: Record<string, string>) {
  const who = e.player_id ? names[e.player_id] : "";
  switch (e.event_type) {
    case "GOAL": return `⚽ GOAL — ${who}`;
    case "SHOT": {
      const details = [typeof e.outcome === "string" ? e.outcome.replaceAll("_", " ") : null,
        typeof e.xg === "number" ? `xG ${e.xg}` : null].filter(Boolean);
      return `Shot${who ? ` — ${who}` : ""}${details.length ? ` (${details.join(", ")})` : ""}`;
    }
    case "SUBSTITUTION": return `⇅ ${names[e.player_on_id as string]} on for ${names[e.player_off_id as string]}`;
    case "RED_CARD": return `■ Red card — ${who}`;
    case "TACKLE": return `Tackle won — ${who}`;
    case "OFFSIDE": return `⚑ Offside — ${who ?? "flag"}${e.margin_m != null ? ` (${e.margin_m}m)` : ""}`;
    default: return `${eventLabel(e)}${who ? ` — ${who}` : ""}`;
  }
}

export interface StudioProps {
  match?: string;
  part?: string;
  startT?: number;
  momentId?: string;
}

/** Loads one match (part) and opens the studio on it; `match` / `part` scope every API call. */
export default function StudioLoader({ match, part, startT, momentId }: StudioProps) {
  const [data, setData] = useState<MatchBundle | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setScope(match, part);
    loadMatch().then(setData).catch((e) => setErr(String(e)));
  }, [match, part]);
  if (err) return <div className="p-8 text-rose-300">Could not reach the intelligence API at {API}. Start the backend first.<br />{err}</div>;
  if (!data) return <div className="p-8 text-slate-300 animate-pulse">Loading match & running intelligence pipeline…</div>;
  if (data.meta.live && data.meta.live.started === false) {
    return (
      <div className="min-h-screen bg-[#060a13] text-slate-200 p-8 space-y-3">
        <div className="text-lg font-bold">This half has not started yet.</div>
        <Link href={`/match/${match}/studio?part=${data.meta.live.part}`} className="inline-block px-3 py-1.5 rounded bg-rose-600 font-bold text-sm">● Watch the live half</Link>
      </div>
    );
  }
  return <Studio key={`${match}-${part}`} data={data} match={match} part={part} startT={startT} momentId={momentId} />;
}

function Studio({ data, match, part, startT, momentId }: { data: MatchBundle } & StudioProps) {
  const { meta } = data;
  // live match: only what has happened so far (the clock edge advances in real time)
  const live = meta.live && !meta.live.finished ? meta.live : null;
  const [edge, setEdge] = useState(live ? live.t_now : meta.duration);
  useEffect(() => {
    if (!live) return;
    const t0 = performance.now();
    const id = setInterval(() => setEdge(Math.min(meta.duration, live.t_now + ((performance.now() - t0) / 1000) * live.speed)), 1000);
    return () => clearInterval(id);
  }, [live, meta.duration]);
  const cut = live ? Math.floor(edge / 10) * 10 : Infinity;
  // live: a moment appears once all of its evidence lies before the clock edge (available_at), not at its start
  const moments = useMemo(() => data.moments.filter((m) => (m.available_at ?? m.timestamp) <= cut), [data.moments, cut]);
  const events = useMemo(() => data.events.filter((e) => e.timestamp <= cut), [data.events, cut]);
  const story = useMemo(() => {
    if (!live) return data.story;
    const ids = new Set(moments.map((m) => m.id));
    return { ...data.story, edges: data.story.edges.filter((e) => ids.has(e.from) && ids.has(e.to)),
      chapters: data.story.chapters.filter((c) => c.start <= cut).map((c) => ({ ...c, end: Math.min(c.end, cut) })) };
  }, [data.story, moments, cut, live]);
  const timeline = useMemo(() => live
    ? { A: data.timeline.A.slice(0, Math.floor(cut / 60) + 1), B: data.timeline.B.slice(0, Math.floor(cut / 60) + 1) }
    : data.timeline, [data.timeline, cut, live]);
  const isVod = meta.view === "vod";
  // ground truth is the default whenever the match has it; Replay / Live show what a broadcast camera would give
  const hasTruth = meta.view === "truth";
  const hasVod = isVod || !!meta.vod_available;
  const defaultView: ViewMode = hasTruth ? "truth" : "replay";
  const [viewMode, setViewMode] = useState<ViewMode>(defaultView);
  const [frameCache, setFrameCache] = useState<Partial<Record<ViewMode, Frame[]>>>({ [defaultView]: data.frames });
  const [frameErrors, setFrameErrors] = useState<Partial<Record<ViewMode, string>>>({});
  const [showUncertainty, setShowUncertainty] = useState(false);
  const [showWhatIf, setShowWhatIf] = useState(false);
  const [showEvents, setShowEvents] = useState(false);
  const [showTruth, setShowTruth] = useState(false);
  useEffect(() => {
    const need: ViewMode[] = [viewMode, ...(showTruth ? (["truth"] as ViewMode[]) : [])];
    need.filter((m) => !frameCache[m]).forEach((m) => loadFrames(m).then((f) => setFrameCache((c) => ({ ...c, [m]: f }))).catch(() => setFrameErrors((e) => ({ ...e, [m]: "Positions could not load." }))));
  }, [viewMode, showTruth, frameCache]);
  const frames = frameCache[viewMode] ?? EMPTY_FRAMES;
  const hasPredictionZones = useMemo(() => frames.some(frame => frame.rad?.some(radius => radius != null && radius > 0)), [frames]);
  const clock = useReplayClock(live ? Math.max(1, edge) : meta.duration);
  const { t, playing, setPlaying, speed, setSpeed, seek, playWindow, stopAt } = clock;
  // live: follow the edge at 1x until the user scrubs or replays something
  const [follow, setFollow] = useState(!!live);
  useEffect(() => {
    if (!follow) return;
    if (t < edge - 3) seek(edge - 1);
    setSpeed(1);
    if (!playing && t < edge) setPlaying(true);
  }, [follow, edge, t, playing, seek, setSpeed, setPlaying]);
  const goLive = () => { setSelected(null); setFollow(true); seek(edge - 1); };
  const [mode, setMode] = useState<Mode>("analyst");
  const [selected, setSelected] = useState<Moment | null>(null);
  const [player, setPlayer] = useState<string | null>(null);
  const [showLines, setShowLines] = useState(true);
  const [showShape, setShowShape] = useState(true);
  // "what the camera couldn't see" + prediction -> reality (real-tracking matches with a camera view)
  const [showCamPanel, setShowCamPanel] = useState(false);
  const [cameraOn, setCameraOn] = useState(true);
  const [hideInferred, setHideInferred] = useState(false);
  const [activeReveal, setActiveReveal] = useState<Reveal | null>(null);
  const [liveT, setLiveT] = useState<number | null>(null);

  const lastMinute = Math.max(0, Math.ceil(meta.duration / 60) - 1);
  const minute = Math.min(lastMinute, Math.floor(t / 60));
  const offset = meta.clock_offset ?? 0;
  const isCV = meta.source?.type === "computer_vision";
  const isTracking = meta.source?.type === "tracking";
  const transferredCamera = isTracking && meta.source?.camera?.includes("moved onto");
  const names = useMemo(() => Object.fromEntries(meta.players.map((p) => [p.id, p.name])), [meta.players]);
  const score = useMemo(() => {
    const s = { ...(meta.initial_score ?? { A: 0, B: 0 }) };
    meta.goals.forEach((g) => { if (g.timestamp <= t && g.team_id) s[g.team_id]++; });
    return s;
  }, [meta.goals, meta.initial_score, t]);
  const sec = Math.floor(t);
  const fr = frames[Math.min(frames.length - 1, Math.floor(t))];
  const latest = useMemo(() => [...moments].reverse().find((m) => m.timestamp <= sec) ?? null, [moments, sec]);
  const shown = selected ?? latest;
  const inReplay = selected != null && stopAt != null;
  const feed = useMemo(() => events.filter((e) => FEED_TYPES.has(e.event_type) && e.timestamp <= sec).slice(-6).reverse(), [events, sec]);

  const replay = useCallback((m: Moment) => {
    setFollow(false);
    if (selected == null) setLiveT(t);
    setSelected(m);
    playWindow(m.replay.start, m.replay.end, m.replay.end - m.replay.start > 40 ? 2 : 1);
  }, [playWindow, selected, t]);

  const backToLive = () => {
    if (live) return goLive();
    setSelected(null);
    if (liveT != null) seek(liveT);
    setLiveT(null);
    setSpeed(8);
    setPlaying(true);
  };

  useEffect(() => {
    // deep link from the highlights page: play that moment (or jump to a time) once loaded
    const m = momentId ? data.moments.find((x) => x.id === momentId) : null;
    const id = setTimeout(() => (m ? replay(m) : startT != null ? seek(startT) : undefined), 0);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const [cardPlayer, setCardPlayer] = useState<string | null>(null);
  const onPlayerClick = (pid: string) => { setPlayer(pid); setCardPlayer(pid); setMode("player"); };

  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if (e.code === "Space" && (e.target as HTMLElement).tagName !== "SELECT") { e.preventDefault(); setPlaying((p) => !p); }
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [setPlaying]);

  const poss = fr?.p;
  const nObs = (fr?.obs ?? []).filter((o) => o === true).length;
  const nInf = (fr?.obs ?? []).filter((o) => o === false).length;
  const possRow = timeline.A[minute];
  const shapeNow = Object.fromEntries((["A", "B"] as TeamId[]).map((tm) => {
    const r = timeline[tm][Math.min(minute, timeline[tm].length - 1)];
    return [tm, { name: r?.formation ?? "?", lines: r?.formation_lines ?? [] }];
  })) as Record<TeamId, { name: string; lines: string[][] }>;

  return (
    <main className="h-screen flex flex-col bg-[#060a13] text-slate-100 overflow-hidden">
      {/* Scoreboard */}
      <header className="min-h-14 shrink-0 flex flex-wrap items-center gap-x-4 gap-y-1 px-4 py-1 border-b border-white/10 bg-gradient-to-r from-[#0b1222] to-[#060a13]">
        {match && <Link href={`/match/${match}`} className="text-xs text-slate-400 hover:text-white whitespace-nowrap">← Highlights</Link>}
        <Link href="/" className="text-xs text-slate-400 hover:text-white">Lobby</Link>
        {live ? (
          <button onClick={goLive} title={live.note} className={`px-2 py-0.5 rounded text-xs font-black tracking-widest ${follow ? "bg-rose-600" : "bg-white/10 text-rose-300 hover:bg-rose-600/40"}`}>● {follow ? "LIVE" : "GO LIVE"}</button>
        ) : (
          <span className={`shrink-0 whitespace-nowrap px-2 py-0.5 rounded text-xs font-black tracking-widest ${inReplay ? "bg-lime-300 text-black" : "bg-slate-700"}`}>{inReplay ? "REPLAY" : isCV ? (meta.duration < 2400 ? "CLIP REPLAY" : "MATCH REPLAY") : "FULL MATCH"}</span>
        )}
        <div className="flex shrink-0 items-center gap-3 text-lg font-bold">
          <span className="flex items-center gap-2"><span className="w-3 h-3 rounded-sm" style={{ background: meta.teams.A.color }} />{meta.teams.A.name}{poss === "A" && <span className="text-[10px] text-lime-300">●</span>}</span>
          <span className="whitespace-nowrap font-mono text-2xl px-3 py-0.5 rounded bg-white/10">{meta.score_known === false ? "? - ?" : `${score.A} – ${score.B}`}</span>
          <span className="flex items-center gap-2">{poss === "B" && <span className="text-[10px] text-lime-300">●</span>}{meta.teams.B.name}<span className="w-3 h-3 rounded-sm" style={{ background: meta.teams.B.color }} /></span>
        </div>
        <span className="font-mono text-2xl text-lime-300">{fmtClock(t + offset)}</span>
        {isCV && <span className="px-2 py-0.5 rounded text-[10px] font-bold tracking-widest bg-violet-500/30 text-violet-200" title={meta.source?.game}>COMPUTER VISION · POSITIONS UNVERIFIED · {meta.source?.provider ?? (meta.source?.game?.startsWith("skillcorner/") ? "SkillCorner" : "SoccerNet")} · {Math.round((meta.source?.observed_share ?? meta.source?.coverage ?? 0) * 100)}% MEASURED · REST ESTIMATED</span>}
        {(hasVod || isCV) && viewMode !== "truth" && (
          <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-white/10" title="Measured and estimated positions; missed detection does not prove a player is outside the camera">
            {fr ? <>Now: <b className="text-white">{nObs}</b>/{nObs + nInf} {isCV ? "measured" : "observed"} · <b className="text-amber-300">{nInf}</b> inferred</> : "Now: N/A"}
            {fr?.cut && <span className="ml-2 text-rose-300 font-bold">CAMERA CUT</span>}
            {hasVod && <span className="ml-2 text-[10px] text-slate-400 uppercase">{viewMode === "replay" ? "retrospective" : "causal estimate"}</span>}
          </span>
        )}
        {isTracking && <span className="px-2 py-0.5 rounded text-[10px] font-bold tracking-widest bg-sky-500/30 text-sky-200" title={meta.source?.camera ?? meta.source?.dataset}>REAL TRACKING · DFL{viewMode !== "truth" && (transferredCamera ? " · SIMULATED CAMERA" : ` · ${Math.round((meta.source?.observed_share ?? 0) * 100)}% ON CAMERA`)}</span>}
        {viewMode === "truth" && <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-amber-400 text-black">{isTracking ? "TRACKED GROUND TRUTH" : "SYNTHETIC GROUND TRUTH"}</span>}
        <div className="flex items-center gap-2 text-[11px] text-slate-400">
          <span>Replay poss (3&apos;)</span>
          <div className="w-28 h-1.5 rounded bg-white/10 overflow-hidden flex">
            <div style={{ width: `${(possRow?.possession_s ?? 0.5) * 100}%`, background: meta.teams.A.color }} />
            <div className="flex-1" style={{ background: meta.teams.B.color }} />
          </div>
          <span className="font-mono">{Math.round((possRow?.possession_s ?? 0.5) * 100)}%</span>
        </div>
        {data.parts && data.parts.length > 1 && (
          <div className="ml-auto flex rounded overflow-hidden border border-white/10 text-[11px]">
            {data.parts.map((p) => (
              <Link key={p.id} href={`/match/${match}/studio?part=${p.id}`} className={`px-2 py-1 ${p.id === (part ?? data.parts![0].id) ? "bg-lime-300 text-black font-bold" : "hover:bg-white/10"}`}>{p.label}</Link>
            ))}
          </div>
        )}
        <div className={`${data.parts && data.parts.length > 1 ? "" : "ml-auto "}text-[11px] uppercase tracking-widest text-slate-500`}>Match Insights Engine</div>
      </header>

      <div className="flex-1 min-h-0 flex">
        {/* Pitch column */}
        <section className="flex-1 min-w-0 flex flex-col p-3 gap-2">
          <div className="flex items-center gap-2 text-xs text-slate-300">
            {meta.date && <span className="mr-auto">Played {meta.date}</span>}
            {viewMode === "live" && <span title="Positions update when a sample arrives. Replay interpolates between samples." className="text-slate-400">Live · 1 s samples</span>}
            <button type="button" aria-expanded={showEvents} onClick={() => setShowEvents(v => !v)} className="rounded border border-slate-600 px-2 py-1">{showEvents ? "Hide match events" : "Match events"}</button>
            {viewMode !== "truth" && <button type="button" aria-pressed={cameraOn} onClick={() => setCameraOn(v => !v)} className="rounded border border-slate-600 px-2 py-1">{cameraOn ? "Hide camera view" : "Show camera view"}</button>}
            {(isCV || hasVod) && <button type="button" disabled={!hasPredictionZones} title={hasPredictionZones ? "Model uncertainty regions; video-domain coverage is unverified" : "This export contains no prediction radii"} aria-pressed={showUncertainty} onClick={() => setShowUncertainty(v => !v)} className="rounded border border-slate-600 px-2 py-1 disabled:opacity-40">{!hasPredictionZones ? "Prediction zones N/A" : showUncertainty ? "Hide prediction zones" : "Show prediction zones"}</button>}
            <button type="button" disabled={!shown} onClick={() => setShowWhatIf(v => !v)} className="rounded border border-slate-600 px-2 py-1 disabled:opacity-40">{showWhatIf ? "Hide explanation / what-if" : "Explain / What if"}</button>
          </div>
          <div className="flex-1 min-h-0 relative rounded-lg overflow-hidden bg-[#0b2a17]">
            {showEvents && <div className="absolute top-2 right-2 left-2 z-30 flex justify-end"><MatchEvents events={events} meta={meta} onClose={() => setShowEvents(false)} onSeek={nt => { setSelected(null); setFollow(false); seek(Math.min(nt, live ? edge : nt)); }} /></div>}
            {fr ? <Pitch t={t} frames={frames} meta={meta} events={events} causalPlayback={viewMode === "live"}
              focusPlayers={selected ? selected.replay.focus_players : []}
              focusTeam={selected ? selected.replay.focus_team : null}
              overlay={selected ? selected.replay.overlay : null}
              selectedPlayer={mode === "player" ? player : null}
              showLines={showLines} onPlayerClick={onPlayerClick}
              shape={showShape ? shapeNow : null}
              camera={viewMode !== "truth" ? { footprint: cameraOn, hideInferred } : null}
              reveal={viewMode !== "truth" ? activeReveal : null}
              showUncertainty={showUncertainty} truthFrames={showTruth && viewMode !== "truth" ? frameCache.truth ?? null : null} /> :
              <div role="status" className="h-full flex flex-col items-center justify-center gap-2 text-sm text-slate-300">
                <span>{frameErrors[viewMode] ?? `Loading ${viewMode === "live" ? "live estimates" : viewMode === "replay" ? "replay estimates" : "ground truth"}…`}</span>
                {frameErrors[viewMode] && <button className="rounded border border-slate-600 px-3 py-1" onClick={() => { setFrameErrors(e => ({ ...e, [viewMode]: undefined })); setFrameCache(c => ({ ...c })); }}>Retry</button>}
              </div>}
            {showWhatIf && shown && <div role="region" aria-label="Explanation and what-if" className="absolute z-30 top-2 bottom-2 right-2 w-[420px] max-w-[calc(100%-1rem)] rounded-lg border border-slate-600 bg-slate-950/95 flex flex-col">
              <div className="flex justify-between px-3 py-2 border-b border-white/10"><span>Explanation / What if</span><button type="button" aria-label="Close explanation" onClick={() => setShowWhatIf(false)}>✕</button></div>
              <div className="overflow-y-auto p-3"><MomentPanel moment={shown} meta={meta} mode={mode === "fan" ? "fan" : "analyst"} live={!selected} hideWhatIf />{shown.counterfactual && <WhatIf cf={shown.counterfactual} meta={meta} />}</div>
            </div>}
            {selected && (
              <div className="absolute top-2 left-2 right-2 flex items-start gap-2 pointer-events-none">
                <div className="pointer-events-auto bg-black/70 backdrop-blur rounded px-3 py-1.5 text-sm">
                  <span className="text-lime-300 font-bold mr-2">▶ {selected.minute_label}</span>{selected.title}
                  <span className="text-slate-400 ml-2 text-xs">overlay: {selected.replay.overlay}</span>
                </div>
                <button onClick={backToLive} className="pointer-events-auto ml-auto bg-rose-600 hover:bg-rose-500 rounded px-3 py-1.5 text-xs font-bold">{live ? "● Back to live" : "← Back"}</button>
              </div>
            )}
            {fr && showCamPanel && ((isTracking && hasVod) || isCV) && (
              <div className={`absolute ${selected ? "top-12" : "top-2"} left-2 pointer-events-none`}>
                <CameraRealityPanel meta={meta} frame={fr} cameraOn={cameraOn} hideInferred={hideInferred} active={activeReveal} maxT={live ? edge : Infinity} video={isCV}
                  onCamera={(on) => { setCameraOn(on); if (viewMode === "truth") setViewMode("live"); }}
                  onHideInferred={(h) => { setHideInferred(h); if (viewMode === "truth") setViewMode("live"); }}
                  onPlayReveal={(r) => {
                    setSelected(null); setFollow(false); setHideInferred(false); if (!isCV) setViewMode("live"); setActiveReveal(r);
                    playWindow(Math.max(0, r.t - Math.min(r.hidden_s, 10) - 2), r.t + 5, 1);
                  }} />
              </div>
            )}
            {fr && cardPlayer && (hasVod || isCV) && (
              <div className="absolute top-12 right-2 pointer-events-none">
                <PlayerStateCard pid={cardPlayer} sec={sec} mode={viewMode} meta={meta} onClose={() => setCardPlayer(null)} />
              </div>
            )}
            {selected && (
              <div className="absolute bottom-2 left-2 right-2 pointer-events-none flex">
                <div className="flex items-stretch rounded overflow-hidden text-sm font-semibold shadow-lg">
                  <div className="px-2 py-1.5 bg-lime-300 text-black">{selected.minute_label}</div>
                  <div className="px-3 py-1.5 bg-slate-950/90">{selected.narrative.broadcast.lower_third}</div>
                </div>
              </div>
            )}
          </div>
          {/* Controls */}
          <div className="shrink-0 flex items-center gap-3 text-sm">
            <button onClick={() => { setFollow(false); setPlaying((p) => !p); }} className="w-20 py-1 rounded bg-white/10 hover:bg-white/20 font-semibold">{playing ? "❚❚ Pause" : "▶ Play"}</button>
            <div className="flex rounded overflow-hidden border border-white/10">
              {SPEEDS.map((s) => (
                <button key={s} onClick={() => setSpeed(s)} className={`px-2 py-1 font-mono text-xs ${speed === s ? "bg-lime-300 text-black" : "hover:bg-white/10"}`}>{s}x</button>
              ))}
            </div>
            <input type="range" min={0} max={live ? Math.max(1, edge) : meta.duration} step={1} value={t} onChange={(e) => { setFollow(false); seek(Number(e.target.value)); }} className="flex-1 accent-lime-300" />
            {(isCV || (hasVod && viewMode !== "truth")) && <span className="text-[11px] text-slate-400 flex items-center gap-1"><svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="none" stroke="white" strokeDasharray="2 1.5" opacity="0.6" /></svg>inferred</span>}
            {hasVod && (
              <div className="flex rounded overflow-hidden border border-white/10 text-[11px]" title="How hidden players are estimated">
                {([["truth", "Truth"], ["replay", "Camera replay"], ["live", "Camera live"]] as [ViewMode, string][]).map(([m, l]) => (
                  <button key={m} onClick={() => setViewMode(m)} className={`px-2 py-1 ${viewMode === m ? "bg-amber-300 text-black font-bold" : "hover:bg-white/10"}`}>{l}</button>
                ))}
              </div>
            )}
            {(isVod || (hasTruth && hasVod)) && viewMode !== "truth" && <label className="flex items-center gap-1 text-xs text-slate-400" title="Yellow squares: where inferred players really were"><input type="checkbox" checked={showTruth} onChange={(e) => setShowTruth(e.target.checked)} />Error</label>}
            {((isTracking && hasVod) || isCV) && (
              <button onClick={() => { setShowCamPanel((v) => !v); if (!showCamPanel && viewMode === "truth") setViewMode("live"); if (showCamPanel) setActiveReveal(null); }}
                className={`px-2 py-1 rounded text-[11px] font-bold ${showCamPanel ? "bg-amber-300 text-black" : "bg-white/10 hover:bg-white/20 text-amber-200"}`}>
                Camera vs reality
              </button>
            )}
            <label className="flex items-center gap-1 text-xs text-slate-400"><input type="checkbox" checked={showLines} onChange={(e) => setShowLines(e.target.checked)} />Lines</label>
            <label className="flex items-center gap-1 text-xs text-slate-400" title="Detected formation: players joined line by line, back line first"><input type="checkbox" checked={showShape} onChange={(e) => setShowShape(e.target.checked)} />Shape</label>
          </div>
          <div className="shrink-0 h-[22px] flex gap-4 text-xs text-slate-400 overflow-hidden whitespace-nowrap">
            {feed.map((e) => (
              <span key={e.id}><span className="font-mono text-slate-500">{fmtClock(e.timestamp + offset)}</span> <span style={{ color: e.team_id ? meta.teams[e.team_id].color : undefined }}>{eventLine(e, names)}</span></span>
            ))}
          </div>
        </section>

        {/* Intelligence panel */}
        <aside className="w-[440px] shrink-0 border-l border-white/10 flex flex-col bg-[#080d19]">
          <div className="px-4 pt-3 pb-2 border-b border-white/10">
            <div className="text-[11px] uppercase tracking-widest text-slate-400 mb-2 font-bold">Match intelligence</div>
            <div className="grid grid-cols-4 gap-1 p-1 rounded-lg bg-white/5">
              {(["analyst", "fan", "player", "accuracy"] as Mode[]).map((m) => (
                <button key={m} onClick={() => setMode(m)} className={`py-1.5 rounded text-sm font-semibold capitalize ${mode === m ? "bg-lime-300 text-black" : "text-slate-300 hover:bg-white/10"}`}>{m}</button>
              ))}
            </div>
          </div>
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
            {(mode === "analyst" || mode === "fan") && <LiveState timeline={timeline} minute={minute} meta={meta} />}
            {mode === "accuracy" ? (
              <AccuracyPanel />
            ) : mode === "player" ? (
              <PlayerPanel meta={meta} playerId={player} throughT={sec} reconstructionMode={viewMode} onSelect={setPlayer} onMoment={replay} moments={moments} />
            ) : (
              <MomentPanel moment={shown} meta={meta} mode={mode} live={!selected} hideWhatIf />
            )}
            {(mode === "analyst" || mode === "fan") && shown && !selected && (
              <button onClick={() => replay(shown)} className="w-full py-2 rounded bg-lime-300 text-black font-bold text-sm">▶ Replay & explain this moment</button>
            )}
          </div>
        </aside>
      </div>

      {/* Match story */}
      <footer className="h-[150px] shrink-0 border-t border-white/10 bg-[#070c17]">
        <MatchStory story={story} moments={moments} events={events} timeline={timeline} meta={meta} t={t}
          selectedId={selected?.id ?? null} onSelect={replay} onSeek={(nt) => { setSelected(null); setFollow(false); seek(Math.min(nt, live ? edge : nt)); }} />
      </footer>
    </main>
  );
}
