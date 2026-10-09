"use client";
import { memo, useMemo } from "react";
import type { Frame, MatchEvent, MatchMeta, Reveal, TeamId } from "@/lib/types";

const L = 105;
const W = 68;

type Pt = [number, number];

function lerp(a: Pt | null, b: Pt | null, f: number): Pt | null {
  if (!a) return null;
  if (!b) return a;
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
}

function hull(pts: Pt[]): Pt[] {
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o: Pt, a: Pt, b: Pt) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Pt[] = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  const upper: Pt[] = [];
  for (const q of [...p].reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

interface Props {
  t: number;
  frames: Frame[];
  meta: MatchMeta;
  events: MatchEvent[];
  focusPlayers: string[];
  focusTeam: TeamId | null;
  overlay: string | null;
  selectedPlayer: string | null;
  showLines: boolean;
  onPlayerClick: (pid: string) => void;
  showUncertainty?: boolean;
  truthFrames?: Frame[] | null; // synthetic ground truth: draw where inferred players really were
  shape?: Partial<Record<TeamId, { name: string; lines: string[][] }>> | null; // detected formation: draw its lines
  camera?: { footprint: boolean; hideInferred: boolean } | null; // "what the camera couldn't see"
  reveal?: Reveal | null; // prediction -> reality example being played
}

function PitchMarkings() {
  const stripes = Array.from({ length: 10 }, (_, i) => i);
  return (
    <g>
      {stripes.map((i) => (
        <rect key={i} x={i * 10.5} y={0} width={10.5} height={W} fill={i % 2 ? "#14532d" : "#166534"} />
      ))}
      <g fill="none" stroke="rgba(255,255,255,0.55)" strokeWidth={0.25}>
        <rect x={0} y={0} width={L} height={W} />
        <line x1={L / 2} y1={0} x2={L / 2} y2={W} />
        <circle cx={L / 2} cy={W / 2} r={9.15} />
        <rect x={0} y={W / 2 - 20.16} width={16.5} height={40.32} />
        <rect x={L - 16.5} y={W / 2 - 20.16} width={16.5} height={40.32} />
        <rect x={0} y={W / 2 - 9.16} width={5.5} height={18.32} />
        <rect x={L - 5.5} y={W / 2 - 9.16} width={5.5} height={18.32} />
        <rect x={-1.5} y={W / 2 - 3.66} width={1.5} height={7.32} />
        <rect x={L} y={W / 2 - 3.66} width={1.5} height={7.32} />
        <line x1={L / 3} y1={0} x2={L / 3} y2={W} strokeDasharray="0.6 1.2" stroke="rgba(255,255,255,0.15)" />
        <line x1={(2 * L) / 3} y1={0} x2={(2 * L) / 3} y2={W} strokeDasharray="0.6 1.2" stroke="rgba(255,255,255,0.15)" />
      </g>
      <circle cx={L / 2} cy={W / 2} r={0.4} fill="rgba(255,255,255,0.6)" />
    </g>
  );
}

const Markings = memo(PitchMarkings);

export default function Pitch({ t, frames, meta, events, focusPlayers, focusTeam, overlay, selectedPlayer, showLines, onPlayerClick, showUncertainty = true, truthFrames = null, shape = null, camera = null, reveal = null }: Props) {
  const i0 = Math.min(frames.length - 1, Math.floor(t));
  const i1 = Math.min(frames.length - 1, i0 + 1);
  const f = t - i0;
  const fr0 = frames[i0];
  const fr1 = frames[i1];
  const order = meta.player_order;
  const pmap = useMemo(() => Object.fromEntries(meta.players.map((p) => [p.id, p])), [meta.players]);

  const pos: (Pt | null)[] = order.map((_, j) => lerp(fr0.xy[j] as Pt | null, fr1.xy[j] as Pt | null, f));
  // a >40m jump in one second is a restart (ball placed on the spot), not a pass: snap instead of sliding
  const jump = Math.hypot(fr1.b[0] - fr0.b[0], fr1.b[1] - fr0.b[1]) > 40;
  const ball = jump ? fr0.b : lerp(fr0.b, fr1.b, f)!;

  const teamPts = (team: TeamId, outfield = true) =>
    order
      .map((pid, j) => ({ pid, p: pos[j] }))
      .filter(({ pid, p }) => p && pid.startsWith(team) && (!outfield || pmap[pid].position !== "GK"))
      .map(({ pid, p }) => ({ pid, p: p as Pt }));

  const defLine = (team: TeamId) => {
    const xs = teamPts(team).map(({ p }) => (team === "A" ? p[0] : L - p[0])).sort((a, b) => a - b);
    const v = xs.slice(0, 4).reduce((s, x) => s + x, 0) / Math.max(1, Math.min(4, xs.length));
    return team === "A" ? v : L - v;
  };

  // offside line: second-last defender of the team without the ball (or halfway / ball, whichever is further)
  const offsideX = (() => {
    const att = fr0.p;
    if (!att) return null;
    const def: TeamId = att === "A" ? "B" : "A";
    const own = (x: number) => (att === "A" ? x : L - x);
    const xs = teamPts(def, false).map(({ p }) => own(p[0])).sort((a, b) => b - a);
    const v = Math.max(xs[1] ?? L, own(ball[0]), L / 2);
    return att === "A" ? v : L - v;
  })();

  const recent = events.filter((e) => e.timestamp <= t && e.timestamp > t - 2.5 && e.event_type !== "PRESSURE");
  const color = (team: string) => meta.teams[team as TeamId].color;

  const trail: Pt[] = [];
  if (overlay === "counter") {
    for (let k = Math.max(0, i0 - 14); k <= i0; k++) trail.push(frames[k].b);
    trail.push(ball);
  }

  const focusPts = focusTeam ? teamPts(focusTeam) : [];
  const pressers =
    overlay === "press" && focusTeam && ball[0] > 0 && ball[0] < L
      ? [...focusPts].sort((a, b) => Math.hypot(a.p[0] - ball[0], a.p[1] - ball[1]) - Math.hypot(b.p[0] - ball[0], b.p[1] - ball[1])).slice(0, 3)
      : [];
  const hullPts = overlay === "shape" && focusTeam ? hull(focusPts.map(({ p }) => p)) : [];

  return (
    <svg viewBox="-3 -3 111 74" className="w-full h-full select-none" preserveAspectRatio="xMidYMid meet">
      <defs>
        <radialGradient id="glow">
          <stop offset="0%" stopColor="#d9f99d" stopOpacity="0.7" />
          <stop offset="100%" stopColor="#d9f99d" stopOpacity="0" />
        </radialGradient>
        <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="white" />
        </marker>
      </defs>
      <Markings />

      {showLines &&
        (["A", "B"] as TeamId[]).map((team) => {
          const x = defLine(team);
          const strong = overlay === "line" || overlay === "press" || focusTeam === team;
          return (
            <g key={team}>
              <line x1={x} y1={0} x2={x} y2={W} stroke={color(team)} strokeWidth={strong ? 0.35 : 0.2} strokeDasharray="1.2 0.8" opacity={strong ? 0.95 : 0.45} />
              <text x={x + (team === "A" ? 0.6 : -0.6)} y={1.8} fontSize={1.5} fill={color(team)} textAnchor={team === "A" ? "start" : "end"} opacity={0.85}>
                {meta.teams[team].short} line {(team === "A" ? x : L - x).toFixed(0)}m
              </text>
            </g>
          );
        })}

      {showLines && offsideX != null && (
        <g>
          <line x1={offsideX} y1={0} x2={offsideX} y2={W} stroke="white" strokeWidth={0.15} strokeDasharray="0.4 0.4" opacity={0.7} />
          <text x={offsideX} y={W - 0.8} fontSize={1.3} fill="white" textAnchor="middle" opacity={0.75}>offside</text>
        </g>
      )}

      {shape &&
        (["A", "B"] as TeamId[]).map((team) => {
          const s = shape[team];
          if (!s) return null;
          // players per line: this second's fit of the formation (fr0.sl), else the window's grouping
          const groups: number[][] = [];
          if (fr0.sl) {
            order.forEach((pid, j) => {
              const k = fr0.sl![j];
              if (k >= 0 && pid.startsWith(team)) (groups[k] ??= []).push(j);
            });
          } else {
            s.lines.forEach((ln, k) => (groups[k] = ln.map((pid) => order.indexOf(pid)).filter((j) => j >= 0)));
          }
          // each line joins its players across the pitch (current positions); dashed where an end is estimated
          const lines = groups.map((js) =>
            (js ?? []).filter((j) => pos[j] && !(camera?.hideInferred && fr0.obs?.[j] === false)).map((j) => ({ p: pos[j] as Pt, est: fr0.obs?.[j] === false })).sort((a, b) => a.p[1] - b.p[1]),
          );
          if (!lines.some((l) => l.length)) return null;
          return (
            <g key={`shape-${team}`} opacity={0.85}>
              {lines.map((pts, k) =>
                pts.slice(1).map((b, i) => {
                  const a = pts[i];
                  return (
                    <line key={`${k}-${i}`} x1={a.p[0]} y1={a.p[1]} x2={b.p[0]} y2={b.p[1]} stroke={color(team)}
                      strokeWidth={0.45} strokeLinecap="round" strokeDasharray={a.est || b.est ? "1 0.8" : undefined}
                      opacity={a.est || b.est ? 0.6 : 0.95} />
                  );
                }),
              )}
              {(
                <text x={team === "A" ? 1 : L - 1} y={W - 1} fontSize={2} fontWeight={700} fill={color(team)}
                  textAnchor={team === "A" ? "start" : "end"}>{meta.teams[team].short} {s.name}</text>
              )}
            </g>
          );
        })}

      {hullPts.length > 2 && (
        <polygon points={hullPts.map((p) => p.join(",")).join(" ")} fill={color(focusTeam!)} fillOpacity={0.12} stroke={color(focusTeam!)} strokeWidth={0.25} strokeOpacity={0.7} />
      )}

      {pressers.map(({ pid, p }) => (
        <line key={pid} x1={p[0]} y1={p[1]} x2={ball[0]} y2={ball[1]} stroke="#facc15" strokeWidth={0.3} strokeDasharray="0.8 0.5" opacity={0.9} />
      ))}

      {trail.length > 1 && <polyline points={trail.map((p) => p.join(",")).join(" ")} fill="none" stroke="#fde047" strokeWidth={0.45} strokeOpacity={0.8} strokeLinecap="round" />}

      {recent.map((e) => {
        const age = (t - e.timestamp) / 2.5;
        if (e.event_type === "PASS" && e.x != null && e.target_x != null)
          return (
            <line key={e.id} x1={e.x} y1={e.y} x2={e.target_x} y2={e.target_y} stroke={e.success ? "white" : "#f87171"} strokeWidth={0.25} opacity={1 - age} markerEnd="url(#arrow)" strokeDasharray={e.success ? undefined : "0.6 0.4"} />
          );
        if (e.event_type === "SHOT" && e.x != null) {
          const gx = e.team_id === "A" ? L : 0;
          return <line key={e.id} x1={e.x} y1={e.y} x2={gx} y2={W / 2} stroke="#facc15" strokeWidth={0.5} opacity={1 - age} />;
        }
        if ((e.event_type === "TACKLE" || e.event_type === "RED_CARD") && e.x != null)
          return (
            <g key={e.id} opacity={1 - age} transform={`translate(${e.x},${e.y})`}>
              <circle r={2.4} fill="none" stroke={e.event_type === "RED_CARD" ? "#ef4444" : "#facc15"} strokeWidth={0.3} />
            </g>
          );
        return null;
      })}

      {camera?.footprint && fr0.cam && (
        /* the broadcast camera's view this second: everything outside it is dimmed */
        <g pointerEvents="none">
          <path fillRule="evenodd" fill="rgba(2,6,23,0.55)"
            d={`M-3,-3 H108 V71 H-3 Z M${fr0.cam.map((q) => q.join(",")).join(" L")} Z`} />
          <polygon points={fr0.cam.map((q) => q.join(",")).join(" ")} fill="none" stroke="#e2e8f0" strokeWidth={0.25} strokeDasharray="1 0.6" opacity={0.8} />
        </g>
      )}
      {camera?.footprint && !fr0.cam && (
        <rect x={-3} y={-3} width={111} height={74} fill="rgba(2,6,23,0.55)" pointerEvents="none" />
      )}

      {order.map((pid, j) => {
        const p = pos[j];
        const inferred = fr0.obs ? fr0.obs[j] === false : false;
        if (!p || !inferred || camera?.hideInferred) return null;
        const r = fr0.rad?.[j];
        const tp = truthFrames ? lerp(truthFrames[i0]?.xy[j] as Pt | null, truthFrames[i1]?.xy[j] as Pt | null, f) : null;
        return (
          <g key={`u-${pid}`} style={{ pointerEvents: "none" }}>
            {showUncertainty && r && (!reveal || reveal.player_id === pid) ? <circle cx={p[0]} cy={p[1]} r={r} fill={color(pmap[pid].team)} fillOpacity={0.07} stroke={color(pmap[pid].team)} strokeOpacity={0.35} strokeWidth={0.12} strokeDasharray="0.5 0.5" /> : null}
            {tp && (
              <>
                <line x1={p[0]} y1={p[1]} x2={tp[0]} y2={tp[1]} stroke="#facc15" strokeWidth={0.15} opacity={0.8} />
                <rect x={tp[0] - 0.6} y={tp[1] - 0.6} width={1.2} height={1.2} fill="none" stroke="#facc15" strokeWidth={0.2} />
              </>
            )}
          </g>
        );
      })}

      {order.map((pid, j) => {
        const p = pos[j];
        if (!p) return null;
        if (camera?.hideInferred && fr0.obs?.[j] === false) return null;
        const pl = pmap[pid];
        const focus = focusPlayers.includes(pid);
        const sel = selectedPlayer === pid;
        const carrier = fr0.c === pid;
        const estimated = fr0.obs ? fr0.obs[j] === false : false;
        return (
          <g key={pid} transform={`translate(${p[0]},${p[1]})`} onClick={() => onPlayerClick(pid)} className="cursor-pointer" opacity={estimated && !focus && !sel ? 0.45 : 1}>
            {(focus || sel) && <circle r={3.6} fill="url(#glow)" />}
            {(focus || sel) && <circle r={2.2} fill="none" stroke={sel ? "#ffffff" : "#d9f99d"} strokeWidth={0.3} className="pulse-ring" />}
            <circle r={1.35} fill={pl.position === "GK" ? (pl.team === "A" ? "#7f1d1d" : "#0c4a6e") : color(pl.team)}
              stroke={carrier ? "#fde047" : estimated ? "white" : "rgba(0,0,0,0.6)"} strokeWidth={carrier ? 0.35 : estimated ? 0.2 : 0.15}
              strokeDasharray={estimated ? "0.5 0.35" : undefined} />
            <text y={0.5} fontSize={1.3} textAnchor="middle" fill="white" fontWeight={700} style={{ pointerEvents: "none" }}>
              {pl.shirt}
            </text>
            {(focus || sel) && (
              <text y={-2.4} fontSize={1.5} textAnchor="middle" fill="white" style={{ pointerEvents: "none", paintOrder: "stroke" }} stroke="rgba(0,0,0,0.7)" strokeWidth={0.3}>
                {pl.name}
              </text>
            )}
          </g>
        );
      })}

      {reveal && (() => {
        /* prediction -> reality: while hidden the live estimate is labelled PREDICTED; at the reveal the prediction
           (ghost + 80% radius) is compared with where the player actually reappeared */
        const j = order.indexOf(reveal.player_id);
        const tc = pmap[reveal.player_id];
        if (j < 0 || !tc) return null;
        const start = reveal.t - reveal.hidden_s;
        const col = color(tc.team);
        if (t >= start && t < reveal.t && pos[j]) {
          const p = pos[j] as Pt;
          return (
            <g pointerEvents="none">
              <circle cx={p[0]} cy={p[1]} r={2.6} fill="none" stroke="#fde047" strokeWidth={0.3} className="pulse-ring" />
              <text x={p[0]} y={p[1] - 3} fontSize={1.6} fontWeight={800} textAnchor="middle" fill="#fde047" stroke="rgba(0,0,0,0.8)" strokeWidth={0.3} style={{ paintOrder: "stroke" }}>
                PREDICTED · hidden {Math.floor(t - start)}s
              </text>
            </g>
          );
        }
        if (t >= reveal.t && t < reveal.t + 6) {
          const [px, py] = reveal.predicted;
          const [ax, ay] = reveal.actual;
          return (
            <g pointerEvents="none">
              <circle cx={px} cy={py} r={reveal.radius_m} fill={col} fillOpacity={0.08} stroke="#fde047" strokeWidth={0.18} strokeDasharray="0.6 0.5" />
              <circle cx={px} cy={py} r={1.35} fill="none" stroke="#fde047" strokeWidth={0.3} strokeDasharray="0.5 0.35" />
              <line x1={px} y1={py} x2={ax} y2={ay} stroke="#fde047" strokeWidth={0.25} />
              <circle cx={ax} cy={ay} r={2.1} fill="none" stroke="#fde047" strokeWidth={0.35} />
              <text x={(px + ax) / 2} y={Math.min(py, ay) - 2.5} fontSize={1.7} fontWeight={800} textAnchor="middle" fill="#fde047" stroke="rgba(0,0,0,0.85)" strokeWidth={0.35} style={{ paintOrder: "stroke" }}>
                ERROR {reveal.error_m.toFixed(1)} m · {reveal.inside_radius ? "inside" : "outside"} ±{reveal.radius_m.toFixed(1)} m
              </text>
            </g>
          );
        }
        return null;
      })()}

      {fr0.ds && (() => {
        /* dead ball: play has not restarted yet; ring around the ball + what restarts play */
        const label = `${fr0.ds.toUpperCase()} · NOT TAKEN YET`;
        const w = label.length * 0.85 + 2;
        const lx = Math.min(L - w, Math.max(0, ball[0] - w / 2));
        return (
        <g pointerEvents="none">
          <circle cx={ball[0]} cy={ball[1]} r={2.2} fill="none" stroke="#fde047" strokeWidth={0.25} strokeDasharray="0.7 0.5">
            <animateTransform attributeName="transform" type="rotate" from={`0 ${ball[0]} ${ball[1]}`} to={`360 ${ball[0]} ${ball[1]}`} dur="4s" repeatCount="indefinite" />
          </circle>
          <rect x={lx} y={ball[1] < 6 ? ball[1] + 3 : ball[1] - 5.6} width={w} height={2.6} rx={0.6} fill="rgba(0,0,0,0.75)" />
          <text x={lx + w / 2} y={ball[1] < 6 ? ball[1] + 4.8 : ball[1] - 3.8} fontSize={1.5} fontWeight={700} fill="#fde047" textAnchor="middle">
            {label}
          </text>
        </g>
        );
      })()}
      <circle cx={ball[0]} cy={ball[1]} r={0.75} fill={fr0.ds ? "#fde047" : "white"} stroke="#111" strokeWidth={0.15}
        opacity={fr0.bs === "raw" ? 0.45 : 1} strokeDasharray={fr0.bs === "raw" ? "0.3 0.25" : undefined} />
    </svg>
  );
}
