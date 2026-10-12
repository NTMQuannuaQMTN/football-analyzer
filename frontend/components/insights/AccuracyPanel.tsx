"use client";
import { memo, useEffect, useState } from "react";
import { loadEvaluation } from "@/lib/api";

/* eslint-disable @typescript-eslint/no-explicit-any */
const LEVEL_LABEL: Record<string, string> = { "k=18": "18 / 22", "k=14": "14 / 22", "k=10": "10 / 22", "k=6": "6 / 22", broadcast: "Broadcast cam" };

const MASKS: [string, string][] = [["sc_camera", "Transferred camera mask"], ["k=18", "18 / 22"], ["k=14", "14 / 22"], ["k=10", "10 / 22"], ["k=6", "6 / 22"]];

function RealFootball({ rep }: { rep: any }) {
  const lomo = rep.real_lomo?.k7r_v3 ?? rep.real_lomo?.v2 ?? rep.real_lomo?.v1;
  const sevenFold = !!rep.real_lomo?.k7r_v3;
  const rv = rep.reveals;
  const tac = rep.real_tactics;
  if (!lomo && !rv && !tac) return null;
  const ms = (x: any) => (x ? `${x.mean.toFixed(1)}±${x.sd.toFixed(1)}` : "–");
  const ca = tac?.["COUNTERATTACK (shots, DFL CounterAttack flag)"];
  const fm = tac?.["FORMATION (official line-up, first 15 min)"];
  return (
    <div className="space-y-4">
      <div className="text-[11px] uppercase tracking-widest text-lime-300 font-bold">Real football (DFL tracking ground truth)</div>
      {lomo && (
        <div>
          <div className="text-[11px] uppercase tracking-widest text-slate-400 mb-1">Hidden players · {lomo.folds} held-out matches ({sevenFold ? "7-fold: train on 6, test on the 7th" : "leave-one-match-out"}) · mean error m, mean±sd over matches</div>
          <table className="w-full text-xs">
            <thead className="text-slate-400"><tr><th className="text-left font-normal">Visible</th><th className="text-right font-normal">Last seen</th><th className="text-right font-normal">Shape L / R</th><th className="text-right font-normal text-lime-300">Learned L / R</th><th className="text-right font-normal">80% radius</th></tr></thead>
            <tbody className="font-mono">
              {MASKS.filter(([k]) => lomo.results[k]).map(([k, label]) => {
                const r = lomo.results[k];
                return (
                  <tr key={k} className="border-t border-white/5">
                    <td className="font-sans py-0.5">{label}</td>
                    <td className="text-right text-slate-500">{ms(r.last_position?.mae_m)}</td>
                    <td className="text-right text-slate-400">{r.formation_live?.mae_m?.mean.toFixed(1)} / {r.formation_replay?.mae_m?.mean.toFixed(1)}</td>
                    <td className="text-right text-lime-300 font-bold">{ms(r.ml_live?.mae_m)} / {ms(r.ml_replay?.mae_m)}</td>
                    <td className="text-right">{r.ml_live?.radius_coverage_80 ? `${Math.round(r.ml_live.radius_coverage_80.mean * 100)}%` : "–"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="text-[10px] text-slate-500 mt-1">L = live (past only), R = replay (also the next sighting). 80% radius = share of hidden players inside the stated radius (target 80%).</div>
        </div>
      )}
      {rv && (
        <div>
          <div className="text-[11px] uppercase tracking-widest text-slate-400 mb-1">Prediction → reality · {rv.n.toLocaleString()} reappearances (held-out matches)</div>
          <div className="text-xs text-slate-300">
            Live prediction at the second the player came back into view: median error <b className="text-white">{rv.median_error_m} m</b>,
            {" "}{Math.round(rv.within_10m * 100)}% within 10 m, inside the 80% radius <b className="text-white">{Math.round(rv.inside_80pct_radius * 100)}%</b>.
          </div>
          <div className="text-xs text-slate-400 font-mono mt-0.5">
            {Object.entries(rv.by_hidden_s).map(([k, v]: [string, any]) => <span key={k} className="mr-3">hidden {k}: {v.median_error_m} m</span>)}
          </div>
        </div>
      )}
      {rep.clip && (
        <div>
          <div className="text-[11px] uppercase tracking-widest text-slate-400 mb-1">Broadcast video (SoccerNet clip) · {rep.clip.reappearances} reappearances</div>
          <table className="w-full text-xs font-mono">
            <tbody>
              {Object.entries(rep.clip.methods).map(([k, v]: [string, any]) => (
                <tr key={k} className="border-t border-white/5"><td className="font-sans py-0.5">{k.replace("learned_", "learned ").replace("_live", "").replace(/_/g, " ")}</td><td className="text-right">median {v.median_error_m} m</td></tr>
              ))}
            </tbody>
          </table>
          <div className="text-[10px] text-slate-500 mt-1">Reference = a projected CV detection at reappearance, with role-assigned identity. Neither identity nor camera projection is independent ground truth. These are consistency proxies, not verified hidden-player accuracy.</div>
        </div>
      )}
      {tac && (
        <div>
          <div className="text-[11px] uppercase tracking-widest text-slate-400 mb-1">Tactics on real matches</div>
          <table className="w-full text-xs">
            <tbody>
              {ca && (
                <tr className="border-t border-white/5">
                  <td className="py-0.5">Counterattack shots vs DFL&apos;s own label</td>
                  <td className="text-right font-mono">P {ca.all.precision} · R {ca.all.recall} · F1 {ca.all.f1}</td>
                </tr>
              )}
              {ca && (
                <tr><td className="text-slate-500 text-[11px]">held-out 2 matches (not used to design)</td><td className="text-right font-mono text-slate-400">F1 {ca["test (2 held-out matches)"].f1} ({ca["test (2 held-out matches)"].tp}/{ca["test (2 held-out matches)"].tp + ca["test (2 held-out matches)"].fn} found)</td></tr>
              )}
              {fm && (
                <tr className="border-t border-white/5">
                  <td className="py-0.5">Shape vs official line-up</td>
                  <td className="text-right font-mono">exact {fm.exact_match}/{fm.teams} · back line {fm.back_line_match}/{fm.teams}</td>
                </tr>
              )}
              {fm && <tr><td className="text-slate-500 text-[11px]">shape changes per team-half</td><td className="text-right font-mono text-slate-400">{fm.detected_changes_per_team_half}</td></tr>}
            </tbody>
          </table>
          <div className="text-[10px] text-slate-500 mt-1">High press: thresholds were tuned on the simulator and flag 0 real minutes; there are no real press labels to recalibrate against, so it is unvalidated on real football.</div>
        </div>
      )}
      <div className="border-t border-white/10 pt-3 text-[11px] uppercase tracking-widest text-slate-500 font-bold">Synthetic benchmark (simulator ground truth, separate)</div>
    </div>
  );
}

function AccuracyPanel() {
  const [rep, setRep] = useState<any>(null);
  useEffect(() => { loadEvaluation().then(setRep).catch(() => setRep({ error: "N/A" })); }, []);
  if (!rep) return <div className="text-sm text-slate-400 animate-pulse">Loading evaluation…</div>;
  if (rep.error) return <div className="text-sm text-slate-400">{rep.error}</div>;
  const multi = rep.multi;
  const ci = (x: any) => (x ? `${x.mean.toFixed(2)} ±${((x.ci95[1] - x.ci95[0]) / 2).toFixed(2)}` : "–");
  const recon = rep.reconstruction;
  const tac = rep.tactics;
  const keys = Object.keys(tac.full_ground_truth);
  return (
    <div className="space-y-4 text-sm">
      <p className="text-xs text-amber-200/90">Stored benchmark results for their evaluated datasets and models. These do not measure accuracy for the match currently shown. DFL tests use full reference tracking with simulated visibility; they do not establish EPL video accuracy.</p>
      <RealFootball rep={rep} />
      <div>
        <div className="text-[11px] uppercase tracking-widest text-slate-400">Simulator evaluation</div>
        <p className="text-slate-300 mt-1">
          Simulated camera masks hide selected players, then reconstructed positions are compared with the simulator&apos;s complete reference positions.
          Seed {rep.eval_seed}; uncertainty calibrated on a different match (seed {rep.calibration_seed}).
        </p>
      </div>

      {rep.real && !rep.real_lomo && Object.entries(rep.real.folds).map(([fold, f]: [string, any]) => (
        <div key={fold}>
          <div className="text-[11px] uppercase tracking-widest text-slate-400 mb-1">Real football · {fold} · test match {String(f.test).replace("DFL-MAT-", "")} (teams unseen in training)</div>
          <table className="w-full text-xs">
            <thead className="text-slate-400">
              <tr><th className="text-left font-normal">Mask</th><th className="text-right font-normal pl-1">Last</th><th className="text-right font-normal pl-1">CV</th><th className="text-right font-normal pl-1">Shape L/R</th><th className="text-right font-normal pl-1">ML L/R</th><th className="text-right font-normal pl-1">ML ≤5m</th></tr>
            </thead>
            <tbody className="font-mono">
              {Object.entries(f.results).map(([mk, r]: [string, any]) => (
                <tr key={mk} className="border-t border-white/5">
                  <td className="font-sans py-0.5">{mk === "sc_camera" ? "Transferred camera mask" : mk.replace("k=", "").replace("rand=", "rand ") + (mk.startsWith("k=") ? " / 22" : "")}</td>
                  <td className="text-right pl-1 text-slate-500">{r.last_position.mae_m}</td>
                  <td className="text-right pl-1 text-slate-500">{r.constant_velocity.mae_m}</td>
                  <td className="text-right pl-1">{r.formation_live.mae_m}/{r.formation_replay.mae_m}</td>
                  <td className="text-right pl-1 text-lime-300">{r.ml_live.mae_m}/{r.ml_replay.mae_m}</td>
                  <td className="text-right pl-1">{Math.round(r.ml_replay.within_5m * 100)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="text-[10px] text-slate-500 mt-1">{rep.real.dataset}. Hidden-player MAE in metres. L = live (past only), R = replay (retrospective).</div>
        </div>
      ))}

      {multi && (
        <div>
          <div className="text-[11px] uppercase tracking-widest text-slate-400 mb-1">{multi.matches} matches · hidden-player error (m, mean ±95% CI)</div>
          <table className="w-full text-xs">
            <thead className="text-slate-400">
              <tr><th className="text-left font-normal">Visible</th><th className="text-right font-normal pl-2">Freeze</th><th className="text-right font-normal pl-2">Live</th><th className="text-right font-normal pl-2">Replay</th><th className="text-right font-normal pl-2">Beats freeze</th></tr>
            </thead>
            <tbody className="font-mono">
              {Object.entries(multi.aggregate.recon).map(([lvl, r]: [string, any]) => (
                <tr key={lvl} className="border-t border-white/5">
                  <td className="font-sans py-0.5">{LEVEL_LABEL[lvl] ?? lvl}</td>
                  <td className="text-right pl-2 text-slate-500">{ci(r.hold.mae_m)}</td>
                  <td className="text-right pl-2">{ci(r.live.mae_m)}</td>
                  <td className="text-right pl-2 text-lime-300">{ci(r.replay.mae_m)}</td>
                  <td className="text-right pl-2">{Math.round(r.replay_beats_hold_share * 100)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="text-[10px] text-slate-500 mt-1">{multi.note}. Beats freeze = share of matches where replay error &lt; freeze error.</div>
        </div>
      )}

      <div>
        <div className="text-[11px] uppercase tracking-widest text-slate-400 mb-1">Demo match (seed {rep.eval_seed}) · hidden-player position error (m)</div>
        <table className="w-full text-xs">
          <thead className="text-slate-400">
            <tr><th className="text-left font-normal">Visible</th><th className="text-right font-normal">Freeze</th><th className="text-right font-normal">Live</th><th className="text-right font-normal">Replay</th><th className="text-right font-normal">≤5 m</th><th className="text-right font-normal">≤10 m</th><th className="text-right font-normal">±r hit</th></tr>
          </thead>
          <tbody className="font-mono">
            {Object.entries(recon).map(([lvl, r]: [string, any]) => (
              <tr key={lvl} className="border-t border-white/5">
                <td className="font-sans py-0.5">{LEVEL_LABEL[lvl] ?? lvl}</td>
                <td className="text-right text-slate-500">{r.hold_last_seen_baseline.mae_m}</td>
                <td className="text-right">{r.live.mae_m}</td>
                <td className="text-right text-lime-300">{r.replay.mae_m}</td>
                <td className="text-right">{Math.round(r.replay.within_5m * 100)}%</td>
                <td className="text-right">{Math.round(r.replay.within_10m * 100)}%</td>
                <td className="text-right">{Math.round(r.replay["radius_coverage (target 0.80)"] * 100)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="text-[10px] text-slate-500 mt-1">Freeze = leave hidden players where last seen. Live uses only the past; Replay also uses the next sighting (retrospective). ±r hit: share of true positions inside the reported uncertainty circle (target 80%).</div>
      </div>

      <div>
        <div className="text-[11px] uppercase tracking-widest text-slate-400 mb-1">Tactical detection F1</div>
        <table className="w-full text-xs">
          <thead className="text-slate-400">
            <tr><th className="text-left font-normal">State used</th>{keys.map((k) => <th key={k} className="text-right font-normal pl-2">{k.split(" ")[0].replace("ATTACKING_TRANSITION", "ATT TR").replace("DEFENSIVE_TRANSITION", "DEF TR").replace("TACTICAL_", "").replace("COUNTERATTACK", "COUNTER").replace("HIGH_PRESS", "PRESS")}</th>)}</tr>
          </thead>
          <tbody className="font-mono">
            <tr className="border-t border-white/5"><td className="font-sans py-0.5">Full ground truth</td>{keys.map((k) => <td key={k} className="text-right pl-2">{tac.full_ground_truth[k].f1}</td>)}</tr>
            {["k=6", "broadcast"].filter((l) => tac[l]).flatMap((l) => (["observed_only", "reconstructed_replay"] as const).map((v) => (
              <tr key={l + v} className="border-t border-white/5">
                <td className={`font-sans py-0.5 ${v === "observed_only" ? "text-slate-400" : ""}`}>{LEVEL_LABEL[l]} · {v === "observed_only" ? "visible only" : "reconstructed"}</td>
                {keys.map((k) => <td key={k} className={`text-right pl-2 ${v === "reconstructed_replay" ? "text-lime-300" : "text-slate-400"}`}>{tac[l][v][k].f1}</td>)}
              </tr>
            )))}
          </tbody>
        </table>
        <div className="text-[10px] text-slate-500 mt-1">This synthetic benchmark keeps ball and events fixed across visibility masks. Counterattack results here do not test broadcast ball-detection errors.</div>
      </div>
    </div>
  );
}

export default memo(AccuracyPanel);
