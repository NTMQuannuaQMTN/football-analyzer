"""INGEST -> INTERPRET -> EXPLAIN -> RENDER -> PERSONALIZE orchestration."""
import glob
import json
import os
import time
from pathlib import Path
from threading import Lock

import numpy as np
from typing import Optional

from backend.analytics.movement import player_movement, speeds
from backend.analytics.passing import player_event_stats
from backend.analytics.possession import possession_sequences
from backend.analytics.shape import lines_per_second
from backend.counterfactual.engine import counterfactual_for
from backend.models.match import MatchData
from backend.moments.detector import MomentDetector
from backend.moments.story import build_story, chapters
from backend.narratives.generator import get_generator
from backend.tactics.counterattack import detect_counterattacks
from backend.tactics.high_press import detect_high_press
from backend.tactics.shift import detect_shifts
from backend.tactics.tactical_state import build_timeline, tactical_label, window_metrics
from backend.reconstruction.estimator import EVIDENCE, SRC_INF, SRC_OBS, reconstruct, roster_timeline
from backend.reconstruction.observation import observe
from simulator.match_generator import generate_match
from simulator.scenarios import GROUND_TRUTH

EVAL_REPORT = os.path.join(os.path.dirname(__file__), "..", "data", "eval", "report.json")
ABSTAIN_MIN_OBSERVED = 2.5  # avg observed outfield players per team-minute below which we don't claim team tactics


def frames_from(truth: MatchData, rec=None, cut=None):
    """Render frames: reconstructed positions + per-player source / confidence / uncertainty."""
    out = []
    for t, f in enumerate(truth.frames):
        if rec is None:
            out.append({**f, "obs": [None if p is None else True for p in f["xy"]]})
            continue
        xy, ob, cf, rd = [], [], [], []
        for j in range(len(truth.order)):
            if rec.src[t, j] == 0:
                xy.append(None), ob.append(None), cf.append(None), rd.append(None)
            else:
                p = rec.pos[t, j]
                xy.append([round(float(p[0]), 1), round(float(p[1]), 1)])
                ob.append(bool(rec.src[t, j] == SRC_OBS))
                cf.append(round(float(rec.conf[t, j]), 2))
                rd.append(round(float(rec.rad[t, j]), 1))
        out.append({"t": f["t"], "b": f["b"], "p": f["p"], "c": f["c"], "xy": xy, "obs": ob, "conf": cf, "rad": rd,
                    "cut": bool(cut[t]) if cut is not None else False})
    return out


GENERIC = {"FC", "F.C.", "VfL", "SSV", "SV", "SC", "TSG", "Bayer", "Fortuna", "AFC", "CF", "United", "City"}


def prose_name(name):
    """Short team name for sentences: '1. FC Köln' -> 'Köln', 'FC Bayern München' -> 'Bayern'."""
    words = [w for w in name.split() if w.isalpha() and len(w) >= 3]
    return next((w for w in words if w not in GENERIC), words[0] if words else name)


class MatchIntelligence:
    def __init__(self, seed: int = 7, source: Optional[str] = None):
        t0 = time.time()
        self.seed = seed
        self.source = source
        # INGEST: synthetic simulator, or any file in the same contract (e.g. cv/build_match.py output)
        raw = json.load(open(source)) if source else generate_match(seed)
        self.raw_meta = dict(raw["meta"])  # as ingested (the lobby card reads it)
        # ground truth, when the match has it, is what we analyse ("truth"); the camera estimate ("vod") is kept to
        # compare against, and is what a match without ground truth (CV from video) has to rely on
        self.view = (raw["meta"].get("view") or "cv") if source else os.getenv("VIEW", "truth")
        # a file may carry its own live / truth renderings (e.g. backend/reconstruction/dfl_match.py)
        self.alt_frames = raw.pop("alt_frames", None) or {} if source else {}
        # estimates' calibrated confidence was checked on this input type (e.g. SkillCorner re-entries)
        self.uncertainty_validated = bool(((raw.get("meta") or {}).get("source") or {}).get("uncertainty_validated"))
        # prediction -> reality tests (backend/reconstruction/dfl_match.py): live prediction at each reappearance
        self.reveals = raw.pop("reveals", None) or [] if source else []
        self.truth_m, self.recon, self.obs = None, {}, None
        self._frames_cache = {}
        if source:
            self.m = MatchData(raw)
        else:
            # OBSERVE -> RECONSTRUCT: the synthetic match as a broadcast camera sees it, hidden players estimated.
            # VIEW=vod: analytics run on the reconstruction (ground truth only for evaluation / overlay).
            # VIEW=truth (default): analytics run on ground truth; the reconstruction is there to compare against.
            self.truth_m = MatchData(raw)
            on, _ = roster_timeline(self.truth_m)
            self.obs = observe(self.truth_m, "broadcast", seed)
            vis = self.obs.vis & on
            self.estimator = os.getenv("RECON_MODEL", "formation")
            self.recon = {mode: self._reconstruct(vis, mode) for mode in ("replay", "live")}
            rraw = {k: v for k, v in raw.items() if k != "truth"}
            rraw["frames"] = frames_from(self.truth_m, self.recon["replay"], self.obs.cut)
            rec = self.recon["replay"]
            rraw["meta"] = {**raw["meta"], "view": "vod", "reconstruction": {
                "mode": "replay", "retrospective": True, "camera": "simulated broadcast (seeded)",
                "observed_share": round(float((rec.src == SRC_OBS).sum() / max(1, (rec.src > 0).sum())), 3),
                "evidence_kinds": EVIDENCE}}
            if self.view == "vod":
                self.m = MatchData(rraw)
            else:
                raw["meta"] = {**raw["meta"], "view": "truth", "vod_available": True,
                               "reconstruction": rraw["meta"]["reconstruction"]}
                self.m = self.truth_m = MatchData(raw)
        # INTERPRET
        self.timeline = build_timeline(self.m)
        self.seqs = possession_sequences(self.m)
        self.speeds = speeds(self.m)
        # top speed / sprints only from observed->observed seconds (estimated jumps are not running)
        obs = np.array([[o is True for o in (f.get("obs") or [True] * len(f["xy"]))] for f in self.m.frames])
        both = obs & np.vstack([obs[:1], obs[:-1]])
        self.speeds_observed = np.where(both, self.speeds, np.nan)
        self.presses = detect_high_press(self.m, self.timeline, self.seqs)
        self.counters = detect_counterattacks(self.m, self.seqs)
        self.shifts = detect_shifts(self.timeline)
        self.coverage = self._coverage_by_minute()
        for team in ("A", "B"):
            for row, cov in zip(self.timeline[team], self.coverage[team]):
                row["visibility"] = cov
        self.moments = MomentDetector(self.m, self.timeline, self.seqs, self.presses, self.counters, self.shifts).run()
        for mo in self.moments:
            self._attach_coverage(mo)
            from backend.tactics.position_evidence import position_evidence, reason_from_positions
            mo["position_reasoning"] = reason_from_positions(mo, position_evidence(self.m, mo["team"], mo["timestamp"]))
        if "replay" in self.alt_frames and self.view == "truth":
            for mo in self.moments:
                hs = self._hidden_state(mo)
                if hs:
                    mo["hidden_state"] = hs
        self.story = build_story(self.moments, self.timeline)
        from backend.tactics.changes import tactical_changes
        self.changes = tactical_changes(self)
        # EXPLAIN / RENDER
        from backend.narratives import generator as gen_mod
        gen_mod.TEAM_NAMES.update({t: v["name"] for t, v in self.m.meta["teams"].items()})
        gen_mod.SHORT.update({t: prose_name(v["name"]) for t, v in self.m.meta["teams"].items()})
        self.gen = get_generator()
        ctx = {"teams": self.m.meta["teams"]}
        for mo in self.moments:
            mo["narrative"] = self.gen.generate(mo, ctx)
            mo["counterfactual"] = None if source else counterfactual_for(mo, self.timeline, seed)
        self.build_seconds = round(time.time() - t0, 2)

    def _hidden_state(self, mo, half_window=5):
        """What the camera couldn't see at this moment: the moment team's shape read from (a) the players the camera
        showed, (b) the camera-based reconstruction, (c) ground truth, averaged over +-5 s. Engine numbers only."""
        cam = self.alt_frames["replay"]
        team = mo["team"]
        t0 = int(mo["timestamp"])
        ts = range(max(0, t0 - half_window), min(len(cam), t0 + half_window + 1))
        js = [j for j, pid in enumerate(self.m.order) if pid.startswith(team) and not self.m.is_gk[j]]
        own = (lambda p: p[0]) if team == "A" else (lambda p: 105.0 - p[0])

        def metrics(pts, ball):
            if len(pts) < 3:
                return None
            xs = sorted(own(p) for p in pts)
            ys = [p[1] for p in pts]
            near = sum(1 for p in pts if ((p[0] - ball[0]) ** 2 + (p[1] - ball[1]) ** 2) ** 0.5 <= 15)
            return {"def_line_m": float(np.mean(xs[:4])), "width_m": float(max(ys) - min(ys)), "near_ball": float(near)}

        acc = {"observed_only": [], "reconstructed": [], "truth": []}
        n_obs, n_inf, n_hc = [], [], []
        for t in ts:
            cf, tf = cam[t], self.m.frames[t]
            obs = [cf["xy"][j] for j in js if cf["xy"][j] is not None and cf["obs"][j] is True]
            rec = [cf["xy"][j] for j in js if cf["xy"][j] is not None]
            tru = [tf["xy"][j] for j in js if tf["xy"][j] is not None]
            n_obs.append(len(obs))
            inf = [j for j in js if cf["xy"][j] is not None and cf["obs"][j] is False]
            n_inf.append(len(inf))
            n_hc.append(sum(1 for j in inf if (cf.get("conf") or [None] * len(cf["xy"]))[j] is not None and cf["conf"][j] >= 0.5))
            for k, pts in (("observed_only", obs), ("reconstructed", rec), ("truth", tru)):
                v = metrics(pts, tf["b"])
                if v:
                    acc[k].append(v)
        if not acc["truth"]:
            return None
        avg = {k: ({m: round(float(np.mean([r[m] for r in v])), 1) for m in v[0]} if v else None) for k, v in acc.items()}
        err = {}
        for m in ("def_line_m", "width_m", "near_ball"):
            tr = avg["truth"][m]
            err[m] = {"observed_only": round(abs(avg["observed_only"][m] - tr), 1) if avg["observed_only"] else None,
                      "reconstructed": round(abs(avg["reconstructed"][m] - tr), 1) if avg["reconstructed"] else None}
        return {"team": team, "window_s": [ts.start, ts.stop - 1], "observed_players": round(float(np.mean(n_obs)), 1),
                "inferred_players": round(float(np.mean(n_inf)), 1), "high_confidence_inferred": round(float(np.mean(n_hc)), 1),
                "outfield_players": len(js), "metrics": avg, "abs_error_vs_truth": err,
                "note": "Shape of this team as a broadcast camera would have shown it, as reconstructed, and as it really was"}

    def _reconstruct(self, vis, mode):
        """formation: team-shape estimator (calibrated on synthetic seed 11).
        ml: learned correction trained on DFL real ground truth (cv/weights/hidden_*), see backend/reconstruction/ml.py."""
        if self.estimator == "ml":
            from backend.reconstruction.inference import ml_available, reconstruct_ml
            from backend.reconstruction.ml import SIM_COARSE
            if ml_available():
                on, role = roster_timeline(self.truth_m)
                teams = np.array([p[0] for p in self.truth_m.order])
                rc = np.array([SIM_COARSE.get(role.get(p) or "CM", "CM") for p in self.truth_m.order])
                return self._no_teleport(reconstruct_ml(self.truth_m, vis, mode, on, role, None, teams, rc), mode)
        rec = reconstruct(self.truth_m, vis, mode)
        rec.meta["estimator"] = "formation"
        return self._no_teleport(rec, mode)

    @staticmethod
    def _no_teleport(rec, mode):
        """Estimates may not move faster than a sprint between seconds (observed positions are never moved)."""
        from backend.reconstruction.postprocess import limit_speed
        rec.pos = limit_speed(rec.pos, rec.src == SRC_OBS, rec.src > 0, causal=mode == "live")
        return rec

    # ---------------------------------------------------------------- observed vs inferred
    def _visible(self):
        f0 = self.m.frames
        return [[(o is True) for o in (f.get("obs") or [])] for f in f0]

    def _coverage_by_minute(self):
        """Per team-minute: avg observed / inferred players and mean inferred confidence."""
        out = {"A": [], "B": []}
        team_of = [pid[0] for pid in self.m.order]
        n_min = len(self.timeline["A"])
        for k in range(n_min):
            fr = self.m.frames[k * 60:(k + 1) * 60]
            for team in ("A", "B"):
                obs = inf = 0
                confs = []
                for f in fr:
                    ob = f.get("obs") or [p is not None for p in f["xy"]]
                    for j, o in enumerate(ob):
                        if team_of[j] != team or o is None:
                            continue
                        if o:
                            obs += 1
                        else:
                            inf += 1
                            if f.get("conf") and (self.m.meta.get("source", {}).get("type") != "computer_vision"
                                                  or getattr(self, "uncertainty_validated", False)):
                                value = f["conf"][j]
                                if value is not None and np.isfinite(value):
                                    confs.append(value)
                n = max(1, len(fr))
                out[team].append({"observed_avg": round(obs / n, 1), "inferred_avg": round(inf / n, 1),
                                  "inferred_conf": round(sum(confs) / len(confs), 2) if confs else None,
                                  "abstain": obs / n - 1 < ABSTAIN_MIN_OBSERVED})  # minus the keeper
        return out

    def _attach_coverage(self, mo):
        team = mo["team"]
        a, b = int(mo["timestamp"] // 60), int(min(self.m.duration, mo.get("end_timestamp", mo["timestamp"])) // 60) + 1
        rows = self.coverage[team][a:b] or self.coverage[team][a:a + 1]
        obs = sum(r["observed_avg"] for r in rows) / len(rows)
        inf = sum(r["inferred_avg"] for r in rows) / len(rows)
        cf = [r["inferred_conf"] for r in rows if r["inferred_conf"] is not None]
        conf_inf = sum(cf) / len(cf) if cf else (None if inf > 0.05 else 1.0)
        share = obs / max(1e-6, obs + inf)
        reliability = share + (1 - share) * (conf_inf if conf_inf is not None else 0.0)
        mo["coverage"] = {"team": team, "observed_avg": round(obs, 1), "inferred_avg": round(inf, 1),
                          "observed_share": round(share, 2), "inferred_mean_confidence": round(conf_inf, 2) if conf_inf is not None else None,
                          "reliability": round(reliability, 2), "view": self.view}
        mo["confidence"] = round(mo["confidence"] * (0.6 + 0.4 * reliability), 2)
        if mo["type"] == "FORMATION_SHIFT" and share < 0.5:
            # mostly estimated players: on DFL truth-vs-reconstruction the timing of such a shift matched a real one in
            # 6/9 cases but the new formation name in only 3/9, so the label is phrased as possible and capped
            mo["title"] = mo["title"].replace(" switch to ", " possible switch to ", 1)
            mo["confidence"] = min(mo["confidence"], 0.5)
            mo["formation_label_reliability"] = "low: most of the team was estimated, the new formation name is uncertain"
        position_based = ("Tactical trigger", "Structural change", "Pressing", "Territory", "Recovery", "Exposed defence",
                          "Exposed space", "Opponent shape", "Control")
        for ev in mo["evidence"]:
            ev["basis"] = "observed + inferred positions" if ev["label"] in position_based and inf > 0.05 else "observed events"
            if self.m.meta.get("source", {}).get("type") == "computer_vision":
                ev["basis"] = "video-derived positions; identity/calibration unverified" if ev["label"] in position_based else "recorded event feed; some events estimated"
        # substitution and shape moments: visibility stays in the coverage header, not in the analysis reasons
        if inf > 0.05 and mo["type"] not in ("SUBSTITUTION_IMPACT", "FORMATION_SHIFT"):
            uncertainty = f"mean confidence {conf_inf:.0%}" if conf_inf is not None else "uncertainty N/A; no calibrated confidence available"
            mo["evidence"].append({"label": "Visibility", "detail": f"On average {obs:.0f} of {obs + inf:.0f} players observed; "
                                   f"{inf:.0f} estimated from trajectories and team shape ({uncertainty})",
                                   "value": round(share, 2), "unit": "observed share", "timestamp": mo["timestamp"],
                                   "source": "reconstruction", "basis": "coverage"})

    def resolve_mode(self, mode):
        """Do not relabel a retrospective CV export as live or ground truth."""
        if mode == "default":
            mode = "truth" if self.view == "truth" else "replay"
        if mode not in ("truth", "replay", "live"):
            raise ValueError(f"Unknown reconstruction mode: {mode}")
        source = self.m.meta.get("source") or {}
        if source.get("type") == "computer_vision" and mode != "replay" and mode not in self.alt_frames:
            raise ValueError("This video export provides retrospective replay only; live estimates and ground truth are N/A")
        return mode

    def player_states(self, t: float, mode: str = "replay"):
        """Machine-readable per-player state at time t (observed / inferred, confidence, uncertainty, evidence)."""
        mode = self.resolve_mode(mode)
        t = int(max(0, min(self.m.duration, t)))
        rec = self.recon.get(mode)
        selected_frames = self.frames(mode) if mode in self.alt_frames else self.m.frames
        out = []
        for j, pid in enumerate(self.m.order):
            if rec is not None:
                if rec.src[t, j] == 0:
                    continue
                p = rec.pos[t, j]
                obs = rec.src[t, j] == SRC_OBS
                out.append({"player_id": pid, "position": {"x": round(float(p[0]), 1), "y": round(float(p[1]), 1)},
                            "state_source": "observed" if obs else "inferred", "confidence": round(float(rec.conf[t, j]), 2),
                            "uncertainty_radius": round(float(rec.rad[t, j]), 1),
                            "evidence": rec.evidence_list(t, j), "retrospective": bool(rec.meta.get("retrospective")) and not obs})
            else:
                f = selected_frames[t]
                p = f["xy"][j]
                if p is None:
                    continue
                o = (f.get("obs") or [True] * len(f["xy"]))[j]
                cf, rd = (f.get("conf") or [None] * len(f["xy"]))[j], (f.get("rad") or [None] * len(f["xy"]))[j]
                kind = "tracking_masked_estimate" if self.m.meta.get("source", {}).get("type") == "tracking" else "cv_estimate"
                is_cv = self.m.meta.get("source", {}).get("type") == "computer_vision"
                inferred_conf_ok = not is_cv or (getattr(self, "uncertainty_validated", False) and not o)
                out.append({"player_id": pid, "position": {"x": p[0], "y": p[1]},
                            "state_source": "observed" if o else "inferred", "confidence": (1.0 if o else cf) if inferred_conf_ok else None,
                            "observation_timestamp": (f.get("observed_t") or [None] * len(f["xy"]))[j] if o else None,
                            "uncertainty_radius": (None if is_cv else 0.0) if o else rd,
                            "evidence": ["detected_foot; calibration/identity unverified" if is_cv else "observed"] if o else [kind]})
        return out

    def player_state(self, pid: str, t: float, mode: str = "replay"):
        """One player's state at t plus observed history (last / next sighting) for the UI."""
        mode = self.resolve_mode(mode)
        t = int(max(0, min(self.m.duration, t)))
        st = next((p for p in self.player_states(t, mode) if p["player_id"] == pid), None)
        if st is None:
            return {"player_id": pid, "on_pitch": False}
        j = self.m.idx[pid]
        obs = [(k, f["xy"][j]) for k, f in enumerate(self.frames(mode)) if (f.get("obs") or [None] * len(f["xy"]))[j] is True]
        before = [o for o in obs if o[0] <= t]
        after = [o for o in obs if o[0] > t]
        hist = {"last_seen_t": before[-1][0] if before else None, "last_seen_position": before[-1][1] if before else None,
                "seconds_since_seen": (t - before[-1][0]) if before else None,
                "observed_share_so_far": round(len(before) / max(1, t + 1), 2)}
        if mode == "replay":  # retrospective only: the next sighting is future information
            hist.update({"next_seen_t": after[0][0] if after else None, "next_seen_position": after[0][1] if after else None})
        review = {}
        if self.m.meta.get("source", {}).get("type") == "computer_vision":
            from backend.reconstruction.measurement_review import projected_motion_review
            review["measurement_diagnostic"] = projected_motion_review(self.frames(mode), j, t)
        return {**st, "timestamp": t, "mode": mode, "history": hist,
                **review,
                "estimator": (self.recon.get(mode).meta.get("estimator", "formation") if self.recon.get(mode)
                              else self.m.meta.get("source", {}).get("estimator", self.view))}

    def frames(self, mode: str = "replay"):
        """Frames for rendering; each carries "sl": every player's line in the current formation (-1 = none)."""
        mode = self.resolve_mode(mode)
        if mode in self._frames_cache:
            return self._frames_cache[mode]
        if mode in self.alt_frames:
            fr = self.alt_frames[mode]
        elif mode == "truth" and self.truth_m is not None:
            fr = frames_from(self.truth_m)
        elif mode in ("replay", "live") and mode in self.recon:
            fr = frames_from(self.truth_m, self.recon[mode], self.obs.cut)
        else:
            fr = self.m.frames
        if self.m.meta.get("source", {}).get("type") == "computer_vision":
            if mode != 'live':
                from backend.analytics.deadball import dead_ball_frames
                fr = dead_ball_frames(fr, self.m.events)
            # Tracking-domain confidence excludes perception/identity error.
            # Retain raw values in the private export, not as video probabilities, unless the estimates' confidence
            # was validated on this kind of input (source.uncertainty_validated): then keep it for estimated players.
            keep = getattr(self, "uncertainty_validated", False)
            fr = [{**f, "conf": [c if keep and o is False else None for c, o in zip(f.get("conf") or [None] * len(f["xy"]),
                                                                              f.get("obs") or [None] * len(f["xy"]))],
                   "rad": [None if observed else radius for observed,radius in zip(
                       f.get("obs") or [False]*len(f["xy"]), f.get("rad") or [None]*len(f["xy"]))]}
                  for f in fr]
        for f, sl in zip(fr, lines_per_second(fr, self.m.order, self.m.is_gk, self.timeline, causal=mode == "live")):
            f["sl"] = sl
        self._frames_cache[mode] = fr
        return fr

    def evaluation(self):
        try:
            rep = json.load(open(EVAL_REPORT))
        except Exception:
            return {"error": "no evaluation report yet: run .venv/bin/python -m backend.reconstruction.evaluate"}
        try:  # many-match evaluation (backend.reconstruction.multi_eval): aggregate only
            multi = json.load(open(EVAL_REPORT.replace("report.json", "multi_report.json")))
            rep["multi"] = {k: multi[k] for k in ("matches", "failed", "calibration_seed", "note", "aggregate")}
        except Exception:
            pass
        real = {}
        for f in sorted(glob.glob(os.path.join(os.path.dirname(EVAL_REPORT), "real", "ml_results_f*.json"))):
            r = json.load(open(f))
            from backend.reconstruction.real_benchmark import split
            res = r.get("results") or (r["test"] if isinstance(r.get("test"), dict) else None)
            real[f"fold{r['fold']}"] = {"train": r["train"], "val": r["val"], "test": r.get("test_match") or split(r["fold"])[2],
                                        "results": res}
        syn = os.path.join(os.path.dirname(EVAL_REPORT), "real", "ml_synthetic_f0.json")
        if real:
            rep["real"] = {"dataset": "DFL/IDSSE open data (CC BY 4.0), 7 matches, TRACAB ground truth; masks from SkillCorner broadcast footprints (MIT)",
                           "folds": real}
        if os.path.exists(syn):
            rep["ml_on_synthetic"] = json.load(open(syn))
        rdir = os.path.join(os.path.dirname(EVAL_REPORT), "real")
        for key, fname in (("real_lomo", "lomo_report.json"), ("real_tactics", "tactics_real.json"), ("reveals", "reveals_report.json"),
                           ("clip", "clip_reveals.json")):
            fp = os.path.join(rdir, fname)
            if os.path.exists(fp):
                rep[key] = json.load(open(fp))
        return rep

    # ---------------------------------------------------------------- queries
    def state_at(self, t: float) -> dict:
        t = max(0, min(self.m.duration, t))
        k = min(int(t // 60), len(self.timeline["A"]) - 1)
        f = self.m.frames[int(t)]
        teams = {}
        for team, opp in (("A", "B"), ("B", "A")):
            row, orow = self.timeline[team][k], self.timeline[opp][k]
            teams[team] = {
                "tactical_state": tactical_label(row, orow),
                "formation": row["formation"],
                "press_intensity": row["press_index"],
                "def_line_m": row["def_line_s"], "mid_line_m": row["mid_line_s"], "width_m": row["width_s"],
                "depth_m": row["depth_s"], "compactness_m": row["compactness_s"],
                "possession_pct": round(100 * row["possession_s"], 1), "territory_pct": round(100 * row["territory_s"], 1),
                "rolling_5min": window_metrics(self.timeline, team, max(0, k - 4), k + 1),
            }
            cov = self.coverage[team][k]
            teams[team]["visibility"] = cov
            if cov["abstain"]:
                teams[team]["tactical_state"] = "INSUFFICIENT_VISIBILITY"
        live = [mo for mo in self.moments if mo["timestamp"] <= t]
        ps = self.player_states(t)
        n_obs = sum(p["state_source"] == "observed" for p in ps)
        hc = sum(1 for p in ps if p["state_source"] == "inferred" and (p.get("confidence") or 0) >= 0.7)
        cov_val = (n_obs + sum((p.get("confidence") or 0) for p in ps if p["state_source"] == "inferred")) / max(1, len(ps))
        perception_unverified = self.m.meta.get("source", {}).get("type") == "computer_vision"
        return {"timestamp": t, "minute": k, "score": self.m.score_at(t), "possession": f["p"],
                "ball": f["b"], "teams": teams, "view": self.view,
                "tactical_analysis_mode": "retrospective",
                "tactical_analysis_basis": "Full-minute metrics and sequence-smoothed five-minute formations can use later frames; these summaries are not live predictions",
                "coverage": {"observed_players": n_obs, "inferred_players": len(ps) - n_obs,
                             "high_confidence_inferred": hc, "coverage": round(cov_val, 3), "on_pitch": len(ps),
                             "camera_cut": bool(f.get("cut", False)),
                             "perception_unverified": perception_unverified,
                             "tactical_reliability": "unverified" if perception_unverified else "high" if cov_val >= 0.85 else "reduced" if cov_val >= 0.6 else "insufficient"},
                "players": ps,
                "active_moment_id": live[-1]["id"] if live else None}

    def player_profile(self, pid: str, t: Optional[float] = None, mode: str = "default") -> dict:
        mode = self.resolve_mode(mode)
        t = self.m.duration if t is None else int(max(0, min(self.m.duration, t)))
        p = self.m.player_by_id[pid]
        # Use only the selected view's prefix, rather than cached truth movement
        # or the remainder of the current minute. Events are prefix-filtered too.
        selected_frames = self.frames(mode)[:int(t) + 1]
        profile_match = MatchData({**self.m.raw, "meta": {**self.m.meta, "duration": t},
                                  "frames": selected_frames,
                                  "events": [e for e in self.m.events if e["timestamp"] <= t]})
        selected_speeds = speeds(profile_match)
        obs = np.array([[o is True for o in (f.get("obs") or [True] * len(f["xy"]))]
                        for f in selected_frames])
        both = obs & np.vstack([obs[:1], obs[:-1]])
        observed_speeds = np.where(both, selected_speeds, np.nan)
        mv = player_movement(profile_match, selected_speeds, int(t))[pid]
        mo = player_movement(profile_match, observed_speeds, int(t))[pid]
        mv = {**mv, "top_speed_kmh": mo["top_speed_kmh"], "sprint_seconds": mo["sprint_seconds"],
              "note": "distance includes estimated off-screen movement; top speed and sprints use observed seconds only"}
        st = player_event_stats(profile_match, t)[pid]
        # Precomputed tactical moments can use future windows. Keep their links
        # in retrospective profiles; live profiles rely on observed events only.
        mos = [] if mode == "live" else [mo for mo in self.moments if mo["timestamp"] <= t and (
            pid in mo["replay"]["focus_players"] or mo.get("player_id") == pid)]
        j = self.m.idx[pid]
        seen = [(f.get("obs") or [True] * len(f["xy"]))[j] if f["xy"][j] is not None else None for f in selected_frames]
        on = [o for o in seen if o is not None]
        visibility = {"observed_seconds": sum(1 for o in on if o), "inferred_seconds": sum(1 for o in on if not o),
                      "observed_share": round(sum(1 for o in on if o) / max(1, len(on)), 2)}
        narrative = self.gen.player_summary(p, st, mos)
        headline_kind = "LIVE ESTIMATES" if mode == "live" else "CLIP ESTIMATES" if self.m.meta.get("source", {}).get("type") == "computer_vision" else "REPLAY SUMMARY" if mode == "replay" else "PLAYER SUMMARY"
        narrative = {**narrative, "headline": f"{p['name'].upper()} — {headline_kind}"}
        return {"player": p, "until": t, "mode": mode, "movement": mv, "stats": st, "visibility": visibility,
                "moment_scope": "Precomputed retrospective moments are excluded from live profiles" if mode == "live" else "retrospective",
                "moments": [{"id": mo["id"], "title": mo["title"], "minute_label": mo["minute_label"]} for mo in mos],
                "narrative": narrative}

    def summary(self, until: Optional[float] = None) -> dict:
        """Whole-match (or up to `until`, for a live match) analysis: team stats, players, goals, highlights."""
        T = self.m.duration if until is None else int(max(0, min(self.m.duration, until)))
        evs = [e for e in self.m.events if e["timestamp"] <= T]
        offset = self.m.meta.get("clock_offset") or 0
        poss = self.m.poss[:T + 1]
        has_truth = self.view == "truth"
        teams = {}
        for tm in ("A", "B"):
            mine = [e for e in evs if e.get("team_id") == tm]
            kind = lambda *k: [e for e in mine if e["event_type"] in k]
            passes = kind("PASS")
            shots = kind("SHOT")
            on = self.m.team_mask(tm, outfield=False)
            with np.errstate(all="ignore"):
                dist = float(np.nansum(self.speeds[:T + 1][:, on])) / 1000
            rows = [r for r in self.timeline[tm] if r["minute"] * 60 <= T]
            forms = [r.get("formation") for r in rows if r.get("formation") not in (None, "?")]
            teams[tm] = {
                "goals": len(kind("GOAL")), "shots": len(shots), "shots_on_target": sum(1 for e in shots if e.get("on_target")),
                "xg": round(sum(e.get("xg") or 0 for e in shots), 2), "passes": len(passes),
                "passes_completed": sum(1 for e in passes if e.get("success")),
                "pass_accuracy": round(sum(1 for e in passes if e.get("success")) / max(1, len(passes)), 3),
                "tackles_won": sum(1 for e in kind("TACKLE") if e.get("success", True)),
                "fouls": len(kind("FOUL")), "corners": len(kind("CORNER")), "offsides": len(kind("OFFSIDE")),
                "yellow_cards": sum(1 for e in kind("CARD") if "Yellow" in (e.get("label") or "")),
                "red_cards": len(kind("RED_CARD")), "possession_seconds": int((poss == (0 if tm == "A" else 1)).sum()),
                "distance_km": round(dist, 1) if has_truth else None,
                "press_index": round(float(np.mean([r["press_index"] for r in rows])), 1) if rows else None,
                "formation": max(set(forms), key=forms.count) if forms else None,
                "formations": [{"from_minute": r["minute"], "formation": r["formation"]} for i, r in enumerate(rows)
                               if r.get("formation") not in (None, "?") and (i == 0 or rows[i - 1].get("formation") != r.get("formation"))],
            }
            # statistics the source event feed does not record are unknown, not zero
            for k in (self.m.meta.get("source") or {}).get("unavailable_stats") or []:
                teams[tm][k] = None
        players = []
        for p in self.m.players:
            pr = self.player_profile(p["id"], T)
            if pr["visibility"]["observed_seconds"] + pr["visibility"]["inferred_seconds"] == 0:
                continue
            players.append({"id": p["id"], "name": p["name"], "team": p["team"], "shirt": p.get("shirt"),
                            "position": p.get("position"), **{k: pr["stats"].get(k, 0) for k in ("passes", "completed", "tackles", "shots", "goals", "assists", "xg")},
                            "distance_km": pr["movement"]["distance_km"] if has_truth else None,
                            "top_speed_kmh": pr["movement"]["top_speed_kmh"] if has_truth else None})
        clock = lambda t: f"{int((t + offset) // 60) + 1}'"
        goals = [{"timestamp": e["timestamp"], "minute": clock(e["timestamp"]), "team": e["team_id"],
                  "scorer": self.m.player_by_id.get(e["player_id"], {}).get("name") if e.get("player_id") else None,
                  "assist": self.m.player_by_id.get(e.get("assist_id") or "", {}).get("name")} for e in evs if e["event_type"] == "GOAL"]
        cards = [{"timestamp": e["timestamp"], "minute": clock(e["timestamp"]), "team": e["team_id"],
                  "player": self.m.player_by_id.get(e["player_id"], {}).get("name") if e.get("player_id") else None,
                  "card": "red" if e["event_type"] == "RED_CARD" else "yellow"} for e in evs if e["event_type"] in ("CARD", "RED_CARD")]
        subs = [{"timestamp": e["timestamp"], "minute": clock(e["timestamp"]), "team": e["team_id"],
                 "on": self.m.player_by_id.get(e["player_on_id"], {}).get("name"),
                 "off": self.m.player_by_id.get(e["player_off_id"], {}).get("name")} for e in evs if e["event_type"] == "SUBSTITUTION"]
        highlights = []
        for mo in sorted(self.moments, key=lambda m: m["timestamp"]):
            if mo.get("available_at", mo["timestamp"]) > T:  # live: only moments whose evidence is all in the past
                continue
            n = mo.get("narrative") or {}
            highlights.append({"id": mo["id"], "timestamp": mo["timestamp"], "end_timestamp": mo["end_timestamp"],
                               "minute_label": mo["minute_label"], "type": mo["type"], "team": mo["team"], "title": mo["title"],
                               "importance": mo["importance"], "confidence": mo["confidence"],
                               "headline": (n.get("fan") or {}).get("headline"), "summary": (n.get("analyst") or {}).get("summary"),
                               "bullets": (n.get("analyst") or {}).get("bullets", []), "why": n.get("why_it_matters"),
                               "coverage": mo.get("coverage")})
        chapters_ = [{**c, "end": min(c["end"], T)} for c in self.story["chapters"] if c["start"] <= T]
        changes = {tm: [c for c in self.changes.get(tm, []) if c["available_at"] <= T or T >= self.m.duration] for tm in ("A", "B")}
        return {"until": T, "duration": self.m.duration, "clock_offset": offset, "view": self.view, "teams": teams, "changes": changes,
                "players": players, "goals": goals, "cards": cards, "substitutions": subs, "highlights": highlights,
                "chapters": chapters_, "score": self.m.score_at(T)}

    def ground_truth(self):
        return {"script": None if self.source else GROUND_TRUTH,
                "detected": {"high_press": [{k: v for k, v in p.items() if k not in ("final_third_recovery_chains", "bypasses")} for p in self.presses],
                             "shifts": self.shifts, "chapters": chapters(self.timeline),
                             "counterattacks_with_shot": [c for c in self.counters if c["outcome"] != "lost"]}}


_cache = {}
_cache_locks = {}
_cache_guard = Lock()
_DEFAULT = object()


def get_intel(seed: int = 7, source=_DEFAULT) -> MatchIntelligence:
    """Default match: MATCH_FILE=data/....json, else the synthetic match. The lobby passes `source` explicitly
    (a match file path, or None for the synthetic match with this seed)."""
    src = (os.getenv("MATCH_FILE") or None) if source is _DEFAULT else source
    src = str(Path(src).resolve()) if src else None
    identity = (seed, src)
    # Six parallel studio endpoints must share one cold construction. Other
    # matches can still load independently; failures are never cached.
    with _cache_guard:
        lock = _cache_locks.setdefault(identity, Lock())
    with lock:
        for _ in range(3):
            stamp = Path(src).stat() if src else None
            version = (stamp.st_mtime_ns, stamp.st_size, stamp.st_ino) if stamp else None
            key = (*identity, version)
            if key in _cache:
                return _cache[key]
            value = None
            if src is None:  # synthetic: the shipped precomputed analysis when it was built by this code
                from backend import precomputed
                value = precomputed.load(seed)
            value = value or MatchIntelligence(seed, src)
            after = Path(src).stat() if src else None
            if stamp and (after.st_mtime_ns, after.st_size, after.st_ino) != version:
                continue  # Never label a changed file as the version read earlier.
            with _cache_guard:
                for previous in list(_cache):
                    if previous[:2] == identity:
                        del _cache[previous]
                _cache[key] = value
            return value
    raise RuntimeError('Match file repeatedly changed during analysis; retry after export completes')
