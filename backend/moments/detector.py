"""Moment detection: turns detections + key events into explainable moments.

Each moment answers WHAT HAPPENED -> WHAT CHANGED -> WHAT DID IT CAUSE -> WHY IT MATTERED.
Every number placed in a moment comes from the analytics layer (never invented).
"""
from typing import Dict, List, Optional

import numpy as np

from backend.analytics.passing import is_backward
from backend.models.match import MatchData, other, to_own
from backend.tactics.tactical_state import window_metrics

TEAM_NAMES = {"A": "Ashford City", "B": "Belmont Rovers"}


CLOCK_OFFSET = 0.0  # real-match clips start mid-game; set per match by MomentDetector


def clock(t: float) -> str:
    t += CLOCK_OFFSET
    return f"{int(t // 60)}:{int(t % 60):02d}"


def minute(t: float) -> str:
    return f"{int((t + CLOCK_OFFSET) // 60) + 1}'"


class MomentDetector:
    def __init__(self, m: MatchData, timeline, seqs, presses, counters, shifts):
        self.m, self.tl, self.seqs = m, timeline, seqs
        self.presses, self.counters, self.shifts = presses, counters, shifts
        self.moments: List[dict] = []
        self.names = {p["id"]: p["name"] for p in m.players}
        global CLOCK_OFFSET
        CLOCK_OFFSET = float(m.meta.get("clock_offset") or 0)
        TEAM_NAMES.update({t: v["name"] for t, v in m.meta["teams"].items()})

    # ------------------------------------------------------------------ helpers
    def name(self, pid):
        return self.names.get(pid, pid) or "unidentified player"

    def ev(self, eid) -> Optional[dict]:
        return next((e for e in self.m.events if e["id"] == eid), None)

    def wm(self, team, m0, m1):
        return window_metrics(self.tl, team, m0, m1)

    def nearest_players(self, team, t, k=3):
        t = int(min(max(t, 0), self.m.T - 1))
        mask = self.m.team_mask(team)
        ids = [pid for pid, j in self.m.idx.items() if mask[j]]
        b = self.m.ball[t]
        d = [(np.linalg.norm(self.m.xy[t, self.m.idx[p]] - b), p) for p in ids
             if not np.isnan(self.m.xy[t, self.m.idx[p], 0])]
        return [p for _, p in sorted(d)[:k]]

    def players_bypassed(self, defending_team, t):
        """Defending outfield players caught upfield of the ball (in their own frame)."""
        t = int(min(max(t, 0), self.m.T - 1))
        mask = self.m.team_mask(defending_team)
        own = self.m.own_xy(defending_team)[t][mask]
        bx = self.m.own_ball(defending_team)[t, 0]
        return int(np.sum(own[:, 0] > bx + 1))

    def seq_for_goal(self, goal):
        return next((s for s in self.seqs if s["goal"] and s["goal"]["id"] == goal["id"]), None)

    def add(self, mo):
        mo["id"] = f"m{len(self.moments):02d}-{mo['type'].lower()}"
        mo["minute_label"] = minute(mo["timestamp"])
        mo["clock"] = clock(mo["timestamp"])
        mo["importance"] = round(float(np.clip(mo["importance"], 0.2, 0.99)), 2)
        mo["confidence"] = round(float(np.clip(mo["confidence"], 0.4, 0.98)), 2)
        mo.setdefault("links", [])
        self.moments.append(mo)
        return mo

    @staticmethod
    def delta(before, after, keys):
        return {k: round(after[k] - before[k], 1) for k in keys
                if after.get(k) is not None and before.get(k) is not None}

    # ------------------------------------------------------------------ detectors
    def run(self) -> List[dict]:
        self.goal_moments()
        self.press_moments()
        self.counter_moments()
        self.sub_moments()
        self.shift_moments()
        self.red_card_moments()
        # a press onset that coincides with a formation change is the same story: keep the shift
        shifts = [mo for mo in self.moments if mo["type"] == "FORMATION_SHIFT"]
        self.moments = [mo for mo in self.moments if not (
            mo.get("subtype") == "high_press" and any(
                s["team"] == mo["team"] and abs(s["timestamp"] - mo["timestamp"]) <= 180 for s in shifts))]
        self.moments.sort(key=lambda x: x["timestamp"])
        for i, mo in enumerate(self.moments):
            mo["id"] = f"m{i:02d}-{mo['type'].lower().replace('_', '-')}"
        return self.moments

    def goal_moments(self):
        for g in [e for e in self.m.events if e["event_type"] == "GOAL"]:
            team, t = g["team_id"], g["timestamp"]
            seq = self.seq_for_goal(g)
            shot = next((e for e in self.m.events if e["event_type"] == "SHOT" and 0 <= t - e["timestamp"] <= 1.0
                         and e["team_id"] == team), {"xg": g.get("xg"), "timestamp": t})
            rec_t = seq["start"] if seq else t - 10
            rec_ev = self.ev(seq["start_event"]) if seq else None
            opp = other(team)
            opp_back = [p for p in self.m.events_between(rec_t - 12, rec_t + 0.1, ("PASS",), opp) if is_backward(p)]
            passes = seq["passes"] if seq else []
            counter = next((c for c in self.counters if c.get("goal") == g["id"]), None)
            press = next((p for p in self.presses if p["team"] == team and p["start"] <= rec_t < p["end"]), None)
            context = ("counterattack" if counter else "high press" if press else "sustained possession")
            rec_x = to_own(team, rec_ev["x"], rec_ev["y"])[0] if rec_ev and "x" in rec_ev else None
            score = g["score"]
            evidence = []
            if press:
                evidence.append(self._evi("Tactical context", f"{TEAM_NAMES[team]} high press active",
                                          press["peak_press_index"], "press index", press["start"]))
            if opp_back:
                evidence.append(self._evi("Forced backwards", f"{TEAM_NAMES[opp]} made {len(opp_back)} backward passes before losing the ball",
                                          len(opp_back), "passes", opp_back[0]["timestamp"]))
            if rec_ev and seq["cause"] in ("tackle", "interception"):
                evidence.append(self._evi("Recovery", f"{self.name(rec_ev['player_id'])} won the ball ({seq['cause']}) {round(rec_x)}m from own goal",
                                          round(rec_x, 1), "m", rec_t))
            if counter:
                evidence.append(self._evi("Transition", f"Ball moved {counter['progression_m']}m in {counter['seconds']}s with {counter['passes']} pass(es)",
                                          counter["speed_ms"], "m/s", rec_t))
                byp = self.players_bypassed(opp, t - 2)
                evidence.append(self._evi("Exposed defence", f"{byp} {TEAM_NAMES[opp]} outfield players caught upfield of the ball",
                                          byp, "players", t - 2))
            else:
                evidence.append(self._evi("Build-up", f"{len(passes)} passes in {round(t - rec_t, 1)}s", len(passes), "passes", rec_t))
            if shot.get("location_known", True):
                evidence.append(self._evi("Shot", f"{self.name(g['player_id'])} shot from {shot['distance']}m", shot["xg"], "xG", t))
            else:
                evidence.append(self._evi("Shot", "Shot location not visible on the live camera (replay / close-up)", "n/a", "", t))
            evidence.append(self._evi("Outcome", f"Goal - {score['A']}-{score['B']}", 1, "goal", t))
            involved = sorted({g["player_id"], *(p["player_id"] for p in passes), *([g["assist_id"]] if g.get("assist_id") else []),
                               *([rec_ev["player_id"]] if rec_ev else [])})
            won = seq and seq["cause"] in ("tackle", "interception")
            chain = [{"stage": "EVENT", "label": f"{seq['cause'].title()} by {self.name(seq['player_id'])}" if won else "Spell of possession", "timestamp": rec_t}]
            if press:
                chain.insert(0, {"stage": "TACTICAL_CHANGE", "label": "High press", "timestamp": press["start"]})
            if opp_back:
                chain.insert(len(chain) - 1, {"stage": "STATE_CHANGE", "label": f"{len(opp_back)} forced backward passes", "timestamp": opp_back[0]["timestamp"]})
            chain += [{"stage": "CONSEQUENCE", "label": f"{len(passes)} passes, {round(t - rec_t, 1)}s", "timestamp": rec_t},
                      {"stage": "OUTCOME", "label": f"Goal ({shot['xg']} xG)", "timestamp": t}]
            self.add({
                "type": "GOAL_BUILD_UP", "team": team, "timestamp": t, "end_timestamp": t,
                "title": f"Goal {TEAM_NAMES[team]} - {self.name(g['player_id'])}",
                "subtype": context,
                "importance": 0.82 + 0.1 * (abs(score["A"] - score["B"]) <= 1) + 0.05 * (t > 4500),
                "confidence": 0.95,
                "trigger": {"event_type": "GOAL", "event_id": g["id"], "description": f"{self.name(g['player_id'])} scores"},
                "before": {"score": {k: v - (1 if k == team else 0) for k, v in score.items()}, "recovery_own_x": rec_x,
                           "opp_backward_passes": len(opp_back)},
                "after": {"score": score, "xg": shot["xg"], "passes": len(passes), "duration_s": round(t - rec_t, 1)},
                "deltas": {}, "evidence": evidence,
                "consequences": [f"Score {score['A']}-{score['B']}"],
                "outcome": {"type": "GOAL", "event_id": g["id"], "timestamp": t, "description": f"Goal {score['A']}-{score['B']}"},
                "chain": chain,
                "replay": {"start": max(0, min(rec_t - 10, t - 20)), "end": t + 4, "focus_players": involved,
                           "focus_team": team, "overlay": "counter" if counter else "press" if press else "shape"},
                "related_events": [g["id"], shot["id"]] + ([rec_ev["id"]] if rec_ev else []) + [p["id"] for p in passes],
                "window": [rec_t, t],
            })

    def press_moments(self):
        for p in self.presses:
            team, m0, m1 = p["team"], p["start_min"], p["end_min"]
            opp = other(team)
            before, after = self.wm(team, max(0, m0 - 6), m0), self.wm(team, m0, m1)
            ob, oa = self.wm(opp, max(0, m0 - 6), m0), self.wm(opp, m0, m1)
            chains = p["final_third_recovery_chains"]
            shot_chains = [c for c in chains if c["shot"]]
            goals = [c for c in chains if c["goal"]]
            mins_b, mins_a = max(1, m0 - max(0, m0 - 6)), max(1, m1 - m0)
            rec_rate_b = before["final_third_recoveries"] / mins_b * 10
            rec_rate_a = after["final_third_recoveries"] / mins_a * 10
            evidence = [
                self._evi("Tactical trigger", f"{TEAM_NAMES[team]} pressing intensity {before['press_intensity']} -> {after['press_intensity']}",
                          round(after["press_intensity"] - before["press_intensity"], 1), "index pts", p["start"]),
                self._evi("Structural change", f"Defensive line (out of possession) advanced {round(after['def_line_oop_m'] - before['def_line_oop_m'], 1)}m",
                          round(after["def_line_oop_m"] - before["def_line_oop_m"], 1), "m", p["start"]),
                self._evi("Immediate effect", f"{TEAM_NAMES[opp]} backward passes: {round(ob['opp_backward_passes'] / mins_b * 10, 1)} -> {round(oa['opp_backward_passes'] / mins_a * 10, 1)} per 10 min",
                          oa["opp_backward_passes"], "passes", p["start"]),
                self._evi("Recovery", f"{after['final_third_recoveries']} final-third recoveries ({round(rec_rate_b, 1)} -> {round(rec_rate_a, 1)} per 10 min)",
                          after["final_third_recoveries"], "recoveries", p["start"]),
            ]
            if shot_chains:
                c = shot_chains[0]
                evidence.append(self._evi("Outcome", f"{len(shot_chains)} shots came directly from final-third recoveries; first {c['seconds_to_shot']}s after the regain",
                                          len(shot_chains), "shots", c["timestamp"]))
            if goals:
                evidence.append(self._evi("Outcome", f"Goal scored from a pressing regain at {clock(goals[0]['timestamp'])}", len(goals), "goals", goals[0]["timestamp"]))
            if p["bypasses"]:
                evidence.append(self._evi("Risk", f"Press bypassed {len(p['bypasses'])} time(s)", len(p["bypasses"]), "bypasses", p["bypasses"][0]["start"]))
            demo = goals[0] if goals else (shot_chains[0] if shot_chains else (chains[0] if chains else None))
            rp_t = demo["timestamp"] if demo else p["start"] + 60
            chain = [{"stage": "TACTICAL_CHANGE", "label": "High press", "timestamp": p["start"]},
                     {"stage": "STATE_CHANGE", "label": f"Line +{round(after['def_line_oop_m'] - before['def_line_oop_m'], 1)}m", "timestamp": p["start"]},
                     {"stage": "CONSEQUENCE", "label": f"Opponent forced backward ({oa['opp_backward_passes']} back passes)", "timestamp": p["start"]},
                     {"stage": "CONSEQUENCE", "label": f"{after['final_third_recoveries']} final-third recoveries", "timestamp": p["start"]}]
            if goals:
                chain.append({"stage": "OUTCOME", "label": "Goal", "timestamp": goals[0]["timestamp"]})
            elif shot_chains:
                chain.append({"stage": "OUTCOME", "label": f"{len(shot_chains)} shots", "timestamp": shot_chains[0]["timestamp"]})
            self.add({
                "type": "TACTICAL_TURNING_POINT", "subtype": "high_press", "team": team,
                "timestamp": p["start"], "end_timestamp": p["end"],
                "title": f"{TEAM_NAMES[team]} switch on the high press" if p["start"] < 3600 else f"{TEAM_NAMES[team]} squeeze the game",
                "importance": 0.55 + 0.15 * bool(goals) + min(0.15, len(shot_chains) * 0.03) + min(0.1, (p["peak_press_index"] - 55) / 300),
                "confidence": 0.6 + min(0.3, (p["peak_press_index"] - 55) / 120) + 0.05 * bool(shot_chains),
                "trigger": {"event_type": "TACTICAL_CHANGE", "description": "Pressing intensity crossed the high-press threshold"},
                "before": before, "after": after,
                "deltas": self.delta(before, after, ["press_intensity", "def_line_oop_m", "territory_pct", "final_third_recoveries", "shots", "xg", "possession_pct"]),
                "evidence": evidence,
                "consequences": [f"{after['final_third_recoveries']} final-third recoveries", f"{after['shots']} shots ({after['xg']} xG)"]
                + ([f"{len(p['bypasses'])} press bypass(es) conceded"] if p["bypasses"] else []),
                "outcome": ({"type": "GOAL", "event_id": goals[0]["goal"], "timestamp": goals[0]["timestamp"], "description": "Goal from pressing regain"} if goals else
                            {"type": "SHOTS", "timestamp": shot_chains[0]["timestamp"], "description": f"{len(shot_chains)} shots from regains"} if shot_chains else
                            {"type": "TERRITORY", "description": f"Territory {after['territory_pct']}%"}),
                "chain": chain,
                "replay": {"start": max(0, rp_t - 18), "end": rp_t + (demo["seconds_to_shot"] or 6 if demo else 10) + 4,
                           "focus_players": self.nearest_players(team, rp_t, 4), "focus_team": team, "overlay": "press"},
                "related_events": [c["recovery_event"] for c in chains],
                "window": [p["start"], p["end"]],
                "detection": {k: v for k, v in p.items() if k not in ("final_third_recovery_chains",)},
            })
            for b in p["bypasses"]:
                if b["shot"]:
                    self.pressure_break(team, b, after)

    def pressure_break(self, press_team, b, press_state):
        team = other(press_team)
        shot = self.ev(b["shot"])
        t0 = b["start"]
        seq = next(s for s in self.seqs if s["start"] == t0 and s["team"] == team)
        reach_t = t0 + b["seconds_to_final_third"]
        byp = self.players_bypassed(press_team, reach_t)
        players = sorted({seq["player_id"], *(p["player_id"] for p in seq["passes"]),
                          *(p["receiver_id"] for p in seq["passes"] if p.get("receiver_id")), shot["player_id"]})
        long_ball = max(seq["passes"], key=lambda p: p["distance"]) if seq["passes"] else None
        evidence = [
            self._evi("Context", f"{TEAM_NAMES[press_team]} pressing at {press_state['press_intensity']} with line at {press_state['def_line_oop_m']}m",
                      press_state["press_intensity"], "press index", t0),
            self._evi("Escape", f"{self.name(long_ball['player_id'])} played a {round(long_ball['distance'])}m pass over the press" if long_ball else "Press beaten on the dribble",
                      round(long_ball["distance"], 1) if long_ball else 0, "m", long_ball["timestamp"] if long_ball else t0),
            self._evi("Exposed space", f"{byp} pressing players taken out of the game", byp, "players", reach_t),
            self._evi("Speed", f"Final third reached in {b['seconds_to_final_third']}s", b["seconds_to_final_third"], "s", reach_t),
            self._evi("Outcome", f"Shot by {self.name(shot['player_id'])} ({shot['outcome'].replace('_', ' ')})", shot["xg"], "xG", shot["timestamp"]),
        ]
        self.add({
            "type": "PRESSURE_BREAK", "team": team, "timestamp": shot["timestamp"], "end_timestamp": shot["timestamp"],
            "title": f"{TEAM_NAMES[team]} play through the press",
            "importance": 0.5 + shot["xg"] + 0.015 * byp, "confidence": 0.8,
            "trigger": {"event_type": "PASS", "event_id": long_ball["id"] if long_ball else None, "description": "Long ball over the press"},
            "before": {"press_intensity": press_state["press_intensity"], "def_line_oop_m": press_state["def_line_oop_m"]},
            "after": {"players_bypassed": byp, "seconds_to_final_third": b["seconds_to_final_third"], "xg": shot["xg"]},
            "deltas": {}, "evidence": evidence,
            "consequences": [f"{byp} pressers bypassed", "Shot conceded"],
            "outcome": {"type": "SHOT", "event_id": shot["id"], "timestamp": shot["timestamp"], "description": f"Shot {shot['outcome']}"},
            "chain": [{"stage": "TACTICAL_CHANGE", "label": "Opponent high press", "timestamp": t0},
                      {"stage": "EVENT", "label": "Long ball over the press", "timestamp": long_ball["timestamp"] if long_ball else t0},
                      {"stage": "STATE_CHANGE", "label": f"{byp} pressers bypassed", "timestamp": reach_t},
                      {"stage": "CONSEQUENCE", "label": "Midfield exposed", "timestamp": reach_t},
                      {"stage": "OUTCOME", "label": f"Shot conceded ({shot['xg']} xG)", "timestamp": shot["timestamp"]}],
            "replay": {"start": max(0, t0 - 8), "end": shot["timestamp"] + 4, "focus_players": players, "focus_team": team, "overlay": "counter"},
            "related_events": [shot["id"]] + [p["id"] for p in seq["passes"]],
            "window": [t0, shot["timestamp"]],
        })

    def counter_moments(self):
        for c in self.counters:
            if c["goal"] or c["outcome"] != "shot" or c["xg"] < 0.1:
                continue  # goals are covered by GOAL_BUILD_UP
            if any(mo["type"] == "PRESSURE_BREAK" and abs(mo["window"][0] - c["start"]) < 3 for mo in self.moments):
                continue
            team, opp = c["team"], other(c["team"])
            shot = self.ev(c["shot"])
            t0 = c["start"]
            k = int(t0 // 60)
            opp_line = self.tl[opp][k]["def_line"]
            byp = self.players_bypassed(opp, t0 + c["seconds"])
            self.add({
                "type": "COUNTERATTACK", "team": team, "timestamp": shot["timestamp"], "end_timestamp": shot["timestamp"],
                "title": f"{TEAM_NAMES[team]} counterattack",
                "importance": min(0.8, 0.4 + 0.6 * shot["xg"] + 0.01 * byp), "confidence": 0.78,
                "trigger": {"event_type": "POSSESSION_CHANGE", "description": f"Regain by {self.name(c['recovery_player'])}"},
                "before": {"recovery_own_x": c["recovery_own_x"], "opp_def_line_m": opp_line},
                "after": {"progression_m": c["progression_m"], "seconds": c["seconds"], "passes": c["passes"], "xg": shot["xg"]},
                "deltas": {},
                "evidence": [
                    self._evi("Recovery", f"{self.name(c['recovery_player'])} regains {round(c['recovery_own_x'])}m from own goal", c["recovery_own_x"], "m", t0),
                    self._evi("Opponent shape", f"{TEAM_NAMES[opp]} defensive line at {opp_line}m", opp_line, "m", t0),
                    self._evi("Transition", f"{c['progression_m']}m in {c['seconds']}s, {c['passes']} pass(es)", c["speed_ms"], "m/s", t0),
                    self._evi("Exposed defence", f"{byp} players caught upfield", byp, "players", t0 + c["seconds"]),
                    self._evi("Outcome", f"Shot by {self.name(shot['player_id'])} ({shot['outcome'].replace('_', ' ')})", shot["xg"], "xG", shot["timestamp"]),
                ],
                "consequences": [f"{byp} opponents bypassed", "Shot"],
                "outcome": {"type": "SHOT", "event_id": shot["id"], "timestamp": shot["timestamp"], "description": f"Shot {shot['outcome']}"},
                "chain": [{"stage": "EVENT", "label": "Ball regained", "timestamp": t0},
                          {"stage": "STATE_CHANGE", "label": "Opponent out of shape", "timestamp": t0},
                          {"stage": "CONSEQUENCE", "label": f"{c['progression_m']}m in {c['seconds']}s", "timestamp": t0 + c["seconds"]},
                          {"stage": "OUTCOME", "label": f"Shot ({shot['xg']} xG)", "timestamp": shot["timestamp"]}],
                "replay": {"start": max(0, t0 - 8), "end": shot["timestamp"] + 4, "focus_players": c["players"], "focus_team": team, "overlay": "counter"},
                "related_events": [shot["id"]], "window": [t0, shot["timestamp"]],
            })

    def sub_moments(self):
        for s in [e for e in self.m.events if e["event_type"] == "SUBSTITUTION"]:
            team, t = s["team_id"], s["timestamp"]
            k = int(t // 60)
            before, after = self.wm(team, k - 6, k), self.wm(team, k, k + 10)
            on = s["player_on_id"]
            acts = [e for e in self.m.events_between(t, t + 900, None, team) if e["player_id"] == on]
            pres = sum(1 for e in acts if e["event_type"] == "PRESSURE")
            rec = sum(1 for e in acts if e["event_type"] == "POSSESSION_CHANGE")
            right = sum(1 for e in acts if e["event_type"] == "PRESSURE" and to_own(team, e["x"], e["y"])[1] > 44)
            assists = [e for e in self.m.events_between(t, t + 1200, ("GOAL",), team) if e.get("assist_id") == on]
            self.add({
                "type": "SUBSTITUTION_IMPACT", "team": team, "timestamp": t, "end_timestamp": t + 900,
                "title": f"{self.name(on)} on for {self.name(s['player_off_id'])}",
                "importance": 0.45 + 0.2 * bool(assists) + min(0.15, pres / 100),
                "confidence": 0.7,
                "trigger": {"event_type": "SUBSTITUTION", "event_id": s["id"], "description": f"{self.name(s['player_off_id'])} -> {self.name(on)}"},
                "before": before, "after": after,
                "deltas": self.delta(before, after, ["press_intensity", "width_m", "def_line_oop_m", "territory_pct", "final_third_recoveries", "shots"]),
                "evidence": [
                    self._evi("Change", f"{self.name(on)} replaces {self.name(s['player_off_id'])} ({s['role']})", 1, "sub", t),
                    self._evi("Player activity", f"{pres} pressures ({right} on the right), {rec} recoveries in first 15 min", pres, "pressures", t),
                    self._evi("Team effect", f"Press intensity {before['press_intensity']} -> {after['press_intensity']}",
                              round((after["press_intensity"] or 0) - (before["press_intensity"] or 0), 1), "index pts", t),
                ] + ([self._evi("Outcome", f"Assist for the goal at {clock(assists[0]['timestamp'])}", 1, "assists", assists[0]["timestamp"])] if assists else []),
                "consequences": [f"{pres} pressures from {self.name(on)}"] + (["Assist"] if assists else []),
                "outcome": ({"type": "ASSIST", "event_id": assists[0]["id"], "timestamp": assists[0]["timestamp"], "description": "Assisted a goal"}
                            if assists else {"type": "NONE", "description": "No direct goal involvement"}),
                "chain": [{"stage": "EVENT", "label": "Substitution", "timestamp": t},
                          {"stage": "STATE_CHANGE", "label": f"Fresh legs on the {s['role']}", "timestamp": t}]
                + ([{"stage": "OUTCOME", "label": "Assist", "timestamp": assists[0]["timestamp"]}] if assists else []),
                "replay": {"start": max(0, t - 8), "end": t + 20, "focus_players": [on, s["player_off_id"]], "focus_team": team, "overlay": "shape"},
                "related_events": [s["id"]], "window": [t, t + 900], "player_id": on,
            })

    def shift_moments(self):
        used = set()
        reds = [e["timestamp"] for e in self.m.events if e["event_type"] == "RED_CARD"]
        for sh in self.shifts:
            if sh["formation_before"] == sh["formation_after"]:
                continue
            if any(abs(sh["start"] - r) <= 180 for r in reds):
                continue  # forced by the red card: explained by RED_CARD_IMPACT
            team, k = sh["team"], sh["minute"]
            t = k * 60
            before, after = self.wm(team, k - 6, k), self.wm(team, k, k + 10)
            used.add((team, k))
            # find pressing structure on the right side
            def right_share(a, b):
                ps = self.m.events_between(a, b, ("PRESSURE",), team)
                if not ps:
                    return 0.0
                return round(100 * sum(1 for e in ps if to_own(team, e["x"], e["y"])[1] > 44) / len(ps), 1)
            rb, ra = right_share(t - 360, t), right_share(t, t + 600)
            goals = self.m.events_between(t, t + 600, ("GOAL",), team)
            ev = [
                self._evi("Tactical trigger", f"Shape changed {sh['formation_before']} -> {sh['formation_after']}", sh["score"], "shift score", t),
                self._evi("Structural change", f"Defensive line {'+' if sh['deltas']['def_line'] >= 0 else ''}{sh['deltas']['def_line']}m, width {'+' if sh['deltas']['width'] >= 0 else ''}{sh['deltas']['width']}m",
                          sh["deltas"]["def_line"], "m", t),
                self._evi("Pressing", f"Press intensity {before['press_intensity']} -> {after['press_intensity']}; right-side share of pressures {rb}% -> {ra}%",
                          round((after["press_intensity"] or 0) - (before["press_intensity"] or 0), 1), "index pts", t),
                self._evi("Recovery", f"Final-third recoveries {before['final_third_recoveries']} -> {after['final_third_recoveries']}",
                          after["final_third_recoveries"], "recoveries", t),
                self._evi("Territory", f"Territory {before['territory_pct']}% -> {after['territory_pct']}%",
                          round(after["territory_pct"] - before["territory_pct"], 1), "pct pts", t),
            ]
            if goals:
                ev.append(self._evi("Outcome", f"Goal at {clock(goals[0]['timestamp'])}", 1, "goals", goals[0]["timestamp"]))
            g_seq = self.seq_for_goal(goals[0]) if goals else None
            rp = (g_seq["start"] - 12, goals[0]["timestamp"] + 4) if g_seq else (t, t + 25)
            self.add({
                "type": "FORMATION_SHIFT", "team": team, "timestamp": t, "end_timestamp": t + 600,
                "title": f"{TEAM_NAMES[team]} switch to {sh['formation_after']}",
                "importance": 0.6 + min(0.15, sh["score"] / 60) + 0.15 * bool(goals),
                "confidence": 0.62 + min(0.3, sh["score"] / 30),
                "trigger": {"event_type": "TACTICAL_CHANGE", "description": f"{sh['formation_before']} -> {sh['formation_after']}"},
                "before": {**before, "right_press_share_pct": rb}, "after": {**after, "right_press_share_pct": ra},
                "deltas": {**self.delta(before, after, ["press_intensity", "def_line_oop_m", "width_m", "territory_pct", "final_third_recoveries", "shots", "xg"]),
                           "right_press_share_pct": round(ra - rb, 1)},
                "evidence": ev,
                "consequences": [f"{after['final_third_recoveries']} final-third recoveries", f"Territory {after['territory_pct']}%", f"{after['shots']} shots"],
                "outcome": ({"type": "GOAL", "event_id": goals[0]["id"], "timestamp": goals[0]["timestamp"], "description": "Goal"} if goals
                            else {"type": "TERRITORY", "description": f"Territory {after['territory_pct']}%"}),
                "chain": [{"stage": "TACTICAL_CHANGE", "label": f"{sh['formation_before']} -> {sh['formation_after']}", "timestamp": t},
                          {"stage": "STATE_CHANGE", "label": f"Right-side press {rb}% -> {ra}%", "timestamp": t},
                          {"stage": "CONSEQUENCE", "label": f"{after['final_third_recoveries']} final-third recoveries", "timestamp": t}]
                + ([{"stage": "OUTCOME", "label": "Goal", "timestamp": goals[0]["timestamp"]}] if goals else []),
                "replay": {"start": max(0, rp[0]), "end": rp[1], "focus_players": self.nearest_players(team, rp[1] - 6, 4),
                           "focus_team": team, "overlay": "shape"},
                "related_events": [g["id"] for g in goals], "window": [t, t + 600],
                "detection": sh,
            })
        # remaining big shifts not explained by press / formation / red card -> turning points.
        # DOMINANT last: it is usually the mirror image of the opponent dropping deep.
        for sh in sorted(self.shifts, key=lambda x: x["label_after"] == "DOMINANT"):
            t = sh["start"]
            if (sh["team"], sh["minute"]) in used or any(abs(t - r) <= 180 for r in reds):
                continue
            if any(mo["type"] in ("TACTICAL_TURNING_POINT", "FORMATION_SHIFT") and abs(t - mo["timestamp"]) <= 180 for mo in self.moments):
                continue
            if sh["label_after"] not in ("LOW_BLOCK", "HIGH_PRESS", "DOMINANT") or sh["score"] < 3:
                continue
            self.turning_point(sh)

    def turning_point(self, sh):
        team, k = sh["team"], sh["minute"]
        t = k * 60
        before, after = self.wm(team, k - 6, k), self.wm(team, k, k + 12)
        counters = [c for c in self.counters if c["team"] == team and t <= c["start"] < t + 900]
        cgoals = [c for c in counters if c["goal"]]
        label = {"LOW_BLOCK": "drop into a low block", "HIGH_PRESS": "push up", "DOMINANT": "take territorial control"}[sh["label_after"]]
        g = self.ev(cgoals[0]["goal"]) if cgoals else None
        self.add({
            "type": "TACTICAL_TURNING_POINT", "subtype": sh["label_after"].lower(), "team": team, "timestamp": t, "end_timestamp": t + 900,
            "title": f"{TEAM_NAMES[team]} {label}",
            "importance": 0.5 + min(0.15, sh["score"] / 40) + 0.2 * bool(cgoals),
            "confidence": 0.6 + min(0.3, sh["score"] / 25),
            "trigger": {"event_type": "TACTICAL_CHANGE", "description": f"{sh['label_before']} -> {sh['label_after']}"},
            "before": before, "after": after,
            "deltas": self.delta(before, after, ["def_line_oop_m", "depth_m", "width_m", "possession_pct", "territory_pct", "press_intensity", "shots", "xg"]),
            "evidence": [
                self._evi("Tactical trigger", f"{'Block dropped' if sh['label_after'] == 'LOW_BLOCK' else 'Block pushed up'}: defensive line {before['def_line_oop_m']}m -> {after['def_line_oop_m']}m",
                          round((after["def_line_oop_m"] or 0) - (before["def_line_oop_m"] or 0), 1), "m", t),
                self._evi("Ceded the ball" if sh["label_after"] == "LOW_BLOCK" else "Possession",
                          f"Possession {before['possession_pct']}% -> {after['possession_pct']}%",
                          round(after["possession_pct"] - before["possession_pct"], 1), "pct pts", t),
            ] + ([
                self._evi("Transitions", f"{len(counters)} counterattacks launched in the next 15 min", len(counters), "counters", t),
                self._evi("Outcome", f"{sum(1 for c in counters if c['outcome'] != 'lost')} ended in shots" + (f"; goal at {clock(g['timestamp'])}" if g else ""),
                          sum(1 for c in counters if c["outcome"] != "lost"), "shots", t),
            ] if sh["label_after"] == "LOW_BLOCK" else [
                self._evi("Territory", f"Territory {before['territory_pct']}% -> {after['territory_pct']}%",
                          round(after["territory_pct"] - before["territory_pct"], 1), "pct pts", t),
                self._evi("Outcome", f"{after['shots']} shots ({after['xg']} xG) in the following minutes", after["shots"], "shots", t),
            ]),
            "consequences": ([f"{len(counters)} counterattacks"] if sh["label_after"] == "LOW_BLOCK" else [f"Territory {after['territory_pct']}%"])
            + [f"{after['shots']} shots ({after['xg']} xG)"],
            "outcome": ({"type": "GOAL", "event_id": g["id"], "timestamp": g["timestamp"], "description": "Counterattack goal"} if g
                        else {"type": "SHOTS", "description": f"{after['shots']} shots"}),
            "chain": [{"stage": "TACTICAL_CHANGE", "label": sh["label_after"].replace("_", " ").title(), "timestamp": t},
                      {"stage": "STATE_CHANGE", "label": f"Line {before['def_line_oop_m']}m -> {after['def_line_oop_m']}m", "timestamp": t},
                      {"stage": "CONSEQUENCE", "label": f"{len(counters)} counterattacks", "timestamp": t}]
            + ([{"stage": "OUTCOME", "label": "Goal", "timestamp": g["timestamp"]}] if g else []),
            "replay": ({"start": cgoals[0]["start"] - 10, "end": g["timestamp"] + 4, "focus_players": cgoals[0]["players"], "focus_team": team, "overlay": "counter"}
                       if g else {"start": t, "end": t + 25, "focus_players": [], "focus_team": team, "overlay": "shape"}),
            "related_events": [g["id"]] if g else [], "window": [t, t + 900], "detection": sh,
        })

    def red_card_moments(self):
        for r in [e for e in self.m.events if e["event_type"] == "RED_CARD"]:
            team, t = r["team_id"], r["timestamp"]
            opp = other(team)
            k = int(t // 60)
            end_k = min(len(self.tl[team]), k + 14)
            b_opp, a_opp = self.wm(opp, k - 8, k), self.wm(opp, k + 1, end_k)
            b_t, a_t = self.wm(team, k - 8, k), self.wm(team, k + 1, end_k)
            goals = self.m.events_between(t, t + 1200, ("GOAL",), opp)
            self.add({
                "type": "RED_CARD_IMPACT", "team": team, "timestamp": t, "end_timestamp": t + 840,
                "title": f"Red card: {self.name(r['player_id'])} ({TEAM_NAMES[team]})",
                "importance": 0.8 + 0.1 * bool(goals), "confidence": 0.9,
                "trigger": {"event_type": "RED_CARD", "event_id": r["id"], "description": r.get("reason", "")},
                "before": {"team": b_t, "opponent": b_opp}, "after": {"team": a_t, "opponent": a_opp},
                "deltas": {**{f"opp_{k2}": v for k2, v in self.delta(b_opp, a_opp, ["possession_pct", "territory_pct", "shots", "xg", "final_third_recoveries"]).items()},
                           **{f"team_{k2}": v for k2, v in self.delta(b_t, a_t, ["def_line_m", "width_m", "depth_m"]).items()}},
                "evidence": [
                    self._evi("Event", f"{self.name(r['player_id'])} sent off - {r.get('reason', '')}", 1, "red card", t),
                    self._evi("Structural change", f"{TEAM_NAMES[team]} {b_t['formation']} -> {a_t['formation']}, line {b_t['def_line_m']}m -> {a_t['def_line_m']}m",
                              round(a_t["def_line_m"] - b_t["def_line_m"], 1), "m", t),
                    self._evi("Control", f"{TEAM_NAMES[opp]} possession {b_opp['possession_pct']}% -> {a_opp['possession_pct']}%, territory {b_opp['territory_pct']}% -> {a_opp['territory_pct']}%",
                              round(a_opp["territory_pct"] - b_opp["territory_pct"], 1), "pct pts", t),
                    self._evi("Pressure", f"{TEAM_NAMES[opp]} shots {b_opp['shots']} -> {a_opp['shots']} ({a_opp['xg']} xG) in comparable windows",
                              a_opp["shots"], "shots", t),
                ] + ([self._evi("Outcome", f"{TEAM_NAMES[opp]} scored at {clock(goals[0]['timestamp'])}", len(goals), "goals", goals[0]["timestamp"])] if goals else []),
                "consequences": [f"{TEAM_NAMES[team]} dropped to {a_t['formation']}", f"{TEAM_NAMES[opp]} territory {a_opp['territory_pct']}%"],
                "outcome": ({"type": "GOAL", "event_id": goals[0]["id"], "timestamp": goals[0]["timestamp"], "description": "Opponent goal"} if goals
                            else {"type": "TERRITORY", "description": f"Territory {a_opp['territory_pct']}%"}),
                "chain": [{"stage": "EVENT", "label": "Red card", "timestamp": t},
                          {"stage": "STATE_CHANGE", "label": f"{b_t['formation']} -> {a_t['formation']}", "timestamp": t},
                          {"stage": "TACTICAL_CHANGE", "label": f"{TEAM_NAMES[opp]} push up", "timestamp": t + 60},
                          {"stage": "CONSEQUENCE", "label": f"Territory {a_opp['territory_pct']}%", "timestamp": t + 120}]
                + ([{"stage": "OUTCOME", "label": "Goal", "timestamp": goals[0]["timestamp"]}] if goals else []),
                "replay": {"start": max(0, t - 20), "end": t + 10, "focus_players": [r["player_id"]] + self.nearest_players(opp, t, 2),
                           "focus_team": team, "overlay": "line"},
                "related_events": [r["id"]] + [g["id"] for g in goals], "window": [t, t + 840], "player_id": r["player_id"],
            })

    @staticmethod
    def _evi(label, detail, value, unit, ts):
        return {"label": label, "detail": detail, "value": value, "unit": unit, "timestamp": ts, "source": "analytics"}
