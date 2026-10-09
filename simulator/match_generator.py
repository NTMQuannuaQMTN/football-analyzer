"""Deterministic synthetic match generator (1 Hz frames + event stream).

Coordinates: pitch 105 x 68 m, world frame. Team A attacks +x, Team B attacks -x.
"Own frame" for a team = coordinates where that team attacks +x (B is rotated 180deg).
"""
import math
from typing import Dict, List, Optional

import numpy as np

from .scenarios import (ALL_FLAGS, FORMATIONS, L, RED_CARDS, ROLE_MAP, ROSTERS, SCRIPTS,
                        SUBSTITUTIONS, TEAMS, W, tactic_at)

DURATION = 5400
CENTER = np.array([L / 2, W / 2])


def own(team: str, p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return p.copy() if team == "A" else np.array([L - p[0], W - p[1]])


to_world = own  # rotation by 180deg is an involution


def opp(team: str) -> str:
    return "B" if team == "A" else "A"


def xg_model(own_xy) -> float:
    dx, dy = L - own_xy[0], own_xy[1] - W / 2
    d = math.hypot(dx, dy)
    a1 = math.atan2(dy - 3.66, dx)
    a2 = math.atan2(dy + 3.66, dx)
    angle = abs(a2 - a1)
    return round(1 / (1 + math.exp(-(-0.2 - 0.14 * d + 1.2 * angle))), 3)


class MatchSimulator:
    def __init__(self, seed: int = 7, flags=ALL_FLAGS):
        self.rng = np.random.default_rng(seed)
        self.flags = set(flags)
        self.seed = seed
        self.players: Dict[str, dict] = {}
        self.order: List[str] = []
        for team, roster in ROSTERS.items():
            for shirt, name, role in roster:
                pid = f"{team}{shirt}"
                self.order.append(pid)
                self.players[pid] = {"id": pid, "team": team, "shirt": shirt, "name": name,
                                     "start_role": role}
        self.role: Dict[str, Optional[str]] = {pid: p["start_role"] for pid, p in self.players.items()}
        self.on = {pid for pid, r in self.role.items() if r is not None}
        self.formation = {"A": "4-3-3", "B": "4-4-2"}
        self.noise = {pid: (self.rng.uniform(0, 6.3, 2), self.rng.uniform(22, 60, 2),
                            self.rng.uniform(1.5, 3.5)) for pid in self.order}
        self.pos: Dict[str, np.ndarray] = {}
        self.vel: Dict[str, np.ndarray] = {pid: np.zeros(2) for pid in self.order}
        self.t = 0
        self.ball = CENTER.copy()
        self.poss = "A"
        self.carrier: Optional[str] = None
        self.events: List[dict] = []
        self.truth_counters: List[dict] = []  # ground truth only: never part of the observable stream
        self.frames: List[dict] = []
        self.score = {"A": 0, "B": 0}
        self.freeze_until = -1
        self.kickoff_team = "A"
        self.kickoff_taker: Optional[str] = None
        self.counter_until = {"A": -1, "B": -1}
        self.script_steps: Dict[int, tuple] = {}
        self.script_end = -1
        self.scripts_done = set()
        self.pressure_log: Dict[str, int] = {}
        self.carry_vec: Optional[np.ndarray] = None
        self.pass_point: Optional[np.ndarray] = None
        self.freeze_until = 10**9  # skip pressing logic while placing players
        for team in ("A", "B"):
            for pid, tgt in self._targets(team, tactic_at(team, 0, self.flags), CENTER, False, 0).items():
                self.pos[pid] = tgt.copy()
        self._start_kickoff("A", 0, immediate=True)

    # ------------------------------------------------------------------ helpers
    def own_ball(self, team):
        return own(team, self.ball)

    def team_players(self, team, outfield=False):
        return [p for p in self.order if p in self.on and self.players[p]["team"] == team
                and not (outfield and self.role[p] == "GK")]

    def by_role(self, team, role):
        for p in self.team_players(team):
            if self.role[p] == role:
                return p
        return None

    def emit(self, etype, team=None, player=None, xy=None, **kw):
        ev = {"id": f"e{len(self.events):05d}", "timestamp": float(self.t), "event_type": etype,
              "team_id": team, "player_id": player}
        if xy is not None:
            ev["x"], ev["y"] = round(float(xy[0]), 1), round(float(xy[1]), 1)
        for k, v in kw.items():
            if isinstance(v, np.bool_):
                v = bool(v)
            if isinstance(v, (np.floating, float)):
                v = round(float(v), 3)
            if isinstance(v, np.ndarray):
                v = [round(float(c), 1) for c in v]
            ev[k] = v
        self.events.append(ev)
        return ev

    def pressure_on(self, pid) -> float:
        team = self.players[pid]["team"]
        p = self.pos[pid]
        s = 0.0
        for q in self.team_players(opp(team)):
            d = np.linalg.norm(self.pos[q] - p)
            s += max(0.0, 1 - d / 5.5)
        return min(1.0, s)

    def nearest(self, team, point, exclude=(), outfield=True):
        cands = [p for p in self.team_players(team, outfield) if p not in exclude]
        return min(cands, key=lambda p: np.linalg.norm(self.pos[p] - point))

    def change_possession(self, team, player, cause, xy=None):
        prev = self.poss
        self.poss, self.carrier = team, player
        self.counter_until[opp(team)] = -1
        if prev != team:
            self.emit("POSSESSION_CHANGE", team, player, xy if xy is not None else self.pos[player],
                      from_team=prev, cause=cause)
            tac = tactic_at(team, self.t, self.flags)
            if tac.counter and own(team, self.pos[player])[0] < 55:
                self.counter_until[team] = self.t + 14
                self.truth_counters.append({"team": team, "t": float(self.t)})

    def offside_line(self, team, ball_w=None) -> float:
        """Own-frame x beyond which `team`'s attackers are offside: second-last opponent, the ball or halfway."""
        ball_w = self.ball if ball_w is None else ball_w
        opp_x = sorted((own(team, self.pos[q])[0] for q in self.team_players(opp(team)) if q in self.pos), reverse=True)
        second_last = opp_x[1] if len(opp_x) > 1 else L
        return max(second_last, own(team, ball_w)[0], L / 2)

    # ------------------------------------------------------------------ positioning
    def _targets(self, team, tac, ball_w, in_poss, t):
        bo = own(team, ball_w)
        line = tac.block + 0.35 * (bo[0] - L / 2) + (5 if in_poss else 0)
        line = float(np.clip(line, 10, 58))
        depth = tac.depth * (1.15 if in_poss else 1.0)
        width = tac.width_in if in_poss else tac.width_out
        shift_y = 0.3 * (bo[1] - W / 2)
        countering = in_poss and self.counter_until[team] >= t
        form = FORMATIONS[self.formation[team]]
        offside = self.offside_line(team, ball_w) if self.pos else L
        out = {}
        for pid in self.team_players(team):
            role = self.role[pid]
            ln, lat = form[role]
            if role == "GK":
                x, y = 4 + max(0.0, (line - 25) * 0.25), W / 2 + 0.2 * (bo[1] - W / 2)
            else:
                x = line + (ln - 1) / 2 * depth
                y = W / 2 + lat * width / 2 + shift_y
                if not in_poss and tac.hold_forwards and ln >= 3:
                    x = max(x, 50)
                if countering and ln >= 2:
                    x = max(x, bo[0] + 4 + (ln - 2) * 9)
                ph, per, amp = self.noise[pid]
                x += amp * math.sin(2 * math.pi * t / per[0] + ph[0])
                y += amp * math.sin(2 * math.pi * t / per[1] + ph[1])
                # hold runs on the offside line; deeper roles stay staggered between the lines
                x = min(x, offside - 0.6 - max(0.0, 3.1 - ln) * 8)
                if not in_poss and self.freeze_until < t:
                    # defend goal-side: the back line keeps between the ball and its own goal, midfielders recover
                    if ln <= 1.5:
                        x = min(x, max(5.5, bo[0] - 3.5))
                    elif ln <= 2.5 and bo[0] < 45:
                        x = min(x, bo[0] + 4)
                    if bo[0] < 32 and ln <= 2.5:  # ball near our box: narrow and protect the middle
                        y = W / 2 + (y - W / 2) * 0.5 + 0.25 * (bo[1] - W / 2)
            if self.freeze_until >= t:  # restart: stay in own half
                x = min(x, L / 2 - 2)
                if pid == self.kickoff_taker:
                    x, y = L / 2 - 0.5, W / 2
            out[pid] = to_world(team, (float(np.clip(x, 1, L - 1)), float(np.clip(y, 1, W - 1))))
        # in possession: the two nearest team-mates come short to offer the carrier a safe option
        if in_poss and self.freeze_until < t and self.pos and self.carrier in self.pos:
            mates = sorted((p for p in self.team_players(team, True) if p != self.carrier),
                           key=lambda p: np.linalg.norm(self.pos[p] - ball_w))[:2]
            for pid in mates:
                d = self.pos[pid] - ball_w
                n = float(np.linalg.norm(d))
                if n > 11:
                    support = ball_w + d / n * 9.5
                    out[pid] = out[pid] + 0.55 * (support - out[pid])
        # pressing / engaging the ball
        if not in_poss and self.freeze_until < t and self.pos:
            on_right = bo[1] > W / 2 + 8
            n, strength = 1, 0.35
            if (L - bo[0]) <= tac.press_reach:
                n, strength = tac.press_n, tac.press_strength
                if tac.right_press and on_right:
                    n, strength = n + 1, min(1.0, strength + 0.2 * tac.right_press)
            goal_side = to_world(team, (bo[0] - 1.2, bo[1]))
            outfield = sorted(self.team_players(team, True), key=lambda p: np.linalg.norm(self.pos[p] - ball_w))
            for i, pid in enumerate(outfield[:n]):
                s = strength if i == 0 else strength * 0.75
                tgt = goal_side if i == 0 else ball_w + (self.pos[pid] - ball_w) * 0.35
                out[pid] = out[pid] + s * (tgt - out[pid])
        return out

    def _move(self, targets, t):
        for pid, tgt in targets.items():
            if pid not in self.pos:
                continue
            d = tgt - self.pos[pid]
            dist = float(np.linalg.norm(d))
            if dist < 1e-6:
                desired = np.zeros(2)
            else:
                maxs = 8.2 if dist > 12 else 6.0
                desired = d / dist * min(dist * 0.65, maxs)
            self.vel[pid] = 0.6 * desired + 0.4 * self.vel[pid]
            self.pos[pid] = self.pos[pid] + self.vel[pid]
            self.pos[pid][0] = float(np.clip(self.pos[pid][0], 0.5, L - 0.5))
            self.pos[pid][1] = float(np.clip(self.pos[pid][1], 0.5, W - 0.5))

    # ------------------------------------------------------------------ actions
    def do_pass(self, to_pid, success=True, intercept_team=None, through=0.0, interceptor=None, check_offside=True):
        c = self.carrier
        team = self.poss
        start = self.pos[c].copy()
        if success and check_offside:
            line = self.offside_line(team)
            rx = own(team, self.pos[to_pid])[0]
            if rx > line + 0.3:
                self.emit("OFFSIDE", team, to_pid, self.pos[to_pid], passer_id=c,
                          margin_m=round(float(rx - line), 1), line_x=round(float(line), 1))
                q = self.nearest(opp(team), self.pos[to_pid])
                self.change_possession(opp(team), q, "offside", self.pos[to_pid])
                return
        target = self.pos[to_pid].copy()
        if through:
            target = to_world(team, own(team, target) + np.array([through, 0]))
        dist = float(np.linalg.norm(target - start))
        fwd = float(own(team, target)[0] - own(team, start)[0])
        self.emit("PASS", team, c, start, target_x=round(float(target[0]), 1),
                  target_y=round(float(target[1]), 1), receiver_id=to_pid if success else None,
                  success=success, pressure=self.pressure_on(c), distance=dist,
                  progression=round(fwd, 1))
        if success:
            self.carrier = to_pid
            self.pass_point = target if through else None
            if through:
                self.vel[to_pid] = (target - self.pos[to_pid]) * 0.5
        else:
            o = opp(team)
            q = interceptor or self.nearest(o, target)
            # interceptor steps into the passing lane
            d = target - self.pos[q]
            step = min(4.0, float(np.linalg.norm(d)))
            if np.linalg.norm(d) > 0:
                self.pos[q] = self.pos[q] + d / np.linalg.norm(d) * step
            self.change_possession(o, q, "interception")

    def do_carry(self, dist, speed_dir=None):
        team = self.poss
        bo = self.own_ball(team)
        if speed_dir is None:
            goal = np.array([L, W / 2])
            d = goal - bo
            d = d / (np.linalg.norm(d) + 1e-9)
            d = d + np.array([0, self.rng.normal(0, 0.25)])
            d = d / np.linalg.norm(d)
        else:
            d = speed_dir
        new_o = bo + d * dist
        new_o = np.array([np.clip(new_o[0], 1, L - 3), np.clip(new_o[1], 1.5, W - 1.5)])
        self.carry_vec = to_world(team, new_o) - self.ball

    def do_shot(self, goal=False, outcome=None):
        c, team = self.carrier, self.poss
        o_xy = own(team, self.pos[c])
        xg = xg_model(o_xy)
        if goal:
            outcome = "goal"
        elif outcome is None:
            r = self.rng.random()
            outcome = "saved" if r < 0.4 else ("off_target" if r < 0.8 else "blocked")
        self.emit("SHOT", team, c, self.pos[c], xg=xg, outcome=outcome,
                  on_target=outcome in ("goal", "saved"), distance=round(float(np.hypot(L - o_xy[0], o_xy[1] - W / 2)), 1))
        if goal:
            self.score[team] += 1
            assist = None
            for ev in reversed(self.events[:-1]):
                if ev["event_type"] == "PASS" and ev.get("receiver_id") == c and ev["success"]:
                    assist = ev["player_id"]
                    break
                if ev["event_type"] in ("POSSESSION_CHANGE",):
                    break
            self.emit("GOAL", team, c, self.pos[c], assist_id=assist, xg=xg,
                      score={"A": self.score["A"], "B": self.score["B"]})
            net = to_world(team, (L + 1.2, float(np.clip(own(team, self.pos[c])[1], W / 2 - 3, W / 2 + 3))))
            self._start_kickoff(opp(team), self.t + 22)
            self.ball = net
        else:
            o = opp(team)
            if outcome == "blocked":
                q = self.nearest(o, self.pos[c])
            else:
                q = self.by_role(o, "GK")
            self.change_possession(o, q, "shot_" + outcome)

    def _start_kickoff(self, team, until, immediate=False):
        self.freeze_until = until
        self.kickoff_team = team
        self.kickoff_taker = self.by_role(team, "ST") or self.by_role(team, "RS") or self.by_role(team, "CAM")
        self.carrier = None
        self.poss = team
        if immediate:
            self.ball = CENTER.copy()
            self.pos[self.kickoff_taker] = CENTER.copy()

    # ------------------------------------------------------------------ decisions
    def decide(self, t):
        c, team = self.carrier, self.poss
        tac = tactic_at(team, t, self.flags)
        dtac = tactic_at(opp(team), t, self.flags)
        bo = own(team, self.pos[c])
        pressure = self.pressure_on(c)
        countering = self.counter_until[team] >= t
        # pressure events
        for q in self.team_players(opp(team), True):
            d = float(np.linalg.norm(self.pos[q] - self.pos[c]))
            if d < 4.0 and t - self.pressure_log.get(q, -99) >= 8:
                self.pressure_log[q] = t
                self.emit("PRESSURE", opp(team), q, self.pos[q], target_player_id=c,
                          pressure=round(1 - d / 4.0, 2))
        # --- football decision: compare the value of shooting, passing to each team-mate and carrying on ---
        opps = [q for q in self.team_players(opp(team)) if q in self.pos]
        opp_o = np.array([own(team, self.pos[q]) for q in opps])
        opp_out = np.array([own(team, self.pos[q]) for q in opps if self.role[q] != "GK"])

        def threat(o):  # how dangerous possession at own-frame point o is (chance of a goal soon)
            return xg_model(o) + 0.06 * (max(0.0, o[0]) / L) ** 2

        def lane_min(a, b):  # closest opponent to the passing lane a -> b (ignoring the ends)
            ab = b - a
            n2 = float(ab @ ab) + 1e-9
            tt = np.clip(((opp_o - a) @ ab) / n2, 0.12, 0.92)
            return float(np.min(np.linalg.norm(opp_o - (a + tt[:, None] * ab), axis=1)))

        # shot: xG discounted for outfield defenders standing in the shooting triangle
        goal = np.array([L, W / 2])
        blockers = 0
        for q in opp_out:
            if bo[0] < q[0] < L and abs(q[1] - (bo[1] + (W / 2 - bo[1]) * (q[0] - bo[0]) / max(1.0, L - bo[0]))) < 2.5 + 3.66 * (q[0] - bo[0]) / max(1.0, L - bo[0]):
                blockers += 1
        dist_goal = float(np.linalg.norm(goal - bo))
        shot_v = xg_model(bo) * (0.6 ** blockers) * (1 - 0.5 * pressure)
        through = bo[0] >= 60 and blockers == 0 and not any(q[0] > bo[0] and abs(q[1] - bo[1]) < 12 for q in opp_out)
        shoot = (through and dist_goal <= 13) or (shot_v >= 0.16 and self.rng.random() < 0.6) \
            or (shot_v >= 0.07 and self.rng.random() < 0.08)
        if bo[0] >= 70 and shoot:
            self.do_shot()
            return
        # tackle?
        tackle_p = 0.08 * pressure * dtac.press_strength + (0.006 if pressure > 0 else 0)
        if self.rng.random() < tackle_p:
            q = self.nearest(opp(team), self.pos[c])
            self.emit("TACKLE", opp(team), q, self.pos[q], opponent_id=c, success=True)
            self.change_possession(opp(team), q, "tackle")
            return
        # carry: run at goal when there is space ahead (a 1v1 is carried, not passed back)
        fwd = goal - bo
        fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        ahead = bo + fwd * 6
        space = float(np.min(np.linalg.norm(opp_o - ahead, axis=1))) if len(opp_o) else 20.0
        keep = float(np.clip(1 - 0.5 * pressure - (0.3 if space < 3 else 0.0), 0.2, 1))
        carry_v = keep * threat(ahead) + (0.02 if through else 0.0)
        hold_v = threat(bo) * (1 - 0.4 * pressure)
        # passes: value = P(complete) * threat at the receiver - P(lost) * danger of losing it there
        mates = [p for p in self.team_players(team) if p != c and (self.role[p] != "GK" or (pressure > 0.6 and bo[0] < 35))]
        best = None
        cands = []
        for m in mates:
            mo = own(team, self.pos[m])
            dist = float(np.linalg.norm(mo - bo))
            if dist < 5 or dist > 50:
                continue
            openness = float(np.min(np.linalg.norm(opp_o - mo, axis=1)))
            lane = lane_min(bo, mo)
            p_ok = tac.pass_acc + 0.04 - 0.16 * pressure - 0.005 * max(0.0, dist - 15) - (0.14 if openness < 2.5 else 0.0) \
                - (0.28 if lane < 1.5 else 0.10 if lane < 3 else 0.0)
            p_ok -= 0.06 * dtac.press_strength * (1 if (L - own(opp(team), self.pos[m])[0]) <= dtac.press_reach else 0)
            p_ok = float(np.clip(p_ok, 0.05, 0.97))
            gain = threat(mo) + 0.0012 * tac.directness * max(0.0, mo[0] - bo[0]) + (0.01 if countering and mo[0] > bo[0] else 0)
            gain = max(gain, 0.9 * threat(bo)) + 0.01  # keeping the ball has value of its own
            loss = 0.6 * threat(np.array([L - mo[0], W - mo[1]])) + 0.06  # losing the ball always costs
            v = p_ok * gain - (1 - p_ok) * loss
            if through and threat(mo) < 1.3 * threat(bo):
                continue  # through on goal: only a square ball to a better-placed team-mate, never a pass back
            cands.append((v, m, p_ok))
        if cands:
            vs = np.array([x[0] for x in cands])
            pr = np.exp((vs - vs.max()) / 0.006)
            pr /= pr.sum()
            k = int(self.rng.choice(len(cands), p=pr))
            best = cands[k]
        # act: pass when it beats keeping the ball (tempo sets how readily), or when pressed with a safe outlet
        rate = 0.10 * tac.tempo + 0.30 * pressure + (0.2 if self.role[c] == "GK" else 0.0)
        if best is not None and (best[0] > max(carry_v, hold_v) * (1.15 - 0.15 * tac.tempo) or pressure > 0.55) \
                and self.rng.random() < rate + (0.12 if best[0] > carry_v * 1.5 else 0.0):
            self.do_pass(best[1], success=self.rng.random() < best[2])
            return
        if pressure > 0.5 and not countering and not through:
            q = self.nearest(opp(team), self.pos[c])
            away = own(team, self.pos[c]) - own(team, self.pos[q])
            away = away / (np.linalg.norm(away) + 1e-9)
            away = 0.6 * away + 0.4 * fwd  # shield but keep facing forward
            away = away / (np.linalg.norm(away) + 1e-9)
            self.do_carry(1.5 + self.rng.random(), away)
        else:
            spd = 6.5 + self.rng.random() if countering or through else (2.0 + 2.5 * self.rng.random() if space > 4 else 1.2)
            self.do_carry(spd)

    def run_script_step(self, action, args):
        team = self.poss
        if action == "pass":
            to = next((self.by_role(team, r) for r in args["to"] if self.by_role(team, r) not in (None, self.carrier)), None)
            if to is None:  # role absent in this (counterfactual) shape: most advanced teammate
                mates = [p for p in self.team_players(team, True) if p != self.carrier]
                to = max(mates, key=lambda p: own(team, self.pos[p])[0])
            if args.get("intercept"):
                self.do_pass(to, success=False)
            else:
                # scripted moves still obey the offside law: the receiver times his run to stay level with the
                # second-last defender (or the ball) when the pass is played
                line = self.offside_line(team)
                ox, oy = own(team, self.pos[to])
                if ox > line - 0.5:
                    self.pos[to] = to_world(team, np.array([line - 0.5, oy]))
                self.do_pass(to, success=True, through=args.get("through", 0.0), check_offside=False)
        elif action == "carry":
            self.do_carry(args["dist"])
        elif action == "tackle":
            o = opp(team)
            q = self.by_role(o, args["by"]) or self.nearest(o, self.ball)
            self.pos[q] = self.ball + (self.pos[q] - self.ball) / (np.linalg.norm(self.pos[q] - self.ball) + 1e-9) * 1.0
            self.emit("PRESSURE", o, q, self.pos[q], target_player_id=self.carrier, pressure=0.9)
            self.emit("TACKLE", o, q, self.pos[q], opponent_id=self.carrier, success=True)
            self.change_possession(o, q, "tackle")
        elif action == "shot":
            self.do_shot(goal=args.get("goal", False), outcome=args.get("outcome"))

    # ------------------------------------------------------------------ main loop
    def scheduled_events(self, t):
        for st, team, off_s, on_s, req in SUBSTITUTIONS:
            if st == t and all(f in self.flags for f in req):
                off_id, on_id = f"{team}{off_s}", f"{team}{on_s}"
                self.role[on_id] = self.role[off_id]
                self.role[off_id] = None
                self.on.discard(off_id)
                self.on.add(on_id)
                self.pos[on_id] = np.array([L / 2, W - 0.5 if team == "A" else 0.5])
                self.vel[on_id] = np.zeros(2)
                if self.carrier == off_id:
                    self.carrier = on_id
                self.emit("SUBSTITUTION", team, on_id, self.pos[on_id], player_off_id=off_id,
                          player_on_id=on_id, role=self.role[on_id])
        for rt, team, shirt, remap, req in RED_CARDS:
            if rt == t and all(f in self.flags for f in req):
                pid = f"{team}{shirt}"
                xy = self.pos[pid].copy()
                self.emit("RED_CARD", team, pid, xy, reason="Denied goal-scoring opportunity")
                self.on.discard(pid)
                self.role[pid] = None
                for p in self.team_players(team):
                    if self.role[p] in remap:
                        self.role[p] = remap[self.role[p]]
                self.formation[team] = "4-4-1"
                self.script_steps.clear()
                q = self.nearest(opp(team), xy)
                self.change_possession(opp(team), q, "free_kick", xy)

    def sync_formations(self, t):
        for team in ("A", "B"):
            tac = tactic_at(team, t, self.flags)
            cur = self.formation[team]
            if tac.formation != cur and (cur, tac.formation) in ROLE_MAP:
                m = ROLE_MAP[(cur, tac.formation)]
                for p in self.team_players(team):
                    self.role[p] = m.get(self.role[p], self.role[p])
                self.formation[team] = tac.formation

    def check_scripts(self, t):
        if self.script_steps or self.freeze_until >= t or self.carrier is None:
            return
        for sc in SCRIPTS:
            if sc.id in self.scripts_done or t < sc.earliest or not all(f in self.flags for f in sc.flags):
                continue
            if sc.cond(self):
                self.scripts_done.add(sc.id)
                for dt, action, args in sc.steps:
                    self.script_steps[t + dt] = (action, args)
                self.script_end = t + max(s[0] for s in sc.steps)
                return

    def step(self, t):
        self.t = t
        self.carry_vec = None
        self.pass_point = None
        self.scheduled_events(t)
        self.sync_formations(t)
        if self.freeze_until >= 0 and t == self.freeze_until:
            self.freeze_until = -1
            self.carrier = self.kickoff_taker
            self.pos[self.carrier] = CENTER.copy()
            self.ball = CENTER.copy()
            self.emit("KICK_OFF", self.kickoff_team, self.carrier, CENTER)
        elif self.freeze_until < t and self.carrier is not None:
            self.check_scripts(t)
            if t in self.script_steps:
                self.run_script_step(*self.script_steps.pop(t))
            elif t <= self.script_end:
                self.do_carry(1.0)
            else:
                self.decide(t)

        if self.freeze_until >= t and t == self.freeze_until - 6:
            self.ball = CENTER.copy()  # ball brought back for the restart
        ball_ref = self.ball if self.carrier is None else self.pos[self.carrier]
        if self.freeze_until >= t:
            ball_ref = CENTER
        if self.carry_vec is not None:
            ball_ref = self.ball + self.carry_vec
        targets = {}
        for team in ("A", "B"):
            tac = tactic_at(team, t, self.flags)
            targets.update(self._targets(team, tac, ball_ref, team == self.poss, t))
        if self.carrier is not None:
            targets.pop(self.carrier, None)
        self._move(targets, t)
        if self.carrier is not None:
            c = self.carrier
            if self.carry_vec is not None:
                new = self.ball + self.carry_vec
                self.vel[c] = new - self.pos[c]
                self.pos[c] = new
            elif self.pass_point is not None:
                self.pos[c] = self.pass_point
            self.ball = self.pos[c].copy()
        self.record(t)

    def record(self, t):
        xy = []
        for pid in self.order:
            if pid in self.on:
                xy.append([round(float(self.pos[pid][0]), 1), round(float(self.pos[pid][1]), 1)])
            else:
                xy.append(None)
        self.frames.append({"t": t, "b": [round(float(self.ball[0]), 1), round(float(self.ball[1]), 1)],
                            "p": self.poss if self.freeze_until < t else None, "c": self.carrier, "xy": xy})

    def run(self) -> dict:
        for t in range(DURATION + 1):
            self.step(t)
        players = []
        for pid in self.order:
            p = self.players[pid]
            players.append({**p, "position": p["start_role"] or "SUB"})
        return {
            "truth": self.ground_truth(),
            "meta": {"seed": self.seed, "flags": sorted(self.flags), "duration": DURATION,
                     "pitch": {"length": L, "width": W}, "teams": TEAMS, "player_order": self.order,
                     "final_score": dict(self.score)},
            "players": players,
            "events": self.events,
            "frames": self.frames,
        }


    def ground_truth(self) -> dict:
        """What the simulator *knows* (used only for evaluation, never by the analytics)."""
        by_min = {}
        changes = []
        for team in ("A", "B"):
            rows, prev = [], None
            for k in range(DURATION // 60):
                tac = tactic_at(team, k * 60 + 30, self.flags)
                rows.append({"minute": k, "tactic": tac.label, "formation": tac.formation,
                             "high_press": tac.press_n >= 3 and tac.press_strength >= 0.85,
                             "low_block": tac.block <= 22})
            by_min[team] = rows
            for t in range(DURATION):
                tac = tactic_at(team, t, self.flags)
                if prev is not None and tac.label != prev.label:
                    changes.append({"team": team, "t": t, "from": prev.label, "to": tac.label,
                                    "formation_change": tac.formation != prev.formation})
                prev = tac
        # did each counter-intent recovery develop? (kept the ball and reached the final third within 15 s)
        for c in self.truth_counters:
            t0, team = int(c["t"]), c["team"]
            dev = False
            for f in self.frames[t0:t0 + 16]:
                if f["p"] != team:
                    break
                if own(team, f["b"])[0] >= 70:
                    dev = True
                    break
            c["developed"] = dev
        recov = [{"t": e["timestamp"], "team": e["team_id"], "cause": e["cause"]} for e in self.events
                 if e["event_type"] == "POSSESSION_CHANGE" and e.get("cause") in ("tackle", "interception")]
        return {"tactic_by_minute": by_min, "tactic_changes": changes, "counter_starts": self.truth_counters,
                "recoveries": recov}


def generate_match(seed: int = 7, flags=ALL_FLAGS) -> dict:
    return MatchSimulator(seed, flags).run()


if __name__ == "__main__":
    import json, sys, time
    t0 = time.time()
    m = generate_match()
    print("sim seconds", round(time.time() - t0, 2), "score", m["meta"]["final_score"], file=sys.stderr)
    from collections import Counter
    print(Counter(e["event_type"] for e in m["events"]), file=sys.stderr)
    for e in m["events"]:
        if e["event_type"] in ("GOAL", "SUBSTITUTION", "RED_CARD"):
            print(e, file=sys.stderr)
    if len(sys.argv) > 1:
        json.dump(m, open(sys.argv[1], "w"))
