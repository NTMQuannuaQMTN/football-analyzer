"""Every tactical and formation change, for both teams, with the evidence around it.

Three kinds of change, read from the engine's per-minute timeline (no second analytics system):
  FORMATION  the per-minute shape (template fit, already smoothed over time) changes, e.g. 4-2-3-1 -> 4-4-2;
             includes which players moved between lines (defence / midfield / attack) as the shape is read
  TACTIC     the team's tactical state changes (HIGH_PRESS, LOW_BLOCK, DOMINANT, BALANCED); one-minute blips are
             treated as noise (a state must last >= MIN_STATE_MIN minutes to count)
  STRUCTURE  a detected change point in line height / width / depth / pressing (backend/tactics/shift.py) that is not
             already a formation or state change at the same time

For each change: before / after window metrics (up to WIN minutes, never past the next change of the same team),
deltas, what preceded it (substitutions, goals, cards in the 3 minutes before), what followed it (shots, xG, goals for
and against while the new state lasted, max WIN minutes), how much of the team was observed (reconstructed matches),
and a short hedged reading. Wording is sequence, never cause: "followed", "preceded".
"""
from typing import Dict, List, Optional

import numpy as np

from backend.tactics.tactical_state import tactical_label

WIN = 5
MIN_STATE_MIN = 2
STATE_TEXT = {"HIGH_PRESS": "high press", "LOW_BLOCK": "low block", "DOMINANT": "territorial control", "BALANCED": "balanced"}
DELTA_KEYS = [("def_line_oop_m", "defensive line (out of possession)", "m"), ("width_m", "width", "m"), ("compactness_m", "compactness", "m"),
              ("possession_pct", "possession", "%"), ("territory_pct", "territory", "%"), ("press_intensity", "press index", "")]


def _line_names(n_lines):
    return {2: ["defence", "attack"], 3: ["defence", "midfield", "attack"],
            4: ["defence", "defensive midfield", "attacking midfield", "attack"]}.get(n_lines, [f"line {i + 1}" for i in range(n_lines)])


def _player_lines(lines):
    names = _line_names(len(lines or []))
    return {pid: names[i] for i, line in enumerate(lines or []) for pid in line}


def _runs(seq, min_len=1):
    """[(value, start, end)] runs; runs shorter than min_len are absorbed into the previous run (noise)."""
    runs = []
    for k, v in enumerate(seq):
        if runs and runs[-1][0] == v:
            runs[-1][2] = k + 1
        else:
            runs.append([v, k, k + 1])
    if min_len > 1:
        merged = []
        for r in runs:
            if merged and (r[2] - r[1] < min_len or merged[-1][0] == r[0]):
                merged[-1][2] = r[2]
            else:
                merged.append(list(r))
        runs = merged
    return [tuple(r) for r in runs]


# the same smoothed per-minute values the tactical labels are computed from (raw per-minute values are noisier and
# could contradict the label, e.g. a "low block" with a higher raw line)
SMOOTH = {"def_line_oop_m": ("def_line_oop_s", 1), "width_m": ("width_s", 1), "compactness_m": ("compactness_s", 1),
          "possession_pct": ("possession_s", 100), "territory_pct": ("territory_s", 100), "press_intensity": ("press_index", 1)}


def _smoothed(rows, m0, m1):
    out = {}
    for key, (col, scale) in SMOOTH.items():
        v = [r[col] for r in rows[max(0, m0):max(m0 + 1, m1)] if r.get(col) is not None]
        out[key] = round(float(np.mean(v)) * scale, 1) if v else None
    return out


def _context(m, team, t, name):
    opp = "B" if team == "A" else "A"
    pre = [e for e in m.events if t - 180 <= e["timestamp"] <= t + 30 and e["event_type"] in ("SUBSTITUTION", "GOAL", "CARD", "RED_CARD")]
    out = []
    for e in pre:
        who = "" if e["team_id"] is None else ("" if e["team_id"] == team else "opponent ")
        if e["event_type"] == "SUBSTITUTION":
            out.append(f"{who}substitution {int(e['timestamp'] // 60) + 1}': {name(e.get('player_on_id'))} on for {name(e.get('player_off_id'))}")
        elif e["event_type"] == "GOAL":
            out.append(f"{'goal scored' if e['team_id'] == team else 'goal conceded'} {int(e['timestamp'] // 60) + 1}'")
        else:
            out.append(f"{who}card {int(e['timestamp'] // 60) + 1}'")
    del opp
    return out


def _followed(m, team, t0, t1):
    opp = "B" if team == "A" else "A"
    shots = [e for e in m.events if t0 <= e["timestamp"] < t1 and e["event_type"] == "SHOT"]
    goals = [e for e in m.events if t0 <= e["timestamp"] < t1 and e["event_type"] == "GOAL"]
    mine = [e for e in shots if e["team_id"] == team]
    theirs = [e for e in shots if e["team_id"] == opp]
    return {"shots_for": len(mine), "xg_for": round(sum(e.get("xg") or 0 for e in mine), 2),
            "shots_against": len(theirs), "xg_against": round(sum(e.get("xg") or 0 for e in theirs), 2),
            "goals_for": sum(e["team_id"] == team for e in goals), "goals_against": sum(e["team_id"] == opp for e in goals),
            "window_s": [t0, t1]}


def _reliability(I, team, m0, m1):
    rows = (getattr(I, "coverage", None) or {}).get(team, [])[max(0, m0):max(m0 + 1, m1)]
    if not rows:
        return None
    obs = float(np.mean([r["observed_avg"] for r in rows]))
    inf = float(np.mean([r["inferred_avg"] for r in rows]))
    return round(obs / max(1e-6, obs + inf), 2)


def _reading(ch):
    """Short hedged reading built only from the measured numbers above."""
    d = ch["deltas"]
    parts = []
    if ch["type"] == "FORMATION":
        unsure = (ch.get("observed_share") or 1) < 0.5
        parts.append(f"{'Possible change of shape' if unsure else 'Shape read as'} {ch['from']} -> {ch['to']}"
                     + (" (most of the team was estimated)" if unsure else "")
                     + (", lasting only a minute" if ch.get("brief") else ""))
        if ch.get("moved"):
            parts.append("; ".join(f"{x['name']} {x['from']} -> {x['to']}" for x in ch["moved"][:4]))
    elif ch["type"] == "TACTIC":
        parts.append(f"From {STATE_TEXT.get(ch['from'], ch['from'])} to {STATE_TEXT.get(ch['to'], ch['to'])}")
    else:
        parts.append("Structural change in how the team is positioned")
    moves = []
    for key, label, unit in DELTA_KEYS:
        v = d.get(key)
        if v is not None and abs(v) >= (3 if unit == "m" else 5 if unit == "%" else 8):
            moves.append(f"{label} {'+' if v > 0 else ''}{v}{unit}")
    if moves:
        parts.append(", ".join(moves))
    f = ch["followed"]
    parts.append(f"followed by {f['shots_for']} shot(s) for ({f['xg_for']} xG) and {f['shots_against']} against "
                 f"({f['xg_against']} xG)" + (f", {f['goals_for']} goal(s) for" if f["goals_for"] else "")
                 + (f", {f['goals_against']} against" if f["goals_against"] else ""))
    if ch.get("preceded_by"):
        parts.append("preceded by " + "; ".join(ch["preceded_by"]))
    return ". ".join(p[0].upper() + p[1:] for p in parts if p) + "."


def tactical_changes(I, until: Optional[float] = None) -> Dict[str, List[dict]]:
    """{team: [change, ...]} in time order. With `until` (live), only changes whose after-window has fully elapsed."""
    m, tl = I.m, I.timeline
    offset = (m.meta.get("clock_offset") or 0)
    name = lambda pid: (m.player_by_id.get(pid) or {}).get("name") or (pid or "?")
    shirt = lambda pid: (m.player_by_id.get(pid) or {}).get("shirt")
    out = {}
    for team in ("A", "B"):
        opp = "B" if team == "A" else "A"
        rows, orows = tl[team], tl[opp]
        n = len(rows)
        forms = [r.get("formation") if r.get("formation") not in (None, "?") else None for r in rows]
        labels = [tactical_label(r, o) for r, o in zip(rows, orows)]
        cands = []
        for v, s, e in _runs(forms)[1:]:
            if v is None or forms[s - 1] is None:
                continue
            before, after = _player_lines(rows[s - 1].get("formation_lines")), _player_lines(rows[s].get("formation_lines"))
            moved = [{"player_id": p, "name": name(p), "shirt": shirt(p), "from": before[p], "to": after[p]}
                     for p in after if p in before and before[p] != after[p]]
            cands.append({"type": "FORMATION", "minute": s, "from": forms[s - 1], "to": v, "lasts_min": e - s,
                          "brief": e - s < MIN_STATE_MIN, "moved": moved})
        lab_runs = _runs(labels, MIN_STATE_MIN)
        for (pv, _, _), (v, s, e) in zip(lab_runs, lab_runs[1:]):
            if v != pv:
                cands.append({"type": "TACTIC", "minute": s, "from": pv, "to": v, "lasts_min": e - s, "brief": False})
        taken = {c["minute"] for c in cands}
        for sh in getattr(I, "shifts", []) or []:
            if sh["team"] == team and not any(abs(sh["minute"] - k) <= 2 for k in taken):
                cands.append({"type": "STRUCTURE", "minute": sh["minute"], "from": sh.get("label_before"), "to": sh.get("label_after"),
                              "lasts_min": None, "brief": False, "score": sh.get("score")})
        cands.sort(key=lambda c: (c["minute"], c["type"]))
        for i, c in enumerate(cands):
            k = c["minute"]
            nxt = next((x["minute"] for x in cands[i + 1:] if x["minute"] > k), n)
            prv = max([x["minute"] for x in cands[:i] if x["minute"] < k] or [0])
            b0, a1 = max(prv, k - WIN), min(nxt, k + WIN, n)
            before, after = _smoothed(rows, b0, k), _smoothed(rows, k, max(a1, k + 1))
            c["deltas"] = {key: (round(after[key] - before[key], 1) if after.get(key) is not None and before.get(key) is not None else None)
                           for key, _, _ in DELTA_KEYS}
            c["before"] = {key: before.get(key) for key, _, _ in DELTA_KEYS}
            c["after"] = {key: after.get(key) for key, _, _ in DELTA_KEYS}
            t = k * 60
            c.update(team=team, timestamp=t, clock=f"{int((t + offset) // 60)}:{int((t + offset) % 60):02d}",
                     minute_label=f"{int((t + offset) // 60) + 1}'", windows={"before_min": [b0, k], "after_min": [k, a1]},
                     preceded_by=_context(m, team, t, name), followed=_followed(m, team, t, a1 * 60),
                     observed_share=_reliability(I, team, b0, a1), available_at=float(a1 * 60))
            c["basis"] = ("ground truth positions" if I.view == "truth" else
                          "measured + estimated positions" if (c["observed_share"] or 1) >= 0.5 else
                          "mostly estimated positions: the shape reading is uncertain" if c["type"] == "FORMATION" else
                          "mostly estimated positions")
            c["reading"] = _reading(c)
            c["id"] = f"chg-{team}-{k:03d}-{c['type'].lower()}"
        if until is not None:
            cands = [c for c in cands if c["available_at"] <= until]
        out[team] = cands
    return out
