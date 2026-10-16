"""Full-match real-broadcast re-entry benchmark on SkillCorner Open Data (20 A-League 2024/25 matches, MIT).

SkillCorner tracking is produced by computer vision from the broadcast. Per player and frame it says whether the
player was detected in the broadcast image (`is_detected`); undetected positions are SkillCorner's own
extrapolations and are NEVER used here. Detected positions are SkillCorner's measurements, not independent
ground truth (PROXY TRUTH, typically ~1 m-level noise but unverified here). Identities, roles and possession come from
SkillCorner. Same protocol as `gsr_benchmark`: live re-entry (100 % real visibility) and replay with EXT s of
extra hiding after a re-entry.

Note on overlap: the camera FOOTPRINTS of these matches are what `sc_camera` training masks are built from (camera
behaviour only; no player positions from these matches are ever used for training).

    .venv/bin/python -m backend.reconstruction.skillcorner_benchmark --methods base,final_v3
"""
import argparse
import glob
import json
import os

import numpy as np

from backend.reconstruction.datasets import L, SC_DIR, W
from backend.reconstruction.gsr_benchmark import OUT, run, summarize

ROLE = {"GK": "GK", "CB": "CB", "LCB": "CB", "RCB": "CB", "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
        "DM": "DM", "LDM": "DM", "RDM": "DM", "LM": "WM", "RM": "WM", "AM": "AM", "LW": "W", "RW": "W",
        "LF": "ST", "RF": "ST", "CF": "ST", "ST": "ST", "CM": "CM"}
LINE_LAT = {"GK": (0, 0), "CB": (1, 0), "LCB": (1, -.3), "RCB": (1, .3), "LB": (1, -.85), "RB": (1, .85),
            "LWB": (1.3, -.9), "RWB": (1.3, .9), "DM": (1.6, 0), "LDM": (1.6, -.3), "RDM": (1.6, .3), "LM": (2, -.85),
            "RM": (2, .85), "AM": (2.45, 0), "LW": (2.6, -.8), "RW": (2.6, .8), "LF": (2.9, -.3), "RF": (2.9, .3),
            "CF": (3.1, 0), "ST": (3.1, 0), "CM": (2, 0)}


RESTART_KIND = {"throw_in": "Throw-in", "free_kick": "Free kick", "goal_kick": "Goal kick", "corner": "Corner",
                "goal": "Goal · kick-off", "penalty": "Penalty"}


def methods_learned(methods):
    return [m for m in methods.split(",") if m != "base"]


def restart_kind(label):
    """'throw_in_for' -> 'Throw-in'; '' -> None."""
    if not label:
        return None
    base = label.rsplit("_", 1)[0] if label.endswith(("_for", "_against")) else label
    return RESTART_KIND.get(base, "Dead ball")


def provider_stoppages(mdir, period, f0, T):
    """Dead-ball spells from SkillCorner's own interruption labels: play stops after a possession whose
    game_interruption_after is set and restarts with the next possession whose game_interruption_before is set (its
    reception starts ~1 s after the restart). Returns (alive (T,) bool, [(start_s, restart_s, kind, after_label)],
    possession start times)."""
    import csv
    rows = list(csv.DictReader(open(glob.glob(os.path.join(mdir, "*_dynamic_events.csv"))[0])))
    poss = sorted([r for r in rows if r["period"] == str(period) and r["event_type"] == "player_possession"],
                  key=lambda r: int(r["frame_start"]))
    ts = lambda fr: (int(fr) - f0) / 10
    spells = []
    for i, r in enumerate(poss):
        after = r.get("game_interruption_after") or ""
        if not after:
            continue
        s = ts(r["frame_end"]) + 1.0
        if i + 1 >= len(poss):
            continue
        nxt = poss[i + 1]  # play restarts with the very next possession (its restart label can be missing)
        if ts(nxt["frame_start"]) - s > 300:
            continue
        e = max(s + 1.0, ts(nxt["frame_start"]) - 1.0)
        if after in ("goal_for", "goal_against"):  # a goal stops play until ITS kick-off, not the next touch
            for q in poss[i + 1:]:
                if ts(q["frame_start"]) - s > 180:
                    break
                if (q.get("game_interruption_before") or "") in ("goal_for", "goal_against"):
                    e = max(e, ts(q["frame_start"]) - 1.0)
                    break
                if q.get("game_interruption_after"):
                    break  # another stoppage first: this goal's kick-off label is missing; never swallow live play
        # the stoppage type is how play STOPPED (the next row's restart label can be stray, e.g. a goal restart label
        # on a possession after a goal kick); goal stoppages are confirmed against counted goals when displayed
        spells.append((s, e, restart_kind(after) or restart_kind(nxt.get("game_interruption_before") or ""), after))
    alive = np.ones(T, bool)
    for s, e, _, _ in spells:
        alive[max(0, int(np.ceil(s))):min(T, int(np.floor(e)))] = False
    return alive, spells, [ts(r["frame_start"]) for r in poss]


def load_match(mdir):
    mid = os.path.basename(mdir)
    meta = json.load(open(glob.glob(os.path.join(mdir, "*_match.json"))[0]))
    pl, pw = float(meta.get("pitch_length") or 105), float(meta.get("pitch_width") or 68)
    home = meta["home_team"]["id"]
    players = {p["id"]: p for p in meta["players"]}
    per = {}
    with open(glob.glob(os.path.join(mdir, "*tracking_extrapolated.jsonl"))[0]) as fh:
        for line in fh:
            fr = json.loads(line)
            if fr.get("period") in (1, 2) and fr["frame"] % 10 == 0:
                per.setdefault(fr["period"], []).append(fr)
    sides = meta.get("home_team_side") or ["left_to_right", "right_to_left"]
    for period, frames in sorted(per.items()):
        flip = sides[period - 1] == "right_to_left"
        ids = sorted({p["player_id"] for fr in frames for p in fr["player_data"]})
        col = {pid: j for j, pid in enumerate(ids)}
        T, N = len(frames), len(ids)
        pos = np.full((T, N, 2), np.nan)
        on = np.zeros((T, N), bool)
        ball = np.full((T, 2), np.nan)
        ball_z = np.full(T, np.nan)
        poss = np.full(T, -1)
        for t, fr in enumerate(frames):
            for p in fr["player_data"]:
                j = col[p["player_id"]]
                on[t, j] = True
                if p.get("is_detected") and p.get("x") is not None:
                    pos[t, j] = (p["x"] * L / pl, p["y"] * W / pw)
            b = fr.get("ball_data") or {}
            if b.get("x") is not None:
                ball[t] = (b["x"] * L / pl, b["y"] * W / pw)
                ball_z[t] = b["z"] if b.get("z") is not None else np.nan
            g = (fr.get("possession") or {}).get("group")
            poss[t] = 0 if g == "home team" else 1 if g == "away team" else -1
        # on the pitch = the match sheet's playing time in this period (the provider lists nobody while the
        # broadcast shows replays / close-ups, but the players are still on the pitch)
        frames_no = np.array([fr["frame"] for fr in frames])
        sheet_on = np.zeros((T, N), bool)
        have_sheet = np.zeros(N, bool)
        for j, sid in enumerate(ids):
            for bp in (players[sid].get("playing_time") or {}).get("by_period") or []:
                if bp["name"] == f"period_{period}":
                    sheet_on[:, j] = (frames_no >= int(bp["start_frame"])) & (frames_no <= int(bp["end_frame"]))
                    have_sheet[j] = True
        on = np.where(have_sheet[None], sheet_on | on, on)
        pos = pos + [L / 2, W / 2]
        ball = ball + [L / 2, W / 2]
        if flip:  # home attacks +x in every half
            pos = np.stack([L - pos[..., 0], W - pos[..., 1]], -1)
            ball = np.stack([L - ball[:, 0], W - ball[:, 1]], 1)
        ball_seen = np.isfinite(ball).all(1)
        last = np.array([L / 2, W / 2])
        for t in range(T):  # causal forward fill of missing ball samples
            if np.isfinite(ball[t]).all():
                last = ball[t].copy()
            else:
                ball[t] = last
        team = np.array(["A" if players[i]["team_id"] == home else "B" for i in ids])
        acr = [players[i]["player_role"]["acronym"] for i in ids]
        role = np.array([ROLE.get(a, "CM") for a in acr])
        anchors = []
        for a in acr:
            ln, lat = LINE_LAT.get(a, (2, 0))
            anchors.append(((ln - 1) / 2 * 30 - 15, -lat * 22))
        tracked = np.array([bool(fr["player_data"]) for fr in frames])
        f0 = int(frames[0]["frame"])
        alive, spells, starts = provider_stoppages(mdir, period, f0, T)
        yield f"{mid}_p{period}", {"clip": f"{mid}_p{period}", "game": mid, "pos": pos, "on": on, "ball": ball, "tracked": tracked, "ball_seen": ball_seen, "ball_z": ball_z,
                                   "team": team, "role": role, "poss": poss, "alive": alive, "stoppages": spells, "f0": f0,
                                   "possession_starts": starts,
                                   "anchors": anchors, "acronyms": acr}


def sc_clips():
    for mdir in sorted(glob.glob(str(SC_DIR / "data" / "matches" / "*"))):
        if os.path.isdir(mdir):
            for name, clip in load_match(mdir):
                yield "skillcorner", clip


def uncertainty_record(rows, method):
    """What the exports may claim about the uncertainty of `method` on SkillCorner re-entries: 80%-radius coverage and
    the worst gap between predicted and observed P(err <= 5 m) over probability bins with >= 200 rows."""
    out = {"basis": f"SkillCorner A-League re-entries (proxy truth = provider detections), model {method}",
           "applies_to": "estimated players only; measured positions remain unverified"}
    for mode in ("live", "replay"):
        r = [x for x in rows if x["mode"] == mode and f"err:{method}" in x]
        if not r:
            return None
        e, p5, r80 = (np.array([x[f"{k}:{method}"] for x in r]) for k in ("err", "p5", "r80"))
        b = np.digitize(p5, [0.2, 0.4, 0.6, 0.8])
        gap = max(abs(p5[b == k].mean() - (e[b == k] <= 5).mean()) for k in range(5) if (b == k).sum() >= 200)
        out[f"{mode}_r80_coverage"] = round(float((e <= r80).mean()), 3)
        out[f"{mode}_p5_max_bin_gap"] = round(float(gap), 3)
        out[f"{mode}_n"] = len(r)
    return out


def flank_check():
    """Convention check (no model error involved): own-frame y of right- vs left-sided players when detected."""
    out = {"right": [], "left": []}
    for _, c in sc_clips():
        for j, a in enumerate(c["acronyms"]):
            side = "right" if a in ("RB", "RCB", "RM", "RW", "RWB") else "left" if a in ("LB", "LCB", "LM", "LW", "LWB") else None
            if side:
                y = c["pos"][:, j, 1]
                y = y if c["team"][j] == "A" else W - y
                out[side].append(np.nanmean(y))
    return {k: float(np.nanmean(v)) for k, v in out.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="base,final_v3")
    ap.add_argument("--name", default="skillcorner")
    ap.add_argument("--flank-check", action="store_true")
    a = ap.parse_args()
    if a.flank_check:
        print(flank_check())
        raise SystemExit
    rows = run(a.methods.split(","), sc_clips())
    s = summarize(rows)
    (OUT / f"{a.name}_summary.json").write_text(json.dumps(s, indent=1))
    (OUT / f"{a.name}_rows.json").write_text(json.dumps(rows))
    for m in methods_learned(a.methods):
        rec = uncertainty_record(rows, m)
        if rec:
            (OUT / f"sc_uncertainty_{m}.json").write_text(json.dumps(rec, indent=1))
    for mode, res in s.items():
        print(f"\n{mode}: n={res['n']} halves={res['clips']} matches={len(res['games'])}")
        for nm, v in res["methods"].items():
            print(f"  {nm:28s} mean {v['mean']:.2f} median {v['median']:.2f} <=5m {v['within_5m']:.1%} match-mean {v['game_mean']:.2f}"
                  + (f" r80cov {v['r80_coverage']:.1%} r80 {v['r80_mean_m']:.1f}" if "r80_coverage" in v else "") + f" by_gap {v['by_gap']}")
