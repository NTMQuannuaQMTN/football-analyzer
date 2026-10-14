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
        yield f"{mid}_p{period}", {"clip": f"{mid}_p{period}", "game": mid, "pos": pos, "on": on, "ball": ball, "tracked": tracked, "ball_seen": ball_seen,
                                   "team": team, "role": role, "poss": poss, "alive": np.ones(T, bool),
                                   "anchors": anchors, "acronyms": acr}


def sc_clips():
    for mdir in sorted(glob.glob(str(SC_DIR / "data" / "matches" / "*"))):
        if os.path.isdir(mdir):
            for name, clip in load_match(mdir):
                yield "skillcorner", clip


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
    for mode, res in s.items():
        print(f"\n{mode}: n={res['n']} halves={res['clips']} matches={len(res['games'])}")
        for nm, v in res["methods"].items():
            print(f"  {nm:28s} mean {v['mean']:.2f} median {v['median']:.2f} <=5m {v['within_5m']:.1%} match-mean {v['game_mean']:.2f}"
                  + (f" r80cov {v['r80_coverage']:.1%} r80 {v['r80_mean_m']:.1f}" if "r80_coverage" in v else "") + f" by_gap {v['by_gap']}")
