"""Real football data for hidden-player reconstruction.

DFL / IDSSE open data (Bassek et al. 2025, Sci Data, CC BY 4.0; figshare 10.6084/m9.figshare.28196177)
  7 Bundesliga / 2. Bundesliga matches, TRACAB Gen5 in-stadium optical tracking: all 22 players + ball,
  25 Hz, identities and playing positions. FULL-PITCH GROUND TRUTH (no broadcast view).
SkillCorner Open Data (MIT, github.com/SkillCorner/opendata)
  10 A-League matches of BROADCAST tracking at 10 fps: per frame the camera's footprint on the pitch
  (image_corners_projection) and per-player is_detected flags. Hidden positions there are SkillCorner's own
  extrapolations, so they are NOT used as ground truth. We use the real camera behaviour (pan/zoom/cuts and
  detection misses) to mask DFL ground truth realistically.

Unified representation (one half = one segment; 1 Hz; metres, 0..105 x 0..68; home attacks +x):
  {"match_id","segment","t","player_id","team_id","role","x","y","visible","camera"} via rows(); arrays via load_segments().
"""
import glob
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DFL_DIR = ROOT / "data" / "dfl"
SC_DIR = ROOT / "data" / "skillcorner"
CACHE = ROOT / "data" / "eval" / "real"
L, W = 105.0, 68.0
HZ = 25

# DFL playing positions -> coarse roles (used for role priors / per-role error)
ROLE = {"TW": "GK", "IVL": "CB", "IVR": "CB", "IVZ": "CB", "LV": "FB", "RV": "FB",
        "DML": "DM", "DMR": "DM", "DMZ": "DM", "DLM": "CM", "DRM": "CM", "ZM": "CM", "ZO": "AM",
        "OLM": "WM", "ORM": "WM", "LM": "WM", "RM": "WM", "LA": "W", "RA": "W", "HL": "W", "HR": "W",
        "STZ": "ST", "STL": "ST", "STR": "ST", "HST": "ST"}
ROLES = ["GK", "CB", "FB", "DM", "CM", "AM", "WM", "W", "ST"]


# ----------------------------------------------------------------------------- DFL
def dfl_matches():
    return sorted({Path(p).stem.split("_")[-1] for p in glob.glob(str(DFL_DIR / "DFL_04_03_positions_raw_observed_*.xml"))})


def _match_info(mid):
    f = glob.glob(str(DFL_DIR / f"DFL_02_01_matchinformation_*_{mid}.xml"))[0]
    root = ET.parse(f).getroot()
    g = root.find(".//General")
    info = {"match_id": mid, "title": g.get("MatchTitle"), "competition": g.get("CompetitionName"),
            "home": g.get("HomeTeamId"), "away": g.get("GuestTeamId"), "players": {}}
    for team in root.iter("Team"):
        for p in team.iter("Player"):
            pos = p.get("PlayingPosition")
            info["players"][p.get("PersonId")] = {"team": "A" if team.get("TeamId") == info["home"] else "B",
                                                  "role": ROLE.get(pos, "CM") if pos else "SUB", "dfl_position": pos,
                                                  "shirt": p.get("ShirtNumber"), "name": p.get("Shortname"),
                                                  "starting": p.get("Starting") == "true"}
        info["lineup_" + ("A" if team.get("TeamId") == info["home"] else "B")] = team.get("LineUp")
    return info


def dfl_cache_path(mid,hz_out):
    if isinstance(hz_out,bool) or not isinstance(hz_out,(int,np.integer)) or hz_out<=0 or HZ%hz_out:
        raise ValueError('DFL sampling must be an integer divisor of 25 Hz: 1, 5 or 25')
    return CACHE / (f'dfl_{mid}.npz' if hz_out==1 else f'dfl_{mid}_{hz_out}hz.npz')


def initial_home_orientation(pos,team,roles):
    """Known full-tracking setup at the first retained frame; never use a future half mean."""
    keep=(np.asarray(team)=='A')&(np.asarray(roles)=='GK')
    values=np.asarray(pos)[0,keep,0];values=values[np.isfinite(values)]
    if not len(values):raise ValueError('Initial home goalkeeper unavailable; an explicit attack direction is required')
    return bool(values.mean()>L/2)


def parse_dfl(mid, hz_out=1):
    """Stream 25 Hz XML at 1/5/25 Hz; use a separate cache for each rate.

    Timestamps are relative to the first retained source frame, not verified
    match kickoff or broadcast clock. Existing reconstruction models use 1 Hz.
    """
    cache = dfl_cache_path(mid,hz_out)
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        info=d['info'].item();segments=list(d['segments'])
        recorded=info.get('sampling_hz',1)
        if recorded!=hz_out:raise ValueError('DFL cache sampling rate does not match the requested rate')
        return info,segments
    info = _match_info(mid)
    info['sampling_hz']=hz_out
    f = glob.glob(str(DFL_DIR / f"DFL_04_03_positions_raw_observed_*_{mid}.xml"))[0]
    step = HZ // hz_out
    data = {}  # (section, person) -> {N: (x, y)}
    ball = {}  # section -> {N: (x, y, poss, status)}
    sect = person = team = None
    for ev, el in ET.iterparse(f, events=("start", "end")):
        if ev == "start" and el.tag == "FrameSet":
            sect, person, team = el.get("GameSection"), el.get("PersonId"), el.get("TeamId")
        elif ev == "end" and el.tag == "Frame":
            n = int(el.get("N"))
            if n % step == 0:
                x, y = float(el.get("X")), float(el.get("Y"))
                if team == "BALL":
                    ball.setdefault(sect, {})[n] = (x, y, int(el.get("BallPossession") or 0), int(el.get("BallStatus") or 0))
                elif person in info["players"]:
                    data.setdefault((sect, person), {})[n] = (x, y)
            el.clear()
        elif ev == "end" and el.tag == "FrameSet":
            el.clear()
    segments = []
    pids = sorted(info["players"])
    for sect in sorted(ball):
        ns = np.array(sorted(ball[sect]))
        T = len(ns)
        pos = np.full((T, len(pids), 2), np.nan)
        index = {n: i for i, n in enumerate(ns)}
        for j, pid in enumerate(pids):
            for n, (x, y) in data.get((sect, pid), {}).items():
                if n in index:
                    pos[index[n], j] = (x + L / 2, y + W / 2)
        b = np.array([ball[sect][n] for n in ns], dtype=float)
        bxy = np.stack([b[:, 0] + L / 2, b[:, 1] + W / 2], 1)
        # orient: home team attacks +x in every segment (rotate 180 degrees if needed)
        team = np.array([info["players"][p]["team"] for p in pids])
        roles=np.array([info['players'][p]['role'] for p in pids])
        if initial_home_orientation(pos,team,roles):
            pos = np.stack([L - pos[..., 0], W - pos[..., 1]], -1)
            bxy = np.stack([L - bxy[:, 0], W - bxy[:, 1]], 1)
        poss = np.where(b[:, 2] == 1, 0, np.where(b[:, 2] == 2, 1, -1))
        segments.append({"section": sect, "pos": pos, "ball": bxy, "poss": poss, "alive": b[:, 3] == 1,
                         "pids": pids, "team": team, "role": roles,
                         'sampling_hz':hz_out,'timestamps_s':(ns-ns[0])/HZ,
                         'orientation_basis':'Home goalkeeper at the first retained frame; known full-tracking setup, not inferred broadcast identity'})
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, info=np.array(info, dtype=object), segments=np.array(segments, dtype=object))
    return info, segments


def rows(info, seg, vis=None, camera="broadcast_sim", every=1):
    """Unified row representation (generator)."""
    for t in range(0, len(seg["pos"]), every):
        for j, pid in enumerate(seg["pids"]):
            x, y = seg["pos"][t, j]
            if np.isnan(x):
                continue
            timestamp=float(seg['timestamps_s'][t]) if 'timestamps_s' in seg else float(t)/seg.get('sampling_hz',1)
            yield {"match_id": info["match_id"], "segment": seg["section"], "timestamp": timestamp, "player_id": pid,
                   "team_id": seg["team"][j], "role": seg["role"][j], "x": round(float(x), 2), "y": round(float(y), 2),
                   "visible": bool(vis[t, j]) if vis is not None else True, "camera": camera if vis is not None else "full_pitch"}


# ----------------------------------------------------------------------------- SkillCorner camera footprints
def skillcorner_camera_cache_path(hz_out):
    if isinstance(hz_out,bool) or not isinstance(hz_out,(int,np.integer)) or hz_out<=0 or 10%hz_out:
        raise ValueError('SkillCorner sampling must be an integer divisor of 10 Hz: 1, 2, 5 or 10')
    return CACHE / ('skillcorner_cameras.npz' if hz_out==1 else f'skillcorner_cameras_{hz_out}hz.npz')


def skillcorner_cameras(hz_out=1):
    """Per SkillCorner match-period: 1 Hz arrays of camera footprint corners (T,4,2), ball (T,2),
    detected-player count, and detection rate inside the footprint. Coordinates converted to 0..105 x 0..68."""
    cache = skillcorner_camera_cache_path(hz_out)
    if cache.exists():
        return list(np.load(cache, allow_pickle=True)["cams"])
    cams = []
    for mdir in sorted(glob.glob(str(SC_DIR / "data" / "matches" / "*"))):
        files = glob.glob(os.path.join(mdir, "*tracking_extrapolated.jsonl"))
        if not files:
            continue
        per = {}
        with open(files[0]) as fh:
            for i, line in enumerate(fh):
                if i % (10 // hz_out):
                    continue
                fr = json.loads(line)
                per.setdefault(fr.get("period"), []).append(fr)
        meta = glob.glob(os.path.join(mdir, "*match.json"))
        pl, pw = 105.0, 68.0
        if meta:
            mj = json.load(open(meta[0]))
            pl, pw = float(mj.get("pitch_length") or 105), float(mj.get("pitch_width") or 68)
        for period, frames in per.items():
            if period is None:
                continue
            corners, ballxy, ndet, rate = [], [], [], []
            for fr in frames:
                c = fr.get("image_corners_projection")
                bd = fr.get("ball_data") or {}
                players = fr.get("player_data") or []
                det = [p for p in players if p.get("is_detected")]
                if c and all(v is not None for v in (c.values() if isinstance(c, dict) else c)):
                    if isinstance(c, dict):
                        q = [[c["x_top_left"], c["y_top_left"]], [c["x_top_right"], c["y_top_right"]],
                             [c["x_bottom_right"], c["y_bottom_right"]], [c["x_bottom_left"], c["y_bottom_left"]]]
                    else:
                        q = np.array(c, dtype=float).reshape(4, 2).tolist()
                    q = np.array(q, dtype=float) * [L / pl, W / pw] + [L / 2, W / 2]
                else:
                    q = np.full((4, 2), np.nan)
                bx = bd.get("x")
                ballxy.append([bx * L / pl + L / 2, bd.get("y") * W / pw + W / 2] if bx is not None and bd.get("y") is not None else [np.nan, np.nan])
                corners.append(q)
                ndet.append(len(det))
                inside = [p for p in players if p.get("x") is not None and not np.isnan(q).any() and point_in_quad(
                    np.array([[p["x"] * L / pl + L / 2, p["y"] * W / pw + W / 2]]), q)[0]]
                rate.append(sum(1 for p in inside if p.get("is_detected")) / len(inside) if inside else np.nan)
            cams.append({"match": os.path.basename(mdir), "period": period, "corners": np.array(corners),
                         "ball": np.array(ballxy), "n_detected": np.array(ndet), "det_rate": np.array(rate)})
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, cams=np.array(cams, dtype=object))
    return cams


def reconstruction_input_fingerprints(matches,masks):
    """Hash the parsed data actually consumed; raw-source provenance is separate.

    Existing caches may predate raw-file provenance. This prevents feature-cache
    reuse after those consumed arrays change; it does not certify raw/cache agreement.
    """
    import hashlib
    paths=[]
    for match in matches:
        parse_dfl(match)
        paths.append(dfl_cache_path(match,1))
    if 'sc_camera' in masks:
        skillcorner_cameras()
        paths.append(skillcorner_camera_cache_path(1))
    result={}
    for path in sorted(set(paths)):
        before=path.stat()
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise RuntimeError('Reconstruction input changed while hashing; retry after the writer finishes')
        result[path.name]=digest.hexdigest()
    return result


def point_in_quad(P, q):
    """P (n,2) inside convex-ish quadrilateral q (4,2) (consistent winding)."""
    P = np.asarray(P, dtype=float)
    s = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        s.append((b[0] - a[0]) * (P[:, 1] - a[1]) - (b[1] - a[1]) * (P[:, 0] - a[0]))
    s = np.stack(s, 1)
    return np.all(s >= 0, 1) | np.all(s <= 0, 1)


def dfl_ball_height(mid, hz_out=1):
    """Per section: ball height (m) per output sample = max Z within that second (TRACAB measures the ball in 3D).
    Aligned with parse_dfl segments. Cached."""
    cache = CACHE / f"dfl_ballz_{mid}.npz"
    if cache.exists():
        return dict(np.load(cache, allow_pickle=True)["z"].item())
    f = glob.glob(str(DFL_DIR / f"DFL_04_03_positions_raw_observed_*_{mid}.xml"))[0]
    step = HZ // hz_out
    per = {}
    sect, is_ball = None, False
    for ev, el in ET.iterparse(f, events=("start", "end")):
        if ev == "start" and el.tag == "FrameSet":
            sect, is_ball = el.get("GameSection"), el.get("TeamId") == "BALL"
        elif ev == "end" and el.tag == "Frame":
            if is_ball:
                n = int(el.get("N"))
                k = n - n % step
                z = float(el.get("Z") or 0)
                d = per.setdefault(sect, {})
                d[k] = max(d.get(k, 0.0), z)
            el.clear()
        elif ev == "end" and el.tag == "FrameSet":
            el.clear()
    out = {s: np.array([d[k] for k in sorted(d)]) for s, d in per.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, z=np.array(out, dtype=object))
    return out
