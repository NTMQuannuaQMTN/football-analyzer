"""Match catalog for the lobby: which matches exist, their status, and a whole-match summary.

  real tracking   data/dfl_<match>_h1.json + _h2.json (backend/reconstruction/dfl_match.py), one match, two halves
  computer vision data/cv_*.json (cv/build_match.py), a clip of a broadcast
  EPL test match  data/epl_<match>_h1.json + _h2.json: a whole EPL broadcast (SoccerNet video, local only under its NDA)
                  run through our own CV + hidden-player model; when present it replaces the synthetic match in the lobby
  synthetic       the scripted simulator (a test match with exact ground truth; not meant to look like real football)
LIVE: one recorded real match (LIVE_MATCH, DFL tracking) is replayed as a live feed: its clock runs in real time from
LIVE_STARTED_MIN minutes before the server started, through the 1st half into the 2nd. Labelled as a replay.
"""
import glob
import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import date
from functools import lru_cache
from pathlib import Path

from backend.pipeline import get_intel
from simulator.match_generator import generate_match

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LIVE_SEED = 7  # synthetic test match
# Fortuna Düsseldorf vs F.C. Hansa Rostock (2. Bundesliga); the r13 copy has the same truth and events but a camera
# estimate built without event pins (pins fit a clock lag over the whole half, i.e. future information in live)
LIVE_MATCH = os.getenv("LIVE_MATCH") or ("dfl-J03WOY-r13" if (DATA / "dfl_J03WOY_r13_h1.json").exists() else "dfl-J03WOY")
SHOW_SUPERSEDED = os.getenv("SHOW_SUPERSEDED", "0") == "1"
LIVE_SPEED = 1.0
LIVE_NOTE = "Simulated live feed: a recorded real match (DFL tracking data) replayed in real time"
LIVE_STARTED_MIN = float(os.getenv("LIVE_STARTED_MIN", "20"))
_SERVER_START = time.time()
PART_LABEL = {"h1": "1st half", "h2": "2nd half", "clip": "Clip", "full": "Full match"}


def video_details(meta):
    """Use exported provenance; a provider match ID is not an ISO date."""
    src = meta.get("source", {})
    game = src.get("game", "")
    match_date = None
    for value in (meta.get("date"), src.get("date"), game.split("/")[-1][:10]):
        try:
            match_date = date.fromisoformat(value).isoformat()
            break
        except (TypeError, ValueError):
            continue
    provider = src.get("provider") or ("SoccerNet" if game.startswith("england_epl/") else "Broadcast")
    competition = src.get("competition") or ("Premier League" if game.startswith("england_epl/") else None)
    return {"date": match_date, "competition": competition,
            "label": 'Experimental reconstruction · EPL' if src.get('experimental') else f"Computer vision · {provider}"}


@lru_cache(maxsize=4)
def _synthetic_meta(seed):
    """A lobby card needs simulator facts, not two reconstruction passes (the shipped precomputed analysis already
    holds them; otherwise only the simulator runs)."""
    pre = precomputed_meta(seed)
    return pre if pre is not None else generate_match(seed)["meta"]


def precomputed_meta(seed):
    from backend import precomputed
    I = precomputed.load(seed)
    return I.raw_meta if I is not None and hasattr(I, "raw_meta") else None


@lru_cache(maxsize=128)
def _file_meta(path: str, mtime: float) -> dict:
    d = json.load(open(path))
    return {"meta": d["meta"], "goals": [e for e in d["events"] if e["event_type"] == "GOAL"]}


def file_meta(path) -> dict:
    return _file_meta(str(path), os.path.getmtime(path))


def _dfl_info(mid: str) -> dict:
    f = glob.glob(str(DATA / "dfl" / f"DFL_02_01_matchinformation_*_{mid}.xml"))
    if not f:
        return {}
    g = ET.parse(f[0]).getroot().find(".//General")
    return {"date": (g.get("KickoffTime") or "")[:10], "season": g.get("Season"), "matchday": g.get("MatchDay"),
            "result": g.get("Result"), "competition": g.get("CompetitionName")}


@lru_cache(maxsize=1)
def _catalog_static() -> tuple:
    out = []
    halves = {}
    for p in sorted(glob.glob(str(DATA / "dfl_*_h[12].json"))):
        key, half = Path(p).stem.rsplit("_", 1)
        halves.setdefault(key, {})[half] = p
    for key, parts in halves.items():
        meta=file_meta(parts[next(iter(parts))])['meta']
        mid = meta.get('source',{}).get('match_id') or "DFL-MAT-" + key.split("_", 1)[1]
        info = _dfl_info(mid)
        label='Experimental reconstruction · DFL' if meta.get('source',{}).get('experimental') else 'Real tracking · DFL open data'
        # r13 re-exports are the current version of these matches (same truth / events; newer camera estimate)
        out.append({"id": key.replace("_", "-"), "kind": "tracking", "label": label,
                    "parts": {h: parts[h] for h in sorted(parts)}, "info": info})
    # broadcast tracking (SkillCorner open data, backend/reconstruction/skillcorner_match.py): one match, two halves;
    # observed = provider broadcast detections, hidden players = our model's estimates
    sk = {}
    for p in sorted(glob.glob(str(DATA / "skc_*_h[12].json"))):
        key, half = Path(p).stem.rsplit("_", 1)
        sk.setdefault(key, {})[half] = p
    for key, parts in sk.items():
        meta = file_meta(parts[next(iter(parts))])["meta"]
        src = meta.get("source", {})
        out.append({"id": key.replace("_", "-"), "kind": "computer_vision",
                    "label": "Broadcast tracking · SkillCorner · hidden players estimated",
                    "parts": {h: parts[h] for h in sorted(parts)},
                    "info": {"date": src.get("date") or meta.get("date"), "competition": src.get("competition"),
                             "full_match": True}})
    for p in sorted(glob.glob(str(DATA / "cv_*.json"))):
        if "backup" in p:
            continue
        details = video_details(file_meta(p)["meta"])
        if file_meta(p)['meta'].get('source',{}).get('experimental'):details['label']='Experimental reconstruction · EPL'
        out.append({"id": "cv-" + Path(p).stem[3:].replace("_", "-")[:60], "kind": "computer_vision",
                    "label": details["label"], "parts": {"clip": p}, "info": details})
    # whole EPL match, both halves of the broadcast through our own CV (cv/build_match.py + apply_exports re-inference)
    epl = {}
    for p in sorted(glob.glob(str(DATA / "epl_*_h[12].json"))):
        key, half = Path(p).stem.rsplit("_", 1)
        epl.setdefault(key, {})[half] = p
    for key, parts in epl.items():
        details = video_details(file_meta(parts[next(iter(parts))])["meta"])
        out.append({"id": key.replace("_", "-"), "kind": "computer_vision",
                    "label": "Test match · EPL broadcast video · our CV, hidden players estimated",
                    "parts": {h: parts[h] for h in sorted(parts)},
                    "info": {**details, "full_match": True, "test_match": True}})
    out.append({"id": f"sim-{LIVE_SEED}", "kind": "synthetic", "label": "Synthetic test match · scripted simulator",
                "parts": {"full": None}, "info": {"competition": "Synthetic test league"}})
    return tuple(out)


def entry(mid):
    return next((m for m in _catalog_static() if m["id"] == mid), None)


def intel(mid=None, part=None):
    """MatchIntelligence for one part of a catalogued match (None: the server's default match)."""
    if mid is None:
        return get_intel()
    m = entry(mid)
    if m is None:
        raise KeyError(mid)
    part = part or next(iter(m["parts"]))
    if part not in m["parts"]:
        raise KeyError(f"{mid}/{part}")
    return get_intel(LIVE_SEED, m["parts"][part])


def live_clock(mid):
    """Seconds of the live match played so far, across its parts (None if this match is not live)."""
    if mid != LIVE_MATCH or entry(mid) is None:
        return None
    return (time.time() - _SERVER_START) * LIVE_SPEED + LIVE_STARTED_MIN * 60


def _part_spans(mid):
    """[(part, start, duration)] with parts played back to back."""
    m = entry(mid)
    spans, start = [], 0.0
    for h, p in m["parts"].items():
        d = float(file_meta(p)["meta"]["duration"]) if p else float(intel(mid, h).m.duration)
        spans.append((h, start, d))
        start += d
    return spans


def part_until(mid, part):
    """Seconds of `part` played so far (None: not started); the part's full duration if the match is not live."""
    t = live_clock(mid)
    for h, start, d in _part_spans(mid):
        if h == part:
            if t is None:
                return d
            return None if t < start else min(t - start, d)
    raise KeyError(part)


def live_info(mid, part=None):
    t = live_clock(mid)
    if t is None:
        return None
    spans = _part_spans(mid)
    current = next((h for h, s, d in spans if t < s + d), spans[-1][0])
    part = part or current
    until = part_until(mid, part)
    _, s, d = next(x for x in spans if x[0] == part)
    return {"t_now": round(until or 0.0, 1), "speed": LIVE_SPEED, "started": until is not None,
            "finished": t >= s + d, "match_finished": t >= spans[-1][1] + spans[-1][2], "part": current, "note": LIVE_NOTE}


def _teams(meta):
    return {k: {"name": v["name"], "short": v.get("short") or v["name"][:3].upper(), "color": v["color"]} for k, v in meta["teams"].items()}


def superseded(m) -> bool:
    """A catalogued file replaced by a newer version of the same match: earlier experimental candidates (seven-hour
    session), originals for which a re-inferred r13 export exists, and the synthetic match once the EPL test match exists. Still reachable by URL; listed only with
    SHOW_SUPERSEDED=1. The live match is never hidden."""
    if m["id"] == LIVE_MATCH:
        return False
    if m["kind"] == "synthetic":  # the whole EPL test match replaces it when present (still reachable by URL)
        return any((e["info"] or {}).get("test_match") for e in _catalog_static())
    p = next(iter(m["parts"].values()))
    src = file_meta(p)["meta"].get("source") or {}
    if src.get("session") == "seven-hour" or "trial7h" in Path(p).name:
        return True
    ids = {e["id"] for e in _catalog_static()}
    if m["id"].startswith("dfl-") and not m["id"].endswith("-r13") and f"{m['id']}-r13" in ids:
        return True
    if m["id"].startswith("cv-") and not m["id"].startswith("cv-r13-") and "cv-r13-" + m["id"][3:] in ids:
        return True
    return False


def lobby():
    """Cards for the lobby. Cheap: reads file metadata only (the live match is analysed at server start)."""
    cards = []
    for m in _catalog_static():
        if not SHOW_SUPERSEDED and superseded(m):
            continue
        if m["kind"] == "synthetic":
            meta = _synthetic_meta(LIVE_SEED)
            cards.append({"id": m["id"], "kind": m["kind"], "label": m["label"], "teams": _teams(meta),
                          "score": dict(meta["final_score"]), "score_known": True, "status": "finished", "competition": m["info"].get("competition"),
                          "date": None, "parts": [{"id": "full", "label": PART_LABEL["full"]}],
                          "note": "Scripted, rule-based match used to test tactic detection against exact ground truth; not real football"})
            continue
        li = live_info(m["id"])
        if li and not li["match_finished"]:
            I = intel(m["id"], li["part"])
            until = li["t_now"]
            cards.append({"id": m["id"], "kind": m["kind"], "label": m["label"], "teams": _teams(I.m.meta),
                          "score": I.m.score_at(until), "status": "live",
                          "minute": int((until + (I.m.meta.get("clock_offset") or 0)) // 60) + 1,
                          "competition": m["info"].get("competition"), "date": m["info"].get("date"),
                          "season": m["info"].get("season"), "matchday": m["info"].get("matchday"),
                          "parts": [{"id": h, "label": PART_LABEL.get(h, h)} for h in m["parts"]], "note": li["note"],
                          "live_part": li["part"]})
            continue
        metas = {h: file_meta(p) for h, p in m["parts"].items()}
        first = next(iter(metas.values()))["meta"]
        last_key = list(metas)[-1]
        last = metas[last_key]
        score = dict(last["meta"].get("initial_score") or {"A": 0, "B": 0})
        for g in last["goals"]:
            score[g["team_id"]] += 1
        src = first.get("source", {})
        card = {"id": m["id"], "kind": m["kind"], "label": m["label"], "teams": _teams(first), "score": score,
                "score_known": first.get("score_known", True),
                "status": "finished", "competition": m["info"].get("competition") or src.get("competition"),
                "date": m["info"].get("date"), "season": m["info"].get("season"), "matchday": m["info"].get("matchday"),
                "parts": [{"id": h, "label": PART_LABEL.get(h, h)} for h in m["parts"]]}
        if m["info"].get("test_match"):
            shares = [(v["meta"].get("source") or {}).get("observed_share") for v in metas.values()]
            shares = [x for x in shares if x is not None]
            card["note"] = ("Whole match from the broadcast video through our own CV"
                            + (f" ({round(100 * sum(shares) / len(shares))}% of player-seconds detected)" if shares else "")
                            + "; replays and close-ups excluded, every other position is a model estimate. Player identities "
                              "are role-based guesses from the lineup, not shirt numbers. Events: SoccerNet action labels. "
                              "No reference tracking exists, so positional accuracy cannot be measured on this match")
        elif m["kind"] == "computer_vision" and m["info"].get("full_match"):
            share = src.get("observed_share")
            card["note"] = (f"Full match from broadcast detections ({round(share * 100)}% of player-seconds detected in the 1st half); "
                            "every other position is a model estimate. Events: provider CV, not an official feed")
            miss = (last["meta"].get("source") or {}).get("goals_not_in_provider_events")
            if miss:
                off = last["meta"].get("final_score") or {}
                card["score"] = {"A": off.get("A"), "B": off.get("B")}  # the official result, not just located goals
                card["note"] += (f". Final score is the official result; {sum(miss.values())} goal(s) could not be located "
                                 "in time from the provider data, so they have no goal moment")
        elif m["kind"] == "computer_vision":
            card.update(video_details(first))
            s, e = src.get("start", 0), src.get("end", 0)
            score_note = "score is for this clip" if card["score_known"] else "score N/A"
            card["note"] = f"{int(s // 60)}:{int(s % 60):02d}–{int(e // 60)}:{int(e % 60):02d} of half {src.get('half', 1)}; {score_note}"
        elif m["kind"] == "tracking" and len(m["parts"]) < 2:
            card["note"] = "Only one half available"
        cards.append(card)
    live = [c for c in cards if c["status"] == "live"]
    past = sorted((c for c in cards if c["status"] != "live"), key=lambda c: c.get("date") or "", reverse=True)
    return live + past  # live first, then newest past match first


_summaries = {}


def match_summary(mid):
    """Whole-match analysis, cached for finished matches until a part's file changes (a live match changes every
    minute and is always recomputed)."""
    m = entry(mid)
    if m is None:
        raise KeyError(mid)
    if live_clock(mid) is not None:
        return _match_summary(mid, m)
    key = (mid, tuple((p, os.stat(p).st_mtime_ns) if p else (None, None) for p in m["parts"].values()))
    if key not in _summaries:
        if len(_summaries) > 32:
            _summaries.clear()
        _summaries[key] = _match_summary(mid, m)
    return _summaries[key]


def _match_summary(mid, m):
    """Whole-match analysis: every part summarised, then combined (stats summed, possession by time on the ball)."""
    live = live_clock(mid) is not None
    parts = []
    for h in m["parts"]:
        until = part_until(mid, h) if live else None
        if live and until is None:
            continue  # this half has not started yet
        I = intel(mid, h)
        parts.append({"id": h, "label": PART_LABEL.get(h, h), **I.summary(until)})
    I0 = intel(mid, next(iter(m["parts"])))
    combined = {}
    for tm in ("A", "B"):
        c = {}
        for k in ("goals", "shots", "shots_on_target", "passes", "passes_completed", "tackles_won", "fouls", "corners",
                  "offsides", "yellow_cards", "red_cards", "possession_seconds"):
            vals = [p["teams"][tm][k] for p in parts]
            c[k] = None if any(v is None for v in vals) else sum(vals)  # unknown in any part -> unknown
        c["xg"] = round(sum(p["teams"][tm]["xg"] for p in parts), 2)
        c["pass_accuracy"] = round(c["passes_completed"] / max(1, c["passes"]), 3)
        dists = [p["teams"][tm]["distance_km"] for p in parts]
        c["distance_km"] = round(sum(dists), 1) if all(d is not None for d in dists) else None
        presses = [p["teams"][tm]["press_index"] for p in parts if p["teams"][tm]["press_index"] is not None]
        c["press_index"] = round(sum(presses) / len(presses), 1) if presses else None
        combined[tm] = c
    tot = combined["A"]["possession_seconds"] + combined["B"]["possession_seconds"]
    for tm in ("A", "B"):
        combined[tm]["possession"] = round(combined[tm]["possession_seconds"] / max(1, tot), 3)
    # players across parts
    pl = {}
    for p in parts:
        for x in p["players"]:
            y = pl.setdefault(x["id"], {**x, "passes": 0, "completed": 0, "tackles": 0, "shots": 0, "goals": 0,
                                         "assists": 0, "xg": 0.0, "distance_km": None, "top_speed_kmh": None})
            for k in ("passes", "completed", "tackles", "shots", "goals", "assists"):
                y[k] += x[k]
            y["xg"] = round(y["xg"] + (x["xg"] or 0), 2)
            if x["distance_km"] is not None:
                y["distance_km"] = round((y["distance_km"] or 0) + x["distance_km"], 2)
                y["top_speed_kmh"] = max(y["top_speed_kmh"] or 0, x["top_speed_kmh"] or 0)
    last = parts[-1]
    meta = I0.m.meta
    lob = next(c for c in lobby() if c["id"] == mid)
    return {"id": mid, "kind": m["kind"], "label": m["label"], "status": lob["status"], "minute": lob.get("minute"),
            "teams": _teams(meta), "score": lob["score"], "score_known": lob.get("score_known", True), "competition": lob.get("competition"), "date": lob.get("date"),
            "season": lob.get("season"), "matchday": lob.get("matchday"), "note": lob.get("note"),
            "parts": [{k: v for k, v in p.items() if k not in ("players",)} for p in parts],
            "team_stats": combined, "players": sorted(pl.values(), key=lambda x: -(x["goals"] * 10 + x["completed"])),
            "live": live_info(mid) if lob["status"] == "live" else None, "until": last["until"]}
