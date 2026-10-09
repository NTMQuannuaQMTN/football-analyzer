"""FastAPI surface. Run: uvicorn backend.api.main:app --port 8000 (from repo root).

Every /match/... endpoint takes optional `match` (lobby id, see /matches) and `part` (h1 / h2 / clip / full)
query parameters; without them it serves the default match (MATCH_FILE, else the synthetic match).
"""
import asyncio
import json
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse

from backend import catalog
from backend.models.schemas import Event, Moment
from backend.pipeline import MatchIntelligence, get_intel

app = FastAPI(title="Synthetic Match Insights Engine", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.on_event("startup")
def warm():
    get_intel()
    for m in catalog._catalog_static():  # analyse the live match (both halves) up front
        if m["id"] == catalog.LIVE_MATCH:
            for h in m["parts"]:
                catalog.intel(m["id"], h)


def scope(match: Optional[str] = None, part: Optional[str] = None) -> MatchIntelligence:
    """Which match (lobby id) and part a request is about; none = the default match."""
    try:
        return catalog.intel(match, part)
    except KeyError as e:
        raise HTTPException(404, f"unknown match/part: {e}")


# ---------------------------------------------------------------- lobby
@app.get("/matches")
def matches():
    """Lobby: live and past matches."""
    return catalog.lobby()


@app.get("/matches/{mid}")
def match_detail(mid: str):
    """Whole-match highlights and analysis (all parts; a live match up to its current minute)."""
    try:
        return catalog.match_summary(mid)
    except KeyError:
        raise HTTPException(404, "match not found")


# ---------------------------------------------------------------- one match (part)
@app.get("/health")
def health():
    return {"ok": True, "build_seconds": get_intel().build_seconds}


@app.get("/match")
def match(match: Optional[str] = None, part: Optional[str] = None, I: MatchIntelligence = Depends(scope)):
    goals = [e for e in I.m.events if e["event_type"] == "GOAL"]
    entry = catalog.entry(match) if match else None
    parts = [{"id": h, "label": catalog.PART_LABEL.get(h, h)} for h in entry["parts"]] if entry else None
    return {**I.m.meta, "players": I.m.players, "goals": goals, "live": catalog.live_info(match, part or (next(iter(entry["parts"])) if entry else None)) if match else None,
            "parts": parts,
            "key_events": [e for e in I.m.events if e["event_type"] in ("GOAL", "SUBSTITUTION", "RED_CARD")]}


@app.get("/match/frames")
def frames(start: int = 0, end: Optional[int] = None, mode: str = "default", I: MatchIntelligence = Depends(scope)):
    """mode: default (truth when the match has ground truth, else replay) | truth | replay (retrospective camera
    estimate) | live (causal camera estimate: no future frames)."""
    end = I.m.duration if end is None else end
    return {"player_order": I.m.order, "mode": mode, "view": I.view, "frames": I.frames(mode)[max(0, start):end + 1]}


@app.get("/match/reveals")
def reveals(I: MatchIntelligence = Depends(scope)):
    """Prediction -> reality: for every player who came back into the camera view after >= 5 s hidden, what the
    live (causal) model predicted at that second, the 80% radius, and where the player actually was."""
    summary = (I.m.meta.get("source") or {}).get("error_vs_truth", {}).get("reveals")
    return {"summary": summary, "reveals": I.reveals}


@app.get("/evaluation")
def evaluation(I: MatchIntelligence = Depends(scope)):
    """Reconstruction + tactical accuracy against ground truth (data/eval/)."""
    return I.evaluation()


@app.get("/match/players/states")
def player_states(t: float = 0, mode: str = "replay", I: MatchIntelligence = Depends(scope)):
    return I.player_states(t, mode)


@app.get("/match/players/{pid}/state")
def player_state(pid: str, t: float = 0, mode: str = "replay", I: MatchIntelligence = Depends(scope)):
    """Observed vs inferred state of one player at t, with confidence, uncertainty, evidence and observed history."""
    return I.player_state(pid, t, mode)


@app.get("/match/state")
def state(t: float = Query(0, ge=0), I: MatchIntelligence = Depends(scope)):
    return I.state_at(t)


@app.get("/match/timeline")
def timeline(I: MatchIntelligence = Depends(scope)):
    return I.timeline


@app.get("/match/events", response_model=List[Event])
def events(start: float = 0, end: float = 1e9, types: Optional[str] = None, team: Optional[str] = None,
           I: MatchIntelligence = Depends(scope)):
    return I.m.events_between(start, end, types.split(",") if types else None, team)


@app.get("/match/moments", response_model=List[Moment])
def moments(I: MatchIntelligence = Depends(scope)):
    return I.moments


def _moment(I, moment_id):
    mo = next((m for m in I.moments if m["id"] == moment_id), None)
    if not mo:
        raise HTTPException(404, "moment not found")
    return mo


@app.get("/match/moments/{moment_id}", response_model=Moment)
def moment(moment_id: str, I: MatchIntelligence = Depends(scope)):
    return _moment(I, moment_id)


@app.get("/match/moments/{moment_id}/overlay")
def overlay(moment_id: str, I: MatchIntelligence = Depends(scope)):
    """Minimal broadcast-graphics payload for a moment."""
    mo = _moment(I, moment_id)
    return {"id": mo["id"], "timestamp": mo["timestamp"], "type": mo["type"], "team": mo["team"],
            "lower_third": mo["narrative"]["broadcast"]["lower_third"], "headline": mo["narrative"]["fan"]["headline"],
            "highlight_players": mo["replay"]["focus_players"], "importance": mo["importance"],
            "confidence": mo["confidence"]}


@app.get("/match/players")
def players(t: Optional[float] = None, I: MatchIntelligence = Depends(scope)):
    return [I.player_profile(p["id"], t) for p in I.m.players]


@app.get("/match/players/{pid}")
def player(pid: str, t: Optional[float] = None, I: MatchIntelligence = Depends(scope)):
    if pid not in I.m.player_by_id:
        raise HTTPException(404, "player not found")
    return I.player_profile(pid, t)


@app.get("/match/story")
def story(I: MatchIntelligence = Depends(scope)):
    return I.story


@app.get("/match/ground-truth")
def ground_truth(I: MatchIntelligence = Depends(scope)):
    return I.ground_truth()


@app.get("/match/stream")
async def stream(speed: float = 10.0, start: float = 0, I: MatchIntelligence = Depends(scope)):
    """SSE replay: events + moment notifications at `speed`x real time.
    Stand-in for an Azure Event Hubs / Web PubSub live feed."""

    async def gen():
        t = start
        evs = [e for e in I.m.events if e["timestamp"] >= start and e["event_type"] != "PRESSURE"]
        mos = [m for m in I.moments if m["timestamp"] >= start]
        i = j = 0
        while t <= I.m.duration:
            while i < len(evs) and evs[i]["timestamp"] <= t:
                yield f"event: match_event\ndata: {json.dumps(evs[i])}\n\n"
                i += 1
            while j < len(mos) and mos[j]["timestamp"] <= t:
                m = mos[j]
                yield f"event: moment\ndata: {json.dumps({'id': m['id'], 'title': m['title'], 'type': m['type'], 'timestamp': m['timestamp']})}\n\n"
                j += 1
            if int(t) % 10 == 0:
                st = I.state_at(t)
                yield f"event: state\ndata: {json.dumps({'timestamp': t, 'score': st['score'], 'possession': st['possession']})}\n\n"
            t += 1
            await asyncio.sleep(1 / speed)

    return StreamingResponse(gen(), media_type="text/event-stream")
