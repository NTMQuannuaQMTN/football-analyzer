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

API_PREFIX = "/api"  # public mount path on Vercel (vercel.json rewrites /api/* to this service)


class StripApiPrefix:
    """Serve every route both at /<route> (local: uvicorn on :8000) and /api/<route> (Vercel services routing,
    which forwards the original /api/... path)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            if path == API_PREFIX or path.startswith(API_PREFIX + "/"):
                scope = dict(scope, path=path[len(API_PREFIX):] or "/",
                             raw_path=(path[len(API_PREFIX):] or "/").encode())
        await self.app(scope, receive, send)


class CdnCache:
    """Let the CDN (Vercel edge) answer repeat GETs about finished matches without running the function: their analysis
    only changes with a new deployment, which purges the CDN cache. A live match (or the lobby while one is live)
    changes every minute and is never cached; /health is never cached."""
    LONG = "public, max-age=0, s-maxage=86400, stale-while-revalidate=86400"
    SHORT = "public, max-age=0, s-maxage=15"

    def __init__(self, app):
        self.app = app

    def policy(self, scope):
        from urllib.parse import parse_qs
        path = scope.get("path", "")
        if path.startswith(API_PREFIX + "/"):
            path = path[len(API_PREFIX):]
        if path in ("/health", "/match/stream") or not (path.startswith("/match") or path == "/evaluation"):
            return None
        if path == "/matches":
            return self.SHORT if catalog.live_clock(catalog.LIVE_MATCH) is not None else self.LONG
        q = parse_qs(scope.get("query_string", b"").decode())
        mid = path.split("/", 2)[2] if path.startswith("/matches/") else (q.get("match") or [None])[0]
        if mid is None:  # the default match: MATCH_FILE, else the synthetic match (never live)
            return self.LONG
        return None if catalog.live_clock(mid) is not None else self.LONG

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "GET":
            return await self.app(scope, receive, send)
        policy = self.policy(scope)

        async def send_with_cache(message):
            if policy and message["type"] == "http.response.start" and message.get("status") == 200:
                message = dict(message, headers=list(message.get("headers", [])) + [(b"cache-control", policy.encode())])
            await send(message)
        await self.app(scope, receive, send_with_cache)


app = FastAPI(title="Synthetic Match Insights Engine", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CdnCache)
app.add_middleware(StripApiPrefix)


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


def reconstruction_mode(I, mode):
    try:
        return I.resolve_mode(mode)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


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
    date_meta = catalog.video_details(I.m.meta) if I.m.meta.get("source", {}).get("type") == "computer_vision" else {}
    return {**I.m.meta, "date": I.m.meta.get("date") or date_meta.get("date") or (entry or {}).get("info", {}).get("date"), "players": I.m.players, "goals": goals, "live": catalog.live_info(match, part or (next(iter(entry["parts"])) if entry else None)) if match else None,
            "parts": parts,
            "key_events": [e for e in I.m.events if e["event_type"] in ("GOAL", "SUBSTITUTION", "RED_CARD")]}


@app.get("/match/frames")
def frames(start: int = 0, end: Optional[int] = None, mode: str = "default", I: MatchIntelligence = Depends(scope)):
    """mode: default (truth when the match has ground truth, else replay) | truth | replay (retrospective camera
    estimate) | live (causal camera estimate: no future frames)."""
    mode = reconstruction_mode(I, mode)
    end = I.m.duration if end is None else end
    return {"player_order": I.m.order, "mode": mode, "view": I.view, "frames": I.frames(mode)[max(0, start):end + 1]}


@app.get("/match/reveals")
def reveals(I: MatchIntelligence = Depends(scope)):
    """Tracking-ground-truth or CV-proxy reappearance checks, distinguished by source."""
    source = I.m.meta.get("source") or {}
    report = source.get("error_vs_cv_proxy") if source.get("type") == "computer_vision" else source.get("error_vs_truth")
    # Older exports stored CV proxies under error_vs_truth; the UI still labels
    # these as consistency checks rather than ground-truth accuracy.
    if report is None:
        report = source.get("error_vs_truth", {})
    summary = report.get("reveals")
    return {"summary": summary, "reveals": I.reveals}


@app.get("/evaluation")
def evaluation(I: MatchIntelligence = Depends(scope)):
    """Reconstruction + tactical accuracy against ground truth (data/eval/)."""
    return I.evaluation()


@app.get("/match/players/states")
def player_states(t: float = 0, mode: str = "replay", I: MatchIntelligence = Depends(scope)):
    return I.player_states(t, reconstruction_mode(I, mode))


@app.get("/match/players/{pid}/state")
def player_state(pid: str, t: float = 0, mode: str = "replay", I: MatchIntelligence = Depends(scope)):
    """Observed vs inferred state of one player at t, with confidence, uncertainty, evidence and observed history."""
    return I.player_state(pid, t, reconstruction_mode(I, mode))


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
def players(t: Optional[float] = None, mode: str = "default", I: MatchIntelligence = Depends(scope)):
    mode = reconstruction_mode(I, mode)
    return [I.player_profile(p["id"], t, mode) for p in I.m.players]


@app.get("/match/players/{pid}")
def player(pid: str, t: Optional[float] = None, mode: str = "default", I: MatchIntelligence = Depends(scope)):
    if pid not in I.m.player_by_id:
        raise HTTPException(404, "player not found")
    return I.player_profile(pid, t, reconstruction_mode(I, mode))


@app.get("/match/changes")
def changes(I: MatchIntelligence = Depends(scope)):
    """Every tactical and formation change for both teams, with before / after evidence (backend/tactics/changes.py)."""
    return I.changes


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
