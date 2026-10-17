#!/usr/bin/env bash
# Starts the intelligence API (:8000) and the studio UI (:3000).
#   ./run.sh                                  synthetic match
#   MATCH_FILE=data/cv_<name>.json ./run.sh   real match from cv/build_match.py
#   MATCH_FILE=data/dfl_J03WMX_h1.json ./run.sh  real top-down tracking (backend/reconstruction/dfl_match.py)
set -e
cd "$(dirname "$0")"
[ -d .venv ] || { uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt; }
[ -d frontend/node_modules ] || (cd frontend && npm install)

if [ -n "$MATCH_FILE" ] && [ ! -f "$MATCH_FILE" ]; then
  echo "MATCH_FILE not found: $MATCH_FILE"; echo "Available:"; ls data/cv_*.json data/dfl_*.json 2>/dev/null || echo "  (none — run cv.build_match first)"; exit 1
fi

# Ports: API_PORT (default 8000) and WEB_PORT (default 3000). This checkout's own servers are stopped and restarted;
# a port held by another program is never touched; the next free port is used instead.
API_PORT="${API_PORT:-8000}"; WEB_PORT="${WEB_PORT:-3000}"
.venv/bin/python scripts/dev_server_guard.py --root "$PWD" --stop --api-port "$API_PORT" --web-port "$WEB_PORT"
free_port() { local p=$1; while lsof -nP -iTCP:$p -sTCP:LISTEN >/dev/null 2>&1; do p=$((p + 1)); done; echo $p; }
for _ in $(seq 1 20); do lsof -nP -iTCP:$API_PORT -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 0.25; done
for _ in $(seq 1 20); do lsof -nP -iTCP:$WEB_PORT -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 0.25; done
NEW_API=$(free_port "$API_PORT"); NEW_WEB=$(free_port "$WEB_PORT")
[ "$NEW_API" != "$API_PORT" ] && echo "API port $API_PORT is used by another program; using $NEW_API"
[ "$NEW_WEB" != "$WEB_PORT" ] && echo "Web port $WEB_PORT is used by another program; using $NEW_WEB"
API_PORT=$NEW_API; WEB_PORT=$NEW_WEB

# refresh the shipped synthetic-match analysis if the code changed (data/precomputed; commit it so the deploy loads it)
(.venv/bin/python -m backend.precomputed --if-stale >/dev/null 2>&1 &)
.venv/bin/uvicorn backend.api.main:app --port "$API_PORT" &
API=$!
trap 'kill $API 2>/dev/null || true' EXIT
echo "Building match intelligence${MATCH_FILE:+ from $MATCH_FILE}... API http://localhost:$API_PORT  UI http://localhost:$WEB_PORT"
(cd frontend && NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL:-http://localhost:$API_PORT}" npm run dev -- --port "$WEB_PORT")
