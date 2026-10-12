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

# Stop only verified listeners belonging to this checkout. Refuse foreign ports.
.venv/bin/python scripts/dev_server_guard.py --root "$PWD" --stop
for port in 8000 3000; do
  for _ in $(seq 1 20); do lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 0.25; done
  if lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is used by another program:"; lsof -nP -iTCP:$port -sTCP:LISTEN; exit 1
  fi
done

.venv/bin/uvicorn backend.api.main:app --port 8000 &
API=$!
trap 'kill $API 2>/dev/null || true' EXIT
echo "Building match intelligence${MATCH_FILE:+ from $MATCH_FILE}..."
(cd frontend && NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL:-http://localhost:8000}" npm run dev -- --port 3000)
