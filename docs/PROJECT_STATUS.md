# Project status: MatchSight AI

Purpose of this file: give a reviewer (human or AI model) enough context to decide **what to improve next**. It describes what the repo does, how it is built, what has been measured, and what is weak. All numbers come from files in `data/eval/` or from runs noted below. Nothing here is estimated.

Snapshot date: 2026-10-08 (updated after the autonomous improvement session).

---

## 1. What the product is

A football **match intelligence engine** with a broadcast-studio style web UI. Given player and ball positions plus match events, it:

1. **Interprets** each minute: possession, territory, pressing intensity, defensive line, width, compactness, and team shape (standard formations such as 4-2-3-1, read from the back line forward).
2. **Detects tactics:** high press, counterattacks, attacking/defensive transitions, tactical shifts, formation changes, low block / dominance chapters.
3. **Finds key moments** (goals, pressure breaks, tactical turning points, shape changes, substitution and red-card impact). Each has an evidence chain (trigger → state change → tactical change → consequence → outcome), importance, confidence and coverage.
4. **Builds a match story:** a graph of moments linked with deliberately non-causal relations ("preceded", "followed_during").
5. **Explains** each moment for three audiences (analyst / fan / player) using only numbers produced by the engine. An LLM backend is stubbed for Azure / Microsoft Foundry.
6. **Reconstructs hidden players.** A broadcast camera shows only about a quarter of the players, so the engine estimates off-screen positions with confidence and an uncertainty radius. This works in two modes:
   - **live:** causal, past only.
   - **replay:** retrospective, also uses the next sighting.
7. **Serves it all** through a FastAPI backend and a Next.js UI:
   - a **lobby** of live and past matches;
   - a **match page** with whole-match highlights and analysis;
   - a **studio** with pitch replay, overlays, the match-story timeline and personalised panels.

### Data the app runs on

| Kind | Source | In the lobby | Ground truth? |
|---|---|---|---|
| Real tracking | DFL / IDSSE open data: 7 Bundesliga / 2. Bundesliga matches, 25 Hz TRACAB, all 22 players + ball in 3D, plus the official event feed. CC BY 4.0. | All 7 matches, both halves. One (Düsseldorf vs Rostock) is replayed in real time as the **live** match. | Yes |
| Computer vision from broadcast video | SoccerNet (research / NDA licence): a 10-minute clip of Arsenal vs Liverpool 2016, processed by our CV pipeline | 1 clip | No |
| Synthetic | Our scripted simulator (`simulator/`) | 1 test match | Yes (exact) |
| Camera behaviour | SkillCorner open data (MIT): real broadcast camera footprints and detection rates | Used to mask DFL truth realistically | n/a |

When a match has ground truth, the **truth is the default view and all analysis runs on it**. The camera-only estimate ("Camera replay" / "Camera live") is kept for comparison, and for future matches that only have video.

---

## 2. How to run

```bash
./run.sh                                   # API :8000 + UI :3000 (http://localhost:3000 = lobby)
.venv/bin/python -m pytest -q tests        # 18 tests, all passing at snapshot
```

Environment variables:
- `LIVE_MATCH` (default `dfl-J03WOY`): which match is replayed as live.
- `LIVE_STARTED_MIN` (default `20`): how far into it the live clock starts.
- `VIEW=vod`: analyse the camera estimate instead of truth.
- `RECON_MODEL=ml`: use the learned estimator on the synthetic match.
- `NARRATIVE_BACKEND=foundry`: LLM stub.

Stack:
- **Backend:** Python 3.11 (uv venv): FastAPI, NumPy/SciPy, PyTorch (MPS).
- **CV:** ultralytics YOLO11 + BoT-SORT, OpenCV.
- **Frontend:** Next.js 16 + React 19 + Tailwind 4. This Next version has breaking changes; read `frontend/AGENTS.md`.
- **Codebase:** about 7.6k lines of Python and about 2k of TypeScript.

The repo is **not a git repository**. Large data (`data/dfl`, `data/skillcorner`, `data/sn_*`, `cv/weights`, `cv/out`) is git-ignored.

---

## 3. Architecture and file map

Pipeline: `INGEST → (OBSERVE → RECONSTRUCT) → INTERPRET → EXPLAIN → RENDER → PERSONALIZE`

| Stage | Files | Notes |
|---|---|---|
| Data contract | `backend/models/match.py`, `schemas.py` | One JSON contract `{meta, players, events, frames}` for simulator, CV and DFL files. 1 Hz frames, metres, 105×68 pitch. |
| Simulator | `simulator/match_generator.py`, `scenarios.py` | Deterministic and seeded. Scripted tactical phases (press, counter, sub, shift, red card) give exact ground truth. Offside law enforced, including scripted moves (fixed 2026-10-08). |
| Real data | `backend/reconstruction/datasets.py` (DFL parser, ball height, SkillCorner cameras), `dfl_match.py` (DFL half → app file: events, dead-ball / restart labels, goal-area logic, camera estimate) | Event clock aligned per half using on-camera players only. |
| Camera / observation | `backend/reconstruction/observation.py` (synthetic camera), `real_benchmark.py` (`mask_sc`: real SkillCorner footprints moved onto the DFL ball) | |
| Hidden-player estimator | `estimator.py` (team-shape model: role anchors, block fit, keeper model, live forward pass, replay backward pass + inverse-variance fusion + reachability), `ml.py` (MLP correction + calibrated σ, trained on DFL), `inference.py` (`reconstruct_ml`, `predict_hidden_state` API), `postprocess.py` (event pins, shape context, speed limit / no-teleport) | |
| Analytics / tactics | `backend/analytics/*` (shape incl. template-fit formations and per-second line assignment), `backend/tactics/*` | |
| Moments / story | `backend/moments/detector.py`, `story.py` | |
| Narratives | `backend/narratives/generator.py` | Template-based. Test `test_narratives_use_only_engine_numbers` rejects invented numbers. |
| What-if | `backend/counterfactual/engine.py` | Re-simulates the synthetic match without a scripted event. Synthetic only. |
| Orchestration | `backend/pipeline.py` (`MatchIntelligence`, `summary()`), `backend/catalog.py` (lobby, live clock, whole-match summary across halves) | |
| API | `backend/api/main.py` | `/matches`, `/matches/{id}`. Every `/match/...` endpoint takes `?match=&part=`. |
| CV pipeline | `cv/detect_track.py` (YOLO fine-tuned on SoccerNet-GSR + BoT-SORT), `build_match.py` (team colours, identity by role, ball path from touches, rebuild with football logic), `roles.py`, `gsr*.py`, `render_check.py` | Works on one clip; see weaknesses. |
| Evaluation | `backend/reconstruction/evaluate.py`, `multi_eval.py` (100 synthetic matches), `ml.py --fold` (real DFL folds), `dfl_match.py --tune` | |
| UI | `frontend/app/page.tsx` (lobby), `app/match/[id]/page.tsx` (highlights + analysis), `app/match/[id]/studio/page.tsx` → `components/studio/Studio.tsx`; `components/pitch/Pitch.tsx`, `timeline/MatchStory.tsx`, `insights/*`, `personalization/PlayerPanel.tsx` | |
| Tests | `tests/test_engine.py`, `test_reconstruction.py`, `test_ml_leakage.py` | |
| Docs | `README.md`, `docs/NEXT_STEPS.md`, `docs/HIDDEN_STATE_RECONSTRUCTION.md`, `docs/COMPUTER_VISION_PLAN.md` | |

---

## 4. Progress checklist

Legend: ✅ done and verified · 🟡 partial / works with caveats · ❌ not done

### Core engine
- ✅ Per-minute tactical state and press index. Formation detection is a template fit, smoothed over time with a switch penalty (Viterbi). Against the official DFL line-ups: exact 5/14, back line 13/14; 1.1 changes per team-half.
- 🟡 Detectors for high press, counterattack, transitions and tactical shift.
  - Synthetic: §5.5.
  - Real: counterattack checked against DFL's labels; high press unvalidated and not firing (§5.4).
- ✅ Moments with evidence chain, coverage and basis tags; non-causal story graph.
- ✅ Analyst / fan / player narratives; the test enforces "no invented numbers".
- 🟡 Wording is "the sequence suggests X drove Y". This is hedged but still causal-sounding; a reviewer may want it softer.
- 🟡 LLM narratives: `FoundryNarrativeGenerator` calls Azure OpenAI / Foundry. Set `NARRATIVE_BACKEND=foundry`, `AZURE_AI_ENDPOINT`, `AZURE_AI_KEY`. Any reply with a number not in the evidence falls back to the template. Not run against a live endpoint here.
- 🟡 What-if counterfactuals: synthetic match only.

### Hidden-player reconstruction
- ✅ Live (causal) vs replay (retrospective) with a leakage test.
- ✅ Calibrated uncertainty: the 80% radius covers 80–81% on real held-out matches.
- ✅ Learned correction trained on DFL with match-level splits: leave-one-match-out over all 7 matches (§5.1). The app uses each match's held-out model.
- ✅ No teleporting: replay steps over 10 m/s went from about 750 per half to 0.
- 🟡 Live mode still jumps when a player reappears, because it can't use the future.
- ✅ Prediction → reality is measured on 21,775 real reappearances (§5.2) and shown in the studio (**Camera vs reality**).
- ❌ Shape-context blending ("player's usual spot per shape and phase") was implemented and tested, but **made results worse**, so it is disabled (weight 0). The v2 slot-memory features did help.
- ❌ Event pins are about 5–6 m off at 1 Hz, so they're only used softly, with marginal gain.

### Real-data presentation (DFL)
- ✅ Truth view; dead-ball labels (throw-in, corner, goal kick, free kick, kick-off) with the ball placed at the restart spot.
- ✅ Goal kept in the net until kick-off; over-the-bar detection uses ball height.
- ✅ Shape lines drawn per second, dashed where an end is estimated.
- 🟡 "Dead ball" without a known restart type: about 18 s per half remain unlabelled.

### Computer vision (broadcast → positions)
- ✅ Detection + tracking with a fine-tuned detector. On SoccerNet-GSR held-out: precision 0.872→0.913, recall 0.953→0.972, ball AP 0.67→0.716.
- ✅ Pitch calibration from SoccerNet homographies, team colours, lineups from labels, ball path rebuilt from touches.
- 🟡 Runs on **one 10-minute clip** only.
- ❌ **Player identity:** shirt numbers are not read. Identity accuracy was about 35% in our role-based evaluation. With oracle identities, off-screen error falls from about 12.9 m to 6.8 m (`cv/eval_roles.py`). This is the biggest gap.
- ❌ CV confidence is not calibrated end to end against labelled video.
- 🟡 The CV clip was not rebuilt after the latest speed-limit change.

### Product / UI
- ✅ Lobby (live + past), match page (highlights with filters, team stats, shapes, chapters, top players, cards / subs), studio with deep links (`?part=&moment=`).
- ✅ Live mode follows the clock edge; only content up to the edge is shown.
- 🟡 "Live" is a recorded match replayed in real time and labelled as such. There is no real live data ingestion (the Event Hubs stand-in is an SSE replay at `/match/stream`).
- 🟡 Moment detection uses the whole match, including windows that end after the live edge. A true live system would detect moments later than our replay shows them.
- ❌ No auth, no deployment, no persistence beyond files; not responsive for mobile.

### Engineering
- ✅ 20 tests pass. They cover engine contracts, reconstruction beating hold-last, observed positions exact, live never using the future (v1 and v2 features), and inferred positions carrying uncertainty. Also covered: schemas, story links, synthetic counterfactual labels, reveal error integrity, and the LLM number guard.
- ❌ No git history, no CI, no frontend tests.
- 🟡 Startup builds analysis in-process (synthetic about 20–30 s; each DFL half a few seconds on first request).

---

## 5. Measured results

Reproduce:
- `ml.py --lomo --all --feat 2` then `lomo_report --compare` (hidden players);
- `reveal_report` (prediction → reality);
- `backend.tactics.real_eval` (real tactics);
- `multi_eval --matches 100` (synthetic).

Outputs are in `data/eval/` and `data/eval/real/`.

### 5.1 Hidden players on real data: 7-fold over the 7 DFL matches (train on 6, test on the 7th)

Recipe: train on the 6 other matches. The last part of every training half (t ≥ 2200 s, 60 s gap) picks the epoch count and calibrates uncertainty. The model is then refit on all rows of the 6 matches. The camera mask is a real SkillCorner broadcast footprint. Values are mean error on hidden players over the 7 test matches (`ml.py --k7 --all --feat 3 --refit --tag k7r`).

| Recipe (real camera) | Live | Replay | 80% radius coverage (live / replay) |
|---|---|---|---|
| Team-shape estimator | 14.2 | 10.3 | – |
| Leave-one-match-out v2 (5 train + validation match) | 8.29 | 6.92 | 0.80 / 0.80 |
| 7-fold v2, no refit | 8.32 ± 0.28 | 7.01 ± 0.29 | 0.81 / 0.81 |
| 7-fold v2 + refit | 8.20 ± 0.22 | 6.82 ± 0.27 | 0.81 / 0.83 |
| **7-fold v3 + refit (current)** | **8.08 ± 0.27** | **6.74 ± 0.25** | 0.81 / 0.82 |

- Refit beat no refit on 6/7 matches (live) and 7/7 (replay).
- v3 (adds dead-ball / set-piece context) beat v2 on 7/7 in both modes.
- The weakest situations are dead ball (about 46% of hidden player-seconds) and wide roles (wingers, wide midfielders, full-backs).
- Each app match uses the fold model that never saw it.
- The production model `final_v3` is trained on all 7 matches with the same recipe; its accuracy is the 7-fold estimate.

### 5.2 Prediction → reality (real, all 14 halves, held-out 7-fold models)

| Measure | Result |
|---|---|
| Reappearances | 21,775 |
| Median / mean error | **7.78 m** / 9.36 m |
| Within 5 m | 29.6% |
| Within 10 m | 63.5% |
| Inside the 80% radius | **79.6%** |

Confidence falls with time hidden, and higher confidence means lower error.

### 5.2b Broadcast video (SoccerNet Arsenal–Liverpool clip, 344 reappearances after ≥ 5 s off screen; `cv/clip_eval.py`)

| Method | Median error | Mean error |
|---|---|---|
| Last seen | 23.6 m | 25.6 m |
| Team shape | 17.8 m | 21.8 m |
| Learned f0 | 16.6 m | 20.0 m |
| Learned final_v3 | 18.8 m | 20.8 m |

- Learned final_v3 vs f0: mean difference +0.78 m, 95% CI [−0.05, 1.63]. Against team shape: −1.04 m, 95% CI [−2.37, 0.23]. **No measurable difference between methods on video:** CV detection and role-based identity (about 35% exact) dominate the error.
- The tracking-calibrated radius covers only about 40% on video, so the video path now scales it by 2.67. That scale was fitted on the first half of the clip; on the second half coverage is 0.92.

### 5.3 Does reconstruction change the tactical reading? (real, 125 key moments, held-out 7-fold models)

For the moment's team, ±5 s: camera-only vs reconstructed vs truth. On average the camera showed 2.2 of 10 outfield players.

| Metric | Camera-only error | Reconstructed error |
|---|---|---|
| Defensive line | 11.6 m (computable in 72/125 moments) | **3.2 m** (125/125) |
| Team width | 15.6 m | **6.8 m** |
| Players within 15 m of ball | 0.46 | 0.36 (on camera anyway) |

### 5.4 Tactics on REAL matches (`data/eval/real/tactics_real.json`)

| Check | Label source | Result |
|---|---|---|
| Counterattack shots | DFL's own `CounterAttack` flag on 167 shots (14 positive) | P 0.75 · R 0.43 · F1 0.55. Old detector P 0.64 · R 0.50 · F1 0.56. Designed on 5 dev matches; on the 2 held-out matches F1 is 0.40 (old 0.55). **Not an improvement on real data.** |
| Formation | Official DFL line-up, first 15 min | exact 5/14, back line 13/14 |
| Formation stability | n/a | changes per team-half **6.9 → 1.1**; SHAPE CHANGE moments over 14 halves **55 → 13** |
| High press | none (weak association only) | **0 real minutes flagged.** Thresholds tuned on the simulator do not transfer: the real press-index p97 is 57, against 92 in the simulator. Unvalidated on real football. |

### 5.5 Synthetic (100 matches, re-run on the 2026-10-09 simulator)

The simulator was rewritten to play more like football (see 5.6). The benchmark got harder:

| Visible | Replay error before | Replay error now |
|---|---|---|
| 18 | 3.07 m | 3.33 m |
| 14 | 3.47 m | 4.25 m |
| 10 | 3.99 m | 4.87 m |
| 6 | 5.41 m | 7.29 m |
| Broadcast camera | 3.90 m | 4.10 m |

The replay estimate still beats freezing players in 100% of matches.

Tactic F1 (full truth), before → now:

| Tactic | Before | Now |
|---|---|---|
| High press | 0.757 | **0.895** (out-of-possession line threshold 48 m, chosen on dev seeds 100–111) |
| Tactical shift | 0.840 | 0.720 |
| Counterattack | 0.386 | 0.371 |
| Attacking transition | 0.72 | 0.79 |
| Defensive transition | 0.73 | 0.81 |

### 5.6 Simulator realism vs real DFL (`simulator/realism.py`)

| Measure | Before | Now | Real |
|---|---|---|---|
| Backward-pass share | 58% | 23% | 24% |
| Final third, backwards | 80% | 34% | 27% |
| Through on goal → pass back | 63% | 11% | 11% |
| Passes per minute | 16.3 | 10.5 | 9.1 |
| Shots per 90 | 27 | 30 | 23 |
| Pass success | 54% | 67% | 76% |
| Median pass length | 20 m | 18 m | 12.5 m |

Changes:
- Value-based decisions: shoot, pass or carry, judged on xG, lane interceptions and threat.
- Defenders stay goal-side and protect the box.
- Two support runs towards the carrier.
- When clean through, never pass back.

---|---|---|
| Replay hidden error (18 / 14 / 10 / 6 visible / broadcast) | 3.07 / 3.48 / 3.99 / 5.42 / 3.90 | 3.07 / 3.47 / 3.99 / 5.41 / 3.90 |
| High press F1 (full truth) | 0.729 | **0.757** (precision up: minimum 3 minutes + must target the opponent's deep build-up) |
| Counterattack F1 | 0.304 | **0.386** (staged detector: recovery → disorganised defence → fast progression → entry) |
| Tactical shift F1 | 0.838 | 0.840 |
| Transitions F1 | 0.72 / 0.73 | 0.72 / 0.73 |

---

## 6. Known weaknesses (ranked)

1. **High press is unvalidated on real football** and currently never fires there; there are no real press labels to recalibrate against.
2. **Counterattack on real data** is about F1 0.55, with only 14 labelled positives. The redesign helped synthetic, not real.
3. **Uncertainty is slightly overconfident at reappearance moments** (77.8% inside the 80% radius; 74% after 30 s+ hidden).
4. **CV identity is unchanged** (shirt numbers not read; about 35% identity accuracy). Phase skipped this session.
5. **1 Hz resolution is unchanged.** The 5 Hz pipeline was not attempted.
6. **Live is a recorded match replayed in real time.** Moments and reveals use full-half context: the reveal summary counts the whole half, and moment windows extend past the edge.
7. **Foundry narratives** are wired (env vars, number guard, fallback) but untested against a real endpoint: there are no credentials here.

---

## 7. Candidate improvements

| # | Improvement | Why |
|---|---|---|
| A | Hand-label high-press phases on 2–3 DFL halves, then recalibrate the press index on real data | High press is the only headline tactic with no real validation |
| B | Recalibrate the radius per time-hidden bucket on validation matches | Reveal coverage 77.8% → 80% |
| C | Temporal model (GRU over the last N seconds of team state), compared under leave-one-match-out | v2 features showed temporal context helps |
| D | Shirt-number recognition (SoccerNet Jersey, PARSeq, track voting) | Biggest CV gap |
| E | 5 Hz reconstruction | Event alignment, pins, ball flight |
| F | True causal moment detection for live | Honest live latency |

---

## 8. Rules any change should respect

- **Never claim accuracy that wasn't measured.** Every number in UI or docs must come from a run that can be reproduced in this repo.
- Keep **live vs replay** separation: live must never use future data (`tests/test_reconstruction.py`).
- Keep **match-level splits** for anything trained on DFL; flag matches the model trained on.
- Narratives may only use engine-computed numbers (`test_narratives_use_only_engine_numbers`). Wording stays non-causal ("followed", "suggests").
- **Licences:**
  - DFL: CC BY 4.0, credit Bassek et al. 2025.
  - SkillCorner: MIT.
  - SoccerNet: research / NDA; don't show raw broadcast frames publicly.
  - No league or club logos. Don't redistribute footage.
- Estimated positions must stay visibly different from observed ones in the UI (dashed, confidence, ± radius).

---

## 9. Open questions for the reviewer

1. Should the product focus on **real tracking data** (DFL-style providers) and treat video CV as a later add-on, or is broadcast-only CV the core goal?
2. Is it worth hand-labelling real tactical phases (A) before any further detector work?
3. Should the synthetic match stay visible in the lobby at all?
4. Which audience matters most for the demo: analysts (evidence, accuracy) or fans (highlights, narratives)?
