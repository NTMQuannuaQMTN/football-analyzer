# What to do next

A practical plan for the team, in priority order. Every number below comes from runs in this repo (`data/eval/`).

## Where we are

- **Synthetic demo works end to end.** It's a simulated match seen through a simulated broadcast camera: hidden players are reconstructed, then tactics, moments, the story, explanations and the Analyst / Fan / Player / Accuracy views follow.
- **Reconstruction is proven against ground truth.**
  - Synthetic, 100 matches: replay error 3.1–5.4 m (18 down to 6 players visible).
  - Real Bundesliga tracking, teams unseen in training: the learned model reaches **7.0 m** (replay) under a real broadcast camera, against 10.0–10.4 m for the team-shape estimator and 15.5–16.0 m for last position.
- **Real video pipeline runs** on a SoccerNet clip (Arsenal 3–4 Liverpool). Two weak links remain: **who is who** (shirt numbers aren't read) and the stock detector's ball and referee classes.

## 1. Today: make the demo bulletproof (≈1 hour)

```bash
./run.sh                                   # lobby at localhost:3000: live (synthetic) + past matches (DFL, CV clip)
MATCH_FILE=data/cv_2016_08_14_18_00_arsenal_3_4_liverpool_h1_1440_2040.json ./run.sh   # sets the default match for un-scoped API calls
.venv/bin/python -m pytest -q tests        # 9 tests must pass
```

Checklist before presenting:
- [ ] Start once and wait about 30 s; the backend builds everything at startup.
- [ ] Click **SHAPE 66'** in Match Story: the replay plays and the evidence panel shows coverage ("7/11 observed").
- [ ] Turn on **Error**: yellow squares show where inferred players really were.
- [ ] Click a dashed (inferred) player: the card shows confidence, ±radius, last / next seen and evidence.
- [ ] Open the **Accuracy** tab: synthetic, 100-match and real-football tables load.
- [ ] Close other heavy apps. The Mac has 24 GB and the backend plus Next dev server use several GB.

### Test on real top-down matches (DFL tracking)
Real Bundesliga matches, every player tracked by in-stadium cameras (TRACAB). They're shown through a real broadcast camera footprint (SkillCorner), so about 75% of players are hidden and estimated.
- **Truth:** the real position of every player.
- **Error:** shows each estimate against the real position.
- **Fair test:** the estimator never trained on these matches.

| File | Match | Half | Score at end of file | Hidden-player error (replay / live) |
|---|---|---|---|---|
| `data/dfl_J03WMX_h1.json` | 1. FC Köln vs FC Bayern München | 1 | 0–1 | 7.0 m / 8.1 m |
| `data/dfl_J03WMX_h2.json` | 1. FC Köln vs FC Bayern München | 2 | 1–2 | 7.4 m / 8.9 m |
| `data/dfl_J03WN1_h1.json` | VfL Bochum vs Bayer Leverkusen (Adli red card 8') | 1 | 2–0 | 7.1 m / 8.4 m |
| `data/dfl_J03WN1_h2.json` | VfL Bochum vs Bayer Leverkusen | 2 | 3–0 | 7.0 m / 8.8 m |

```bash
./run.sh   # then in the lobby: 1. FC Köln vs FC Bayern München (Coman's goal at 7:48)
.venv/bin/python -m backend.reconstruction.dfl_match --match DFL-MAT-J03WOH --half 1   # other matches (training matches: flagged in meta)
```
**Views when a match has ground truth (the synthetic match and the DFL files):**
- **Truth** is the default. All analysis (tactics, shapes, moments, stories) runs on the real positions.
- **Camera replay / Camera live** show what a broadcast camera would give, with hidden players estimated. They're for comparison, and for future matches that have no ground truth (video only). Turn on **Error** to see each estimate against the real position.
- How hidden players are estimated: their last and next sightings (where and when they were seen again) plus the team shape (formation slot, learned correction).

**Dead ball:** for a throw-in, free kick, goal kick or corner, the ball goes to where play restarts and shows "THROW-IN · NOT TAKEN YET" until play resumes. Events come from the official DFL feed: passes, shots with xG, tackles, cards and subs.

**How a hidden player is estimated** (`backend/reconstruction/dfl_match.py`, `postprocess.py`). Each choice was made on the validation match, never the test match (`--tune`).
1. **Camera sightings** are exact and never moved.
2. **Team shape prior:** each player's slot in the current block, adapted from when they were seen.
3. **Correction model:** trained on DFL tracking; replay also uses the next sighting.
4. **Event pins:** the player named in the event feed is pulled halfway to the ball.
   - The event clock is aligned per half using on-camera players only.
   - Pins are still about 5–6 m off with one sample per second, so they're soft. Hard anchors were worse.
5. **Speed limit (8.5 m/s):** in replay, steps over 10 m/s went from about 750 per half to 0, and error improved by about 0.2 m.
   - Live still jumps when a player comes back on camera: it can't use the future, so it snaps to the truth.
6. **Tried and not used:** blending in "the player's usual spot for this shape and phase". It made replay worse in both folds; the correction model already uses the shape.

**Ball:** when play stops, the ball stays where it went out, then moves to the restart spot. After a goal it stays in the net until the kick-off.

**Shapes** are read from the back line forward, goalkeeper excluded: 4-3-3 means 4 defenders, 3 midfielders, 3 forwards.
- The team's average positions over 5 minutes are fitted to standard formations (with 10, 9 or 8 outfielders after red cards).
- On true DFL positions the fit matched the official line-up exactly for 7 of 13 teams, with the right back-line count for 12 of 13.
- Real teams move between neighbouring shapes (4-2-3-1 ↔ 4-1-4-1) during a match.

## 2. Demo script (5 minutes)

1. **Live match (30 s):** play at 8x. Point at "17/22 observed · 5 inferred": *"the camera never shows everyone."*
2. **Hidden state (60 s):** pause on a camera cut ("0/22 observed"): every player is estimated, with growing uncertainty circles. Switch **Camera live / Camera replay**: live uses only the past; replay also uses the next sighting and says so. **Truth** (the default) shows the real positions.
3. **Proof (60 s):** turn on **Error**, then open **Accuracy**. One sentence: *"On real Bundesliga matches with teams it never saw, we cut hidden-player error by a third versus the formation estimator, and our 80% uncertainty circles contain the true position 80% of the time."*
4. **Tactical shift → story → why (60 s):** click **SHAPE 66'**. Read out the evidence chain and the "sequence suggests" wording; we don't over-claim causation.
5. **What if + personalisation (45 s):** expand What If (labelled synthetic counterfactual), then switch Fan and Player mode.
6. **Real footage (45 s):** restart with `MATCH_FILE=…`. Show the CV badge and the goal credited to Walcott, and note *"Insufficient visibility"*: the system abstains when it can't see enough.

Say the limitations out loud (see section 5). Judges reward honesty about what's measured.

## 3. Highest-value engineering work (in order)

### 3.1 Read shirt numbers, to fix identity in real video
Why first: with correct identities, off-screen error drops from about 12.9 m to 6.8 m (`cv/eval_roles.py`, oracle mode). It's the biggest gap between the real-VOD path and the benchmark.
- **Data:** SoccerNet Jersey Number Recognition (2023), plus the legibility labels from Koshkina et al. (2024).
- **Model:** a legibility classifier first (is a number readable?), then fine-tune PARSeq on legible torso crops only. Vote per track and only commit when several confident readings agree.
- **Plug-in point:** pass the read identity as a detection hint (`(slot_id, xy, role_index)`); `RoleEstimator` already supports this (`cv/roles.py`).
- **Done when:** identity accuracy on SoccerNet-GSR held-out clips is ≥ 70% (today about 35%).
- **Compute:** use an Azure ML GPU; this laptop swaps during training.

### 3.2 More real ground truth and more folds
- **Leave-one-match-out on all 7 DFL matches:** run `.venv/bin/python -m backend.reconstruction.ml --fold N` after extending `split()` beyond 2 folds.
- **Add SoccerTrack v2** (CC BY 4.0, 10 university matches, full pitch) as extra training and test data in the same row format (`backend/reconstruction/datasets.py`).
- **Done when:** results are reported as mean ± spread over ≥ 5 test matches.

### 3.3 Close the synthetic ↔ real gap
The real-trained model is slightly worse on the simulator (domain gap). Two options:
- Fine-tune on mixed synthetic and real samples, with a domain flag feature.
- Fit the simulator's movement (speeds, pressing distances, line heights) to DFL statistics.

**Done when:** one estimator wins on both benchmarks, so `RECON_MODEL=ml` can become the default.

### 3.4 Calibrate the real-video path end to end
Run the full CV pipeline on SoccerNet-GSR validation clips and score positions, identities and teams against their labels, so real-match confidence values are calibrated rather than synthetic-calibrated.

## 4. Microsoft / Azure (when the above is stable)

| Need | Service | Where it plugs in |
|---|---|---|
| LLM narratives that only rephrase evidence | Microsoft Foundry | `FoundryNarrativeGenerator` in `backend/narratives/generator.py`; enable with `NARRATIVE_BACKEND=foundry`. Keep the test `test_narratives_use_only_engine_numbers` passing; it rejects invented numbers. |
| GPU training (detector, jersey numbers, reconstruction folds) | Azure Machine Learning | `cv/gsr_train.py`, `backend/reconstruction/ml.py` |
| Hosting the API and UI | Azure Container Apps | `backend/api/main.py` (stateless; builds at startup), `frontend` (`next build`) |
| Large data and weights (about 5 GB) | Azure Blob Storage | `data/dfl`, `data/skillcorner`, `cv/weights` (all git-ignored) |
| Live event feed | Azure Event Hubs | replaces the SSE stand-in `/match/stream` |
| CI | GitHub Actions | run `pytest`, `tsc`, `eslint` on every push |

## 5. Licences and honesty checklist for a public demo

- [ ] **Credit DFL / IDSSE** (CC BY 4.0) on the Accuracy slide: *Bassek et al., Sci Data 2025*.
- [ ] **Credit SkillCorner** (MIT) for the camera footprints.
- [ ] **SoccerNet broadcast video is under an NDA (research / non-commercial).** Don't show raw broadcast frames publicly. The minimap reconstruction from them is fine to show; check the NDA wording before any public release.
- [ ] **Don't use Premier League or club logos.** The synthetic teams (Ashford City, Belmont Rovers) are fictional on purpose.
- [ ] **Say it plainly:**
  - Off-screen positions are *estimates*, with stated confidence.
  - Real-data accuracy is measured on 2 test matches.
  - Real-match identities are approximate until shirt numbers are read.
  - What-if results are synthetic.

## 6. Command cheat-sheet

```bash
# reconstruction evaluation (synthetic): one match / 100 matches
.venv/bin/python -m backend.reconstruction.evaluate
.venv/bin/python -m backend.reconstruction.multi_eval --matches 100 --workers 8

# real-data model (DFL + SkillCorner masks); results in data/eval/real/
.venv/bin/python -m backend.reconstruction.ml --fold 0

# serve variants
VIEW=vod ./run.sh              # analyse the camera estimate instead of ground truth (default: truth)
RECON_MODEL=ml ./run.sh        # learned estimator on the synthetic demo

# real video pipeline (SoccerNet)
.venv/bin/python -m cv.detect_track --start 1440 --end 2040
.venv/bin/python -m cv.build_match
.venv/bin/python -m cv.render_check --start 1800 --end 1815   # overlay check vs video
```
