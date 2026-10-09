# Computer Vision Track — From Match Video to the Insights Engine

**Goal:** replace the synthetic simulator with tracking + event data extracted from real match video, **without changing anything downstream**. The engine only needs the same raw dict the simulator produces:

```text
{ meta, players, events, frames }   # frames: 1 Hz (or faster) player + ball positions in metres, 105 x 68 pitch
```

`backend/models/match.py::MatchData` is the single ingest point. If CV output matches this contract, then tactical detection, moments, the story graph, the narratives and the UI all work unchanged.

> Note: the hackathon brief is about *synthetic* events, so CV is an extension, not a requirement. Build it only once the synthetic demo is solid, and present it as "the same engine running on real footage".

---

## 0. How to run it (SoccerNet, implemented in `cv/`)

```bash
uv pip install --python .venv/bin/python -r requirements-cv.txt
.venv/bin/hf auth login                                   # account with the SoccerNet NDA accepted
.venv/bin/python -m cv.soccernet --half 1                 # Arsenal 3-4 Liverpool (2016-17): labels, calibration, 720p video
.venv/bin/python -m cv.detect_track --start 1440 --end 2040 --show   # 24:00-34:00, YOLO11 + ByteTrack on MPS (~5 min)
.venv/bin/python -m cv.build_match                        # check printed team colours; flip with --a-team away
MATCH_FILE=data/cv_<slug>.json ./run.sh                   # same studio UI, now on real footage
```

- **Calibration:** we don't click anything. `1_field_calib_ccbv.json` provides a pitch→image homography every 0.5 s, and we invert it to put each player's feet in pitch metres.
- **Shot filtering:** a frame counts as live only if it has enough grass, a confident homography, an image centre on the pitch and small players. `Labels-cameras.json` is used only to skip replays; its close-up labels are unreliable.
- **Events:** passes, possession changes and pressure are inferred from who owns the ball. Shots, goals, offsides and set pieces come from `Labels-v2.json`.
- `python -m cv.soccernet --list` shows all 104 EPL games.
- **Off-screen players:** `python -m cv.offscreen_model` trains an MLP on Metrica open tracking data (`data/metrica/`), with broadcast camera views simulated on top. It reports error on a held-out match: 7.5 m, against 9.7 m for "freeze at last seen" (`cv/weights/offscreen_metrics.json`).
- **Identity:** players take their real shirt numbers and names from the official lineup (`Labels-caption.json`), including substitutions. Each second, visible detections are matched to lineup roles; anyone unmatched is estimated by the model and drawn faded and dashed in the UI.
- **End-to-end check:** `python -m cv.eval_roles` measures team-shape error (identity-free, about 7 m) and identity accuracy (about 35% after camera cuts). Identity is the weak link; the fix is shirt-number recognition, using the SoccerNet Jersey Number dataset with a scene-text model such as PARSeq.

## 0b. Training on SoccerNet-GSR (video + ground-truth minimap)

The data is `SoccerNet/SN-GSR-2025` on Hugging Face (public, no NDA): 200 broadcast clips of 30 s, with every person's box, pitch position, role, team and jersey number. Validation clips 1–12 are held out for evaluation only.

```bash
.venv/bin/python -m cv.gsr eval --clips 12 --stride 10 --model <weights>   # score on held-out clips
.venv/bin/python -m cv.gsr_train export-yolo && .venv/bin/python -m cv.gsr_train train-yolo --epochs 5
.venv/bin/python -m cv.gsr_train export-jersey && .venv/bin/python -m cv.gsr_train train-jersey
```

**Detector**, held-out clips, scored with "oracle" calibration (the per-frame mapping is fitted to the labelled players, so calibration error is excluded):

| | Stock YOLO11m (COCO) | Fine-tuned YOLO11s (`cv/weights/yolo_gsr.pt`) |
|---|---|---|
| People: precision / recall | 87.2% / 95.3% | **91.3% / 97.2%** |
| Position error | 0.36 m | **0.33 m** |
| Ball found (within 20 px) | 67.0% | **71.6%** |
| Team (colour clustering) | 77.9% | 77.9% |

The fine-tuned model also separates **referees** (dropped before team clustering) and **goalkeepers**. Training took 1.7 h on an M5 laptop: YOLO11s, 1280 px, half the frames. On an Azure ML GPU, use YOLO11m, all frames, batch 16.

**Shirt numbers** have not been solved yet:

| Approach | Right number per track, new matches |
|---|---|
| ResNet18, 99-way classifier | 7% (memorised the training players) |
| Digit heads + anti-shortcut augmentation | 12% |
| PARSeq scene-text reader, no fine-tuning | 1 of 20 crops |

Most broadcast crops show no legible number. The next step is a **legibility classifier** trained on Koshkina et al. 2024 legibility labels, then fine-tuning PARSeq on legible torso crops only. With identities known, off-screen error drops from 12.9 m to 6.8 m (`cv/eval_roles.py`, oracle mode).

**Video check:** `python -m cv.render_check --start 1800 --end 1815` draws our ball and players back onto the broadcast frames next to the minimap (`cv/out/check/`).

## 1. Pipeline

```text
VIDEO ──► 1 DETECT ──► 2 TRACK ──► 3 IDENTIFY TEAMS ──► 4 CALIBRATE PITCH ──► 5 PROJECT TO METRES
                                                                                    │
      MatchData ◄── 8 EXPORT CONTRACT ◄── 7 DETECT EVENTS ◄── 6 SMOOTH / FILL GAPS ◄┘
```

| Step | What it does | Starting point (don't train from scratch) |
|---|---|---|
| 1 Detect | Boxes for players, goalkeepers, referees, ball | YOLO (v8/11) or RT-DETR, fine-tuned on football data |
| 2 Track | Keep a stable ID per player across frames | ByteTrack / BoT-SORT (both ship with Ultralytics) |
| 3 Teams | Assign A / B / referee | K-means on jersey colour crops, or SigLIP/CLIP embeddings + clustering |
| 4 Calibrate | Map image to pitch plane, per frame | Pitch keypoint model (line intersections, centre circle) → homography |
| 5 Project | Box bottom-centre → (x, y) metres | `cv2.perspectiveTransform` with the step 4 homography |
| 6 Smooth | Remove jitter, fill short occlusions, cap speed at ~11 m/s | Kalman filter or Savitzky–Golay; interpolate gaps under 2 s |
| 7 Events | PASS / SHOT / TACKLE / POSSESSION_CHANGE | Rule-based from ball ownership first (nearest player within ~1.5 m); add an action-spotting model later |
| 8 Export | Write the `{meta, players, events, frames}` JSON | New `ingest/cv_adapter.py` |

Hardest parts, in order: **ball tracking** (small, blurred, occluded), **pitch calibration on broadcast camera cuts**, **re-identifying players** after they leave the frame. Broadcast video only shows part of the pitch, so expect partial tracking. Make every downstream metric tolerate `null` positions; the MatchData arrays already use NaN.

---

## 2. Data (use licensed / open sources only)

Premier League broadcast footage is copyrighted. Do **not** train on scraped match video. Use:

- **SoccerNet**: tracking, camera calibration, action spotting, and *Game State Reconstruction* (close to exactly this task). Academic licence; requires an NDA for the videos.
- **Metrica Sports sample data**: open tracking + event data (no video). Ideal for testing steps 6–8 and the engine itself.
- **SkillCorner open data**: broadcast-derived tracking for full matches.
- **StatsBomb open data**: events + 360 freeze frames, useful for checking event detection.
- **Roboflow Universe football datasets**: small labelled sets for player / ball / pitch-keypoint detection to fine-tune on.

---

## 3. Suggested plan

**Phase 0 — Prove the contract (½ day, no CV).**
Write `ingest/metrica_adapter.py` that converts Metrica open tracking + events into `MatchData`. Run the existing engine on it. This shows the engine works on real movement, and it gives you a reference for what the CV output has to look like.

**Phase 1 — Positions from video (2–3 days).**
Steps 1–5 on a single *tactical/wide camera* clip (much easier than broadcast). Pretrained detector + ByteTrack + colour clustering + a manually clicked homography (four pitch points) is enough for a first demo.

**Phase 2 — Robustness (3–5 days).**
Fine-tune the detector (especially the ball) on Roboflow/SoccerNet data. Train or adopt a pitch-keypoint model so calibration is automatic per frame. Add smoothing.

**Phase 3 — Events (3+ days).**
Ball-ownership rules → PASS / POSSESSION_CHANGE / SHOT. Evaluate against StatsBomb/Metrica events. Add a SoccerNet action-spotting model if needed.

**Phase 4 — Live (later).**
Stream frames through Azure Event Hubs → a CV worker on Azure Container Apps (GPU) → `MatchData` windows → the existing API/SSE.

---

## 4. Evaluation: use the synthetic engine as ground truth

This is our strongest technical story. Render the synthetic match (we know every position and event) to a simple top-down or 2.5D video, run the CV pipeline on it, and compare:

- position error (metres, RMSE per player)
- ID switches per minute
- event precision / recall (pass, shot, possession change)
- **insight agreement**: does the engine still detect the high press at 16', the 4-3-3 → 4-2-3-1 switch at 66', and the red-card impact, when it runs on CV output instead of the simulator's positions?

The last metric links CV quality to *product* quality, not just detection accuracy.

---

## 5. Azure mapping

| Need | Service |
|---|---|
| Store video + model artefacts | Azure Blob Storage |
| Fine-tuning jobs on GPU, experiment tracking | Azure Machine Learning |
| Model hosting for the CV worker | Azure Container Apps (GPU) or Azure ML managed endpoints |
| Live frame/event stream | Azure Event Hubs |
| Narratives on top of CV-derived moments | Microsoft Foundry (plugs into `NarrativeGenerator`) |
| CI for the pipeline | GitHub Actions |

---

## 6. Repo changes when you start

```text
cv/
  detect_track.py      # steps 1–3
  calibrate.py         # step 4
  project.py           # steps 5–6
  events.py            # step 7
ingest/
  metrica_adapter.py   # phase 0
  cv_adapter.py        # step 8 → MatchData contract
```

`backend/pipeline.py::MatchIntelligence` gets a `source` argument (`synthetic` | `metrica` | `cv`). Nothing else changes.

## 7. Risks

- **Licensing:** footage and some datasets have usage restrictions; check them before a public demo.
- **Ultralytics YOLO is AGPL-3.0:** fine for a hackathon. For a commercial product, use RT-DETR via a permissive implementation, or buy a licence.
- **Broadcast camera:** partial view, replays and cut-aways. Detect and skip non-live shots (simple shot-type classifier).
- **Compute:** real-time multi-object tracking needs a GPU. Process offline first.
