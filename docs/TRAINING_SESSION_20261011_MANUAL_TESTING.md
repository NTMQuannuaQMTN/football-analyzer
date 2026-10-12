# Local training review and manual testing

This guide reviews the isolated 11 October training candidates. Do not overwrite restored exports to test a candidate.

## Environment and baseline

```sh
cd /Users/qnrj/Code/epl-hackathon
pwd
ls -l .venv/bin/python
.venv/bin/python -c 'import torch, cv2; print(torch.__version__, cv2.__version__, torch.backends.mps.is_available())'
.venv/bin/python -m pytest -q
cd frontend
npm run lint
npm run build -- --webpack
```

Expected: the Python path exists; tests report their actual pass/fail counts; lint and webpack build succeed. If `.venv/bin/python` is missing, first check that the terminal is in the repository above. Rebuilding the environment requires the repository dependencies plus the ML packages recorded in the session artifact; the minimal `requirements.txt` alone does not install the complete training environment. Do not launch expensive runs until imports and required local assets are available.

## Review model evidence

Read [the session report](TRAINING_SESSION_20261011.md). Candidate models are under `cv/out/training_20261011/`; original production weights remain in `cv/weights/`. Detection recall is not pitch-calibration accuracy. Tracking-based hidden-position errors exclude perception errors. Match IDs used for training, validation and testing must be disjoint within the stated experiment.

The detection audit can be repeated into a new result filename:

```sh
.venv/bin/python -m cv.evaluate_session --model cv/out/training_20261011/detector/weights/best.pt --out cv/out/training_20261011/detection_repeat.json
```

Expected: 18 test-game clips, 450 sampled frames, counts and paired conditions matching the report. The reported oracle-calibrated footpoint score must never be described as independent camera accuracy.

## Replay gate, scorers and uncertainty

```sh
.venv/bin/python -m pytest -q tests/test_camera_intervals.py tests/test_camera_footprint.py tests/test_event_identity.py tests/test_touch_cues.py tests/test_ml_leakage.py tests/test_position_evidence.py
```

Expected: shot-end intervals classify replay footage correctly, camera polygons are homogeneous-sign invariant, ambiguous goalscorers remain unknown, touches do not become observations, live features ignore future observations, and uncalibrated position uncertainty stays N/A.

For private Arsenal review, open `data/soccernet/england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/1_720p.mkv` in a local video player. Inspect 30:10 (goal sequence), 30:15 (close-up), and 30:38/30:50 (replays). The local overlays `cv/out/training_20261011/arsenal_clock_review.jpg`, `arsenal_calibration_1810.jpg` and `arsenal_refined_1810.jpg` are diagnostic evidence only. They are not embedded in the web application or published. Comparing painted-line alignment is useful, but cannot score hidden-player accuracy.

## Application presentation

Start the preserved baseline:

```sh
MATCH_FILE=data/cv_2016_08_14_18_00_arsenal_3_4_liverpool_h1_1440_2040.json ./run.sh
```

Open the Arsenal match in the lobby. Check the played date, Walcott goal credit and the existing Show/Hide prediction zones button. Open a moment's explanation overlay: effect/trade-off first; expand measured changes and positional reasoning when needed; numeric comparisons use an HTML table. Missing uncertainty displays N/A. Player identities from video roles remain explicitly unverified. Position-derived mechanisms should use conditional language, and recorded outcomes should be separate.

Training competes for local GPU/CPU resources. Benchmark playback after training ends; apparent speed while training is not a clean frontend performance comparison. Browser GUI checks and clean playback frame-rate measurements were not run in this environment.


## Private video and readable measurements

```sh
.venv/bin/python -m cv.session_report
open cv/out/training_20261011/review.html
.venv/bin/python -m cv.private_review --match data/cv_2016_08_14_18_00_arsenal_3_4_liverpool_h1_1440_2040.json --out cv/out/training_20261011/review_baseline.mp4
open cv/out/training_20261011/review_baseline.mp4
```

The movie covers clip seconds 345–405 (video seconds 1785–1845). The left side is the private local broadcast and the right side is the archived pitch view. Solid players are measured slots; rings are inferred. Shirt identities are role-assigned and unverified. Source positions are 1 Hz; interpolation to 10 fps is presentation only. Replay/close-up images cannot validate current-time positions, so the movie warns when the annotation gate excludes them. Keep the movie private; it contains restricted broadcast footage and is not served by the application.

Pause at video 1800, 1810 and the goal at approximately 1811.7. Compare visible kit colours, goalkeeper/end of pitch, penalty-area markings, relative distances and player count. Record camera/feet alignment separately from identity accuracy. Mark unreadable identities N/A. A hidden player cannot be rated against unobserved truth; later reappearance is only a consistency check unless a reliable same-player identity is available. Do not treat a smooth animation as proof of accurate reconstruction.

The HTML tables distinguish oracle-camera detector diagnostics, independent marking calibration and exact-ID hidden-player tracking tests. Check the sample counts and conditions before comparing numbers. Lower mean error does not guarantee correct named EPL players.

## Compare the private candidate

```sh
.venv/bin/python -m cv.private_review --match cv/out/training_20261011/arsenal_candidate.json --out cv/out/training_20261011/review_candidate.mp4
open cv/out/training_20261011/review_candidate.mp4
```

Expected: uniform pitch, camera outline, measured/inferred symbols and synchronised broadcast clock. This candidate worsened the measured Arsenal consistency diagnostic. Its shirt identities remain role-assigned and unverified. The archived movie is useful for visual review; the paired f0/new-model numeric comparison uses `arsenal_control.json` and `arsenal_candidate.json`, both sharing corrected detections and camera gating.

For optional web review, stop the existing backend only and start the private candidate below. Do not launch a second process on occupied port 8000. This selects a private input without replacing an active export. Restore the archived MATCH_FILE command afterward.

```sh
MATCH_FILE=cv/out/training_20261011/arsenal_candidate.json NARRATIVE_BACKEND=deterministic .venv/bin/uvicorn backend.api.main:app --port 8000
```

## Check live-mode safeguards

With the archived CV backend running:

```sh
curl -i 'http://localhost:8000/match/frames?start=0&end=0&mode=live'
curl -i 'http://localhost:8000/match/frames?start=0&end=0&mode=truth'
curl -i 'http://localhost:8000/health'
.venv/bin/python -m pytest -q tests/test_video_modes.py
```

Expected: the first two requests return HTTP 400 because this export is retrospective; health returns 200. CV confidence displays N/A; an inferred radius is a model radius, not a validated probability. In a DFL match's live mode, an unseen player's position and history must come from visibility-masked reconstruction. No future sighting or concealed ground-truth coordinate should appear. This was checked through the actual API for DFL-J03WOY half 1, player A33 at second 0.

The HTML report includes expandable role, ball-in-play and confidence groups with counts. These are tracking-domain results; ball-in-play status is not a detailed tactical phase. Raw and speed-limited predictions appear separately. Reappearance references are unverified CV detections, not ground truth.
