# Local training and reconstruction review — 11 October 2026

Status: **completed with accuracy limitations**. Session began at 03:56:49 UTC with a planned five-hour deadline of 08:56:49 UTC (16:56:49 Singapore). Host pauses/wall-clock jumps extended execution and evaluation beyond that window. This was not five continuous GPU-training hours. No further training is running. Candidate weights and private evidence remain in `cv/out/training_20261011/`; active exports and production checkpoints were preserved.

## Scope and evidence conditions

The requested priorities are visible-player pitch placement, retrospective hidden-player trajectories using later sightings and match context, and tactical explanations from those positions. Future sightings are allowed only in replay reconstruction. A trained model cannot establish exact coordinates for every unseen player when broadcast footage contains no corresponding position labels.

Available local assets: 115 SoccerNet-GSR clips from six game IDs (57 train, 58 valid); seven DFL matches with full tracking; ten SkillCorner matches supplying camera footprints; the Arsenal–Liverpool broadcast and existing exports; Metrica data and archived weights. GSR labels cover people **visible in the image**, not all hidden players. DFL experiments use exact visible tracking coordinates and identities under transferred camera masks; these conditions exclude video detection, calibration and identity errors. They are not EPL broadcast accuracy measurements.

SoccerNet video use is research-oriented and some assets require NDA access; no new footage was downloaded and private frames/crops remain ignored local artifacts. Public access does not establish unrestricted redistribution or commercial demo rights. See the [official data guidance](https://www.soccer-net.org/data) and [FAQ](https://www.soccer-net.org/faq).

## Initial checks actually executed

- `git status --short`: checkout `539f23b` plus the prior baseline-reset presentation changes and reset documentation.
- `.venv/bin/python -m pytest -q`: **20 passed**, 250 existing NumPy deprecation warnings, 43.80 s.
- `npm run lint`: passed.
- `npm run build`: failed with Turbopack worker-port permission error.
- `npm run build -- --webpack`: passed.
- Local hardware check: 24 GiB unified memory, 10 logical CPUs; PyTorch 2.14.1, MPS available outside the sandbox. Package versions are recorded in `cv/out/training_20261011/session.json`.

## Experiments and splits

### Detection, pitch markings and shirt numbers

The former GSR split used the first 12 clips of game 2 for testing and the remaining clips of that same game for validation. The new split isolates whole games:

| Purpose | Game IDs | Clips | Detector images, stride 15 | Marking images, stride 30 | Jersey crops / tracklets |
|---|---|---:|---:|---:|---:|
| Training | 4, 6, 9 | 57 | 2,850 | 1,425 | 34,129 / 933 |
| Validation | 3, 5 | 40 | 2,000 | 1,000 | 26,257 / 691 |
| Testing | 2 | 18 | 900 | 450 | 9,710 / 275 |

Detector starts from the existing generic YOLO11s checkpoint, not the old football checkpoint whose epoch selection saw game 2. Its bounded run completed **two epochs**; the epoch timer showed a long pause/wall-clock jump (945 s after epoch 1, 7,155 s after epoch 2). The time limit stopped training. This is not five hours of uninterrupted detector training. A temporary `caffeinate -i` process now inhibits idle sleep for the remaining local session. MPS reports some operations without deterministic implementations; fixed seeds do not imply bit-identical GPU results.

Paired detection audit: all 18 test-game clips, 450 frames sampled every 30 frames, 960 px inference, confidence 0.15, image-box assignment at IoU ≥ 0.5. People include players, goalkeepers and referees.

| Method | Reference people | Predicted boxes | Matched boxes | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Generic YOLO11s | 6,592 | 7,033 | 5,916 | 0.841 | 0.897 |
| Archived generic YOLO11m | 6,592 | 6,951 | 5,969 | 0.859 | 0.905 |
| Two-epoch candidate | 6,592 | 7,648 | 6,177 | 0.808 | 0.937 |

Recall increased, but false positives increased too. This is a trade-off, not an established overall improvement. Mean footpoint error under **label-fitted oracle calibration** is approximately 0.33 m for the generic model and 0.32 m for the candidate. Exact finite-footpoint counts are recorded in the detector artifacts and readable HTML tables. The final matcher maximises the number of pairs above IoU 0.5 before their summed overlap; all models were re-scored consistently. Earlier maximum-overlap-only outputs are retained under `initial_matching/`. These matched-only samples differ, and neither score validates image-to-pitch calibration or identities.

The marking network learns 17 named ground lines; the geometric solver uses only its image predictions and known field dimensions. It abstains when fewer than four independent lines constrain a camera. Pitch-player labels are scoring-only. The labelled-line solver diagnostic supported 383/450 test frames, 5,602 people: mean 3.86 m, median 0.71 m, p90 4.97 m. That uses annotated lines and is an upper-bound diagnostic, **not the learned camera's result**. The learned marking model stopped after 54 epochs using validation early stopping. On 450 held-out frames, marking Dice was 0.865; calibration supported 354 frames and 5,092 labelled people, with mean 4.20 m, median 1.22 m and p90 5.61 m. An interim validation-camera check (100 frames / 1,233 people / 70 supported frames) had mean 15.76 m, median 2.42 m and p90 27.06 m. This substantial variation and the visibly misplaced line segments in private Arsenal frames mean the model is **not enabled in production**. It also hallucinated lines on a goal close-up, demonstrating that independent shot gating remains necessary. Final jersey results are recorded below; the identity candidate failed and is not deployed.

### Hidden players

Training matches: J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9. Validation: J03WN1. Test: J03WMX. Feature versions 3 and 4 are compared with two shuffle seeds; validation chooses epochs and candidates and calibrates uncertainty. Test positions never enter features. Version 4 adds causal recent ball pace and, in replay only, explicit interpolation/velocity between the last and next visible sightings. Labels and inference operate at **1 Hz**.

Each training mode has 1,296,739 rows, and validation has 90,546 rows across four masks. Rows are correlated player-timestamps, not independent matches. A fresh first-segment rebuild matched the cached v3 feature/target prefix exactly for both live and replay. Future cache generation now binds filenames to matches, masks, stride, seed, feature version and feature-source hash; older caches lack that manifest. Held-out reconstruction results are recorded below, including the raw replay teleportation regression.

### Calibration refinement

An optional image-paint alignment experiment refines a prior camera without player truth or future frames. Private frame 1810 shows visibly better line alignment, but that alone is not an accuracy result. On controlled validation perturbations: 196 frames, 3,455 labelled people; mean error 1.68 → 1.41 m, while RMSE worsened 2.12 → 2.47 m. On held-out game 2: 86 frames, 1,250 people, **zero accepted refinements**, mean unchanged at 1.61 m. This did not demonstrate held-out improvement and is not enabled by default. The experiment perturbs label-fitted cameras; it does not measure absolute Arsenal calibration accuracy.

## Verified engineering failure sources and changes

1. **Replay contamination:** camera annotations describe shots ending at their boundaries. Old code treated them as starts. At video time 1838 s, replay footage was eligible as live observation. Corrected intervals classify 1810 as real-time main camera, 1815 as close-up, and 1838/1850 as replay. Replays and close-ups are now excluded before tracker updates. The [SoccerNet-v2 paper](https://orbi.uliege.be/bitstream/2268/253781/1/SoccerNet-v2.pdf) describes the ending shot and boundary convention. This offline annotation gate is not a deployed live camera classifier.
2. **Tracker state:** A four-frame ablation found **no tracker frame-counter difference** between shared and separate ball models: counters advanced once per person frame in both conditions. Tracked-box counts were also identical (14, 1, 1, 1). The hypothesised double update was not reproduced with the installed Ultralytics version. Separation is defensive isolation only and has no demonstrated accuracy benefit. The paired labelled tracking audit below locates substantial lost recall in tracker output and shows an association-versus-ID-switch trade-off. Named Arsenal identity accuracy remains unverified.
3. **Camera footprint:** sampled grid hulls could omit valid edge positions and depended on arbitrary homogeneous matrix sign. The footprint now clips the pitch against image-frustum half-planes exactly and is sign invariant. Calibration error still affects it.
4. **Identity versus event facts:** regenerated goals previously chose the nearest inferred player as scorer. Goals now resolve unambiguous lineup goal facts; unresolved/ambiguous scorers remain unknown. This preserves Walcott attribution without hardcoding his name. Caption text timestamps are delayed and are not player-position timestamps.
5. **Invented observations:** ball-touch pins could overwrite measured feet and mark hidden positions observed. They are now soft cues applied only to hidden estimates, with observation flags unchanged.
6. **Tactical certainty:** missing inferred confidence was replaced with 50%. It remains N/A now. New position reasoning separates measurement, interpretation, mechanism, recorded outcome, trade-off and alternatives. Radius sensitivity is explicitly not a joint confidence interval. The UI uses expandable explanations and an HTML table, with no JSON viewer. Moment confidence is labelled heuristic evidence strength.

## Commands executed for this session

```sh
.venv/bin/python -u -m cv.train_session --hours 1.6
.venv/bin/python -u -m backend.reconstruction.train_session --deadline 1791706609
.venv/bin/python -u -m cv.train_pitch_lines --prepare-only
.venv/bin/python -u -m cv.train_jersey_session --prepare-only
.venv/bin/python -u -m cv.training_queue --after-pid 20078 --deadline 1791706609
.venv/bin/python -u -m cv.evaluate_session --model cv/weights/yolo11s.pt --out cv/out/training_20261011/detector_baseline_yolo11s.json
.venv/bin/python -u -m cv.evaluate_session --model cv/out/training_20261011/detector/weights/best.pt --out cv/out/training_20261011/detector_candidate_people.json
.venv/bin/python -u -m cv.eval_refinement --part val
.venv/bin/python -u -m cv.eval_refinement --part test
.venv/bin/python -m compileall -q cv backend
git diff --check
```

Additional Python snippets inspected labels, hardware, package versions and cache consistency, and generated/inspected private broadcast clock and calibration overlays. Tests executed after initial changes: **26 passed**, 250 warnings, 291.30 s while training competed for resources. Additional focused tests passed for box matching, event identity, camera intervals/footprint, calibration abstention and touch-cue semantics. Frontend lint and webpack production build passed after the evidence-panel changes. Final regression results and the complete source-file inventory are recorded below.

## Acceptance and limitations

| Requirement | Current status |
|---|---|
| Accurate localisation of every visible player | Not achieved; imperfect detection, calibration and identity remain |
| Hidden-player predictions using future sightings, pace and role | Implemented and evaluated; modest DFL gain, Arsenal consistency regression; isolated candidate |
| Separate causal and retrospective features | Future-mutation tests passed for feature versions 1–4 |
| Position-grounded tactical explanations | Implemented; numeric geometry and radius-envelope tests passed |
| Integrated candidate game and private video comparison | Private exports and synchronised movies generated and inspected; production candidate promotion declined |
| No paid computation or footage downloads | Met |

Main remaining limitations: no full hidden-player tracking ground truth aligned to the Arsenal video; role-based identities; weak/absent jersey-legibility labels; uncertain video calibration; domain gap from Bundesliga tracking to EPL broadcast; only one strictly held-out tracking test match in the current experiment. Synthetic or oracle-visible performance cannot establish real broadcast accuracy. Ranked recommended actions appear below.

Manual review instructions: [training manual guide](TRAINING_SESSION_20261011_MANUAL_TESTING.md).


## Reconstruction results (completed)

Validation selected v4/seed22026 for both modes. Validation MAE: live 6.710 m, replay 5.556 m. Best v3 in the same session: live 6.762 m, replay 5.597 m. Therefore the added features have only a small validation advantage; changes from old f0 also include the training procedure. Old f0 checkpoint metadata confirms the identical five training / one validation / one test match split.

| Mode | Mask | Player-timestamps | Old mean m | New mean m | New median m | New p90 m | 80% radius coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| live | sc_camera | 97,116 | 8.20 | 7.85 | 5.90 | 16.90 | 81.4% |
| live | k=14 | 46,720 | 4.88 | 4.83 | 3.47 | 10.24 | 81.7% |
| live | k=10 | 70,080 | 5.86 | 5.81 | 4.31 | 12.04 | 82.0% |
| live | k=6 | 93,440 | 7.35 | 7.23 | 5.44 | 15.04 | 80.8% |
| replay | sc_camera | 97,116 | 7.05 | 6.69 | 4.71 | 14.97 | 81.0% |
| replay | k=14 | 46,720 | 3.86 | 3.80 | 2.71 | 7.83 | 82.6% |
| replay | k=10 | 70,080 | 4.80 | 4.67 | 3.45 | 9.63 | 83.2% |
| replay | k=6 | 93,440 | 6.50 | 6.20 | 4.69 | 12.98 | 81.5% |

These are raw corrections, evaluated with exact player IDs and oracle visible positions/context. One held-out match does not establish generalisation to EPL video. Radius coverage is evaluated in the tracking domain only. Sample counts are correlated timestamps, not independent matches. Hidden-to-hidden raw replay steps over 10 m/s increased from 1,683 to 1,805: a regression that cannot be hidden behind mean error. Existing speed limiting is evaluated separately, preserving observations and using future anchors only in replay. Goalkeeper-zone and event-cue priors are excluded from this controlled comparison.

The initial candidate detector, the independent camera network and jersey network remain isolated. No active export or production weight has been replaced.


## Additional causal/provenance safeguards

The CV export is retrospective: ball interpolation, team-colour voting and role assignment use the complete clip. The API now rejects requests to relabel that export as `live` or `truth` when no corresponding explicit frames exist. Feature-level live tests also mutate future ball position, possession and priors, not just future player sightings. Tactical moment detection uses before/after windows and remains retrospective; streamed playback does not make those explanations real-time.

The DFL benchmark uses pre-match playing-position metadata for role labels. Its legacy parser chooses canonical direction from whole-period keeper means; the report therefore treats direction as oracle preprocessing, not a tested live direction estimator. The raw XML clock audit completed for all seven matches / 14 halves: no non-1-second sampled steps, and initial keeper sides agreed with whole-period directions. This does not validate an online orientation estimator. Existing 1 Hz caches lack stored frame-number arrays, and non-default frequency cache keys are a remaining limitation; the reported experiments all use 1 Hz.

Arsenal candidate generation completed from 3,001 sampled frames (5 Hz, video 1440–2040), with 1,566 accepted main-camera samples. The candidate has 601 frames, 22 finite roster positions per frame and an observed share of 0.334. This is a coverage count, not accuracy. Reappearance proxy: 282 samples, median 17.42 m; it includes uncertain role identities and calibration, so it is **not ground-truth position error**. The source metadata labels it separately from truth. The goal clock now uses the actual goal annotation at clip 371.68 s, separately from the earlier shot annotation; scorer facts identify Theo Walcott. Active match data remain unchanged.


## Controlled postprocessing and real-clip regression

Under the **same existing speed limiter**, old/new hidden-position means are live 8.11 → 7.78 m and replay 6.83 → 6.46 m (97,116 hidden player-timestamps each). Both have zero hidden-to-hidden steps above 10 m/s across 91,112 one-second transitions. New 80%-radius coverage is 81.8% live / 82.0% replay. This comparison excludes keeper-zone and event-cue priors; its observations and IDs are exact tracking data.

Private Arsenal control/candidate use identical stock-YOLO11m raw detections, original CCBV calibration, corrected replay gate and role assignments. Only hidden-model weights differ. The reappearance consistency mean **worsened 18.44 → 18.82 m**, median **16.06 → 17.42 m**, on 282 role-assigned references. These references are not tracking ground truth. Both exports retain 72 hidden-related transitions above 10 m/s out of 9,531 transitions, maximum 48.7 m/s. These include transitions into preserved observations: an identity mismatch or bad calibration can defeat a movement limiter at reappearance. No Arsenal improvement is supported, so neither candidate replaces the archived match.

The private synchronised movies `review_baseline.mp4` and `review_candidate.mp4` each contain 600 readable frames at 10 fps, 1440×620. Source coordinates are 1 Hz; interpolation is presentation only. Actual frame 1810 was inspected: visible shirt identities still conflict with role-assigned pitch markers. A factual Walcott goal record does not validate the identity of every coordinate. Close-ups/replays carry a warning and are excluded from fresh observations. Broadcast footage remains local and is not embedded in the website.

## Tracking diagnostic on identical detections

Two validation clips from games 3/5 and two test clips from game 2 are sampled at 5 Hz. These are paired motion ablations with identical raw detections, not an end-to-end detector comparison. ID switches count identity changes between matched sightings including gaps; full HOTA/IDF1 was not run.

| Split | Method | GT people | Matched | Recall | ID switches / matched transitions |
|---|---|---:|---:|---:|---:|
| Validation | Raw detections | 4,918 | 4,456 | 90.6% | N/A |
| Validation | Legacy tracker | 4,918 | 3,029 | 61.6% | 247 / 2,983 |
| Validation | Association guard | 4,918 | 3,223 | 65.5% | 312 / 3,176 |
| Test | Raw detections | 4,374 | 4,036 | 92.3% | N/A |
| Test | Legacy tracker | 4,374 | 3,241 | 74.1% | 187 / 3,195 |
| Test | Association guard | 4,374 | 3,322 | 75.9% | 262 / 3,276 |

The guard recovers additional sightings but increases identity switches, including their rate. It is **not promoted**. Disabling camera motion or using the pitch-focused flow guard did not resolve the trade-off. These results support prioritising track association and identity repair over simply training the hidden model longer.

## Verified application and causal safeguards

Final full Python regression: **42 passed**, 250 existing NumPy deprecation warnings, 246.53 s. Frontend lint and webpack production build passed after the latest presentation changes. `compileall` and `git diff --check` passed. Local API smoke test returned 22 player slots, 12 observed at clip second 370, N/A CV confidence, and one moment with position reasoning. Unsupported CV `live`/`truth` requests returned HTTP 400. The backend serves the preserved archive with deterministic narration.

Live player-state/history now select the visibility-masked reconstruction, rather than falling back to full tracking truth. Live formation-line labels use only past/current estimated frames, not the smoothed retrospective truth timeline. Mutation tests cover future player, ball, possession, prior and formation-timeline changes. All CV probabilities are withheld in the API because tracking-domain confidence does not measure perception/identity error; private inferred radii remain illustrative model radii. Tactical windows and outcome explanations remain retrospective.

Baseline preservation was checked from file modification times and exact archived Arsenal `xy`, `obs`, `ball` equality: 601 frames unchanged; no active match export or production `.pt` weight modified during this session. GUI browser checks were **not run** because no controllable browser surface was available. Private image/movie inspection and API verification were run; frontend playback frame rate was not measured under clean idle conditions.


## Final detector and jersey results

The detector continuation completed three additional epochs, selecting the best by validation mAP50–95 (0.234). Its epoch elapsed time reached 6,685 s because of another host pause; the one-hour training setting did not guarantee a one-hour wall-clock finish. Together with the initial two epochs, **five detector epochs** completed. The queued low-learning-rate final run was **not run** because its safety deadline elapsed. This is recorded rather than counting waiting time as training.

The continued detector was scored under identical CPU inference/matching conditions on 450 held-out images, 6,592 labelled people: 6,836 predictions, 6,151 matches, precision **89.98%**, recall **93.31%**. Archived generic YOLO11m had precision **85.87%**, recall **90.55%**. Both improved on this test, but the clips share one game and do not establish Arsenal camera/identity accuracy. Conditional oracle-camera footpoint errors: 5,899 matched finite samples, mean 0.31 m, median 0.20 m, p90 0.61 m. This excludes camera error. The detector is available as an opt-in local checkpoint, not silently substituted into every export.

The jersey network completed four epochs. Validation threshold 0.95 gave 44 correct / 52 decided tracks (84.6% precision), just 7.5% coverage out of 691 tracks. No tested threshold met the 90% validation-precision target. At that conservative threshold, held-out test produced **2 correct / 7 decided tracks**, 28.6% precision and 2.5% coverage over 275 tracks / 9,710 crops. Crop accuracy was 7.7%. This is an unsuccessful identity experiment, not a reliable identity solution. It was not deployed. Front-view/unreadable-number crops lack legibility labels; future voting is retrospective.

GPU jobs and final CPU evaluations are finished. The temporary sleep-inhibition process was stopped. Saved weights are private experimental artifacts, and the web backend still uses the archived match with deterministic narration.

## Tactical definitions and remaining scope

Width is max minus min outfield y in metres; depth is max minus min own-team x; defensive-line height is the mean of the four lowest own-team outfield x coordinates. At least seven finite outfield positions are required. These are geometric proxies, not verified coach intent. Radius-based box envelopes show sensitivity, not joint calibrated confidence. Measured changes of at least 2 m support conditional mechanisms; otherwise the UI keeps the existing event explanation. Outcome descriptions refer to the recorded retrospective sequence. Detailed passing-lane, overload and counterattack models were not newly trained in this session.

## Additional commands actually executed

```sh
.venv/bin/python -u -m cv.train_session --hours 1.0 --out cv/out/training_20261011/detector_continuation --data cv/out/training_20261011/detector_data/data.yaml --initial cv/out/training_20261011/detector/weights/best.pt
.venv/bin/python -m cv.evaluate_session --model cv/weights/yolo11m.pt --out cv/out/training_20261011/detector_baseline_yolo11m.json
.venv/bin/python -m cv.evaluate_session --model cv/out/training_20261011/detector_continuation/detector/weights/best.pt --out cv/out/training_20261011/detector_continuation_people.json
.venv/bin/python -m cv.eval_jersey_session --device mps
.venv/bin/python -m pytest -q
npm run lint
npm run build -- --webpack
.venv/bin/python -m cv.session_report
.venv/bin/python -m cv.private_review --match cv/out/training_20261011/arsenal_candidate.json --out cv/out/training_20261011/review_candidate.mp4
```

Python snippets also ran paired old/new speed-limited reconstruction evaluation, streamed the DFL XML clock audit, generated corrected private Arsenal control/candidate exports, checked local API live history, and verified archive equality. Their output artifacts are under the session directory. Logs are under `/private/tmp/football_*.log`; logs alone are not position-accuracy evidence. A malformed documentation shell heredoc failed and was replaced using a patch; it did not alter model outputs.

## Recommended next work

1. Repair and evaluate player association across camera motion/cuts, including reappearance identity continuity; higher recall with more identity switches is insufficient.
2. Build a reliable camera estimator and manually label a permitted multi-clip calibration test, including close-ups, occlusion and orientation. Keep oracle camera diagnostics separate.
3. Obtain permitted, aligned EPL video/full-player tracking labels. More broadcast images alone cannot label hidden positions. Add legibility labels before jersey-number training.
4. Evaluate the new detector through the complete perception pipeline on multiple independent matches before promotion. Preserve the archive and compare named-player position accuracy, not just box recall.
5. Re-evaluate hidden models once perception/identity is reliable; report uncertainty by off-camera duration and regressions at reappearance. Detailed tactical reasoning must remain conditional until its inputs are trustworthy.

No claim is made that every visible or hidden EPL player is accurately located. A hidden player may also be a missed detection inside the image; forcing its estimate outside the camera would manufacture location evidence. These models do not apply that workaround.

## Files modified or created

- `.gitignore`
- `backend/analytics/shape.py`
- `backend/api/main.py`
- `backend/models/schemas.py`
- `backend/narratives/generator.py`
- `backend/pipeline.py`
- `backend/reconstruction/inference.py`
- `backend/reconstruction/ml.py`
- `cv/build_match.py`
- `cv/clip_extras.py`
- `cv/detect_track.py`
- `cv/gsr_train.py`
- `cv/soccernet.py`
- `frontend/components/insights/MomentPanel.tsx`
- `frontend/components/pitch/CameraRealityPanel.tsx`
- `frontend/components/pitch/PlayerStateCard.tsx`
- `frontend/components/studio/Studio.tsx`
- `frontend/lib/types.ts`
- `tests/test_ml_leakage.py`
- `backend/reconstruction/session_evaluation.py`
- `backend/reconstruction/train_session.py`
- `backend/tactics/position_evidence.py`
- `cv/audit_dfl_clock.py`
- `cv/audit_tracker_state.py`
- `cv/broadcast_motion.py`
- `cv/eval_jersey_session.py`
- `cv/eval_refinement.py`
- `cv/eval_tracking_session.py`
- `cv/evaluate_session.py`
- `cv/event_identity.py`
- `cv/pitch_lines.py`
- `cv/private_review.py`
- `cv/refine_calibration.py`
- `cv/session_report.py`
- `cv/touch_cues.py`
- `cv/train_jersey_session.py`
- `cv/train_pitch_lines.py`
- `cv/train_session.py`
- `cv/training_queue.py`
- `docs/AI_IMPLEMENTATION_STATUS.md`
- `docs/BASELINE_RESET.md`
- `docs/TRAINING_SESSION_20261011.md`
- `docs/TRAINING_SESSION_20261011_MANUAL_TESTING.md`
- `tests/test_broadcast_motion.py`
- `tests/test_calibration_refinement.py`
- `tests/test_camera_footprint.py`
- `tests/test_camera_intervals.py`
- `tests/test_detection_matching.py`
- `tests/test_event_identity.py`
- `tests/test_pitch_lines.py`
- `tests/test_position_evidence.py`
- `tests/test_touch_cues.py`
- `tests/test_video_modes.py`

Ignored local models, datasets, images, movies and evaluation artifacts are listed by `cv/out/training_20261011/`; none are web-hosted footage.
