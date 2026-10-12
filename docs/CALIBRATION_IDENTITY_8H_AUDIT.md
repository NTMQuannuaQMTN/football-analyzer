# Eight-hour calibration and identity audit

Status: completed bounded eight-hour work window. Started 11 October 2026, 15:09:31 UTC (23:09:31 Singapore); deadline 23:09:31 UTC (07:09:31 Singapore, 12 October). Work is local, using existing assets and dependencies. Host pauses or interruptions will be reported, not counted as continuous GPU computation.

## Review first

**No experimental checkpoint, tracking setting, calibration or active match export was promoted.** The expanded camera and tracking gates fail, and context training trades clean-input accuracy for stale-context robustness. The useful production changes are honest provenance/status, timestamp-safe player panels and live rendering, cache/clock/identity guards, and safer startup. Accurate named reconstruction of all 22 EPL players remains incomplete.

- Main private review: `cv/out/audit8_20261011/review.html` (native HTML tables; diagnostic links expand on demand).
- Your useful action: [18-frame landmark/shirt review instructions](CALIBRATION_IDENTITY_8H_USER_ACTIONS.md). No extra training or paid data is required from you. Independent real human CSV is still missing. Assistant-provisional readings and UI-smoke clicks are explicitly excluded from human-ground-truth claims.
- Latest test evidence: **132 Python tests passed**, plus the separate actual-component causal-rendering check; frontend lint and webpack production build passed. Detailed timestamps, conditions, unsuccessful commands and limitations follow chronologically. Final preservation/time accounting is recorded in the closeout section below.
- Seven fresh nested-match folds: **28 models / 840 full epochs**, same test conditions within each control/candidate pair. Stale-ball mean improves, clean mean regresses; all seven per-match and observation-age tables are retained. These German tracking/simulated-camera results do not establish EPL broadcast accuracy.
- Camera live holds its latest 1 Hz sample; clip replay interpolates retrospectively. Existing tactical timelines/stories are explicitly retrospective and are not a new real-time tactical engine.

Earlier “running” sections below record progress at their stated times; later completion/failed-gate sections supersede their pending decisions without erasing the earlier evidence.

## Initial state

The previous session's uncommitted source changes and private candidates are retained. Production match exports and checkpoints are not replaced. Initial checks actually ran: **42 tests passed**, 250 existing NumPy deprecation warnings, 87.17 s; frontend lint passed; webpack production build passed. No new dependencies were installed. `trackeval` and `motmetrics` are absent; `lap`, Ultralytics, PyTorch, OpenCV and SciPy are available.

Local assets include six GSR games with image boxes, pitch markings, visible pitch coordinates and track identities; one Arsenal broadcast with CCBV camera estimates; and DFL full-position tracking. GSR images label visible people, not all hidden players. The previous whole-game split is preserved: train 4/6/9, validation 3/5, test 2. Test results must not select candidate parameters.

## Work plan and gates

1. Separate raw detection, tracking association, coordinate mapping and named identity errors. Record counts, finite coordinates, discontinuities and evaluation conditions.
2. Audit camera geometry using image markings as inputs and independent labelled feet as scoring-only references. Distinguish model-estimated cameras from label-fitted diagnostics.
3. Compare tracking settings on identical cached detections. Select settings on validation games; evaluate the chosen candidate once on the held-out game. No future observations in causal association.
4. Produce private frozen-frame review and a manual annotation workflow. Ground truth and manual evaluation labels must never enter normal inference.
5. Integrate only evidence-supported, reviewable changes; run regressions and inspect actual output. Document failed experiments and unresolved accuracy.

## User actions

See [the separate action checklist](CALIBRATION_IDENTITY_8H_USER_ACTIONS.md). Manual review is optional for continued automated work, but missing labels limit real-video accuracy claims.

## Measured findings so far (11 October, 16:13 UTC)

### Tracking and missing visible people

Generic YOLO11m raw detections were cached once per frame. Nine tracker settings were compared on 1,200 validation frames across eight clips from games 3/5. Selection allowed at most a 1.5 percentage-point precision loss and required non-decreasing anonymous identity F1. The selected setting retains high/new thresholds 0.30/0.35, disables score fusion and raises the match threshold to 0.95. It is saved privately in `cv/out/audit8_20261011/selected_tracker.yaml`; production tracker settings remain unchanged.

The frozen selection was tested on all 18 game-2 clips, 2,700 frames at 5 Hz, **39,437 labelled visible-person observations**:

| Metric | Legacy | Selected |
| --- | ---: | ---: |
| Visible-person precision, IoU ≥ 0.5 | 90.79% | 89.54% |
| Visible-person recall | 73.57% | 79.85% |
| Detection F1 | 81.28% | 84.41% |
| Anonymous identity F1 | 47.55% | 51.22% |
| Matched ID switches / transitions | 2,052 / 28,611 | 1,820 / 31,084 |

These are anonymous image-space tracks, including referees and goalkeepers. They do **not** verify shirt numbers, full lineups, pitch coordinates or hidden players. Global one-to-one trajectory assignment uses future frames only in evaluation, never association. The implementation is tested but is not the absent official TrackEval/GS-HOTA package. The continued detector's best eligible validation tracker was its legacy configuration, so the generic detector's settings cannot be transferred blindly. Game 2 was also examined in the previous session: it is held out from training/selection, not a newly untouched external test.

### Timestamp bias

The historical exporter averages observations over `[second, second+1)` and labels that average at the beginning of the second. A controlled **exact-visible-coordinate/identity** experiment isolates this timing error: 20,330 common validation samples have mean 0.904 m, median 0.758 m and p90 1.807 m displacement from the integer-time reference; 7,594 held-out samples have mean 0.766 m, median 0.568 m and p90 1.724 m. Actual mean source offsets are +0.391 s and +0.388 s. Selecting the exact integer-time observation has zero error **by construction**, not evidence of a trained-model improvement. `--sampling nearest` is an opt-in private export option; the legacy default remains unchanged. Actual measurement timestamps are retained in new exports. Future-bucket averaging is disallowed by the utility's causal mode.

### Camera instability and label problems

Three parallel pitch lines plus one transverse line do not constrain a full homography, even if noisy points make a numerical matrix appear full rank. Enforcing two distinct lines in each family removes a demonstrably unstable labelled example. A viewed GSR validation frame, `SNGS-058/000541.jpg`, has a named penalty-area line drawn along the goal line; labelled feet diagnose about 14 m mismatch for that correspondence. This is evidence for a specific annotation failure, not a claim that all GSR labels are unreliable.

Train-only cleanup uses cross-validated reference-foot cameras to check line labels: 1,273 of 1,425 training frames pass the trust criteria; 21 line names in 21 frames are corrected. Validation and test labels are unchanged in the generated rows. Those coordinates supervise training cleanup only and never enter camera inference.

Unconstrained nonlinear fitting can lower median error while catastrophically increasing the mean. Validation learned-camera DLT has 24.12 m mean over 12,355 supported feet; image-residual fitting has 62.14 m on the same support. Neither is promoted. Finite pitch-segment consensus reduces tails but changes coverage: on its common validation set, DLT mean 12.48 m becomes 4.76 m over 10,731 feet, with 626/1,000 frames supported.

An image-only physical plausibility filter selected on validation accepts 402/1,000 validation frames, 6,986 feet: mean 2.21 m, median 1.16 m, p90 4.93 m. Its **frozen** held-out result accepts 338/450 frames, 4,839 feet: mean 1.91 m, median 1.07 m, p90 3.99 m. On those exact held-out feet, unfiltered DLT mean is 2.09 m. Rejection changes coverage; this is not whole-coverage accuracy. Further, 51 accepted test frames still have per-frame p90 above 5 m. These checks indicate plausibility, not calibrated confidence. They use image grass, projective horizon and scale; reference feet score only.

The first 36-epoch tiny-camera continuation with cleaned labels does not establish a geometry improvement: full validation DLT mean worsens to 43.46 m; segment-consensus mean 4.72 m remains high and coverage differs. Matched raw-label continuation and local pretrained ResNet camera experiments continue separately. Training loss is not an accuracy claim.

### Commands actually executed

Initial checks: `.venv/bin/python -m pytest -q`; frontend lint and webpack production build (42 tests passed; lint/build passed).

Experiments used `.venv/bin/python -m` with: `cv.audit_tracking` (validation generic/candidate and held-out generic), `cv.audit_camera` (full validation/held-out cameras), `cv.audit_camera_quality` (selection/frozen test filter), `cv.clean_line_labels`, `cv.train_camera_audit`, `cv.audit_temporal`, `cv.audit_foot_measurements`, `cv.audit_broadcast prepare`, `cv.manual_camera_review`, and `cv.build_match ... --sampling nearest --out cv/out/audit8_20261011/arsenal_timestamp_candidate.json`. Configuration, model hashes and sample lists are retained in the private artifact directory. Person-embedding training is queued, not yet reported as executed.

Focused automated checks actually ran: 15 tests across temporal samples, camera geometry/quality, pitch lines, label cleanup, embedding loss and identity metrics; separately 2 manual-camera evaluation tests. Full final regressions remain pending.

## Further measured findings (17:05 UTC)

### Coverage and placement together

The integrated held-out evaluation uses the same 450 camera timestamps, **6,592 labelled visible-person observations**, cached stock YOLO11m detections, IoU matching before pitch scoring, and image-only camera estimates. Camera abstentions and missed detections count against the all-labelled within-five-metres recall. Conditional mean errors exclude those missing samples explicitly.

| Pipeline | Matched, camera-supported observations | Conditional mean error | All-labelled within-5 m recall |
| --- | ---: | ---: | ---: |
| Legacy tracker, static camera, Kalman feet | 3,587 | 1.993 m | 50.09% |
| Selected tracker, static camera, Kalman feet | 3,880 | 2.005 m | 54.20% |
| Selected tracker, propagated camera, Kalman feet | 4,123 | 1.945 m | 57.99% |
| Raw detections, propagated camera | 4,651 | 1.967 m | 65.32% |

Raw detections have no confirmed track/lineup identity. These results concern visible players, goalkeepers and referees in game 2, not all 22 named EPL players or hidden positions. The full validation evaluation is still running. The roughly two-metre conditional result must not be described as whole-match accuracy.

### Causal camera propagation

Ground-paint optical flow uses current/past images, forward/back consistency and RANSAC; no reference coordinates enter it. Validation selected propagation age at most three seconds with a ten-metre anchor-agreement gate. It expands validation support from 402 to 483 of 1,000 frames. Frozen test support rises from 338 to 359 of 450 frames (4,839 to 5,167 feet); conditional mean changes 1.906 to 1.873 m. On the common 4,839 feet, mean changes 1.906 to 1.837 m; the 328 added feet have mean 2.394 m.

A 2,000-resample clip bootstrap gives a common-set mean-change interval of −0.197 to +0.054 m, crossing zero. This supports increased coverage under these conditions, not a strong statistically established placement gain. All held-out clips belong to one game; the interval does not establish EPL generalisation.

Camera uncertainty bins fitted on validation reference feet cover 81.97% of 5,065 supported held-out feet with their nominal 80% empirical radii; 102 further test feet have no supported bin. These are reference-foot camera errors, excluding detector, identity and hidden-player errors. They are not deployed probabilities for Arsenal estimates.

### Arsenal-specific failures

All five learned-camera estimates on reviewed main-camera Arsenal frames fail image-only plausibility checks. Existing CCBV estimates pass those checks but visibly misalign with some painted lines. A safe-looking camera is therefore not sufficient evidence of accurate calibration.

A paint-alignment clock sweep selects −0.5 s on the first part of the clip (37 eligible images): chamfer distance 26.26 → 21.79 pixels. On the later 38 images, 24.71 → 20.73 pixels. This is an image-line proxy, not independent position ground truth; an inspected 1800-second overlay has a locally worse corner with the offset. The offset remains private and opt-in, not promoted.

Exact track-to-role locking gives a small oracle-visible identity gain on game 2 (ID F1 98.075 → 98.247%, 10 → 8 role switches), but the private Arsenal export has **more** observed-to-observed steps above 10 m/s (361 versus 346 under nearest sampling). It is not promoted. Anonymous track continuity does not establish shirt-number identity.

### Pending experiments and honest feasibility bounds

The raw-label tiny-camera control has no operating point meeting the validation gate; the cleaned-label tiny continuation also does not establish broad improvement. The local pretrained ResNet marking model is still training. Appearance training and paired Arsenal re-extraction are queued, followed by an image-only bounded physical-camera experiment.

The physical camera family can fit training/validation correspondences: a diagnostic half-player fit gives held-out-half mean 0.199 m over 258 training feet and 0.151 m over 274 validation feet. **This uses labelled correspondences and is not neural-network inference accuracy.** It only establishes that the seven-parameter camera family can represent these examples. The image-only model still needs training and independent evaluation.

### Additional commands actually executed

`cv.audit_camera_motion` (validation/frozen test), `cv.audit_camera_uncertainty`, `cv.audit_comparisons`, `cv.audit_broadcast_cameras`, `cv.audit_calibration_clock`, `cv.audit_role_aliases`, `cv.audit_physical_family`, and `cv.audit_visible_pipeline --part test` completed. `cv.audit_visible_pipeline --part val` remains running. Private nearest/stable-role exports and timestamp-continuity comparisons were built; no active export was overwritten. A 60-second private review movie was rendered and equal-scale pitch rendering checked. The local annotation page was opened in Chrome and its image/table/frame selector inspected; click/undo/export GUI steps remain unverified. Focused causal-calibration, appearance-cost, role-association, retrieval, uncertainty, API timestamp and physical-camera tests ran successfully; the final suite is pending.

## Verification and additional results (17:40 UTC)

The mid-session full suite actually ran: **71 passed**, 250 existing warnings, 276.11 s under concurrent evaluation/training load. Frontend lint and webpack production build passed; `git diff --check` passed. A subsequent catalog regression adds one test (4 catalog tests passed in 10.73 s); final full-suite verification is still pending further work.

### Active match and API checks

All **19 active file exports** were executed through `MatchIntelligence`, sampled at start/middle/end, and their summaries/states checked for JSON serialisability. No integer-second contract/array-length/non-finite-present-coordinate defects were found. Each source's available/rejected replay/live/truth modes are recorded separately. This is functionality verification, not identity or reconstruction accuracy; legacy precomputed live exports were not retrained or independently recertified by this smoke check.

The preserved Arsenal archive has 4,373 measured and 8,849 estimated player-timestamps, with 395 measured-to-measured steps above 10 m/s among 3,551 transitions. This differs from the earlier regenerated f0 control; those counts must not be conflated. Other source-provider tracking can contain genuine out-of-field points; this audit preserves them rather than clipping evaluation data.

The backend was restarted using only its verified project PID, keeping the archived Arsenal selection. Actual HTTP checks passed for health, 13 catalog cards, the 601-frame export, the current state and frontend HTTP 200. CV live/truth requests returned explicit HTTP 400. Walcott remains the factual goalscorer; calibration and role-slot identities remain unverified. Live/truth API rejection is checked independently from file mode availability.

SkillCorner card provider/date/unknown-score handling was corrected: provider IDs are not dates, exported dates are used, and `score_known: false` propagates to lobby/summary so the UI displays `? - ?`. The studio metadata inherits the same source date. CV tactical reliability is now `unverified`, even when measurement coverage is high. Measurement coverage is not a probability of positional accuracy.

A cold local lobby benchmark took **86.506 s**, because one synthetic card built full reconstruction and tactical analysis. Cached simulator metadata replaces that card-only work; the executed after benchmark took **15.909 s** cold and **0.0064 s** warm for the same 13 cards. These measurements ran under concurrent jobs and include live-source initialisation; they are not browser playback FPS or controlled hardware throughput claims. Full synthetic analysis is retained when opened. An initial cold HTTP catalog check timed out; warm corrected HTTP checks succeeded. Two smoke-script contract/date assumptions were corrected before the successful check; no unsupported dates or test pass claims are retained.

### Continued detector in the same camera pipeline

The previous session's continued four-class detector was evaluated with the same frozen static/propagated camera sample list and its own validation-selected tracker. On the 6,592 held-out visible-person observations, raw detections plus propagated camera reach **67.29% within 5 m** (4,791 supported matches, mean 1.925 m), versus generic raw detection 65.32%. With confirmed tracks, however, its selected legacy tracker reaches **55.48%** (3,938 supported matches, mean 1.878 m), below the generic selected tracker's 57.99%. Different supported sets preclude claiming that the lower conditional mean is a paired placement improvement. Better raw detection does not automatically improve full confirmed coverage.

### Colour and marking failures

Generic colour clustering on 450 fixed test timestamps scores 4,512/5,273 matched outfield crops correctly (85.57%) after the best two-way team permutation. Label-aware exclusion of goalkeeper/referee/unmatched crops scores 5,086/5,273 (96.45%). The latter is an **oracle diagnostic**, not deployable identity accuracy. It excludes errors in pitch orientation, detection and track voting. Full validation and the continued detector's actual role-based exclusion are pending; a partial validation attempt omitted 359 not-yet-cached timestamps and is not treated as a complete result.

The 46-epoch ResNet marking experiment fails the predeclared full-validation operating gate. DLT mean is 7.16 m over 13,426 supported feet; image-fit mean is 20.86 m; consensus mean is 4.16 m over 12,415 feet. One rejected quality operating point supports 612/1,000 frames with mean 2.54 m but p90 5.90 m, exceeding the five-metre tail criterion. The camera test candidate was therefore **not run**, rather than relaxing the gate after seeing the failure. Appearance training and physical-camera training remain independent experiments.

### Actual failure visualisations

`cv.render_visible_failures` regenerated anonymous tracker measurements from frozen test caches and verified rendered errors against scored errors. Ten private contact sheets include eight high-error frames from distinct clips and two low-error controls. The 13.63-metre `SNGS-032/000181` case and 0.31-metre `SNGS-026/000481` control were actually inspected. The high-error blurred frame has only one supported matched track despite many visible people; its one-point conditional mean does not establish whole-frame accuracy.

Open `cv/out/audit8_20261011/visible_failures/review.html` and `active_matches/review.html` for native HTML tables and images. The main application does not embed private football video. Chrome later exposed no accessible window; no full studio GUI/FPS check is claimed.


## Further diagnosis and experiments (18:05 UTC)

### Reset-local tracker IDs were incorrectly shared across camera cuts

Installed Ultralytics `BYTETracker.reset()` resets the global ID counter. The previous producer resets tracking after unsupported/replay images, but the exporter grouped all detections by bare integer ID across the entire clip. Consequently, unrelated people in different camera intervals were treated as one anonymous trajectory before role assignment. In the original Arsenal cache, **262 of 676 native IDs occur in multiple supported intervals** (35 intervals; 1,208 shot/ID pairs). This proves a namespace collision, not that all such collisions were different named players.

`cv.track_namespace` now assigns stable IDs to `(supported interval, native ID)` using only past/current camera flags. `cv.detect_track` emits those IDs with their native ID and interval. `cv.build_match` qualifies old flagged caches by default; `--track-namespace legacy` explicitly reproduces former grouping. No position is changed by qualification and source caches remain untouched. Cross-shot person matching and shirt names remain unverified. A missing camera flag now requires explicit legacy mode rather than silently guessing shot boundaries.

Two preserved cached paired stock-detector runs also contain collisions: legacy tracker 236/650 IDs, selected tracker 173/556. Their copied qualified caches preserve all 17,443 and 20,177 person measurements respectively. Four private nearest-time exports use identical model f0, detections, ball, cameras and event data within each pair. The role-assigned reappearance proxies have different sample sets: legacy native/qualified mean 18.52/18.43 m (315/297 samples), selected native/qualified 18.83/17.52 m (291/283). **These are not paired independent accuracy gains**; reference identities are still guessed, and coverage changed. Active match JSONs were not rewritten.

Executed: `cv.qualify_cached_tracks --run-dir .../broadcast/{legacy,selected}_raw --out .../broadcast/{legacy,selected}_qualified`; four `cv.build_match` commands retained in `broadcast/build_pair_results.json`, with `--sampling nearest --recon-fold f0 --track-namespace legacy` for native controls and shot qualification for candidates. Two namespace tests check cut separation, unchanged measurements, ball ID preservation, idempotence and prefix causality.

### Full validation coverage is substantially lower than the held-out game

The full generic camera/detection validation run completed on **1,000 timestamps and 17,064 labelled visible-person observations**. Legacy/static/Kalman reaches 24.43% all-labelled within-5 m recall (4,598 supported matches; mean 2.140 m). Selected/static reaches 28.25% (5,328; 2.184 m); selected/propagated reaches 33.41% (6,295; 2.161 m). Raw/propagated reaches 39.89% (7,582; 2.233 m). The held-out game's 57.99% confirmed coverage does not represent all games. The missing-support denominator remains explicit; the two-metre conditional means must not conceal missing people.

### Appearance helps anonymous tracking but does not verify player names

A locally cached pretrained ResNet was fine-tuned on training-game torso crops for 43 epochs. The selected checkpoint is `person_embedding/best.pt`, SHA256 `3fb842bd8be3b11b47afe9cf924f383c5db00558794b8a73489794e265ecca11`. Validation alone selected the checkpoint and spatial/cosine association gates. Retrospective labelled-crop held-out retrieval rises from 83/260 to 198/260 across clips and 48/269 to 150/269 between track thirds. This uses a labelled gallery within each match/half/team and is **not causal association or named identity accuracy**.

The actual causal association test uses current/past detected crops without labelled IDs or team labels. Over 18 clips, 2,700 frames and 39,437 reference people, anonymous identity F1 rises **51.22% to 53.38%**; switches fall 1,820 to 1,740 over 31,084/31,087 matched transitions. Precision changes 89.54% to 89.45%, recall 79.846% to 79.854%. This remains weak for assigning all 22 names. A private Arsenal appearance run is being generated from the existing cached detections; production exports/checkpoints remain unchanged.

### Hidden predictions are sensitive to perception errors

`cv.audit_hidden_sensitivity` evaluated the previously validation-selected v4 models on DFL match J03WMX, two halves, 1 Hz, with fixed simulated broadcast visibility and **91,165 identical common hidden player-timestamps per condition**. Exact roster, identity, roles, ball, possession and phase are supplied except for the explicitly corrupted component. Hidden reference coordinates score only. Live features use past/current available measurements; replay may use future visible measurements. Synthetic corruption magnitudes were not fitted to EPL errors.

| Input condition | Live mean error | Replay mean error | Live nominal-80% coverage | Replay nominal-80% coverage |
| --- | ---: | ---: | ---: | ---: |
| Exact visible measurements | 8.28 m | 7.05 m | 81.20% | 80.75% |
| Independent visible error, intended mean 2 m | 9.38 m | 7.78 m | 72.35% | 72.86% |
| Visible camera offset +5 m along x | 9.02 m | 8.30 m | 72.25% | 63.29% |
| Same-team identity swaps in 10% of frames | 8.82 m | 7.50 m | 79.22% | 78.44% |
| Available measurements delayed 1 s | 8.41 m | 7.20 m | 75.67% | 76.50% |
| Ball offset +5 m along x | 8.37 m | 7.13 m | 80.73% | 80.22% |

The exact-input mean differs from the previous full benchmark because this comparison uses the common hidden intersection including the delayed-visibility condition. On 90,546 common samples with a previous sighting, clean live/replay mean is 8.25/7.03 m versus last-observed 16.94 m; 619 never-seen samples are excluded from both paired baselines. Age groups, medians and p90 are retained in the artifact. The large coverage loss under calibration error means clean-tracking uncertainty cannot be presented as calibrated EPL confidence.

A bounded CPU robustness experiment has started: five training matches, one validation match, separate live/replay models, clean-control and noisy-input variants with the same maximum update budget. Noise combines independent visible-foot error with 60-second coherent camera offsets; no hidden measurements are exposed. Validation selection requires lower average clean/noisy error and at most 0.3 m clean degradation. Checkpoints stay private; failed validation candidates will not proceed to test. This is an experiment, not a claimed improvement.

### Team-colour evidence and manual landmark provenance

The now-complete generic validation colour diagnostic has 13,662 matched outfield crops: all-person clustering 88.22%, label-aware role exclusion 94.28%, no missing cache timestamps. On held-out crops, the continued detector's actual predicted-player exclusion scores 5,175/5,386 (96.08%) versus 5,061/5,386 (93.97%) on the exact same subset without exclusion. Seven matched true-player crops were excluded by its role prediction. This is team-colour clustering after evaluation-only two-way permutation, not pitch-side assignment, names or hidden accuracy.

Three assistant-provisional landmark clicks on the actually viewed Arsenal 1800-second frame were scored against unchanged CCBV, giving approximately 2.67, 3.50 and 4.74 m errors. They have explicit unverified-click/orientation notes and `assistant_provisional` provenance; **they are not independent human ground truth or training labels**. The user can confirm or replace them through the private annotation page. No user CSV has yet been supplied.


## Regression checks, robustness and presentation (18:26 UTC)

The updated full automated suite actually completed: **80 passed**, 250 existing warnings, 108.99 s. Frontend webpack production build passed after the first presentation corrections; final lint/build checks will include the subsequent missing-radius and evidence-label changes. These checks do not replace accuracy evaluation.

### Robust-input training result

All four bounded CPU fine-tuning experiments finished. Clean-control and noisy-input variants use equal-sized training corpora, the same maximum optimiser budget, seeds and stopping policy; actual early stopping yielded 15/16 live epochs and 15/15 replay epochs. Both variants calibrate their radii on the identical combined clean/noisy validation distribution. The baseline's previous calibration was clean-input, so part of the coverage difference is recalibration rather than learning.

The validation-selected noisy-input candidate was evaluated without test fitting on the same **91,165 common hidden samples** and controlled stresses described above:

| Input condition | Live clean-control / robust mean | Replay clean-control / robust mean | Robust live / replay nominal-80% coverage |
| --- | ---: | ---: | ---: |
| Exact visible measurements | 8.21 / 8.33 m | 7.01 / 7.04 m | 81.9% / 83.5% |
| Independent visible error, intended mean 2 m | 9.30 / 8.69 m | 7.74 / 7.46 m | 80.5% / 80.6% |
| +5 m camera offset | 8.95 / 9.01 m | 8.25 / 8.24 m | 75.0% / 67.0% |
| Same-team identity swaps | 8.73 / 8.64 m | 7.48 / 7.42 m | 81.1% / 81.6% |
| Measurement delay 1 s | 8.35 / 8.45 m | 7.18 / 7.20 m | 78.0% / 80.9% |
| +5 m ball offset | 8.29 / 8.42 m | 7.07 / 7.10 m | 81.6% / 83.2% |

This supports a limited improvement against random foot noise, with a clean-input trade-off. Systematic camera error still gives poor replay coverage. Neither variant is promoted as a reliable EPL model. Commands actually executed: `cv.train_hidden_robustness --deadline 1791756571`, then `cv.audit_hidden_sensitivity` for each selected clean-control/robust live/replay pair with separate output directories. Configuration, feature cache hashes, exact checkpoints, histories and results are retained in `hidden_robustness/`.

The private appearance-qualified Arsenal comparison holds all detections, camera, role aliases and measured coordinates constant across f0, original v4 and robust v4. All have 4,255 measured positions and **376 measured-to-measured steps above 10 m/s among 3,623 transitions**, maximum 51.93 m/s. Off-camera retraining cannot repair those input discontinuities. On the same 281 unverified role-reappearance proxies, mean changes 17.53 → 17.39 → 16.60 m; this is consistency with guessed role references, not hidden ground-truth accuracy. Video-radius scale 2.67 remains explicitly transferred/unvalidated.

A numeric CLI fold was also diagnosed: `--recon-fold 0` arrived as a string and searched for `hidden_live_0.pt`, causing optional diagnostics to fail and reconstruction to fall back. Numeric folds are now normalised to `f0`-style tags before loading. An actual full numeric-0 export equals the explicit-f0 export exactly and records the real f0 model. The earlier private appearance attempt with this incorrect argument failed its camera-extra diagnostic and is excluded from comparisons; it was rebuilt correctly.

### Real cache-concurrency and file-refresh checks

A six-thread benchmark executed the real SkillCorner 4039 second-half analysis. Before the change, six concurrent cold calls constructed six distinct `MatchIntelligence` objects. After per-source locking, they construct one shared object. Wall times were 1.005 and 1.153 s under concurrent workloads: **no latency improvement is claimed from those timings**; repeated computation and duplicate objects are eliminated. Different sources remain independently loadable.

Cache identity now canonicalises paths and includes file mtime/size/inode. A changed export invalidates cached analysis without rerunning YOLO. Failed or repeatedly changing files do not become stale successful entries. Three focused tests passed, and the 80-test full suite includes them. The backend was restarted with the preserved Arsenal source and deterministic local narration; active files/checkpoints are unchanged.

### Actual browser rendering and controls

A separate headless Chrome profile uses existing local Chrome, not the user's normal browser session. The Arsenal studio was loaded and its screenshots actually inspected. At 1x playback over 20.0003 s under concurrent training/evaluation, it recorded **1,126 animation callbacks (56.30 Hz), 1,124 player-position changes (56.20 Hz), median 16.7 ms, p90 16.8 ms, maximum 50.1 ms**, no long tasks and no new resource requests. These are headless development-browser presentation measurements, not verified physical-display FPS, source sampling or positional accuracy.

The first screenshot showed the score wrapping vertically and a confident low-block label despite unverified CV geometry. The header now wraps its secondary fields while keeping the score on one line, labels the short Arsenal source `CLIP REPLAY`, uses the actual source provider, and calls projected points measurements. Players, including goalkeepers, use just the two team colours. CV tactical geometry and evidence accuracy are labelled unverified/N/A; estimated metrics and event records have distinct evidence tags.

The preserved Arsenal archive has **no prediction-radius arrays**, so its prediction-zone button now explicitly shows disabled `Prediction zones N/A`; no uncertainty circles are invented. The actual 370-second Walcott explanation opens in an absolute overlay on the match view, closes correctly, uses native lists/tables, and contains no raw JSON or embedded video. A route test initially assumed half `h1`; the catalog's actual part is `clip`, and the corrected route was checked successfully. Further zone-show/hide checks use a source that actually includes radii.

The annotation page's actual frame selection, canvas click, undo, native table and CSV download were checked in the isolated profile, with no JavaScript exceptions in the successful run. Its fabricated smoke-test click is explicitly labelled **UI SMOKE TEST ONLY** in `browser/test_downloads/`, never scored as a landmark or used for training. An audit-script import and frame-option assumption were corrected before the successful check. User-made annotations remain absent.

The local GSR inventory was reconfirmed: **57 training clips** (games 4/6/9) and **58 validation-package clips** (games 2/3/5), total 115 clips/86,250 labelled image frames. The custom evaluation uses validation games 3/5 and held-out game 2; the provider task page advertises a different validation count and additional test clips, which are not present locally. No new independent games were discovered or downloaded.

## Rejected refinements and private playback diagnosis (18:55 UTC)

### Current-image paint alignment does not establish a held-out improvement

The existing bounded `cv.refine_calibration` algorithm was evaluated against the frozen propagated camera, using current images and stock-detector boxes only. Labelled visible feet were read after fixing each candidate camera, for scoring alone. Failed paint/physical checks retain the original camera and support is identical. The validation gate required at least 0.1 m paired mean reduction with non-increasing p90.

| Split | Same labelled feet | Supported / total frames | Original mean / p90 | Refined mean / p90 | Accepted refinements |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 8,363 | 483 / 1,000 | 2.166 / 4.902 m | 1.963 / 4.634 m | 276 |
| Held-out game | 5,167 | 359 / 450 | 1.873 / 3.949 m | 1.891 / 4.008 m | 27 |

The validation-selected method **failed to generalise**. Among the 337 feet in actually changed held-out frames, mean worsened 1.245 → 1.527 m, although 16 of 27 changed frames improved and 11 worsened. A lower image-paint distance is therefore not a reliable positional accuracy gate. No parameters were changed using these test failures, and no active export was refined. Executed `cv.audit_paint_refinement --part val`, then unchanged `--part test`; readable native tables are in `paint_refinement/{val,test}.html`.

### Ground-based affine tracker compensation was rejected on validation

The same 20,453 labelled people across eight validation clips/1,200 frames were compared under native sparse optical flow, causal ground-paint affine compensation and no compensation. Ground flow reads past/current paint only; unsupported or non-affine motion falls back to identity. Two synthetic tests verify transform direction and unsupported-image fallback.

| Motion compensation | Precision | Recall | Anonymous identity F1 | Switches / transitions |
| --- | ---: | ---: | ---: | ---: |
| Existing sparse optical flow | 87.10% | 74.93% | 46.67% | 702 / 15,134 |
| Ground-paint affine | 87.36% | 74.69% | 43.72% | 967 / 15,084 |
| None | 91.02% | 66.82% | 33.69% | 1,666 / 13,475 |

The native baseline exactly reproduces the earlier selected-tracker aggregate. Neither alternative passes the predeclared gate (identity F1 gain ≥0.5 pp; precision loss ≤0.5 pp; recall loss ≤2 pp). **Held-out testing was deliberately skipped**. This rules out the tested affine implementation as a supported fix; it does not establish that all possible camera compensation is ineffective. Executed `cv.audit_ground_tracking --part val`; selection is explicitly null.

### Missing measurements are not evidence that players are outside the camera

`cv.audit_broadcast_coverage` inventories all 3,001 cached 5 Hz timestamps over the 600-second Arsenal clip; the identical camera flags mark 1,566 timestamps as supported main-camera views. Confirmed projected person observations are 17,443 legacy / 20,177 selected / 20,155 appearance, with medians 12 / 14 / 14 per supported frame. These counts include officials, possible false positives and anonymous people: they are **not named-player recall**.

At video second **1800**, legacy has 2 projected tracks and 17 unassigned boxes (9 detector scores ≥0.5); selected and appearance each have **5 projected tracks and 14 unassigned boxes**, including **6 scores ≥0.5**. Therefore these misses cannot all be explained by low detector confidence or off-camera visibility. Confirmation/association and projection rejection remain separate losses. No independent EPL visible-person inventory is available to score recall. Records missing confidence are counted separately, never given an invented score.

The native HTML coverage table and time series are `broadcast_coverage/review.html`; the actual plot was inspected. The first script executions exposed different cache record schemas (unsupported projections lack confidence, unsupported cameras lack class); those were handled explicitly and the completed output verified. No model or cache was rewritten.

Two **private 30-second review movies** cover video 1785–1815 at 10 display frames/s, 300 frames each, 1440×620:

- `broadcast/anonymous_review.mp4`: current/past 5 Hz measurements only, at most 0.25 s old. White anonymous confirmed tracks and amber unassigned boxes; no hidden or named positions. Source seek was checked against the cached frame (1785.003 s, PSNR 43.98 dB including compression). Slightly out-of-pitch points remain drawn; there is no forced outward projection.
- `broadcast/robust_role_review.mp4`: retrospective robust-model candidate, role-assigned numbers explicitly unverified; solid measurements versus ring estimates. This does not imply calibrated hidden EPL accuracy.

Both 1800-second posters were actually inspected. The anonymous poster shows missing confirmation for visibly detected red-shirted people, while the role view fills a roster using uncertain identities and hidden estimates. Use the anonymous video to review perception first. These movies stay local and are never embedded in the application. Executed `cv.private_tracking_review` and `cv.private_review` with explicit private paths, `--start 345 --duration 30 --fps 10`; writer frame count, dimensions and FPS were read back.

The image-only physical-camera experiment is still running. At epoch 66 its best fixed validation-subset mean is 5.202 m, p90 12.178 m over 3,595 accepted feet/208 of 250 frames; this is progress, not an accepted full evaluation. A bounded continuation with stronger supervised world-position loss is queued into a separate private directory, using the same training/validation split and no test labels. The original training source was archived before the optional resume/loss-weight CLI was added. No physical-camera checkpoint is promoted.

## Visibility semantics and causal safeguards (19:06 UTC)

### Geometrically hidden is different from unobserved

`cv.audit_hidden_visibility` explicitly separated the frozen full hidden set, **97,116 samples** in both halves of held-out DFL match J03WMX. Camera polygons and hidden truth enter evaluation grouping only, not model inputs. Simulated SkillCorner visibility is regenerated with the identical seed and asserted equal to the earlier `sc_camera` mask.

| Reference condition | Samples | Baseline live mean | Baseline replay mean | Baseline live / replay prediction inside camera |
| --- | ---: | ---: | ---: | ---: |
| Truly outside a known camera footprint | 47,181 | 6.82 m | 5.35 m | 4.11% / 2.83% |
| Inside footprint, but observation missed | 809 | 3.71 m | 1.64 m | 88.50% / 95.30% |
| Camera footprint unavailable / cut | 49,126 | 8.91 m | 8.05 m | N/A |

The robust-input model gives live/replay means 6.95/5.39 m outside, 4.11/1.72 m inside-but-missed, and 8.91/8.01 m when the camera is unavailable. Its nominal-80% coverage is 81.85/84.42%, 85.41/96.04% and 83.64/84.20% respectively. This is clean oracle-visible tracking with exact identities/context, not actual Arsenal accuracy. For truly outside players, occasional predicted camera overlap is a genuine model error in this controlled setup; for missed in-camera players, inside-camera predictions are usually appropriate. A hard outside-polygon rule would turn correct estimates into errors for that second group. Half of unavailable samples occur without a usable footprint, so "off-camera" was also an imprecise name for the aggregate.

All metre metrics and native tables are in `hidden_visibility/review.html`; no point was moved outside a camera, no failure was hidden and no model was fitted on these test groups. One new membership test checks that missing cameras and nonfinite positions are not misclassified as known inside points.

### Simulated camera initialisation read future balls

The `k=<n>` benchmark's camera centre previously initialised from the mean of the first five ball positions. This is a small but actual future-information dependency in simulated early visibility. It now initialises from the current first ball, or midfield if missing. Prefix and future-mutation tests pass; all subsequent camera pan updates use only current/past balls. Existing `sc_camera` stress/visibility masks are unchanged, so the reported same-mask held-out comparisons remain valid. Older `k`-mask experiments retain their original conditions and are not silently relabelled as regenerated causal data.

The exact pre-change `real_benchmark.py`, `ml.py` and robustness-training source plus hashes were archived in `reconstruction_source_before_camera_init/`. Feature-cache keys now include complete source hashes of masking, dataset parsing, the prior and the feature module; changing helper implementations can no longer reuse an incompatible feature cache merely because the wrapper's source stayed unchanged. Existing checkpoints and earlier result artifacts remain untouched. Focused causal/noise checks actually passed **11 tests**; the full suite immediately before these additions passed **82 tests**, 250 existing warnings, 126.63 s. A later full check will include the new tests.

### Tactical time windows are explicit

Actual browser screenshots exposed an apparent count mismatch: the header counts the current frame, while the panel averages a minute. This is an aggregation difference, not evidence that both show the same timestep. The header now says "Now"; the table says "Avg. measured / inferred" with one decimal. A short note distinguishes minute averages from current-frame counts, and expandable time-window text explains three-minute smoothed metrics and five-minute shape windows. Actual headless controls still passed; native tables, two team colours, no embedded broadcast video and the N/A zone state remain intact. Frontend lint and webpack build passed after this change, and the rendered text was checked directly.


## Camera discontinuities and fresh causal-mask evaluation (19:52 UTC)

### A camera error can teleport a correctly tracked person

The private anonymous continuity audit holds current detections and qualified IDs fixed. At video **1444.2 → 1444.4 seconds**, track 17 moves only **10.86 image pixels**, but its projected pitch position moves **82.45 metres in 0.2 seconds** (412.27 m/s). Reprojecting the same current foot through the old/new camera matrices accounts for **82.01 metres** of that change. The actual adjacent broadcast images were inspected: the main view and people are nearly unchanged. Both camera estimates have high source confidence and pass grass/scale plausibility. This is evidence of a camera-origin failure before hidden-player inference. Incorrect pose correspondence versus source-video clock alignment remains unresolved; no independent landmarks establish the exact answer. Across 1,570 matrices, no simple image-axis reflection was found.

At 5 Hz the appearance branch has 18,318 strict same-track, same-shot, consecutive transitions. Median speed is 4.44 m/s and 5,026 exceed 10 m/s. On the same-camera-bin subset, 559/11,100 exceed that threshold; when a 2 Hz camera bin changes, 4,467/7,218 exceed it. These flags diagnose discontinuity, not football speed ground truth. The source video length and matrix count agree with 2 Hz sampling, but the matrix source names a different HQ video, so semantic/clock correctness is still unverified. Native tables and the actually viewed worst-pair image are in `anonymous_continuity/review.html`.

`Calibration.at` now rejects samples from a different supported-camera interval by default; explicit `allow_cross_shot=True` reproduces legacy behaviour. Nine of 1,579 cached supported main images previously borrowed a matrix from a neighbouring non-main shot, affecting 98 selected-cache person measurements. The guard rejects those nine; it cannot repair a wrong matrix whose nominal timestamp is still inside the main shot. Four focused camera boundary/interval tests passed. Active exports remain preserved.

### Frozen temporal camera review helps one case but is insufficient

`cv.camera_continuity` uses current/past ground-paint flow and the earlier validation-selected **three-second** propagation / **ten-metre** anchor-agreement thresholds. External CCBV anchors and shot labels remain retrospective; this is not a live broadcast camera model. `cv.reproject_camera_review` reprojects the identical appearance cache, with unchanged boxes and IDs, into a private directory. Unsupported cameras/positions are explicit; no estimates are moved outside the camera and no named roles or hidden predictions enter this experiment.

On **18,175 exact common transitions**, speed flags above 10 m/s change **4,937 → 4,856**; p90 changes **21.855 → 21.375 m/s**, maximum **412.27 → 343.93 m/s**. The inspected track-17 example changes to **0.502 m/s**, consistent with its small image motion. Person measurement support drops **20,155 → 20,001** (154 fewer), so this is not a complete solution. A separate integer-anchor-only variant falls to **8,851** measurements and is rejected for coverage regression. No candidate is promoted and no active match export is rewritten. Two camera continuity tests passed; a default rerun produces byte-identical 13,073,193-byte detection output.

The private `camera_continuity/review.html` includes before/after native tables and **eight-second movies**, starting at video 1440. Both contain 80 frames at 10 display Hz / 1440×620; readback and the actual 1444.4-second comparison poster were checked. The source matrices are explicitly declared normalized-image → pitch-metres. The manual scorer can evaluate a frozen candidate at exact cached timestamps, without fitting or nearest-time substitution. Four manual-scoring tests passed. The three assistant-provisional 1800-second landmarks give unchanged errors (2.67/3.50/4.74 m), as expected where the original anchor is retained; these are not independent human ground truth.

### Regenerated training after the causal-mask fix

The camera-initialization change affects **114 visibility entries across 3,198,062 checked array cells**, all within the first eight seconds of the fourteen local DFL halves. This small scope is recorded; no accuracy gain is inferred from it. Clean-control and robust-input models were retrained with new feature-source hashes, in `hidden_robustness_causal_masks/`, rather than overwriting old experiments. Training finished (live 15/16 epochs, replay 15/15), and both frozen candidates were evaluated with the same 91,165 common held-out samples. Actual commands/timestamps/checkpoints are in `evaluation_commands.json`.

| Condition | Live clean / robust mean | Replay clean / robust mean | Robust live / replay nominal-80% coverage |
| --- | ---: | ---: | ---: |
| Exact visible measurements | 8.21 / 8.33 m | 7.01 / 7.04 m | 82.0% / 83.4% |
| Independent visible error, intended mean 2 m | 9.30 / 8.69 m | 7.74 / 7.46 m | 80.6% / 80.5% |
| +5 m camera offset | 8.95 / 9.00 m | 8.25 / 8.24 m | 75.0% / 66.9% |
| Same-team identity swaps | 8.73 / 8.64 m | 7.48 / 7.42 m | 81.1% / 81.6% |
| Measurement delay 1 s | 8.35 / 8.45 m | 7.18 / 7.20 m | 78.1% / 80.8% |
| +5 m ball offset | 8.29 / 8.41 m | 7.07 / 7.10 m | 81.6% / 83.1% |

The conclusion remains limited robustness to independent measurement noise with a clean-input trade-off, not correction of systematic camera error. Coverage varies with the fitted radius; current clean-control and robust candidates use the same validation/calibration conditions. Earlier experiments retain their original masks and calibration rather than being relabelled. No checkpoint is promoted.

### Physical-camera training: first run fails the gate

The bounded MPS run completed **130 epochs over 7,200.28 seconds**. Full validation has 1,000 frames, 989 supported, 16,943 labelled feet: mean **4.094 m**, median **2.399 m**, p90 **10.120 m**. No image-only quality operating point satisfies the frozen mean ≤3 m, p90 ≤5 m, coverage ≥40% gate; the existing inside=.75/scale=.25 point supports 814 frames/14,041 feet with mean 4.641 m and p90 11.041 m. Held-out testing is **not run**, and the model is not promoted. A separate 1.8-hour continuation with stronger training-only world-position loss is running, with the same split/gate and a bounded deadline.

The full suite after boundary/continuity/causal-mask changes passed **90 tests**, 250 existing warnings, 141.59 seconds. Two additional candidate-scoring tests subsequently passed in the four-test focused check; a final full check will include them.


## Failure gallery and constrained fallback hypothesis (19:55 UTC)

`cv.render_camera_discontinuities` generated ten distinct-time before/after image pairs with native tables, selected by the largest original anonymous projected jumps. In these selected cases, the temporal candidate corrects six to below 10 m/s; four remain 343.93/320.48/226.20/193.06 m/s. All four report weak spatial support from ground homography. The actual 1499.2/1499.4 and 2001.2/2001.4 images were viewed: large pitch jumps accompany small image motion. The gallery is a symptom-selected review aid, not an unbiased improvement metric. It contains no named roles, hidden model or ground truth. Command: `cv.render_camera_discontinuities`.

A separate bounded similarity-transform fallback is now being evaluated on GSR validation only. It uses the same current/past white-ground points and forward/backward consistency; it allows less spatial area only for a four-parameter transform with bounded scale/rotation/translation. This approximation can fail under perspective changes. The existing homography/default inference remains unchanged. A predeclared gate requires at least 48.3% validation support, mean ≤2.216 m and p90 ≤5.002 m versus the original propagated-camera result. Test is allowed only if that validation gate passes; thresholds will not be tuned using Arsenal or held-out errors.

The initial synthetic fallback fixture had fewer usable corners than required and failed (1 passed, 1 failed); it was corrected to a narrow region with sufficient corners, keeping the support requirements unchanged. The meaningful translation/direction and no-ground-support tests then passed (2 tests). Actual command: `cv.audit_camera_motion --part val --similarity-fallback --out cv/out/audit8_20261011/camera_similarity`.

Latest actual full regression: **92 passed**, 250 warnings, 98.05 seconds. Frontend lint and webpack production build passed after adding CV highlights/statistics/formation/name provenance labels. The rendered highlights, accuracy and player panels were checked in the isolated Chrome profile, and screenshots actually inspected. The combined dashboard now has ten native tables, no raw JSON or embedded videos, no broken images and no browser exceptions. Training continuation is still running independently.

## Dense camera coverage and foreground motion (20:46 UTC)

### Correcting the fallback hypothesis

The four remaining large temporal-camera jumps were labelled “weak spatial support”, but inspection of the actual diagnostics showed adequate spatial extent and low inlier fractions (about .38–.55). The initial narrow-ground-area explanation was not supported. A constrained similarity fallback produced identical validation summaries to strict homography on 6,000 dense frames; held-out support increased by only four frames/60 labelled feet. It did not fix the four private Arsenal jumps. The original inference path stays unchanged.

### Excluding moving foreground from camera flow

`cv.camera_foreground` masks current and previous **detected person boxes** out of ground-feature selection. It keeps the existing homography requirements: 16 inliers, .6 inlier fraction, .015 spatial extent, forward/backward consistency and bounded propagation. Actual detections enter this flow; reference boxes and future images do not. Two focused tests cover resized foreground masks and background translation. External camera anchors/shot annotations still make the private Arsenal path retrospective.

Camera-only validation on eight clips/1,200 5 Hz frames supports 603 → 635 frames and 9,918 → 10,493 labelled feet; mean changes 1.499 → 1.519 m, p90 2.752 → 2.805 m, within-5 m recall over all 20,169 pitch-labelled people 46.864% → 49.507%. This satisfies the existing validation tolerances (mean regression ≤.05 m, p90 ≤.1 m, nondecreasing coverage and negligible recall loss), without threshold tuning on test.

Held-out camera-only evaluation covers 18 clips/2,700 frames from one game. Support increases 2,006 → 2,069 frames and 28,720 → 29,605 feet, mean 1.868 → 1.810 m, p90 3.982 → 3.774 m. Within-5 m recall over **37,968 pitch-labelled** people improves 70.304% → 72.880%. On the **28,314 common** feet, mean changes 1.820 → 1.798 m and p90 3.816 → 3.770 m: modest placement improvement as well as additional support. This is visible-player GSR evidence, not all-player EPL accuracy. A fresh run with image, detector-box and reference-content cache keys reproduces validation/test summaries exactly; `dense_foreground_repro/reproduction.json` records the comparison. Camera estimation is independent of scoring labels.

### Actual detector/tracker integration

The same selected BOTSORT and cached detector were rerun using those frozen dense camera matrices. Image matching and predictions are identical between camera conditions. Conditional errors exclude unmatched, unsupported or missing pitch references, while recall uses **all labelled visible people**, including referees.

| Split / sample conditions | Camera | Scored matches | Mean | Median | P90 | Within 5 m / all GT |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Validation: 8 clips, 1,200 frames, 20,453 GT people | Strict | 7,264 | 1.474 m | .966 m | 2.725 m | 34.054% |
| Same validation | Foreground masked | 7,512 | 1.491 m | .973 m | 2.733 m | 35.154% |
| Test: 18 clips, 2,700 frames, 39,437 GT people | Strict | 23,303 | 1.933 m | 1.177 m | 4.076 m | 54.857% |
| Same test | Foreground masked | 23,906 | 1.901 m | 1.158 m | 3.932 m | 56.444% |

The integration gate carries the same previously used camera tolerances; its result was recorded after validation, not presented as an independently pre-registered new experiment. Test followed the passing validation check. The all-person denominator differs from the camera-only pitch-labelled denominator; these percentages are intentionally not combined. Earlier sparse 1,000/450-frame results also have different conditions and are not a direct before/after comparison.

### Complete private broadcast continuity

On the same 18,175 exact anonymous consecutive transitions, original → temporal → foreground-masked camera results are: >10 m/s flags **4,937 → 4,856 → 4,712**, p90 **21.855 → 21.375 → 20.549 m/s**, maximum **412.27 → 343.93 → 147.97 m/s**. Supported person observations are 20,155 / 20,001 / 20,048. The candidate fixes nine of ten originally worst, symptom-selected jump cases to below 10 m/s; that selected proportion is not an unbiased accuracy metric. Major residual jumps remain, so no camera or match export is promoted. The actual original 1444.4-second video clock shows about 24:04 and 1800 shows 29:59; this does not support an assumed ten-second global clock offset. Exact HQ-video calibration correspondence still needs independent landmark verification.

Commands actually executed: `cv.audit_dense_camera` foreground validation/test and fresh reproduction; `cv.audit_visible_pipeline --dense-camera-source ... --part val/test --out .../visible_foreground`; `cv.reproject_camera_review` foreground private reprojection; `cv.render_camera_discontinuities --candidate ... --out ...`; `cv.audit_foreground_results`; `cv.private_tracking_review --start 0 --duration 8 --fps 10`. Readable results: `camera_foreground_arsenal/review.html`; private original/candidate images and movies remain local.

## Raw tracking time, generalisation and data terms (20:46 UTC)

The raw DFL frame/UTC audit read all seven XML files and fourteen halves. Every retained 1 Hz ball-frame count agrees with its cache; no retained duplicate frame number, frame-number gap or UTC gap was found. This establishes that corpus cadence, not broadcast synchronisation or event alignment. Raw source hashes and first/last UTC are in `dfl_input_audit/raw_timing.json`, with native tables in its HTML page.

The importer previously ignored requested sampling rate in its cache name and silently used `25//hz_out`, yielding 12.5 Hz when asked for 10 Hz. It now accepts integer divisors of 25, uses rate-specific caches, records actual elapsed source-frame timestamps and checks cache cadence. The 1 Hz reconstruction adapter rejects incompatible rates rather than labelling them 1 Hz. New raw-half orientation uses the first retained home keeper rather than a future half-average; it agrees with the original direction in all fourteen existing halves. A fresh 5 Hz J03WMX import has 14,142/15,052 half samples; all 2,829/3,011 common integer-second positions exactly match the preserved 1 Hz cache. No original raw file or legacy 1 Hz cache was overwritten. Four sampling/orientation tests and focused causal/provenance checks passed.

All seven existing v3 leave-one-match-out checkpoint pairs were evaluated with their target match explicitly excluded from checkpoint training metadata. No retraining, selection or recalibration occurred in this multi-match audit. Equal-match macro mean live/replay errors: clean 8.513/7.087 m, independent 2 m visible noise 9.611/7.829 m, 5 m camera offset 9.234/8.260 m, identity swaps 9.097/7.651 m, 1 s delay 8.623/7.230 m and 5 m ball offset 8.564/7.107 m. Per-match sample counts/medians/p90 and uncertainty coverage are in `hidden_multimatch/review.html`; aggregate medians/p90 are not invented from fold summaries. Simulated masks, exact context/identities and synthetic corruptions limit transfer to real EPL. Existing checkpoint calibration uses training time tails rather than an independent external calibration set.

Provider terms were checked without downloading match data or contacting anyone. The [DFL dataset paper](https://www.nature.com/articles/s41597-025-04505-y) confirms seven German Bundesliga/2. Bundesliga matches under CC BY 4.0; licensed footage is excluded. [Hudl/StatsBomb metadata](https://github.com/hudl/open-data) lists EPL 2015/16 and 2003/04, with no listed EPL 360 availability. These are event/lineup resources, not continuous hidden-position truth. The actual five-page [custom licence](https://github.com/hudl/open-data/blob/master/LICENSE.pdf), updated 8 September 2023, restricts redistribution/commercial exploitation and requires attribution/logo for published analysis. Terms clarification is needed before a public/commercial hackathon integration. No StatsBomb match dataset was imported. Licence-document hash and checked primary URLs are in `licensing_checks.json`.

Latest completed full regression before the newest loader/cache tests: **101 passed**, 250 existing warnings, 113.36 seconds. New dense-input/cache tests passed in focused checks; the final full run will include them. Frontend lint and production build actually passed after the last provenance-label edits.

## Tactical geometry and confidence stress (20:50 UTC)

`cv.audit_tactical_sensitivity` evaluated the frozen clean-control and robust-input candidates on the same J03WMX halves/masks. Each condition has **11,457 frame-team samples** with at least seven active outfield players and at least one inferred player. Reference geometry is scored only after inference; no hidden truth enters features. Production width/depth/four-lowest-outfield defensive-line calculations and radius sensitivity bounds are reused. Raw hidden estimates are used without a speed limiter or event cue, and live/replay remain separate.

| Clean-control input / mode | Width mean absolute error | Depth mean absolute error | Defensive-line mean absolute error | Defensive-line sensitivity-envelope coverage |
| --- | ---: | ---: | ---: | ---: |
| Exact visible, live | 5.97 m | 4.46 m | 2.85 m | 92.1% |
| 2 m visible noise, live | 6.33 m | 5.11 m | 3.12 m | 89.9% |
| 5 m camera offset, live | 6.03 m | 4.59 m | 4.72 m | 72.5% |
| Exact visible, replay | 5.01 m | 4.14 m | 2.58 m | 91.5% |
| 2 m visible noise, replay | 5.35 m | 4.69 m | 2.86 m | 88.3% |
| 5 m camera offset, replay | 5.05 m | 4.26 m | 4.92 m | 63.4% |

Robust training reduces noisy-input live depth error 5.11 → 4.82 m and replay depth 4.69 → 4.51 m, with clean-input trade-offs; it does not fix systematic camera offsets. Robust defensive-line envelope coverage under the 5 m offset is 70.4% live / 62.2% replay. Full mean/median/p90/count/envelope-span tables are in `tactical_sensitivity/clean_control/review.html` and `.../robust/review.html`.

These are geometric sensitivity envelopes, **not nominal-80% joint confidence intervals**. Observed-player radii are zero, deliberately exposing the existing limitation that perception error is excluded. High clean coverage can result from broad envelopes, not high coordinate precision. A calculated 2 m tactical change can be smaller than reconstruction geometry error; the current threshold detects a calculated change, not statistical significance or coaching intent. Comparative-window uncertainty is still missing. Existing CV narratives already mark perception unverified/tactical interpretation tentative; no generated tactical intention or causal outcome was treated as reference truth.

The foreground pipeline's pooled validation-fit visible radius is **1.861 m**, over 7,512 supported IoU matches. Held-out coverage is **71.78%**, over 23,906 matches, against nominal 80%. Predicted/observed within-5 m frequencies are 95.70% / 93.11%. Thus this radius is not transferred to the EPL app. On 23,101 common supported detector/tracker matches, strict → foreground mean changes 1.911 → 1.894 m, p90 3.971 → 3.927 m; the new support remains separate from this paired comparison.

The raw-time audit totals **40,113 retained 1 Hz samples** across fourteen halves. Seven-match stress evaluation totals **620,625 common hidden player-timestamps per mode**, with per-match counts shown rather than silently pooling percentiles. The foreground private movie was read back (80 frames, 10 Hz, 1440×620) and its 1444.4-second poster actually viewed. The huge jump is corrected but some dots remain questionable against the painted pitch, so visual smoothness is not accepted as accuracy. A failure-gallery caption overlap was corrected using a separate header strip, regenerated, and the actual adjacent 2001.2/2001.4 images inspected.

The updated dashboard passed actual isolated Chrome checks: ten native tables, no raw JSON blocks, no embedded video, no broken images and no browser exceptions. Its first unprivileged browser attempt failed with a local-network sandbox restriction; the authorised isolated local check then passed. Latest full regression actually completed **104 passed**, 250 warnings, 145.79 seconds.

## Feature-input reproducibility guard (20:50 UTC)

Feature caches now additionally hash the **parsed DFL and camera-cache bytes actually consumed**, after ensuring those caches exist. Source/config hashes alone previously could reuse features after data arrays changed. This is input provenance, not proof that an old parsed cache agrees with its raw XML. The separate raw-time/source audit covers the checked local files. Hashing rejects files that change while being read. SkillCorner camera cache names now distinguish 1/2/5/10 Hz, and non-divisor/fractional rates fail instead of silently reusing the 1 Hz cache. Default 1 Hz arrays and active outputs are preserved; higher-rate SkillCorner inference is not claimed or evaluated.

Eight focused sampling/provenance/causal tests actually passed after these additions. The earlier 104-test full result remains valid for that earlier source state; the next full run will include the two new tests. No training gain is attributed to cache-key changes, and the completed candidates keep their original commands/input conditions.

## Expanded validation changes the camera decision (21:00 UTC)

The foreground candidate was checked, with unchanged parameters, on **all 40 available validation clips / 6,000 frames**. Existing stock CPU detection caches already covered these frames; **zero new detection frames were generated**. This is additional validation after the original eight-clip selection and held-out evaluation, not a fresh untouched external test. No thresholds were changed.

Camera-only strict → foreground support increases 2,642 → 2,769 frames and 45,591 → 47,865 labelled feet. Mean changes **2.129 → 2.201 m**, p90 **4.783 → 4.903 m**; within-5 m recall over 101,388 pitch-labelled people improves **40.767% → 42.642%**. Both mean and p90 regress beyond the original +.05/+ .1 m tolerances, so the expanded gate **fails**. On 45,514 common feet, mean changes 2.130 → 2.137 m and p90 4.785 → 4.795 m; new, harder support also contributes to the aggregate regression. The gate is not relaxed post hoc.

The actual selected detector/tracker integration on these same 6,000 frames / **102,075 all-person GT observations** gives scored matches 34,087 → 35,492, mean **2.133 → 2.224 m**, p90 **4.698 → 4.862 m**, within-5 m/all-GT **30.374% → 31.464%**. This also exceeds the quality tolerances. The earlier single-game held-out gains remain honestly recorded, but are insufficient to justify promotion. No production camera, tracker or export is replaced.

Commands actually executed: `cv.audit_dense_camera --part val --foreground-exclusion --all-validation --out .../dense_foreground_full_validation`, followed by `cv.audit_visible_pipeline --part val --dense-camera-source .../dense_foreground_full_validation/val.json --out .../visible_foreground_full_validation`. Native camera-only tables: `dense_foreground_full_validation/val.html`.

On 18,238 exact original/foreground-common broadcast transitions, 245 original >10 m/s flags become ≤10 and **eight new flags** appear. This pairwise set differs from the three-camera common set above. A new gallery ranked by the candidate's remaining failures shows ten actual residual discontinuities, including 1961.2/1961.4 and 1603.2/1603.4 seconds; the actual first pair was inspected and both images show the same midfield view. Candidate H/foot mapping is still wrong there. These four timestamps were added to the manual annotation page, now **18 frames**. Current-foot association does not prove that an anonymous ID belongs to a named player.

The annotation page now has an explicit Arsenal/Liverpool/N/A team selector, avoiding ambiguous free-text team labels. `cv.evaluate_manual_identity` can link an independently clicked foot to a current anonymous track, distinguish unconfirmed detections and reject ambiguous associations. It never fits identities, rewrites exports or uses pitch/formation to choose the person. Conflicting human shirt/team labels on one track require manual review. Three association tests passed, including close feet, missing detections and replay exclusion. An additional importer test prevents deliberately fabricated UI-smoke clicks being scored as human ground truth; **eight** focused manual tests passed. The real human CSV is still absent, so human identity and landmark scoring remain **not run**.


## Additional gates, identities and tactical windows (21:45 UTC)

The visible-foot anchor-agreement candidate was evaluated with the same frozen 10 m / 3 s policy on all forty validation clips. Its 2,765 supported frames / 47,764 feet have mean **2.302 m**, median 1.287 m and p90 **5.121 m**, worse than the foreground candidate's 2.201 / 4.903 m mean/p90. Its gate fails; held-out testing is **not run**. The CLI now rejects this experiment without a separate explicit `--out`, preventing it from overwriting the earlier foreground evidence. The negative command actually produced the expected `ValueError`; this is a checked guard, not a passed accuracy result.

The stronger world-loss physical-camera continuation completed **101 epochs / 5,431.64 s**, rather than consuming its entire allowed maximum. Full validation supports 989/1,000 images and 16,943 feet: mean **3.754 m**, median 2.260 m, p90 **9.001 m**. The frozen grass/scale filter supports 814 frames / 14,016 feet: mean **4.211 m**, p90 **9.761 m**, with 334 accepted frames having per-frame p90 above 5 m. No gate satisfies mean <=3 m, p90 <=5 m and coverage >=40%; held-out testing is **not run**. This improves its predecessor's validation mean 4.094 m but remains unreliable; neither physical camera is promoted.

`cv.tactical_window_metrics` additionally evaluates the production 60-second averages and five-minute before/after shape windows, with out-of-possession filtering for defensive-line comparisons. It does not confuse raw frame error with error of a change: a constant spatial bias can partly cancel between windows. There are **178 overlapping window-team comparisons** per metric/input condition, not 178 independent tactical events or intention labels. Subsequent windows are retrospective evaluation outputs; they never enter live position inference.

For the clean-control exact-visible condition, live/replay width-change mean absolute errors are **2.529/2.452 m**, depth-change errors **1.987/2.065 m**, and out-of-possession line-change errors **1.784/1.653 m**. At the existing 2 m change threshold, live width has 125 estimated changes: 99 agree in direction with a reference change >=2 m, 19 have reference magnitude below 2 m and seven have opposite direction. Replay has 124: 107 / 13 / four. These are calculation sensitivity checks, not verified tactical intent or causal consequence. Full native tables: `tactical_windows/clean_control/review.html` and `.../robust/review.html`. Two tests cover constant-bias cancellation and below-threshold/opposite-direction accounting. Comparative-window confidence is still not calibrated.

The jersey input audit finds **34,129 training / 26,257 validation / 9,710 test crops**, 933/691/275 tracks, and zero inconsistent number labels within the retained tracks. Validation has seven numbers absent from training (2, 37, 42, 43, 47, 59, 99; 2,873 crops); test has number 42 absent from training (553 crops). No legibility labels exist. The actually inspected private gallery contains readable wrong-number predictions as well as blurred/front-view crops; unreadability alone does not explain every failure. The older .95 fallback had only 44/52 correct validation tracks (84.6%), so a high vote score is not proof of reliable identity.

The new local-cached ResNet jersey continuation completed **nine full epochs**, seed 11026, learning rate .0001, batch 128, unchanged architecture/augmentation, with early stopping. Best validation epoch one has .95-threshold precision **61/92 = 66.3%**, coverage 13.3% of 691 tracks; .5 threshold is 166/557 = 29.8%. None of the fixed .5/.7/.85/.95 thresholds reaches the required 90% precision on >=20 tracks. No fallback is accepted, the held-out test is **not run**, and no jersey weights are promoted. This is a negative training result, not an improvement claim.

The private anonymous/unconfirmed movie now covers 1785–1815 seconds at **300 frames / 10 Hz / 1440×620** and was read back. Its 1800-second poster was actually inspected: five tracker-confirmed anonymous pitch points plus thirteen additional unconfirmed detections are shown through the same unverified camera. These can include officials, duplicate/false detections and unsupported named identities; the extra points are not additional verified lineup players. The first render failed on a legacy record missing `cls`; the producer now records known class/score and the renderer refuses to project unknown-class legacy records. The partial file was overwritten by the successful complete render. Four manual-association tests pass, including rejection of unknown classes. No broadcast video is embedded in the application.

Camera quality scale/grid metadata now explicitly identifies **canonical 1920×1080 reference pixels**, rather than silently implying that these are native 1280×720 video pixels. Compatibility keys/numerical thresholds remain unchanged. One scaling-metadata test passes; this change does not establish an accuracy gain.

The restarted API was actually exercised across **13 match catalog entries / 20 parts**, using **69 successful requests**: health/catalog/meta, first frame/state for each part, Arsenal moments/reveals/timeline, correct factual `scorer_name` Theo Walcott, and rejection of CV live/truth modes. Frontend HTTP returned 200. Two review-helper attempts failed (missing `PYTHONPATH`, then an incorrect expected scorer field), were corrected, and the final run completed; those helper failures are not hidden application failures. Native results are in `api_final_review.html`.

The full Python regression completed **121 passed**, 250 existing warnings, **48.37 seconds**. The one subsequently added training-input causality test also passed separately; a later final full run will include it. Frontend lint and webpack production build passed again after replacing the misleading synthetic “ball always observed” sentence with its actual fixed-context benchmark condition. These tests do not verify real EPL coordinates. The updated isolated Chrome dashboard actually has **sixteen native HTML tables**, zero broken images, zero raw JSON blocks, zero embedded videos and zero browser exceptions; its screenshot was inspected.

## Ball availability and hidden-context stress (21:45 UTC)

`cv.audit_ball_context` reads the same 3,001 private 5 Hz timestamp rows and cached stock predictions without changing inference. Of **1,579 labelled main-view samples**, 1,272 have a raw ball candidate; only **606** have any candidate score >=.2 and **91** have multiple candidates >=.2. Scores are uncalibrated detector outputs. After camera/field filtering, 1,256 samples have pitch ball candidates; the existing temporal filter accepts **928** of 2,379 projected candidates. Longest consecutive main-view run without an accepted ball sample is 24 samples (4.8 s at 5 Hz); camera cuts break this run. The 896 <=1-second accepted transitions have p90 apparent speed 30.57 m/s and maximum 44.83 m/s; no ball truth exists, so these are diagnostics, not ball accuracy or proof of false detections.

Before inferred-touch/event postprocessing, the mirrored 601-frame exporter context has **301 detection buckets, 290 held locations, nine future interpolations and one initial midfield fallback**. Accepted buckets average [s,s+1) detections, so they are retrospective rather than exact-time causal measurements. Touch carrying, nearest-player snapping, dead-ball and goal constraints subsequently alter this path. The original archived export has a finite ball in all 601 frames and no per-frame `bs` source label. Finite does not mean detected. Airborne balls additionally violate the ground-plane homography assumption. The 1800-second candidate image was inspected; the highest-score box lies near the visible ball while other candidates remain ambiguous. No annotation or corrected ball truth is fabricated.

A separate frozen-model stress uses a clean-control V4 candidate, one held-out DFL match/two halves and the **same 97,116 hidden player-timestamps** per mode/condition. Visible positions/IDs/roles and match phase remain exact; ball/possession are altered only as stated. A held ball uses the current/past exact sample every 30 seconds; unknown possession is -1 in both the prior and learned features. No training or model selection uses these stress results.

| Input condition | Live mean / p90 | Replay mean / p90 | Live / replay nominal-80% radius coverage |
| --- | ---: | ---: | ---: |
| Exact ball and possession | 7.79 / 16.77 m | 6.65 / 14.97 m | 86.00% / 85.80% |
| Ball observed every 30 s | 11.03 / 24.85 m | 9.06 / 21.51 m | 69.06% / 72.42% |
| Possession unknown | 8.82 / 19.73 m | 7.28 / 16.62 m | 82.04% / 83.64% |
| Midfield fallback + possession unknown | 12.78 / 27.37 m | 10.20 / 23.53 m | 62.85% / 67.49% |

The stale ball's mean displacement from reference is 25.17 m; the midfield fallback is 36.84 m. These are deliberately severe controlled examples, not fitted Arsenal error frequencies. They show that exact-context tracking benchmarks understate one real-video input limitation. Earlier six-condition stress has a smaller delayed-observation intersection (91,165 samples), so its means are not a before/after comparison here. Tests verify causal holds, no reference mutation and explicit fill/missing-run labels. The first diagnostic run failed on a NumPy scalar JSON write and a separate stress helper initially used an abbreviated match identifier/incorrect metric key; both were corrected and complete outputs inspected. No failed invocation is counted as a successful result.

Readable private pages: `ball_context/review.html` and `ball_conditioning/review.html`. Source hashes, model hashes, conditions and sample counts are retained with the outputs. No ball correction or match export is promoted.

## Targeted context continuation and expanded tracking validation (running)

The ball stress motivates an isolated, existing-architecture V4 continuation: exact-visible inputs with ball observations every 30 seconds and unknown possession in alternating 60-second blocks. It has **five training matches / one validation / one excluded evaluation match**, fixed masks/seeds, 10-second training and 5-second validation stride. Hidden reference positions enter training targets and validation scoring only. Equal-size clean-control and augmented corpora use the same optimiser/update limits; validation accepts only improved mean clean/corrupted error and clean regression <=.3 m. Caches bind actual parsed-data/camera bytes and dependency code hashes. The fixed deadline is 22:15 UTC; no new dependency/service/data download is used.

Four CPU continuations already completed (live control 15 epochs, live augmented 25, replay control 15, replay augmented 20). Validation clean/control -> augmented errors are live **6.654 -> 6.785 m**, replay **5.321 -> 5.378 m**; corrupted-context errors improve live **9.015 -> 8.048 m**, replay **6.863 -> 6.118 m**. Both augmented candidates pass this exploratory validation criterion, with clean trade-offs. These are validation results only; held-out evaluation is running and no promotion is claimed. This augmentation has no explicit ball-missingness feature and cannot fix a wrong camera or named identity.

The original tracking selection used eight validation clips. Its frozen motion/appearance/legacy settings are now being evaluated on **all forty validation clips** with identical existing detections, separate outputs and no grid search/reselection. This is validation expansion after earlier held-out inspection, not a fresh external test. The original selection files are preserved. A checked CLI guard rejects expanded validation without a separate output directory. Results will be appended after completion.


## Fixed-budget context results and expanded gates (22:20 UTC)

The earlier context continuations used equal **maximum** budgets but different actual epoch counts. A new comparison removes that confound: every live/replay, clean-control/context-augmented variant completed **30 full epochs / 3,690 optimiser updates**. The four local CPU fits took approximately 138 seconds including cached feature preparation; evaluation and interval calculation ran separately. Same training/validation/excluded match split, existing V4 architecture, fixed seeds and masks. No production checkpoint changed.

The held-out comparison uses the same **97,116 hidden player-timestamps per mode/condition**, one previously reviewed German match/two halves, exact visible IDs/positions/roles/phase and simulated visibility. It is not EPL video accuracy.

| Input / mode | Control mean | Context-trained mean | Approximate paired mean change and 95% interval |
| --- | ---: | ---: | --- |
| Exact context, live | 7.79 m | 7.95 m | +0.169 m [0.104, 0.235] |
| Stale ball every 30 s, live | 11.09 m | 10.13 m | −0.954 m [−1.164, −0.757] |
| Possession unknown, live | 8.81 m | 8.79 m | −0.015 m [−0.186, 0.142] |
| Midfield ball + unknown possession, live | 12.75 m | 12.61 m | −0.140 m [−0.278, 0.002] |
| Exact context, replay | 6.62 m | 6.80 m | +0.174 m [0.109, 0.241] |
| Stale ball every 30 s, replay | 9.06 m | 8.32 m | −0.737 m [−0.943, −0.543] |
| Possession unknown, replay | 7.24 m | 7.24 m | −0.003 m [−0.145, 0.122] |
| Midfield ball + unknown possession, replay | 10.18 m | 9.77 m | −0.407 m [−0.558, −0.265] |

Individual means above are rounded artifact values; paired changes use the saved unrounded errors. Intervals use seed 11026 / 2,000 draws / 99 contiguous 60-second blocks, resampling separately within each half and weighting by scored-player counts. The 120-second (50 blocks) and 300-second (21 blocks) sensitivity checks also support stale-context gain and clean-context regression. They are **within-match diagnostics**, not across-match generalisation intervals; trajectory dependence across blocks may remain. Unknown-possession-only effects are inconclusive. Midfield live improvement is inconclusive at 60/300 seconds; no favourable block length is selected post hoc.

Clean nominal-80% radius coverage decreases 87.60% → 82.51% live and 86.45% → 81.88% replay; stale-ball coverage improves 69.85% → 76.13% live and 71.71% → 76.28% replay. Better coverage in one condition is not universal uncertainty calibration. This trade-off is recorded rather than presented as a clean accuracy improvement. Native results and approximate intervals: `context_fixed_budget/evaluation_clean_control/review.html`, `.../evaluation_robust/review.html`, `.../paired_review.html`.

The frozen forty-clip tracking expansion completed on **6,000 frames / 102,075 labelled person observations**:

| Tracker | Precision | Recall | Anonymous identity F1 | ID switches / matched transitions |
| --- | ---: | ---: | ---: | ---: |
| Legacy | 92.01% | 64.12% | 40.98% | 4,957 / 64,527 |
| Selected motion | 90.21% | 74.51% | 46.67% | 4,354 / 75,128 |
| Selected appearance | 90.22% | 74.64% | 47.49% | 4,165 / 75,258 |

Appearance versus selected motion passes its original identity-F1 gain and precision checks. Selected motion versus legacy loses **1.79 precision points**, exceeding the fixed 1.5-point limit, so that gate fails. The settings are not promoted. These are anonymous image-space identities, not shirt-number or named EPL identity accuracy. After the completed metric run, the same saved results were rendered into a native table/gate page; no metric was rerun or threshold retuned for that presentation.

The full camera coverage decomposition is **2,638 common frames / 45,514 feet**, 131 newly supported frames / 2,351 feet and four lost frames / 77 feet. Common mean strict → foreground is 2.130 → 2.137 m; new support mean/p90 is **3.448 / 6.767 m**; lost support strict mean/p90 is 1.659 / 3.336 m. More coverage comes with harder, less accurate added cases, explaining part of the aggregate regression. Native table: `camera_coverage_tradeoffs/review.html`. No calibration gate is relaxed.

Actual commands included: `.venv/bin/python -m cv.audit_context_comparison`, `.venv/bin/python -m cv.audit_camera_tradeoffs`, and `.venv/bin/python -m cv.audit_appearance_tracking --all-validation --out cv/out/audit8_20261011/appearance_full_validation` (frozen existing settings/detections; exact selection config stored with output). Context training configurations and all four epoch/update histories are saved in `context_fixed_budget/`. Paired comparison rejects differing half/time/player indices rather than relying on matching counts; its test passed.

## UI honesty, causal player profiles and safe startup (22:20 UTC)

A new `measurement_diagnostic` reports contradictions between **two recent measured projections only**, using source timestamps when present, current/past data, a maximum five-second pair gap and ten-second freshness. Apparent displacement above 10 m/s is a **review camera or identity** flag, not verified physical speed, a confidence probability or a coordinate repair. Inferred points and future-source observations are excluded. Of 4,373 measured Arsenal slots, **439** carry this diagnostic under that scope, which differs from the narrower 395-transition statistic earlier in this report. Small jumps do not certify a correct position. At relative second four, role alias A24 implies 35.67 m/s over one second; that does not prove Bellerin ran at this speed.

The studio now says **POSITIONS UNVERIFIED** for CV sources and **PROJECTED** for measured CV points, retaining observed/inferred status and all coordinates. The actual A24 warning was clicked and inspected in isolated Chrome; no browser exception. `browser/studio_measurement_conflict.png` preserves the before-profile-fix view; `browser/studio_profile_final.png` is the inspected final view.

During this UI check, the player panel at second four was requesting statistics through second 60 and displaying “LIVE IMPACT.” It now requests the exact selected second and reconstruction mode, rejects stale/future profile responses when seeking, computes movement from that mode's selected frame prefix, filters events through the timestamp and omits future-derived tactical moment links in live profiles. Its headline is **CLIP ESTIMATES / LIVE ESTIMATES / REPLAY SUMMARY / PLAYER SUMMARY**, as appropriate. CV movement labels say projected distance/apparent speed/sprints because even measured points and role assignments remain unverified. The final screenshot actually shows **CLIP ESTIMATES through 24:04**, matching selected clock 24:04; no future minute is used. Two causal/profile tests passed, including changing later positions/events and hidden reference positions without changing an earlier live profile.

Existing tactical timeline calculation still uses full-minute averages, temporal formation smoothing and some subsequent outcome windows. This audit **does not implement a new causal tactical engine**. State payloads now explicitly label tactical analysis **retrospective**, and the frontend shows “Retrospective estimated geometry” for CV sources, “Retrospective tactical summary” otherwise, and “Replay poss (3')”. Live position inference and player profiles remain separate from these retrospective tactical calculations. Previously recorded tactical sensitivity results are evaluation of geometry, not real-time intent inference or coaching intention labels.

`run.sh` no longer broadly kills processes by command name. `scripts/dev_server_guard.py` checks all listeners on 3000/8000 before stopping anything, verifies project-root cwd and the expected backend/frontend executable, rechecks ownership before TERM and refuses foreign/unverifiable listeners. No SIGKILL fallback. Two focused tests passed. The actual current-project startup/restart succeeded; an actual other-checkout-root stop was refused, and both project PIDs remained running afterwards. The final local startup serves the preserved Arsenal export with **NARRATIVE_BACKEND=local**. No external narration call is required.

Latest full regression actually completed **131 passed**, 250 existing NumPy warnings, **54.61 seconds**. Latest frontend lint and webpack production build both passed. `compileall backend cv scripts tests` and `git diff --check` passed at the recorded pre-final checkpoint; final checks will be recorded separately. A rerun of the local 69-request API/catalog regression after the profile changes succeeded across 13 matches / 20 parts, with frontend HTTP 200. The first sandboxed attempt was blocked from localhost by OS permissions; the authorised local check passed. No automatic approval-review rejection occurred. Factual source events still assign the 370-second goal to A14 / Theo Walcott; this is a verified scorer label, **not proof of a verified A14 tracking identity**.

## Fresh nested-match context comparison (running)

A final bounded V3 comparison starts from random initialisation in every fold rather than original six-match pretrained weights: otherwise the designated validation match could already have been seen in pretraining. Each fold has **five training matches / one distinct validation match / one excluded test match**, seed 11026 + fold, 30 complete epochs and equal recorded control/candidate optimiser updates. Both mode validation gates must pass before candidate test evaluation. All seven available German matches were previously reviewed; this is not a newly untouched EPL test set. Exact visible/context limitations and simulated masks remain explicit. Training/evaluation deadline is **22:40 UTC**, leaving time for final inspection and documentation before the eight-hour deadline.

Executed command: `.venv/bin/python -m cv.train_context_crossvalidation --deadline 1791758400 --out cv/out/audit8_20261011/context_crossvalidation`. The read-only `.venv/bin/python -m cv.audit_context_crossvalidation --out cv/out/audit8_20261011/context_crossvalidation` validates disjoint partitions, successful evaluation records and equal-budget gates, and renders native per-match/age/coverage tables. Final fold counts/results will be appended after the job finishes. No model promotion is implied by a validation pass.


## Seven-fold context results (completed 22:26 UTC)

All **seven folds completed**, with four fresh V3 fits per fold: **28 models / 840 full training epochs / 102,480 optimiser updates** in total. Within each fold/mode the control and candidate have identical updates; counts range 3,600–3,720 per fit because each five-match training corpus has a different size. Both fixed validation gates passed in every fold before candidate test evaluation. Start 22:11:10, finish 22:26:14 UTC: approximately **904 seconds**, including feature generation, training and excluded-match evaluation. The job completed before its 22:40 deadline. No weights were promoted.

The seven excluded matches have **660,036 scored hidden player-timestamps per mode/condition** in total. These counts differ from the earlier 620,625 delayed-observation-intersection stress. Each row below is an **equal-match average of that match's mean error**, not a pooled mean or averaged percentile. All match counts, median/p90/radius coverage and observation-age results are in `context_crossvalidation/review.html`. Evaluation artifact means are rounded to .01 m; the displayed macro figures inherit that resolution.

| Context | Live control → candidate | Replay control → candidate |
| --- | ---: | ---: |
| Exact ball/possession | 8.760 → 8.946 m | 7.443 → 7.694 m |
| Ball observed every 30 s | 11.779 → 10.827 m | 9.664 → 8.806 m |
| Possession unknown | 10.091 → 9.789 m | 8.126 → 7.969 m |
| Midfield ball + unknown possession | 14.563 → 13.933 m | 11.303 → 10.477 m |

The trade-off repeats across the broader match evaluation: stale-context gain, clean-context regression. These models still assume exact visible positions and identities, and cannot correct the measured camera/role errors in the Arsenal clip. Same transferred camera-mask source is used across these German matches; this is not an independent camera-domain or EPL test. Seven overlapping training folds are not seven independently trained production systems or independent evidence of real-world accuracy.

Observation age is measured from the last **past** visible sample. “Never seen” is separate from a finite interval; replay can still use later visible observations. Equal-match mean errors for live mode illustrate why long absence must not receive high confidence:

| Last-observation age | Hidden N (seven matches) | Exact context, control → candidate | Stale ball, control → candidate |
| --- | ---: | ---: | ---: |
| ≤3 s | 100,487 | 3.051 → 2.929 m | 3.857 → 3.267 m |
| >3–10 s | 142,537 | 6.427 → 6.459 m | 8.359 → 7.577 m |
| >10–30 s | 208,237 | 10.056 → 10.347 m | 13.761 → 12.700 m |
| >30 s | 203,936 | 11.753 → 12.111 m | 15.989 → 14.844 m |
| Never seen | 4,839 | 13.266 → 12.921 m | 14.040 → 13.336 m |

Even with exact visible measurements, >30-second stale-context live estimates remain about **15 m wrong**. The appropriate next step is context/source availability and calibrated uncertainty, together with camera/identity fixes; another undifferentiated training extension is insufficient. The maximum .3 m clean **validation** tolerance is not proof that every held-out match improves. The private models remain experiments pending an independently labelled real-broadcast test and an explicit deployment decision.

Additional actual verification: final Python rerun **131 passed / 250 warnings / 65.93 seconds**, after adding the context-only tactical scoring option. The new local profile/state check completed **15 successful contract requests**, verifying exact selected time/clamping, synthetic and tracking live profiles excluding retrospective moments, unsupported CV live/truth/invalid modes returning 400, and explicitly retrospective tactical state metadata. Native results: `api_profile_final_review.html`.

Selected-player 1× playback was measured for **20.009 seconds** in isolated headless Chrome while local V3 training was active: **1,201 requestAnimationFrame callbacks (60.02 Hz), 1,199 pitch-position changes (59.92 Hz), median/p90 interval 16.7 ms, maximum 16.8 ms, zero long tasks**. Player state/profile requests occur at integer-second changes, not every animation callback. This is local presentation throughput, not 60 Hz detection, positional accuracy or proof of the user's physical display performance. `browser/playback_selected_final.json` retains the measurement conditions and actual request timings.

A bounded diagnostic now scores how the frozen context candidates affect production width/depth/defensive-line calculations across these same excluded matches. It uses no new training or selection and does not score tactical intent. Command actually launched: `.venv/bin/python -m cv.audit_context_tactics --training cv/out/audit8_20261011/context_crossvalidation --out cv/out/audit8_20261011/context_tactics --deadline 1791759300` (22:55 UTC). The first already-completed identical-source check is reused; remaining checks have separate execution logs and per-match native tables. Final measured results will follow after completion.


## Context error propagated to tactics (completed 22:30 UTC)

All **14 model checks across seven excluded matches** completed before the 22:55 diagnostic deadline. The first completed identical-source control was reused; thirteen additional checks ran with separate logs. No training, test-based checkpoint selection, radii refit, event cue or speed limiter. Each mode/condition scores **78,406 frame-team samples** total, with at least seven active outfield players and at least one inferred position. Visible positions, roles, IDs, possession and phase remain exact; ball context is the stated controlled input. Truth enters scoring after inference.

Values below are **equal-match means of absolute geometry error**, not pooled percentiles, verified tactical intent or causal outcomes:

| Mode / context | Width control → candidate | Depth control → candidate | Defensive line control → candidate |
| --- | ---: | ---: | ---: |
| Live / exact | 7.015 → 7.297 m | 5.302 → 5.327 m | 3.443 → 3.555 m |
| Live / ball every 30 s | 7.076 → 7.239 m | 5.722 → 5.444 m | 5.721 → 5.027 m |
| Replay / exact | 5.737 → 5.803 m | 4.798 → 4.870 m | 2.986 → 3.238 m |
| Replay / ball every 30 s | 5.797 → 5.822 m | 5.125 → 4.945 m | 4.612 → 3.980 m |

Context training improves stale-context line/depth error while width slightly regresses. A lower player-position average does not establish better tactical measurements generally. These several-metre errors constrain interpretation of a 2 m tactical-change threshold; window-change errors remain separate as previously measured. The reused production envelopes exclude observed-player perception error and are sensitivity bounds, not joint nominal-80% confidence intervals. Individual match sample counts, median/p90, envelope coverage and span are available in `context_tactics/review.html` and per-fold pages. They do not establish EPL tracking or tactical accuracy.

## Backward-seek state safeguard (22:33 UTC)

A further UI review found that the small player-state popup retained the previous response while seeking or changing player/mode. Even after fixing the larger statistics panel, a later position/history could briefly appear at an earlier clock. `PlayerStateCard.tsx` now binds each response to player, request mode and requested second; later timestamps or a changed player/mode are hidden until replaced. At most one second of past lag is allowed during forward playback. Failed requests cannot retain arbitrarily old state. No inference coordinate is changed.

The actual local browser check delayed **delivery of real API state responses by five seconds**, without inventing any output. Keyboard Home sought from 24:04 to 24:00: while pending the old popup was absent and the player statistics did not show 24:04; after delivery it showed the genuine 24:00 state with no 35.67 m/s later warning. Zero browser exceptions. The final screenshot was actually inspected (`browser/studio_seek_final.png`); the pending screenshot and test conditions are retained separately. Latest frontend lint and webpack production build passed after this fix.

The first faster-playback measurement exhausted Chrome's default resource-timing buffer, so its captured request list was incomplete even though animation counts were valid. That artifact is retained as `browser/playback_selected_8x_before_seek_fix.json`. The subsequent check enlarges/clears the timing buffer and runs against the final seek safeguard; final faster-playback values will be recorded after inspection. No truncated request count is claimed as the complete number of requests.


## Presentation causality and final faster-playback checks (22:45 UTC)

A source-level review exposed a second live-view issue: `Pitch` interpolated between the current and next 1 Hz position for **every** mode. While each live sample is inferred causally, displaying a blend with the next sample lets later observations affect an earlier fractional playback time. Live rendering now holds the latest available sample; retrospective replay retains interpolation. `tests/frontend_pitch_causality.cjs` executes the actual TSX component with installed TypeScript/React server rendering. Changing future coordinates, ball and possession leaves the current live SVG **identical**, changes retrospective rendering as expected, and becomes visible exactly at the next timestamp. The behavioural check actually passed; no new test framework or dependency was added.

`Studio` also previously fell back to its default frame array when another mode was loading. For a tracking/synthetic truth default, this briefly presented ground truth under a Camera live label. The fallback is removed: the pitch and associated camera/player overlays wait for the selected mode, expose a loader and show **Now: N/A** rather than a false zero-player count. Failed frame requests have a visible Retry action. Switching Truth → Camera live with **real local live-frame delivery delayed five seconds** actually showed 22 truth points before selection, **zero points while loading**, then 22 live state slots after delivery, with zero browser exceptions. The loader screenshot was inspected. This is presentation isolation; ground truth remains evaluation-only for the reconstruction.

Camera live says **Live · 1 s samples**. A step once per second is expected for these 1 Hz causal samples; it does not claim 25 Hz live broadcast ingestion. Replay remains smooth because it is retrospective. Existing full-sequence tactical/story summaries remain explicitly retrospective; this change does not implement causal tactical analysis.

Final selected-player retrospective playback measurements use the enlarged Chrome timing buffer and the preserved Arsenal archive, local development server. Background load was not separately measured.

| Speed / condition | Duration | Pitch-position updates | Median / p90 callback interval | Long tasks | Completed measured profile/state requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8×, selected player | 20.008 s | 59.58 Hz | 16.7 / 16.7 ms | 0 | 318 |
| 32×, selected player | 10.006 s | 57.77 Hz | 16.7 / 16.7 ms | 0 | 638 |

The 8× recorded request maximum/mean durations are **5.0 / 2.99 ms**; 32× **5.6 / 2.85 ms**, respectively. These are local Resource Timing measurements for completed state/profile requests, not guaranteed end-to-end service latency or physical-screen FPS. At 32× source seconds advance rapidly, so request rates rise; frame animation does not create new detection samples. No source positions or measured FPS were fabricated.

Latest frontend lint passes **without the new hook warning**; the warning introduced by an unstable empty-array fallback was fixed using a stable constant. Latest webpack production build and the behavioural causal-pitch check passed. The final Python result remains **131 passed / 250 existing warnings**, and is not re-run for display-label-only report changes. Compile/diff checks will be confirmed at closeout.

The native main dashboard now hides its long diagnostic-link list behind “Explore”; the latest seven-fold trade-off table is visible first and the 18-frame annotation link is prominent. New context/geometry tables use human-readable labels while machine metric keys remain unchanged. The actual three-page browser review found **17 / 4 / 3 native tables**, zero raw JSON blocks, zero embedded videos, zero broken images and zero exceptions; screenshots were inspected. Latest report-label presentation will be checked again without rerunning training or prediction.


## Final source/annotation review (22:58 UTC)

The causal pitch now also derives its displayed formation count from the current causal line assignments. It does not fall back to retrospective/truth line groups or formation names. The actual-component test additionally mutates the retrospective formation label and groups and confirms that the current causal SVG remains unchanged. This fixes a presentation leak without implementing a new tactical timeline engine.

The existing analytical basis is now stated in the tactical panel: **tracked ground truth / synthetic ground truth / source positions including estimates**. In a truth-default DFL/synthetic match, changing the pitch to Camera live does not silently recompute its prebuilt tactical timeline from the live estimates. That timeline remains retrospective ground-truth analysis, now labelled explicitly; mode-specific causal tactical analytics remain **incomplete**. CV clip analytics use the source measurements/estimates and remain unverified.

Independent human annotation scoring remains **not run**. The three assistant-provisional 1800-second landmarks are retained for provisional diagnostics, not relabelled as human truth. The camera and identity importers now reject their known assistant/non-independent provenance notes under the default human basis, require the explicit `assistant_provisional` basis and label that output accordingly. The actual negative CLI produced the expected rejection and **no human-score artifact**. Deliberately fabricated UI-smoke rows remain rejected for every basis. One new provenance test verifies both rejection and explicitly declared provisional import.

Latest full Python regression completed **132 passed / 250 existing NumPy warnings / 47.95 seconds** after the provenance changes. Latest frontend lint passed without warnings; webpack production build and actual-component future/formation mutation checks passed. The final live-mode GUI probe again measured 22 truth slots → **zero while loading** → 22 separate live slots, zero browser exceptions. Actual API/catalog checks remain 69 successful requests across 13 matches / 20 parts, plus fifteen profile/state/time-mode contract requests; no later backend runtime logic was changed.

SHA-256 preservation verification at 22:51 UTC checked **all 88 initially protected files unchanged: six CV JSON files including the backup, plus 82 production checkpoints**. A separate check verified **all 19 active exports unchanged** against the earlier active-export audit. These are separate, overlapping inventories, not 88 matches. The legacy DFL J03WMX 1 Hz cache still has SHA-256 `ff951e24b10298b55e8cb1f57c8182ea931e72178bddb1debfcdf4c4df63ce88`. All **28** saved nested-match checkpoint hashes and actual training/validation/test metadata were checked: five training matches, separate validation/test, correct feature version/mode. Metadata checks complement, not replace, the feature-causality tests.

Private figures, videos, feature caches and experimental weights remain under ignored `cv/out/audit8_20261011/`; `git check-ignore` confirmed the review page and candidate checkpoint. Reports/manual instructions are repository Markdown. No commit, paid API call, purchase, external training job, newly imported restricted match data or outgoing message was made. Public provider terms were researched as documented earlier; this does not grant rights to redistribute the private assets.

## Reviewable files and checks

The checkout already had uncommitted work before this eight-hour session. Current `git status` includes those earlier-session files too; the following lists the relevant review groups, not a claim that every untracked file was created today.

| Area | Relevant changed/created files | Purpose / evidence |
| --- | --- | --- |
| Runtime/source honesty | `backend/catalog.py`, `backend/pipeline.py`, `backend/api/main.py`, `backend/reconstruction/measurement_review.py` | Canonical source cache; selected-time/mode profiles; projection conflict diagnostic; retrospective tactical metadata; actual API checks |
| Timestamp/cache/camera/tracking safeguards | `backend/reconstruction/datasets.py`, `backend/reconstruction/real_benchmark.py`, `cv/soccernet.py`, `cv/detect_track.py`, `cv/build_match.py`, camera/namespace/appearance helpers listed in the chronological sections | Explicit rates/provenance, causal masks, source observation times, camera-shot boundaries and ID namespaces; no active XY overwrite |
| Training/evaluation | `cv/train_hidden_robustness.py`, `cv/train_context_robustness.py`, `cv/train_context_crossvalidation.py`, `cv/audit_context_comparison.py`, `cv/audit_context_crossvalidation.py`, `cv/audit_tactical_sensitivity.py`, `cv/audit_context_tactics.py`, other camera/tracking/ball audit modules | Frozen conditions, separate match gates, exact paired indices, unsuccessful experiments and native tables |
| Private annotation and reports | `cv/manual_camera_review.py`, `cv/evaluate_manual_camera.py`, `cv/evaluate_manual_identity.py`, `cv/audit_dashboard.py`, `cv/session_report.py` | Independent landmark/shirt workflow, explicit provisional basis, CSV safeguards, native readable tables and expandable diagnostics |
| Frontend | `Studio.tsx`, `Pitch.tsx`, `PlayerStateCard.tsx`, `PlayerPanel.tsx`, `LiveState.tsx`, insight/source labels and `frontend/lib/{api,types}.ts` | Two team colours, measured/estimated provenance, prediction-zone button, explanation overlay, selected-time guards and causal/live loading isolation |
| Startup | `run.sh`, `scripts/dev_server_guard.py` | Verify project ownership before stopping 3000/8000; refuse foreign ports; actual startup and refusal checks |
| Regression | `tests/test_*` modules named with the relevant sections, `tests/frontend_pitch_causality.cjs` | 132 Python tests plus actual-component rendering checks; no new framework |
| Documentation | This report, `CALIBRATION_IDENTITY_8H_USER_ACTIONS.md`, cumulative `AI_IMPLEMENTATION_STATUS.md` | Commands, actual outcomes, failures, visual evidence, user actions and honest limits |

Commands actually executed for final verification include:

```sh
.venv/bin/python -m pytest -q
node tests/frontend_pitch_causality.cjs
cd frontend
npm run lint
npm run build -- --webpack
cd ..
.venv/bin/python -m compileall -q backend cv scripts tests
git diff --check
.venv/bin/python scripts/dev_server_guard.py --root /Users/qnrj/Code/epl-hackathon
```

The browser/API helpers in `/private/tmp/football_audit8_*` exercised actual local endpoints and isolated Chrome; final screenshots/response conditions are retained in the ignored artifact root. Training/evaluation commands above and their saved configs/execution records give exact checkpoint paths, seeds, split/mask conditions and deadlines. Training deadlines are historical Unix timestamps: repeating the experiment later requires an explicit new future deadline and a new output directory, not blindly copying an expired time.

## Acceptance and next priorities

| Criterion | Result | Evidence / remaining limit |
| --- | --- | --- |
| Reproducible camera/detection/tracking/hidden baselines and separated error causes | **Passed under documented local conditions** | Existing assets, fixed samples/configs/hashes, camera-only versus full visible pipeline, controlled context stresses, actual images |
| Improved full-pipeline EPL accuracy / all named players reliably located | **Not achieved / cannot verify hidden accuracy** | Expanded camera/tracker gates fail; named role aliases and camera remain unverified; no independent hidden EPL truth |
| Useful private failure visualization and user annotation workflow | **Passed operationally; human scoring not run** | Images/movies/readback, eighteen-frame page, tested CSV export/import; real human CSV absent |
| Live position/profile/rendering future isolation | **Passed relevant automated and actual UI checks** | Future mutation invariance, past/current masks/holds, selected prefixes, no cached truth while live loads; CV live remains unsupported |
| Honest calibrated real-video uncertainty | **Incomplete** | CV confidence N/A; model radii exclude perception/identity uncertainty and controlled coverage varies with context |
| Tactical evidence/uncertainty diagnostics | **Passed for geometry diagnostics only** | Per-match width/depth/line/window errors and sensitivity coverage; no coach-intent or causal-outcome labels |
| New causal tactical engine / end-to-end real live ingestion | **Not implemented / not verified** | Existing timelines/stories explicitly retrospective; truth-based versus source-estimated analytical basis labelled |
| Existing application, startup and regression checks | **Passed executed checks** | 132 Python tests, separate rendering check, lint/build, 69+15 HTTP checks, actual Chrome checks; own physical display/manual accuracy still requires review |
| Active outputs and production checkpoints preserved | **Passed hash checks** | 19 active exports unchanged; 88 initially protected files unchanged; candidates remain private |

Priority order for the next work:

1. Obtain independent **landmark and readable shirt/team clicks** from the eighteen supplied frames, projection off. Resolve camera pose/timestamp/orientation versus identity before replacing a model. Use the retained unchanged-camera and private-candidate scorers; do not fit/test the same labels and call that held-out accuracy.
2. Fix camera support/quality and anonymous/named association together with separate gates. Camera-cut namespaces/current-foot anchoring prevent specific defects but do not verify a person or camera. Add independent angles/another permitted clip; retain coverage penalties for missed/abstained observations.
3. Expose and train with **ball availability/age, possession uncertainty and perception quality**. The same model cannot treat exact, held, inferred-touch and unknown context as equally trustworthy. Test clean and corrupted conditions, calibrate regions conditionally and preserve the clean-input regressions rather than hiding them.
4. Seek permitted synchronised EPL full-position labels if available; events/lineups or more broadcast video alone cannot score hidden-player truth. Provider rights and usage restrictions remain as documented. No paid/restricted acquisition is authorised.
5. Build mode-specific tactical analytics from the selected estimates with comparative-window uncertainty, keeping observed outcomes and inferred intent separate. Ground-truth demo summaries and live position estimates must not be conflated.

The immediate user action is in the separate Markdown guide. You do not need to retrain or rerun detection to inspect the delivered outputs.

## Final camera provenance check (23:05 UTC)

The DFL live/replay camera masks are transferred SkillCorner footprints moved onto the tracking match, not measured broadcast coverage for that DFL match. The studio now labels them **SIMULATED CAMERA**, and its camera review panel states this limitation. No camera footprint or player coordinate changed. Actual isolated Chrome verified the final label and the real delayed loading sequence: 22 truth slots → zero pending slots → 22 live estimate slots, with no runtime exceptions. The causal-rendering component check, compile/diff checks, frontend lint and webpack production build passed again after this display change. This does not validate real-video location or identity.

The final guide path check found **27 ready private review/movie files** and four human-scoring outputs that are correctly absent until you supply the independent CSV and run the documented scorers. An initial check incorrectly required those conditional outputs to exist and failed; the corrected check distinguishes them explicitly (`user_guide_paths_verified.json`). Human scoring remains not run. Final demo health returned `{ok: true}` and the Arsenal studio returned HTTP 200.

## Eight-hour closeout (23:09:50 UTC)

The authorised window ran from **15:09:31 to 23:09:31 UTC**, eight elapsed wall-clock hours; final handoff was recorded at **2026-10-11T23:09:50.412791+00:00**. This is not eight hours of GPU training. Training, evaluation, engineering, output inspection and regression checks are separately documented above. No training/evaluation job remains running.

The heartbeat began at **2026-10-11T18:34:50.363042+00:00**: the initial approximately **3 h 25 min** was not heartbeat-instrumented, so uninterrupted execution over that initial interval cannot be independently established. The instrumented interval ends at **2026-10-11T23:09:25.849536+00:00**, with **550** samples and maximum adjacent gap **30.072 s**. No large host pause is observed in that interval. Heartbeat continuity is not a GPU utilisation measurement. Raw accounting is retained in `heartbeat.jsonl` and `closeout.json`.

Only the audit-owned isolated Chrome, heartbeat and caffeinate helpers were considered for cleanup, with exact process-command ownership checks. Cleanup: **58886: TERM sent to verified audit helper; 68034: TERM sent to verified audit helper; 87992: TERM sent to verified audit helper**. Existing frontend/backend servers were left running; final backend health is **True**, frontend HTTP **200**.

Final verification: **132 Python tests passed**, 250 existing NumPy warnings; actual-component causal rendering, frontend lint, webpack production build, compile and diff checks passed. The final actual screenshot was inspected after the simulated-camera label change. Preservation checks confirm 88 protected assets and 19 active exports unchanged (overlapping inventories). No experimental model/calibration/tracker configuration was promoted.

The completed deliverable is this bounded engineering audit and reviewed safeguards, not a claim of reliable all-player EPL reconstruction. Your next action is the eighteen-frame independent annotation in the separate Markdown guide. Four human-score HTML outputs correctly remain absent until those labels are supplied and scored.

## Follow-up: independent user CSV received (12 October 2026)

The previous missing-human-label status applies to the completed eight-hour window. Your subsequently supplied CSV now provides 17 landmark labels on nine frames and 23 identity clicks. Unchanged-camera mean/median/P90 landmark error is **15.55 / 3.79 / 55.23 m**, N=17. Frozen temporal/foreground candidates score **5.06 / 5.05 m mean** on those same selected landmarks, but neither is deployed; full validation failed and residual 1961.4 error remains 20.72 m. These selected failure frames do not measure match-wide or hidden-player accuracy.

The current export already contains one source penalty and five free kicks; frontend filtering hid them. Added native-table event overlay, seekable set-piece markers/feed and separate body-click association (nine tracks/five unconfirmed detections/nine unsupported or unmatched; not ground-contact truth). One supplied team label needs review. All 19 active exports remain unchanged. See [follow-up results and manual tests](MANUAL_REVIEW_FOLLOWUP.md) and the updated user-action guide. The earlier eight-hour metrics and promotion decisions remain preserved.
