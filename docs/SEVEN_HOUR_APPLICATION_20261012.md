# Seven-hour training and match application — 12 October 2026

Status: running. This is not a completed training report or an accuracy improvement claim.

## Session and inputs

Requested window: 13:33:49–20:33:49 Singapore (05:33:49–12:33:49 UTC), 12 October. Session root: `cv/out/apply_train7h_20261012_053349`. GPU/CPU training began about 13:46 after inspection and baseline checks. Setup, evaluation and application are included in the seven-hour window; this is not seven uninterrupted GPU hours.

Seven locally available DFL matches have full tracking ground truth, two halves each. No corresponding DFL broadcast video is available here. DFL evaluates reconstruction from exact visible tracking with simulated camera masks, identities, coarse roles and declared ball/game context. It cannot establish broadcast detection/calibration accuracy.

Arsenal–Liverpool, 14 August 2016, is the only locally available EPL broadcast clip. Four other EPL half exports can be updated from their stored measured positions, but their corresponding local videos and complete hidden-player truth are missing. Existing inferred coordinates are masked before inference. No estimated coordinates are used as training labels. No footage or weights were downloaded.

## Training and separation

DFL: fixed seed 11026; existing V3 architecture; seven folds, each five training matches, one validation match, one excluded test match; 150 requested complete epochs per model. Live and replay models are separate. Clean-control and context-robust candidates are selected using validation conditions, never held-out test error. Earlier sessions have reviewed these test matches: this is not a fresh independent external benchmark. Source/config snapshots and actual optimizer-update counts are stored per fold.

Sequential MPS queue: pitch-marking continuation, physical camera, appearance embedding, jersey recognition, then a physical-camera control if time remains. Each stage has its own bounded duration and the common deadline. Stages that cannot finish are not counted as completed. Camera/person models use existing game-separated datasets; the EPL close-up head remains the frozen diagnostic classifier trained previously. Category recognition is not proof of camera accuracy.

## Application

Original model files and match exports are retained. New exports have `trial7h_preview` or `trial7h_trained` in their names and explicit experimental metadata/catalog labels. Preview uses the prior completed cross-validation checkpoints; trained uses this session only after all seven folds complete. No global production-weight replacement occurs.

DFL application verifies the checkpoint excludes the target match from training and validation. Event pins are disabled to avoid full-match clock fitting and later event information. Metrics report hidden mean/median/P90 in metres, sample counts, observation-age groups, last-observed paired baseline and uncertainty-radius coverage. Observed zero error means exact tracking was supplied; it is not broadcast-camera accuracy. Legacy postprocessing differs, so the old-export comparison does not isolate a model-only change.

EPL application freezes complete camera checkpoints, rejects unsupported views, uses current-image camera predictions and cached anonymous detections, and exports a footprint derived from the same camera matrices as measured positions. Named-player assignment, missing detections, ball measurements and hidden-player accuracy remain unverified. Private review movies stay outside the web application. New appearance/jersey checkpoints require their own evaluation gates before any identity promotion.

## Actual checks and early results

- Initial Python suite: 137 passed, 250 existing warnings, 95.51 s (`/private/tmp/football_7h_baseline_tests.log`).
- Integration suite before application fixes: 138 passed, 250 warnings, 407.89 s (`/private/tmp/football_7h_regression.log`).
- Focused application/camera regressions: four passed, 0.55 s. Null inactive substitutes are excluded, never converted to `(0,0)`.
- Compile and whitespace checks ran successfully before the later application fixes; final checks remain pending.
- First preview DFL J03WMX half 1 live hidden error: N=47,800; mean 9.153 m, median 7.206 m, P90 19.014 m. Legacy mean 7.822 m. This is worse, with different postprocessing; not a training improvement.

Application attempts exposed two defects: null substitute coordinates made ragged arrays, and a double pitch-orientation transform changed observed rounding ties by up to 0.1414 m. Candidate export now serializes original observed measurements exactly. The original-export observed error remains separately scored rather than asserted away. Failed logs are archived in the session root. Full application, final aggregate metrics, browser verification and model gates are pending.

## Reproducible commands actually executed

```sh
.venv/bin/python -u /private/tmp/football_7h_supervisor.py
.venv/bin/python -m pytest tests/test_candidate_application_metrics.py tests/test_camera_footprint.py -q
```

Exact queued commands: `plan.json`, `gpu_plan.json`; actual commands/PIDs/stage times: `processes.json`, `gpu_running.json`, `gpu_execution.json`, `application_started.json`. Full test logs above record the executed baseline/regression commands. `heartbeat.jsonl` records job return codes. A heartbeat is not evidence that a model improved.

## Review and remaining actions

Use the separate [manual guide](SEVEN_HOUR_APPLICATION_20261012_MANUAL_TESTING.md). Read native HTML metric tables in `review_preview.html` and, once available, `review_trained.html`. Final results must include actual completed folds/stages, coverage and uncertainty limitations, application failures, original-file hash checks, and any blocked candidates. Do not present an unfinished run as completed.

For more EPL validation, supply only footage/tracking you are permitted to use, with timestamps and pitch-coordinate conventions. A video without synchronized all-player ground truth supports visible-player review but cannot score hidden-player accuracy. This is optional; existing local work continues without it.

## Mid-session verification

All14 preview DFL halves were exported. Hidden N=665,019 per mode; sample-weighted mean9.312m live and7.869m replay. Legacy mean8.434m live and6.884m replay under different event/postprocessing conditions. Preview is worse in both modes. All candidate observed coordinates match truth after serialization. Full later regression: **140 passed**,250 existing warnings,421.58s (`/private/tmp/football_7h_application_regression.log`). Frontend lint and webpack production build passed; actual event and causal-rendering checks passed. Native result-table screenshot was rendered and visually inspected at `/private/tmp/football-7h-review.png` (13/14 snapshot before the final half arrived); isolated browser closed afterward. Browser connector was unavailable; direct headless Chrome was used. New app routes still await EPL publication/server refresh.

Marking continuation completed36 complete epochs (179 batches each); best selection cost remained6.779m, equal to the starting selected checkpoint. No calibration improvement is claimed. Physical-camera experiment started next. Camera validation/user-landmark scoring runs independently, without feeding reference coordinates into prediction. The trained DFL comparisons and EPL application remain pending.

Frozen marking-camera check: selected supported N3,405 feet, mean6.032m, median1.461m, P9010.566m; fixed mean≤3m/P90≤5m gate fails. Initial and selected support differ, so use the paired-common rows in `camera_evaluation/paired_evaluation/review.html` for a matched comparison. User landmark check:12/17 scored, mean1.667m/P902.712m; five unresolved. These visible reference-foot/landmark diagnostics do not score identities or hidden EPL players. Test remains not run after failed camera validation gate.

Preview application completed:14 DFL halves and five EPL half/clip exports. Arsenal private start/penalty/goal movies exist; one actual start frame was decoded with OpenCV and visually inspected. `ffmpeg` CLI was missing, so OpenCV was used for the review frame. Camera geometry and shirt identities remain visibly imperfect; no EPL position-accuracy claim is made. Original measured coordinates in the other four EPL half exports were verified unchanged (23,137/25,493/25,810/23,441 observations). The later positional-role mismatch was fixed with explicit CF/RWB/etc mapping; unknown roles fail instead of being guessed. Focused application/catalog suite8 passed15.16s after that fix. Earlier catalog/application/camera9 passed16.19s. These focused checks follow the140-test full run; no142-test full-run claim is made.

Catalog date/grouping was checked for all seven DFL preview cards. Fixed the experimental EPL label being overwritten during provenance refresh and raised the metadata cache capacity to128 to accommodate original/preview/trained exports without32-entry cache thrashing. Preview server refresh started successfully; API health200. Full candidate catalog/studio route verification remains pending until the background verifier records results. All seven-hour training jobs continue independently.

Preview demo verification completed: health200;12 candidate catalog cards; Arsenal match API200; studio route200. A rendered headless-browser screenshot at clip4s/24:04 was inspected (`/private/tmp/football-7h-demo.png`): camera footprint,12 measured/10 inferred header, two team colours, single prediction-zone button, unverified tactical geometry, N/A confidence and set-piece timeline are visible. This confirms rendering, not position/identity accuracy. The isolated browser was closed; demo and training remain running. Final trained-phase browser review must still be done by the user; the background verifier checks HTTP/data availability only.
