# Seven-hour training and match application — 12 October 2026

Status: session ended. See recorded closeout results below for completed and incomplete stages. No improvement is assumed.

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


## Automated closeout artifacts

# Seven-hour application — recorded results

Generated 2026-10-12T21:18:52+08:00

Supervisor ended: True. This is an artifact-based report; missing stages remain incomplete.

DFL uses exact visible tracking and simulated visibility. EPL hidden truth is missing. No broadcast accuracy is inferred from DFL error. Live uses past sightings; replay may use later sightings. Legacy event postprocessing differs. No production weights are promoted.

## preview: 14/14 DFL halves

| Match | Half | Mode | Hidden N | Mean m | Median m | P90 m | Legacy mean m | Observed preserved |
|---|---:|---|---:|---:|---:|---:|---:|---|
| DFL-MAT-J03WMX | 1 | live | 47800 | 9.153 | 7.206 | 19.014 | 7.822 | True |
| DFL-MAT-J03WMX | 1 | replay | 47800 | 7.474 | 5.474 | 16.444 | 6.391 | True |
| DFL-MAT-J03WMX | 2 | live | 50912 | 9.203 | 7.111 | 19.337 | 8.797 | True |
| DFL-MAT-J03WMX | 2 | replay | 50912 | 7.56 | 5.492 | 16.665 | 6.844 | True |
| DFL-MAT-J03WN1 | 1 | live | 44137 | 8.915 | 6.964 | 18.304 | 7.952 | True |
| DFL-MAT-J03WN1 | 1 | replay | 44137 | 7.294 | 5.482 | 15.79 | 6.527 | True |
| DFL-MAT-J03WN1 | 2 | live | 44923 | 8.995 | 7.117 | 18.427 | 8.303 | True |
| DFL-MAT-J03WN1 | 2 | replay | 44923 | 7.326 | 5.412 | 16.25 | 6.537 | True |
| DFL-MAT-J03WOH | 1 | live | 46833 | 9.691 | 7.616 | 20.443 | 8.881 | True |
| DFL-MAT-J03WOH | 1 | replay | 46833 | 7.904 | 5.803 | 17.204 | 7.158 | True |
| DFL-MAT-J03WOH | 2 | live | 46445 | 10.014 | 7.642 | 21.355 | 9.071 | True |
| DFL-MAT-J03WOH | 2 | replay | 46445 | 8.597 | 6.297 | 18.828 | 7.65 | True |
| DFL-MAT-J03WOY | 1 | live | 47106 | 9.06 | 7.1 | 18.537 | 8.296 | True |
| DFL-MAT-J03WOY | 1 | replay | 47106 | 8.002 | 5.825 | 17.236 | 6.943 | True |
| DFL-MAT-J03WOY | 2 | live | 49825 | 9.459 | 7.424 | 19.54 | 8.497 | True |
| DFL-MAT-J03WOY | 2 | replay | 49825 | 8.226 | 6.2 | 18.091 | 7.058 | True |
| DFL-MAT-J03WPY | 1 | live | 45687 | 8.916 | 7.08 | 18.659 | 7.902 | True |
| DFL-MAT-J03WPY | 1 | replay | 45687 | 7.571 | 5.692 | 16.58 | 6.472 | True |
| DFL-MAT-J03WPY | 2 | live | 51067 | 10.008 | 7.94 | 20.632 | 9.099 | True |
| DFL-MAT-J03WPY | 2 | replay | 51067 | 8.377 | 6.168 | 18.668 | 7.448 | True |
| DFL-MAT-J03WQQ | 1 | live | 45933 | 8.996 | 7.102 | 18.924 | 8.283 | True |
| DFL-MAT-J03WQQ | 1 | replay | 45933 | 7.711 | 5.798 | 16.787 | 6.897 | True |
| DFL-MAT-J03WQQ | 2 | live | 46070 | 8.88 | 7.1 | 18.508 | 8.028 | True |
| DFL-MAT-J03WQQ | 2 | replay | 46070 | 7.739 | 5.763 | 17.182 | 6.567 | True |
| DFL-MAT-J03WR9 | 1 | live | 46028 | 9.073 | 7.256 | 18.54 | 7.948 | True |
| DFL-MAT-J03WR9 | 1 | replay | 46028 | 7.634 | 5.78 | 16.47 | 6.549 | True |
| DFL-MAT-J03WR9 | 2 | live | 52253 | 9.832 | 7.71 | 20.387 | 8.987 | True |
| DFL-MAT-J03WR9 | 2 | replay | 52253 | 8.576 | 6.14 | 19.261 | 7.196 | True |

live: hidden N=665,019, sample-weighted mean=9.312 m. Overall median/P90 not computed by averaging row quantiles.
replay: hidden N=665,019, sample-weighted mean=7.869 m. Overall median/P90 not computed by averaging row quantiles.

EPL application marker: True; other EPL half exports: 4/4. Hidden accuracy: N/A. Named identities and video calibration remain unverified.

## trained: 14/14 DFL halves

| Match | Half | Mode | Hidden N | Mean m | Median m | P90 m | Legacy mean m | Observed preserved |
|---|---:|---|---:|---:|---:|---:|---:|---|
| DFL-MAT-J03WMX | 1 | live | 47800 | 8.739 | 6.888 | 18.381 | 7.822 | True |
| DFL-MAT-J03WMX | 1 | replay | 47800 | 7.184 | 5.247 | 15.815 | 6.391 | True |
| DFL-MAT-J03WMX | 2 | live | 50912 | 8.993 | 6.826 | 18.921 | 8.797 | True |
| DFL-MAT-J03WMX | 2 | replay | 50912 | 7.459 | 5.3 | 16.382 | 6.844 | True |
| DFL-MAT-J03WN1 | 1 | live | 44137 | 8.628 | 6.673 | 18.001 | 7.952 | True |
| DFL-MAT-J03WN1 | 1 | replay | 44137 | 7.191 | 5.235 | 15.784 | 6.527 | True |
| DFL-MAT-J03WN1 | 2 | live | 44923 | 8.899 | 6.923 | 18.832 | 8.303 | True |
| DFL-MAT-J03WN1 | 2 | replay | 44923 | 7.223 | 5.162 | 16.488 | 6.537 | True |
| DFL-MAT-J03WOH | 1 | live | 46833 | 9.639 | 7.495 | 20.634 | 8.881 | True |
| DFL-MAT-J03WOH | 1 | replay | 46833 | 7.603 | 5.558 | 16.576 | 7.158 | True |
| DFL-MAT-J03WOH | 2 | live | 46445 | 9.674 | 7.355 | 20.586 | 9.071 | True |
| DFL-MAT-J03WOH | 2 | replay | 46445 | 8.316 | 5.941 | 18.608 | 7.65 | True |
| DFL-MAT-J03WOY | 1 | live | 47106 | 8.901 | 6.986 | 18.27 | 8.296 | True |
| DFL-MAT-J03WOY | 1 | replay | 47106 | 7.778 | 5.547 | 16.951 | 6.943 | True |
| DFL-MAT-J03WOY | 2 | live | 49825 | 9.222 | 7.29 | 19.225 | 8.497 | True |
| DFL-MAT-J03WOY | 2 | replay | 49825 | 8.065 | 5.941 | 17.879 | 7.058 | True |
| DFL-MAT-J03WPY | 1 | live | 45687 | 8.624 | 6.807 | 18.171 | 7.902 | True |
| DFL-MAT-J03WPY | 1 | replay | 45687 | 7.22 | 5.442 | 15.847 | 6.472 | True |
| DFL-MAT-J03WPY | 2 | live | 51067 | 9.93 | 7.823 | 20.64 | 9.099 | True |
| DFL-MAT-J03WPY | 2 | replay | 51067 | 8.215 | 5.991 | 18.547 | 7.448 | True |
| DFL-MAT-J03WQQ | 1 | live | 45933 | 8.937 | 6.963 | 18.926 | 8.283 | True |
| DFL-MAT-J03WQQ | 1 | replay | 45933 | 7.526 | 5.532 | 16.589 | 6.897 | True |
| DFL-MAT-J03WQQ | 2 | live | 46070 | 8.684 | 6.929 | 18.134 | 8.028 | True |
| DFL-MAT-J03WQQ | 2 | replay | 46070 | 7.418 | 5.595 | 16.349 | 6.567 | True |
| DFL-MAT-J03WR9 | 1 | live | 46028 | 8.872 | 7.072 | 18.297 | 7.948 | True |
| DFL-MAT-J03WR9 | 1 | replay | 46028 | 7.443 | 5.59 | 15.88 | 6.549 | True |
| DFL-MAT-J03WR9 | 2 | live | 52253 | 9.637 | 7.44 | 20.439 | 8.987 | True |
| DFL-MAT-J03WR9 | 2 | replay | 52253 | 7.815 | 5.731 | 17.318 | 7.196 | True |

live: hidden N=665,019, sample-weighted mean=9.111 m. Overall median/P90 not computed by averaging row quantiles.
replay: hidden N=665,019, sample-weighted mean=7.614 m. Overall median/P90 not computed by averaging row quantiles.

EPL application marker: True; other EPL half exports: 4/4. Hidden accuracy: N/A. Named identities and video calibration remain unverified.

## Training completion

DFL complete folds: 7/7. Trainer finished: True.

- camera_markings: exit code 0, elapsed 93.4 min. Exit code zero alone is not an accuracy gate.
- physical_camera: exit code 0, elapsed 120.1 min. Exit code zero alone is not an accuracy gate.
- appearance: exit code 0, elapsed 9.3 min. Exit code zero alone is not an accuracy gate.
- jersey: exit code 0, elapsed 23.0 min. Exit code zero alone is not an accuracy gate.
- physical_control: exit code 0, elapsed 126.0 min. Exit code zero alone is not an accuracy gate.

## Camera validation

| Camera | Common frames | Feet N | Mean m | P90 m | Gate |
|---|---:|---:|---:|---:|---|
| initial | 197 | 3383 | 5.625 | 12.473 | False |
| selected | 197 | 3383 | 5.954 | 9.874 | False |

Full support and user-landmark results: `camera_evaluation/paired_evaluation/review.html`. No camera test/promotion after a failed validation gate.

preview demo: health, 12 candidate cards, Arsenal API and studio HTTP route verified. Rendered-browser/manual UI review remains required.

trained demo: health, 12 candidate cards, Arsenal API and studio HTTP route verified. Rendered-browser/manual UI review remains required.

The running DFL config retained an old prose phrase “Thirty complete epochs”; actual eligibility checks use the requested150 epochs and equal updates. Source snapshots/configs are retained. This wording discrepancy does not relax the gate.

## Failures and limitations

- Attempt 1: Inactive substitute coordinate None requires explicit NaN padding, not zero or ragged arrays. Handling: True.
- Attempt 2: Legacy exports changed observed coordinates during postprocessing. Handling: True.
- Attempt 3: Double orientation transform changes one-decimal .05 rounding ties for observed players; maximum 0.1414m. Handling: Serialize original already-observed measurement, not double-transformed copy.
- Attempt 4: SkillCorner role_slot contains named positions, not Arsenal numeric formation slots. Handling: Explicit documented position-role mapping; unknown labels rejected; completed preview reused.

closeout_checks.json: [{'command': ['/Users/qnrj/Code/epl-hackathon/.venv/bin/python', '-m', 'compileall', '-q', 'cv', 'backend', 'tests'], 'exit_code': 0, 'output': ''}, {'command': ['git', 'diff', '--check'], 'exit_code': 0, 'output': ''}]

## Original-file preservation

107 manifest checks (overlap), failures: 0. Detailed paths in originals_preservation.json.

Camera/appearance/jersey experiments are not automatically deployed. Unsupported close-up/set-piece angles must abstain. DFL exact IDs/roles/context do not exist reliably in the EPL pipeline. No inferred positions were treated as ground truth. More synchronized licensed EPL video/tracking remains an optional user input.
