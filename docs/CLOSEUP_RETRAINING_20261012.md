# Close-up-aware retraining — 12 October 2026

Status: training and frozen evaluation completed. Local start **09:59:04 Singapore / 01:59:04 UTC**. Overall training cutoff **10:19:04 Singapore / 02:19:04 UTC**, with a separate fifteen-minute calibration limit and four maximum calibration epochs. Models remain private candidates; existing match exports and production weights are preserved.

## Why this run

The user's corrected landmark review demonstrates camera errors before hidden-position inference. Training a hidden-position model again with the same wrong visible coordinates would not establish a repair. This run trains view handling and continues the existing pitch-marking model under explicit comparisons.

The Arsenal penalty's local annotation interval **1731.975–1745.697 seconds** is a real-time behind-goal view. Close-ups surround it and subsequent replay views have their own timestamps. A close-up can identify a shirt/body, but cannot necessarily establish a globally calibrated ground position. A behind-goal image may be useful with a separate supported calibration; it must not inherit sideline-camera assumptions. Replay frames must not update current-time observations.

## Stage 1: view recognition

A frozen locally cached ResNet18 image encoder and trained logistic head recognise the annotated sideline category versus close-ups, alternate angles and other views. Encoder features read only the current image, with whole-frame letterboxing. No explicit ball, player coordinate, shirt identity, manual world coordinate, numeric event timing or future image is an inference feature. Visible scoreboard/graphics remain in the image; implicit visual correlations cannot be excluded. The image head does **not** infer replay time, calibration validity or player coordinates.

- Fixed seed 12026; 300 full-head optimisation steps completed, in about 60 seconds including extraction.
- Only the existing permitted local Arsenal video/annotations and locally cached ImageNet weights; no download/service call.
- Samples every three seconds, one-second shot-boundary margin, fifteen-second temporal-block boundary buffer.
- Entire shots stay in one split. Validation midpoint blocks 2/6, test 4/8 of 300 seconds; remaining eligible shots train.
- **All manual-review shots withheld** (including previously selected frames without a CSV click); replay-labelled images excluded from every split to avoid duplicated replay-origin footage.
- Train: 334 samples / 90 shots (242 annotated sideline, 92 other). Validation: 118 / 23 shots. Test: 117 / 35 shots. Separate manual-shot review: 71 / 10 shots.
- Feature normalization uses training samples only. Training/threshold selection never uses test or manual-review labels. Optional event-nearby group labels enter scoring metadata only.
- Validation gate: at least 20 accepted samples, ≥98% category precision and ≤5% unsupported false-accept rate; widest eligible fixed threshold from .5/.7/.85/.95/.99. Selected epoch 300 / threshold 0.5.

| Split | N | Annotated sideline | Unsupported | Sideline retained | Unsupported falsely accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 118 | 94 | 24 | 94 | 0 |
| Held-out temporal test | 117 | 89 | 28 | 88 | 0 |
| Separate manual-shot review | 71 | 56 | 15 | 56 | 0 |

The fixed grass-coverage baseline (>35% green pixels) falsely accepts **9/24 validation**, **18/28 test** and **7/15 review** unsupported samples. These are view-category tests, not coordinate accuracy. Within the held-out set-piece-nearby subset (±15 s), **N=42**: 28 annotated sideline retained, 14 unsupported rejected. Test includes 25 close-up samples and **only one alternate-angle sample**; review has eleven close-up and four alternate samples. These counts are too small to establish broad alternate-angle accuracy.

An actual six-frame gallery was generated and inspected. The sole test disagreement at **1227 seconds** is annotated “Main camera center,” but the image visibly has tightly framed large players. That is consistent with conservative view rejection; it also demonstrates that camera-origin annotations are not independent labels of zoom/geometry support. The reported score remains the original annotation comparison; no test label was silently changed after seeing the result. Raw scores are uncalibrated. Results from temporal parts of one match do **not** establish generalisation across EPL matches or stadiums.

Private results: `cv/out/closeup_retrain_20261012_015904/views/review.html`, `results.json` (machine metrics), `actual_view_examples.jpg` and saved source/config/checkpoint.

## Stage 2: pitch-marking calibration continuation

Existing `ResNetPitch`, cleaned GSR training rows and the earlier frozen checkpoint are reused. Training games 4/6/9; validation 3/5; previously reviewed game2 remains excluded from updates and selection. The Arsenal CSV is not a training target. No architecture replacement.

Maximum four full epochs / fifteen-minute stage budget; geometry measured every epoch on the same **250 validation images**. Full 1,000-image line validation is separate from quarter-sample coordinate evaluation. Train loss/dice do not prove positional improvement. Partial epochs, if the time limit interrupts one, are explicitly flagged.

A frozen initial geometry result is saved before updates, and its state is retained as the best checkpoint. A worse continuation cannot replace it merely because it is the first new epoch. Selection uses the existing 20 m capped error/abstention loss; this is not reported as position MAE. Supported uncapped error and coverage are separate.

Starting geometry: **197/250 supported frames**, **3,383 labelled people**, mean **5.625 m**, median **1.345 m**, P90 **12.473 m**, capped selection cost **7.033 m**. This is GSR camera-only evaluation at labelled visible feet, not detection/named identity/hidden/EPL accuracy.

Epoch 1 completes all 179 training batches but worsens mean to 8.020 m and selection cost to 7.336 m; the starting state is retained. Final epoch and paired comparisons follow below. An operational camera gate remains ≥40% supported frames, supported mean ≤3 m and P90 ≤5 m. Failed gates must not be relaxed to deploy this run.

## Files and checks

Changed/created: `cv/broadcast_views.py`, `cv/train_broadcast_views.py`, `cv/train_camera_audit.py`, `cv/audit_closeup_retrain.py`, `tests/test_broadcast_views.py`, this report/manual guide, cumulative status. Existing runtime inference, frontend and active match data have not been replaced by this experiment.

Baseline relevant camera/manual checks actually ran before edits: **20 passed / 2.22 s**. New focused checks: **23 passed / 2.02 s**. Full regression after changes: **137 passed / 250 existing NumPy warnings / 165.10 s**. Compile and diff checks passed. Frontend lint/build were not rerun in this training-only change; their prior follow-up passes remain recorded in `MANUAL_REVIEW_FOLLOWUP.md`.

Executed training commands (the saved run.json records the exact Python path and historical absolute deadline):

```sh
.venv/bin/python -u -m cv.train_broadcast_views --out cv/out/closeup_retrain_20261012_015904/views --device mps --epochs 300 --deadline 1791771544.1862578
.venv/bin/python -u -m cv.train_camera_audit --arch resnet18 --rows cv/out/audit8_20261011/camera_training/rows.json --initial cv/out/audit8_20261011/camera_resnet_clean/best.pt --device mps --epochs 4 --geometry-every 1 --hours .25 --deadline 1791771544.1862578 --out cv/out/closeup_retrain_20261012_015904/camera
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q cv tests
git diff --check
```

For reproduction later, use a new output directory and an explicitly new future deadline. Existing weights are required locally; these commands do not download them. The source/config versions are retained with the experiment. No paid services, restricted data acquisition or external messages.

## Acceptance / remaining work

| Criterion | Current result |
| --- | --- |
| Real local training executed | Both stages complete |
| Set-piece close-ups/alternate shots distinguished from sideline assumptions | Category dataset/tests and separately grouped measurements complete; alternate sample support limited |
| Replay/current timestamps remain separate | Replay images excluded; existing runtime safeguards unchanged |
| Manual labels kept outside training | All corresponding shots withheld; manual coordinate labels never used |
| Improved calibration under the same evaluation conditions | Not established; camera gate failed |
| Accurate all-player Arsenal positions / close-up global reconstruction | Not achieved/verified |
| Safe production promotion | None; no evidence sufficient for a blanket replacement |

The user does not need to retrain or relabel the same forty CSV rows. The highest-value additional evidence is another permitted match/angle with independent visible ground landmarks and corrected shot/zoom labels. Exact global positions cannot be recovered from a tightly cropped face alone.

## Final results and closeout

Actual training ran **09:59:04–10:08:27 Singapore**, **562.9 elapsed seconds** (~9 min 23 s). View stage 59.78 s; calibration 503.11 s; both exit 0. These are local stage wall times, not measured GPU utilisation. Four calibration epochs each completed 179 batches, **716 updates** total; 300 full-head updates used frozen image embeddings. No training remains running.

| Epoch | Supported frames /250 | People N | Mean m | Median m | P90 m | Capped selection cost m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 197 | 3383 | 5.625 | 1.345 | 12.473 | 7.033 |
| 1 | 198 | 3396 | 8.020 | 1.465 | 14.962 | 7.336 |
| 2 | 196 | 3361 | 14.804 | 1.439 | 10.962 | 7.039 |
| 3 | 198 | 3405 | 6.032 | 1.461 | 10.566 | 6.779 |
| 4 | 196 | 3362 | 69.985 | 1.382 | 13.817 | 7.239 |

Epoch three was selected by the original capped error/abstention objective. This does **not** establish lower uncapped position error. On the common **197 frames / 3,383 people**, initial→selected mean is **5.625→5.954 m**, median **1.345→1.446 m**, P90 **12.473→9.874 m**. One newly supported frame adds 22 observations; it does not explain away the common-sample mean regression. Both initial and selected fail the fixed ≤3 m mean/≤5 m P90 gate. Epoch four's 69.985 m supported mean illustrates severe residual homography outliers; it was not selected. Camera test scoring is **not run: validation gate failed**; no promotion.

The independently supplied user landmarks are evaluation-only and use image-predicted lines, never reference fitting. Both learned cameras solve **12/17** labels (five unscored). Initial mean/median/P90: **1.625 / 1.273 / 3.170 m**; selected: **1.667 / 1.687 / 2.712 m**. On those **same twelve** labels, current CCBV mean is **19.068 m**. This shows a useful supported-region alternative to the current bad mapping, not a claim that further training improved the initial learned camera or reconstructs every player. The actual selected 1444.4 overlay was inspected.

Additional command actually executed twice (second pass corrects descriptive camera provenance and records hashes; unchanged predictions/selection):

```sh
.venv/bin/python -u -m cv.audit_closeup_retrain --root cv/out/closeup_retrain_20261012_015904 --device mps
node tests/frontend_match_events.cjs
node tests/frontend_pitch_causality.cjs
```

The separate actual-component event and causal-rendering checks passed after training. Full regression remains 137 passed / 250 existing warnings; no new runtime inference behaviour was introduced by the private evaluator. Compile/diff checks passed. Private HTML pages were checked for native tables, and actual view/overlay image outputs were inspected. Frontend lint/build: **not rerun**, because no frontend change in this training session; previous verified passes are unchanged.

Preservation recheck: **88 protected assets unchanged** (six CV exports plus 82 production checkpoints), and **19 active exports unchanged**, overlapping inventories. Recorded in `preservation_verified.json`. The corrected CSV remains evaluation-only. All checkpoints/figures/features are under ignored `cv/out/closeup_retrain_20261012_015904/`; source/config versions and exact execution times are retained. No paid calls, purchases, footage downloads or external training.

Open the root `review.html` for native tables and links. The current app still uses the preserved source export: this run does not yet repair its player/ball paths. The safe next engineering work is an independently evaluated camera selector with paint/geometry checks and temporal propagation, plus per-shot alternate-camera support. A scene-category score alone cannot validate a homography. Further indiscriminate training on the same data is not justified by these mixed results.

Final focused checks after the private evaluator/provenance update: **9 passed / 0.10 s**, executed with `.venv/bin/python -m pytest -q tests/test_broadcast_views.py tests/test_manual_camera_evaluation.py tests/test_manual_candidate_camera.py`. `git check-ignore` confirmed candidate weights/review remain private. Local backend health returned `ok: true`; frontend studio HTTP200. No pending training job.
