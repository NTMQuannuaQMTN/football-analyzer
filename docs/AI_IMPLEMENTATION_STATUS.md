# Implementation status

## 11 October 2026 — return to the archived reconstruction baseline

User requested a return to the beginning of the match view and off-camera prediction work. Current checkout was already at original commit `539f23b`; saved exports retained later transformations. Snapshotted all 19 active exports, restored the Arsenal YOLO11m/offscreen-MLP archive, and removed later camera abstention/display fields in other exports. Verified archived Arsenal xy/obs/ball exactly, unchanged DFL ground truth/observations and 34,028 restored DFL model points. No new training, inference or downloads. Camera diagnostics and verified scorer metadata retained; simple score/date/zone-button/what-if-overlay presentation fixes preserved.

Automated tests: 20 passed; frontend lint/build and diff check passed. Running API serves all 601 archived Arsenal frames with 22 roster slots and no experimental display layer; studio HTTP 200. Browser visual checks and accuracy metrics not measured. Earlier phase reports are absent from this checkout; this document records the reset only and does not reconstruct missing reports.

See [reset report, limitations and manual checks](BASELINE_RESET.md). Snapshot: `data/reset_backups/20261011_113625/`. Reconstruction accuracy remains unresolved. Next step is manual baseline review, followed by evidence-based calibration/visible-player diagnosis before further hidden-player model changes.

## 11 October 2026 — five-hour local training session (running)

User authorised local retraining for visible pitch placement, retrospective hidden-player movement and tactical analysis. Whole-game GSR splits, isolated detector/marking/jersey candidates, and DFL live/replay feature experiments are running. Replay shot-end interval handling, ball/person tracker separation, exact camera footprints, factual scorer resolution, soft touch cues and honest missing-confidence handling have been implemented. Expanded tactical explanations distinguish measured geometry from conditional mechanisms.

No candidate has replaced the restored exports. The two-epoch detector improved held-out recall but worsened precision; optional paint refinement did not improve its held-out controlled test. Learned camera, jersey and hidden-player results are pending. See [running session report](TRAINING_SESSION_20261011.md) and [manual guide](TRAINING_SESSION_20261011_MANUAL_TESTING.md); final regression results and acceptance decisions will be appended when the experiments finish.


## 11 October 2026 — local training session results

Completed two detector runs (five epochs total), 54 marking epochs, four jersey epochs and eight structured hidden-position experiments. Host pauses extended wall time beyond the planned five-hour window; no uninterrupted five-hour GPU run is claimed. The third detector run was skipped at its deadline. All training/evaluation jobs ended.

Continued detector: held-out visible-person precision 90.0%, recall 93.3%, versus archived generic 11m 85.9% / 90.5% on the same 450 images. Hidden replay mean: raw 7.05 → 6.69 m; same-speed-limiter 6.83 → 6.46 m on 97,116 oracle-visible DFL player-timestamps. Arsenal reappearance proxy worsened 18.44 → 18.82 m; jersey identity test failed (2 correct of 7 decided tracks). Camera estimation and tracker identity remain unreliable. No experimental weights or match exports were promoted.

Integrated replay-gate, camera-footprint, factual scorer, soft cue, causal API/history, missing-confidence and conditional tactical-evidence safeguards. Final regression: 42 passed; frontend lint/webpack build, compile and diff checks passed. Actual API and private movie checks ran; browser GUI/playback FPS checks did not run. Reports preserve the initial baseline and unsuccessful experiments.

See [final training report](TRAINING_SESSION_20261011.md), [manual guide](TRAINING_SESSION_20261011_MANUAL_TESTING.md), and local `cv/out/training_20261011/review.html` native tables. Reliable all-player EPL reconstruction remains incomplete; priorities are calibration, association/identity and permitted aligned labels.

## 11–12 October 2026 — eight-hour calibration/identity audit (running)

User authorised eight hours of local engineering, with user actions recorded in Markdown. Started 15:09:31 UTC; deadline 23:09:31 UTC. Initial regression: 42 tests passed; frontend lint and webpack build passed. Active exports/checkpoints are hash-manifested and preserved.

Frozen held-out tracking selection improves anonymous visible-person recall 73.57 → 79.85% and identity F1 47.55 → 51.22%, with precision 90.79 → 89.54%. Integrated camera/tracker evaluation places 57.99% of 6,592 labelled visible observations within 5 m, versus 50.09% for its legacy control; conditional supported mean is 1.945 m. These are one held-out GSR game, not verified EPL names or hidden positions. Learned cameras still fail all five reviewed Arsenal frames. Appearance and physical-camera experiments remain pending. No model/export promotion is justified yet.

See [running audit and evidence](CALIBRATION_IDENTITY_8H_AUDIT.md) and [private review actions](CALIBRATION_IDENTITY_8H_USER_ACTIONS.md). Final regression, candidate decisions and time accounting will be appended after the window.


## 12 October 2026 — eight-hour audit completed

Completed the authorised 15:09:31–23:09:31 UTC local work window; handoff recorded 2026-10-11T23:09:50.412791+00:00. This is elapsed engineering time, not eight hours of GPU training. Heartbeat covers the final approximately 4 h 35 min; initial approximately 3 h 25 min was not heartbeat-instrumented. All training/evaluation jobs ended. Only verified audit helper processes were cleaned up; frontend/backend remain running.

Expanded camera/tracker gates failed. Seven fresh nested-match folds trained 28 models / 840 epochs: live equal-match mean error under stale ball improves 11.779 → 10.827 m, clean error regresses 8.760 → 8.946 m; replay stale 9.664 → 8.806 m, clean 7.443 → 7.694 m. These are German tracking with simulated camera visibility, not verified EPL broadcast accuracy. Geometry diagnostics also have mixed results. No production model/calibration/tracker setting or active export was promoted.

Integrated timestamp-safe profiles/position cards, causal live rendering/formation labels, no cached truth during live loading, projected-measurement conflict warnings, explicit retrospective/ground-truth tactical basis and simulated-camera labels, annotation provenance guards and ownership-safe startup. Final 132 Python tests passed (250 existing warnings), separate actual-component rendering check, lint/webpack build, compile/diff, 69+15 HTTP checks and actual browser probes passed. Preservation: 88 protected assets and 19 active exports unchanged, overlapping inventories. Real human scoring and reliable named/all-player EPL reconstruction remain incomplete.

See [completed audit](CALIBRATION_IDENTITY_8H_AUDIT.md) and [user actions/manual testing](CALIBRATION_IDENTITY_8H_USER_ACTIONS.md). The useful next action is eighteen independent landmark/shirt review frames; no additional training is required to inspect the private outputs.


## 12 October 2026 — user annotations and set-piece display follow-up

Received the user's private CSV unchanged: 40 labels, 17 landmarks on nine frames and 23 shirt/team clicks. Current camera landmark mean/median/P90 **15.55 / 3.79 / 55.23 m**, N=17; frozen temporal/foreground candidates mean 5.06 / 5.05 m on the same selected labels. Residual errors and failed wider gates prohibit a blanket accuracy claim; no model/camera/export promotion. Landmark identities and click precision remain annotation assumptions.

The existing source penalty at 29:02 and five free kicks were excluded by frontend event filtering. Added a native-table Match events overlay, six seekable set-piece markers, richer recent feed and truthful empty tactical-explanation text. A separately declared body-box identity-review mode links nine tracked boxes/five unconfirmed detections, leaves nine unsupported/unmatched and never treats body clicks as foot/pitch truth. A 1510-second supplied team label needs review. All 19 active exports remain unchanged.

Final verification: **134 Python tests passed**, 250 existing warnings; actual event/causal-rendering checks, frontend lint/webpack build, compile/diff and real local browser overlay/timeline/private-table checks passed. Camera/player/ball accuracy and deeper penalty/set-piece tactical interpretation remain incomplete. See [follow-up report](MANUAL_REVIEW_FOLLOWUP.md), [updated actions](CALIBRATION_IDENTITY_8H_USER_ACTIONS.md), and private `manual_review/summary.html`.


User confirmed the1510-second label is Liverpool #14. Corrected only its team field in the private workspace review CSV, archived the original, re-ran foot/body identity reviews and all three camera scores, and updated readable results/instructions. Counts and landmark metrics remain unchanged; the earlier team-label action is resolved. Active match exports and production models are unchanged.


## 12 October 2026 — close-up-aware retraining completed

User requested another training run with set-piece close-ups accounted for. Local training completed 09:59:04–10:08:27 Singapore, approximately 9 min 23 s within the 20-minute cutoff. Frozen ResNet18 view head: 300 updates; existing marking model four complete epochs / 716 updates. All manual-review shots withheld from scene training; replay images excluded; GSR camera train/validation/test match separation retained.

Temporal scene test N=117: all 28 unsupported samples rejected,88/89 annotated sideline retained (25 close-ups, only one alternate sample). This is one-match category recognition, not coordinate accuracy. Selected calibration fails fixed validation gate: common 3,383 feet mean 5.625→5.954m, P90 12.473→9.874m. No camera test/promotion. User-landmark diagnostic: 12/17 supported, selected mean 1.667 m; five unresolved. Same 12 current CCBV mean 19.068 m; initial learned mean 1.625 m, so further training is not a mean-error improvement over that learned initial checkpoint.

137 tests passed / 250 existing warnings; separate frontend event/causality checks, compile/diff and actual figure/table inspection passed. Frontend lint/build not rerun for training-only changes.88 protected assets and 19 active exports unchanged (overlap). See [training results](CLOSEUP_RETRAINING_20261012.md),[manual review](CLOSEUP_RETRAINING_20261012_MANUAL_TESTING.md), private `cv/out/closeup_retrain_20261012_015904/review.html`. Models remain experimental; all-player EPL/close-up global reconstruction remains incomplete.

## 12 October 2026 — seven-hour DFL/EPL application session running

User authorized seven hours of retraining and applying models to the available matches. Session 13:33:49–20:33:49 Singapore; GPU/CPU trainers began about 13:46. Seven DFL whole-match cross-validation folds (150 requested epochs), sequential local MPS perception/identity experiments, additive experimental exports and ground-truth comparisons are running. DFL has tracking but no local broadcast video; EPL has one local Arsenal broadcast and four other half exports without full hidden truth. No inferred labels are used as training targets. Original exports/production weights are retained.

Initial suite137 passed; integration138 passed; focused application/camera four passed. Initial application exposed null substitute arrays and orientation/rounding ties; both are documented and corrected. First preview live hidden error J03WMX half1 N47,800 mean9.153m versus legacy7.822m is worse, with different postprocessing, and is not an improvement claim. Final metrics/application/browser checks are pending. See [running report](SEVEN_HOUR_APPLICATION_20261012.md) and [manual guide](SEVEN_HOUR_APPLICATION_20261012_MANUAL_TESTING.md). Running artifacts record commands, timestamps, folds, gates, failures and actual updates.

## 12 October 2026 — stoppage display and plotting follow-up

Fixed missing EPL stoppage display by connecting explicit SoccerNet stop/restart intervals and existing dead-ball estimates to the DFL-style pitch overlay. Annotations/estimated states have distinct labels; restart-only events do not invent preparation time, and live does not receive future labels. Goal-kick contracts/display are supported only when source annotations exist. Replay no longer blends a measurement into an inference with measured styling, animates large video discontinuities as a run, or flies a dead ball into a restart sample. Coordinates and production models/exports are unchanged.

An API check caught the initial misplaced enrichment hook; corrected with frame-serving integration coverage. Final146 Python tests passed/250 existing warnings; frontend lint/webpack build and actual TSX event/causal/stoppage checks passed. Actual frame/player API calls and a rendered penalty-stoppage screenshot were verified. Camera/identity errors and off-camera reconstruction remain unresolved; no accuracy improvement claimed. Seven-hour GPU work continues independently; DFL trainer completed seven folds and trained application is running. See [follow-up](STOPPAGE_AND_POSITION_FOLLOWUP_20261012.md) and [manual testing](STOPPAGE_AND_POSITION_FOLLOWUP_20261012_MANUAL_TESTING.md).

## 12 October 2026 — Arsenal goal, extra-view and future-anchor follow-up

Goal facts/timing verified against local labels/video: Theo Walcott at video1811.68/clip371.68 in31st minute; spatial arrangement remains unverified. Found current camera propagation rejecting fresh anchors with12.70/11.02m disagreement and role-slot observation gaps despite visible scorer. Private extra-view tool now retains close-up/behind-goal detections; general local detector handles close-up scales better, while current-view pitch calibration remains unsupported there. Replays are explicitly excluded from current-time inference until aligned.

Added final retrospective feasible-gap bridge after learned corrections/touch cues, measured anchors preserved and conflicting pairs reported; no live future leakage. Additive Arsenal candidate:356 feasible gaps,232 speed-budget conflicts (205 adjacent observed pairs),22 large hidden-related steps unchanged. Same14 DFL halves/N665,019 replay mean7.613803→7.613797m: no meaningful accuracy improvement. Fresh person-role class-agnostic suppression recorded in detection provenance; current cached goal input is COCO-person, and sampled goal counts remain unchanged, so this is not a current goal correction.

149 tests passed/250 existing warnings, compile/diff passed; actual private goal image and local candidate frame/goal APIs verified. No frontend source changed; lint/build not rerun this turn. Demo loaded additive candidate; seven-hour training continues. Accurate named goal/set-piece trajectories and camera/identity conflicts remain incomplete. See [report](GOAL_CLOSEUP_REPLAY_FOLLOWUP_20261012.md) and [manual review/actions](GOAL_CLOSEUP_REPLAY_FOLLOWUP_20261012_MANUAL_TESTING.md).


Seven-hour closeout artifacts generated from actual outputs. See `cv/out/apply_train7h_20261012_053349/results.html` and appended recorded results in `docs/SEVEN_HOUR_APPLICATION_20261012.md`. Missing folds/applications remain explicitly incomplete; no model improvement is inferred from exit codes.

## 13 October 2026 — reconstruction training, validation and match application

There is no accessible dataset with broadcast video plus independent all-player truth for the same match, verified locally and on official / paper pages; see [dataset_inventory.md](dataset_inventory.md). The session therefore used a hybrid of three sources:
- **DFL:** direct truth with transferred real camera footprints;
- **SoccerNet-GSR:** real broadcast re-entries, direct truth;
- **SkillCorner A-League:** 20 matches of real broadcast re-entries, proxy truth.

A new unified 7-fold benchmark scores every method on identical rows (1.32 M hidden player-seconds).

**Model selection.** The rule was the mean of full-context and degraded broadcast-context validation-tail error, among models that never trained on the tail. It selected `r13_mlp_aug_ens3`: a 3-seed v3 MLP ensemble with degraded-context augmentation. It passed a non-inferiority gate against the incumbent `k7r_v3` / `final_v3` and was promoted to `inference.PRODUCTION`. The old model can still be selected with `RECON_PRODUCTION=:final_v3`.

| Test | Selected | Incumbent | Formation | Notes |
|---|---|---|---|---|
| DFL held-out | live 8.10 / replay 6.66 m | 8.23 / 6.74 m | 14.28 / 10.54 m | wins 7/7 matches |
| DFL degraded context | 6.18 / 5.43 m | 7.77 / 6.98 m | | |
| GSR | live 6.74 m | 6.99 m | | |
| SkillCorner | live 7.70 m | 7.88 m | | wins 20/20 |

The 80 % radius covers 80 % on DFL and 78 % / 81 % on SkillCorner. It is about 70 % on GSR-like inputs; a game-split scale of about 1.2× fixes that.

**Rejected:** a GRU temporal model was best on DFL (7.72 / 6.37 m) but worse on both real-broadcast tests. This result is reported, not deployed.

**Applied** with the promoted model (held-out folds for DFL):
- 20 A-League matches added to the app (`skc-*`) from SkillCorner broadcast detections. Hidden players are estimated through replays using the match sheet; events are the provider's; goals reproduce the official score in 19/20 matches, and the one gap is flagged.
- 7 DFL matches with a new camera estimate (`dfl-*-r13`).
- The Arsenal clip and 2 legacy SkillCorner EPL matches (`cv-r13-*`).

All 59 exports pass the quality checks. All 32 analyses ran, with basis-tagged evidence chains: [match_analysis_report.md](match_analysis_report.md).

Under reconstruction, team geometry holds within 1.2–3.4 m per minute, but formation labels agree with truth only 57 % of the time; tactical moments are therefore labelled experimental.

**Bugs found and fixed:**
- per-event SkillCorner coordinate normalisation;
- cumulative scores;
- goal-attribution window;
- tracking gaps during replays;
- estimates pulled off the pitch;
- goal without a scorer crashing moments and the API;
- provider-missing statistics shown as 0;
- causal narrative wording softened.

**Checks:** 162 Python tests pass (13 new); frontend lint, type-check and rendering checks pass. API end-to-end checks passed, and lobby / match / studio screenshots were inspected.

**Not achieved:** end-to-end video → hidden-player accuracy against truth, which needs data that does not exist locally. Identity on our own video is also unchanged at about 35 %.

See:
- [evaluation](reconstruction_evaluation.md)
- [model card](reconstruction_model_card.md)
- [training plan](reconstruction_training_plan.md)
- [test guide](reconstruction_test_guide.md)

## 14 October 2026 — follow-up: ten weak areas

1. **Live moments** now appear only once all their evidence is in the past (`available_at`), in both the backend and the studio.
2. **A-League possession changes and kick-offs** are derived from provider possessions. Possession spells, recoveries and counters now exist (14 counterattack moments; previously 0).
3. **Ball not tracked by the provider** is drawn hollow and labelled as its last known position.
4. **Validated confidence** is shown for SkillCorner-derived estimates (live P(err ≤ 5 m) within 0.03). Unreliable team-minutes fell from 64 % to 27 %.
5. **Horizon-aware calibration:** long-unseen coverage is 79.5–80.3 % instead of 77–78 % on held-out DFL (replay worst deviation 4.4 → 1.9 points).
6. **Low-visibility formation shifts** are titled "possible" and capped, based on a DFL check: right formation name in 3 of 9.
7. **Detectors validated** against SkillCorner phase labels. Counter P 0.62 / R 0.23; high press does not work on real data (F1 0.05). A tuned low-block threshold was rejected because of its precision loss.
8. **Identity-swap evaluation:** at about 65 % swaps (our video identity level), error rises from 6.1 to 13.6 m.
9. **`run.sh`** uses the next free port when another program holds 8000 or 3000, and never stops it.
10. **The lobby** hides 37 superseded copies (`SHOW_SUPERSEDED=1` shows them); the live match uses the pin-free r13 copy.

169 Python tests pass (20 in `test_reconstruction_r13.py`); lint, type-check and the frontend rendering checks pass. `./run.sh` was verified end to end with port 8000 occupied, and studio screenshots were inspected.
