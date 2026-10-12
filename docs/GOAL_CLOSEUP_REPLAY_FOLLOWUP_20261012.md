# Arsenal goal, extra views and retrospective motion

## Verified findings

The supplied local Arsenal–Liverpool match has a factual Theo Walcott goal annotation at video1811.68s / clip371.68s, in the31st minute. The original preserved export rounded this to clip370s; the trained export retains371.68s. At video1810s the actual broadcast clock reads30:09; this is consistent with a31st-minute goal. Names/timing come from annotations, not role-position guesses. This does not verify the reconstructed player arrangement.

Current goal calibration has a concrete disagreement: at video1810/1811 the camera-continuity policy rejects fresh image-camera anchors because they differ from its propagated camera by12.70/11.02m on its reference grid. Both are model estimates; the policy's choice is not verified by full ground truth. Goal actor A14 is inferred at clip370–374 even though the actual broadcast shows the scorer. Identity mapping and observation coverage remain unresolved. Do not snap that slot to the ball and call it a measured Walcott position.

The original broadcast cache saved only main-camera images. For example all cached images at1709–1709.8s are absent despite a real-time close-up. Replay and close-up exclusions therefore lose image observations, not just pitch calibration. Replays also show an earlier unknown match time; their screen timestamp cannot be treated as current match time.

## Implemented work

- `backend/reconstruction/replay_bridge.py`: final retrospective bridge after learned corrections/touch cues. Feasible gaps preserve measured endpoints and constrain every adjacent step by the existing8.5m/s model budget while allowing model-derived deviations. Leading/trailing gaps use the available anchor. Off-pitch/missing segments break continuity. Incompatible anchors remain unchanged and produce explicit diagnostics. This is not a live feature; no future observations enter live prediction.
- `cv/build_match.py`: final bridge for new CV exports, with diagnostics stored in source metadata. Original measurements are preserved; no projection outside the camera polygon or fake identity correction.
- `cv/apply_replay_bridge.py`: additive application to the completed trained Arsenal export, `data/cv_arsenal_replay_bridge.json`. Earlier exports and production weights stay unchanged. Anonymous/private reviews and source facts remain distinct from named estimates.
- `cv/review_extra_views.py`: extracts ten real local frames around penalty preparation, behind-goal kick, goal and replay. Records anonymous image detections, box-bottom proxies, clipping, scene/time status and experimentally calibrated pitch coordinates only where current-view line calibration passes the existing plausibility gate. Replay frames never provide current-time pitch observations. Generic local YOLO is used for close-up scales; the GSR detector returned zero detections in several close-ups. Neither detector score establishes position confidence.
- `cv/detect_track.py` and fresh `cv/audit_broadcast.py` runs use class-agnostic suppression for mutually competing person-role labels, recording it in provenance. The private GSR review shows duplicate boxes under different labels. The active cached goal detections came from a generic COCO person-class detector, so this change does not explain or repair that current cached goal arrangement. Existing caches are not silently mixed with new settings.

## Measured results

Arsenal:356 feasible hidden gaps bridged.232 measured-anchor pairs exceed the configured speed budget:205 adjacent observed-to-observed transitions and27 hidden gaps. Some require30–45m/s and need camera/identity/time review. Speeds just above8.5m/s may be real sprints; the budget is not a universal physiological maximum. Hidden-related steps over10m/s remain22 before/after. The remaining discontinuities cannot be removed while preserving contradictory measurements. No teleportation-elimination claim is made.

Same fourteen held-out DFL trained halves, replay only, hidden N665,019: mean7.613803m before and7.613797m after. Difference is practically zero. No parameter was tuned using those scores. Live unchanged. This safeguards motion consistency, not an accuracy improvement. Read native tables in `cv/out/goal_closeup_followup_20261012/dfl_review.html`.

Extra views after class-agnostic detection: close-ups at1669/1725/1747/1817s yield1/3/2/7 anonymous person detections. Behind-goal penalty views1734/1742s yield8/12 detections. Their global pitch calibration remains unsupported. Of ten sampled views, only the two real-time sideline goal images support the experimental line-camera gate; support is not ground-truth accuracy. Replay samples1840/1850 are excluded from current-time inference. Suppression reduces counts at1840 from19 to17 and1850 from14 to13, but leaves goal counts16/18 unchanged; no precision/recall gain is asserted without labels.

## Tests and limitations

Four targeted bridge/touch tests passed. Full suite **149 passed**,250 existing warnings,111.19s (`/private/tmp/football_goal_bridge_tests.log`). Compile/diff checks passed. Actual goal detection image and extra-view outputs were inspected. No frontend source changed this turn; frontend lint/build not rerun (last pass was preceding stoppage follow-up). Demo/API refresh verification pending at report creation.

Commands executed: `.venv/bin/python -m pytest tests/test_replay_bridge.py tests/test_touch_cues.py -q`; full `pytest -q`; three `cv.review_extra_views` runs to distinct directories; `cv.apply_replay_bridge`; fixed14-half DFL scoring; compile/diff. No new training, download, paid call or production checkpoint replacement. Existing seven-hour jobs continue.

Goal spatial fidelity, correct named all-player trajectories, close-up pitch calibration, replay match-time/identity alignment and residual observed jumps remain incomplete. Close-up pixel observations are useful evidence but cannot recover the whole pitch from a cropped torso. Use explicit field landmarks/camera parameters and verified identities before merging extra-view observations into named trajectories. See the separate manual guide for optional user review.

## Runtime verification

Loaded the additive bridge candidate using the project-owned listener guard; training was not stopped. Match API200;601 replay frames; existing penalty stoppage retained; goal API confirms371.68s / Theo Walcott / factual scorer verified. The factual scorer flag does not verify the player-position slot. Fresh detector suppression did not change either sampled goal-frame person count, so it must not be cited as a fix for the current goal reconstruction. Prior cached measurements/model checkpoints remain preserved. Overall outcome: extra-view review and final future-anchor bridge implemented; realistic named goal/set-piece reconstruction and contradictory measured jumps remain unresolved pending calibration/identity evidence.
