# Stoppage and pitch-position follow-up — 12 October 2026

## Findings

The seven-hour training session is still running; it has not established better EPL accuracy. Preview DFL mean hidden error is9.312m live and7.869m replay, versus legacy8.434m and6.884m under different postprocessing. The marking-camera validation gate still fails. EPL has no complete hidden-player ground truth.

Three display/contract problems were reproduced:

1. Arsenal preview contains233 `bs=dead_ball` samples and36 goal-state samples, but zero `ds` fields. The pitch's existing DFL stoppage overlay depends on `ds`, so these EPL stoppages did not appear.
2. Source events include five free kicks and a penalty; a restart timestamp alone does not establish the preparation interval. Stop/restart annotations were disconnected from the frame display.
3. Replay interpolated current observed coordinates toward next-second inferred coordinates, while retaining current observed styling. It also rendered large camera/identity jumps as continuous movement. A dead-ball sample could interpolate toward the next restart position before the state changed.

## Changes

- `backend/analytics/deadball.py`: non-mutating retrospective display enrichment from explicit SoccerNet stop/restart annotations; half-open intervals end at the restart. An intervening annotated shot/goal prevents pairing with a later unrelated restart. Unsupported/unpaired existing dead-ball ball-state samples get an explicitly estimated label; no arbitrary preparation time is invented around isolated restart events.
- `backend/pipeline.py`: apply enrichment to CV replay frames only. Existing DFL `ds` is preserved. Live does not obtain a future annotated restart. Existing stored matches gain display fields without retraining/re-exporting or changing player/ball coordinates.
- `frontend/components/pitch/Pitch.tsx`: distinguish annotated stoppages from estimated ball state and existing DFL truth state. Avoid interpolation across observed/inferred transitions, camera cuts/footprint availability changes or video jumps over12m per source second. Same-status nearby replay samples retain smooth presentation. No observed coordinate is moved or projected outside a camera polygon. Dead-ball rendering holds the current ball sample.
- Event contract/export/frontend accept goal kicks when supplied by the actual source. No absent corners, goal kicks, takers or outcomes are invented.
- Tests cover interval boundaries, missing labels, play resumption, no coordinate mutation, preserving DFL states, live causality and actual TSX rendering.

## Actual output checks

Arsenal preview replay enrichment yields269 stoppage frames:113 free-kick,77 penalty,43 throw-in,35 kick-off and one generic estimated dead-ball sample. Penalty preparation: clip225≤t<302; restart at302 is not marked pending. Free-kick preparation at4≤t<51;51 is not pending. These are annotation-based retrospective intervals, not a learned visual set-piece classifier or exact referee-whistle timings.

Initial targeted tests before changes:9 passed0.66s. New dead-ball/video-mode tests:9 passed0.86s. Actual event-overlay/timeline and pitch TSX checks passed, including no invented corner, truthful label text, future-mutation invariance, video discontinuity handling and no dead-ball flight into restart. Full suite, frontend lint/build and refreshed-browser/API verification are pending at report creation.

Commands executed include `.venv/bin/python -m pytest tests/test_deadball_display.py tests/test_video_modes.py -q`, `node tests/frontend_match_events.cjs`, `node tests/frontend_pitch_causality.cjs`; later full logs are `/private/tmp/football_stoppage_regression.log`, `/private/tmp/football_stoppage_lint.log`, `/private/tmp/football_stoppage_build.log`.

## Position accuracy remains incomplete

The animation fixes do not improve camera-to-pitch coordinates or hidden-player model accuracy. Observed means a detection was mapped to a role slot; it does not verify the named shirt identity or calibration. An unobserved player can be inside the footprint because of a missed detection, calibration error or wrong association. Forcing every estimate outside the polygon would conceal those failures and create edge clustering/teleportation.

The current camera estimate fails its visible-foot validation gate. Off-camera models trained with exact DFL visibility/identities/context do not inherit those reliable inputs from EPL footage. Seven-hour training continues independently; its held-out metrics must determine whether any model improves. Close-ups, reverse angles and replay scenes need separate calibration/support handling; set-piece annotations do not validate player coordinates during those views.

Next accuracy work: matched visible-foot and anonymous-track review on multiple clips, reliable camera/identity labels, scoring perception separately from conditional hidden reconstruction, and verification of ball/possession inputs. Full EPL hidden-position accuracy remains blocked by missing synchronized all-player truth. No new paid computation or footage download is required for these display fixes.

## Final verification

The first API check caught an integration error in this patch: the enrichment call had landed in `player_state` rather than `frames`. Corrected it, added an actual frame-serving integration test, and restarted only verified project demo listeners. First full run145 passed did not detect that misplaced hook and is superseded by the final check. Final full suite: **146 passed**,250 existing warnings,150.76s (`/private/tmp/football_stoppage_regression_final.log`). Corrected targeted integration/video suite10 passed0.37s. Frontend lint and webpack production build passed. Compile/diff checks passed.

Actual local `/match/frames` verification:601 replay frames; free-kick at4, penalty at225/301, kick-off at372; pending labels absent at51/302/407. `/match/players/A14/state` returns200. Browser screenshot of clip225/27:45 was rendered and visually inspected (`/private/tmp/football-stoppage-demo-final.png`): **PENALTY · ANNOTATED STOPPAGE**, yellow ring and source-event feed are visible. Only the isolated browser was closed; demo and seven-hour GPU jobs remain running. DFL trainer finished all seven complete folds; trained application is running. Those metrics are not conflated with this display fix.

Acceptance: missing EPL dead-ball display **fixed for supported source intervals/states**; DFL behavior **preserved in tests**; source set-piece contract/timeline **supported**, including goal kicks only when provided; live future-label leakage **prevented in integration tests**. Learned visual set-piece recognition, missing source annotations, camera/player-identity accuracy and off-camera position accuracy **remain incomplete**. No model accuracy improvement is claimed. No original exports or production checkpoints were modified by this follow-up.
