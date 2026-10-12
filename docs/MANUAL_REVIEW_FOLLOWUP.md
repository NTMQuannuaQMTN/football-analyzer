# Manual review and missing set pieces — 12 October 2026

The supplied annotations confirm a substantial camera-calibration problem. This follow-up exposes the existing set-piece events and scores the uploaded labels; it does **not** claim to repair the entire match reconstruction.

## Inputs and camera results

Copied `/Users/qnrj/Downloads/arsenal_manual_review.csv` unchanged to the private review directory. SHA-256: `7beaafbfa10f12661ea5d5f80996e66af3db43a5c650176aff7499d5abd7dd14`. It contains 40 labels: 17 landmarks on nine selected frames and 23 shirt/team clicks. Labels are evaluation-only; no camera/model was fitted or trained on them. Scoring assumes the supplied landmark identities, world coordinates and click locations are correct. These selected failure frames are not a random or match-wide accuracy sample.

| Camera | Scored landmarks | Mean error m | Median m | P90 m |
| --- | ---: | ---: | ---: | ---: |
| Current CCBV | 17 | 15.55 | 3.79 | 55.23 |
| Frozen temporal candidate | 17 | 5.06 | 3.69 | 9.31 |
| Frozen foreground candidate | 17 | 5.05 | 3.69 | 9.27 |

At video 1444.4 seconds, the current camera's three errors are 49.42, 81.29 and 63.95 m. The adjacent frame has essentially the same view. The private foreground candidate reduces these to 1.51, 9.42 and 5.35 m, but the 1961.4-second landmark still has 20.72 m error. The wider camera validation previously failed. No candidate was promoted, and no player/ball coordinate or active export was rewritten.

The screenshot's projected yellow markings visibly miss painted pitch lines. A player being classified as measured only means that a detection was used; its pitch position and named identity can still be wrong. Missing detections do not prove a player is outside the view.

## Shirt clicks

The strict ground-foot association links none of the 23 clicks: eight lack a supported current-main-camera measurement and fifteen have no detected foot within the fixed 20-pixel rule. Inspected clicks at 1510 and 1810 seconds lie on bodies/shirts, not feet. The interface's earlier ground-contact instruction was insufficiently clear for this usage.

Added an explicit **body** review mode. It links only a unique containing current-image person box, rejects overlapping boxes and unsupported camera shots, and never produces a ground-contact reference or a pitch-accuracy score. It links nine tracked boxes and five unconfirmed detections; nine are unsupported/unmatched. No linked-track conflicts were found under the supplied labels, which does not prove identities are correct.

**Resolved by your confirmation:** the player at video second 1510, (857, 297), is Liverpool #14. Corrected only that team field in the saved workspace CSV and retained the as-received original. The Downloads file remains unchanged. For positional references click feet; for identity-only review you may click the shirt/body and use the explicitly labelled body mode. Replay/close-up labels remain useful visual notes but cannot verify current match positions.

## Missing penalty and set pieces

The preserved export already contains one SoccerNet penalty annotation at clip second **302 / displayed clock 29:02**, and **five free kicks**. There are no corner annotations in this ten-minute clip. The frontend feed excluded these event types, while the story timeline only rendered detected tactical moments. This was a presentation omission, independent of calibration/tracking errors.

Changes:

- Added a **Match events** button above the pitch. Its overlay uses a native HTML table with clock, event, team and source; clock links seek to the event.
- Added six amber set-piece timeline diamonds: the existing penalty and five free kicks. A marker seeks to the event without inventing a tactical narrative.
- Included restarts, penalties, fouls and cards in the recent event feed; preserved direct/indirect free-kick labels.
- Removed `undefined` shot details when outcome/xG is absent.
- Clarified that no detected tactical explanation does not mean no recorded incident exists.
- Retained the existing live event-window filtering. Replay permits navigation through its recorded clip events.

A penalty annotation alone does not establish the taker, whether it was saved/missed, or what the positioning caused. Those require separately aligned evidence. This change makes the event visible; a deeper set-piece tactical explanation remains incomplete because the coordinates are unverified.

## Files changed/created

`frontend/lib/matchEvents.ts`, `frontend/components/studio/MatchEvents.tsx`, `frontend/components/studio/Studio.tsx`, `frontend/components/timeline/MatchStory.tsx`, `frontend/components/insights/MomentPanel.tsx`, `cv/evaluate_manual_identity.py`, `tests/test_manual_identity_association.py`, `tests/frontend_match_events.cjs`, this report, the user-action guide and cumulative status. Private CSV, scores, summary table and screenshots are under ignored `cv/out/audit8_20261011/`.

## Commands actually executed

```sh
cp /Users/qnrj/Downloads/arsenal_manual_review.csv cv/out/audit8_20261011/manual_review/arsenal_manual_review.csv
.venv/bin/python -m cv.evaluate_manual_camera --csv cv/out/audit8_20261011/manual_review/arsenal_manual_review.csv
.venv/bin/python -m cv.evaluate_manual_camera --csv cv/out/audit8_20261011/manual_review/arsenal_manual_review.csv --camera-run cv/out/audit8_20261011/camera_continuity/appearance_raw --name temporal_camera_scores
.venv/bin/python -m cv.evaluate_manual_camera --csv cv/out/audit8_20261011/manual_review/arsenal_manual_review.csv --camera-run cv/out/audit8_20261011/camera_foreground_arsenal/appearance_raw --name foreground_camera_scores
.venv/bin/python -m cv.evaluate_manual_identity --csv cv/out/audit8_20261011/manual_review/arsenal_manual_review.csv
.venv/bin/python -m cv.evaluate_manual_identity --csv cv/out/audit8_20261011/manual_review/arsenal_manual_review.csv --point-kind body --name manual_identity_body_links
.venv/bin/python -m pytest -q
node tests/frontend_match_events.cjs
node tests/frontend_pitch_causality.cjs
cd frontend
npm run lint
npm run build -- --webpack
cd ..
.venv/bin/python -m compileall -q cv tests
git diff --check
```

The full Python baseline ran before edits: **132 passed / 250 existing warnings / 102.68 s**; baseline lint passed. Final results follow below. The first event-rendering test failed because its regex greedily combined five attributes into one match; the bounded attribute matcher was corrected and the actual component test passed. An attempted read of `backend/tactics/moments.py` failed because the detector is actually `backend/moments/detector.py`; the correct implementation was inspected. No test pass is inferred from source reading.

## Manual testing

1. Refresh the [Arsenal studio](http://localhost:3000/match/cv-2016-08-14-18-00-arsenal-3-4-liverpool-h1-1440-2040/studio?part=clip&t=4). Click **Match events** above the pitch. Expect a table overlay, no JSON viewer and no embedded broadcast video.
2. Scroll **inside the overlay** to the penalty row at 29:02. Click its clock. Expect the match clock and recent feed to show the source penalty. It may be a camera cut; the displayed inferred positions are not verified penalty setup.
3. Close the overlay. Expect six amber diamonds along the story timeline. Click the direct free-kick diamond at 24:51; expect that clock and event label.
4. Open `cv/out/audit8_20261011/manual_review/summary.html`. Expect the three-camera comparison table, per-landmark errors, and separate strict-foot/body-box links.
5. The 1510-second team-label warning is resolved: the corrected review shows Liverpool #14. The as-received CSV is retained for provenance.

Actual isolated Chrome checked the running app: one event table, penalty jump to 29:02, free-kick jump to 24:51, six timeline markers, zero embedded videos and zero runtime exceptions. `manual_followup/events_overlay.png` and `penalty_selected.png` were captured and visually inspected. All 19 active exports remain byte-identical (`manual_followup/exports_preserved.json`). Existing frontend/backend servers remain running.

## Acceptance and next steps

| Criterion | Result |
| --- | --- |
| Supplied manual labels scored against unchanged current/candidate cameras | Passed under stated annotation conditions |
| Penalty/free-kick events visible and seekable in the actual app | Passed |
| Native readable tables and private-only broadcast review | Passed |
| Shirt clicks distinguished from ground-contact references | Passed; the 1510-second team label is now confirmed/corrected |
| Accurate full-match camera/player/ball reconstruction | Not achieved |
| Evidence-based penalty outcome/taker/set-piece tactical mechanism | Not verified/implemented |

Next priority: correct the calibration and named association under separate held-out gates; add more independent landmarks/angles and another permitted clip before promoting a camera candidate. Rebuild ball/player paths only after that gate, keeping detected/held/interpolated sources distinct. Do not repair apparent geometry by arbitrarily moving players outside a bad camera polygon.

## Final verification

Full regression after the body-association changes: **134 passed / 250 existing NumPy warnings / 388.54 s**. The separate actual-component event overlay/timeline test and causal-pitch rendering test passed. Frontend lint and webpack production build passed; compile and diff checks passed. An optional collection check completed with 134 tests collected; collection is not a test pass. Runtime varied substantially from the 102.68-second baseline; no inference/performance improvement is claimed from these test durations.

The three private review pages were checked in actual isolated Chrome: summary **two tables**, camera scoring **one**, body association **two**; zero raw JSON blocks, zero embedded videos and zero runtime exceptions. The actual summary screenshot was inspected. The audit-owned isolated browser was then closed with its exact process/profile verified; user browsers and demo servers were left running. No training/model promotion, match regeneration, paid call or data download occurred.

## Confirmed team-label correction

You confirmed Liverpool #14 at second1510. Only that CSV team field changed; all40rows, clicks and landmark labels were preserved. Original CSV: `manual_review/arsenal_manual_review_as_received.csv`; corrected SHA-256: `b6443e79ed3c9621105a90fa5c1323d696f0949c0d33d5baa3e60d6bb6af69be`. Re-ran both identity-review modes and all three unchanged-camera scorers. The corrected row links to anonymous image track91 in explicit body mode. Association counts and landmark metrics are unchanged. No source code/model/active match changed, and no additional unit-test run was needed for this single-label correction.
