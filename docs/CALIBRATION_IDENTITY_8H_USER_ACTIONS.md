# Calibration and identity review: actions for you

The automated eight-hour audit is complete. You do not need to rerun training. Existing match exports and production weights remain preserved; experimental candidates were not promoted. The frontend and backend are running for review.

## Latest update — your CSV was received (12 October 2026)

Your 40 labels were copied unchanged and scored. Open the readable results:

```sh
open cv/out/audit8_20261011/manual_review/summary.html
```

The current camera has **15.55 m mean landmark error** across 17 selected landmarks; private continuity candidates are about **5.05 m**, but retain errors and have not replaced the active match. These are landmark results, not all-player accuracy. See [follow-up report and manual tests](MANUAL_REVIEW_FOLLOWUP.md).

**Resolved:** you confirmed the player at second **1510**, (857, 297), is **Liverpool #14**. The saved review CSV and scoring outputs are corrected; the original is retained as `arsenal_manual_review_as_received.csv`. No further action is needed for that team label. Several clicks are on shirts/bodies. Body-box associations are now supported separately and do not verify player feet or pitch coordinates:

```sh
open cv/out/audit8_20261011/manual_review/manual_identity_body_links.html
```

For future identity-only clicks you may mark a body/shirt; for positional references mark the ground-contact point. Eight supplied identity clicks lack current-main-camera support, and one additional body click has no containing detection; they remain N/A. Do not re-label a replay as current play.

Refresh the web app and use **Match events** above the pitch. The source penalty at **29:02** and five free kicks appear in a native table and as amber timeline diamonds. Clicking a clock/diamond seeks to the event. The annotation alone does not establish penalty outcome/taker or verified tactical positioning. No corner was invented for this clip.

Older sections below retain the eight-hour handoff instructions and historical missing-label status; this received-CSV update supersedes that status. You do not need to redo all 18 frames or retrain to inspect these results.

## Start here

Open [the Arsenal review studio](http://localhost:3000/match/cv-2016-08-14-18-00-arsenal-3-4-liverpool-h1-1440-2040/studio?part=clip&t=4), then do the **18-frame private annotation** below. This is the action that can resolve real camera/identity ambiguity. Automatic tests and German tracking experiments cannot substitute for those labels. The active match and production weights are preserved because experimental gates/trade-offs do not justify replacing them.

For a readable overview, run `open cv/out/audit8_20261011/review.html`. Tables are shown normally; the long diagnostic list expands only when you want details. Startup and live/replay testing steps are at the end of this guide.

## What I need from you

1. Review the local Arsenal frames and comparison outputs listed in the audit report. Pause the image; judge the measured player feet separately from the player name/number.
2. For a readable shirt number, record the number and the matching visible player. Use **N/A** when it is obscured; do not infer a number from expected formation or from the goal event.
3. Mark clearly visible pitch landmarks (touchline/halfway intersections and penalty-area corners). Do not mark shadows, advertising borders or airborne goal-frame corners as pitch landmarks. Skip ambiguous landmarks.
4. Record whether each camera overlay agrees with the painted lines, and whether a projected player is on the correct side of each nearby line. A plausible-looking dot is not sufficient evidence.

The local annotation page is ready:

```sh
cd /Users/qnrj/Code/epl-hackathon
open cv/out/audit8_20261011/manual_review/review.html
```

Choose a frame, select a visible landmark or readable shirt number, and click its ground contact point. The optional camera-paint projection shows the existing CCBV estimate; turn it off while marking landmarks to avoid being guided by a potentially wrong estimate. The page uses an HTML table and exports CSV, not a JSON viewer. Copy `arsenal_manual_review.csv` from your Downloads into `cv/out/audit8_20261011/manual_review/`. File-based local storage may vary by browser, so export before closing. No network service is used. Start with seconds **1800 and 1810**; skip replay/close-up frames for current-time position evaluation.

For shirt numbers, choose the explicit Arsenal / Liverpool / N/A team selector. Both sides can have the same number. An obscured number remains N/A; the note is optional.

After copying the CSV, this command scores your landmark clicks against the existing camera estimate and creates a readable HTML table:

```sh
.venv/bin/python -m cv.evaluate_manual_camera --csv cv/out/audit8_20261011/manual_review/arsenal_manual_review.csv
open cv/out/audit8_20261011/manual_review/manual_camera_scores.html
```

Expected: only main-camera landmarks receive errors in metres; replay/close-up landmarks show N/A. The command does not refit the camera or train on your clicks. Shirt labels are inventoried separately because a readable number is not ground truth for off-camera positions. The manual scoring command has automated tests; it has not yet been run against real user annotations because no exported CSV is present.

Images and movies contain existing private broadcast assets; keep them local and do not upload or redistribute them.

## Other review outputs

```sh
open cv/out/audit8_20261011/visible_failures/review.html
open cv/out/audit8_20261011/active_matches/review.html
open cv/out/audit8_20261011/broadcast_cameras/review.html
open cv/out/audit8_20261011/arsenal_timestamp_review.mp4
```

The first page shows scored **GSR visible-player** failures and controls. Green squares are evaluation references; red rings are projected detections. Its error values do not measure Arsenal hidden-player accuracy. The second page checks all 19 active file exports; “passed” means the analysis executed, not that its estimates are correct. The third page compares private Arsenal camera overlays with no independent ground truth. The movie is retrospective and interpolates 1 Hz points for display; its shirt identities remain unverified role assignments.

In the app, check that SkillCorner cards show their source dates (Liverpool–Manchester City: 2019-11-10; Manchester City–Liverpool: 2020-07-02), their SkillCorner provider label, and `? - ?` for unknown scores. Arsenal remains a **clip** with a clip score, not a full-match result. Check the prediction-zone button and measured/estimated labels in the studio. The preserved Arsenal archive has no radii, so it correctly shows `Prediction zones N/A`; SkillCorner sources with heuristic radii support show/hide, with calibration still unverified. Automated HTTP/build and isolated headless-browser checks passed, including animation, annotation click/undo/CSV export and the explanation overlay. Please still judge visible-window responsiveness and annotation accuracy yourself; headless throughput does not verify your screen or the coordinates.

## Data that would unlock hidden-player accuracy testing

If you already have permission to use another EPL match with synchronised full-player tracking, provide its local video/tracking paths, coordinate definitions, sampling frequency, clock offset and usage terms. Do not purchase or download a restricted dataset for this audit. Broadcast video alone cannot reveal ground truth for unseen players.

## What not to do

- Do not move estimates outside the camera simply because they were not detected.
- Do not label future replay images as observations at the current match time.
- Do not overwrite `data/cv_*.json` or `cv/weights/*.pt` to test experimental output.
- Do not treat smooth playback or a factual goalscorer record as verified player identity.

If you cannot review immediately, the audit will still finish with its automated evidence and will mark manually unverified cases explicitly.


## Verified dataset options and remaining permissions

Checked against provider sources on 11 October 2026; no additional data was downloaded.

| Source | Available information | Utility and restriction |
| --- | --- | --- |
| [SkillCorner current open data](https://github.com/SkillCorner/opendata) | Ten 2024/25 Australian A-League matches; 10 Hz player/ball coordinates, possession, camera polygon and `is_detected`; extrapolated hidden positions | Useful for broadcast visibility and tactical examples. Hidden extrapolations are model outputs, not independent hidden ground truth. Repository [MIT notice](https://github.com/SkillCorner/opendata/blob/master/LICENSE) must be retained; it does not grant rights to external broadcast video. This current sample is not EPL. |
| [SoccerNet GSR](https://www.soccer-net.org/tasks/game-state-reconstruction) | Broadcast clips with pitch positions, roles, teams and jersey labels; official metric GS-HOTA | Useful for visible perception/identity. Local six-game evaluation differs from the provider's published splits and is labelled accordingly. The task page alone does not establish redistribution rights; use existing permitted local assets and confirm the dataset/video agreement before publishing. |
| [Metrica sample data](https://github.com/metrica-sports/sample-data) | Two anonymised CSV matches plus a third FIFA-format sample; synchronised tracking/events, documented pitch dimensions | Useful for full-position motion/tactics. No team/competition names establish an EPL sample. The README requests attribution but does not provide a general licence grant for every intended use; confirm terms before public redistribution. Synchronised broadcast footage is not promised. |

For meaningful EPL hidden-player validation, the most useful contribution is an already authorised **paired full-player tracking and broadcast sample**, with the provider's actual usage terms, clock offset and named identity mapping. Event-only EPL data can validate dates/scorers but cannot validate unseen positions. There is no need to buy anything for this session.

## Cached identity fix without rerunning detection

The audit found integer tracker IDs reused after camera resets. New exports separate those IDs by supported camera interval. You can review a candidate from the existing cache without running YOLO again:

```sh
.venv/bin/python -m cv.build_match cv/out/2016_08_14_18_00_arsenal_3_4_liverpool_h1_1440_2040 --a-team home --sampling nearest --out cv/out/audit8_20261011/manual_candidate.json
```

This is an **experimental export**, not a verified repair of all named players. The default source fix qualifies tracks, but no active match has been overwritten. Use `--track-namespace legacy` only to reproduce old unsafe grouping. Do not move or hide failed detections to manufacture agreement with the camera polygon.

The assistant's three provisional 1800-second landmark clicks can be viewed in `manual_review/assistant_provisional_scores.html`; their click precision and landmark interpretation need your confirmation. Please create your own CSV rather than accepting those as reference truth.


## Readable combined results

```sh
open cv/out/audit8_20261011/review.html
open cv/out/audit8_20261011/broadcast/namespace_comparison.html
```

The combined page uses HTML tables and figures. Its hidden-position errors are controlled tracking tests, not verified Arsenal accuracy. The camera experiment's progress section is labelled running until evaluation completes. No JSON inspection is needed.

For private candidate video review, the original timestamp movie remains available. Candidate files in `broadcast/` are experimental replay estimates; all numbered identities remain unverified. The goal record names Walcott, but this does not validate a role-assigned dot's identity. The actual 370-second goal explanation can be opened at:

`http://localhost:3000/match/cv-2016-08-14-18-00-arsenal-3-4-liverpool-h1-1440-2040/studio?part=clip&t=370`

This is a short source clip, not a full-match reconstruction. Broadcast footage is kept out of the main web app.

## Start here: compare actual detections before guessed identities

```sh
open cv/out/audit8_20261011/broadcast/anonymous_review.mp4
open cv/out/audit8_20261011/broadcast/robust_role_review.mp4
open cv/out/audit8_20261011/broadcast_coverage/review.html
```

Both movies cover video seconds 1785–1815. Pause at **15 seconds into the movie**, corresponding to video second 1800. In the anonymous view, white boxes/points are confirmed anonymous tracks; amber boxes are detections without supported projected tracks. Several visible, high-scoring detections are still missing from the projected set. Please judge the foot-to-pitch mapping before trying to judge the numbered identities in the role view.

The role movie includes retrospective hidden estimates and unverified role-assigned numbers. It is a diagnostic candidate, not the active match and not verified Arsenal accuracy. Do not assume that a ring outside a polygon proves its player was off-camera. Manual landmark and readable-shirt clicks above are the most useful next input; no further model rerun is needed from you now.

Paint refinement improved validation but worsened the held-out game; the readable comparison is in `paint_refinement/test.html`. Ground-based tracker compensation also failed validation. These failed candidates have not replaced your active matches.


## Camera jump: the most useful review now

```sh
open cv/out/audit8_20261011/camera_continuity/review.html
open cv/out/audit8_20261011/camera_continuity/original_8s.mp4
open cv/out/audit8_20261011/camera_continuity/guarded_8s.mp4
```

These private movies cover video 1440–1448. Pause at **4.4 seconds** and compare the left-goal camera lines. The temporal candidate fixes this inspected jump but still has major discontinuities elsewhere and is not promoted. Review the new **1444.2 and 1444.4** frames in the annotation page, with projection off, and mark only clearly visible ground-line intersections. Landmark orientation is described in that page; leave ambiguous points unmarked.

After exporting your own CSV, you can compare the exact frozen candidate without retraining:

```sh
.venv/bin/python -m cv.evaluate_manual_camera --csv cv/out/audit8_20261011/manual_review/arsenal_manual_review.csv --camera-run cv/out/audit8_20261011/camera_continuity/appearance_raw --name temporal_camera_scores
open cv/out/audit8_20261011/manual_review/temporal_camera_scores.html
```

Expected: errors in metres for supported exact timestamps, N/A for absent cameras; no refitting or hidden truth. Run the earlier baseline command on the same CSV for a fair landmark comparison. Your CSV is still missing, so real human-grounded scoring remains not run. This review is optional for completing the bounded engineering session, but necessary before trusting EPL camera accuracy.


The remaining-jump gallery is also ready:

```sh
open cv/out/audit8_20261011/camera_discontinuities/review.html
```

It shows ten actual adjacent video-frame pairs and a readable table of image movement versus camera-induced pitch movement. It is selected to expose failures, so do not interpret six corrected examples as a general 60% accuracy score. Four examples remain badly wrong. These images are private review only.

## Latest candidate: foreground-masked camera motion

```sh
open cv/out/audit8_20261011/camera_foreground_arsenal/review.html
open cv/out/audit8_20261011/camera_foreground_arsenal/jumps/review.html
open cv/out/audit8_20261011/camera_foreground_arsenal/foreground_8s.mp4
```

Compare the original eight-second movie above with this candidate at **4.4 seconds**. The candidate masks detected people out of camera-motion features; it does not move hidden dots outside the camera. Nine of ten selected worst cases become plausible in speed, but major jumps elsewhere remain. This is a private experiment and has not changed your active match.

Use your independently marked CSV to compare it directly:

```sh
.venv/bin/python -m cv.evaluate_manual_camera --csv cv/out/audit8_20261011/manual_review/arsenal_manual_review.csv --camera-run cv/out/audit8_20261011/camera_foreground_arsenal/appearance_raw --name foreground_camera_scores
open cv/out/audit8_20261011/manual_review/foreground_camera_scores.html
```

The most useful action remains **independent ground-landmark clicks with projection turned off**, followed by readable shirt/team labels. No extra training command is needed from you. Do not distribute these private broadcast movies.

Additional readable audits:

```sh
open cv/out/audit8_20261011/hidden_multimatch/review.html
open cv/out/audit8_20261011/dfl_input_audit/review.html
```

The multi-match table reports seven German matches with simulated broadcast visibility, not seven verified EPL reconstructions.

Two further data sources were checked: [DFL tracking/event data](https://www.nature.com/articles/s41597-025-04505-y) is CC BY 4.0 with footage excluded; [Hudl/StatsBomb open data](https://github.com/hudl/open-data) lists EPL 2015/16 and 2003/04 event/lineup data, without continuous hidden tracking. Its [custom licence](https://github.com/hudl/open-data/blob/master/LICENSE.pdf) restricts redistribution and commercial exploitation and includes accreditation obligations. If you later want it in a public/commercial hackathon demo, confirm the intended-use terms first. No match data was downloaded or imported, and no message was sent on your behalf.

## Remaining failures and readable-shirt review

The larger forty-clip validation check failed the original camera quality gate. The encouraging selected examples have not replaced your match. Review the candidate's remaining failures here:

```sh
open cv/out/audit8_20261011/camera_foreground_arsenal/remaining_jumps/review.html
open cv/out/audit8_20261011/dense_foreground_full_validation/val.html
```

The annotation page now includes **1603.2/1603.4 and 1961.2/1961.4 seconds**, alongside the previously fixed 1444.2/1444.4 example. Please mark independent ground-line intersections with projection off. For readable shirts, choose the explicit **Arsenal / Liverpool / N/A** team selector and click the player's ground contact point. Leave unclear numbers or teams N/A.

After exporting the real CSV, this optional command links your clicks to anonymous image tracks:

```sh
.venv/bin/python -m cv.evaluate_manual_identity --csv cv/out/audit8_20261011/manual_review/arsenal_manual_review.csv
open cv/out/audit8_20261011/manual_review/manual_identity_links.html
```

Expected: a table showing linked anonymous IDs, unconfirmed detections, ambiguous/missing cases and conflicting human labels. This does not establish automatic jersey recognition, pitch accuracy or unseen-player truth. Older CSV files without an explicit team remain incomplete for named-label consistency; no team is guessed from notes. Your annotations are evaluation-only. Do not use the deliberately fabricated CSV in `browser/test_downloads/`; the importer rejects those UI-smoke rows.


## Start the app and check the latest review safeguards

If it is already running, open [the Arsenal studio at 24:04](http://localhost:3000/match/cv-2016-08-14-18-00-arsenal-3-4-liverpool-h1-1440-2040/studio?part=clip&t=4). Otherwise:

```sh
cd /Users/qnrj/Code/epl-hackathon
MATCH_FILE=data/cv_2016_08_14_18_00_arsenal_3_4_liverpool_h1_1440_2040.json NARRATIVE_BACKEND=local ./run.sh
```

Startup now refuses to stop another project's listener on 3000/8000. If it reports a foreign listener, close that application yourself and retry; no unrelated process is killed automatically. This exact project startup was checked. The existing `.venv/bin/python` is present in this workspace; run the commands from the project root. If using a different checkout, create its environment using that checkout's setup instructions first.

1. Pause at 24:04 and click Arsenal #24. Expect **PROJECTED**, a **Review camera or identity** warning and **CLIP ESTIMATES through 24:04**. The flagged 35.67 m/s is an inconsistent projection/role identity, not Bellerin's verified running speed. Positions remain unverified; apparent speed statistics apply their existing movement filters and need not equal the raw two-point diagnostic.
2. Seek backward/forward. Player statistics should follow the selected clock and never show a later minute. CV live/truth requests remain unsupported; the clip is retrospective. Other tracking/synthetic sources expose separate live/replay modes where available.
3. Expect **Replay poss (3')** and **retrospective** tactical summary labels. Those existing smoothed summaries are not causal real-time predictions. Missing detections can include people inside the camera area; do not rate an inferred point correct simply because it lies outside the polygon.
4. Check the single prediction-zone button; Arsenal's archived radii are absent, so **Prediction zones N/A** is expected. No radii are fabricated. Broadcast clips stay in private local review files, not the application.

Additional readable private pages and video:

```sh
open cv/out/audit8_20261011/review.html
open cv/out/audit8_20261011/ball_context/review.html
open cv/out/audit8_20261011/appearance_full_validation/review.html
open cv/out/audit8_20261011/context_fixed_budget/paired_review.html
open cv/out/audit8_20261011/context_crossvalidation/review.html
open cv/out/audit8_20261011/camera_foreground_arsenal/unconfirmed_review.mp4
```

The last movie shows anonymous confirmed/unconfirmed detections for 1785–1815 seconds, not a verified named 22-player lineup. Tables distinguish counts, mean/p90 errors, coverage and live/replay conditions. The full camera/tracking gates failed, and stale-context training has clean-input trade-offs; these candidates have not replaced your active matches.

**The useful action for you is the 18-frame annotation page above**, especially 1444.2/1444.4, 1603.2/1603.4 and 1961.2/1961.4 seconds. Projection off for landmark clicks; explicit team for readable shirts; N/A for anything unclear. Export and copy the real CSV to the stated private folder. No extra training, paid service or footage download is needed. Hidden-player EPL accuracy remains unverified without permitted synchronised full-player position labels.


For the downstream tactical trade-offs, open:

```sh
open cv/out/audit8_20261011/context_tactics/review.html
```

Expect separate live/replay, exact/stale-context rows and per-match counts. Improved average player error does not mean every tactical metric improves: width can regress while defensive-line error improves. The table scores geometry only, not a coach's intention or an alternate match outcome.

When seeking backward with a player selected, the popup should clear a later state while the current request loads; it should not show a future position/history at the earlier clock. Forward playback may briefly show a state at most one second old. The larger player statistics panel also rejects future-time responses. Both behaviours were checked against actual local API responses; no predicted coordinates were fabricated for the test.


## Live versus replay display check

On a DFL or synthetic source with Truth / Camera replay / Camera live buttons, switch from Truth to **Camera live**. Expect a loading message while that mode's positions arrive, followed by **Live · 1 s samples**. Cached truth must not appear briefly under the live label. If the request fails, use Retry. **Now: N/A** is expected while loading.

Camera live holds the latest available 1 Hz estimate, so movement can step once per second. Camera replay interpolates between samples and can use later observations; that smoother playback is retrospective. Arsenal's CV archive supports clip replay, not verified live ingestion. The behavioural future-mutation check can be rerun with:

```sh
node tests/frontend_pitch_causality.cjs
```

Expected: “Causal pitch rendering: future mutation invariance and retrospective interpolation passed.” It tests actual rendering behaviour, not player-position accuracy. Local 8×/32× selected-player replay checks measured about 60/58 pitch-position updates per second; please still judge your own visible window.


The tactical panel now states its analytical basis. On a truth-default DFL/synthetic match, its prebuilt retrospective summary uses tracked/synthetic ground truth even when you switch the pitch to Camera live; those are separate estimates. Mode-specific causal tactical analytics remain unfinished. CV clip summaries use unverified source measurements/estimates.

Do not submit `assistant_provisional_1800.csv` as human labels. It contains three expressly provisional assistant readings; the scorer now rejects it under the default human basis. Your independent CSV should come from your own review of the page. The UI-smoke CSV is also rejected. Final automated verification: **132 Python tests passed**, separate causal-rendering check passed, frontend lint/production build passed. Real human accuracy scoring remains not run.

## DFL camera visibility is simulated

In the DFL tracking demo, **SIMULATED CAMERA** means transferred broadcast-shaped footprints are moved onto this match's ball. They are useful controlled visibility masks; they are not the measured camera coverage of the original DFL broadcast. Check real calibration/identity with the private Arsenal frames above. The live view holds one-second samples; replay interpolates between them.

## Close-up-aware training completed (12 October 2026)

See [the new training report](CLOSEUP_RETRAINING_20261012.md) and [its review guide](CLOSEUP_RETRAINING_20261012_MANUAL_TESTING.md). Open `cv/out/closeup_retrain_20261012_015904/review.html` for native tables and actual examples. The view classifier and four calibration epochs completed; the broader camera gate failed, so the active match remains preserved. All manual-review shots stayed outside scene training, and your world labels were scoring-only. No immediate further action is required from you.
