# Review the Arsenal goal and close-ups

Open the additive bridge candidate:

`http://localhost:3000/match/cv-arsenal-replay-bridge/studio?part=clip&t=368`

The factual goal is clip371.68s (31st minute). Compare the visible player arrangement with these private local images. The named dots remain unverified role slots; scorer facts do not verify the dot identity. Feasible gaps use the next observation; impossible observed pairs are retained as conflicts, not repaired by moving measurements.

```sh
open cv/out/goal_closeup_followup_20261012/extra_views_nms/review.html
open cv/out/goal_closeup_followup_20261012/review.html
open cv/out/goal_closeup_followup_20261012/dfl_review.html
```

The first page shows ten private broadcast frames with image-space detection markers and native tables. Close-up body-box bottoms are proxies, not verified feet; cropped bodies have no pitch coordinate. Replay samples show **original match time N/A** and are excluded from current-time inference. No broadcast video is embedded in the web application.

Optional actions that would help resolve the remaining errors:

1. At1810 and1811.68s, verify which anonymous detection is Walcott and which defenders are identifiable. Record team/shirt/detection ID and whether the shirt is readable or guessed. These observations must not be treated as ground-truth foot coordinates.
2. For1734/1742s penalty views, identify visible field landmarks with known pitch coordinates. At least four well-distributed non-collinear point correspondences, or sufficient named lines in both pitch directions, are needed; arbitrary shirt clicks cannot calibrate a view.
3. For replay1840/1850s, supply the corresponding original play timestamp if you can establish it. Do not infer it from the replay screen time.
4. Inspect large conflicts in the second page and record camera-angle change, wrong identity or suspect ground-contact point. The reported8.5m/s budget can flag genuine fast sprints too.

No immediate model re-run is needed for the bridge candidate. Fresh class-agnostic detections require a new cache/config and proper validation; existing cached measurements have not been replaced. Training is still bounded by the original seven-hour deadline.

Automated regression:

```sh
.venv/bin/python -m pytest tests/test_replay_bridge.py tests/test_touch_cues.py -q
```

Expected: source coordinates unchanged, feasible paths reach the next sighting within the speed budget, impossible anchors flagged, substitutions/off-pitch gaps not linked. DFL position error should not be described as improved: the measured change was effectively zero.
