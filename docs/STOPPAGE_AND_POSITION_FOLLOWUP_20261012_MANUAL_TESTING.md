# Test stoppages and plotting

Open the current experimental Arsenal clip:

`http://localhost:3000/match/cv-trial7h-arsenal-preview/studio?part=clip&t=4`

1. At clip4s (24:04), expect **FREE KICK · ANNOTATED STOPPAGE** with a yellow ball/ring. Seek51s: that pending label ends at the annotated restart. This is retrospective annotation evidence, not visual recognition.
2. Seek225s (27:45) and301s (29:01): expect **PENALTY · ANNOTATED STOPPAGE**. At302s (29:02), the pending label ends. Match events and timeline retain the source penalty. Do not infer its taker/outcome from preparation geometry.
3. Seek372s through406s: expect the annotated kick-off stoppage. At407s it ends. A generic **DEAD BALL · ESTIMATED** means the ball-state estimate exists but a supported restart interval is absent.
4. Open a DFL half with existing dead-ball states: its original restart label and **NOT TAKEN YET** behavior remain. These use DFL tracking/event evidence.
5. During observed/inferred changes, cuts or very large coordinate jumps, the current sample holds until the next timestamp. Nearby same-status replay samples still interpolate. Dead ball does not fly toward the next restart sample. This avoids fabricated display movement; it does not make stored predictions accurate.
6. Review source events in the native table. Existing five free kicks/one penalty remain; absent corners or goal kicks must not appear. Goal-kick display is supported only when the source supplies it.

Compare with the private movies in `cv/out/apply_train7h_20261012_053349/epl_preview/` using the video and clip timestamps. Broadcast video stays outside the web app. Camera calibration and role-assigned shirts remain unverified. Record separate errors for camera/feet, missing detections, team/identity and hidden estimates; a shirt/body click is not a foot-position label. For positional failures, a clip timestamp and visible feet/team/shirt are useful optional evidence.

Automated checks:

```sh
.venv/bin/python -m pytest tests/test_deadball_display.py tests/test_video_modes.py -q
node tests/frontend_match_events.cjs
node tests/frontend_pitch_causality.cjs
```

Backend restart is required to load frame enrichment; frontend changes hot reload. The ongoing training does not need restarting. No match/model re-run is required to add these display labels to stored exports.
