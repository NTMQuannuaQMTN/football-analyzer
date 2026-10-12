# Match-view and off-camera baseline reset

11 October 2026. Requested rollback to a known starting point, not an accuracy improvement.

## Restored state

The checkout was already clean at commit `539f23b`, with the original roster renderer and replay clock. Generated files still contained later camera constraints and display-only detection fields. No Git reset or deletion of models was performed.

The Arsenal–Liverpool export now uses `data/cv_run3_yolo11m_backup.json`: YOLO11m detections and the original Metrica-trained `cv/weights/offscreen_mlp.pt` export, 601 one-second frames and 22 roster slots. Positions, visibility flags and ball coordinates match that archive exactly. Saved camera geometry remains a display diagnostic, and verified Theo Walcott goal metadata/date are retained. Archived prediction confidence and radii are absent and displayed as N/A, not fabricated.

For the other 18 saved matches, removed camera-constraint metadata and restored recorded model positions that had been withheld. DFL live/replay views regained 34,028 points; ground-truth frames and observed coordinates were verified unchanged. SkillCorner coordinates were unchanged. This is not a complete rollback of every historical model/data change in those matches.

Current exports were backed up before replacement in `data/reset_backups/20261011_113625/`, with a per-file manifest. The original archive and all model weights remain intact. The backup directory is ignored by Git. The Arsenal metadata records the archive SHA-256 and snapshot location.

The active view has one roster marker layer and original interpolation. No plain-detector overlay, experimental conditional grid decoder, forced camera-edge placement, or 1 Hz rendering throttle is active. Original model estimates can still appear inside calibrated coverage; this reset does not fix calibration, identity, teleportation or hidden-position accuracy.

## Preserved presentation controls

- Unknown score `? - ?`; archived missing confidence/radius N/A.
- Arsenal date visible; original team colours.
- Camera view and one show/hide prediction-zone button above the pitch, zones initially hidden.
- Explanation/what-if opens over the pitch; counterfactual tables remain HTML tables.

Changed `.gitignore`, Studio, MomentPanel, PlayerStateCard and frontend types for these controls; generated data exports restored separately. Reconstruction source code and weights were not changed.

## Executed verification

- Repository status/log, source and archive inspections.
- Python snapshot/restore commands and archive/measurement assertions.
- `.venv/bin/python -m pytest -q`: **20 passed**, 250 existing NumPy deprecation warnings, 43.65 s.
- Frontend `npm run lint` and `npm run build -- --webpack`: passed.
- `git diff --check`: passed.
- `MATCH_FILE=data/cv_2016_08_14_18_00_arsenal_3_4_liverpool_h1_1440_2040.json ./run.sh`: frontend/backend restarted.
- Actual local API comparison: all 601 frames match archived coordinates/visibility, 22 roster slots, no broadcast-observation or camera-constraint fields. Metadata shows original offscreen MLP, date 2016-08-14, Walcott/A14 goal. Studio HTTP 200.

Browser interaction/FPS and real-video position accuracy were **not measured**. No model inference, detector rerun, training or footage download was needed. Counts of restored points measure rollback completeness, not accuracy.

## Manual review and recovery

1. Open the Arsenal studio and hard-refresh with Cmd+Shift+R. Play at 1x. One roster layer should appear, with solid observed and dashed inferred markers, without `?` detection markers.
2. Toggle camera coverage and prediction zones independently. The archive has no uncertainty radii, so this clip may show no zones; missing radii must remain N/A.
3. Open Explain / What if, close it, and seek/replay. Check score/date and the Walcott goal attribution.
4. Compare a few integer-second positions against the private local broadcast. Record errors rather than treating the reset as successful accuracy validation. Future work should establish camera registration and visible identity reliability before changing hidden-player prediction again.
5. To recover the newer generated outputs, copy the `.json` match files from the snapshot directory back to `data/` and restart. Do not copy the manifest as a match. This restores data only; the prior experimental source files are absent from the current checkout.

The reset and API consistency checks passed. Manual visual agreement with the real game remains unverified.
