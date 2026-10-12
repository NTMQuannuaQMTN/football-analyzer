# Review the seven-hour candidates

Run from the repository root. Session deadline: **20:33:49 Singapore, 12 October 2026**. Keep the machine awake/on power. The supervisor owns a bounded `caffeinate` process. Do not start a second training supervisor.

## Check progress

```sh
tail -n 8 cv/out/apply_train7h_20261012_053349/dfl_training_supervisor.log
tail -n 8 cv/out/apply_train7h_20261012_053349/camera_markings.log
tail -n 8 cv/out/apply_train7h_20261012_053349/application.log
```

`preview_applied.json` or `trained_applied.json` means that application stage completed. `finished.json` inside a training directory means that trainer ended; inspect its complete-epoch/fold counts before calling it complete. `supervisor_finished.json` records exit codes. A traceback indicates a real failure, not successful application.

## DFL ground-truth comparison

When the file exists:

```sh
open cv/out/apply_train7h_20261012_053349/review_preview.html
open cv/out/apply_train7h_20261012_053349/review_trained.html
```

Tables distinguish live and retrospective errors, hidden sample counts and legacy errors. Smaller metre error is better. Compare equal match/half/time/player conditions; old event postprocessing differs. These are simulated-visibility tracking experiments, not DFL broadcast-video evaluation. Counts are correlated player-timestamps, not independent matches.

At `http://localhost:3000`, select **Experimental reconstruction · DFL**. Choose a half and compare tracking ground truth with inferred mode. Examine long hidden intervals and returns to observation; observed coordinates must agree with tracking, while estimates may be wrong. Keep prediction zones hidden or shown using the single upper button.

## EPL qualitative review

When published, open:

`http://localhost:3000/match/cv-trial7h-arsenal-preview/studio?part=clip&t=4`

Replace `preview` with `trained` once that candidate exists. Compare matched timestamps with the private clips:

```sh
open cv/out/apply_train7h_20261012_053349/epl_preview/start_private.mp4
open cv/out/apply_train7h_20261012_053349/epl_preview/penalty_private.mp4
open cv/out/apply_train7h_20261012_053349/epl_preview/goal_private.mp4
```

Use the `epl_trained` directory for later movies. Broadcast video remains private and is not embedded in the web application. Review the camera outline, observed-player feet, player count, ball placement and identity separately. Liverpool #14 is Liverpool, as corrected in your review. Unsupported close-ups/reverse angles should abstain, not inherit a misleading sideline camera. Record timestamp/team/shirt/problem type; a clicked shirt is not a pitch foot-position label.

Other experimental EPL exports have no corresponding local video/full hidden truth. Treat their new inferred positions and transferred uncertainty as unverified. Do not infer accuracy from a plausible picture.

## Optional inputs you can supply

- Permitted DFL/EPL footage plus synchronized all-player tracking, units/orientation and clock alignment.
- Visible-frame review: pitch landmarks, actual feet, team/shirt and a clear distinction between readable and guessed identity.
- Separate penalty/set-piece wide view, close-up and reverse-angle timestamps.

Never label off-pitch substitutes at `(0,0)`, use prior predictions as truth, or approve a model solely because training loss fell.
