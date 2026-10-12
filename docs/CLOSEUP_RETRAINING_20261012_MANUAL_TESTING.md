# Close-up-aware training review

Training outputs are private experiments; the currently running Arsenal demo retains its existing, unverified coordinates.

1. From `/Users/qnrj/Code/epl-hackathon`, open:

   ```sh
   open cv/out/closeup_retrain_20261012_015904/views/review.html
   open cv/out/closeup_retrain_20261012_015904/views/actual_view_examples.jpg
   ```

   Expect a native table and six actual images. The behind-goal/close-up examples are rejected for the sideline model, not declared useless video. The image at 1227 seconds disagrees with the original camera-category annotation and appears tightly framed. Do not change its test label just to make accuracy perfect.

2. Review [the training report](CLOSEUP_RETRAINING_20261012.md). Compare identical validation frames and sample counts. The close-up classifier score is not a camera position score. Alternate-angle test support is only one sample.

3. The existing app's **Match events** overlay still navigates to the penalty annotation at 29:02. This moment can be a behind-goal/close-up view with inferred global positions. Its event fact and projected geometry are separate evidence. Do not rate those positions correct because a penalty label exists.

4. When the calibration stage finishes, open the experiment's root `review.html`. Expect baseline/new-epoch geometry, supported counts, selection cost and the frozen camera gate. If the gate fails, expect **not deployed**, with the existing production files unchanged.

5. For future close-up annotation, mark a readable shirt/body for identity only; click feet for a ground-contact reference. Only mark a pitch landmark if the ground intersection and its pitch identity are clear. A goalpost/crossbar is above ground and cannot be treated as a pitch-plane landmark.

No immediate new action is required from you. More independent permitted angles would help test generalisation. Do not download restricted footage or buy data for this run.
