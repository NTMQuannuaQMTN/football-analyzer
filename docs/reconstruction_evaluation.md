# Hidden-player reconstruction: evaluation (13 October 2026)

All numbers here come from files in `data/eval/recon_benchmark/` that were produced by the commands in §9. Errors are Euclidean distances in metres on a 105 × 68 pitch, at 1 Hz. **Only hidden player-seconds are scored.** Visible players are passed through unchanged and never counted as successful predictions.

## 1. Data and matching quality

The full inventory is in [dataset_inventory.md](dataset_inventory.md).

| Evaluation | Matches | Truth | Visibility | What it measures |
|---|---|---|---|---|
| A. DFL 7-fold | 7 Bundesliga / 2. BL matches, 14 halves | **Direct**: TRACAB in-stadium tracking | **Proxy**: real SkillCorner broadcast footprints moved onto the DFL ball, plus nearest-k levels | Hidden-player error with every position known, identities exact |
| B. DFL degraded | Same | Direct | Proxy, plus GSR-like context: 30 s clips, no possession or ball-status feed, no positional roles | Robustness to broadcast-like input |
| C. SoccerNet-GSR re-entries | 6 games, 114 clips of 30 s (local train + valid splits) | **Direct**, but only at the second a player re-enters the frame | **Real** broadcast framing | Hidden-player error under real broadcast visibility. Inputs are the labelled visible positions (oracle perception and identity). |
| D. SkillCorner re-entries | 20 A-League matches, 40 halves | **Proxy**: SkillCorner's own broadcast detections | **Real** | Same, at full-match scale |

**Not achieved:** no match has both broadcast video and independent full-pitch truth. The whole chain (video → detection → calibration → identity → hidden-player estimate) therefore **cannot be scored against truth on any available match**. Perception was measured separately in earlier sessions (§6).

**Synchronisation:**
- DFL has no video, so nothing needs syncing.
- GSR labels are per frame, so they are synchronised by construction.
- SkillCorner detections and events share the provider frame clock. Event locations were checked against tracking: median 0.4–0.7 m per half once per-event orientation is undone (§7, finding 3).
- No interval was excluded for synchronisation reasons.

## 2. Splits

- **Outer folds:** fold *i* tests on `dfl_matches()[i]` and trains on the other 6. In order:
  - J03WMX Köln–Bayern
  - J03WN1 Bochum–Leverkusen
  - J03WOH
  - J03WOY Düsseldorf–Rostock
  - J03WPY
  - J03WQQ
  - J03WR9
- **Validation:** the time-blocked tail (t ≥ 2200 s, with a 60 s gap) of every half of the 6 training matches. It is used for epoch choice, σ calibration and architecture selection. **The test match is never used for any of these.** The benchmark refuses checkpoints whose metadata does not record the scored match as held out (`benchmark.check_held_out`, tested).
- **Final models** (`foldfinal`) are trained on all 7 matches with the same recipe and have no DFL test set. They are the only models applied to GSR, SkillCorner and the app's broadcast-derived matches, which are all external to DFL.
- **Caveats:**
  - With 7 matches, the cross-validation estimate is the honest one, and there is no single held-out split.
  - Fortuna Düsseldorf appears in 5 matches.
  - The DFL test matches had been examined in earlier sessions, so they are not pristine.

## 3. Methods compared

| Method | Description |
|---|---|
| `last_position` | Freeze at the last sighting |
| `constant_velocity` | Last sighting plus decaying last velocity |
| `formation` | Team-shape estimator: role anchors, block fit, keeper model. Live is forward only; replay fuses forward and backward. |
| `k7r_v3` | **Existing model (incumbent):** v3 MLP correction to the formation prior; 7-fold, refit on all rows of the training matches |
| `r13_mlp` | v3 MLP retrained in the new harness, without refit (control) |
| `r13_mlp_ens3` | 3-seed ensemble of `r13_mlp` |
| `r13_gru` | v3 row features + a GRU over per-second context: own track, prior, ball, visible team centroids and counts, possession, ball status, cut. Live window t−19…t; replay t±12 (bidirectional). |
| **`r13_mlp_aug_ens3`** | **Selected.** `r13_mlp_ens3` trained with **context-degradation augmentation**: the training data also contains independent 30 s clips with no possession, no ball status and no positional roles or anchors. |

`+speed` means the production post-processing is applied: the GK zone clamp, plus a speed limit of 8.5 m/s (live: forward only; replay: both directions). The tables below use `+speed` for the learned models, matching what the app shows.

## 4. Results

### 4.1 DFL 7-fold, real-footprint mask (A)

The `sc_camera` mask uses 2 mask instances per half. There are 1,316,306 hidden player-seconds per method, which is about 75 % of on-pitch player-seconds. Source: `report_final.json`.

| Method | Mean | Median | RMSE | P90 | ≤ 2 m | ≤ 5 m | ≤ 10 m | Match-mean (sd) | 80 % radius coverage | Mean radius |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Last position | 16.93 | 13.27 | 22.19 | 37.00 | 9.1 % | 22.4 % | 40.1 % | 16.89 (1.13) | n/a | n/a |
| Constant velocity | 16.55 | 12.80 | 22.00 | 36.85 | 11.9 % | 24.8 % | 41.9 % | 16.50 (1.22) | n/a | n/a |
| Formation, live | 14.28 | 10.73 | 18.72 | 31.59 | 8.1 % | 24.7 % | 47.4 % | 14.24 (1.27) | 40.7 % | 9.4 |
| Formation, replay | 10.54 | 7.64 | 14.11 | 23.32 | 11.6 % | 34.2 % | 60.6 % | 10.50 (1.29) | 36.2 % | 6.1 |
| k7r_v3, live (incumbent) | 8.23 | 6.21 | 11.02 | 17.62 | 16.2 % | 41.6 % | 70.2 % | 8.21 (0.51) | 81.0 % | 12.1 |
| k7r_v3, replay (incumbent) | 6.74 | 4.78 | 9.35 | 14.96 | 22.5 % | 51.7 % | 78.1 % | 6.72 (0.61) | 83.2 % | 10.5 |
| r13_mlp, live | 8.30 | 6.27 | 11.10 | 17.76 | 16.0 % | 41.3 % | 69.8 % | 8.28 (0.48) | 80.0 % | 12.0 |
| r13_mlp, replay | 6.90 | 4.93 | 9.52 | 15.21 | 21.4 % | 50.5 % | 77.3 % | 6.88 (0.66) | 81.9 % | 10.4 |
| r13_mlp_ens3, live | 8.10 | 6.12 | 10.87 | 17.37 | 16.8 % | 42.2 % | 70.7 % | 8.09 (0.47) | 80.3 % | 11.7 |
| r13_mlp_ens3, replay | 6.72 | 4.78 | 9.32 | 14.90 | 22.5 % | 51.7 % | 78.1 % | 6.70 (0.63) | 81.0 % | 10.0 |
| r13_gru, live | **7.72** | 5.76 | 10.46 | 16.49 | 17.5 % | 44.3 % | 73.3 % | 7.71 (0.38) | 79.8 % | 11.1 |
| r13_gru, replay | **6.37** | 4.51 | 8.89 | 13.95 | 23.2 % | 54.0 % | 80.6 % | 6.35 (0.54) | 81.7 % | 9.6 |
| **r13_mlp_aug_ens3, live** | 8.10 | 6.11 | 10.87 | 17.40 | 16.8 % | 42.3 % | 70.7 % | 8.09 (0.47) | 80.2 % | 11.7 |
| **r13_mlp_aug_ens3, replay** | 6.66 | 4.72 | 9.26 | 14.79 | 23.2 % | 52.2 % | 78.4 % | 6.64 (0.63) | 80.3 % | 9.8 |

Prediction coverage is 100 % of hidden player-seconds for the formation and learned methods, and 99.1 % for last position and constant velocity, which cannot predict a player before his first sighting.

**Per match**, live / replay mean, `+speed`:

| Match | Formation | k7r_v3 (incumbent) | Selected (aug_ens3) | GRU |
|---|---|---|---|---|
| J03WMX | 13.50 / 10.05 | 7.96 / 6.62 | 7.87 / 6.48 | 7.51 / 6.16 |
| J03WN1 | 12.91 / 9.02 | 7.57 / 5.86 | 7.43 / 5.77 | 7.19 / 5.69 |
| J03WOH | 16.35 / 12.03 | 9.23 / 7.78 | 9.02 / 7.74 | 8.43 / 7.30 |
| J03WOY | 14.43 / 9.98 | 8.07 / 6.53 | 7.98 / 6.45 | 7.63 / 6.16 |
| J03WPY | 13.13 / 9.74 | 8.08 / 6.41 | 8.05 / 6.34 | 7.73 / 6.07 |
| J03WQQ | 13.52 / 9.83 | 7.94 / 6.39 | 7.81 / 6.31 | 7.43 / 6.06 |
| J03WR9 | 15.88 / 12.86 | 8.63 / 7.44 | 8.44 / 7.40 | 8.02 / 7.03 |


- **Selected vs incumbent:** better on **7 / 7** matches in both modes. Margins are 0.03–0.21 m live and 0.04–0.14 m replay.
- **GRU vs incumbent:** better on 7 / 7, by 0.35–0.80 m live and 0.18–0.48 m replay.

**Fewer visible players** (simulated nearest-k masks):

| Method | k = 14 | k = 10 | k = 6 |
|---|---|---|---|
| Formation, live / replay | 7.86 / 6.57 | 9.20 / 7.57 | 11.99 / 10.05 |
| Incumbent, live / replay | 4.90 / 3.92 | 5.84 / 4.71 | 7.32 / 6.12 |
| Selected, live / replay | 4.80 / 3.87 | 5.75 / 4.67 | 7.22 / 6.06 |
| GRU, live / replay | 4.82 / 3.90 | 5.69 / 4.65 | 7.01 / 5.92 |

### 4.2 Accuracy by visibility, horizon, role and phase

DFL, `sc_camera`, all 7 test matches. Columns 3–8 are mean error in metres for formation, incumbent and selected, first live then replay. The last column is the selected model's 80 % radius coverage (live / replay).

| Group | n | Formation (L) | Incumbent (L) | Selected (L) | Formation (R) | Incumbent (R) | Selected (R) | Selected coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0–2 s since last seen | 138,696 | 2.93 | 1.78 | 1.69 | 2.88 | 1.55 | 1.51 | 81 % / 80 % |
| 3–5 s | 150,235 | 6.46 | 4.02 | 3.94 | 5.35 | 3.32 | 3.26 | 81 % / 80 % |
| 6–10 s | 182,732 | 10.51 | 6.45 | 6.36 | 8.23 | 5.33 | 5.25 | 82 % / 81 % |
| 11–30 s | 409,715 | 16.25 | 9.52 | 9.39 | 11.73 | 7.66 | 7.57 | 81 % / 81 % |
| 31–60 s | 234,402 | 19.76 | 11.34 | 11.15 | 13.85 | 9.01 | 9.00 | 78 % / 79 % |
| 61–120 s | 133,191 | 21.22 | 11.42 | 11.25 | 15.84 | 9.85 | 9.72 | 77 % / 78 % |
| > 120 s | 55,544 | 20.87 | 10.37 | 10.37 | 15.03 | 9.17 | 8.90 | 78 % / 81 % |
| Never seen yet | 11,791 | 19.20 | 12.32 | 11.60 | 14.39 | 9.92 | 9.43 | 83 % / 84 % |
| 0 visible team-mates | 779,673 | 17.13 | 9.53 | 9.39 | 12.61 | 8.07 | 7.97 | 80 % / 81 % |
| 1–2 visible team-mates | 205,221 | 12.82 | 7.44 | 7.32 | 8.77 | 5.64 | 5.57 | 81 % / 81 % |
| 3–5 visible team-mates | 219,352 | 8.87 | 5.90 | 5.80 | 6.88 | 4.51 | 4.46 | 80 % / 79 % |
| 6+ visible team-mates | 112,060 | 7.75 | 5.19 | 5.08 | 6.60 | 3.88 | 3.85 | 77 % / 77 % |
| **Outside camera footprint** | 618,069 | 11.55 | 6.96 | 6.84 | 8.27 | 5.32 | 5.25 | 80 % / 79 % |
| **In view, detection missed** (occlusion proxy) | 10,814 | 4.52 | 3.36 | 3.28 | 2.35 | 1.57 | 1.55 | 78 % / 79 % |
| **Camera cut** (replay / close-up, nobody visible) | 687,423 | 16.89 | 9.44 | 9.31 | 12.71 | 8.10 | 8.01 | 81 % / 81 % |
| **Long unseen (> 30 s)** | 423,137 | 20.37 | 11.24 | 11.08 | 14.63 | 9.29 | 9.21 | 78 % / 79 % |
| In possession | 352,344 | 13.39 | 8.07 | 7.99 | 10.39 | 6.73 | 6.68 | 82 % / 81 % |
| Out of possession | 358,848 | 12.78 | 7.01 | 6.97 | 9.58 | 5.98 | 5.92 | 82 % / 81 % |
| Dead ball | 605,114 | 15.70 | 9.04 | 8.84 | 11.20 | 7.20 | 7.09 | 78 % / 79 % |
| Within 5 s of a possession change | 525,454 | 13.22 | 7.64 | 7.54 | 10.06 | 6.45 | 6.36 | 81 % / 80 % |
| Within 5 s of a restart | 247,630 | 15.57 | 8.38 | 8.25 | 10.49 | 6.55 | 6.47 | 80 % / 78 % |
| GK | 138,619 | 9.10 | 5.33 | 5.24 | 8.92 | 5.00 | 4.90 | 76 % / 77 % |
| CB | 271,610 | 13.89 | 7.55 | 7.38 | 10.09 | 6.11 | 6.00 | 81 % / 82 % |
| FB | 192,284 | 15.45 | 8.84 | 8.78 | 11.29 | 7.24 | 7.21 | 81 % / 81 % |
| Wide midfielder | 187,585 | 16.18 | 9.66 | 9.55 | 11.73 | 7.80 | 7.78 | 80 % / 80 % |
| ST | 166,539 | 14.53 | 8.81 | 8.61 | 10.62 | 7.15 | 7.03 | 80 % / 80 % |

Roles are the DFL playing positions mapped to coarse roles, which makes them reliable. Directly observed players are not scored; they are exact tracking here, not detections. Category 5 of the brief, low-confidence or ambiguous identity, **does not exist on DFL**, where identity is exact. Its effect is the identity gap in §6.

### 4.3 Uncertainty calibration

Reliability of P(error ≤ 5 m) for the selected model on DFL, `sc_camera`. Each cell is predicted → observed.

| Predicted bin | Live | Replay |
|---|---|---|
| 0.0–0.2 | 0.13 → 0.13 | 0.15 → 0.16 |
| 0.2–0.4 | 0.28 → 0.26 | 0.31 → 0.31 |
| 0.4–0.6 | 0.49 → 0.47 | 0.48 → 0.48 |
| 0.6–0.8 | 0.66 → 0.64 | 0.68 → 0.68 |
| 0.8–1.0 | 0.92 → 0.91 | 0.93 → 0.93 |

The 80 % radius covers 80.2 % (live) and 80.3 % (replay) overall, falling to 77–79 % after more than 30 s unseen. **Off DFL, the radius is too narrow:** see §4.5.

### 4.4 Broadcast-context robustness (B)

DFL scored on independent 30 s clips with no possession, no ball status and no roles or anchors. Only player-seconds after a sighting in the clip are scored. Values are the mean over folds.

| Method | Validation tail, live | Validation tail, replay | Held-out test, live | Held-out test, replay |
|---|---:|---:|---:|---:|
| Formation | n/a | n/a | 15.33 | 13.95 |
| Incumbent k7r_v3 | 7.76 (in-sample) | 6.70 (in-sample) | 7.77 | 6.98 |
| r13_mlp | 7.99 | 7.80 | — | — |
| r13_mlp_ens3 | 7.78 | 7.66 | 7.81 | 8.02 |
| r13_gru | 8.19 | 8.02 | 8.14 | 8.28 |
| **r13_mlp_aug_ens3** | **6.11** | **5.20** | **6.18** | **5.43** |

"In-sample" because the incumbent was refit on rows that include the validation tail, so its validation numbers are optimistic. That is why it is excluded from selection (§5).

### 4.5 Real broadcast re-entries (C: GSR, direct truth; D: SkillCorner, proxy truth)

These use the final models trained on all 7 DFL matches. GSR and SkillCorner were never used in training.

**C. SoccerNet-GSR**, 6 games:
- Live: 1,344 re-entries (114 clips). By gap: 353 at 1–2 s, 473 at 3–5 s, 335 at 6–10 s, 183 at 11–30 s.
- Replay: 3,000 points, from the first 3 s after re-entry, additionally hidden.

| Method | Live mean | Median | RMSE | ≤ 5 m | ≤ 10 m | 80 % radius coverage | Replay mean | Replay coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Last position | 12.55 | 9.93 | 16.25 | 23 % | 51 % | n/a | n/a | n/a |
| Constant velocity | 11.19 | 8.33 | 14.94 | 30 % | 58 % | n/a | n/a | n/a |
| Formation | 7.69 | 6.49 | 9.38 | 36 % | 74 % | 41 % | 4.69 | 41 % |
| Incumbent final_v3 | 6.99 | 5.65 | 8.77 | 45 % | 78 % | 70 % | 3.20 | 76 % |
| r13_mlp_ens3 | 6.95 | 5.54 | 8.76 | 45 % | 77 % | 72 % | 3.22 | 75 % |
| r13_gru | 7.46 | 5.90 | 9.44 | 42 % | 75 % | 65 % | 3.40 | 72 % |
| **Selected** | **6.74** | **5.37** | **8.42** | **47 %** | **80 %** | 70 % | **3.06** | 71 % |

The selected model beats the incumbent in 5 of 6 games live (it loses game 6: 7.24 vs 6.83 m) and 6 of 6 replay.

**D. SkillCorner A-League**, 20 matches:
- Live: 94,918 re-entries.
- Replay: 226,050 points.

| Method | Live mean | Median | ≤ 5 m | ≤ 10 m | 80 % radius coverage | Replay mean | Replay coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Last position | 14.80 | 10.82 | 24 % | 47 % | n/a | n/a | n/a |
| Constant velocity | 13.68 | 9.33 | 30 % | 52 % | n/a | n/a | n/a |
| Formation | 10.62 | 7.31 | 34 % | 64 % | 47 % | 3.40 | 63 % |
| Incumbent final_v3 | 7.88 | 5.68 | 45 % | 74 % | 78.5 % | 2.91 | 85 % |
| r13_mlp_ens3 | 7.76 | 5.62 | 45 % | 74 % | 77 % | 2.93 | 81 % |
| r13_gru | 8.85 | 6.08 | 42 % | 70 % | 70 % | 3.20 | 77 % |
| **Selected** | **7.70** | **5.55** | **45 %** | **74 %** | 78 % | **2.81** | 81 % |

The selected model beats the incumbent on 20 of 20 matches in both modes. By real gap length (live, selected, SkillCorner):

| Gap | Mean error | Re-entries |
|---|---:|---:|
| 1–2 s | 3.24 m | 16,119 |
| 3–5 s | 5.05 m | 21,809 |
| 6–10 s | 7.19 m | 18,510 |
| 11–30 s | 10.48 m | 25,840 |

**Broadcast radius scale (E6),** fitted on one group of GSR games and tested on the other, then swapped (`gsr_radius_scale.json`):
- Live: the selected model's coverage goes from 73 % / 67 % to **84 % / 76 %** with scales 1.27 / 1.17.
- Replay: from 70 % / 71 % to **79 % / 81 %** with scales 1.18 / 1.20.

So for GSR-like inputs (no possession or roles), the DFL-calibrated radius should be widened by about 1.2×. SkillCorner inputs, which include roles and possession, reach 78 % / 81 % without scaling, so the app applies no scale to them. The Arsenal video export keeps the earlier ×2.67 video scale, which also absorbs identity errors.

## 5. Experiments: what was tried and kept

Selection rule, declared before the augmented results existed: **minimise the mean of full-context and degraded-context validation-tail error**, per mode, among models that never trained on the tail. Promotion then requires being no worse than the incumbent on the DFL test, the degraded test, GSR and SkillCorner. The promotion gate looks at test data; this is disclosed, and it is a binary non-inferiority check, not a search.

| # | Hypothesis | Change | Validation (full / degraded, live) | Held-out DFL live / replay | External (GSR live, SkillCorner live) | Cost | Decision |
|---|---|---|---|---|---|---|---|
| E1 | The harness reproduces the incumbent | Retrain v3 MLP, no refit | 8.42 / 7.99 | 8.30 / 6.90 | n/a | ~2 min | Control; reproduces within 0.07 / 0.16 m (incumbent has refit) |
| E2 | Seed ensembling reduces variance | 3 seeds, σ combined, recalibrated | 8.24 / 7.78 | 8.10 / 6.72 | 6.95, 7.76 | 3× training | Superseded by E7 |
| E3 | Temporal context helps | GRU over 20 s (live) / ±12 s (replay) | **7.85** / 8.19 | **7.72 / 6.37** (7/7 wins) | **7.46 (worse), 8.85 (worse)** | ~45 min per fold on 3 threads | **Rejected for deployment:** best on DFL, but worse on both real-broadcast tests and under degraded context. It overfits to clean, uncorrelated DFL observations and context. |
| E4 | Calibration drifts with horizon | Inspection only | Coverage 77–79 % beyond 30 s | n/a | n/a | n/a | Not implemented; a small effect |
| E5 | Refit after epoch selection | Not run for new models | n/a | n/a | n/a | n/a | Skipped: it makes the validation tail in-sample (see the incumbent) |
| E6 | The DFL radius is too narrow on broadcast | GSR game-split scale | n/a | n/a | Coverage ≈ 70 % → 76–84 % | none | **Kept as a documented setting** for GSR-like inputs |
| E7 | Broadcast inputs lack context the model relies on | Add degraded 30 s clips to training and validation | 8.25 / **6.11** | 8.10 / 6.66 | **6.74, 7.70** | +10 % data | **Selected and promoted** |

The best-known checkpoint was never overwritten. The incumbent `cv/weights/hidden_*_final_v3.pt` and every fold checkpoint are kept. The promoted model lives in `cv/weights/recon/r13_mlp_aug_ens3/`.

## 6. Perception errors

These were measured in earlier sessions on GSR held-out game 2 and are not re-run here (sources: `docs/CALIBRATION_IDENTITY_8H_AUDIT.md`, `cv/out/audit8_20261011`).

| Stage | Result |
|---|---|
| Visible-person detection (selected tracker) | Precision 89.5 %, recall 79.9 % (39,437 labelled people) |
| Anonymous identity F1 | 51.2 %, with 1,820 switches in 31,084 transitions |
| Visible feet → pitch (camera + tracker, Kalman) | 57.99 % of labelled visible observations within 5 m; conditional mean 1.95 m on supported detections |
| Lineup identity on the Arsenal clip (role-based) | ≈ 35 % exact in an earlier evaluation |
| Hidden error with oracle vs CV identities (Arsenal reappearances, earlier session) | ≈ 6.8 m vs ≈ 12.9 m |

**These perception errors add to the hidden-player errors above, and they currently dominate on our own video pipeline.** The GSR and SkillCorner rows in §4.5 isolate the model by using labelled or provider observations.

## 7. Ranked error sources and failure cases

1. **Identity on our own video** (§6): role-based identity is ≈ 35 % correct. With oracle identity the hidden error roughly halves. This is the biggest gap between what DFL measures and what our own CV delivers.
2. **Long camera cuts and long-unseen players.** 52 % of hidden player-seconds fall inside camera cuts (replays and close-ups in the real SkillCorner footprints), with a mean of 9.3 m live. Players unseen for more than 30 s average 11.1 m live / 9.2 m replay.
3. **Domain shift in context and observation noise.** The GRU result shows that models can silently overfit to DFL's exact positions, possession and ball-status feed. Two other domain issues found this session:
   - provider events are normalised per event to the attacking direction (fixed and tested);
   - the DFL radius under-covers on GSR.
4. **Dead balls and wide roles:** dead ball 8.8 m live; wide midfielders 9.6 m and full-backs 8.8 m, the worst roles.
5. **Live re-entry jumps:** live mode cannot use the next sighting, so estimates jump when a player reappears. Replay removes the jumps with the both-direction speed limit, which leaves 0 estimated steps above 10 m/s in the exported files (`export_checks`).
6. **Calibration** is slightly overconfident after 30 s or more unseen (77–79 % coverage), and too narrow on GSR-like inputs (§4.5).

## 8. Runtime and resources

- Hardware: Apple M5, 10 cores, 24 GB RAM; all CPU. MPS was slower than CPU for the GRU.
- Feature cache build: ≈ 90 s for 7 matches, plus ≈ 70 s for degraded clips. Stored in `data/eval/real/recon_cache` (≈ 1 GB).
- Training times:
  - one MLP fold: ≈ 1–2 min unloaded;
  - 3-seed augmented ensemble, 7 folds + final: ≈ 60 min on 3 threads while sharing the CPU;
  - one GRU fold: ≈ 15 min unloaded, ≈ 45 min on 3 threads with 4 jobs sharing.
- Benchmark: ≈ 60–180 s per fold per method set. GSR ≈ 1 min; SkillCorner ≈ 10 min.
- Inference: a full half (≈ 2,900 s × 22 players), live + replay, takes ≈ 5–10 s on CPU.

## 9. Reproduce

```bash
.venv/bin/python -m backend.reconstruction.train_hidden --arch mlp --fold all --tag r13_mlp_aug_ens3 --seeds 0,1,2 --epochs 12 --augment
.venv/bin/python -m backend.reconstruction.train_hidden --arch mlp --fold final --tag r13_mlp_aug_ens3 --seeds 0,1,2 --epochs 12 --augment
.venv/bin/python -m backend.reconstruction.benchmark run --run r13_aug --methods r13_mlp_aug_ens3
.venv/bin/python -m backend.reconstruction.benchmark report --runs r13_base,r13_mlp,r13_ens,r13_gru,r13_aug --name final
.venv/bin/python -c "from backend.reconstruction.benchmark import degraded_run as d; d(['r13_mlp_aug_ens3'],'validation'); d(['k7r_v3','r13_mlp_ens3','r13_mlp_aug_ens3','r13_gru'],'test',name='test_all')"
.venv/bin/python -c "from backend.reconstruction.benchmark import full_validation_run as f; f(['r13_mlp_aug_ens3','r13_gru'])"
.venv/bin/python -m backend.reconstruction.gsr_benchmark --methods base,final_v3,r13_mlp_ens3_final,r13_mlp_aug_ens3_final,r13_gru_final --name gsr_all
.venv/bin/python -m backend.reconstruction.skillcorner_benchmark --methods base,final_v3,r13_mlp_ens3_final,r13_mlp_aug_ens3_final,r13_gru_final --name sc_all
```

The other runs (`r13_base`, `r13_mlp`, `r13_ens`, `r13_gru`) follow the same pattern; see [reconstruction_test_guide.md](reconstruction_test_guide.md).

## 10. Remaining limitations and next steps

- **No end-to-end video truth.** The most valuable next data is a licensed broadcast of a match with full tracking (for example WorldPose under its FIFA agreement, or a TRACAB match with its feed).
- **Identity on video** is the largest error source. Next technical step: jersey-number recognition on GSR with track voting, then measure end-to-end with GSR identities instead of oracle ones.
- **The GRU's DFL gain (−0.4 to −0.5 m) does not transfer.** A GRU trained with degraded-context augmentation *and* observation noise or correlated-miss masks is the natural follow-up. It was not run, for lack of compute time.
- **Live reappearance jumps** remain by design.
