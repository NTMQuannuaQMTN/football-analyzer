# Reconstruction training plan (session of 13 October 2026)

This plan was written after the Phase 0 audit and the baseline run. The results it led to are in [reconstruction_evaluation.md](reconstruction_evaluation.md).

## 1. Existing capabilities (audited)

| Area | Status at session start | Where |
|---|---|---|
| Data contract | One JSON contract `{meta, players, events, frames}` (1 Hz, metres, 105 × 68, home attacks +x) plus optional `alt_frames` (live / replay) and `reveals` | `backend/models/match.py`, `backend/pipeline.py` |
| DFL ground truth | 7 matches parsed to 1 Hz. Home team oriented by the first-frame goalkeeper. | `backend/reconstruction/datasets.py` |
| Camera visibility | Real SkillCorner footprints moved onto the DFL ball (`sc_camera`), nearest-k, random | `real_benchmark.py` |
| Hidden-player model | Team-shape prior (`estimator.py`, live and replay) + v3 MLP correction + σ decile calibration (`ml.py`). Production: `final_v3` (all 7 matches). Held-out: `k7r_{i}_v3`. | `cv/weights/hidden_*` |
| Post-processing | GK zone clamp; speed limit (live forward only, replay both directions). Soft event pins in the legacy DFL export. | `postprocess.py`, `dfl_match.py` |
| Video → pitch | YOLO11 (fine-tuned on GSR) + BoT-SORT, team colours, SoccerNet / learned cameras, role-based identity | `cv/` |
| Analytics | Tactical timeline, press / counter / shift detectors, moments with evidence chains, non-causal story graph, narratives with a number guard | `backend/analytics`, `tactics`, `moments`, `narratives` |
| Tests | **149 passed** (48 s) at session start (`.venv/bin/python -m pytest -q tests -p no:cacheprovider`) | `tests/` |

Problems found in the audit:

1. **Inconsistent protocols.** Hidden-player error was reported in at least three incompatible ways:
   - `ml.py evaluate_test`: 8.08 / 6.74 m;
   - the app export path, with event pins and different post-processing: 9.11 / 7.61 m;
   - reveal-only metrics.
   
   No single harness scored every method on the same masks and rows.
2. **No real-broadcast hidden-player test.** DFL masks are transferred from A-League cameras. The Arsenal clip has no truth, and its proxy (role-based identity) is dominated by identity errors.
3. **Repeated use of the test matches.** The 7 DFL test matches have been looked at in earlier sessions. They are not a pristine benchmark, and the docs must say so.
4. **Calibration transfer.** The 80 % radius is calibrated on DFL truth. Its coverage on broadcast data had only been measured on the Arsenal clip, using role-based identities.
5. **Event pins in the legacy DFL export.** They use an event-clock lag fitted over the whole half, so the legacy *live* view uses some future information. Pins were disabled in the seven-hour session and stay disabled here.

## 2. Data already available

Summary only; the full list is in [dataset_inventory.md](dataset_inventory.md).

- **Direct positions, simulated visibility:** DFL, 7 matches / 14 halves.
- **Direct truth for visible players, real broadcast visibility:** SoccerNet-GSR, 115 clips / 6 games. Used for the new real-broadcast re-entry test.
- **Proxy truth (provider CV detections), real broadcast visibility:** SkillCorner A-League, 20 matches / 40 halves. Used for the new full-match re-entry test, and newly reconstructed for the app.
- **No truth:** the Arsenal–Liverpool video clip and 2 legacy SkillCorner EPL matches.

## 3. Data still needed

- Broadcast video with independent full-pitch tracking **for the same match**. It does not exist locally, and no accessible public source was found.
- The GSR test split (needs the NDA password; not attempted).
- Jersey and identity labels for broadcast reconstruction of named players.

## 4. Known limitations going in

- Only 7 DFL matches, and Düsseldorf plays in 5 of them, so test teams are partly seen in other folds.
- DFL visibility is simulated: real camera footprints but a transferred position, with no occlusion model beyond the measured SkillCorner detection rate.
- The model is trained on exact visible positions and true identities. Real video adds detection, calibration (≈ 2 m conditional mean on GSR visible feet) and identity errors (anonymous identity F1 ≈ 51 % on held-out GSR game 2).
- The model works at 1 Hz.

## 5. Baseline (Phase 0), recorded

Command:

```bash
.venv/bin/python -m backend.reconstruction.benchmark run --run r13_base --methods base,k7r_v3
.venv/bin/python -m backend.reconstruction.benchmark report --runs r13_base --name baseline
```

Setup:
- 7 outer folds (test = one DFL match);
- mask seeds `EVAL_SEED=777000 + 1000·fold + 100·half + 10·instance`, disjoint from the training seeds;
- `sc_camera` with 2 instances per half;
- existing checkpoints `cv/weights/hidden_{live,replay}_k7r_{i}_v3.pt`.

Output: `data/eval/recon_benchmark/report_baseline.json`.

Hidden players under the real-footprint mask (`sc_camera`), all 7 test matches, 1,316,306 hidden player-seconds per method:

| Method | Mean | Median | RMSE | ≤ 5 m | Match-mean (sd) | 80 % radius coverage |
|---|---:|---:|---:|---:|---:|---:|
| Last position | 16.93 | 13.27 | 22.19 | 22.4 % | 16.89 (1.13) | n/a |
| Constant velocity | 16.55 | 12.80 | 22.00 | 24.8 % | 16.50 (1.22) | n/a |
| Formation, live | 14.28 | 10.73 | 18.72 | 24.7 % | 14.24 (1.27) | 40.7 % |
| Formation, replay | 10.54 | 7.64 | 14.11 | 34.2 % | 10.50 (1.29) | 36.2 % |
| **Existing k7r_v3, live** | **8.29** | 6.23 | 11.12 | 41.5 % | 8.27 (0.50) | 80.7 % |
| Existing k7r_v3, live + speed limit | 8.23 | 6.21 | 11.02 | 41.6 % | 8.21 (0.51) | 81.0 % |
| **Existing k7r_v3, replay** | **6.96** | 4.96 | 9.62 | 50.3 % | 6.94 (0.61) | 81.8 % |
| Existing k7r_v3, replay + speed limit | 6.74 | 4.78 | 9.35 | 51.7 % | 6.72 (0.61) | 83.2 % |

Real-broadcast re-entry baselines, run on the same day:
- **GSR** (direct truth, 6 games): live formation 7.69 m vs final_v3 6.99 m.
- **SkillCorner** (proxy truth, 20 matches): live formation 10.62 m vs final_v3 7.88 m.

## 6. Execution plan

1. **Harness.** Build `benchmark.py` (DFL 7-fold, every method on identical rows), `gsr_benchmark.py` (GSR re-entries) and `skillcorner_benchmark.py` (A-League re-entries). ✅
2. **Shared model module.** `hidden_models.py`: one preprocessing path for training, benchmark and app; MLP / GRU / ensemble checkpoints with an `arch` key. ✅
3. **Training script.** `train_hidden.py`: 7-fold by match. Epoch selection and calibration use only the time-blocked tail (t ≥ 2200 s, 60 s gap) of the training halves. Architectures are chosen by the same validation tail, **never by test error**.
4. **Experiments,** one change at a time:
   - E1: v3 MLP retrained in the new harness (control, no refit).
   - E2: 3-seed MLP ensemble (variance reduction).
   - E3: GRU temporal context: the player's own track, the prior, ball, team centroids, possession, ball status and cuts over the last 20 s (live) or ±12 s (replay).
   - E4: calibration per time-hidden bucket, if coverage drifts with horizon.
   - E5: re-fit the winner on all rows of the 6 training matches (the earlier recipe gave about −0.1 m).
5. **Select** on validation, then report the test results of every candidate, including the losers.
6. **Final model.** Train on all 7 matches; it is used for GSR, SkillCorner and the app. Its DFL accuracy is the 7-fold estimate.
7. **Reconstruct.**
   - 20 A-League matches (new, from real broadcast detections);
   - 2 legacy SkillCorner EPL matches;
   - Arsenal clip (re-inference on its stored CV observations);
   - 7 DFL matches (held-out fold models, simulated camera).
8. **Analyse** every processed match with the existing engine. Write a grounded per-match report and a summary with basis tags (ground truth / observed / model-inferred / heuristic).
9. **Tests and documentation.** Add tests for live causality, held-out checks, SkillCorner orientation, the exporter contract and benchmark rows. Write the model card, evaluation, analysis report and test guide.

## 7. Outcome

Recorded after execution; the details are in [reconstruction_evaluation.md](reconstruction_evaluation.md).

**Done:** steps 1–9.

**Experiments:**
- E1, E2, E3 and E7 were run.
- E4 was inspected only.
- E5 was skipped, because refit makes the validation tail in-sample.
- E6 was kept as a documented setting.
- **E7 (context-degradation augmentation) was not in the original plan.** It was added after E3's GRU failed on the real-broadcast tests and the failure was reproduced on DFL validation data.

**Selected and promoted:** `r13_mlp_aug_ens3`.

**Not done:**
- An end-to-end video-truth evaluation (no such data).
- A GRU with augmentation and observation noise (compute time).
- Jersey-based identity.
