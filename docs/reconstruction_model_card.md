# Model card: hidden-player reconstruction `r16_blend` (current) / `r13_mlp_aug_ens3`

> **16 October 2026:** production is now `r16_blend`, an equal blend of `r13_mlp_aug_ens3` (described below) and `r16_top3`. `r16_top3` is an MLP, a mixture density network and a zone classifier, the three best of 11 algorithms screened. The blend is better on every gate: DFL held-out live 8.04 m (was 8.10), replay 6.61 m (was 6.66); GSR 6.66 / 3.00 m; SkillCorner 7.61 / 2.67 m. Calibration is unchanged. Checkpoints are in `cv/weights/recon/r16_blend/`; each refers to its two member files by relative path. Evidence: `docs/reconstruction_evaluation.md` §12. The rest of this card describes the `r13_mlp_aug_ens3` member, which is unchanged.

| | |
|---|---|
| Version | `r13_mlp_aug_ens3` (promoted 13 October 2026). It replaces `final_v3`, which is kept and can be selected with `RECON_PRODUCTION=:final_v3`. |
| Checkpoints | `cv/weights/recon/r13_mlp_aug_ens3/hidden_{live,replay}_foldfinal.pt`: an ensemble manifest plus 3 member files `*_s{0,1,2}.pt`, trained on all 7 DFL matches. Held-out fold models: `hidden_{mode}_fold{0..6}.pt`. Every checkpoint records its `train` and `test` matches, seed, epochs, validation history, feature version and the source digest. |
| Code | `backend/reconstruction/hidden_models.py` (loading, preprocessing, inference) and `train_hidden.py` (training). The app path is `inference.reconstruct_ml` / `dfl_match.estimate`, which use the same `hidden_models.predict_segment` as the benchmark. |

## What it does

For every on-pitch player who is **not observed** at a given second, the model corrects the team-shape estimator's guess of where that player is. It outputs:
- a position;
- a calibrated 80 % radius;
- P(error ≤ 5 m).

Observed players are never changed.

There are two modes:
- **Live:** uses only data up to the current second. Tested: changing the future does not change live output.
- **Replay:** also uses the next sighting. It is retrospective and is labelled that way in the app.

## Architecture and inputs

- **Prior:** the team-shape estimator (`estimator.py`).
- **Model:** 3 MLPs (256-256-128), each predicting a 2-D correction and log σ, in the player's own-team frame. The ensemble averages the means and combines σ² with the spread of the members, then recalibrates on validation.
- **Features:** feature version 3, 66 for live and 75 for replay.
  - The prior and the player's last sighting, with time since, velocity and offset from his team at that sighting.
  - The ball now and at the last sighting.
  - Visible team-mate and opponent counts, centroids and the 3 nearest of each.
  - Team extent, the opponents' offside line, and ball velocity.
  - Possession, ball in or out of play, camera cut, dead-ball / set-piece context, and coarse role.
  - Replay only: the next sighting and its slot.
- **Uncertainty:** the σ decile maps to an empirical 80 % radius and P(err ≤ 5 m), fitted on the validation tail.
- **Post-processing:** an unseen goalkeeper is kept within 25 m of his goal line. The speed limit is 8.5 m/s: forward only in live, both directions in replay.

## Training data and supervision

- **Positions: direct supervision.** DFL / IDSSE open data: 7 German matches, TRACAB in-stadium tracking, CC BY 4.0.
- **Visibility: proxy.** Seven mask instances per half:
  - 2 × real SkillCorner broadcast camera footprints with real detection rates, moved onto the DFL ball;
  - nearest-k with 16 / 12 / 8 / 6 visible;
  - 10 visible at random.
- **Augmentation:** independent 30 s clips without possession, ball status or positional roles or anchors. These imitate broadcast inputs where that context is missing.
- **Protocol:** 7-fold by match. Epochs, calibration and the choice of architecture come from the time-blocked tail (t ≥ 2200 s) of the training halves, never from a test match. Selection rule: the mean of full-context and degraded-context validation error.

No broadcast video, no synthetic data and no SkillCorner or GSR positions were used for training. SkillCorner camera footprints shape only the training masks.

## Measured performance

Full tables are in [reconstruction_evaluation.md](reconstruction_evaluation.md).

| Setting | Live | Replay | 80 % radius coverage |
|---|---|---|---|
| DFL held-out, real camera footprint (7-fold; 1.32 M hidden player-seconds) | 8.10 m (incumbent 8.23; formation 14.28; last position 16.93) | 6.66 m (incumbent 6.74; formation 10.54) | 80 % / 80 % |
| DFL held-out, broadcast-like context | 6.18 m (incumbent 7.77) | 5.43 m (incumbent 6.98) | not measured |
| **SoccerNet-GSR real-broadcast re-entries** (6 games, direct truth, oracle perception) | 6.74 m (incumbent 6.99; formation 7.69; last position 12.55) | 3.06 m | **70 % / 71 %**: widen by ≈ 1.2× for such inputs |
| **SkillCorner real-broadcast re-entries** (20 matches, proxy truth) | 7.70 m (incumbent 7.88; formation 10.62) | 2.81 m | 78 % / 81 % |

Error grows with time unseen. On DFL, live:

| Time unseen | Mean error |
|---|---:|
| 0–2 s | 1.7 m |
| 6–10 s | 6.4 m |
| 11–30 s | 9.4 m |
| more than 30 s | 11.1 m |

## Intended use

- Estimating where unseen players probably are, for **team-level tactical context**: defensive line, width, compactness, shape. Always show an uncertainty region, and never present an estimate as an observation.
- Comparing camera-only, reconstructed and true views where truth exists.

## Do not trust the model for

- **Individual-player claims about a hidden player:** runs, distances, involvement. The error is often 5–15 m.
- **Players unseen for more than 30 s, and long camera cuts** (replays, close-ups). Error is about 9–11 m and calibration slightly overconfident.
- **Our own video pipeline without verified identity.** With role-based identity (≈ 35 % correct), errors roughly double. The `cv_*` exports carry a ×2.67 radius scale and are experimental.
- **Inputs without possession or roles** (GSR-like): the radius under-covers, at about 70 %.
- **Live mode at the moment of reappearance:** estimates jump to the new sighting.
- **Other competitions, women's football, youth or amateur football:** not evaluated. Training covers 7 German men's matches; external tests cover EPL / UEFA broadcast clips (GSR) and the A-League.
- **Synthetic matches:** the model was not tuned for them, and the formation estimator is the default there.

## Rejected alternative

A GRU with temporal context was best on DFL (live 7.72 m, replay 6.37 m). It was worse on both real-broadcast tests (GSR live 7.46 m, SkillCorner live 8.85 m) and under degraded context, so it was not deployed. Its checkpoints are kept in `cv/weights/recon/r13_gru/` for research.

## Licences

- DFL data: CC BY 4.0. Credit Bassek et al. 2025.
- SkillCorner open data: MIT.
- SoccerNet: NDA / research terms. SoccerNet is used here for evaluation only, and no footage is redistributed.
