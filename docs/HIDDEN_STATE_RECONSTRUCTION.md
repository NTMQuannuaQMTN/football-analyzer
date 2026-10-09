# Hidden-player reconstruction: datasets, model and results

## Dataset decision

No public dataset has elite **broadcast video and true full-pitch positions of the same match** under a licence suitable for a public demo. Broadcast-derived tracking (SkillCorner, PFF) estimates its own off-screen players, so those positions are not ground truth. The datasets that do have full ground truth (in-stadium tracking, panoramas) have no broadcast view. We therefore combine two sources:

| Dataset | What it has | Licence | Grade | Use |
|---|---|---|---|---|
| **DFL / IDSSE open data** (Bassek et al. 2025, figshare 10.6084/m9.figshare.28196177) | 7 Bundesliga / 2. BL matches, TRACAB in-stadium tracking of all 22 players + ball, 25 Hz, identities and positions | CC BY 4.0 | **A (combined)** | **Primary ground truth**: train / val / test |
| **SkillCorner Open Data** (github.com/SkillCorner/opendata) | 20 A-League matches of broadcast tracking (10 fps), per-frame camera footprint on the pitch, `is_detected` per player | MIT | **B** | **Real broadcast camera behaviour** (pan, zoom, cuts, detection misses), transferred onto DFL truth as masks |
| Synthetic simulator (this repo) | exact positions of all 22 every second | ours | A (synthetic) | Controlled benchmark (separate from real) |
| SoccerNet-GSR | broadcast clips + pitch positions of *visible* players | research / SoccerNet terms | C | Detector fine-tune / evaluation only; no hidden-player truth |
| SoccerTrack v2 | 10 university matches, panoramic 4K + pitch coordinates | CC BY 4.0 | B | Future: amateur full-state data (no broadcast view) |
| Metrica sample data | 3 anonymised matches of full tracking | attribution requested | B | Earlier off-screen MLP |
| StatsBomb 360 | broadcast visible area + visible players at events | non-commercial | B, not used | Licence unsuitable |
| PFF FC WC 2022 | broadcast-derived tracking, 64 matches | request-gated | C | Hidden positions are estimates, not truth |
| FOOTPASS (2025) | 54 matches, broadcast + game-state | availability unclear | C | Watch |

**Masking:** the `sc_camera` mask moves each real SkillCorner footprint onto the DFL ball position (corners − SkillCorner ball + DFL ball), keeping real cuts (37% of seconds) and the real 97% in-footprint detection rate. As secondary benchmarks: nearest-k to the camera for 18/16/14/12/10/8/6 visible, and random masking.

**Split (match level):** Fortuna Düsseldorf plays in 5 of the 7 matches. Train = those 5; validation and test = Köln–Bayern and Bochum–Leverkusen, swapped across 2 folds, so **test teams are never seen in training**. The test match is not used for model selection or calibration.

## Model

- **Baselines:** last position; constant velocity (decaying); team-shape / formation estimator (`backend/reconstruction/estimator.py`, live and replay).
- **Learned model** (`backend/reconstruction/ml.py`): an MLP (256-256-128) predicting a **correction to the formation estimate** plus log σ.
  - **Features:** prior estimate, last-seen position / velocity / time since seen, ball now and at last sighting, visible teammate and opponent centroids and counts, the 3 nearest visible teammates and 3 nearest visible opponents (relative), possession, ball in play, camera cut, coarse role. Replay also adds the next-sighting position and time until it.
  - **Why an MLP:** about 1.3 M samples; history is summarised in the features; trains on a laptop CPU in minutes. Relationships enter through pooled neighbour features (GNN-lite).
  - **Training:** Huber loss on the correction + Gaussian NLL on σ (mean detached); AdamW, cosine schedule, 12 epochs, best epoch chosen on validation.
- **Uncertainty:** σ deciles on the validation match → empirical 80% radius and P(error ≤ 5 m). On the unseen test match the true position falls inside the 80% radius **80.6% (live) / 80.9% (replay)**.
- **Live vs replay:** separate models; `tests/test_ml_leakage.py` proves live features ignore the future.

## Real-data results (hidden players, metres; test teams unseen)

Columns: last position | constant velocity | formation live / replay | **ML live / replay** | ML replay RMSE | ML replay ≤5 m / ≤10 m

Fold 0, test Köln–Bayern:

| Mask | Last | CV | Formation L/R | ML L/R | RMSE | ≤5 / ≤10 m |
|---|---|---|---|---|---|---|
| real broadcast camera | 16.02 | 15.63 | 13.8 / 10.42 | **8.2 / 7.05** | 9.77 | 50% / 76% |
| k=18 | 9.22 | 8.39 | 6.8 / 5.95 | **3.85 / 3.04** | 4.56 | 85% / 96% |
| k=16 | 11.21 | 10.23 | 7.16 / 6.12 | **4.4 / 3.38** | 4.91 | 81% / 96% |
| k=14 | 12.76 | 11.77 | 7.8 / 6.67 | **4.88 / 3.86** | 5.52 | 76% / 94% |
| k=12 | 13.75 | 12.79 | 8.5 / 7.2 | **5.36 / 4.32** | 6.08 | 71% / 92% |
| k=10 | 15.09 | 14.26 | 9.23 / 7.75 | **5.86 / 4.8** | 6.64 | 66% / 90% |
| k=8 | 16.65 | 15.91 | 10.3 / 8.82 | **6.6 / 5.55** | 7.55 | 59% / 86% |
| k=6 | 18.08 | 17.64 | 11.37 / 10.26 | **7.35 / 6.5** | 8.64 | 51% / 81% |
| rand=10 | 3.62 | 3.1 | 3.55 / 2.24 | **2.59 / 1.46** | 1.98 | 97% / 100% |

Fold 1, test Bochum–Leverkusen:

| Mask | Last | CV | Formation L/R | ML L/R | RMSE | ≤5 / ≤10 m |
|---|---|---|---|---|---|---|
| real broadcast camera | 15.47 | 15.0 | 13.9 / 10.03 | **8.34 / 6.97** | 9.76 | 52% / 77% |
| k=18 | 9.62 | 8.72 | 7.0 / 6.38 | **3.9 / 3.26** | 4.96 | 83% / 96% |
| k=16 | 10.37 | 9.33 | 6.95 / 6.11 | **4.29 / 3.57** | 5.22 | 79% / 95% |
| k=14 | 11.92 | 10.77 | 7.5 / 6.34 | **4.88 / 3.97** | 5.58 | 74% / 94% |
| k=12 | 13.21 | 12.02 | 7.98 / 6.66 | **5.34 / 4.37** | 6.08 | 70% / 92% |
| k=10 | 14.24 | 13.16 | 8.7 / 7.17 | **5.84 / 4.76** | 6.52 | 66% / 90% |
| k=8 | 15.94 | 14.99 | 9.74 / 8.0 | **6.55 / 5.44** | 7.42 | 60% / 87% |
| k=6 | 17.66 | 16.89 | 12.13 / 11.1 | **7.57 / 6.6** | 8.79 | 50% / 80% |
| rand=10 | 3.14 | 2.67 | 3.29 / 2.26 | **2.38 / 1.37** | 1.83 | 98% / 100% |

Per role and tactical state (fold 0, real camera mask, replay): ML beats the formation estimator for every role (e.g. GK 7.9 → 4.5 m, CB 10.4 → 6.7 m, ST 10.7 → 7.8 m) and in every state (in/out of possession, dead ball, ball in any third).

## Synthetic benchmark (ML trained on real data, applied to the simulator; 10 matches)

| Mask | Formation L/R | ML L/R |
|---|---|---|
| broadcast | 5.12 / 4.02 | 5.25 / 4.11 |
| k=14 | 4.13 / 3.47 | 5.13 / 4.31 |
| k=10 | 4.98 / 4.01 | 5.69 / 4.6 |
| k=6 | 6.89 / 5.33 | 6.77 / 5.46 |

The real-trained model does **not** help on the simulator (domain gap: simulated movement is simpler, and the formation estimator was tuned on synthetic seed 11). The synthetic demo therefore keeps the formation estimator by default (`RECON_MODEL=ml` switches); the real CV path (`cv/build_match.py`) uses the learned model.

## Limitations

- **No real broadcast and full truth for the same match:** camera behaviour is transferred from other matches (A-League) onto Bundesliga truth, relative to the ball. Footprint perspective may not exactly match the DFL stadiums.
- **Small dataset:** 7 matches, one recurring team (Düsseldorf); 2 folds only.
- **Identity assumed known in evaluation;** in real VOD, identity errors (no shirt-number reading yet) add on top.
- **The keeper-zone constraint** is applied at inference only.
