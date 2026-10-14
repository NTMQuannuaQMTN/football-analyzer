# Reconstruction test guide

How to reproduce and check the 13 October 2026 reconstruction work. All commands run from the repo root.

## 1. Automated tests

```bash
.venv/bin/python -m pytest -q tests -p no:cacheprovider          # full suite
.venv/bin/python -m pytest -q tests/test_reconstruction_r13.py    # this session's regression tests (~10 s)
(cd frontend && npm run lint && npx tsc --noEmit -p .)
```

`tests/test_reconstruction_r13.py` covers:

| Test | Guarantees |
|---|---|
| `test_live_context_window_never_reads_the_future` | The GRU's live window at time t is unchanged when every input after t is randomised. |
| `test_replay_context_window_is_retrospective` | Replay windows do read later seconds. This is documented, not hidden. |
| `test_live_gru_prediction_is_causal_end_to_end` | Live features + context + GRU output up to t do not depend on any data after t. |
| `test_benchmark_refuses_checkpoints_that_saw_the_test_match` | The benchmark refuses a checkpoint that trained on the scored match, or that has no recorded held-out match. |
| `test_fold_split_is_match_level` | Each of the 7 folds holds out exactly one whole match; `final` has no test match. |
| `test_reentries_need_a_prior_sighting_and_a_gap` | Real-broadcast re-entry extraction (GSR / SkillCorner). |
| `test_skillcorner_event_coordinates_undo_attacking_side_normalisation` | Provider event coordinates are de-normalised per event (see the orientation finding below). |
| `test_catalog_groups_skillcorner_halves` | A-League halves appear as one two-half lobby match, with the "model estimate" note. |
| `test_reinferred_export_keeps_measured_positions_and_never_reads_stored_estimates` | Re-inference uses measured positions only and copies them unchanged. |
| `test_goal_without_a_located_shot_does_not_break_the_engine` | Engine fix: a goal with no located shot no longer crashes moment detection. |

Earlier guarantees still hold: live never uses the future (`test_reconstruction.py`, `test_ml_leakage.py`), narratives use only engine numbers, and so on.

## 2. Reproduce the evaluation

```bash
# features for training (cached in data/eval/real/recon_cache, ~90 s)
.venv/bin/python -m backend.reconstruction.train_hidden --build-only --tag x
# train the 7 held-out folds (test match never read) and the all-match final model
.venv/bin/python -m backend.reconstruction.train_hidden --build-only --augment --tag x                      # degraded clips (~70 s)
.venv/bin/python -m backend.reconstruction.train_hidden --arch mlp --fold all --tag r13_mlp_aug_ens3 --seeds 0,1,2 --augment
.venv/bin/python -m backend.reconstruction.train_hidden --arch mlp --fold final --tag r13_mlp_aug_ens3 --seeds 0,1,2 --augment
.venv/bin/python -m backend.reconstruction.train_hidden --arch gru_ctx --fold all --tag r13_gru --epochs 10
# model selection: validation-tail errors only
.venv/bin/python -c "from backend.reconstruction.train_hidden import validation_table as v; import json; print(json.dumps(v(['r13_mlp','r13_mlp_ens3','r13_gru']), indent=1))"
# DFL 7-fold benchmark (every method, identical rows), then the report
.venv/bin/python -m backend.reconstruction.benchmark run --run r13_base --methods base,k7r_v3
.venv/bin/python -m backend.reconstruction.benchmark run --run r13_gru --methods r13_gru
.venv/bin/python -m backend.reconstruction.benchmark report --runs r13_base,r13_mlp,r13_ens,r13_gru --name final
# real-broadcast tests (external; DFL-trained final models)
.venv/bin/python -m backend.reconstruction.gsr_benchmark --methods base,final_v3,r13_mlp_aug_ens3_final --name gsr_final
.venv/bin/python -m backend.reconstruction.skillcorner_benchmark --methods base,final_v3,r13_mlp_aug_ens3_final --name sc_final
```

The exact model names and output files of this session are listed in [reconstruction_evaluation.md](reconstruction_evaluation.md).

## 3. Reproduce the reconstructed matches and the analysis

```bash
.venv/bin/python -m backend.reconstruction.skillcorner_match --model r13_mlp_aug_ens3_final --all   # 20 A-League matches -> data/skc_*_h{1,2}.json
.venv/bin/python -m backend.reconstruction.apply_exports --model r13_mlp_aug_ens3_final --tag r13   # Arsenal clip + 2 SkillCorner EPL -> data/cv_r13_*.json
.venv/bin/python -m backend.reconstruction.apply_dfl --model r13_mlp_aug_ens3 --tag r13   # 7 DFL matches -> data/dfl_*_r13_h{1,2}.json
.venv/bin/python -m backend.analysis_report --matches all                                  # docs/match_reports/*.md, data/eval/match_reports/*.json
```

## 4. Manual checks in the app

Start with `./run.sh`, open http://localhost:3000, then check:

1. **Lobby:**
   - The A-League matches show as `FT` with the label "Broadcast tracking · SkillCorner · hidden players estimated" and a note giving the detected share.
   - DFL matches re-exported this session show "camera estimate r13".
2. **Studio for an A-League match** (`/match/skc-<id>/studio?part=h1`):
   - Detected players are drawn as measured; estimated players are dashed, with a ± radius.
   - The player card says identities are SkillCorner's and unverified.
   - Switching to *Live* shows only past-based estimates. Players jump when they reappear, which is expected in live mode.
3. **Orientation:**
   - The home team attacks +x in both halves.
   - A goal event (`skc-1886347`, 2nd half) sits near the attacking goal of the scoring team.
4. **DFL studio:**
   - *Truth / Replay / Live* switch correctly.
   - The Error overlay shows estimated-vs-truth lines only for hidden players.
5. **Match story:** every moment links to a timestamp. Moments on A-League matches carry the "video-derived positions; identity/calibration unverified" basis.
6. **Failed or partial processing:** `data/eval/match_reports/index.json` lists any match whose analysis failed, with the error.

## 5. Known results to sanity-check against

These are copied from `reconstruction_evaluation.md`. If a rerun differs by more than about 0.05 m, the cache or the seeds changed.

- DFL `sc_camera` (1,316,306 hidden player-seconds):
  - existing `k7r_v3`: live 8.29 m (8.23 m `+speed`), replay 6.96 m (6.74 m `+speed`);
  - selected `r13_mlp_aug_ens3+speed`: live 8.10 m, replay 6.66 m.
- GSR re-entries (1,344 live, 3,000 replay): formation 7.69 m live; `final_v3` 6.99 / 3.20 m; selected 6.74 / 3.06 m.
- SkillCorner re-entries (94,918 live, 226,050 replay): formation 10.62 m live; `final_v3` 7.88 / 2.91 m; selected 7.70 / 2.81 m.
- Exports: `data/eval/export_quality_r13.json` should list 59 files with 0 issues. A-League goals should reproduce the official score in 19 of 20 matches (`skc-1899585` is flagged).
