# Real tracking · DFL open data: dfl-J03WMX-r13

## h1

**Data basis:** In-stadium tracking of all players (ground truth); official event feed. Positions: `ground_truth`. Events: `recorded_event (official DFL feed)`.

Hidden-player model: `r13_mlp_aug_ens3`.

| | A | B |
|---|---:|---:|
| Goals | 0 | 1 |
| Shots | 5 | 4 |
| xG (engine model) | 0.27 | 0.31 |
| Passes | 227 | 331 |
| Pass accuracy | 0.819 | 0.873 |
| Possession (s) | 1343 | 1486 |
| Press index (heuristic) | 25.6 | 27.5 |
| Last formation (heuristic) | 5-4-1 | 4-1-4-1 |

**Formations over time (heuristic):** A: 0' 4-2-3-1, 20' 5-4-1, 45' 3-5-2; B: 0' 4-1-4-1, 25' 4-1-3-2

**Counterattacks detected (experimental):** 3

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
| 5' | TACTICAL_TURNING_POINT | B | FC Bayern München push up | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 8' | GOAL_BUILD_UP | B | Goal FC Bayern München - Kingsley Coman | 0.95 | 1.0 | recorded goal; build-up sequence from events |
| 26' | FORMATION_SHIFT | B | FC Bayern München switch to 4-1-3-2 | 0.92 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups; 57% minute agreement truth vs reconstruction) |
| 42' | TACTICAL_TURNING_POINT | B | FC Bayern München take territorial control | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |

### Evidence chains (top 3 by importance)

**8' Goal FC Bayern München - Kingsley Coman** (GOAL_BUILD_UP). Chain: EVENT: Spell of possession → CONSEQUENCE: 1 passes, 7.3s → OUTCOME: Goal (0.04 xG)

- [recorded_event] Build-up: 1 passes in 7.3s
- [recorded_event] Shot: Kingsley Coman shot from 20.5m
- [recorded_event] Outcome: Goal - 0-1
- Followed by (not proof of causation): Score 0-1
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [463, 473], "observed_players": 0.0, "inferred_players": 10.0, "high_confidence_inferred": 0.0, "outfield_players": 10, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 53.8, "width_m": 31.0, "near_ball": 1.1}, "truth": {"def_line_m": 52.8, "width_m": 34.4`

**26' FC Bayern München switch to 4-1-3-2** (FORMATION_SHIFT). Chain: TACTICAL_CHANGE: 4-1-4-1 -> 4-1-3-2 → STATE_CHANGE: Right-side press 21.1% -> 16.7% → CONSEQUENCE: 1 final-third recoveries

- [heuristic over ground_truth positions] Tactical trigger: Shape changed 4-1-4-1 -> 4-1-3-2
- [ground_truth positions] Structural change: Defensive line -19.17m, width -8.93m
- [ground_truth positions] Pressing: Press intensity 37.7 -> 22.2; right-side share of pressures 21.1% -> 16.7%
- [recorded_event] Recovery: Final-third recoveries 1 -> 1
- [ground_truth positions] Territory: Territory 62.5% -> 44.1%
- Followed by (not proof of causation): 1 final-third recoveries, Territory 44.1%, 0 shots
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [1495, 1505], "observed_players": 2.1, "inferred_players": 7.9, "high_confidence_inferred": 4.8, "outfield_players": 10, "metrics": {"observed_only": {"def_line_m": 57.2, "width_m": 13.9, "near_ball": 4.0}, "reconstructed": {"def_line_m": 41.5, "width_m": 28.4, "near_ball":`

**5' FC Bayern München push up** (TACTICAL_TURNING_POINT). Chain: TACTICAL_CHANGE: High Press → STATE_CHANGE: Line 35.7m -> 39.0m → CONSEQUENCE: 0 counterattacks

- [heuristic over ground_truth positions] Tactical trigger: Block pushed up: defensive line 35.7m -> 39.0m
- [recorded_event] Possession: Possession 71.5% -> 45.3%
- [ground_truth positions] Territory: Territory 26.0% -> 51.1%
- [recorded_event] Outcome: 1 shots (0.04 xG) in the following minutes
- Followed by (not proof of causation): Territory 51.1%, 1 shots (0.04 xG)
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [235, 245], "observed_players": 8.1, "inferred_players": 1.9, "high_confidence_inferred": 1.9, "outfield_players": 10, "metrics": {"observed_only": {"def_line_m": 42.2, "width_m": 31.2, "near_ball": 4.3}, "reconstructed": {"def_line_m": 41.5, "width_m": 39.1, "near_ball": 4`

### Tactical & formation changes (every change, both teams)

**Team A** (9 changes)
- 2' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +8.2m, possession -15.3%, territory +31.3%. Followed by 1 shot(s) for (0.05 xG) and 0 against (0 xG). _(ground truth positions)_
- 5' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -13.5m, possession +8.5%, territory -41.7%. Followed by 0 shot(s) for (0 xG) and 1 against (0.04 xG), 1 against. _(ground truth positions)_
- 15' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +17.0m, width -5.2m, territory +45.7%, press index +24.6. Followed by 0 shot(s) for (0 xG) and 1 against (0.04 xG). _(ground truth positions)_
- 18' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -28.2m, width -7.8m, possession -44.0%, territory -37.6%, press index -39.2. Followed by 0 shot(s) for (0 xG) and 1 against (0.15 xG). _(ground truth positions)_
- 21' [FORMATION] Shape read as 4-2-3-1 -> 5-4-1. Ellyes Skhiri defensive midfield -> defence; E. Martel defensive midfield -> midfield; D. Ljubicic attacking midfield -> midfield; F. Kainz attack -> midfield. Defensive line (out of possession) +3.4m, width +5.6m, possession +9.0%, press index +17.6. Followed by 1 shot(s) for (0.1 xG) and 0 against (0 xG). _(ground truth positions)_
- 26' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) +14.4m, possession +30.0%, territory +10.8%. Followed by 1 shot(s) for (0.02 xG) and 0 against (0 xG). Preceded by card 24'. _(ground truth positions)_
- 39' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +16.8m, possession +27.0%, territory +28.3%, press index +24.0. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 43' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -23.9m, width -6.9m, possession -17.2%, territory -53.8%, press index -11.0. Followed by 0 shot(s) for (0 xG) and 1 against (0.07 xG). _(ground truth positions)_
- 46' [FORMATION] Shape read as 5-4-1 -> 3-5-2. B. Schmitz defence -> midfield; Jonas Hector defence -> midfield; F. Kainz midfield -> attack. Defensive line (out of possession) -3.1m, width +4.7m, possession +41.7%, territory -17.4%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_

**Team B** (8 changes)
- 6' [TACTIC] From balanced to high press. Defensive line (out of possession) +13.5m, territory +32.2%, press index +24.7. Followed by 1 shot(s) for (0.04 xG) and 0 against (0 xG), 1 goal(s) for. _(ground truth positions)_
- 9' [TACTIC] From high press to balanced. Width -10.0m, possession -26.7%, press index -34.2. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by goal scored 8'. _(ground truth positions)_
- 11' [TACTIC] From balanced to territorial control. Width +4.8m, territory +7.0%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by goal scored 8'. _(ground truth positions)_
- 13' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -16.5m, territory -28.0%, press index -23.5. Followed by 0 shot(s) for (0 xG) and 1 against (0.09 xG). _(ground truth positions)_
- 15' [TACTIC] From balanced to low block. Defensive line (out of possession) -16.7m, width -9.0m, compactness -3.4m, possession -10.0%, territory -33.0%, press index -9.0. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 17' [TACTIC] From low block to balanced. Defensive line (out of possession) +37.2m, width +8.7m, compactness +3.1m, possession +38.9%, territory +36.8%, press index +30.1. Followed by 2 shot(s) for (0.2 xG) and 0 against (0 xG). _(ground truth positions)_
- 26' [FORMATION] Shape read as 4-1-4-1 -> 4-1-3-2. Leroy Sané attacking midfield -> attack. Defensive line (out of possession) -8.5m, width -10.9m, compactness -3.6m, possession -30.0%, territory -10.8%, press index -26.9. Followed by 0 shot(s) for (0 xG) and 1 against (0.02 xG). Preceded by opponent card 24'. _(ground truth positions)_
- 44' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +17.0m, possession -9.6%, territory +62.4%. Followed by 1 shot(s) for (0.07 xG) and 0 against (0 xG). _(ground truth positions)_

### Does the analysis survive reconstruction? (truth vs camera reconstruction)

Formation agreement per team-minute: 26%. Mean absolute difference per minute: defensive line 1.39 m, width 3.04 m, compactness 1.09 m, press index 4.86. Position-based moments: truth 3, reconstruction 2, recalled within ±120 s 2, reconstruction-only 0.

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Positions are ground truth; the camera view is simulated.

## h2

**Data basis:** In-stadium tracking of all players (ground truth); official event feed. Positions: `ground_truth`. Events: `recorded_event (official DFL feed)`.

Hidden-player model: `r13_mlp_aug_ens3`.

| | A | B |
|---|---:|---:|
| Goals | 1 | 1 |
| Shots | 4 | 8 |
| xG (engine model) | 1.06 | 0.87 |
| Passes | 234 | 192 |
| Pass accuracy | 0.769 | 0.755 |
| Possession (s) | 1643 | 1368 |
| Press index (heuristic) | 26.6 | 18.0 |
| Last formation (heuristic) | 4-2-3-1 | 3-4-3 |

**Formations over time (heuristic):** A: 0' 4-2-3-1; B: 0' 3-4-3, 35' 4-1-4-1, 45' 4-3-3

**Counterattacks detected (experimental):** 2

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
| 59' | TACTICAL_TURNING_POINT | B | FC Bayern München drop into a low block | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 62' | SUBSTITUTION_IMPACT | A | K. Schindler on for F. Kainz | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 62' | SUBSTITUTION_IMPACT | A | Jan Thielmann on for D. Selke | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 62' | SUBSTITUTION_IMPACT | A | D. Huseinbašić on for B. Schmitz | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 62' | SUBSTITUTION_IMPACT | B | Joao Cancelo on for T. Müller | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 67' | TACTICAL_TURNING_POINT | A | 1. FC Köln drop into a low block | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 71' | SUBSTITUTION_IMPACT | B | Leon Goretzka on for Ryan Gravenberch | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 71' | SUBSTITUTION_IMPACT | B | E. Choupo-Moting on for Kingsley Coman | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 81' | FORMATION_SHIFT | B | FC Bayern München switch to 4-1-4-1 | 0.88 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups; 57% minute agreement truth vs reconstruction) |
| 81' | GOAL_BUILD_UP | A | Goal 1. FC Köln - D. Ljubicic | 0.95 | 1.0 | recorded goal; build-up sequence from events |
| 82' | SUBSTITUTION_IMPACT | A | Kristian Pedersen on for D. Ljubicic | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 82' | SUBSTITUTION_IMPACT | A | T. Lemperle on for L. Maina | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 85' | SUBSTITUTION_IMPACT | B | J. Musiala on for N. Mazraoui | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 85' | SUBSTITUTION_IMPACT | B | Mathys Tel on for Leon Goretzka | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 88' | COUNTERATTACK | B | FC Bayern München counterattack | 0.78 | 1.0 | heuristic; vs DFL CounterAttack flags F1 0.55; vs SkillCorner quick-break phases P 0.62 / R 0.23 (experimental) |
| 89' | TACTICAL_TURNING_POINT | A | 1. FC Köln drop into a low block | 0.78 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 89' | GOAL_BUILD_UP | B | Goal FC Bayern München - J. Musiala | 0.95 | 1.0 | recorded goal; build-up sequence from events |

### Evidence chains (top 3 by importance)

**81' Goal 1. FC Köln - D. Ljubicic** (GOAL_BUILD_UP). Chain: EVENT: Spell of possession → CONSEQUENCE: 1 passes, 119.7s → OUTCOME: Goal (0.767 xG)

- [recorded_event] Build-up: 1 passes in 119.7s
- [recorded_event] Shot: D. Ljubicic shot from 11.0m
- [recorded_event] Outcome: Goal - 1-1
- Followed by (not proof of causation): Score 1-1
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [2099, 2109], "observed_players": 1.2, "inferred_players": 8.8, "high_confidence_inferred": 3.5, "outfield_players": 15, "metrics": {"observed_only": {"def_line_m": 101.3, "width_m": 11.8, "near_ball": 3.3}, "reconstructed": {"def_line_m": 79.0, "width_m": 20.0, "near_ball"`

**89' Goal FC Bayern München - J. Musiala** (GOAL_BUILD_UP). Chain: EVENT: Spell of possession → CONSEQUENCE: 3 passes, 8.3s → OUTCOME: Goal (0.05 xG)

- [recorded_event] Build-up: 3 passes in 8.3s
- [recorded_event] Shot: J. Musiala shot from 19.4m
- [recorded_event] Outcome: Goal - 1-2
- Followed by (not proof of causation): Score 1-2
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [2589, 2599], "observed_players": 2.8, "inferred_players": 7.2, "high_confidence_inferred": 5.7, "outfield_players": 15, "metrics": {"observed_only": {"def_line_m": 80.5, "width_m": 24.7, "near_ball": 3.2}, "reconstructed": {"def_line_m": 69.2, "width_m": 42.9, "near_ball":`

**81' FC Bayern München switch to 4-1-4-1** (FORMATION_SHIFT). Chain: TACTICAL_CHANGE: 3-4-3 -> 4-1-4-1 → STATE_CHANGE: Right-side press 50.0% -> 34.8% → CONSEQUENCE: 0 final-third recoveries → OUTCOME: Goal

- [heuristic over ground_truth positions] Tactical trigger: Shape changed 3-4-3 -> 4-1-4-1
- [ground_truth positions] Structural change: Defensive line +10.65m, width +11.87m
- [ground_truth positions] Pressing: Press intensity 6.9 -> 23.9; right-side share of pressures 50.0% -> 34.8%
- [recorded_event] Recovery: Final-third recoveries 1 -> 0
- [ground_truth positions] Territory: Territory 38.7% -> 43.9%
- [recorded_event] Outcome: Goal at 88:14
- Followed by (not proof of causation): 0 final-third recoveries, Territory 43.9%, 2 shots
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [2095, 2105], "observed_players": 0.0, "inferred_players": 10.0, "high_confidence_inferred": 5.8, "outfield_players": 15, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 9.6, "width_m": 18.4, "near_ball": 5.2}, "truth": {"def_line_m": 16.6, "width_m": 21.`

### Tactical & formation changes (every change, both teams)

**Team A** (10 changes)
- 49' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) +11.1m, width +4.8m, possession +8.3%, territory +9.0%, press index -9.9. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 52' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +6.2m, width -4.0m, possession -6.0%, territory +43.6%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 55' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -9.2m, territory -22.1%. Followed by 1 shot(s) for (0.02 xG) and 1 against (0.04 xG). _(ground truth positions)_
- 60' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +14.6m, width +3.1m, possession +16.0%, territory +10.6%, press index +19.0. Followed by 1 shot(s) for (0.03 xG) and 0 against (0 xG). _(ground truth positions)_
- 69' [TACTIC] From territorial control to low block. Defensive line (out of possession) -33.3m, width -13.7m, compactness -4.8m, possession -17.6%, territory -50.4%, press index -22.4. Followed by 0 shot(s) for (0 xG) and 3 against (0.18 xG). _(ground truth positions)_
- 74' [TACTIC] From low block to territorial control. Defensive line (out of possession) +35.4m, width +9.1m, compactness +3.4m, territory +54.3%, press index +31.4. Followed by 1 shot(s) for (0.24 xG) and 1 against (0.06 xG). Preceded by opponent substitution 26': Leon Goretzka on for Ryan Gravenberch; opponent substitution 26': E. Choupo-Moting on for Kingsley Coman. _(ground truth positions)_
- 76' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -12.0m, width +5.2m, possession +28.0%, territory -25.0%. Followed by 0 shot(s) for (0 xG) and 1 against (0.11 xG). _(ground truth positions)_
- 80' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +9.0m, width -8.1m, possession -24.6%, territory +43.3%, press index -18.8. Followed by 1 shot(s) for (0.77 xG) and 0 against (0 xG), 1 goal(s) for. _(ground truth positions)_
- 85' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -7.1m, width +4.8m, possession +16.9%, territory -38.6%, press index +14.4. Followed by 0 shot(s) for (0 xG) and 1 against (0.42 xG). Preceded by substitution 37': Kristian Pedersen on for D. Ljubicic; substitution 37': T. Lemperle on for L. Maina; opponent substitution 40': J. Musiala on for N. Mazraoui; opponent substitution 40': Mathys Tel on for Leon Goretzka. _(ground truth positions)_
- 89' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) -10.9m, width -4.7m, possession -6.3%, territory -25.8%, press index -12.7. Followed by 0 shot(s) for (0 xG) and 1 against (0.05 xG), 1 against. Preceded by goal conceded 44'. _(ground truth positions)_

**Team B** (15 changes)
- 47' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +14.8m, width +5.7m, territory +23.0%, press index +8.1. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 50' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -10.8m, possession -11.3%, territory -31.0%, press index -11.6. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 52' [TACTIC] From balanced to low block. Defensive line (out of possession) -20.3m, width -6.3m, territory -37.0%, press index -16.0. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 54' [TACTIC] From low block to balanced. Defensive line (out of possession) +18.6m, width +5.9m, possession +9.8%, territory +25.6%. Followed by 1 shot(s) for (0.04 xG) and 1 against (0.02 xG). _(ground truth positions)_
- 60' [TACTIC] From balanced to low block. Defensive line (out of possession) -12.7m, width -10.0m, compactness -4.2m, possession -16.5%, territory -14.1%, press index -9.1. Followed by 0 shot(s) for (0 xG) and 1 against (0.03 xG). _(ground truth positions)_
- 63' [TACTIC] From low block to balanced. Defensive line (out of possession) +5.6m, width +9.4m, compactness +4.0m, possession +12.5%, territory +7.8%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by opponent substitution 17': K. Schindler on for F. Kainz; opponent substitution 17': Jan Thielmann on for D. Selke; opponent substitution 17': D. Huseinbašić on for B. Schmitz; substitution 17': Joao Cancelo on for T. Müller. _(ground truth positions)_
- 67' [STRUCTURE] Structural change in how the team is positioned. Possession +18.5%, territory +13.5%, press index +20.6. Followed by 1 shot(s) for (0.05 xG) and 0 against (0 xG). _(ground truth positions)_
- 70' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +24.7m, width -4.7m, possession +14.4%, territory +47.3%, press index +26.0. Followed by 2 shot(s) for (0.14 xG) and 0 against (0 xG). _(ground truth positions)_
- 73' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -16.5m, possession -14.3%, territory -48.9%, press index -37.4. Followed by 2 shot(s) for (0.17 xG) and 1 against (0.24 xG). Preceded by substitution 26': Leon Goretzka on for Ryan Gravenberch; substitution 26': E. Choupo-Moting on for Kingsley Coman. _(ground truth positions)_
- 80' [TACTIC] From balanced to low block. Defensive line (out of possession) -18.2m, width -6.3m, compactness -4.5m, possession -26.0%, territory -32.0%, press index -8.3. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 81' [FORMATION] Shape read as 3-4-3 -> 4-1-4-1. Joao Cancelo midfield -> defence; Joshua Kimmich midfield -> defensive midfield; Serge Gnabry attack -> attacking midfield; Leroy Sané attack -> attacking midfield. Defensive line (out of possession) -3.6m, width +6.3m, possession +37.5%, territory -13.0%, press index +29.9. Followed by 0 shot(s) for (0 xG) and 1 against (0.77 xG), 1 against. Preceded by goal conceded 36'. _(ground truth positions)_
- 83' [TACTIC] From low block to balanced. Defensive line (out of possession) +22.1m, width +8.3m, compactness +5.1m, possession +10.7%, territory +33.8%, press index -13.6. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by goal conceded 36'; opponent card 36'; opponent substitution 37': Kristian Pedersen on for D. Ljubicic; opponent substitution 37': T. Lemperle on for L. Maina. _(ground truth positions)_
- 91' [FORMATION] Shape read as 4-1-4-1 -> 4-3-3. Joshua Kimmich defensive midfield -> midfield; Serge Gnabry attacking midfield -> midfield; J. Musiala attacking midfield -> midfield; Leroy Sané attacking midfield -> attack. Defensive line (out of possession) +10.1m, possession +8.4%, territory +29.4%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by goal scored 44'; opponent card 45'. _(ground truth positions)_
- 91' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +10.1m, possession +8.4%, territory +29.4%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by goal scored 44'; opponent card 45'. _(ground truth positions)_
- 93' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -8.1m, width -4.8m, possession +23.8%, territory -27.8%, press index +11.5. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by opponent card 45'. _(ground truth positions)_

### Does the analysis survive reconstruction? (truth vs camera reconstruction)

Formation agreement per team-minute: 61%. Mean absolute difference per minute: defensive line 1.85 m, width 2.89 m, compactness 1.13 m, press index 3.52. Position-based moments: truth 5, reconstruction 5, recalled within ±120 s 3, reconstruction-only 2.

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Positions are ground truth; the camera view is simulated.
