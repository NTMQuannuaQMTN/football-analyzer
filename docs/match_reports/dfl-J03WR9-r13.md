# Real tracking · DFL open data: dfl-J03WR9-r13

## h1

**Data basis:** In-stadium tracking of all players (ground truth); official event feed. Positions: `ground_truth`. Events: `recorded_event (official DFL feed)`.

Hidden-player model: `r13_mlp_aug_ens3`.

| | A | B |
|---|---:|---:|
| Goals | 1 | 0 |
| Shots | 10 | 1 |
| xG (engine model) | 0.88 | 0.07 |
| Passes | 290 | 191 |
| Pass accuracy | 0.838 | 0.812 |
| Possession (s) | 1461 | 1274 |
| Press index (heuristic) | 26.3 | 13.1 |
| Last formation (heuristic) | 4-1-3-2 | 4-2-3-1 |

**Formations over time (heuristic):** A: 0' 4-1-3-2, 20' 3-4-3, 35' 4-1-3-2; B: 0' 4-2-3-1, 45' 3-4-3

**Counterattacks detected (experimental):** 4

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
| 6' | TACTICAL_TURNING_POINT | B | 1. FC Kaiserslautern drop into a low block | 0.87 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 14' | GOAL_BUILD_UP | A | Goal Fortuna Düsseldorf - M. Karbownik | 0.95 | 1.0 | recorded goal; build-up sequence from events |
| 28' | TACTICAL_TURNING_POINT | A | Fortuna Düsseldorf drop into a low block | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 44' | TACTICAL_TURNING_POINT | B | 1. FC Kaiserslautern drop into a low block | 0.88 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |

### Evidence chains (top 3 by importance)

**14' Goal Fortuna Düsseldorf - M. Karbownik** (GOAL_BUILD_UP). Chain: EVENT: Spell of possession → CONSEQUENCE: 3 passes, 14.7s → OUTCOME: Goal (0.056 xG)

- [recorded_event] Build-up: 3 passes in 14.7s
- [recorded_event] Shot: M. Karbownik shot from 17.1m
- [recorded_event] Outcome: Goal - 1-0
- Followed by (not proof of causation): Score 1-0
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [788, 798], "observed_players": 4.3, "inferred_players": 5.7, "high_confidence_inferred": 2.9, "outfield_players": 10, "metrics": {"observed_only": {"def_line_m": 80.3, "width_m": 28.1, "near_ball": 4.8}, "reconstructed": {"def_line_m": 72.2, "width_m": 47.4, "near_ball": 3`

**6' 1. FC Kaiserslautern drop into a low block** (TACTICAL_TURNING_POINT). Chain: TACTICAL_CHANGE: Low Block → STATE_CHANGE: Line 34.3m -> 28.9m → CONSEQUENCE: 0 counterattacks

- [heuristic over ground_truth positions] Tactical trigger: Block dropped: defensive line 34.3m -> 28.9m
- [ground_truth positions] Ceded the ball: Possession 59.2% -> 57.3%
- [ground_truth positions] Transitions: 0 counterattacks launched in the next 15 min
- [recorded_event] Outcome: 0 ended in shots
- Followed by (not proof of causation): 0 counterattacks, 0 shots (0 xG)
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [295, 305], "observed_players": 2.4, "inferred_players": 7.6, "high_confidence_inferred": 2.2, "outfield_players": 10, "metrics": {"observed_only": {"def_line_m": 19.9, "width_m": 42.5, "near_ball": 4.0}, "reconstructed": {"def_line_m": 27.0, "width_m": 41.8, "near_ball": 2`

**28' Fortuna Düsseldorf drop into a low block** (TACTICAL_TURNING_POINT). Chain: TACTICAL_CHANGE: Low Block → STATE_CHANGE: Line 49.0m -> 43.3m → CONSEQUENCE: 2 counterattacks

- [heuristic over ground_truth positions] Tactical trigger: Block dropped: defensive line 49.0m -> 43.3m
- [ground_truth positions] Ceded the ball: Possession 75.2% -> 60.0%
- [ground_truth positions] Transitions: 2 counterattacks launched in the next 15 min
- [recorded_event] Outcome: 0 ended in shots
- Followed by (not proof of causation): 2 counterattacks, 2 shots (0.05 xG)
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [1615, 1625], "observed_players": 0.0, "inferred_players": 10.0, "high_confidence_inferred": 0.0, "outfield_players": 10, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 42.3, "width_m": 41.4, "near_ball": 2.7}, "truth": {"def_line_m": 40.9, "width_m": 32`

### Tactical & formation changes (every change, both teams)

**Team A** (12 changes)
- 7' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +12.8m, possession +22.2%, territory +20.8%, press index +27.0. Followed by 3 shot(s) for (0.37 xG) and 0 against (0 xG). _(ground truth positions)_
- 11' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -16.2m, width -6.9m, possession -23.2%, territory -29.2%, press index -23.5. Followed by 1 shot(s) for (0.06 xG) and 0 against (0 xG), 1 goal(s) for. _(ground truth positions)_
- 15' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +11.5m, width -5.7m, possession -26.3%, territory +28.5%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by goal scored 14'. _(ground truth positions)_
- 17' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -11.2m, width +10.0m, compactness +4.0m, possession +17.7%, territory -23.5%. Followed by 1 shot(s) for (0.03 xG) and 1 against (0.07 xG). Preceded by goal scored 14'. _(ground truth positions)_
- 21' [FORMATION] Shape read as 4-1-3-2 -> 3-4-3. Jorrit Hendrix defensive midfield -> defence; Kristoffer Peterson attacking midfield -> midfield; M. Karbownik defence -> midfield; Matthias Zimmermann defence -> midfield. Defensive line (out of possession) +4.1m, width +7.7m, possession +26.8%, press index +20.3. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 26' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +9.6m, width +4.8m, possession +24.5%, territory +13.8%, press index +22.3. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 28' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -17.1m, width -13.5m, compactness -3.5m, possession -27.5%, territory -26.5%, press index -23.8. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 30' [TACTIC] From balanced to low block. Defensive line (out of possession) -18.2m, territory -37.0%, press index -34.0. Followed by 1 shot(s) for (0.03 xG) and 0 against (0 xG). _(ground truth positions)_
- 32' [TACTIC] From low block to balanced. Defensive line (out of possession) +27.9m, width +10.8m, compactness +3.6m, possession +9.5%, territory +35.5%, press index +25.2. Followed by 1 shot(s) for (0.02 xG) and 0 against (0 xG). _(ground truth positions)_
- 36' [FORMATION] Shape read as 3-4-3 -> 4-1-3-2. M. Karbownik midfield -> defence; Matthias Zimmermann midfield -> defence; Jorrit Hendrix defence -> defensive midfield; Kristoffer Peterson attack -> attacking midfield. Width -3.7m, possession -8.5%, territory +17.0%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 40' [TACTIC] From balanced to territorial control. Width -3.7m, possession -12.7%, territory +23.8%. Followed by 2 shot(s) for (0.37 xG) and 0 against (0 xG). _(ground truth positions)_
- 44' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) +15.1m, width -3.6m, possession +28.5%, territory +5.9%, press index +8.3. Followed by 1 shot(s) for (0.02 xG) and 0 against (0 xG). _(ground truth positions)_

**Team B** (8 changes)
- 7' [TACTIC] From balanced to low block. Defensive line (out of possession) -12.7m, width -9.1m, compactness -5.9m, possession -26.9%, territory -23.8%. Followed by 0 shot(s) for (0 xG) and 3 against (0.37 xG). _(ground truth positions)_
- 10' [TACTIC] From low block to balanced. Defensive line (out of possession) +12.0m, width +4.3m, possession +26.1%, territory +27.8%, press index +12.1. Followed by 0 shot(s) for (0 xG) and 1 against (0.06 xG), 1 against. _(ground truth positions)_
- 30' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +17.1m, possession +8.1%, territory +47.7%, press index +8.6. Followed by 0 shot(s) for (0 xG) and 1 against (0.03 xG). _(ground truth positions)_
- 33' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -10.2m, width +4.5m, territory -44.5%. Followed by 0 shot(s) for (0 xG) and 1 against (0.02 xG). _(ground truth positions)_
- 40' [TACTIC] From balanced to low block. Defensive line (out of possession) -12.4m, possession +5.5%, territory -22.8%, press index -13.2. Followed by 0 shot(s) for (0 xG) and 1 against (0.34 xG). _(ground truth positions)_
- 42' [TACTIC] From low block to balanced. Defensive line (out of possession) +3.8m, possession +19.5%. Followed by 0 shot(s) for (0 xG) and 1 against (0.02 xG). _(ground truth positions)_
- 44' [TACTIC] From balanced to low block. Defensive line (out of possession) -5.4m, width -8.0m, compactness -3.5m, possession -33.0%, territory -6.0%. Followed by 0 shot(s) for (0 xG) and 1 against (0.02 xG). _(ground truth positions)_
- 46' [FORMATION] Shape read as 4-2-3-1 -> 3-4-3, lasting only a minute. B. Tomiak defence -> midfield; H. Çiftçi defensive midfield -> midfield; Kenny Prince Redondo attacking midfield -> midfield; Terrence Boyd attack -> midfield. Width -7.0m, compactness -3.0m, possession -16.0%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_

### Does the analysis survive reconstruction? (truth vs camera reconstruction)

Formation agreement per team-minute: 82%. Mean absolute difference per minute: defensive line 1.77 m, width 3.2 m, compactness 1.17 m, press index 3.69. Position-based moments: truth 3, reconstruction 3, recalled within ±120 s 2, reconstruction-only 1.

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Positions are ground truth; the camera view is simulated.

## h2

**Data basis:** In-stadium tracking of all players (ground truth); official event feed. Positions: `ground_truth`. Events: `recorded_event (official DFL feed)`.

Hidden-player model: `r13_mlp_aug_ens3`.

| | A | B |
|---|---:|---:|
| Goals | 0 | 2 |
| Shots | 8 | 8 |
| xG (engine model) | 0.28 | 1.24 |
| Passes | 283 | 139 |
| Pass accuracy | 0.845 | 0.662 |
| Possession (s) | 1718 | 1421 |
| Press index (heuristic) | 27.6 | 15.6 |
| Last formation (heuristic) | 4-1-3-2 | 4-1-4-1 |

**Formations over time (heuristic):** A: 0' 4-1-3-2, 50' 3-4-3; B: 0' 4-1-4-1, 50' 3-4-2-1

**Counterattacks detected (experimental):** 3

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
| 50' | GOAL_BUILD_UP | B | Goal 1. FC Kaiserslautern - K. Kraus | 0.95 | 1.0 | recorded goal; build-up sequence from events |
| 71' | SUBSTITUTION_IMPACT | B | P. Klement on for M. Wunderlich | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 76' | SUBSTITUTION_IMPACT | A | Shinta Appelkamp on for E. Iyoha | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 77' | SUBSTITUTION_IMPACT | A | F. Klaus on for Kristoffer Peterson | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 81' | SUBSTITUTION_IMPACT | B | L. Lobinger on for Terrence Boyd | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 85' | TACTICAL_TURNING_POINT | A | Fortuna Düsseldorf take territorial control | 0.86 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 86' | SUBSTITUTION_IMPACT | A | M. Mansfeld on for R. Hennings | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 87' | SUBSTITUTION_IMPACT | B | Daniel Hanslik on for Kenny Prince Redondo | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 90' | SUBSTITUTION_IMPACT | A | E. Fernandes Neto on for M. Mansfeld | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 96' | FORMATION_SHIFT | A | Fortuna Düsseldorf switch to 3-4-3 | 0.92 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups; 57% minute agreement truth vs reconstruction) |
| 96' | FORMATION_SHIFT | B | 1. FC Kaiserslautern switch to 3-4-2-1 | 0.92 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups; 57% minute agreement truth vs reconstruction) |
| 96' | GOAL_BUILD_UP | B | Goal 1. FC Kaiserslautern - P. Klement | 0.95 | 1.0 | recorded goal; build-up sequence from events |

### Evidence chains (top 3 by importance)

**50' Goal 1. FC Kaiserslautern - K. Kraus** (GOAL_BUILD_UP). Chain: EVENT: Spell of possession → CONSEQUENCE: 1 passes, 27.6s → OUTCOME: Goal (0.116 xG)

- [recorded_event] Build-up: 1 passes in 27.6s
- [recorded_event] Shot: K. Kraus shot from 7.2m
- [recorded_event] Outcome: Goal - 1-1
- Followed by (not proof of causation): Score 1-1
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [285, 295], "observed_players": 0.0, "inferred_players": 10.0, "high_confidence_inferred": 0.0, "outfield_players": 13, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 75.5, "width_m": 15.5, "near_ball": 1.4}, "truth": {"def_line_m": 80.1, "width_m": 36.2`

**96' Goal 1. FC Kaiserslautern - P. Klement** (GOAL_BUILD_UP). Chain: STATE_CHANGE: 1 forced backward passes → EVENT: Spell of possession → CONSEQUENCE: 0 passes, 90.7s → OUTCOME: Goal (0.767 xG)

- [recorded_event] Forced backwards: Fortuna Düsseldorf made 1 backward passes before losing the ball
- [recorded_event] Build-up: 0 passes in 90.7s
- [recorded_event] Shot: P. Klement shot from 11.0m
- [recorded_event] Outcome: Goal - 1-2
- Followed by (not proof of causation): Score 1-2
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [3044, 3054], "observed_players": 5.6, "inferred_players": 4.4, "high_confidence_inferred": 2.5, "outfield_players": 13, "metrics": {"observed_only": {"def_line_m": 89.1, "width_m": 22.4, "near_ball": 3.3}, "reconstructed": {"def_line_m": 72.6, "width_m": 36.7, "near_ball":`

**96' 1. FC Kaiserslautern switch to 3-4-2-1** (FORMATION_SHIFT). Chain: TACTICAL_CHANGE: 4-1-4-1 -> 3-4-2-1 → STATE_CHANGE: Right-side press 33.3% -> 0.0% → CONSEQUENCE: 0 final-third recoveries → OUTCOME: Goal

- [heuristic over ground_truth positions] Tactical trigger: Shape changed 4-1-4-1 -> 3-4-2-1
- [ground_truth positions] Structural change: Defensive line +31.97m, width -5.57m
- [ground_truth positions] Pressing: Press intensity 21.1 -> 22.3; right-side share of pressures 33.3% -> 0.0%
- [recorded_event] Recovery: Final-third recoveries 0 -> 0
- [ground_truth positions] Territory: Territory 18.0% -> 100.0%
- [recorded_event] Outcome: Goal at 95:49
- Followed by (not proof of causation): 0 final-third recoveries, Territory 100.0%, 1 shots
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [2995, 3005], "observed_players": 6.9, "inferred_players": 3.1, "high_confidence_inferred": 2.3, "outfield_players": 13, "metrics": {"observed_only": {"def_line_m": 83.5, "width_m": 21.8, "near_ball": 2.9}, "reconstructed": {"def_line_m": 73.1, "width_m": 21.8, "near_ball":`

### Tactical & formation changes (every change, both teams)

**Team A** (6 changes)
- 49' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) +10.7m, width +4.4m, possession +7.7%, territory +30.9%, press index -26.4. Followed by 0 shot(s) for (0 xG) and 1 against (0.12 xG), 1 against. _(ground truth positions)_
- 59' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) +16.1m, width +7.4m, possession +9.2%, territory +7.4%. Followed by 1 shot(s) for (0.04 xG) and 0 against (0 xG). _(ground truth positions)_
- 74' [STRUCTURE] Structural change in how the team is positioned. Width -11.5m, compactness -3.9m, possession -5.6%, territory +17.6%, press index +22.7. Followed by 1 shot(s) for (0.01 xG) and 1 against (0.04 xG). Preceded by opponent substitution 26': P. Klement on for M. Wunderlich. _(ground truth positions)_
- 86' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +9.6m, width -4.7m, possession +11.6%, territory +31.0%, press index +15.5. Followed by 3 shot(s) for (0.15 xG) and 0 against (0 xG). _(ground truth positions)_
- 95' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -5.7m, width -4.3m, possession +16.0%, territory -30.6%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by opponent card 48'. _(ground truth positions)_
- 96' [FORMATION] Shape read as 4-1-3-2 -> 3-4-3. Jorrit Hendrix defensive midfield -> defence; Dawid Kownacki attack -> midfield; Matthias Zimmermann defence -> midfield; Shinta Appelkamp attacking midfield -> midfield. Defensive line (out of possession) -21.0m, possession -43.3%, territory -52.0%, press index -32.2. Followed by 0 shot(s) for (0 xG) and 1 against (0.77 xG), 1 against. Preceded by opponent card 48'. _(ground truth positions)_

**Team B** (9 changes)
- 49' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -15.1m, possession -7.7%, territory -30.9%, press index -23.3. Followed by 1 shot(s) for (0.12 xG) and 0 against (0 xG), 1 goal(s) for. _(ground truth positions)_
- 55' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +12.1m, possession +9.4%, territory +29.6%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 57' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -9.3m, width +4.2m, possession -9.4%, territory -26.4%, press index -9.1. Followed by 0 shot(s) for (0 xG) and 1 against (0.04 xG). _(ground truth positions)_
- 73' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) -5.3m, width -7.5m, territory -6.4%. Followed by 2 shot(s) for (0.07 xG) and 1 against (0.01 xG). Preceded by substitution 26': P. Klement on for M. Wunderlich. _(ground truth positions)_
- 84' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) -4.0m, width -4.6m. Followed by 0 shot(s) for (0 xG) and 1 against (0.03 xG). Preceded by substitution 36': L. Lobinger on for Terrence Boyd. _(ground truth positions)_
- 87' [TACTIC] From balanced to low block. Defensive line (out of possession) -19.4m, possession +13.3%, territory -34.3%, press index -10.2. Followed by 0 shot(s) for (0 xG) and 3 against (0.15 xG). Preceded by opponent substitution 41': M. Mansfeld on for R. Hennings; substitution 42': Daniel Hanslik on for Kenny Prince Redondo. _(ground truth positions)_
- 95' [TACTIC] From low block to balanced. Defensive line (out of possession) +9.8m, width -4.1m, possession -16.0%, territory +30.6%, press index +15.9. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by card 48'. _(ground truth positions)_
- 96' [FORMATION] Shape read as 4-1-4-1 -> 3-4-2-1. K. Kraus defensive midfield -> defence; B. Tomiak defence -> defensive midfield; Aaron Opoku attacking midfield -> defensive midfield; R. Bormuth defence -> defensive midfield. Defensive line (out of possession) +3.0m, possession +29.0%, territory +30.0%. Followed by 1 shot(s) for (0.77 xG) and 0 against (0 xG), 1 goal(s) for. Preceded by card 48'. _(ground truth positions)_
- 97' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +11.2m, width -5.7m, possession +21.5%, territory +33.0%, press index -16.6. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by goal scored 51'. _(ground truth positions)_

### Does the analysis survive reconstruction? (truth vs camera reconstruction)

Formation agreement per team-minute: 80%. Mean absolute difference per minute: defensive line 2.41 m, width 4.4 m, compactness 1.61 m, press index 3.41. Position-based moments: truth 3, reconstruction 2, recalled within ±120 s 0, reconstruction-only 2.

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Positions are ground truth; the camera view is simulated.
