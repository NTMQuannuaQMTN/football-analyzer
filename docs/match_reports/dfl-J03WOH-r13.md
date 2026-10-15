# Real tracking · DFL open data: dfl-J03WOH-r13

## h1

**Data basis:** In-stadium tracking of all players (ground truth); official event feed. Positions: `ground_truth`. Events: `recorded_event (official DFL feed)`.

Hidden-player model: `r13_mlp_aug_ens3`.

| | A | B |
|---|---:|---:|
| Goals | 0 | 0 |
| Shots | 10 | 8 |
| xG (engine model) | 0.79 | 0.43 |
| Passes | 223 | 168 |
| Pass accuracy | 0.785 | 0.732 |
| Possession (s) | 1461 | 1316 |
| Press index (heuristic) | 24.0 | 18.0 |
| Last formation (heuristic) | 4-1-3-2 | 4-2-3-1 |

**Formations over time (heuristic):** A: 0' 4-1-3-2; B: 0' 4-2-3-1, 30' 4-2-2-2, 45' 3-4-1-2

**Counterattacks detected (experimental):** 0

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
| 21' | TACTICAL_TURNING_POINT | A | Fortuna Düsseldorf take territorial control | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 34' | TACTICAL_TURNING_POINT | A | Fortuna Düsseldorf drop into a low block | 0.84 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 42' | TACTICAL_TURNING_POINT | B | SSV Jahn Regensburg drop into a low block | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |

### Evidence chains (top 3 by importance)

**21' Fortuna Düsseldorf take territorial control** (TACTICAL_TURNING_POINT). Chain: TACTICAL_CHANGE: Dominant → STATE_CHANGE: Line 57.8m -> 36.8m → CONSEQUENCE: 0 counterattacks

- [heuristic over ground_truth positions] Tactical trigger: Block pushed up: defensive line 57.8m -> 36.8m
- [recorded_event] Possession: Possession 66.2% -> 42.2%
- [ground_truth positions] Territory: Territory 89.8% -> 49.7%
- [recorded_event] Outcome: 4 shots (0.26 xG) in the following minutes
- Followed by (not proof of causation): Territory 49.7%, 4 shots (0.26 xG)
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [1195, 1205], "observed_players": 0.0, "inferred_players": 10.0, "high_confidence_inferred": 0.0, "outfield_players": 10, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 36.4, "width_m": 26.3, "near_ball": 0.0}, "truth": {"def_line_m": 39.8, "width_m": 32`

**34' Fortuna Düsseldorf drop into a low block** (TACTICAL_TURNING_POINT). Chain: TACTICAL_CHANGE: Low Block → STATE_CHANGE: Line 40.2m -> 35.8m → CONSEQUENCE: 0 counterattacks

- [heuristic over ground_truth positions] Tactical trigger: Block dropped: defensive line 40.2m -> 35.8m
- [ground_truth positions] Ceded the ball: Possession 44.2% -> 60.6%
- [ground_truth positions] Transitions: 0 counterattacks launched in the next 15 min
- [recorded_event] Outcome: 0 ended in shots
- Followed by (not proof of causation): 0 counterattacks, 2 shots (0.29 xG)
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [1975, 1985], "observed_players": 3.5, "inferred_players": 6.5, "high_confidence_inferred": 4.2, "outfield_players": 10, "metrics": {"observed_only": {"def_line_m": 17.7, "width_m": 18.3, "near_ball": 3.6}, "reconstructed": {"def_line_m": 14.6, "width_m": 30.3, "near_ball":`

**42' SSV Jahn Regensburg drop into a low block** (TACTICAL_TURNING_POINT). Chain: TACTICAL_CHANGE: Low Block → STATE_CHANGE: Line 46.2m -> 21.5m → CONSEQUENCE: 0 counterattacks

- [heuristic over ground_truth positions] Tactical trigger: Block dropped: defensive line 46.2m -> 21.5m
- [ground_truth positions] Ceded the ball: Possession 57.2% -> 23.0%
- [ground_truth positions] Transitions: 0 counterattacks launched in the next 15 min
- [recorded_event] Outcome: 0 ended in shots
- Followed by (not proof of causation): 0 counterattacks, 0 shots (0 xG)
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [2455, 2465], "observed_players": 3.7, "inferred_players": 6.3, "high_confidence_inferred": 2.0, "outfield_players": 10, "metrics": {"observed_only": {"def_line_m": 23.6, "width_m": 23.0, "near_ball": 4.0}, "reconstructed": {"def_line_m": 21.9, "width_m": 20.4, "near_ball":`

### Tactical & formation changes (every change, both teams)

**Team A** (9 changes)
- 5' [TACTIC] From low block to balanced. Defensive line (out of possession) +26.7m, width +19.0m, compactness +8.4m, possession +51.5%, territory +50.2%, press index +24.4. Followed by 1 shot(s) for (0.08 xG) and 1 against (0.15 xG). _(ground truth positions)_
- 9' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +9.4m, width -6.8m, possession -24.0%, territory +19.3%. Followed by 1 shot(s) for (0.03 xG) and 0 against (0 xG). _(ground truth positions)_
- 11' [TACTIC] From territorial control to balanced. Possession +9.7%, territory -23.9%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 16' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +14.1m, possession +16.6%, territory +35.6%, press index +10.1. Followed by 1 shot(s) for (0.06 xG) and 0 against (0 xG). _(ground truth positions)_
- 21' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) -13.2m, width -6.5m, possession -31.5%, territory -10.5%, press index -15.0. Followed by 0 shot(s) for (0 xG) and 1 against (0.07 xG). _(ground truth positions)_
- 24' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -9.7m, possession +10.9%, territory -44.3%. Followed by 2 shot(s) for (0.06 xG) and 0 against (0 xG). _(ground truth positions)_
- 34' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) -11.9m, possession +15.0%, territory -34.0%. Followed by 0 shot(s) for (0 xG) and 1 against (0.04 xG). _(ground truth positions)_
- 42' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) +18.6m, width +6.6m, possession +33.4%, territory +36.6%, press index +18.0. Followed by 2 shot(s) for (0.29 xG) and 0 against (0 xG). _(ground truth positions)_
- 46' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +9.8m, width -9.8m, compactness -3.8m, territory +24.5%, press index +18.5. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_

**Team B** (9 changes)
- 5' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -36.8m, possession -48.8%, territory -55.0%, press index -35.9. Followed by 1 shot(s) for (0.15 xG) and 1 against (0.08 xG). _(ground truth positions)_
- 16' [TACTIC] From balanced to low block. Defensive line (out of possession) -26.2m, possession -16.6%, territory -35.6%, press index -12.4. Followed by 0 shot(s) for (0 xG) and 1 against (0.06 xG). _(ground truth positions)_
- 21' [TACTIC] From low block to balanced. Defensive line (out of possession) +21.5m, width +4.7m, possession +37.4%, territory +20.6%, press index +24.0. Followed by 1 shot(s) for (0.07 xG) and 0 against (0 xG). _(ground truth positions)_
- 26' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +5.7m, possession -31.6%, territory +52.6%, press index -16.5. Followed by 0 shot(s) for (0 xG) and 2 against (0.06 xG). _(ground truth positions)_
- 28' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -15.5m, territory -54.3%, press index -11.0. Followed by 1 shot(s) for (0.03 xG) and 1 against (0.17 xG). _(ground truth positions)_
- 31' [FORMATION] Shape read as 4-2-3-1 -> 4-2-2-2. Andreas Albers attacking midfield -> attack. Defensive line (out of possession) +3.3m, territory +26.3%. Followed by 0 shot(s) for (0 xG) and 2 against (0.11 xG). _(ground truth positions)_
- 36' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +14.6m, width +9.3m, possession +7.0%, territory +27.8%, press index +11.2. Followed by 2 shot(s) for (0.05 xG) and 0 against (0 xG). _(ground truth positions)_
- 40' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -12.8m, width -9.4m, possession -18.4%, territory -33.8%, press index -11.0. Followed by 1 shot(s) for (0.01 xG) and 1 against (0.27 xG). _(ground truth positions)_
- 46' [FORMATION] Shape read as 4-2-2-2 -> 3-4-1-2. M. Thalhammer defensive midfield -> defence; K. Faber defence -> defensive midfield; Andreas Albers attack -> defensive midfield; L. Guwara defence -> defensive midfield. Defensive line (out of possession) -8.8m, width -3.0m, compactness -3.2m, territory -29.9%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_

### Does the analysis survive reconstruction? (truth vs camera reconstruction)

Formation agreement per team-minute: 82%. Mean absolute difference per minute: defensive line 1.95 m, width 3.54 m, compactness 1.51 m, press index 3.0. Position-based moments: truth 3, reconstruction 3, recalled within ±120 s 3, reconstruction-only 0.

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Positions are ground truth; the camera view is simulated.

## h2

**Data basis:** In-stadium tracking of all players (ground truth); official event feed. Positions: `ground_truth`. Events: `recorded_event (official DFL feed)`.

Hidden-player model: `r13_mlp_aug_ens3`.

| | A | B |
|---|---:|---:|
| Goals | 4 | 0 |
| Shots | 5 | 4 |
| xG (engine model) | 1.87 | 0.09 |
| Passes | 195 | 177 |
| Pass accuracy | 0.8 | 0.757 |
| Possession (s) | 1309 | 1403 |
| Press index (heuristic) | 24.9 | 21.1 |
| Last formation (heuristic) | 4-1-3-2 | 4-1-3-2 |

**Formations over time (heuristic):** A: 0' 4-1-3-2, 45' 4-1-4-1; B: 0' 4-2-3-1, 20' 4-1-3-2, 45' 4-4-2

**Counterattacks detected (experimental):** 3

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
| 59' | TACTICAL_TURNING_POINT | A | Fortuna Düsseldorf take territorial control | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 59' | GOAL_BUILD_UP | A | Goal Fortuna Düsseldorf - D. Ginczek | 0.95 | 1.0 | recorded goal; build-up sequence from events |
| 65' | GOAL_BUILD_UP | A | Goal Fortuna Düsseldorf - Dawid Kownacki | 0.95 | 1.0 | recorded goal; build-up sequence from events |
| 65' | SUBSTITUTION_IMPACT | B | Scott Kennedy on for L. Guwara | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 66' | SUBSTITUTION_IMPACT | A | C. Klarer on for A. Hoffmann | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 66' | SUBSTITUTION_IMPACT | A | Shinta Appelkamp on for Kristoffer Peterson | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 73' | SUBSTITUTION_IMPACT | B | P. Owusu on for A. Yildirim | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 74' | SUBSTITUTION_IMPACT | B | C. Makridis on for N. Shipnoski | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 78' | GOAL_BUILD_UP | A | Goal Fortuna Düsseldorf - F. Klaus | 0.95 | 1.0 | recorded goal; build-up sequence from events |
| 84' | SUBSTITUTION_IMPACT | A | E. Iyoha on for Dawid Kownacki | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 84' | SUBSTITUTION_IMPACT | A | Kwadwo Baah on for F. Klaus | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 84' | SUBSTITUTION_IMPACT | A | R. Hennings on for D. Ginczek | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 85' | GOAL_BUILD_UP | A | Goal Fortuna Düsseldorf - Shinta Appelkamp | 0.95 | 1.0 | recorded goal; build-up sequence from events |
| 85' | SUBSTITUTION_IMPACT | B | C. Viet on for J. Mees | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 86' | SUBSTITUTION_IMPACT | B | K. Caliskaner on for Andreas Albers | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 89' | TACTICAL_TURNING_POINT | B | SSV Jahn Regensburg drop into a low block | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |

### Evidence chains (top 3 by importance)

**59' Goal Fortuna Düsseldorf - D. Ginczek** (GOAL_BUILD_UP). Chain: EVENT: Interception by Jorrit Hendrix → CONSEQUENCE: 3 passes, 10.0s → OUTCOME: Goal (0.084 xG)

- [recorded_event] Recovery: Jorrit Hendrix won the ball (interception) 42m from own goal
- [heuristic over ground_truth positions] Transition: Ball moved 32.3m in 5s with 2 pass(es)
- [ground_truth positions] Exposed defence: 5 SSV Jahn Regensburg outfield players caught upfield of the ball
- [recorded_event] Shot: D. Ginczek shot from 13.6m
- [recorded_event] Outcome: Goal - 1-0
- Followed by (not proof of causation): Score 1-0
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [832, 842], "observed_players": 2.2, "inferred_players": 7.8, "high_confidence_inferred": 4.5, "outfield_players": 15, "metrics": {"observed_only": {"def_line_m": 85.0, "width_m": 28.2, "near_ball": 2.2}, "reconstructed": {"def_line_m": 60.0, "width_m": 36.0, "near_ball": 2`

**59' Fortuna Düsseldorf take territorial control** (TACTICAL_TURNING_POINT). Chain: TACTICAL_CHANGE: Dominant → STATE_CHANGE: Line 35.7m -> 49.3m → CONSEQUENCE: 1 counterattacks → OUTCOME: Goal

- [heuristic over ground_truth positions] Tactical trigger: Block pushed up: defensive line 35.7m -> 49.3m
- [recorded_event] Possession: Possession 64.0% -> 59.9%
- [ground_truth positions] Territory: Territory 31.5% -> 76.2%
- [recorded_event] Outcome: 3 shots (0.86 xG) in the following minutes
- Followed by (not proof of causation): Territory 76.2%, 3 shots (0.86 xG)
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [775, 785], "observed_players": 6.2, "inferred_players": 3.8, "high_confidence_inferred": 3.5, "outfield_players": 15, "metrics": {"observed_only": {"def_line_m": 63.8, "width_m": 41.9, "near_ball": 2.8}, "reconstructed": {"def_line_m": 48.4, "width_m": 48.5, "near_ball": 3`

**65' Goal Fortuna Düsseldorf - Dawid Kownacki** (GOAL_BUILD_UP). Chain: EVENT: Spell of possession → CONSEQUENCE: 4 passes, 167.3s → OUTCOME: Goal (0.767 xG)

- [recorded_event] Build-up: 4 passes in 167.3s
- [recorded_event] Shot: Dawid Kownacki shot from 11.0m
- [recorded_event] Outcome: Goal - 2-0
- Followed by (not proof of causation): Score 2-0
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [1152, 1162], "observed_players": 4.3, "inferred_players": 5.7, "high_confidence_inferred": 2.9, "outfield_players": 15, "metrics": {"observed_only": {"def_line_m": 89.7, "width_m": 19.5, "near_ball": 1.6}, "reconstructed": {"def_line_m": 74.0, "width_m": 33.9, "near_ball":`

### Tactical & formation changes (every change, both teams)

**Team A** (8 changes)
- 47' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -8.3m, possession +15.7%, territory -34.0%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 50' [STRUCTURE] Structural change in how the team is positioned. Width +5.7m, possession -6.9%, territory -11.4%, press index -27.1. Followed by 0 shot(s) for (0 xG) and 1 against (0.02 xG). _(ground truth positions)_
- 60' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +10.0m, width -11.3m, possession -16.4%, territory +49.4%, press index -25.5. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by goal scored 14'. _(ground truth positions)_
- 67' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -5.3m, width +9.3m, territory -27.4%, press index +21.3. Followed by 1 shot(s) for (0.01 xG) and 0 against (0 xG). Preceded by opponent card 19'; goal scored 20'; opponent substitution 20': Scott Kennedy on for L. Guwara; substitution 21': C. Klarer on for A. Hoffmann; substitution 21': Shinta Appelkamp on for Kristoffer Peterson. _(ground truth positions)_
- 74' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) -10.6m, width -9.9m, possession -45.0%, territory -21.6%, press index -31.7. Followed by 1 shot(s) for (0.15 xG) and 0 against (0 xG), 1 goal(s) for. Preceded by opponent substitution 28': P. Owusu on for A. Yildirim; opponent substitution 29': C. Makridis on for N. Shipnoski. _(ground truth positions)_
- 79' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +16.9m, possession -6.4%, territory +52.9%, press index +16.2. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by goal scored 33'. _(ground truth positions)_
- 81' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -16.3m, width +5.9m, possession +17.8%, territory -43.9%, press index -13.2. Followed by 1 shot(s) for (0.85 xG) and 1 against (0.02 xG), 1 goal(s) for. Preceded by goal scored 33'. _(ground truth positions)_
- 91' [FORMATION] Shape read as 4-1-3-2 -> 4-1-4-1, lasting only a minute. Kwadwo Baah attacking midfield -> defence; Shinta Appelkamp attacking midfield -> defence; C. Klarer defence -> defensive midfield; Jorrit Hendrix defensive midfield -> attacking midfield. Defensive line (out of possession) +26.0m, width +6.1m, territory +41.6%, press index +12.2. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by opponent card 44'. _(ground truth positions)_

**Team B** (12 changes)
- 50' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) +5.7m, width +4.0m, territory +16.3%, press index +8.0. Followed by 1 shot(s) for (0.02 xG) and 0 against (0 xG). _(ground truth positions)_
- 54' [TACTIC] From balanced to territorial control. Possession -18.7%, territory +20.0%. Followed by 1 shot(s) for (0.04 xG) and 0 against (0 xG). _(ground truth positions)_
- 58' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -13.5m, width -4.3m, possession +11.2%, territory -47.8%. Followed by 0 shot(s) for (0 xG) and 1 against (0.08 xG), 1 against. _(ground truth positions)_
- 64' [TACTIC] From balanced to low block. Defensive line (out of possession) -15.1m, width -8.7m, compactness -4.3m, possession -42.2%, territory -11.3%, press index -18.5. Followed by 0 shot(s) for (0 xG) and 1 against (0.77 xG), 1 against. _(ground truth positions)_
- 66' [FORMATION] Shape read as 4-2-3-1 -> 4-1-3-2. M. Thalhammer defensive midfield -> attacking midfield; Andreas Albers attacking midfield -> attack. Defensive line (out of possession) +18.4m, width +9.0m, possession +21.6%, territory +27.9%. Followed by 1 shot(s) for (0.02 xG) and 1 against (0.01 xG). Preceded by card 19'; goal conceded 20'; substitution 20': Scott Kennedy on for L. Guwara; opponent substitution 21': C. Klarer on for A. Hoffmann; opponent substitution 21': Shinta Appelkamp on for Kristoffer Peterson. _(ground truth positions)_
- 66' [TACTIC] From low block to balanced. Defensive line (out of possession) +18.4m, width +9.0m, possession +21.6%, territory +27.9%. Followed by 1 shot(s) for (0.02 xG) and 1 against (0.01 xG). Preceded by card 19'; goal conceded 20'; substitution 20': Scott Kennedy on for L. Guwara; opponent substitution 21': C. Klarer on for A. Hoffmann; opponent substitution 21': Shinta Appelkamp on for Kristoffer Peterson. _(ground truth positions)_
- 72' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) +6.5m, possession +21.3%, territory +14.0%, press index +12.3. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by card 24'. _(ground truth positions)_
- 76' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +19.0m, possession +30.0%, territory +24.0%, press index +19.9. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by substitution 28': P. Owusu on for A. Yildirim; substitution 29': C. Makridis on for N. Shipnoski; opponent card 29'. _(ground truth positions)_
- 78' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -26.5m, possession -13.1%, territory -49.6%, press index -13.7. Followed by 1 shot(s) for (0.02 xG) and 1 against (0.15 xG), 1 against. Preceded by goal conceded 33'. _(ground truth positions)_
- 83' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +14.5m, possession -8.9%, territory +43.1%, press index +14.8. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 85' [TACTIC] From territorial control to balanced. Territory -12.3%. Followed by 0 shot(s) for (0 xG) and 1 against (0.85 xG), 1 against. Preceded by opponent substitution 39': E. Iyoha on for Dawid Kownacki; opponent substitution 39': Kwadwo Baah on for F. Klaus; opponent substitution 39': R. Hennings on for D. Ginczek; goal conceded 40'. _(ground truth positions)_
- 91' [FORMATION] Shape read as 4-1-3-2 -> 4-4-2, lasting only a minute. P. Owusu attack -> defence; K. Caliskaner attack -> defence; Benedikt Gimber defensive midfield -> midfield; M. Thalhammer attacking midfield -> midfield. Defensive line (out of possession) -20.9m, width -10.6m, compactness -5.1m, territory -41.6%, press index -22.9. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by card 44'. _(ground truth positions)_

### Does the analysis survive reconstruction? (truth vs camera reconstruction)

Formation agreement per team-minute: 44%. Mean absolute difference per minute: defensive line 2.02 m, width 4.3 m, compactness 1.54 m, press index 3.71. Position-based moments: truth 2, reconstruction 2, recalled within ±120 s 1, reconstruction-only 1.

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Positions are ground truth; the camera view is simulated.
