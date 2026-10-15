# Real tracking · DFL open data: dfl-J03WN1-r13

## h1

**Data basis:** In-stadium tracking of all players (ground truth); official event feed. Positions: `ground_truth`. Events: `recorded_event (official DFL feed)`.

Hidden-player model: `r13_mlp_aug_ens3`.

| | A | B |
|---|---:|---:|
| Goals | 2 | 0 |
| Shots | 7 | 7 |
| xG (engine model) | 1.29 | 0.3 |
| Passes | 243 | 185 |
| Pass accuracy | 0.819 | 0.714 |
| Possession (s) | 1401 | 1371 |
| Press index (heuristic) | 30.1 | 17.8 |
| Last formation (heuristic) | 4-1-4-1 | 4-2-3 |

**Formations over time (heuristic):** A: 0' 4-1-4-1; B: 0' 4-1-3-2, 5' 4-2-3

**Counterattacks detected (experimental):** 2

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
| 4' | TACTICAL_TURNING_POINT | A | VfL Bochum 1848 take territorial control | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 8' | RED_CARD_IMPACT | B | Red card: A. Adli (Bayer 04 Leverkusen) | 0.9 | 1.0 | recorded card; after-effects heuristic |
| 19' | GOAL_BUILD_UP | A | Goal VfL Bochum 1848 - P. Förster | 0.95 | 1.0 | recorded goal; build-up sequence from events |
| 21' | TACTICAL_TURNING_POINT | B | Bayer 04 Leverkusen take territorial control | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 30' | TACTICAL_TURNING_POINT | A | VfL Bochum 1848 take territorial control | 0.88 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 34' | GOAL_BUILD_UP | A | Goal VfL Bochum 1848 - T. Asano | 0.95 | 1.0 | recorded goal; build-up sequence from events |

### Evidence chains (top 3 by importance)

**19' Goal VfL Bochum 1848 - P. Förster** (GOAL_BUILD_UP). Chain: EVENT: Spell of possession → CONSEQUENCE: 3 passes, 10.2s → OUTCOME: Goal (0.844 xG)

- [recorded_event] Build-up: 3 passes in 10.2s
- [recorded_event] Shot: P. Förster shot from 7.9m
- [recorded_event] Outcome: Goal - 1-0
- Followed by (not proof of causation): Score 1-0
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [1078, 1088], "observed_players": 1.8, "inferred_players": 8.2, "high_confidence_inferred": 3.8, "outfield_players": 10, "metrics": {"observed_only": {"def_line_m": 95.7, "width_m": 20.6, "near_ball": 1.8}, "reconstructed": {"def_line_m": 67.5, "width_m": 32.6, "near_ball":`

**8' Red card: A. Adli (Bayer 04 Leverkusen)** (RED_CARD_IMPACT). Chain: EVENT: Red card → STATE_CHANGE: 4-1-3-2 -> 4-2-3 → TACTICAL_CHANGE: VfL Bochum 1848 push up → CONSEQUENCE: Territory 62.6% → OUTCOME: Goal

- [ground_truth positions] Event: A. Adli sent off - violentConduct
- [ground_truth positions] Structural change: Bayer 04 Leverkusen 4-1-3-2 -> 4-2-3, line 37.6m -> 34.3m
- [ground_truth positions] Control: VfL Bochum 1848 possession 52.4% -> 44.2%, territory 62.1% -> 62.6%
- [ground_truth positions] Pressure: VfL Bochum 1848 shots 0 -> 2 (0.88 xG) in comparable windows
- [recorded_event] Outcome: VfL Bochum 1848 scored at 18:03
- Followed by (not proof of causation): Bayer 04 Leverkusen dropped to 4-2-3, VfL Bochum 1848 territory 62.6%
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [420, 430], "observed_players": 5.9, "inferred_players": 3.5, "high_confidence_inferred": 3.5, "outfield_players": 10, "metrics": {"observed_only": {"def_line_m": 43.2, "width_m": 20.2, "near_ball": 6.2}, "reconstructed": {"def_line_m": 41.0, "width_m": 20.7, "near_ball": 6`

**34' Goal VfL Bochum 1848 - T. Asano** (GOAL_BUILD_UP). Chain: EVENT: Spell of possession → CONSEQUENCE: 6 passes, 45.2s → OUTCOME: Goal (0.154 xG)

- [recorded_event] Build-up: 6 passes in 45.2s
- [recorded_event] Shot: T. Asano shot from 11.3m
- [recorded_event] Outcome: Goal - 2-0
- Followed by (not proof of causation): Score 2-0
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [2012, 2022], "observed_players": 0.0, "inferred_players": 10.0, "high_confidence_inferred": 0.0, "outfield_players": 10, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 75.3, "width_m": 19.9, "near_ball": 2.2}, "truth": {"def_line_m": 83.5, "width_m": 39`

### Tactical & formation changes (every change, both teams)

**Team A** (9 changes)
- 6' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +20.7m, territory +37.4%, press index +20.0. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 12' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -8.0m, territory -37.0%, press index -21.5. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 18' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +9.8m, possession -23.3%, territory +39.7%. Followed by 2 shot(s) for (0.89 xG) and 0 against (0 xG), 1 goal(s) for. _(ground truth positions)_
- 21' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -21.0m, width +4.0m, possession +27.9%, territory -53.7%. Followed by 0 shot(s) for (0 xG) and 1 against (0.04 xG). Preceded by goal scored 19'. _(ground truth positions)_
- 30' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) +10.4m, territory +28.0%. Followed by 3 shot(s) for (0.35 xG) and 0 against (0 xG), 1 goal(s) for. Preceded by card 28'. _(ground truth positions)_
- 35' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +10.6m, width -7.6m, compactness -3.4m, possession -5.3%, territory +10.1%. Followed by 0 shot(s) for (0 xG) and 1 against (0.02 xG). Preceded by goal scored 34'. _(ground truth positions)_
- 37' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -22.9m, width +4.6m, territory -19.5%, press index -12.4. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by goal scored 34'. _(ground truth positions)_
- 40' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +15.2m, width +9.6m, possession +36.0%, territory +15.5%, press index +50.4. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 42' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -8.3m, width -3.7m, possession -26.6%, press index -41.8. Followed by 2 shot(s) for (0.06 xG) and 0 against (0 xG). _(ground truth positions)_

**Team B** (8 changes)
- 6' [FORMATION] Shape read as 4-1-3-2 -> 4-2-3. N. Amiri attacking midfield -> midfield; Exequiel Palacios defensive midfield -> midfield; M. Diaby attacking midfield -> attack; Florian Wirtz attacking midfield -> attack. Defensive line (out of possession) -8.4m, territory -37.6%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 19' [TACTIC] From balanced to low block. Defensive line (out of possession) -16.2m, width +4.0m, possession +23.2%, territory -43.1%, press index +8.4. Followed by 0 shot(s) for (0 xG) and 2 against (0.89 xG), 1 against. Preceded by goal conceded 19'. _(ground truth positions)_
- 21' [TACTIC] From low block to balanced. Defensive line (out of possession) +13.6m, possession -17.5%, territory +36.0%, press index -10.7. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by goal conceded 19'. _(ground truth positions)_
- 23' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +16.0m, width -5.2m, possession -23.5%, territory +39.8%, press index +11.9. Followed by 1 shot(s) for (0.04 xG) and 0 against (0 xG). Preceded by opponent card 21'. _(ground truth positions)_
- 26' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -6.5m, possession +23.5%, territory -29.8%. Followed by 1 shot(s) for (0.06 xG) and 0 against (0 xG). _(ground truth positions)_
- 30' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) -13.1m, width -4.3m, territory -20.9%, press index -9.6. Followed by 0 shot(s) for (0 xG) and 3 against (0.35 xG), 1 against. Preceded by opponent card 28'. _(ground truth positions)_
- 35' [TACTIC] From balanced to low block. Defensive line (out of possession) -10.1m, width -8.2m, compactness -4.1m, possession +5.3%, territory -10.1%. Followed by 1 shot(s) for (0.02 xG) and 0 against (0 xG). Preceded by goal conceded 34'. _(ground truth positions)_
- 37' [TACTIC] From low block to balanced. Defensive line (out of possession) +13.6m, width +8.1m, compactness +3.2m, possession -14.9%, territory +13.3%, press index +15.3. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by goal conceded 34'. _(ground truth positions)_

### Does the analysis survive reconstruction? (truth vs camera reconstruction)

Formation agreement per team-minute: 66%. Mean absolute difference per minute: defensive line 1.59 m, width 3.11 m, compactness 0.97 m, press index 4.12. Position-based moments: truth 3, reconstruction 3, recalled within ±120 s 3, reconstruction-only 0.

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Positions are ground truth; the camera view is simulated.

## h2

**Data basis:** In-stadium tracking of all players (ground truth); official event feed. Positions: `ground_truth`. Events: `recorded_event (official DFL feed)`.

Hidden-player model: `r13_mlp_aug_ens3`.

| | A | B |
|---|---:|---:|
| Goals | 1 | 0 |
| Shots | 3 | 4 |
| xG (engine model) | 0.5 | 0.47 |
| Passes | 116 | 184 |
| Pass accuracy | 0.621 | 0.658 |
| Possession (s) | 1357 | 1534 |
| Press index (heuristic) | 27.8 | 26.6 |
| Last formation (heuristic) | 4-2-3-1 | 4-2-3 |

**Formations over time (heuristic):** A: 0' 4-2-3-1; B: 0' 4-2-3, 35' 4-4-1

**Counterattacks detected (experimental):** 2

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
| 51' | TACTICAL_TURNING_POINT | B | Bayer 04 Leverkusen take territorial control | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 56' | SUBSTITUTION_IMPACT | A | K. Schlotterbeck on for D. Heintz | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 56' | SUBSTITUTION_IMPACT | A | P. Osterhage on for P. Förster | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 67' | TACTICAL_TURNING_POINT | B | Bayer 04 Leverkusen drop into a low block | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 78' | SUBSTITUTION_IMPACT | B | A. Hložek on for N. Amiri | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 81' | FORMATION_SHIFT | B | Bayer 04 Leverkusen switch to 4-4-1 | 0.92 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups; 57% minute agreement truth vs reconstruction) |
| 81' | SUBSTITUTION_IMPACT | A | M. Broschinski on for P. Hofmann | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 81' | SUBSTITUTION_IMPACT | A | S. Zoller on for T. Asano | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 86' | GOAL_BUILD_UP | A | Goal VfL Bochum 1848 - K. Stöger | 0.95 | 1.0 | recorded goal; build-up sequence from events |
| 89' | SUBSTITUTION_IMPACT | A | Vasileios Lampropoulos on for K. Stöger | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 90' | TACTICAL_TURNING_POINT | B | Bayer 04 Leverkusen drop into a low block | 0.83 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 92' | SUBSTITUTION_IMPACT | B | Noah Mbamba on for M. Diaby | 0.7 | 1.0 | recorded substitution; after-effects heuristic |

### Evidence chains (top 3 by importance)

**86' Goal VfL Bochum 1848 - K. Stöger** (GOAL_BUILD_UP). Chain: EVENT: Tackle by M. Broschinski → CONSEQUENCE: 2 passes, 2.7s → OUTCOME: Goal (0.19 xG)

- [recorded_event] Recovery: M. Broschinski won the ball (tackle) 69m from own goal
- [recorded_event] Build-up: 2 passes in 2.7s
- [recorded_event] Shot: K. Stöger shot from 18.6m
- [recorded_event] Outcome: Goal - 3-0
- Followed by (not proof of causation): Score 3-0
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [2413, 2423], "observed_players": 0.0, "inferred_players": 10.0, "high_confidence_inferred": 0.0, "outfield_players": 15, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 58.7, "width_m": 22.0, "near_ball": 4.0}, "truth": {"def_line_m": 51.9, "width_m": 34`

**81' Bayer 04 Leverkusen switch to 4-4-1** (FORMATION_SHIFT). Chain: TACTICAL_CHANGE: 4-2-3 -> 4-4-1 → STATE_CHANGE: Right-side press 56.0% -> 31.6% → CONSEQUENCE: 0 final-third recoveries

- [heuristic over ground_truth positions] Tactical trigger: Shape changed 4-2-3 -> 4-4-1
- [ground_truth positions] Structural change: Defensive line +23.14m, width +1.95m
- [ground_truth positions] Pressing: Press intensity 4.2 -> 29.6; right-side share of pressures 56.0% -> 31.6%
- [recorded_event] Recovery: Final-third recoveries 0 -> 0
- [ground_truth positions] Territory: Territory 25.5% -> 51.7%
- Followed by (not proof of causation): 0 final-third recoveries, Territory 51.7%, 0 shots
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [2095, 2105], "observed_players": 0.0, "inferred_players": 9.0, "high_confidence_inferred": 0.8, "outfield_players": 11, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 65.8, "width_m": 30.0, "near_ball": 4.8}, "truth": {"def_line_m": 50.0, "width_m": 35.`

**81' S. Zoller on for T. Asano** (SUBSTITUTION_IMPACT). Chain: EVENT: Substitution → STATE_CHANGE: Fresh legs on the RW → OUTCOME: Assist

- [recorded_event] Change: S. Zoller replaces T. Asano (RW)
- [ground_truth positions] Player activity: 5 pressures (0 on the right), 3 recoveries in first 15 min
- [ground_truth positions] Team effect: Press intensity 27.0 -> 16.8
- [recorded_event] Outcome: Assist for the goal at 85:18
- Followed by (not proof of causation): 5 pressures from S. Zoller, Assist
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [2112, 2122], "observed_players": 0.5, "inferred_players": 9.5, "high_confidence_inferred": 5.3, "outfield_players": 15, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 17.6, "width_m": 25.8, "near_ball": 5.1}, "truth": {"def_line_m": 21.7, "width_m": 53.`

### Tactical & formation changes (every change, both teams)

**Team A** (8 changes)
- 53' [TACTIC] From balanced to low block. Defensive line (out of possession) -17.4m, width +3.5m, territory -23.6%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 57' [TACTIC] From low block to balanced. Defensive line (out of possession) +20.1m, possession +18.5%, territory +31.2%, press index +17.3. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by substitution 11': K. Schlotterbeck on for D. Heintz; substitution 11': P. Osterhage on for P. Förster. _(ground truth positions)_
- 61' [TACTIC] From balanced to low block. Defensive line (out of possession) -19.6m, width -13.2m, compactness -4.6m, possession -51.3%, territory -15.2%, press index -30.3. Followed by 0 shot(s) for (0 xG) and 2 against (0.33 xG). _(ground truth positions)_
- 63' [TACTIC] From low block to balanced. Defensive line (out of possession) +7.4m, width +4.9m, possession +38.8%, territory +11.6%. Followed by 0 shot(s) for (0 xG) and 1 against (0.1 xG). _(ground truth positions)_
- 69' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +28.1m, width +6.9m, compactness +3.7m, possession +28.3%, territory +50.6%, press index +16.7. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by opponent card 22'; opponent card 24'. _(ground truth positions)_
- 72' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -22.1m, possession -34.5%, territory -41.4%, press index -10.9. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by opponent card 24'; opponent card 26'. _(ground truth positions)_
- 77' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +18.5m, possession +22.5%, territory +41.7%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 80' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -19.1m, possession -39.1%, territory -39.1%, press index -13.2. Followed by 1 shot(s) for (0.24 xG) and 0 against (0 xG). Preceded by opponent substitution 33': A. Hložek on for N. Amiri. _(ground truth positions)_

**Team B** (12 changes)
- 47' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -10.9m, possession +19.2%, territory -34.2%, press index -10.1. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 51' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +19.8m, width +4.8m, possession +20.0%, territory +26.8%, press index +26.3. Followed by 1 shot(s) for (0.04 xG) and 0 against (0 xG). _(ground truth positions)_
- 58' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -26.0m, territory -42.7%, press index -9.9. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by opponent substitution 11': K. Schlotterbeck on for D. Heintz; opponent substitution 11': P. Osterhage on for P. Förster. _(ground truth positions)_
- 61' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +31.0m, possession +32.0%, territory +23.6%, press index +26.6. Followed by 2 shot(s) for (0.33 xG) and 0 against (0 xG). _(ground truth positions)_
- 64' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -23.7m, possession -34.2%, territory -21.3%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by card 18'. _(ground truth positions)_
- 66' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +8.3m, territory +14.3%. Followed by 1 shot(s) for (0.1 xG) and 0 against (0 xG). Preceded by card 18'. _(ground truth positions)_
- 69' [TACTIC] From territorial control to low block. Defensive line (out of possession) -33.1m, width -4.5m, possession -32.7%, territory -60.8%, press index -46.5. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by card 22'; card 24'. _(ground truth positions)_
- 71' [TACTIC] From low block to balanced. Defensive line (out of possession) +21.5m, width +8.3m, possession +38.8%, territory +43.1%, press index +24.4. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by card 24'. _(ground truth positions)_
- 77' [TACTIC] From balanced to low block. Defensive line (out of possession) -24.4m, width -5.2m, possession -15.6%, territory -36.6%, press index -24.3. Followed by 0 shot(s) for (0 xG) and 1 against (0.24 xG). _(ground truth positions)_
- 81' [FORMATION] Shape read as 4-2-3 -> 4-4-1. M. Diaby attack -> midfield; Jeremie Frimpong attack -> midfield. Defensive line (out of possession) +17.5m, width +3.0m, possession +39.8%, territory +49.7%, press index +27.3. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by substitution 33': A. Hložek on for N. Amiri; opponent substitution 36': M. Broschinski on for P. Hofmann; opponent substitution 36': S. Zoller on for T. Asano. _(ground truth positions)_
- 83' [TACTIC] From low block to balanced. Defensive line (out of possession) +4.2m, possession -8.6%, territory -32.1%. Followed by 0 shot(s) for (0 xG) and 1 against (0.19 xG), 1 against. Preceded by opponent substitution 36': M. Broschinski on for P. Hofmann; opponent substitution 36': S. Zoller on for T. Asano. _(ground truth positions)_
- 89' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +5.5m, territory +33.6%, press index +26.8. Followed by 0 shot(s) for (0 xG) and 1 against (0.07 xG). Preceded by goal conceded 41'; opponent card 44'; card 44'. _(ground truth positions)_

### Does the analysis survive reconstruction? (truth vs camera reconstruction)

Formation agreement per team-minute: 14%. Mean absolute difference per minute: defensive line 1.74 m, width 2.98 m, compactness 1.19 m, press index 5.93. Position-based moments: truth 4, reconstruction 4, recalled within ±120 s 3, reconstruction-only 1.

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Positions are ground truth; the camera view is simulated.
