# Real tracking · DFL open data: dfl-J03WOY-r13

## h1

**Data basis:** In-stadium tracking of all players (ground truth); official event feed. Positions: `ground_truth`. Events: `recorded_event (official DFL feed)`.

Hidden-player model: `r13_mlp_aug_ens3`.

| | A | B |
|---|---:|---:|
| Goals | 2 | 0 |
| Shots | 4 | 3 |
| xG (engine model) | 0.87 | 0.1 |
| Passes | 218 | 206 |
| Pass accuracy | 0.784 | 0.733 |
| Possession (s) | 1380 | 1385 |
| Press index (heuristic) | 23.2 | 26.8 |
| Last formation (heuristic) | 4-2-3-1 | 3-5-2 |

**Formations over time (heuristic):** A: 0' 4-2-3-1; B: 0' 3-5-2

**Counterattacks detected (experimental):** 3

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
| 10' | GOAL_BUILD_UP | A | Goal Fortuna Düsseldorf - Dawid Kownacki | 0.95 | 1.0 | recorded goal; build-up sequence from events |
| 27' | GOAL_BUILD_UP | A | Goal Fortuna Düsseldorf - M. Sobottka | 0.95 | 1.0 | recorded goal; build-up sequence from events |
| 31' | TACTICAL_TURNING_POINT | A | Fortuna Düsseldorf drop into a low block | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 38' | TACTICAL_TURNING_POINT | B | F.C. Hansa Rostock drop into a low block | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 38' | SUBSTITUTION_IMPACT | B | D. Lee on for S. Thill | 0.7 | 1.0 | recorded substitution; after-effects heuristic |

### Evidence chains (top 3 by importance)

**10' Goal Fortuna Düsseldorf - Dawid Kownacki** (GOAL_BUILD_UP). Chain: EVENT: Spell of possession → CONSEQUENCE: 9 passes, 45.6s → OUTCOME: Goal (0.223 xG)

- [recorded_event] Build-up: 9 passes in 45.6s
- [recorded_event] Shot: Dawid Kownacki shot from 18.6m
- [recorded_event] Outcome: Goal - 1-0
- Followed by (not proof of causation): Score 1-0
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [574, 584], "observed_players": 2.6, "inferred_players": 7.4, "high_confidence_inferred": 5.6, "outfield_players": 10, "metrics": {"observed_only": {"def_line_m": 64.7, "width_m": 19.2, "near_ball": 1.7}, "reconstructed": {"def_line_m": 59.5, "width_m": 47.8, "near_ball": 1`

**27' Goal Fortuna Düsseldorf - M. Sobottka** (GOAL_BUILD_UP). Chain: EVENT: Tackle by M. Sobottka → CONSEQUENCE: 1 passes, 4.1s → OUTCOME: Goal (0.449 xG)

- [recorded_event] Recovery: M. Sobottka won the ball (tackle) 78m from own goal
- [recorded_event] Build-up: 1 passes in 4.1s
- [recorded_event] Shot: M. Sobottka shot from 10.9m
- [recorded_event] Outcome: Goal - 2-0
- Followed by (not proof of causation): Score 2-0
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [1597, 1607], "observed_players": 0.0, "inferred_players": 10.0, "high_confidence_inferred": 0.0, "outfield_players": 10, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 56.4, "width_m": 35.8, "near_ball": 2.6}, "truth": {"def_line_m": 57.6, "width_m": 28`

**31' Fortuna Düsseldorf drop into a low block** (TACTICAL_TURNING_POINT). Chain: TACTICAL_CHANGE: Low Block → STATE_CHANGE: Line 48.6m -> 35.9m → CONSEQUENCE: 0 counterattacks

- [heuristic over ground_truth positions] Tactical trigger: Block dropped: defensive line 48.6m -> 35.9m
- [ground_truth positions] Ceded the ball: Possession 32.5% -> 45.2%
- [ground_truth positions] Transitions: 0 counterattacks launched in the next 15 min
- [recorded_event] Outcome: 0 ended in shots
- Followed by (not proof of causation): 0 counterattacks, 0 shots (0 xG)
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [1795, 1805], "observed_players": 3.3, "inferred_players": 6.7, "high_confidence_inferred": 3.9, "outfield_players": 10, "metrics": {"observed_only": {"def_line_m": 60.6, "width_m": 32.5, "near_ball": 0.7}, "reconstructed": {"def_line_m": 43.5, "width_m": 40.7, "near_ball":`

### Does the analysis survive reconstruction? (truth vs camera reconstruction)

Formation agreement per team-minute: 85%. Mean absolute difference per minute: defensive line 1.62 m, width 3.51 m, compactness 1.12 m, press index 4.46. Position-based moments: truth 2, reconstruction 1, recalled within ±120 s 0, reconstruction-only 1.

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Positions are ground truth; the camera view is simulated.

## h2

**Data basis:** In-stadium tracking of all players (ground truth); official event feed. Positions: `ground_truth`. Events: `recorded_event (official DFL feed)`.

Hidden-player model: `r13_mlp_aug_ens3`.

| | A | B |
|---|---:|---:|
| Goals | 1 | 1 |
| Shots | 8 | 7 |
| xG (engine model) | 1.27 | 0.32 |
| Passes | 194 | 248 |
| Pass accuracy | 0.701 | 0.742 |
| Possession (s) | 1468 | 1469 |
| Press index (heuristic) | 22.3 | 22.6 |
| Last formation (heuristic) | 4-2-3-1 | 4-2-3-1 |

**Formations over time (heuristic):** A: 0' 3-5-2, 5' 4-2-3-1; B: 0' 4-2-3-1

**Counterattacks detected (experimental):** 2

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
| 51' | FORMATION_SHIFT | A | Fortuna Düsseldorf switch to 4-2-3-1 | 0.92 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups; 57% minute agreement truth vs reconstruction) |
| 61' | GOAL_BUILD_UP | B | Goal F.C. Hansa Rostock - Lukas Fröde | 0.95 | 1.0 | recorded goal; build-up sequence from events |
| 71' | SUBSTITUTION_IMPACT | A | Jorrit Hendrix on for Shinta Appelkamp | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 72' | SUBSTITUTION_IMPACT | A | E. Iyoha on for Dawid Kownacki | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 77' | GOAL_BUILD_UP | A | Goal Fortuna Düsseldorf - E. Iyoha | 0.95 | 1.0 | recorded goal; build-up sequence from events |
| 82' | SUBSTITUTION_IMPACT | B | T. Meißner on for Frederic Ananou | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 82' | SUBSTITUTION_IMPACT | B | Nils Fröling on for K. Pröger | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 82' | SUBSTITUTION_IMPACT | B | R. Munsy on for J. Verhoek | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 85' | SUBSTITUTION_IMPACT | A | E. Fernandes Neto on for Kristoffer Peterson | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 90' | SUBSTITUTION_IMPACT | A | D. Bunk on for A. Tanaka | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 90' | SUBSTITUTION_IMPACT | A | Kwadwo Baah on for F. Klaus | 0.7 | 1.0 | recorded substitution; after-effects heuristic |

### Evidence chains (top 3 by importance)

**61' Goal F.C. Hansa Rostock - Lukas Fröde** (GOAL_BUILD_UP). Chain: EVENT: Spell of possession → CONSEQUENCE: 6 passes, 25.3s → OUTCOME: Goal (0.145 xG)

- [recorded_event] Build-up: 6 passes in 25.3s
- [recorded_event] Shot: Lukas Fröde shot from 8.3m
- [recorded_event] Outcome: Goal - 2-1
- Followed by (not proof of causation): Score 2-1
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [897, 907], "observed_players": 6.3, "inferred_players": 3.7, "high_confidence_inferred": 3.7, "outfield_players": 13, "metrics": {"observed_only": {"def_line_m": 84.2, "width_m": 25.7, "near_ball": 2.3}, "reconstructed": {"def_line_m": 72.9, "width_m": 31.1, "near_ball": 2`

**77' Goal Fortuna Düsseldorf - E. Iyoha** (GOAL_BUILD_UP). Chain: EVENT: Interception by M. Sobottka → CONSEQUENCE: 2 passes, 4.6s → OUTCOME: Goal (0.269 xG)

- [recorded_event] Recovery: M. Sobottka won the ball (interception) 83m from own goal
- [recorded_event] Build-up: 2 passes in 4.6s
- [recorded_event] Shot: E. Iyoha shot from 11.5m
- [recorded_event] Outcome: Goal - 3-1
- Followed by (not proof of causation): Score 3-1
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [1906, 1916], "observed_players": 3.0, "inferred_players": 7.0, "high_confidence_inferred": 1.9, "outfield_players": 15, "metrics": {"observed_only": {"def_line_m": 85.8, "width_m": 27.3, "near_ball": 1.8}, "reconstructed": {"def_line_m": 77.5, "width_m": 31.8, "near_ball":`

**51' Fortuna Düsseldorf switch to 4-2-3-1** (FORMATION_SHIFT). Chain: TACTICAL_CHANGE: 3-5-2 -> 4-2-3-1 → STATE_CHANGE: Right-side press 0.0% -> 23.9% → CONSEQUENCE: 0 final-third recoveries

- [heuristic over ground_truth positions] Tactical trigger: Shape changed 3-5-2 -> 4-2-3-1
- [ground_truth positions] Structural change: Defensive line -22.66m, width -4.54m
- [ground_truth positions] Pressing: Press intensity 33.6 -> 26.9; right-side share of pressures 0.0% -> 23.9%
- [recorded_event] Recovery: Final-third recoveries 1 -> 0
- [ground_truth positions] Territory: Territory 81.4% -> 58.0%
- Followed by (not proof of causation): 0 final-third recoveries, Territory 58.0%, 1 shots
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [295, 305], "observed_players": 0.8, "inferred_players": 9.2, "high_confidence_inferred": 3.9, "outfield_players": 15, "metrics": {"observed_only": {"def_line_m": 50.1, "width_m": 30.5, "near_ball": 4.0}, "reconstructed": {"def_line_m": 40.0, "width_m": 31.4, "near_ball": 4`

### Does the analysis survive reconstruction? (truth vs camera reconstruction)

Formation agreement per team-minute: 60%. Mean absolute difference per minute: defensive line 1.49 m, width 3.17 m, compactness 1.15 m, press index 4.02. Position-based moments: truth 1, reconstruction 2, recalled within ±120 s 1, reconstruction-only 1.

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Positions are ground truth; the camera view is simulated.
