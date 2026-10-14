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

### Does the analysis survive reconstruction? (truth vs camera reconstruction)

Formation agreement per team-minute: 61%. Mean absolute difference per minute: defensive line 1.85 m, width 2.89 m, compactness 1.13 m, press index 3.52. Position-based moments: truth 5, reconstruction 5, recalled within ±120 s 3, reconstruction-only 2.

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Positions are ground truth; the camera view is simulated.
