# Real tracking · DFL · camera estimate r13: dfl-J03WPY-r13

## h1

**Data basis:** In-stadium tracking of all players (ground truth); official event feed. Positions: `ground_truth`. Events: `recorded_event (official DFL feed)`.

Hidden-player model: `r13_mlp_aug_ens3`.

| | A | B |
|---|---:|---:|
| Goals | 0 | 0 |
| Shots | 7 | 6 |
| xG (engine model) | 0.46 | 0.92 |
| Passes | 363 | 160 |
| Pass accuracy | 0.882 | 0.775 |
| Possession (s) | 1775 | 991 |
| Press index (heuristic) | 34.3 | 14.8 |
| Last formation (heuristic) | 3-4-3 | 3-3-2-2 |

**Formations over time (heuristic):** A: 0' 4-2-3-1, 10' 3-4-2-1, 15' 5-3-2, 25' 3-4-3, 45' 4-1-3-2; B: 0' 3-3-2-2, 45' 4-1-3-2

**Counterattacks detected (experimental):** 8

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
| 7' | TACTICAL_TURNING_POINT | B | 1. FC Nürnberg take territorial control | 0.81 | 1.0 | heuristic; high-press thresholds UNVALIDATED on real football (experimental) |
| 8' | COUNTERATTACK | B | 1. FC Nürnberg counterattack | 0.78 | 1.0 | heuristic; real-data F1 0.55 vs DFL CounterAttack flags (experimental) |
| 16' | FORMATION_SHIFT | A | Fortuna Düsseldorf switch to 5-3-2 | 0.92 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups) |
| 33' | TACTICAL_TURNING_POINT | A | Fortuna Düsseldorf take territorial control | 0.9 | 1.0 | heuristic; high-press thresholds UNVALIDATED on real football (experimental) |
| 46' | FORMATION_SHIFT | A | Fortuna Düsseldorf switch to 4-1-3-2 | 0.87 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups) |
| 46' | FORMATION_SHIFT | B | 1. FC Nürnberg switch to 4-1-3-2 | 0.82 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups) |
| 46' | COUNTERATTACK | B | 1. FC Nürnberg counterattack | 0.78 | 1.0 | heuristic; real-data F1 0.55 vs DFL CounterAttack flags (experimental) |

### Evidence chains (top 3 by importance)

**8' 1. FC Nürnberg counterattack** (COUNTERATTACK). Chain: EVENT: Ball regained → STATE_CHANGE: Opponent out of shape → CONSEQUENCE: 64.9m in 11s → OUTCOME: Shot (0.506 xG)

- [recorded_event] Recovery: James Lawrence regains 8m from own goal
- [ground_truth positions] Opponent shape: Fortuna Düsseldorf defensive line at 79.83m
- [heuristic over ground_truth positions] Transition: 64.9m in 11s, 1 pass(es)
- [ground_truth positions] Exposed defence: 10 players caught upfield
- [recorded_event] Outcome: Shot by C. Daferner (saved)
- Followed by (not proof of causation): 10 opponents bypassed, Shot
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [453, 463], "observed_players": 0.0, "inferred_players": 10.0, "high_confidence_inferred": 0.2, "outfield_players": 10, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 47.4, "width_m": 44.2, "near_ball": 2.0}, "truth": {"def_line_m": 43.3, "width_m": 41.5`

**16' Fortuna Düsseldorf switch to 5-3-2** (FORMATION_SHIFT). Chain: TACTICAL_CHANGE: 3-4-2-1 -> 5-3-2 → STATE_CHANGE: Right-side press 46.7% -> 62.5% → CONSEQUENCE: 0 final-third recoveries

- [heuristic over ground_truth positions] Tactical trigger: Shape changed 3-4-2-1 -> 5-3-2
- [ground_truth positions] Structural change: Defensive line -17.2m, width -9.47m
- [ground_truth positions] Pressing: Press intensity 37.4 -> 15.7; right-side share of pressures 46.7% -> 62.5%
- [recorded_event] Recovery: Final-third recoveries 0 -> 0
- [ground_truth positions] Territory: Territory 53.8% -> 51.7%
- Followed by (not proof of causation): 0 final-third recoveries, Territory 51.7%, 4 shots
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [895, 905], "observed_players": 5.5, "inferred_players": 4.5, "high_confidence_inferred": 4.3, "outfield_players": 10, "metrics": {"observed_only": {"def_line_m": 56.2, "width_m": 30.7, "near_ball": 1.1}, "reconstructed": {"def_line_m": 51.2, "width_m": 51.7, "near_ball": 1`

**46' Fortuna Düsseldorf switch to 4-1-3-2** (FORMATION_SHIFT). Chain: TACTICAL_CHANGE: 3-4-3 -> 4-1-3-2 → STATE_CHANGE: Right-side press 29.4% -> 100.0% → CONSEQUENCE: 0 final-third recoveries

- [heuristic over ground_truth positions] Tactical trigger: Shape changed 3-4-3 -> 4-1-3-2
- [ground_truth positions] Structural change: Defensive line +9.57m, width -13.85m
- [ground_truth positions] Pressing: Press intensity 42.7 -> 23.8; right-side share of pressures 29.4% -> 100.0%
- [recorded_event] Recovery: Final-third recoveries 0 -> 0
- [ground_truth positions] Territory: Territory 47.3% -> 25.0%
- Followed by (not proof of causation): 0 final-third recoveries, Territory 25.0%, 0 shots
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [2695, 2705], "observed_players": 3.4, "inferred_players": 6.6, "high_confidence_inferred": 4.5, "outfield_players": 10, "metrics": {"observed_only": {"def_line_m": 85.2, "width_m": 24.4, "near_ball": 3.0}, "reconstructed": {"def_line_m": 80.0, "width_m": 31.2, "near_ball":`

### Does the analysis survive reconstruction? (truth vs camera reconstruction)

Formation agreement per team-minute: 37%. Mean absolute difference per minute: defensive line 1.67 m, width 3.57 m, compactness 1.09 m, press index 5.58. Position-based moments: truth 7, reconstruction 6, recalled within ±120 s 6, reconstruction-only 0.

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Positions are ground truth; the camera view is simulated.

## h2

**Data basis:** In-stadium tracking of all players (ground truth); official event feed. Positions: `ground_truth`. Events: `recorded_event (official DFL feed)`.

Hidden-player model: `r13_mlp_aug_ens3`.

| | A | B |
|---|---:|---:|
| Goals | 0 | 1 |
| Shots | 9 | 4 |
| xG (engine model) | 0.54 | 0.86 |
| Passes | 240 | 127 |
| Pass accuracy | 0.804 | 0.685 |
| Possession (s) | 1386 | 1698 |
| Press index (heuristic) | 22.2 | 24.1 |
| Last formation (heuristic) | 4-1-3-2 | 3-5-2 |

**Formations over time (heuristic):** A: 0' 4-1-4-1, 15' 4-1-3-2, 35' 4-2-3-1; B: 0' 3-3-2-2, 15' 3-5-2, 50' 4-5-1

**Counterattacks detected (experimental):** 6

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
| 47' | GOAL_BUILD_UP | B | Goal 1. FC Nürnberg - Kwadwo Duah | 0.95 | 1.0 | recorded goal; build-up sequence from events |
| 63' | SUBSTITUTION_IMPACT | A | R. Hennings on for Shinta Appelkamp | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 63' | SUBSTITUTION_IMPACT | A | N. Gavory on for E. Iyoha | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 63' | SUBSTITUTION_IMPACT | A | E. Fernandes Neto on for A. Tanaka | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 64' | TACTICAL_TURNING_POINT | A | Fortuna Düsseldorf take territorial control | 0.83 | 1.0 | heuristic; high-press thresholds UNVALIDATED on real football (experimental) |
| 65' | SUBSTITUTION_IMPACT | B | E. Wekesser on for M. Møller Dæhli | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 65' | SUBSTITUTION_IMPACT | B | L. Schleimer on for Kwadwo Duah | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 76' | SUBSTITUTION_IMPACT | B | Manuel Wintzheimer on for F. Nürnberger | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 76' | SUBSTITUTION_IMPACT | A | Kristoffer Peterson on for F. Klaus | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 81' | FORMATION_SHIFT | A | Fortuna Düsseldorf switch to 4-2-3-1 | 0.92 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups) |
| 83' | SUBSTITUTION_IMPACT | B | S. Fofana on for J. Geis | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 84' | SUBSTITUTION_IMPACT | B | E. Valentini on for Jan Gyamerah | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 96' | FORMATION_SHIFT | B | 1. FC Nürnberg switch to 4-5-1 | 0.88 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups) |

### Evidence chains (top 3 by importance)

**47' Goal 1. FC Nürnberg - Kwadwo Duah** (GOAL_BUILD_UP). Chain: EVENT: Spell of possession → CONSEQUENCE: 1 passes, 3.3s → OUTCOME: Goal (0.567 xG)

- [recorded_event] Build-up: 1 passes in 3.3s
- [recorded_event] Shot: Kwadwo Duah shot from 10.3m
- [recorded_event] Outcome: Goal - 0-1
- Followed by (not proof of causation): Score 0-1
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [78, 88], "observed_players": 0.0, "inferred_players": 10.0, "high_confidence_inferred": 0.5, "outfield_players": 15, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 58.0, "width_m": 25.0, "near_ball": 4.1}, "truth": {"def_line_m": 55.8, "width_m": 31.6, `

**81' Fortuna Düsseldorf switch to 4-2-3-1** (FORMATION_SHIFT). Chain: TACTICAL_CHANGE: 4-1-3-2 -> 4-2-3-1 → STATE_CHANGE: Right-side press 30.4% -> 32.3% → CONSEQUENCE: 1 final-third recoveries

- [heuristic over ground_truth positions] Tactical trigger: Shape changed 4-1-3-2 -> 4-2-3-1
- [ground_truth positions] Structural change: Defensive line +36.77m, width +7.56m
- [ground_truth positions] Pressing: Press intensity 6.6 -> 34.6; right-side share of pressures 30.4% -> 32.3%
- [recorded_event] Recovery: Final-third recoveries 0 -> 1
- [ground_truth positions] Territory: Territory 48.5% -> 84.9%
- Followed by (not proof of causation): 1 final-third recoveries, Territory 84.9%, 0 shots
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [2095, 2105], "observed_players": 0.0, "inferred_players": 10.0, "high_confidence_inferred": 0.5, "outfield_players": 14, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 53.7, "width_m": 37.3, "near_ball": 2.1}, "truth": {"def_line_m": 46.3, "width_m": 42`

**96' 1. FC Nürnberg switch to 4-5-1** (FORMATION_SHIFT). Chain: TACTICAL_CHANGE: 3-5-2 -> 4-5-1 → STATE_CHANGE: Right-side press 27.3% -> 12.5% → CONSEQUENCE: 0 final-third recoveries

- [heuristic over ground_truth positions] Tactical trigger: Shape changed 3-5-2 -> 4-5-1
- [ground_truth positions] Structural change: Defensive line -11.63m, width -5.65m
- [ground_truth positions] Pressing: Press intensity 9.9 -> 22.9; right-side share of pressures 27.3% -> 12.5%
- [recorded_event] Recovery: Final-third recoveries 0 -> 0
- [ground_truth positions] Territory: Territory 31.3% -> 8.5%
- Followed by (not proof of causation): 0 final-third recoveries, Territory 8.5%, 0 shots
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [2995, 3005], "observed_players": 8.9, "inferred_players": 1.1, "high_confidence_inferred": 1.1, "outfield_players": 15, "metrics": {"observed_only": {"def_line_m": 5.9, "width_m": 12.1, "near_ball": 1.7}, "reconstructed": {"def_line_m": 5.9, "width_m": 13.1, "near_ball": 1`

### Does the analysis survive reconstruction? (truth vs camera reconstruction)

Formation agreement per team-minute: 50%. Mean absolute difference per minute: defensive line 2.05 m, width 3.35 m, compactness 1.12 m, press index 4.19. Position-based moments: truth 3, reconstruction 2, recalled within ±120 s 1, reconstruction-only 1.

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Positions are ground truth; the camera view is simulated.
