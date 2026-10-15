# Real tracking · DFL open data: dfl-J03WQQ-r13

## h1

**Data basis:** In-stadium tracking of all players (ground truth); official event feed. Positions: `ground_truth`. Events: `recorded_event (official DFL feed)`.

Hidden-player model: `r13_mlp_aug_ens3`.

| | A | B |
|---|---:|---:|
| Goals | 1 | 0 |
| Shots | 7 | 8 |
| xG (engine model) | 0.68 | 0.84 |
| Passes | 235 | 290 |
| Pass accuracy | 0.834 | 0.803 |
| Possession (s) | 1340 | 1429 |
| Press index (heuristic) | 16.6 | 23.6 |
| Last formation (heuristic) | 3-5-2 | 3-5-2 |

**Formations over time (heuristic):** A: 0' 3-5-2, 30' 3-4-1-2; B: 0' 3-5-2, 45' 4-1-4-1

**Counterattacks detected (experimental):** 1

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
| 16' | TACTICAL_TURNING_POINT | B | FC St. Pauli drop into a low block | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 22' | GOAL_BUILD_UP | A | Goal Fortuna Düsseldorf - R. Hennings | 0.95 | 1.0 | recorded goal; build-up sequence from events |
| 35' | TACTICAL_TURNING_POINT | A | Fortuna Düsseldorf drop into a low block | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 46' | FORMATION_SHIFT | B | FC St. Pauli switch to 4-1-4-1 | 0.92 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups; 57% minute agreement truth vs reconstruction) |

### Evidence chains (top 3 by importance)

**22' Goal Fortuna Düsseldorf - R. Hennings** (GOAL_BUILD_UP). Chain: EVENT: Spell of possession → CONSEQUENCE: 2 passes, 22.5s → OUTCOME: Goal (0.048 xG)

- [recorded_event] Build-up: 2 passes in 22.5s
- [recorded_event] Shot: R. Hennings shot from 24.9m
- [recorded_event] Outcome: Goal - 1-0
- Followed by (not proof of causation): Score 1-0
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [1257, 1267], "observed_players": 0.4, "inferred_players": 9.6, "high_confidence_inferred": 1.0, "outfield_players": 10, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 47.9, "width_m": 29.1, "near_ball": 1.8}, "truth": {"def_line_m": 34.1, "width_m": 47.`

**46' FC St. Pauli switch to 4-1-4-1** (FORMATION_SHIFT). Chain: TACTICAL_CHANGE: 3-5-2 -> 4-1-4-1 → STATE_CHANGE: Right-side press 45.5% -> 0.0% → CONSEQUENCE: 0 final-third recoveries

- [heuristic over ground_truth positions] Tactical trigger: Shape changed 3-5-2 -> 4-1-4-1
- [ground_truth positions] Structural change: Defensive line +22.37m, width -5.91m
- [ground_truth positions] Pressing: Press intensity 25.9 -> 56.4; right-side share of pressures 45.5% -> 0.0%
- [recorded_event] Recovery: Final-third recoveries 0 -> 0
- [ground_truth positions] Territory: Territory 74.8% -> 100.0%
- Followed by (not proof of causation): 0 final-third recoveries, Territory 100.0%, 1 shots
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [2695, 2705], "observed_players": 0.5, "inferred_players": 9.5, "high_confidence_inferred": 0.6, "outfield_players": 10, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 81.1, "width_m": 40.6, "near_ball": 1.9}, "truth": {"def_line_m": 81.3, "width_m": 40.`

**16' FC St. Pauli drop into a low block** (TACTICAL_TURNING_POINT). Chain: TACTICAL_CHANGE: Low Block → STATE_CHANGE: Line 47.0m -> 30.6m → CONSEQUENCE: 0 counterattacks

- [heuristic over ground_truth positions] Tactical trigger: Block dropped: defensive line 47.0m -> 30.6m
- [ground_truth positions] Ceded the ball: Possession 40.8% -> 42.0%
- [ground_truth positions] Transitions: 0 counterattacks launched in the next 15 min
- [recorded_event] Outcome: 0 ended in shots
- Followed by (not proof of causation): 0 counterattacks, 1 shots (0.17 xG)
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [895, 905], "observed_players": 4.7, "inferred_players": 5.3, "high_confidence_inferred": 4.7, "outfield_players": 10, "metrics": {"observed_only": {"def_line_m": 53.1, "width_m": 24.2, "near_ball": 1.5}, "reconstructed": {"def_line_m": 45.1, "width_m": 45.9, "near_ball": 1`

### Tactical & formation changes (every change, both teams)

**Team A** (6 changes)
- 3' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -11.0m, width +3.1m, possession -11.3%, territory -32.2%, press index -13.4. Followed by 1 shot(s) for (0.04 xG) and 0 against (0 xG). _(ground truth positions)_
- 16' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) +20.0m, width +3.9m, territory +25.6%, press index +13.5. Followed by 1 shot(s) for (0.03 xG) and 0 against (0 xG). _(ground truth positions)_
- 31' [FORMATION] Shape read as 3-5-2 -> 3-4-1-2. Kristoffer Peterson midfield -> defensive midfield; M. Karbownik midfield -> defensive midfield; E. Iyoha midfield -> defensive midfield; Jorrit Hendrix midfield -> defensive midfield. Defensive line (out of possession) -18.8m, width -6.8m, possession -9.6%, territory -19.6%, press index -10.9. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 37' [TACTIC] From balanced to low block. Defensive line (out of possession) -12.9m, width -13.7m, compactness -4.7m, possession -29.5%, territory -21.3%, press index -11.9. Followed by 0 shot(s) for (0 xG) and 1 against (0.12 xG). _(ground truth positions)_
- 40' [TACTIC] From low block to balanced. Defensive line (out of possession) +12.1m, width +15.8m, compactness +6.4m, possession +34.5%, territory +17.7%, press index +10.5. Followed by 1 shot(s) for (0.22 xG) and 2 against (0.08 xG). _(ground truth positions)_
- 44' [TACTIC] From balanced to low block. Defensive line (out of possession) -13.5m, width -10.7m, compactness -3.8m, possession -33.0%, territory -16.5%. Followed by 0 shot(s) for (0 xG) and 1 against (0.03 xG). _(ground truth positions)_

**Team B** (9 changes)
- 3' [TACTIC] From low block to balanced. Defensive line (out of possession) +27.7m, width +10.5m, compactness +4.0m, possession +11.3%, territory +32.3%, press index +11.9. Followed by 0 shot(s) for (0 xG) and 1 against (0.04 xG). _(ground truth positions)_
- 11' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +14.2m, territory +37.8%, press index +9.4. Followed by 2 shot(s) for (0.28 xG) and 0 against (0 xG). _(ground truth positions)_
- 13' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -17.2m, width -4.3m, possession -37.4%, territory -19.8%, press index -20.9. Followed by 0 shot(s) for (0 xG) and 2 against (0.33 xG). _(ground truth positions)_
- 18' [TACTIC] From balanced to low block. Defensive line (out of possession) -21.1m, possession +18.9%, territory -36.7%. Followed by 0 shot(s) for (0 xG) and 1 against (0.03 xG). _(ground truth positions)_
- 20' [TACTIC] From low block to balanced. Defensive line (out of possession) +15.2m, territory +9.9%. Followed by 1 shot(s) for (0.17 xG) and 1 against (0.05 xG), 1 against. _(ground truth positions)_
- 36' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +21.4m, width -3.3m, possession +28.0%, territory +26.0%, press index +13.4. Followed by 1 shot(s) for (0.12 xG) and 0 against (0 xG). _(ground truth positions)_
- 40' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -24.0m, possession -29.9%, territory -22.1%, press index -23.7. Followed by 0 shot(s) for (0 xG) and 1 against (0.22 xG). _(ground truth positions)_
- 43' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +12.7m, width +5.5m, possession +21.7%, territory +17.6%, press index +17.1. Followed by 2 shot(s) for (0.08 xG) and 0 against (0 xG). _(ground truth positions)_
- 46' [FORMATION] Shape read as 3-5-2 -> 4-1-4-1. M. Saliakas midfield -> defence; Afeez Aremu midfield -> defence; L. Paqarada midfield -> defence; A. Dźwigała defence -> defensive midfield. Defensive line (out of possession) +12.5m, width -5.4m, possession +11.5%, territory +9.7%, press index +21.9. Followed by 1 shot(s) for (0.03 xG) and 0 against (0 xG). _(ground truth positions)_

### Does the analysis survive reconstruction? (truth vs camera reconstruction)

Formation agreement per team-minute: 32%. Mean absolute difference per minute: defensive line 1.33 m, width 3.9 m, compactness 1.09 m, press index 3.46. Position-based moments: truth 3, reconstruction 3, recalled within ±120 s 3, reconstruction-only 0.

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Positions are ground truth; the camera view is simulated.

## h2

**Data basis:** In-stadium tracking of all players (ground truth); official event feed. Positions: `ground_truth`. Events: `recorded_event (official DFL feed)`.

Hidden-player model: `r13_mlp_aug_ens3`.

| | A | B |
|---|---:|---:|
| Goals | 0 | 0 |
| Shots | 6 | 6 |
| xG (engine model) | 0.56 | 0.25 |
| Passes | 135 | 265 |
| Pass accuracy | 0.741 | 0.811 |
| Possession (s) | 1404 | 1522 |
| Press index (heuristic) | 23.5 | 28.3 |
| Last formation (heuristic) | 4-4-2 | 5-3-1 |

**Formations over time (heuristic):** A: 0' 3-4-1-2, 10' 4-4-2; B: 0' 3-5-2, 15' 5-3-1, 40' 4-2-3

**Counterattacks detected (experimental):** 4

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
| 58' | RED_CARD_IMPACT | B | Red card: B. Fazliji (FC St. Pauli) | 0.9 | 1.0 | recorded card; after-effects heuristic |
| 66' | SUBSTITUTION_IMPACT | A | F. Klaus on for Kristoffer Peterson | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 67' | SUBSTITUTION_IMPACT | A | Kwadwo Baah on for Dawid Kownacki | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 70' | SUBSTITUTION_IMPACT | B | Luca Zander on for M. Saliakas | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 70' | SUBSTITUTION_IMPACT | B | D. Otto on for Lukas Daschner | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 70' | SUBSTITUTION_IMPACT | B | Connor Metcalfe on for Afeez Aremu | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 78' | SUBSTITUTION_IMPACT | A | A. Bodzek on for Shinta Appelkamp | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 82' | SUBSTITUTION_IMPACT | B | J. Eggestein on for M. Hartel | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 82' | SUBSTITUTION_IMPACT | B | I. Matanović on for E. Amenyido | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 86' | FORMATION_SHIFT | B | FC St. Pauli switch to 4-2-3 | 0.81 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups; 57% minute agreement truth vs reconstruction) |
| 92' | SUBSTITUTION_IMPACT | A | A. Hoffmann on for E. Iyoha | 0.7 | 1.0 | recorded substitution; after-effects heuristic |

### Evidence chains (top 3 by importance)

**58' Red card: B. Fazliji (FC St. Pauli)** (RED_CARD_IMPACT). Chain: EVENT: Red card → STATE_CHANGE: 3-5-2 -> 5-3-1 → TACTICAL_CHANGE: Fortuna Düsseldorf push up → CONSEQUENCE: Territory 71.7%

- [ground_truth positions] Event: B. Fazliji sent off - violentConduct
- [ground_truth positions] Structural change: FC St. Pauli 3-5-2 -> 5-3-1, line 47.4m -> 31.0m
- [ground_truth positions] Control: Fortuna Düsseldorf possession 39.0% -> 49.1%, territory 36.8% -> 71.7%
- [ground_truth positions] Pressure: Fortuna Düsseldorf shots 1 -> 4 (0.31 xG) in comparable windows
- Followed by (not proof of causation): FC St. Pauli dropped to 5-3-1, Fortuna Düsseldorf territory 71.7%
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [765, 775], "observed_players": 3.6, "inferred_players": 6.4, "high_confidence_inferred": 6.4, "outfield_players": 15, "metrics": {"observed_only": {"def_line_m": 25.1, "width_m": 6.7, "near_ball": 10.0}, "reconstructed": {"def_line_m": 24.6, "width_m": 7.5, "near_ball": 10`

**86' FC St. Pauli switch to 4-2-3** (FORMATION_SHIFT). Chain: TACTICAL_CHANGE: 5-3-1 -> 4-2-3 → STATE_CHANGE: Right-side press 46.7% -> 36.8% → CONSEQUENCE: 0 final-third recoveries

- [heuristic over ground_truth positions] Tactical trigger: Shape changed 5-3-1 -> 4-2-3
- [ground_truth positions] Structural change: Defensive line -9.91m, width -4.81m
- [ground_truth positions] Pressing: Press intensity 31.4 -> 24.9; right-side share of pressures 46.7% -> 36.8%
- [recorded_event] Recovery: Final-third recoveries 0 -> 0
- [ground_truth positions] Territory: Territory 72.8% -> 72.6%
- Followed by (not proof of causation): 0 final-third recoveries, Territory 72.6%, 3 shots
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [2395, 2405], "observed_players": 0.0, "inferred_players": 9.0, "high_confidence_inferred": 0.0, "outfield_players": 15, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 41.8, "width_m": 42.6, "near_ball": 1.2}, "truth": {"def_line_m": 37.1, "width_m": 43.`

**66' F. Klaus on for Kristoffer Peterson** (SUBSTITUTION_IMPACT). Chain: EVENT: Substitution → STATE_CHANGE: Fresh legs on the RM

- [recorded_event] Change: F. Klaus replaces Kristoffer Peterson (RM)
- [ground_truth positions] Player activity: 13 pressures (1 on the right), 8 recoveries in first 15 min
- [ground_truth positions] Team effect: Press intensity 42.1 -> 13.5
- Followed by (not proof of causation): 13 pressures from F. Klaus
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [1237, 1247], "observed_players": 0.5, "inferred_players": 9.5, "high_confidence_inferred": 2.1, "outfield_players": 14, "metrics": {"observed_only": {"def_line_m": 68.4, "width_m": 9.7, "near_ball": 0.0}, "reconstructed": {"def_line_m": 49.2, "width_m": 36.0, "near_ball": `

### Tactical & formation changes (every change, both teams)

**Team A** (6 changes)
- 56' [FORMATION] Shape read as 3-4-1-2 -> 4-4-2. Jorrit Hendrix defensive midfield -> defence; Kristoffer Peterson defensive midfield -> midfield; M. Karbownik defensive midfield -> midfield; E. Iyoha defensive midfield -> midfield. Defensive line (out of possession) -10.3m, width -3.1m, possession -6.2%, territory -22.0%, press index -14.9. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 57' [TACTIC] From balanced to low block. Defensive line (out of possession) -15.3m, width -9.6m, compactness -4.3m, possession +21.0%, territory +10.7%. Followed by 0 shot(s) for (0 xG) and 1 against (0.05 xG). _(ground truth positions)_
- 60' [TACTIC] From low block to territorial control. Defensive line (out of possession) +55.0m, possession +30.2%, territory +53.9%, press index +44.6. Followed by 4 shot(s) for (0.31 xG) and 0 against (0 xG). Preceded by opponent card 13'; card 14'. _(ground truth positions)_
- 68' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -22.6m, width -3.8m, possession -26.8%, territory -45.0%, press index -23.7. Followed by 0 shot(s) for (0 xG) and 1 against (0.04 xG). Preceded by card 21'; substitution 21': F. Klaus on for Kristoffer Peterson; substitution 22': Kwadwo Baah on for Dawid Kownacki. _(ground truth positions)_
- 78' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) -6.2m, possession -9.2%, territory -14.4%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by opponent card 30'. _(ground truth positions)_
- 91' [STRUCTURE] Structural change in how the team is positioned. Width -6.5m, compactness -3.4m, possession -11.6%, territory -9.4%, press index -26.2. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_

**Team B** (10 changes)
- 56' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +9.6m, width -7.4m, territory +31.2%, press index +16.9. Followed by 1 shot(s) for (0.05 xG) and 0 against (0 xG). _(ground truth positions)_
- 58' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -22.0m, width -20.9m, compactness -8.3m, possession -39.0%, territory -50.3%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 61' [FORMATION] Shape read as 3-5-2 -> 5-3-1. M. Saliakas midfield -> defence; Eric Smith midfield -> defence; L. Paqarada midfield -> defence; Lukas Daschner attack -> midfield. Defensive line (out of possession) -11.3m, possession -21.0%, territory -19.7%, press index -10.4. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by card 13'; opponent card 14'. _(ground truth positions)_
- 62' [TACTIC] From balanced to low block. Defensive line (out of possession) -6.3m, width +6.7m, compactness +3.0m, possession +18.7%. Followed by 0 shot(s) for (0 xG) and 4 against (0.31 xG). Preceded by opponent card 14'. _(ground truth positions)_
- 65' [TACTIC] From low block to balanced. Defensive line (out of possession) +16.5m, width +14.4m, compactness +5.8m, possession +49.1%, territory +17.4%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 73' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) -11.1m, width -4.8m, possession -30.0%, territory -13.6%, press index -22.8. Followed by 0 shot(s) for (0 xG) and 1 against (0.14 xG). Preceded by substitution 25': Luca Zander on for M. Saliakas; substitution 25': D. Otto on for Lukas Daschner; substitution 25': Connor Metcalfe on for Afeez Aremu. _(ground truth positions)_
- 83' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +6.0m, width -3.4m, territory +27.5%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by substitution 37': J. Eggestein on for M. Hartel; substitution 37': I. Matanović on for E. Amenyido. _(ground truth positions)_
- 86' [FORMATION] Shape read as 5-3-1 -> 4-2-3. Jackson Irvine defence -> midfield; J. Eggestein midfield -> attack; I. Matanović midfield -> attack. Width -3.4m, possession -13.1%, territory -17.9%. Followed by 3 shot(s) for (0.12 xG) and 0 against (0 xG). _(ground truth positions)_
- 86' [TACTIC] From territorial control to balanced. Width -3.4m, possession -13.1%, territory -17.9%. Followed by 3 shot(s) for (0.12 xG) and 0 against (0 xG). _(ground truth positions)_
- 92' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +13.0m, possession +7.5%, territory +14.1%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by opponent substitution 47': A. Hoffmann on for E. Iyoha. _(ground truth positions)_

### Does the analysis survive reconstruction? (truth vs camera reconstruction)

Formation agreement per team-minute: 85%. Mean absolute difference per minute: defensive line 1.57 m, width 3.23 m, compactness 1.08 m, press index 3.52. Position-based moments: truth 1, reconstruction 1, recalled within ±120 s 1, reconstruction-only 0.

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Positions are ground truth; the camera view is simulated.
