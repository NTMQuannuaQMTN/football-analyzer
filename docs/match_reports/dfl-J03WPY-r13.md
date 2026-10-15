# Real tracking · DFL open data: dfl-J03WPY-r13

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
| 7' | TACTICAL_TURNING_POINT | B | 1. FC Nürnberg take territorial control | 0.81 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 8' | COUNTERATTACK | B | 1. FC Nürnberg counterattack | 0.78 | 1.0 | heuristic; vs DFL CounterAttack flags F1 0.55; vs SkillCorner quick-break phases P 0.62 / R 0.23 (experimental) |
| 16' | FORMATION_SHIFT | A | Fortuna Düsseldorf switch to 5-3-2 | 0.92 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups; 57% minute agreement truth vs reconstruction) |
| 33' | TACTICAL_TURNING_POINT | A | Fortuna Düsseldorf take territorial control | 0.9 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 46' | FORMATION_SHIFT | A | Fortuna Düsseldorf switch to 4-1-3-2 | 0.87 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups; 57% minute agreement truth vs reconstruction) |
| 46' | FORMATION_SHIFT | B | 1. FC Nürnberg switch to 4-1-3-2 | 0.82 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups; 57% minute agreement truth vs reconstruction) |
| 46' | COUNTERATTACK | B | 1. FC Nürnberg counterattack | 0.78 | 1.0 | heuristic; vs DFL CounterAttack flags F1 0.55; vs SkillCorner quick-break phases P 0.62 / R 0.23 (experimental) |

### Evidence chains (top 3 by importance)

**8' 1. FC Nürnberg counterattack** (COUNTERATTACK). Chain: EVENT: Ball regained → STATE_CHANGE: Opponent out of shape → CONSEQUENCE: 64.9m in 11s → OUTCOME: Shot (0.506 xG)

- [recorded_event] Recovery: James Lawrence regains 8m from own goal
- [ground_truth positions] Opponent shape: Fortuna Düsseldorf defensive line at 79.83m
- [heuristic over ground_truth positions] Transition: 64.9m in 11s, 1 pass(es)
- [ground_truth positions] Exposed defence: 10 players caught upfield
- [recorded_event] Outcome: Shot by C. Daferner (saved)
- Followed by (not proof of causation): 10 opponents bypassed, Shot
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [453, 463], "observed_players": 0.0, "inferred_players": 10.0, "high_confidence_inferred": 0.0, "outfield_players": 10, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 47.4, "width_m": 44.2, "near_ball": 2.0}, "truth": {"def_line_m": 43.3, "width_m": 41.5`

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
- Camera / reconstruction / truth at this moment: `{"team": "A", "window_s": [2695, 2705], "observed_players": 3.4, "inferred_players": 6.6, "high_confidence_inferred": 4.4, "outfield_players": 10, "metrics": {"observed_only": {"def_line_m": 85.2, "width_m": 24.4, "near_ball": 3.0}, "reconstructed": {"def_line_m": 80.0, "width_m": 31.2, "near_ball":`

### Tactical & formation changes (every change, both teams)

**Team A** (7 changes)
- 4' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) +7.4m, width +3.4m, press index +11.3. Followed by 0 shot(s) for (0 xG) and 1 against (0.51 xG). _(ground truth positions)_
- 11' [FORMATION] Shape read as 4-2-3-1 -> 3-4-2-1. M. Sobottka defensive midfield -> defence; M. Karbownik defence -> defensive midfield; Shinta Appelkamp attacking midfield -> defensive midfield; Matthias Zimmermann defence -> defensive midfield. Defensive line (out of possession) +3.9m, width +6.4m, possession +26.4%, territory +14.6%, press index -12.7. Followed by 1 shot(s) for (0.14 xG) and 0 against (0 xG). _(ground truth positions)_
- 16' [FORMATION] Shape read as 3-4-2-1 -> 5-3-2. M. Karbownik defensive midfield -> defence; Matthias Zimmermann defensive midfield -> defence; A. Tanaka defensive midfield -> midfield; F. Klaus attacking midfield -> midfield. Defensive line (out of possession) -13.7m, width -10.2m, possession -37.4%, press index -24.9. Followed by 0 shot(s) for (0 xG) and 2 against (0.15 xG). Preceded by opponent card 14'. _(ground truth positions)_
- 26' [FORMATION] Shape read as 5-3-2 -> 3-4-3. M. Karbownik defence -> midfield; Shinta Appelkamp attack -> midfield; Matthias Zimmermann defence -> midfield; F. Klaus midfield -> attack. Defensive line (out of possession) +3.1m, width +7.4m, possession +29.0%, territory -21.4%, press index +28.1. Followed by 1 shot(s) for (0.02 xG) and 1 against (0.03 xG). _(ground truth positions)_
- 34' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +10.3m, width +5.2m, possession +33.4%, territory +41.0%, press index +22.5. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 37' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -12.2m, width +3.0m, possession -10.4%, territory -36.4%, press index -23.9. Followed by 1 shot(s) for (0.14 xG) and 0 against (0 xG). Preceded by opponent card 34'; opponent card 35'; opponent card 36'. _(ground truth positions)_
- 46' [FORMATION] Shape read as 3-4-3 -> 4-1-3-2. M. Karbownik midfield -> defence; Shinta Appelkamp midfield -> defence; Matthias Zimmermann midfield -> defence; C. Klarer defence -> defensive midfield. Width -5.7m, possession -15.9%, territory +7.8%, press index -19.0. Followed by 0 shot(s) for (0 xG) and 1 against (0.11 xG). _(ground truth positions)_

**Team B** (8 changes)
- 9' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +15.3m, width +8.1m, compactness +3.8m, possession +25.0%, territory +29.0%, press index +13.0. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 11' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -13.5m, width -6.2m, compactness -3.3m, possession -37.0%, territory -35.4%, press index -13.3. Followed by 0 shot(s) for (0 xG) and 1 against (0.14 xG). _(ground truth positions)_
- 17' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) +12.5m, possession +33.2%, territory +19.2%, press index +19.9. Followed by 2 shot(s) for (0.15 xG) and 0 against (0 xG). Preceded by card 14'. _(ground truth positions)_
- 28' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +9.1m, width +4.5m, possession -30.9%, territory +37.8%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 30' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -4.0m, possession +46.8%, territory -27.3%, press index +23.4. Followed by 0 shot(s) for (0 xG) and 1 against (0.02 xG). _(ground truth positions)_
- 33' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) -9.4m, possession -33.1%, territory -27.1%, press index -19.6. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 46' [FORMATION] Shape read as 3-3-2-2 -> 4-1-3-2. F. Nürnberger defensive midfield -> defence; J. Castrop defensive midfield -> attacking midfield; C. Daferner attack -> attacking midfield; L. Tempelmann attacking midfield -> attack. Defensive line (out of possession) -22.2m, width -6.9m, possession +15.9%, territory -7.8%, press index -12.3. Followed by 1 shot(s) for (0.11 xG) and 0 against (0 xG). _(ground truth positions)_
- 46' [TACTIC] From balanced to low block. Defensive line (out of possession) -22.2m, width -6.9m, possession +15.9%, territory -7.8%, press index -12.3. Followed by 1 shot(s) for (0.11 xG) and 0 against (0 xG). _(ground truth positions)_

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
| 64' | TACTICAL_TURNING_POINT | A | Fortuna Düsseldorf take territorial control | 0.83 | 1.0 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 65' | SUBSTITUTION_IMPACT | B | E. Wekesser on for M. Møller Dæhli | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 65' | SUBSTITUTION_IMPACT | B | L. Schleimer on for Kwadwo Duah | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 76' | SUBSTITUTION_IMPACT | B | Manuel Wintzheimer on for F. Nürnberger | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 76' | SUBSTITUTION_IMPACT | A | Kristoffer Peterson on for F. Klaus | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 81' | FORMATION_SHIFT | A | Fortuna Düsseldorf switch to 4-2-3-1 | 0.92 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups; 57% minute agreement truth vs reconstruction) |
| 83' | SUBSTITUTION_IMPACT | B | S. Fofana on for J. Geis | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 84' | SUBSTITUTION_IMPACT | B | E. Valentini on for Jan Gyamerah | 0.7 | 1.0 | recorded substitution; after-effects heuristic |
| 96' | FORMATION_SHIFT | B | 1. FC Nürnberg switch to 4-5-1 | 0.88 | 1.0 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups; 57% minute agreement truth vs reconstruction) |

### Evidence chains (top 3 by importance)

**47' Goal 1. FC Nürnberg - Kwadwo Duah** (GOAL_BUILD_UP). Chain: EVENT: Spell of possession → CONSEQUENCE: 1 passes, 3.3s → OUTCOME: Goal (0.567 xG)

- [recorded_event] Build-up: 1 passes in 3.3s
- [recorded_event] Shot: Kwadwo Duah shot from 10.3m
- [recorded_event] Outcome: Goal - 0-1
- Followed by (not proof of causation): Score 0-1
- Camera / reconstruction / truth at this moment: `{"team": "B", "window_s": [78, 88], "observed_players": 0.0, "inferred_players": 10.0, "high_confidence_inferred": 0.4, "outfield_players": 15, "metrics": {"observed_only": null, "reconstructed": {"def_line_m": 58.0, "width_m": 25.0, "near_ball": 4.1}, "truth": {"def_line_m": 55.8, "width_m": 31.6, `

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

### Tactical & formation changes (every change, both teams)

**Team A** (14 changes)
- 47' [TACTIC] From low block to balanced. Defensive line (out of possession) +20.5m, width +4.7m, possession +39.8%, territory +15.5%, press index +18.8. Followed by 0 shot(s) for (0 xG) and 2 against (0.64 xG), 1 against. Preceded by goal conceded 2'. _(ground truth positions)_
- 51' [TACTIC] From balanced to low block. Defensive line (out of possession) -22.7m, width -4.4m, possession -30.1%, territory -22.2%, press index -28.2. Followed by 1 shot(s) for (0.08 xG) and 0 against (0 xG). _(ground truth positions)_
- 54' [TACTIC] From low block to balanced. Defensive line (out of possession) +19.9m, width +9.7m, compactness +3.7m, possession +33.3%, territory +14.1%, press index +21.1. Followed by 1 shot(s) for (0.05 xG) and 0 against (0 xG). _(ground truth positions)_
- 61' [FORMATION] Shape read as 4-1-4-1 -> 4-1-3-2. Shinta Appelkamp attacking midfield -> attack. Defensive line (out of possession) +4.1m, possession -11.8%, territory +18.8%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 62' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +3.6m, width -8.6m, possession -12.0%, territory +16.0%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 67' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -16.1m, width -4.3m, territory -49.0%, press index -12.7. Followed by 2 shot(s) for (0.12 xG) and 0 against (0 xG). Preceded by opponent substitution 20': E. Wekesser on for M. Møller Dæhli; opponent substitution 20': L. Schleimer on for Kwadwo Duah. _(ground truth positions)_
- 70' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +17.5m, width +8.7m, possession +55.7%, territory +53.8%, press index +15.5. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 78' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -5.2m, width -5.3m, possession -15.9%, territory -20.0%. Followed by 1 shot(s) for (0.05 xG) and 0 against (0 xG). Preceded by opponent substitution 31': Manuel Wintzheimer on for F. Nürnberger; substitution 31': Kristoffer Peterson on for F. Klaus. _(ground truth positions)_
- 80' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +19.0m, width +6.3m, compactness +3.1m, possession +12.5%, territory +29.0%, press index +16.6. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 81' [FORMATION] Shape read as 4-1-3-2 -> 4-2-3-1. M. Karbownik attacking midfield -> defensive midfield; Dawid Kownacki attack -> attacking midfield. Defensive line (out of possession) +17.8m, possession +29.8%, territory +19.2%, press index +23.7. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 88' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -13.0m, possession -14.9%, territory -23.4%, press index -13.1. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by card 41'. _(ground truth positions)_
- 90' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +7.6m, territory +8.0%, press index +18.8. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 92' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -10.7m, width -3.9m, possession -15.2%, territory -16.7%, press index -18.9. Followed by 2 shot(s) for (0.11 xG) and 1 against (0.09 xG). _(ground truth positions)_
- 95' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +7.6m, possession +8.0%, territory +25.4%, press index +10.9. Followed by 1 shot(s) for (0.09 xG) and 0 against (0 xG). Preceded by opponent card 48'. _(ground truth positions)_

**Team B** (11 changes)
- 48' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -10.8m, width -4.4m, possession -31.3%, territory -18.0%, press index -29.8. Followed by 1 shot(s) for (0.07 xG) and 0 against (0 xG). Preceded by goal scored 2'. _(ground truth positions)_
- 51' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +10.5m, width +4.6m, possession +27.7%, territory +24.0%, press index +30.3. Followed by 0 shot(s) for (0 xG) and 1 against (0.08 xG). _(ground truth positions)_
- 57' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -17.3m, width +3.7m, possession -5.2%, territory -44.0%, press index -15.9. Followed by 0 shot(s) for (0 xG) and 1 against (0.05 xG). Preceded by card 10'. _(ground truth positions)_
- 61' [FORMATION] Shape read as 3-3-2-2 -> 3-5-2. J. Geis defensive midfield -> midfield; L. Tempelmann attacking midfield -> midfield; M. Møller Dæhli attacking midfield -> midfield; F. Nürnberger defensive midfield -> midfield. Defensive line (out of possession) -5.6m, possession +5.5%, territory -12.9%, press index -24.5. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 64' [TACTIC] From balanced to low block. Defensive line (out of possession) -11.8m, width -3.2m, possession +16.8%, territory -23.8%, press index +20.2. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by opponent substitution 18': R. Hennings on for Shinta Appelkamp; opponent substitution 18': N. Gavory on for E. Iyoha; opponent substitution 18': E. Fernandes Neto on for A. Tanaka. _(ground truth positions)_
- 66' [TACTIC] From low block to balanced. Defensive line (out of possession) +20.7m, width -3.4m, territory +51.0%, press index +22.3. Followed by 0 shot(s) for (0 xG) and 2 against (0.12 xG). Preceded by opponent substitution 18': R. Hennings on for Shinta Appelkamp; opponent substitution 18': N. Gavory on for E. Iyoha; opponent substitution 18': E. Fernandes Neto on for A. Tanaka; substitution 20': E. Wekesser on for M. Møller Dæhli; substitution 20': L. Schleimer on for Kwadwo Duah. _(ground truth positions)_
- 70' [TACTIC] From balanced to low block. Defensive line (out of possession) -24.9m, width -4.1m, possession -64.0%, territory -45.0%, press index -47.8. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(ground truth positions)_
- 74' [TACTIC] From low block to balanced. Defensive line (out of possession) +17.9m, width +6.9m, possession +57.5%, territory +39.5%, press index +24.3. Followed by 1 shot(s) for (0.13 xG) and 2 against (0.09 xG). Preceded by opponent card 28'. _(ground truth positions)_
- 83' [TACTIC] From balanced to low block. Defensive line (out of possession) -21.3m, width -4.9m, possession -39.8%, territory -30.8%, press index -23.1. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by card 37'. _(ground truth positions)_
- 87' [TACTIC] From low block to balanced. Defensive line (out of possession) +13.2m, width +4.9m, possession +24.6%, territory +23.2%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by substitution 39': E. Valentini on for Jan Gyamerah; opponent card 41'. _(ground truth positions)_
- 96' [FORMATION] Shape read as 3-5-2 -> 4-5-1. E. Wekesser midfield -> defence; Manuel Wintzheimer attack -> midfield. Defensive line (out of possession) -14.5m, width -4.8m, compactness -3.0m, possession -8.8%, territory -20.1%, press index +12.0. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). Preceded by card 48'. _(ground truth positions)_

### Does the analysis survive reconstruction? (truth vs camera reconstruction)

Formation agreement per team-minute: 50%. Mean absolute difference per minute: defensive line 2.05 m, width 3.35 m, compactness 1.12 m, press index 4.19. Position-based moments: truth 3, reconstruction 2, recalled within ±120 s 1, reconstruction-only 1.

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Positions are ground truth; the camera view is simulated.
