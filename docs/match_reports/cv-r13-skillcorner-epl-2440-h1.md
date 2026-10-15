# Computer vision · SkillCorner: cv-r13-skillcorner-epl-2440-h1

## clip

**Data basis:** Broadcast detections (legacy release, no event feed); hidden players estimated. Positions: `observed (SkillCorner legacy broadcast detections) + model_inferred`. Events: `none`.

Hidden-player model: `r13_mlp_aug_ens3_final`.

| | A | B |
|---|---:|---:|
| Goals | 0 | 0 |
| Shots | 0 | 0 |
| xG (engine model) | 0 | 0 |
| Passes | 0 | 0 |
| Pass accuracy | 0.0 | 0.0 |
| Possession (s) | 773 | 1037 |
| Press index (heuristic) | 22.2 | 26.6 |
| Last formation (heuristic) | 4-3-3 | 4-1-3-2 |

**Formations over time (heuristic):** A: 0' 4-3-3; B: 0' 4-1-3-2

**Counterattacks detected (experimental):** 0

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
| 25' | TACTICAL_TURNING_POINT | B | Manchester City take territorial control | 0.62 | 0.42 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |
| 45' | TACTICAL_TURNING_POINT | B | Manchester City drop into a low block | 0.69 | 0.41 | heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental) |

### Evidence chains (top 3 by importance)

**45' Manchester City drop into a low block** (TACTICAL_TURNING_POINT). Chain: TACTICAL_CHANGE: Low Block → STATE_CHANGE: Line 46.3m -> 18.1m → CONSEQUENCE: 0 counterattacks

- [heuristic over observed + model_inferred positions] Tactical trigger: Block dropped: defensive line 46.3m -> 18.1m
- [observed + model_inferred positions] Ceded the ball: Possession 69.7% -> 41.0%
- [observed + model_inferred positions] Transitions: 0 counterattacks launched in the next 15 min
- [recorded_event] Outcome: 0 ended in shots
- [observed + model_inferred positions] Visibility: On average 5 of 11 players observed; 6 estimated from trajectories and team shape (uncertainty N/A; no calibrated confidence available)
- Followed by (not proof of causation): 0 counterattacks, 0 shots (0 xG)

**25' Manchester City take territorial control** (TACTICAL_TURNING_POINT). Chain: TACTICAL_CHANGE: Dominant → STATE_CHANGE: Line 44.8m -> 37.1m → CONSEQUENCE: 0 counterattacks

- [heuristic over observed + model_inferred positions] Tactical trigger: Block pushed up: defensive line 44.8m -> 37.1m
- [recorded_event] Possession: Possession 58.0% -> 62.0%
- [observed + model_inferred positions] Territory: Territory 57.5% -> 56.5%
- [recorded_event] Outcome: 0 shots (0 xG) in the following minutes
- [observed + model_inferred positions] Visibility: On average 5 of 11 players observed; 6 estimated from trajectories and team shape (uncertainty N/A; no calibrated confidence available)
- Followed by (not proof of causation): Territory 56.5%, 0 shots (0 xG)

### Tactical & formation changes (every change, both teams)

**Team A** (4 changes)
- 7' [TACTIC] From low block to balanced. Defensive line (out of possession) +12.3m, possession +22.2%, territory +27.4%, press index -17.4. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 13' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) +4.1m, width +6.0m, possession +17.0%, territory +8.0%, press index +12.2. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 30' [STRUCTURE] Structural change in how the team is positioned. Possession +5.2%, territory +14.8%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 45' [STRUCTURE] Structural change in how the team is positioned. Possession +34.1%, territory +22.3%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_

**Team B** (9 changes)
- 6' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -9.5m, possession -31.3%, territory -41.5%, press index +15.7. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 10' [TACTIC] From balanced to territorial control. Territory +28.2%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 12' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -5.3m, possession -14.0%, territory -20.5%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 19' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +5.0m, possession -8.8%, territory +17.3%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 21' [TACTIC] From territorial control to balanced. Width -4.1m, compactness -3.2m, possession +26.8%, territory -16.7%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 25' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) +9.1m, width +4.7m, territory +11.7%, press index +21.9. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 29' [TACTIC] From balanced to territorial control. Defensive line (out of possession) -9.3m, possession +21.5%, territory +6.0%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 31' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -11.1m, width -5.4m, possession -30.8%, territory -26.5%, press index -43.4. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 45' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) -10.9m, possession -13.9%, territory -22.3%, press index -27.4. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_

### Unreliable segments (visibility reliability < 0.5, or engine abstained)

- Team A, minutes 0–14 (min reliability 0.03; abstained 8 min)
- Team A, minutes 17–18 (min reliability 0.41; abstained 0 min)
- Team A, minutes 20–23 (min reliability 0.35; abstained 0 min)
- Team A, minutes 25–25 (min reliability 0.45; abstained 0 min)
- Team A, minutes 29–35 (min reliability 0.26; abstained 2 min)
- Team A, minutes 37–42 (min reliability 0.0; abstained 3 min)
- Team A, minutes 44–46 (min reliability 0.28; abstained 1 min)
- Team B, minutes 0–8 (min reliability 0.04; abstained 7 min)
- Team B, minutes 10–14 (min reliability 0.17; abstained 4 min)
- Team B, minutes 17–18 (min reliability 0.37; abstained 0 min)
- Team B, minutes 21–24 (min reliability 0.26; abstained 2 min)
- Team B, minutes 28–29 (min reliability 0.37; abstained 0 min)
- Team B, minutes 31–35 (min reliability 0.29; abstained 2 min)
- Team B, minutes 37–42 (min reliability 0.0; abstained 3 min)
- Team B, minutes 44–46 (min reliability 0.35; abstained 0 min)

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Hidden players are model estimates with a calibrated 80% radius; observed positions are provider or CV measurements, not verified truth.
