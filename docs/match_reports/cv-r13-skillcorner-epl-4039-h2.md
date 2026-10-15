# Computer vision · SkillCorner: cv-r13-skillcorner-epl-4039-h2

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
| Possession (s) | 820 | 979 |
| Press index (heuristic) | 18.5 | 23.1 |
| Last formation (heuristic) | 4-2-2-1 | 4-2-2-1 |

**Formations over time (heuristic):** A: 0' 4-2-3-1, 10' 4-2-2-1, 25' 3-4-2, 30' 4-2-3, 45' 4-2-2-1; B: 0' 4-2-3, 20' 4-2-2-1

**Counterattacks detected (experimental):** 0

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
| 56' | FORMATION_SHIFT | A | Manchester City possible switch to 4-2-2-1 | 0.5 | 0.34 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups; 57% minute agreement truth vs reconstruction) |
| 71' | FORMATION_SHIFT | A | Manchester City possible switch to 3-4-2 | 0.5 | 0.36 | heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups; 57% minute agreement truth vs reconstruction) |

### Evidence chains (top 3 by importance)

**56' Manchester City possible switch to 4-2-2-1** (FORMATION_SHIFT). Chain: TACTICAL_CHANGE: 4-2-3-1 -> 4-2-2-1 → STATE_CHANGE: Right-side press 0.0% -> 0.0% → CONSEQUENCE: 0 final-third recoveries

- [heuristic over observed + model_inferred positions] Tactical trigger: Shape changed 4-2-3-1 -> 4-2-2-1
- [observed + model_inferred positions] Structural change: Defensive line -22.39m, width -7.24m
- [observed + model_inferred positions] Pressing: Press intensity 20.1 -> 26.5; right-side share of pressures 0.0% -> 0.0%
- [recorded_event] Recovery: Final-third recoveries 0 -> 0
- [observed + model_inferred positions] Territory: Territory 60.8% -> 36.2%
- Followed by (not proof of causation): 0 final-third recoveries, Territory 36.2%, 0 shots

**71' Manchester City possible switch to 3-4-2** (FORMATION_SHIFT). Chain: TACTICAL_CHANGE: 4-2-2-1 -> 3-4-2 → STATE_CHANGE: Right-side press 0.0% -> 0.0% → CONSEQUENCE: 0 final-third recoveries

- [heuristic over observed + model_inferred positions] Tactical trigger: Shape changed 4-2-2-1 -> 3-4-2
- [observed + model_inferred positions] Structural change: Defensive line -21.54m, width -4.35m
- [observed + model_inferred positions] Pressing: Press intensity 14.5 -> 19.7; right-side share of pressures 0.0% -> 0.0%
- [recorded_event] Recovery: Final-third recoveries 0 -> 0
- [observed + model_inferred positions] Territory: Territory 43.2% -> 37.1%
- Followed by (not proof of causation): 0 final-third recoveries, Territory 37.1%, 0 shots

### Tactical & formation changes (every change, both teams)

**Team A** (7 changes)
- 47' [TACTIC] From balanced to territorial control. Possession +30.0%, territory +28.7%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 50' [TACTIC] From territorial control to balanced. Possession -9.2%, territory -21.3%, press index +10.2. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 56' [FORMATION] Possible change of shape 4-2-3-1 -> 4-2-2-1 (most of the team was estimated). Raheem Sterling attacking midfield -> attack. Defensive line (out of possession) -4.0m, width -5.8m, possession -33.2%, territory -18.6%, press index +13.2. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions: the shape reading is uncertain)_
- 71' [FORMATION] Possible change of shape 4-2-2-1 -> 3-4-2 (most of the team was estimated). Rodrigo Hernandez defensive midfield -> midfield; Ilkay Gündogan defensive midfield -> midfield; Phil Foden attacking midfield -> midfield; Kevin de Bruyne attacking midfield -> attack. Possession -32.6%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions: the shape reading is uncertain)_
- 76' [FORMATION] Possible change of shape 3-4-2 -> 4-2-3 (most of the team was estimated). Joao Cancelo midfield -> defence; Phil Foden midfield -> attack. Possession +21.8%, territory -9.6%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions: the shape reading is uncertain)_
- 86' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) -8.5m, width -5.4m, possession -41.8%, territory -9.6%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 91' [FORMATION] Possible change of shape 4-2-3 -> 4-2-2-1 (most of the team was estimated). Rodrigo Hernandez midfield -> defensive midfield; Ilkay Gündogan midfield -> defensive midfield; Bernardo Silva attack -> attacking midfield; Kevin de Bruyne attack -> attacking midfield. Possession +20.8%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions: the shape reading is uncertain)_

**Team B** (8 changes)
- 48' [TACTIC] From balanced to low block. Defensive line (out of possession) -11.7m, possession -20.0%, territory -17.5%, press index -20.4. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 50' [TACTIC] From low block to balanced. Defensive line (out of possession) +6.6m, width -3.2m, possession +5.2%, territory +21.3%, press index +20.0. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 53' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) +14.9m, width +12.2m, possession +32.1%, territory +13.9%, press index +8.0. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 66' [FORMATION] Possible change of shape 4-2-3 -> 4-2-2-1 (most of the team was estimated). Naby Keita midfield -> defensive midfield; Jordan Henderson midfield -> defensive midfield; Sadio Mane attack -> attacking midfield; Mohamed Salah attack -> attacking midfield. Defensive line (out of possession) -3.9m, press index +20.5. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions: the shape reading is uncertain)_
- 71' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) +17.8m, width +4.7m, possession +12.8%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 76' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +5.7m, width +6.2m, territory +16.2%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 79' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -21.5m, width -4.4m, possession -18.6%, territory -35.0%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 91' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) -4.4m, possession -20.8%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_

### Unreliable segments (visibility reliability < 0.5, or engine abstained)

- Team A, minutes 0–6 (min reliability 0.13; abstained 4 min)
- Team A, minutes 8–8 (min reliability 0.38; abstained 0 min)
- Team A, minutes 10–12 (min reliability 0.0; abstained 3 min)
- Team A, minutes 15–17 (min reliability 0.17; abstained 1 min)
- Team A, minutes 20–21 (min reliability 0.12; abstained 2 min)
- Team A, minutes 25–27 (min reliability 0.0; abstained 3 min)
- Team A, minutes 30–30 (min reliability 0.48; abstained 0 min)
- Team A, minutes 32–39 (min reliability 0.11; abstained 2 min)
- Team A, minutes 41–42 (min reliability 0.27; abstained 1 min)
- Team A, minutes 45–45 (min reliability 0.24; abstained 1 min)
- Team A, minutes 47–49 (min reliability 0.27; abstained 2 min)
- Team B, minutes 0–0 (min reliability 0.37; abstained 0 min)
- Team B, minutes 2–6 (min reliability 0.26; abstained 3 min)
- Team B, minutes 8–8 (min reliability 0.31; abstained 1 min)
- Team B, minutes 10–12 (min reliability 0.0; abstained 3 min)
- Team B, minutes 15–17 (min reliability 0.28; abstained 1 min)
- Team B, minutes 19–21 (min reliability 0.14; abstained 2 min)
- Team B, minutes 25–28 (min reliability 0.0; abstained 3 min)
- Team B, minutes 30–30 (min reliability 0.38; abstained 0 min)
- Team B, minutes 32–32 (min reliability 0.39; abstained 0 min)
- Team B, minutes 34–37 (min reliability 0.37; abstained 0 min)
- Team B, minutes 39–39 (min reliability 0.21; abstained 1 min)
- Team B, minutes 41–42 (min reliability 0.24; abstained 1 min)
- Team B, minutes 44–47 (min reliability 0.25; abstained 2 min)

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Hidden players are model estimates with a calibrated 80% radius; observed positions are provider or CV measurements, not verified truth.
