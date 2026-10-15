# Computer vision · SkillCorner: cv-r13-skillcorner-epl-2440-h2

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
| Possession (s) | 768 | 1060 |
| Press index (heuristic) | 29.0 | 28.2 |
| Last formation (heuristic) | 4-1-4-1 | 4-2-3-1 |

**Formations over time (heuristic):** A: 0' 4-1-4-1; B: 0' 4-2-3-1

**Counterattacks detected (experimental):** 0

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
### Tactical & formation changes (every change, both teams)

**Team A** (5 changes)
- 55' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) +3.7m, possession +16.0%, territory +14.2%, press index +10.4. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 70' [STRUCTURE] Structural change in how the team is positioned. Width +7.4m, press index +9.7. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 74' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +27.0m, width -3.2m, possession -20.5%, territory +43.0%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 76' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -32.7m, width -7.3m, possession -12.8%, territory -41.6%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 93' [STRUCTURE] Structural change in how the team is positioned. Possession +21.9%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_

**Team B** (7 changes)
- 47' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +13.6m, possession +37.2%, territory +20.5%, press index +22.6. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 51' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -10.6m, possession -9.2%, territory -20.5%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 55' [STRUCTURE] Structural change in how the team is positioned. Defensive line (out of possession) -4.6m, possession -14.8%, territory -11.6%, press index -12.7. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 69' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +12.6m, width +4.4m, possession +10.2%, territory +27.9%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 72' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -12.8m, width -3.0m, possession +6.0%, territory -49.7%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 83' [TACTIC] From balanced to territorial control. Defensive line (out of possession) +9.4m, width -4.0m, possession +6.2%, territory +25.2%. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_
- 89' [TACTIC] From territorial control to balanced. Defensive line (out of possession) -6.2m, width +4.2m, possession -7.8%, territory -38.8%, press index -10.7. Followed by 0 shot(s) for (0 xG) and 0 against (0 xG). _(mostly estimated positions)_

### Unreliable segments (visibility reliability < 0.5, or engine abstained)

- Team A, minutes 0–0 (min reliability 0.45; abstained 0 min)
- Team A, minutes 2–3 (min reliability 0.44; abstained 0 min)
- Team A, minutes 5–6 (min reliability 0.15; abstained 2 min)
- Team A, minutes 8–12 (min reliability 0.25; abstained 4 min)
- Team A, minutes 15–15 (min reliability 0.39; abstained 0 min)
- Team A, minutes 18–23 (min reliability 0.16; abstained 1 min)
- Team A, minutes 25–29 (min reliability 0.19; abstained 2 min)
- Team A, minutes 32–33 (min reliability 0.11; abstained 1 min)
- Team A, minutes 35–35 (min reliability 0.43; abstained 0 min)
- Team A, minutes 37–38 (min reliability 0.18; abstained 1 min)
- Team A, minutes 41–43 (min reliability 0.15; abstained 1 min)
- Team A, minutes 45–45 (min reliability 0.34; abstained 0 min)
- Team A, minutes 47–49 (min reliability 0.18; abstained 1 min)
- Team B, minutes 0–0 (min reliability 0.47; abstained 0 min)
- Team B, minutes 2–3 (min reliability 0.36; abstained 0 min)
- Team B, minutes 5–13 (min reliability 0.2; abstained 4 min)
- Team B, minutes 15–15 (min reliability 0.24; abstained 1 min)
- Team B, minutes 18–20 (min reliability 0.12; abstained 2 min)
- Team B, minutes 22–23 (min reliability 0.25; abstained 1 min)
- Team B, minutes 25–30 (min reliability 0.24; abstained 2 min)
- Team B, minutes 32–33 (min reliability 0.19; abstained 1 min)
- Team B, minutes 35–45 (min reliability 0.12; abstained 4 min)
- Team B, minutes 47–48 (min reliability 0.18; abstained 1 min)

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Hidden players are model estimates with a calibrated 80% radius; observed positions are provider or CV measurements, not verified truth.
