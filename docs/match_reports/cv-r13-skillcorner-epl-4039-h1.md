# Computer vision · SkillCorner: cv-r13-skillcorner-epl-4039-h1

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
| Possession (s) | 1004 | 833 |
| Press index (heuristic) | 22.6 | 26.4 |
| Last formation (heuristic) | 4-2-3-1 | 4-3-3 |

**Formations over time (heuristic):** A: 0' 4-2-3-1; B: 0' 4-3-3, 45' 4-2-3

**Counterattacks detected (experimental):** 0

### Timeline

| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |
|---|---|---|---|---:|---:|---|
### Unreliable segments (visibility reliability < 0.5, or engine abstained)

- Team A, minutes 0–3 (min reliability 0.29; abstained 1 min)
- Team A, minutes 8–8 (min reliability 0.49; abstained 0 min)
- Team A, minutes 10–10 (min reliability 0.26; abstained 1 min)
- Team A, minutes 13–14 (min reliability 0.0; abstained 2 min)
- Team A, minutes 17–17 (min reliability 0.37; abstained 0 min)
- Team A, minutes 19–19 (min reliability 0.29; abstained 1 min)
- Team A, minutes 21–27 (min reliability 0.0; abstained 4 min)
- Team A, minutes 29–29 (min reliability 0.35; abstained 0 min)
- Team A, minutes 32–32 (min reliability 0.22; abstained 1 min)
- Team A, minutes 34–34 (min reliability 0.05; abstained 1 min)
- Team A, minutes 36–36 (min reliability 0.28; abstained 1 min)
- Team A, minutes 41–45 (min reliability 0.1; abstained 1 min)
- Team A, minutes 48–48 (min reliability 0.13; abstained 1 min)
- Team B, minutes 0–3 (min reliability 0.28; abstained 1 min)
- Team B, minutes 5–6 (min reliability 0.45; abstained 0 min)
- Team B, minutes 8–8 (min reliability 0.45; abstained 0 min)
- Team B, minutes 10–10 (min reliability 0.18; abstained 1 min)
- Team B, minutes 13–14 (min reliability 0.0; abstained 2 min)
- Team B, minutes 17–17 (min reliability 0.35; abstained 0 min)
- Team B, minutes 19–19 (min reliability 0.27; abstained 1 min)
- Team B, minutes 21–27 (min reliability 0.0; abstained 5 min)
- Team B, minutes 29–34 (min reliability 0.06; abstained 2 min)
- Team B, minutes 36–36 (min reliability 0.29; abstained 1 min)
- Team B, minutes 42–45 (min reliability 0.11; abstained 1 min)
- Team B, minutes 47–48 (min reliability 0.08; abstained 1 min)

### Limitations

- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.
- Hidden players are model estimates with a calibrated 80% radius; observed positions are provider or CV measurements, not verified truth.
