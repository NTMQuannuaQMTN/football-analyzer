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
