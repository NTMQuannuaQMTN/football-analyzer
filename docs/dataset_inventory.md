# Dataset inventory: broadcast video and ground-truth positions

Last checked: 13 October 2026. Local entries were opened and parsed this session. External entries were checked on their official or paper pages; none of them was downloaded.

## Bottom line

**No accessible dataset gives broadcast video of a match and independently measured positions of all 22 players for that same match.** That includes everything stored locally and every public source checked. The three closest real sources each lack one piece:

| Need | DFL (local) | SoccerNet-GSR (local) | SkillCorner A-League (local) |
|---|---|---|---|
| Broadcast video | ❌ none | ✅ 30 s clips | ❌ none (the footprint is derived from it) |
| All-player positions | ✅ TRACAB in-stadium, 25 Hz | ❌ on-screen players only | ✅ but off-screen positions are SkillCorner's own extrapolations |
| Positions measured independently of the broadcast | ✅ | ✅ human-annotated | ❌ computer vision on the broadcast |
| Real broadcast visibility | ❌ (we simulate it) | ✅ | ✅ `is_detected` + camera footprint |

So this project uses a **hybrid** strategy (details at the end). It keeps **direct supervision**, where positions are measured independently, separate from **proxy supervision**, where positions come from someone's computer vision or the visibility pattern is simulated.

## Comparison table

Grades:
- **A**: usable for direct hidden-player supervision.
- **B**: direct truth for visible players only, or real visibility with proxy truth.
- **C**: not usable for accuracy claims here.

| Dataset | Broadcast video available? | Same-match ground truth? | Sync possible? | All 22 + GKs? | Identity consistency | Calibration | Rate / size | Access and licence | Local status | Grade / use |
|---|---|---|---|---|---|---|---|---|---|---|
| **DFL / IDSSE open data** (Bassek et al. 2025, figshare 10.6084/m9.figshare.28196177) | **No.** No video is published for these matches. | Yes: TRACAB in-stadium optical tracking | n/a (no video) | Yes, plus ball in 3D | Yes: DFL person IDs, shirt numbers, playing positions | n/a | 25 Hz, 7 matches, 14 halves (≈ 11.6 h) | CC BY 4.0, open | `data/dfl/` (2.4 GB), parsed to 1 Hz in `data/eval/real/dfl_*.npz` | **A**: main ground truth for train / validation / test, with simulated or transferred visibility |
| **SoccerNet-GSR** (Somers et al., CVPRW 2024; github.com/SoccerNet/sn-gamestate) | **Yes**: 30 s broadcast clips, 1080p, 25 fps | For players **in frame** only: pitch position, team side, role, jersey, clip-level track id | Inherent: labels are per frame | **No.** Off-screen players are unlabelled. | Track ids consistent within a clip; jersey labels when readable | Per-frame pitch-line and camera labels | Locally 115 clips (train 57 + valid 58) from **6 games** (2, 3, 4, 5, 6, 9), 18–21 clips each; full set 200 clips + challenge | Research use under the SoccerNet NDA / terms. Test and challenge splits were not downloaded and need the SoccerNet password; we did not try to work around that. | `data/sn_gsr/` (41 GB) | **B**: (1) detection, calibration and identity evaluation; (2) **real-broadcast re-entry test** for hidden players (true position where a player re-enters the frame) |
| **SkillCorner Open Data** (github.com/SkillCorner/opendata) | **No** video is distributed | Detected players: SkillCorner's CV measurement. Undetected players: SkillCorner's extrapolation, **not truth**. | n/a | 22 positions per frame, but only the detected ones are measurements | SkillCorner player ids and roles | Per-frame **camera footprint on the pitch** | 10 fps, **20** A-League 2024/25 matches locally (40 halves; README text says 10, `matches.json` lists 20) | MIT | `data/skillcorner/` (2 GB) | **B (proxy)**: (1) real camera behaviour transferred onto DFL truth (`sc_camera` mask); (2) **full-match real-broadcast re-entry test with proxy truth** |
| SkillCorner legacy EPL (2 matches, old open-data commit) | No | Detected players only | n/a | Partial | SkillCorner ids | Footprint | 10 fps, 2 matches | MIT | Already converted: `data/cv_skillcorner_epl_{2440,4039}_h{1,2}.json` | **B (proxy)**: reconstruction input for the app; no hidden truth |
| SoccerNet broadcast, Arsenal 3–4 Liverpool 2016 | **Yes**: 720p first half | **No** tracking ground truth. Has action labels, camera labels and CCBV calibration. | Event labels are video-timed | No | No (our CV, role-based identity ≈ 35 %) | `1_field_calib_ccbv.json` + our own camera | 45 min of video; app clip is 10 min (24:00–34:00) | SoccerNet NDA; footage must not be redistributed or shown publicly | `data/soccernet/` (824 MB) | **C** for accuracy (no truth). Video-only demo and qualitative review. |
| Metrica Sports sample data | No | Full tracking (anonymised) | n/a | Yes | Anonymised ids | n/a | 25 Hz, 2 matches locally (3 published) | Attribution requested | `data/metrica/` | B: earlier off-screen MLP. Not used this session because the pitch and coordinate conventions differ and there is no camera information. |
| Synthetic simulator (`simulator/`) | Synthetic "camera" | Exact | Exact | Yes | Exact | Exact | 1 Hz, unlimited seeds | Ours | Generated on demand | **Controlled tests only.** Kept separate from real results. |
| PFF FC / Gradient Sports FIFA World Cup 2022 | Video not distributed (FIFA-owned footage) | Tracking is **derived from the broadcast** (≈ 27 % of broadcast time untracked, per third-party write-ups) | Provider timestamps | Yes, but off-screen positions are estimates | PFF ids | Not public | 30 Hz, 64 matches | Free request form; terms unverified | Not downloaded | C: off-screen positions are not truth, and access is gated. Watch. |
| FOOTPASS (arXiv 2511.16183, 2025) | Broadcast video, 54 full matches | Play-by-play events + game-state variables; tracking is **single-player tracklets** | Aligned by the authors | No | Jersey / role annotations | Unclear | 54 matches | Download location and licence **not confirmed** | Not downloaded | C until access and licence are confirmed |
| WorldPose (ETH / FIFA, ECCV 2024) | Broadcast-camera footage under a **separate FIFA agreement** | Multi-view 3D poses; full-pitch coverage of all players not confirmed | Calibrated | Unclear | Yes, within sequences | Broadcast camera parameters | 88 sequences | Request form, academic use only, footage needs a FIFA agreement | Not requested | C here: gated, non-commercial. Strongest future candidate for real synchronised evaluation. |
| SoccerTrack v2 (2025) | **Not broadcast**: panoramic 4K of the full pitch | Yes, full pitch, all players | Yes | Yes | Yes | Panoramic | 10 university matches | CC BY 4.0 | Not downloaded | B for full-state training in a different domain (amateur, panoramic). Not used this session. |
| Bundesliga synchronised set ("Extraction of Positional Player Data from Broadcast Soccer Videos", arXiv 2110.11107) | Broadcast + 25 Hz positions, synced on the game clock | Yes | Yes (by the authors) | Yes | Yes | n/a | 4 halves | **Not publicly released** as far as could be verified | Not available | C: unavailable. The paper is evidence that this setup is the right one. |
| StatsBomb 360 | No video | Visible area + visible players **at events** only | n/a | No | Anonymous | Visible-area polygon | Event-level | Non-commercial licence | Not used | C |

## What "same match" means for each local source

- **DFL:** no broadcast video of these seven matches was found under a licence we can use. Nothing here aligns a DFL match with a video file, and none is claimed.
- **GSR:** truth and video are the same frames by construction, so no synchronisation is needed. The limitation is that truth exists only on screen.
- **SkillCorner:** detections and extrapolations come from the same CV system. "Truth" at a re-entry is SkillCorner's detection of that player, which is a measurement and not an independent reference.
- **Arsenal–Liverpool:** the video and our own CV outputs come from the same file. There is no reference tracking, so hidden-player accuracy on it **cannot** be computed.

## Recommended use (as implemented this session)

| Purpose | Data | Supervision type |
|---|---|---|
| Train and select hidden-player models | DFL, 7-fold by match | **Direct** positions, **proxy** visibility (SkillCorner footprints moved onto the DFL ball + nearest-k / random masks) |
| Held-out DFL test | DFL test match of each fold | Direct positions, proxy visibility |
| Real-broadcast hidden-player test (external) | SoccerNet-GSR re-entries, 6 games | **Direct** truth at re-entry, **real** visibility; inputs are labelled visible positions (oracle perception) |
| Full-match real-broadcast test (external) | SkillCorner A-League re-entries, 20 matches | **Proxy** truth (SkillCorner detections), **real** visibility |
| Detection / calibration / identity | SoccerNet-GSR (game-separated) | Direct, visible players only. Measured in earlier sessions (see `docs/reconstruction_evaluation.md` §2). |
| Video-only demo | Arsenal–Liverpool clip | None. Qualitative only. |
| Controlled tests | Simulator | Synthetic. Reported separately. |

## Data still needed

1. A licensed broadcast video of the **same** match as a full-pitch tracking file (for example a DFL / TRACAB match with its broadcast feed, or WorldPose footage under its FIFA agreement). This is the only way to score the whole pipeline (video → hidden players) with direct truth, and the only way to measure the error from synchronisation and calibration together with reconstruction error.
2. The GSR test split, which can be obtained under the SoccerNet NDA. It would add more held-out games for the re-entry and perception tests.
3. Player identities on broadcast video (jersey numbers). The SoccerNet jersey task and GSR jersey labels exist. Identity is the largest measured gap between oracle and real perception.

## Sources

- DFL / IDSSE: [Bassek et al. 2025, Sci Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/)
- SoccerNet-GSR: [CVPRW 2024 paper](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Somers_SoccerNet_Game_State_Reconstruction_End-to-End_Athlete_Tracking_and_Identification_on_CVPRW_2024_paper.html), [arXiv 2404.11335](https://arxiv.org/html/2404.11335v1)
- SkillCorner Open Data: github.com/SkillCorner/opendata (local copy with LICENSE)
- FOOTPASS: [arXiv 2511.16183](https://arxiv.org/pdf/2511.16183)
- WorldPose: [project page](https://eth-ait.github.io/WorldPoseDataset/), [ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2927_ECCV_2024_paper.php)
- PFF / Gradient Sports World Cup 2022: [dataset announcement](https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset)
- SoccerTrack v2: [project page](https://atomscott.github.io/SoccerTrack-v2/)
- Synchronised Bundesliga broadcast set: [arXiv 2110.11107](https://arxiv.org/pdf/2110.11107)
