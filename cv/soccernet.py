"""SoccerNet access: download, labels, camera segments and per-frame pitch calibration.

Calibration (`{half}_field_calib_ccbv.json`): one 3x3 homography every 0.5 s mapping pitch
metres (origin = centre spot, x along the length, +y towards the main camera) to 720p pixels.
We invert it to map image points (players' feet) onto the pitch.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "soccernet"
DEFAULT_GAME = "england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool"
CALIB_FPS = 2.0
L, W = 105.0, 68.0


def game_dir(game: str) -> Path:
    return DATA / game


def list_games(prefix: str = "england_epl"):
    from SoccerNet.utils import getListGames
    return [g for g in getListGames("all") if g.startswith(prefix)]


def split_of(game: str) -> str:
    from SoccerNet.utils import getListGames
    for spl in ("train", "valid", "test", "challenge"):
        if game in getListGames(spl):
            return spl
    return "train"


def download(game: str, half: int, video: bool = True):
    """Labels + calibration are public; the 720p video needs `hf auth login` + accepted NDA."""
    from SoccerNet.Downloader import SoccerNetDownloader
    d = SoccerNetDownloader(LocalDirectory=str(DATA))
    files = ["Labels-v2.json", "Labels-cameras.json", f"{half}_field_calib_ccbv.json"]
    if video:
        files.append(f"{half}_720p.mkv")
    d.downloadGame(game=game, files=files, spl=split_of(game))
    return game_dir(game)


def _ms(a) -> float:
    return int(a["position"]) / 1000.0


def labels(game: str, half: int):
    d = json.load(open(game_dir(game) / "Labels-v2.json"))
    out = [{**a, "t": _ms(a)} for a in d["annotations"] if a["gameTime"].startswith(f"{half} ")]
    return d, sorted(out, key=lambda a: a["t"])


def camera_segments(game: str, half: int):
    """Shot labels apply BEFORE each boundary (last frame before abrupt cuts).

    SoccerNet-v2 camera annotations name the shot that has been shown. Treating
    them as starts shifts replays into live play. The unlabelled final tail is
    intentionally not assigned the preceding shot's label.
    """
    d = json.load(open(game_dir(game) / "Labels-cameras.json"))
    ann = sorted([a for a in d["annotations"] if a["gameTime"].startswith(f"{half} ")], key=_ms)
    start = 0.0
    out = []
    for a in ann:
        end = _ms(a)
        if end > start:
            out.append((start, end, a))
        start = max(start,end)
    return out


def main_camera_segments(game: str, half: int):
    """[(start, end)] seconds of annotated real-time main-camera play."""
    segs = []
    for start, end, a in camera_segments(game, half):
        if a["label"].startswith("Main camera") and a.get("replay", "real-time") == "real-time":
            segs.append((start, end))
    return segs


class Calibration:
    def __init__(self, game: str, half: int, min_conf: float = 0.5):
        d = json.load(open(game_dir(game) / f"{half}_field_calib_ccbv.json"))
        self.H = []      # image -> pitch (metres, our 0..105 x 0..68 frame) per calibration frame
        self.conf = []
        shift = np.array([[1, 0, L / 2], [0, 1, W / 2], [0, 0, 1]])
        for p in d["predictions"]:
            if not p:
                self.H.append(None)
                self.conf.append(0.0)
                continue
            h = np.array(p[0]["homography"], dtype=float).reshape(3, 3)
            try:
                self.H.append(shift @ np.linalg.inv(h))
            except np.linalg.LinAlgError:
                self.H.append(None)
            self.conf.append(float(p[0].get("confidence", 1.0)))
        self.min_conf = min_conf
        self.main_segments = segments_cached(game, half)

    def at(self, t: float,causal=False,allow_cross_shot=False):
        if not np.isfinite(t):return None
        # Default remains the historical retrospective nearest sample.
        # A caller explicitly requesting causal sampling must use the past.
        i = int(np.floor(t * CALIB_FPS)) if causal else int(round(t * CALIB_FPS))
        # The closest timestamp can be a replay/close-up at a shot boundary.
        # Such a camera cannot project measurements from the current shot.
        # Explicit legacy sampling is retained for reproducible audits only.
        if not allow_cross_shot and hasattr(self, 'main_segments'):
            interval = next(((a, b) for a, b in self.main_segments if a <= t < b), None)
            if interval is None or not interval[0] <= i / CALIB_FPS < interval[1]:
                return None
        if 0 <= i < len(self.H) and self.H[i] is not None and self.conf[i] >= self.min_conf:
            return self.H[i]
        return None

    @staticmethod
    def project(H, pts):
        """pts: (n, 2) pixel coords at 1280x720 -> (n, 2) pitch metres."""
        pts = np.asarray(pts, dtype=float)
        hp = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
        return hp[:, :2] / hp[:, 2:3]


@lru_cache(maxsize=4)
def segments_cached(game, half):
    return tuple(main_camera_segments(game, half))


def on_main_camera(game, half, t) -> bool:
    return any(a <= t < b for a, b in segments_cached(game, half))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Download one SoccerNet game half (labels, calibration, 720p video)")
    ap.add_argument("--game", default=DEFAULT_GAME)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--list", action="store_true", help="list EPL games")
    a = ap.parse_args()
    if a.list:
        print("\n".join(list_games()))
    else:
        print(download(a.game, a.half, video=not a.no_video))
        print(os.listdir(game_dir(a.game)))
