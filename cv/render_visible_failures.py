"""Private scored GSR contact sheets, with reference positions evaluation-only."""
from html import escape
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils.gmc import GMC

from cv.audit_tracking import FixedMotion
from cv.evaluate_session import iou_matrix, matched_pairs
from cv.gsr import GSR, load_clip, project


def main():
    root = Path('cv/out/audit8_20261011')
    output = root/'visible_failures'
    output.mkdir(parents=True, exist_ok=True)
    score = json.loads((root/'visible_pipeline/test.json').read_text())
    variant = 'selected/learned_propagated/kalman'
    grouped = {}
    for row in score['rows']:
        if row['variant'] == variant:
            grouped.setdefault((row['clip'], Path(row['image']).stem), []).append(row['error_m'])
    ordered = sorted(grouped, key=lambda key: np.mean(grouped[key]), reverse=True)
    chosen = []
    clips_used = set()
    for key in ordered:
        if key[0] not in clips_used:
            chosen.append(key)
            clips_used.add(key[0])
        if len(chosen) == 8:
            break
    chosen += [key for key in ordered[-2:] if key not in chosen]
    camera_source = json.loads((root/'camera_motion/test.json').read_text())
    cameras = {(r['clip'], Path(r['image']).stem): r for r in camera_source['cases'] if r['configuration'] == 'selected'}
    configuration = json.loads((root/'tracking_generic/selection.json').read_text())['configuration']
    cache = next((root/'tracking_generic/cache').glob(f'{score["model_sha256"][:16]}_cpu_*_960_c015_i05'))
    records = []
    cv2.setNumThreads(1)
    for clip in sorted({key[0] for key in chosen}):
        _, frames = load_clip(GSR/'valid'/clip)
        tracker = BOTSORT(SimpleNamespace(**configuration))
        tracker.gmc = FixedMotion()
        motion = GMC(method='sparseOptFlow')
        cv2.setRNGSeed(11026)
        for frame in frames[::5]:
            image = cv2.imread(str(frame['file']))
            raw = np.load(cache/f'{clip}_{frame["file"].stem}.npy')
            tracker.gmc.warp = motion.apply(image)
            tracks = tracker.update(Boxes(raw, image.shape[:2]).numpy(), image)
            key = clip, frame['file'].stem
            if key not in chosen:
                continue
            camera = cameras[key]
            truth = [p for p in frame['people'] if p['role'] in ('player', 'goalkeeper', 'referee')]
            boxes = tracks[:, :4] if len(tracks) else np.empty((0, 4))
            pairs = matched_pairs(iou_matrix(boxes, [p['box'] for p in truth]))
            canvas = np.full((680, 1460, 3), 24, np.uint8)
            displayed = image.copy()
            for index, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(displayed, (x1, y1), (x2, y2), (230, 230, 230), 2)
                cv2.circle(displayed, ((x1+x2)//2, y2), 5, (20, 70, 240), -1)
                cv2.putText(displayed, str(int(tracks[index, 4])), (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, .5, (230, 230, 230), 1)
            canvas[90:630, :960] = cv2.resize(displayed, (960, 540))
            def point(p):
                return int(985+4.1*p[0]), int(150+4.1*p[1])
            cv2.rectangle(canvas, point([0, 0]), point([105, 68]), (36, 86, 39), -1)
            cv2.rectangle(canvas, point([0, 0]), point([105, 68]), (175, 190, 175), 1)
            cv2.line(canvas, point([52.5, 0]), point([52.5, 68]), (175, 190, 175), 1)
            cv2.circle(canvas, point([52.5, 34]), int(9.15*4.1), (175, 190, 175), 1)
            for x in (0, 88.5):
                cv2.rectangle(canvas, point([x, 13.84]), point([x+16.5, 54.16]), (175, 190, 175), 1)
            errors = []
            for index, truth_index in pairs:
                p = truth[truth_index]
                if p['pitch'] is None:
                    continue
                box = boxes[index]
                predicted = project(np.array(camera['H']), [[(box[0]+box[2])/2/1920, box[3]/1080]])[0]
                actual = np.asarray(p['pitch'])
                if not np.isfinite(predicted).all():
                    continue
                error = float(np.linalg.norm(predicted-actual))
                errors.append(error)
                a = point(np.clip(actual, [0, 0], [105, 68]))
                b = point(np.clip(predicted, [0, 0], [105, 68]))
                cv2.line(canvas, a, b, (150, 150, 150), 1)
                cv2.rectangle(canvas, (a[0]-4, a[1]-4), (a[0]+4, a[1]+4), (40, 240, 140), -1)
                cv2.circle(canvas, b, 5, (20, 70, 240), 1)
                cv2.putText(canvas, f'{int(tracks[index, 4])}: {error:.1f}m', (b[0]+5, b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, .28, (200, 200, 200), 1)
            mean = float(np.mean(errors)) if errors else None
            cv2.putText(canvas, f'HELD-OUT GSR | {clip}/{frame["file"].stem} | matched supported mean {mean:.2f}m', (15, 32), cv2.FONT_HERSHEY_SIMPLEX, .63, (230, 230, 230), 1)
            cv2.putText(canvas, 'Labelled visible references score only; no full hidden-player truth or verified shirt names', (15, 62), cv2.FONT_HERSHEY_SIMPLEX, .53, (230, 230, 230), 1)
            cv2.putText(canvas, 'Green squares: reference pitch positions', (980, 490), cv2.FONT_HERSHEY_SIMPLEX, .38, (40, 240, 140), 1)
            cv2.putText(canvas, 'Red rings: predicted feet + estimated camera', (980, 515), cv2.FONT_HERSHEY_SIMPLEX, .38, (100, 140, 240), 1)
            cv2.putText(canvas, 'Out-of-pitch points touch edge for drawing only', (980, 540), cv2.FONT_HERSHEY_SIMPLEX, .35, (200, 200, 200), 1)
            filename = f'{clip}_{frame["file"].stem}.jpg'
            cv2.imwrite(str(output/filename), canvas)
            records.append(dict(clip=clip, frame=frame['file'].stem, image=filename, matched_people=len(errors), mean_m=mean,
                                expected_mean_m=float(np.mean(grouped[key])), camera_age_s=camera['anchor_age_s']))
            assert np.isclose(mean, np.mean(grouped[key]), atol=1e-5), 'Rendered pairs differ from scored pipeline'
        print('Rendered', clip, flush=True)
    html_rows = ''.join('<tr>'+''.join('<td>'+escape(str(r[k]))+'</td>' for k in ['clip', 'frame', 'matched_people', 'mean_m', 'camera_age_s'])+'</tr>' for r in records)
    images = ''.join(f'<h2>{escape(r["clip"])}/{escape(r["frame"])}</h2><img style="width:100%" src="{escape(r["image"])}">' for r in records)
    (output/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Scored visible-player failure review</title><style>body{font:16px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style><h1>Scored visible-player camera/tracking review</h1><p>Eight high-error frames from distinct clips and two low-error controls. Selection is diagnostic after frozen evaluation, not model selection. References are evaluation-only; no match inference reads them.</p><table><thead><tr><th>Clip</th><th>Frame</th><th>Matched people</th><th>Mean error (m)</th><th>Camera age (s)</th></tr></thead><tbody>'+html_rows+'</tbody></table>'+images+'</html>')
    (output/'render_manifest.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
