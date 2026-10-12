"""Isolate colour-team clustering from detection and named identity accuracy.

Reference team labels only score a two-way cluster permutation. The explicit
oracle-exclusion control uses role labels and is not an inference candidate.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import ultralytics

from cv.build_match import color_feat, kmeans
from cv.detect_track import shirt_color
from cv.evaluate_session import iou_matrix, matched_pairs
from cv.gsr import GSR, load_clip


def score(features, labels, outfield_indices):
    clusters, _ = kmeans(np.asarray(features), 2, seed=11026)
    target = np.array([labels[i] == 'right' for i in outfield_indices], int)
    predicted = clusters[outfield_indices]
    correct = max(int((predicted == target).sum()), int((1-predicted == target).sum()))
    return dict(n=len(target), correct=correct, fraction=correct/max(1, len(target)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['val', 'test'], required=True)
    parser.add_argument('--model', default='cv/weights/yolo11m.pt')
    parser.add_argument('--tracking-root', default='cv/out/audit8_20261011/tracking_generic')
    parser.add_argument('--out', default='cv/out/audit8_20261011/shirt_colours')
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    rows = json.loads(Path('cv/out/training_20261011/pitch_lines/data/rows.json').read_text())[args.part]
    from ultralytics import YOLO
    model = YOLO(args.model)
    digest = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    raw_root = Path(args.tracking_root)/'cache'/f'{digest[:16]}_cpu_{ultralytics.__version__}_960_c015_i05'
    grouped = {}
    missing_raw = []
    confusion = Counter()
    role_totals = Counter()
    predicted_role_totals = Counter()
    for clip in sorted({row['clip'] for row in rows}):
        _, frames = load_clip(GSR/'valid'/clip)
        lookup = {f['file'].stem: f for f in frames}
        features, labels, predicted_classes = [], [], []
        roles = Counter()
        cv2.setRNGSeed(11026)
        for row in [r for r in rows if r['clip'] == clip]:
            stem = Path(row['image']).stem.removeprefix(clip+'_')
            cache = raw_root/f'{clip}_{stem}.npy'
            if not cache.is_file():
                missing_raw.append(str(cache))
                continue
            frame = lookup[stem]
            image = cv2.imread(str(frame['file']))
            raw = np.load(cache)
            truth = [p for p in frame['people'] if p['role'] in ('player', 'goalkeeper', 'referee')]
            role_totals.update(p['role'] for p in truth)
            predicted_role_totals.update(model.names[int(b[5])] for b in raw)
            pairs = {a: b for a, b in matched_pairs(iou_matrix(raw[:, :4], [p['box'] for p in truth]))}
            for index, truth_index in pairs.items():
                confusion[(truth[truth_index]['role'], model.names[int(raw[index, 5])])] += 1
            for index, box in enumerate(raw):
                feat = color_feat(dict(colors=shirt_color(image, box[:4])))
                if feat is None:
                    roles['no_colour'] += 1
                    continue
                person = truth[pairs[index]] if index in pairs else None
                label = person['team'] if person is not None and person['role'] == 'player' else None
                features.append(feat)
                labels.append(label)
                predicted_classes.append(model.names[int(box[5])])
                roles[person['role'] if person else 'unmatched_prediction'] += 1
        outfield = [i for i, label in enumerate(labels) if label in ('left', 'right')]
        if len(outfield) >= 6 and len({labels[i] for i in outfield}) == 2:
            generic = score(features, labels, outfield)
            oracle = score([features[i] for i in outfield], [labels[i] for i in outfield], list(range(len(outfield))))
            grouped[clip] = dict(all_person_clustering=generic, oracle_outfield_exclusion=oracle, colour_samples=dict(roles))
            predicted_outfield = [i for i, name in enumerate(predicted_classes) if name == 'player']
            common = [i for i in outfield if i in predicted_outfield]
            if len(common) >= 6 and len({labels[i] for i in common}) == 2:
                remap = {old: new for new, old in enumerate(predicted_outfield)}
                grouped[clip]['predicted_player_only'] = score([features[i] for i in predicted_outfield], [labels[i] for i in predicted_outfield], [remap[i] for i in common])
                grouped[clip]['all_person_same_player_subset'] = score(features, labels, common)
        print(args.part, clip, len(outfield), flush=True)
    summary = {}
    for name in ['all_person_clustering', 'oracle_outfield_exclusion', 'predicted_player_only', 'all_person_same_player_subset']:
        count = sum(r[name]['n'] for r in grouped.values() if name in r)
        correct = sum(r[name]['correct'] for r in grouped.values() if name in r)
        summary[name] = dict(n=count, correct=correct, fraction=correct/max(1, count))
    result = dict(part=args.part, clips=grouped, summary=summary, missing_raw_files=missing_raw, model=str(Path(args.model)), model_sha256=digest, seed=11026,
                  true_role_totals=dict(role_totals), predicted_role_totals=dict(predicted_role_totals), role_confusion={f'{a}->{b}':n for (a,b),n in confusion.items()},
                  conditions='Fixed camera-evaluation timestamps, same cached stock YOLO11m boxes. IoU>=0.5 matches score outfield colour labels; best two-way team permutation uses labels for evaluation only. No side/orientation assignment, track voting, name accuracy or hidden-player score. All-person clustering includes referee, goalkeeper and unmatched crops. Oracle role exclusion is a label-aware diagnostic, not deployable. No training or parameter selection.')
    (root/f'{args.part}.json').write_text(json.dumps(result, indent=2))
    print(summary, 'missing raw:', len(missing_raw), flush=True)


if __name__ == '__main__':
    main()
