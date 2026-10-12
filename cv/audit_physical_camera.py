"""Image-only physical camera evaluation with scoring-only reference positions."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from cv.physical_camera import PhysicalCameraNet, project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['val', 'test'], required=True)
    parser.add_argument('--checkpoint', default='cv/out/audit8_20261011/physical_camera/best.pt')
    parser.add_argument('--out', default='cv/out/audit8_20261011/physical_camera/evaluation')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    net = PhysicalCameraNet().to(args.device)
    net.load_state_dict(torch.load(checkpoint, weights_only=False)['state'])
    net.eval()
    source = Path('cv/out/training_20261011/pitch_lines/data/rows.json')
    rows = json.loads(source.read_text())[args.part]
    cases, errors = [], []
    for index, row in enumerate(rows):
        # Inference reads the image only; labels are read after its output.
        image = cv2.cvtColor(cv2.imread(row['image']), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()[None] / 255
        with torch.no_grad():
            matrix, pose = net(tensor.to(args.device))
            inverse = torch.linalg.inv(matrix)
        people = [p for p in row['people'] if p['pitch'] is not None and p['role'] in ('player', 'goalkeeper', 'referee')]
        result = dict(H=inverse[0].cpu().numpy().tolist(), pose=pose[0].cpu().numpy().tolist())
        if people:
            feet = torch.tensor([[[(p['box'][0]+p['box'][2])/2/1920, p['box'][3]/1080] for p in people]], device=args.device)
            predicted = project(inverse, feet)[0].cpu().numpy()
            values = np.linalg.norm(predicted-np.asarray([p['pitch'] for p in people]), axis=1)
            if np.isfinite(values).all():
                errors.extend(values.tolist())
                result.update(errors_m=values.tolist(), mean_m=float(values.mean()), median_m=float(np.median(values)))
        cases.append(dict(image=row['image'], clip=row['clip'], people=len(people), results=dict(learned_physical=result)))
        if index % 100 == 0:
            print(args.part, index, len(rows), flush=True)
    output = dict(part=args.part, checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  rows_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), cases=cases,
                  summary=dict(learned_physical=dict(frames=len(rows), supported_frames=sum('errors_m' in c['results']['learned_physical'] for c in cases),
                               people=len(errors), mean_m=float(np.mean(errors)) if errors else None, median_m=float(np.median(errors)) if errors else None,
                               p90_m=float(np.percentile(errors, 90)) if errors else None)),
                  conditions='Image-only bounded main-sideline physical camera. Reference visible feet/pitch coordinates score only. No detector, named identity or hidden-player score. Ungated error and gated coverage must be reported separately.')
    (root / f'{args.part}.json').write_text(json.dumps(output, indent=2))
    print(output['summary'], flush=True)


if __name__ == '__main__':
    main()
