"""Frozen whole-game retrieval evaluation; not named-player verification."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cv.person_embedding import PersonEmbedding
from cv.train_person_embedding import predict, retrieval, cross_clip_retrieval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['val', 'test'], required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    root = Path('cv/out/audit8_20261011/person_embedding')
    checkpoint = root / 'best.pt'
    metadata = json.loads((root / 'metadata.json').read_text())
    with np.load(Path('cv/out/training_20261011/jersey/data') / f'{args.part}.npz') as data:
        images, tracks = data['X'], data['track']
    # Same fixed per-track sample as validation checkpoint selection.
    indices = []
    for track in sorted(set(tracks.tolist())):
        candidates = np.flatnonzero(tracks == track)
        indices.extend(candidates[np.linspace(0, len(candidates)-1, min(12, len(candidates)), dtype=int)])
    indices = np.asarray(sorted(indices))
    images, tracks = images[indices], tracks[indices]
    torch.set_num_threads(1)
    torch.manual_seed(11026)
    trained = PersonEmbedding().to(args.device)
    trained.load_state_dict(torch.load(checkpoint, weights_only=False)['state'])
    torch.manual_seed(11026)
    baseline = PersonEmbedding().to(args.device)
    cached = Path('/Users/qnrj/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
    state = {k: v for k, v in torch.load(cached, weights_only=True).items() if not k.startswith('fc.')}
    baseline.encoder.load_state_dict(state, strict=False)
    results = {}
    for name, model in [('local_pretrained_random_projection', baseline), ('validation_selected', trained)]:
        vectors = predict(model, images, args.device)
        results[name] = dict(same_track=retrieval(vectors, tracks, metadata), cross_clip=cross_clip_retrieval(vectors, tracks, metadata))
    output = dict(part=args.part, crops=len(images), tracks=len(set(tracks)), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  results=results, conditions='Whole-game held-out split. Retrospective labelled-crop retrieval within team; not causal association, shirt-number recognition, hidden-player accuracy or verified identities.')
    (root / f'{args.part}_retrieval.json').write_text(json.dumps(output, indent=2))
    print(output, flush=True)


if __name__ == '__main__':
    main()
