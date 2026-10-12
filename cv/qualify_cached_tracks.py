"""Requalify existing private detections without rerunning the detector."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from cv.track_namespace import qualify_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    source, output = Path(args.run_dir), Path(args.out)
    if source.resolve() == output.resolve():
        raise ValueError('Use a separate output directory to preserve the source cache')
    rows = [json.loads(line) for line in (source/'detections.jsonl').open()]
    qualified = qualify_rows(rows)
    old_shots = defaultdict(set)
    for row in qualified:
        for detection in row['dets']:
            if detection['cls'] == 'person' and detection['id'] >= 0:
                old_shots[detection['native_id']].add(detection['shot_id'])
    output.mkdir(parents=True, exist_ok=True)
    with (output/'detections.jsonl').open('w') as stream:
        for row in qualified:
            stream.write(json.dumps(row)+'\n')
    metadata = json.loads((source/'run.json').read_text())
    metadata.update(track_id_namespace='shot-qualified-v1', source_run=str(source),
                    source_detections_sha256=hashlib.sha256((source/'detections.jsonl').read_bytes()).hexdigest(),
                    namespace_audit=dict(native_ids=len(old_shots), reused_native_ids=sum(len(shots)>1 for shots in old_shots.values()),
                                         shot_id_pairs=sum(len(shots) for shots in old_shots.values())),
                    namespace_condition='Same cached measurements; reset-local IDs are distinct across camera gaps. Cross-shot association remains unverified and is handled separately.')
    (output/'run.json').write_text(json.dumps(metadata, indent=2))
    print(metadata['namespace_audit'], flush=True)


if __name__ == '__main__':
    main()
