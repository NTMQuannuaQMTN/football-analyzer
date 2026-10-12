"""Batched validation threshold selection, followed by one held-out jersey test."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cv import gsr_train as g


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='mps');a=ap.parse_args()
    root=Path('cv/out/training_20261011/jersey');g.JERSEY_DIR=root/'data'
    torch.set_num_threads(1);model=g.DigitNet.build().to(a.device)
    model.load_state_dict(torch.load(root/'best.pt',weights_only=False));model.eval()
    with np.load(root/'data/val.npz') as data:
        probabilities=g.predict_jersey(model,data['X'],a.device,bs=256)
        validation={str(t):g.jersey_metrics(probabilities,data['y'],data['track'],t) for t in [.5,.7,.85,.95]}
    print('validation',validation,flush=True)
    eligible=[float(t) for t,v in validation.items() if v['track_precision']>=.9]
    threshold=max(eligible,key=lambda t:validation[str(t)]['track_coverage']) if eligible else .95
    test=g.eval_jersey(model,'test',a.device,conf=threshold)
    history=json.loads((root/'history.json').read_text())
    report=dict(train_games=['4','6','9'],val_games=['3','5'],test_games=['2'],completed_epochs=len(history),
        validation=validation,selected_threshold=threshold,test=test,
        stopped_early='Four training epochs; shared GPU prioritised for detector. Initial CPU audit interrupted and replaced by one batched pass per dataset.',
        note='Track voting is retrospective. No legibility labels for front views. Not deployed; high-threshold precision must be read alongside decision count and coverage.')
    # NumPy comparison counts are converted to Python integers for JSON.
    (root/'results.json').write_text(json.dumps(report,indent=2,default=lambda x:int(x)))
    print(report,flush=True)


if __name__=='__main__':main()
