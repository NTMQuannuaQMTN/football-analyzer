"""Run the remaining GPU experiments after the detector, within the session deadline."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--after-pid",type=int,required=True)
    ap.add_argument("--deadline",type=float,required=True)
    a=ap.parse_args()
    root=Path("cv/out/training_20261011")
    while time.time()<a.deadline:
        try:
            os.kill(a.after_pid,0)
        except ProcessLookupError:
            break
        time.sleep(30)
    rows=[]
    for module,hours in (("cv.train_pitch_lines",1.2),("cv.train_jersey_session",1.0)):
        remaining=(a.deadline-time.time())/3600
        budget=min(hours,remaining-.1)
        if budget<=.05:
            break
        log=root/f"{module.rsplit('.',1)[1]}.log"
        with log.open("w") as f:
            start=time.time()
            result=subprocess.run([sys.executable,"-u","-m",module,"--hours",str(budget)],stdout=f,stderr=subprocess.STDOUT)
        rows.append(dict(module=module,seconds=time.time()-start,exit_code=result.returncode,log=str(log)))
        (root/"queue_results.json").write_text(json.dumps(rows,indent=2))
        print(rows[-1],flush=True)


if __name__=="__main__":
    main()
