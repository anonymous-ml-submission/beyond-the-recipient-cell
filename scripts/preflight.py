#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from leakage_vso.preflight import validate_sample
p=argparse.ArgumentParser(); p.add_argument("--root",required=True); p.add_argument("--sample",action="append",required=True); p.add_argument("--min-in-bounds",type=float,default=.98); a=p.parse_args()
for s in a.sample:
    r=validate_sample(Path(a.root)/"samples"/s,a.min_in_bounds); print(s,json.dumps(r,indent=2))
