#!/usr/bin/env python3
import argparse, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--runs",required=True); p.add_argument("--root",required=True); p.add_argument("--samples",nargs="+",required=True); p.add_argument("--split",default="test"); p.add_argument("--device",default="auto"); a=p.parse_args()
for rd in sorted(Path(a.runs).iterdir()):
    ck=rd/"train"/"best.pt"
    if not ck.exists(): continue
    ev=rd/"eval"; subprocess.run([sys.executable,"scripts/evaluate.py","--checkpoint",str(ck),"--root",a.root,"--samples",*a.samples,"--out",str(ev),"--split",a.split,"--device",a.device],check=True)
