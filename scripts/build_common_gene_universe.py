#!/usr/bin/env python3
"""Build the assay-safe gene universe shared across all paper samples."""
import argparse
from pathlib import Path
import anndata as ad

p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--samples',nargs='+',required=True); p.add_argument('--out',required=True); a=p.parse_args()
sets=[]
for sid in a.samples:
    q=Path(a.root)/'samples'/sid/'raw.h5ad'
    if not q.exists(): raise FileNotFoundError(q)
    x=ad.read_h5ad(q,backed='r'); sets.append(set(map(str,x.var_names))); x.file.close()
common=set.intersection(*sets) if sets else set()
if not common: raise RuntimeError('No common genes across requested samples')
Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text('\n'.join(sorted(common))+'\n')
print(f'{len(common)} common genes across {len(sets)} samples')
