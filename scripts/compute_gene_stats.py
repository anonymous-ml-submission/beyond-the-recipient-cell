#!/usr/bin/env python3
"""Compute TRAIN-DONOR gene abundance/prevalence statistics for matched leakage controls."""
import argparse, pandas as pd, anndata as ad, numpy as np, scipy.sparse as sp
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--root",required=True); p.add_argument("--samples",nargs="+",required=True); p.add_argument("--out",required=True); p.add_argument("--split",default="train"); p.add_argument("--genes",default=None,help="Optional common-gene universe text file"); a=p.parse_args()
requested=set(x.strip() for x in Path(a.genes).read_text().splitlines() if x.strip()) if a.genes else None
sums={}; detected={}; ncell=0
for sid in a.samples:
    d=Path(a.root)/"samples"/sid; cells=pd.read_parquet(d/"cells.parquet"); x=ad.read_h5ad(d/"raw.h5ad")
    if "split" not in cells: raise RuntimeError(f"{sid}: missing split")
    ix=np.where(cells.split.astype(str).to_numpy()==str(a.split))[0]
    if not len(ix): continue
    X=x.X[ix]; ss=np.asarray(X.sum(0)).ravel(); dd=np.asarray((X>0).sum(0)).ravel() if sp.issparse(X) else (X>0).sum(0)
    for g,s,v in zip(x.var_names,ss,dd):
        if requested is not None and str(g) not in requested: continue
        sums[g]=sums.get(g,0.0)+float(s); detected[g]=detected.get(g,0)+int(v)
    ncell+=len(ix)
if ncell==0: raise RuntimeError("No training cells found")
out=pd.DataFrame({"gene":list(sums),"mean_count":[sums[g]/ncell for g in sums],"prevalence":[detected[g]/ncell for g in sums]})
out["log_mean_count"]=np.log1p(out.mean_count); out.to_csv(a.out,index=False); print(out.describe().to_string())
