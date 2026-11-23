#!/usr/bin/env python3
"""Estimate gene leakage ONLY from designated training donors/samples.

Fits each sample independently (spatial neighborhoods never cross sections), then reports
robust median parameters across training samples. Never estimate gamma on test donors.
"""
import argparse, pandas as pd, anndata as ad, numpy as np
from pathlib import Path
from leakage_vso.leakage import estimate_leakage_table
p=argparse.ArgumentParser(); p.add_argument("--root",required=True); p.add_argument("--samples",nargs="+",required=True); p.add_argument("--out",required=True); p.add_argument("--radius-um",type=float,default=40); p.add_argument("--genes",default=None); p.add_argument("--split",default="train")
a=p.parse_args(); allfits=[]
requested=None
if a.genes: requested=[g.strip() for g in Path(a.genes).read_text().splitlines() if g.strip()]
for sid in a.samples:
    d=Path(a.root)/"samples"/sid; cells=pd.read_parquet(d/"cells.parquet"); raw=ad.read_h5ad(d/"raw.h5ad"); cleanp=d/"clean.h5ad"
    if not cleanp.exists(): raise RuntimeError(f"{sid}: leakage fit requires clean.h5ad")
    clean=ad.read_h5ad(cleanp)
    if not raw.var_names.equals(clean.var_names): raise RuntimeError(f"{sid}: raw/clean gene axis mismatch")
    if "split" not in cells: raise RuntimeError(f"{sid}: missing split column")
    mask=cells.split.astype(str).to_numpy()==str(a.split)
    if not mask.any(): continue
    genes=raw.var_names.to_list(); genes=[g for g in genes if requested is None or g in requested]
    ix=np.where(mask)[0]; gi=[raw.var_names.get_loc(g) for g in genes]
    t=estimate_leakage_table(cells.loc[mask,["x_um","y_um"]].to_numpy(),raw.X[ix][:,gi],clean.X[ix][:,gi],genes,radius_um=a.radius_um)
    if len(t): t["sample_id"]=sid; allfits.append(t)
if not allfits: raise RuntimeError(f"No leakage fits produced for split={a.split}")
per=pd.concat(allfits,ignore_index=True); per.to_csv(Path(a.out).with_suffix(".per_sample.csv"),index=False)
agg=per.groupby("gene",as_index=False).agg(gamma=("gamma","median"),beta=("beta","median"),r2=("r2","median"),n_samples=("sample_id","nunique"),n_cells=("n","sum"))
agg=agg.sort_values(["r2","gamma"],ascending=False); Path(a.out).parent.mkdir(parents=True,exist_ok=True); agg.to_csv(a.out,index=False)
print(agg.head(20).to_string(index=False))
