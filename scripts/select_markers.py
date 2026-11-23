#!/usr/bin/env python3
import argparse, pandas as pd, anndata as ad
from pathlib import Path
from leakage_vso.markers import select_exclusive_markers
p=argparse.ArgumentParser(); p.add_argument("--root",required=True); p.add_argument("--sample",required=True); p.add_argument("--cell-type-col",default="first_type"); p.add_argument("--out",required=True); p.add_argument("--top-k",type=int,default=10); a=p.parse_args()
d=Path(a.root)/"samples"/a.sample; c=pd.read_parquet(d/"cells.parquet"); clean=ad.read_h5ad(d/"clean.h5ad")
if a.cell_type_col not in c: raise RuntimeError(f"Missing {a.cell_type_col}; run/retain RCTD/SPLIT metadata")
res=select_exclusive_markers(clean,c[a.cell_type_col],top_k=a.top_k); Path(a.out).parent.mkdir(parents=True,exist_ok=True); res.to_csv(a.out,index=False); print(res.to_string(index=False))
