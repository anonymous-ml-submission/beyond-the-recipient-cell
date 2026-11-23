#!/usr/bin/env python3
"""Attach externally corrected counts to a prepared real sample after exact ID/gene matching."""
import argparse
from pathlib import Path
import anndata as ad, pandas as pd
from leakage_vso.io import read_mtx_bundle, write_h5ad_safe
p=argparse.ArgumentParser(); p.add_argument("--root",required=True); p.add_argument("--sample",required=True); p.add_argument("--clean-h5ad",default=None); p.add_argument("--split-dir",default=None); p.add_argument("--prefix",default="clean_counts"); a=p.parse_args()
d=Path(a.root)/"samples"/a.sample; raw=ad.read_h5ad(d/"raw.h5ad")
if a.clean_h5ad: clean=ad.read_h5ad(a.clean_h5ad)
elif a.split_dir: clean=read_mtx_bundle(a.split_dir,a.prefix)
else: raise ValueError("give --clean-h5ad or --split-dir")
missing_cells=raw.obs_names.difference(clean.obs_names); missing_genes=raw.var_names.difference(clean.var_names)
if len(missing_cells): raise RuntimeError(f"Corrected target missing {len(missing_cells)} raw cells; refusing implicit dropping")
if len(missing_genes): raise RuntimeError(f"Corrected target missing {len(missing_genes)} raw genes; refusing implicit dropping")
clean=clean[raw.obs_names,raw.var_names].copy(); write_h5ad_safe(clean,d/"clean.h5ad"); print(f"attached clean targets {clean.shape} to {a.sample}")
