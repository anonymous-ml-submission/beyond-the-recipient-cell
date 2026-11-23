#!/usr/bin/env python3
import argparse, anndata as ad, numpy as np
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--samples',nargs='+',required=True); a=p.parse_args()
for sid in a.samples:
    d=Path(a.root)/'samples'/sid; raw=ad.read_h5ad(d/'raw.h5ad'); cp=d/'clean.h5ad'; clean=ad.read_h5ad(cp) if cp.exists() else None
    def desc(name,x):
        X=x.X; vals=X.data if hasattr(X,'data') else np.asarray(X).ravel(); vals=np.asarray(vals,float); neg=int((vals<0).sum()); nonint=float(np.mean(np.abs(vals-np.round(vals))>1e-6)) if len(vals) else 0
        print(sid,name,'shape',x.shape,'min',float(vals.min()) if len(vals) else 0,'max',float(vals.max()) if len(vals) else 0,'negative',neg,'noninteger_fraction_nonzero',nonint)
        if neg: raise RuntimeError(f'{sid} {name} contains negative values; incompatible with nonnegative expression model')
    desc('raw',raw)
    if clean is not None:
        if not raw.obs_names.equals(clean.obs_names) or not raw.var_names.equals(clean.var_names): raise RuntimeError(f'{sid}: raw/clean axes differ')
        desc('clean',clean)
