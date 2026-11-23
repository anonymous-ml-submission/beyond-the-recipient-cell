from __future__ import annotations
import numpy as np, pandas as pd
from scipy.spatial import cKDTree


def nearest_source_distance(cells: pd.DataFrame, source_mask: np.ndarray) -> np.ndarray:
    xy=cells[["x_um","y_um"]].to_numpy(float); src=xy[np.asarray(source_mask,bool)]
    if len(src)==0: return np.full(len(cells),np.nan)
    return cKDTree(src).query(xy,k=1)[0]


def binned_halo(dist,values,bins):
    dist=np.asarray(dist,float); values=np.asarray(values,float); bins=np.asarray(bins,float)
    rows=[]
    for a,b in zip(bins[:-1],bins[1:]):
        m=(dist>=a)&(dist<b)&np.isfinite(values)
        rows.append({"d_lo":a,"d_hi":b,"d_mid":0.5*(a+b),"n":int(m.sum()),"mean":float(np.mean(values[m])) if m.any() else np.nan,"sem":float(np.std(values[m],ddof=1)/np.sqrt(m.sum())) if m.sum()>1 else np.nan})
    return pd.DataFrame(rows)
