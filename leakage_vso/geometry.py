from __future__ import annotations
import numpy as np
import pandas as pd


def polygon_area_xy(xy: np.ndarray) -> float:
    xy=np.asarray(xy,dtype=float)
    if len(xy)<3: return 0.0
    x=xy[:,0]; y=xy[:,1]
    return float(abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1)))/2.0)


def polygon_centroid_xy(xy: np.ndarray) -> tuple[float,float]:
    """Area-weighted polygon centroid; robust mean fallback for degenerate polygons."""
    xy=np.asarray(xy,dtype=float)
    if len(xy)<3:
        m=np.nanmean(xy,axis=0); return float(m[0]),float(m[1])
    x=xy[:,0]; y=xy[:,1]; xn=np.roll(x,-1); yn=np.roll(y,-1)
    cross=x*yn-xn*y; a=cross.sum()/2.0
    if not np.isfinite(a) or abs(a)<1e-9:
        m=np.nanmean(xy,axis=0); return float(m[0]),float(m[1])
    cx=((x+xn)*cross).sum()/(6*a); cy=((y+yn)*cross).sum()/(6*a)
    return float(cx),float(cy)


def sort_polygon_vertices(q: pd.DataFrame) -> pd.DataFrame:
    if "vertex_order" in q.columns:
        return q.sort_values("vertex_order",kind="stable")
    return q


def polygon_summary(poly: pd.DataFrame) -> pd.DataFrame:
    """Standardized long polygon table -> cell centroids and polygon areas in pixel units."""
    req={"cell_id","x_px","y_px"}
    if not req.issubset(poly.columns):
        raise ValueError(f"polygon table requires {sorted(req)}")
    rows=[]
    p=poly.copy(); p["cell_id"]=p.cell_id.astype(str)
    for cid,q in p.groupby("cell_id",sort=False):
        q=sort_polygon_vertices(q); xy=q[["x_px","y_px"]].to_numpy(float)
        xy=xy[np.isfinite(xy).all(axis=1)]
        if len(xy)<3: continue
        cx,cy=polygon_centroid_xy(xy)
        rows.append({"cell_id":str(cid),"x_px":cx,"y_px":cy,"polygon_area_px2":polygon_area_xy(xy),"n_vertices":len(xy)})
    return pd.DataFrame(rows)
