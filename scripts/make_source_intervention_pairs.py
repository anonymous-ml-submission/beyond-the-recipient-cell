#!/usr/bin/env python3
"""Build distance/size/cell-class matched source-vs-control intervention pairs.

Paper mode requires marker-specific `control_cell_type`. `--allow-any-control` is engineering-only.
"""
import argparse, pandas as pd, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from leakage_vso.geometry import polygon_summary
p=argparse.ArgumentParser(); p.add_argument('--cells',required=True); p.add_argument('--markers',required=True); p.add_argument('--out',required=True); p.add_argument('--polygons',default=None); p.add_argument('--cell-type-col',default='cell_type'); p.add_argument('--max-distance-um',type=float,default=60); p.add_argument('--distance-caliper-um',type=float,default=5); p.add_argument('--max-log-area-ratio',type=float,default=0.7); p.add_argument('--max-per-marker',type=int,default=1000); p.add_argument('--k-controls',type=int,default=128); p.add_argument('--allow-any-control',action='store_true',help='Engineering only; paper mode requires control_cell_type in marker file'); a=p.parse_args()

def _read_table(path):
    q=Path(path)
    if q.suffix.lower()=='.parquet': return pd.read_parquet(q)
    if q.suffix.lower()=='.csv': return pd.read_csv(q)
    if q.suffix.lower()=='.tsv': return pd.read_csv(q,sep='\t')
    raise ValueError(f'Unsupported table format: {q}')

c=_read_table(a.cells).reset_index(drop=True); m=pd.read_csv(a.markers)
for col in ['cell_id','x_um','y_um',a.cell_type_col]:
    if col not in c: raise RuntimeError(f'missing cells column {col}')
req={'gene','source_cell_type','recipient_cell_type'}
if not req.issubset(m.columns): raise RuntimeError(f'markers must contain {sorted(req)}')
if 'control_cell_type' not in m.columns and not a.allow_any_control:
    raise RuntimeError('Paper mode requires marker column control_cell_type. Use --allow-any-control only for engineering diagnostics.')
if a.polygons:
    ps=polygon_summary(_read_table(a.polygons)); ps=ps[['cell_id','polygon_area_px2']]; c=c.merge(ps,on='cell_id',how='left')
area_col='polygon_area_um2' if 'polygon_area_um2' in c else ('polygon_area_px2' if 'polygon_area_px2' in c else None)
rows=[]; xy=c[['x_um','y_um']].to_numpy(float); types=c[a.cell_type_col].astype(str).to_numpy(); ids=c.cell_id.astype(str).to_numpy()
for _,r in m.iterrows():
    g=str(r.gene); src=str(r.source_cell_type); rec=str(r.recipient_cell_type)
    ctrl_type=str(r.control_cell_type) if 'control_cell_type' in m.columns and pd.notna(r.control_cell_type) else None
    if src==rec: raise RuntimeError(f'{g}: source_cell_type and recipient_cell_type must differ')
    if ctrl_type is not None and ctrl_type==src: raise RuntimeError(f'{g}: control_cell_type must differ from source_cell_type')
    src_ix=np.where(types==src)[0]; rec_ix=np.where(types==rec)[0]
    ctrl_ix=np.where(types==ctrl_type)[0] if ctrl_type is not None else np.where(types!=src)[0]
    if not len(src_ix) or not len(rec_ix) or not len(ctrl_ix): continue
    ts=cKDTree(xy[src_ix]); tc=cKDTree(xy[ctrl_ix]); ds,isrc=ts.query(xy[rec_ix],k=1)
    k=min(max(2,a.k_controls),len(ctrl_ix)); dc,ic=tc.query(xy[rec_ix],k=k)
    if k==1: dc=np.asarray(dc).reshape(-1,1); ic=np.asarray(ic).reshape(-1,1)
    for q,ri in enumerate(rec_ix):
        if ds[q]>a.max_distance_um: continue
        sj=src_ix[int(isrc[q])]; candidates=[]
        for dist,loc in zip(np.atleast_1d(dc[q]),np.atleast_1d(ic[q])):
            cj=ctrl_ix[int(loc)]
            if cj==ri or cj==sj: continue
            derr=abs(float(dist)-float(ds[q]))
            if derr>a.distance_caliper_um: continue
            log_area=np.nan
            if area_col and np.isfinite(c.loc[sj,area_col]) and np.isfinite(c.loc[cj,area_col]) and c.loc[sj,area_col]>0 and c.loc[cj,area_col]>0:
                log_area=abs(np.log(float(c.loc[cj,area_col])/float(c.loc[sj,area_col])))
                if log_area>a.max_log_area_ratio: continue
            score=derr/max(a.distance_caliper_um,1e-6)+(0.0 if not np.isfinite(log_area) else log_area)
            candidates.append((score,cj,float(dist),log_area))
        if not candidates: continue
        _,cj,cd,lar=min(candidates,key=lambda x:x[0])
        row={'gene':g,'recipient_cell_type':rec,'source_cell_type':src,'control_cell_type':str(types[cj]),
             'recipient_cell_id':ids[ri],'source_cell_id':ids[sj],'control_cell_id':ids[cj],
             'source_distance_um':float(ds[q]),'control_distance_um':cd,'distance_error_um':abs(cd-float(ds[q])),
             'log_area_ratio_abs':lar}
        if 'donor_id' in c: row['donor_id']=str(c.loc[ri,'donor_id'])
        if area_col: row.update(source_area=float(c.loc[sj,area_col]) if np.isfinite(c.loc[sj,area_col]) else np.nan, control_area=float(c.loc[cj,area_col]) if np.isfinite(c.loc[cj,area_col]) else np.nan)
        rows.append(row)
out=pd.DataFrame(rows)
if len(out): out=out.sort_values(['gene','distance_error_um','log_area_ratio_abs'],na_position='last').groupby('gene',group_keys=False).head(a.max_per_marker)
out.to_csv(a.out,index=False); print(out.groupby('gene').size() if len(out) else 'no pairs')
