#!/usr/bin/env python3
"""Recipient-specific, donor-stratified virtual contamination halo analysis."""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from leakage_vso.halo import nearest_source_distance,binned_halo
p=argparse.ArgumentParser(); p.add_argument('--pred',required=True); p.add_argument('--cells',required=True); p.add_argument('--markers',required=True); p.add_argument('--sample',default=None,help='Required when prediction archive contains multiple samples'); p.add_argument('--prediction-key',default='auto',choices=['auto','pred','pred_obs','pred_endo']); p.add_argument('--cell-type-col',default='cell_type'); p.add_argument('--out',required=True); p.add_argument('--max-um',type=float,default=100); p.add_argument('--bin-um',type=float,default=5); a=p.parse_args()
z=np.load(a.pred,allow_pickle=True); genes=[str(x) for x in z['genes']]
if a.prediction_key=='auto':
    key='pred' if 'pred' in z else ('pred_obs' if 'pred_obs' in z else None)
else: key=a.prediction_key
if key is None or key not in z: raise RuntimeError(f'Prediction key {key} unavailable; archive keys={list(z.keys())}')
pred=np.asarray(z[key]); ids=np.asarray([str(x) for x in z['cell_id']]);
if 'sample_id' in z:
    sids=np.asarray([str(x) for x in z['sample_id']]); uniq=np.unique(sids)
    if a.sample is None and len(uniq)>1: raise RuntimeError(f'Prediction archive contains multiple samples {uniq.tolist()}; pass --sample')
    sample=str(a.sample) if a.sample is not None else str(uniq[0]); keep=sids==sample; pred=pred[keep]; ids=ids[keep]
elif a.sample is not None:
    raise RuntimeError('--sample was supplied but prediction archive has no sample_id')
cells=pd.read_parquet(a.cells); cells['cell_id']=cells.cell_id.astype(str)
if cells.cell_id.duplicated().any(): raise RuntimeError('cells file contains duplicate cell_id within sample')
missing=set(ids)-set(cells.cell_id)
if missing: raise RuntimeError(f'{len(missing)} prediction cell IDs absent from provided cells file; wrong --sample/cells pairing?')
cells=cells.set_index('cell_id').loc[ids].reset_index(); markers=pd.read_csv(a.markers)
req={'gene','source_cell_type','recipient_cell_type'}
if not req.issubset(markers.columns): raise RuntimeError(f'markers must contain {sorted(req)}')
if a.cell_type_col not in cells: raise RuntimeError(f'cells missing {a.cell_type_col}')
if 'donor_id' not in cells: raise RuntimeError('cells missing donor_id; halo inference must be donor-stratified')
rows=[]; bins=np.arange(0,a.max_um+a.bin_um,a.bin_um)
for _,m in markers.iterrows():
    g=str(m.gene); src=str(m.source_cell_type); rec=str(m.recipient_cell_type)
    if src==rec: raise RuntimeError(f'{g}: source and recipient cell type must differ')
    if g not in genes: continue
    gi=genes.index(g); source=cells[a.cell_type_col].astype(str).to_numpy()==src; recipient=cells[a.cell_type_col].astype(str).to_numpy()==rec
    if source.sum()==0 or recipient.sum()==0: continue
    d=nearest_source_distance(cells,source)
    for donor,ix in cells.groupby('donor_id').groups.items():
        ix=np.asarray(list(ix),dtype=int); keep=recipient[ix] & np.isfinite(d[ix])
        if keep.sum()<3: continue
        tab=binned_halo(d[ix][keep],pred[ix][keep,gi],bins); tab['gene']=g; tab['source_cell_type']=src; tab['recipient_cell_type']=rec; tab['donor_id']=str(donor); rows.append(tab)
out=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(); Path(a.out).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False)
if len(out):
    agg=out.groupby(['gene','source_cell_type','recipient_cell_type','d_lo','d_hi','d_mid'],as_index=False).agg(mean=('mean','mean'),n_donors=('donor_id','nunique'),n_cells=('n','sum'))
    agg.to_csv(Path(a.out).with_suffix('.summary.csv'),index=False)
print(f'halo_rows={len(out)} prediction_key={key}')
